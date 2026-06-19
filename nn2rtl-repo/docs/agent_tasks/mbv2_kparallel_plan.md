# MobileNetV2 Engine K-Parallelism — Implementation Plan

**Status:** READ-ONLY investigation / executable plan. No RTL was edited and no
Vivado was run in producing this document.

**Scope:** Add `P`-lane reduction parallelism to the shared compute engine so the
34 engine-dispatched pointwise (1×1) convs run ~`P×` faster, *byte-exact* to the
current INT8 result. Target: drop the engine's serial MAC cost below the stem
(`conv_810` ≈ 11.4M cyc) so the engine stops gating throughput, then decide
whether the stem needs its own parallelism.

---

## 1. Current engine datapath (what actually runs)

The MobileNetV2 engine top
`output/mobilenet-v2/rtl/nn2rtl_top_engine.v:2167` instantiates `shared_engine`
with **default parameters** — `MAC_COUNT=256`, `WGT_W=4` (INT4 nibble-packed):

```verilog
shared_engine u_shared_engine ( .clk(clk), .rst_n(rst_n), ... );
```

`shared_engine` (`output/rtl/shared_engine_skeleton.v`) wires five sub-blocks.
The compute core is `mac_array` driven by `address_generator` and the FSM.

### 1.1 MAC array shape — 256 **output-channel**-parallel, **input-channel**-SERIAL

`output/rtl/engine/mac_array.v` is the entire compute core:

```verilog
module mac_array #(parameter integer WGT_W = 4) (
    input  wire          clk, rst_n, mac_clear, mac_valid_in,
    input  wire [7:0]    act_byte,                 // ONE activation byte, broadcast
    input  wire [256*WGT_W-1:0] weight_bus,        // 256 lanes * WGT_W bits
    output wire [8191:0] acc_out,                  // 256 * INT32
    output wire          mac_busy );

    for (lane = 0; lane < 256; lane = lane + 1) begin : g_mac
        wire signed [WGT_W-1:0] w_byte = $signed(weight_bus[lane*WGT_W +: WGT_W]);
        wire signed [7:0]       a_byte = $signed(act_byte);          // SAME byte, all 256 lanes
        (* use_dsp="yes" *) reg signed [15:0] mul_q1;
        reg signed [31:0] acc;
        always @(posedge clk)             mul_q1 <= w_byte * a_byte; // stage 1
        always @(posedge clk or negedge rst_n)                       // stage 2
            if (!rst_n)             acc <= 0;
            else if (mac_clear)     acc <= 0;
            else if (mac_valid_q1)  acc <= acc + $signed(mul_q1);
        assign acc_out[lane*32 +: 32] = acc;
    end
```

**Key fact:** every cycle, *one* activation byte (`act_byte`) is broadcast to all
256 lanes; each lane multiplies it by its own per-OC weight and accumulates. The
256 lanes are **256 distinct output channels** of the SAME spatial position and
the SAME input-channel index `k`. The **dot-product reduction over IC is walked
one element per cycle.** Throughput = **1 IC-MAC/cycle/lane = 256 MACs/cycle, but
only 1 step of the IC reduction per cycle.**

### 1.2 What drives the reduction — `address_generator` walks K_TOTAL one/cycle

`output/rtl/engine/address_generator.v` walks the inner loop
`for kh: for kw: for ic:` with `ic` innermost (lines 300–342). `k_index`
increments by 1 every active cycle to `K_TOTAL-1 = IC*KH*KW-1`, emitting one
`weight_rd_addr` (one 256-wide URAM word) and one `act_in_rd_addr` per cycle.
`mac_done` pulses when `k_cnt == k_total_m1`.

The FSM (`shared_engine_skeleton.v:215`) loops:
`ST_RUN` (K_TOTAL cycles) → `ST_REQUANT` → per oc_pass advance (`oc_pass_idx`,
0..ceil(OC/256)-1) → `ST_DRAIN` → next pixel. So per output pixel the engine
spends `ceil(OC/256) * (K_TOTAL + overhead)` cycles, with K_TOTAL serial.

### 1.3 Weight + activation feed (the bandwidth that already exists)

- **Weights:** `output/mobilenet-v2/weights/weight_memory_map.json` →
  `mac_count=256`, `num_banks=8`, `weights_per_bank_per_cycle=32`,
  `bank_useful_bits=256`, `uram_primitive_bits=72`, `uram_primitive_depth=4096`.
  One URAM read returns **256 INT4 weights = 1024 bits/cycle** (`URAM_DATA_W=1024`
  in the skeleton). These 256 weights are the 256 OC lanes for ONE `k`.
- **Activations:** `ACT_BUS_W=2048` (one BRAM beat = 256 INT8 channels of one
  pixel). The bridge / skeleton byte-selects **one** channel byte per cycle
  (`act_in_rd_data_d[ag_act_in_ic_byte_idx_d2*8 +: 8]`, skeleton line 389) and
  broadcasts it. So a whole pixel's 256 channels are already on-chip in one beat;
  the engine just consumes them one channel per cycle.

### 1.4 Requant — 256 OC-parallel, applied AFTER full reduction

`output/rtl/engine/requant_pipeline.v`: 256 lanes, each does
`biased = acc + bias` → `scaled = biased * mult` → `(scaled + HALF) >>> shift` →
INT8 saturate, with `+HALF` round-half-up (line 234) and ACC_W=32 / SCALED_W=65.
Triggered by `requant_valid_in = ag_mac_done_d5` — i.e. **only after the entire
IC reduction has completed and drained** (skeleton lines 423–445).

---

## 2. Where parallelism is cheapest AND byte-exact

For a 1×1 pointwise conv: `out[oc, p] = sat(requant(bias[oc] + Σ_{ic} act[ic,p] * w[oc,ic]))`.

Three axes:

| Axis | What it parallelizes | Cycle effect | Byte-exact? | Cost |
|---|---|---|---|---|
| **(a) Output-channel** | compute >256 OC at once (more lanes) | reduces `oc_passes`, NOT K_TOTAL | trivially exact (independent OC) | **already at 256**; MBv2 max OC=1280 → only 5 passes. K_TOTAL untouched ⇒ NO help to the limiter |
| **(b) Input-channel reduction-tree** | sum `P` partial products per cycle (wider dot-product step) | **K_TOTAL/P cycles** — directly attacks the serial limiter | exact **iff** the P-wide sum is exact-width and the requant still runs once after the FULL reduction (accumulation *order* changes but integer sum is associative & exact at INT32 — see §5) | P× DSPs + P× act bytes/cycle + P× weight words/cycle |
| **(c) Pixel/spatial** | compute P output pixels at once | reduces pixel loop, NOT K_TOTAL | exact (independent pixels) | P× act read ports (different pixels → different BRAM beats); P× lane replication; harder banking |

**Recommendation: axis (b), input-channel reduction-tree parallelism.** It is the
ONLY axis that shrinks `K_TOTAL` (the serial term that dominates: conv_912
K_TOTAL=320 walked one/cycle is the whole cost). It is byte-exact because INT
addition is associative and we keep the full INT32 accumulator and apply
saturation/requant exactly once after the complete reduction. Output-channel
parallelism (a) is already maxed at 256 and does nothing for K_TOTAL; pixel
parallelism (c) needs P independent activation BRAM read ports (P distinct pixel
beats per cycle) which is far more BRAM-port-expensive than (b), where the P
activations are P *channels of the same pixel* and already co-resident in the
single 2048-bit beat already on-chip.

### Why (b) is cheap here specifically

- The **activation** P bytes are P different IC channels of the **same pixel** —
  all already present in the single `act_in_rd_data` 2048-bit beat. No extra BRAM
  read port: just byte-select P bytes from the beat instead of 1
  (combinational fan-out of the already-latched word).
- The **weights** for P consecutive `k` (same 256 OC, IC step) are P consecutive
  URAM words. Bandwidth grows P× (need P×1024 bits/cycle) — addressed in §3.3.

---

## 3. Quantify P = 2, 4, 8

### 3.1 conv_912 engine cycles and engine-limited fps

conv_912: `IC=320, OC=1280, OH=OW=7` ⇒ `K_TOTAL=320`, `oc_passes=ceil(1280/256)=5`,
`npix=49`. Engine cost ≈ `npix * oc_passes * (ceil(K_TOTAL/P) + OVH)`, `OVH≈12`
(mac drain 2 + requant 5 + FSM/drain ~5).

| P | K_TOTAL/P | per (pix,pass) | conv_912 frame cyc | vs current engine (~0.081M) |
|---|---|---|---|---|
| 1 (today) | 320 | 332 | **81,340** | 1.0× |
| 2 | 160 | 172 | 42,140 | 1.93× |
| 4 | 80 | 92 | 22,540 | 3.61× |
| 8 | 40 | 52 | 12,740 | 6.39× |

conv_912 is NOT the engine limiter after offload (it is already 0.081M «
11.4M). The meaningful metric is the **engine-serial sum across all 34 offloaded
convs** (the engine is a single shared resource, time-multiplexed). That sum is
dominated by the high-K_TOTAL, high-pixel-count mid-network pointwise convs
(IC up to 960, OH·OW up to 28×28). Each such conv's cost scales as
`npix · ceil(OC/256) · ceil(K_TOTAL/P)`, so the **whole engine-serial total
scales ~`1/P`** (the `+OVH` term is a few % at these K_TOTAL).

**The governing system-level number is the limiter, not conv_912:**

- Frame fps @200MHz = `200e6 / max(engine_serial_total, stem_811_cyc, ...)`.
- Stem `conv_810` ≈ **11.44M cyc** (§4) is the post-offload limiter and is fixed
  regardless of engine P. So the *first* job of K-parallelism is to push the
  engine-serial total **below 11.4M**; beyond that, engine P gives no
  system speedup until the stem is also parallelized.

> Action item for the executor: dump the per-conv engine cycle cost for all 34
> convs from `nn2rtl_scheduler_schedule.json` / the cost model and sum them to get
> the exact engine-serial total at P=1; divide by P for the projection. The
> structure guarantees ~1/P scaling; the absolute crossover point vs 11.4M
> determines the useful P.

### 3.2 DSP cost (huge headroom)

- Current engine MACs: **256** (one `(* use_dsp *)` 8×8 multiply per lane,
  `mac_array.v:80`). Plus 256 requant multiplies (`requant_pipeline.v:194`,
  `scaled_q2 = biased * mult`, INT33×INT16). ≈ **512 DSP** engine-side.
- U250 = **12,288 DSP48E2**. Current engine ≈ 4.2% of device (consistent with
  the "~0.8%" figure once the spatial-node DSPs are excluded; engine MACs alone
  are the relevant pool).
- K-parallel (b) multiplies the MAC multipliers by P (requant multipliers are
  unchanged — still one per OC lane, fired once per pass):

| P | engine MAC DSP | + requant DSP | total | % of 12,288 |
|---|---|---|---|---|
| 1 | 256 | 256 | 512 | 4.2% |
| 2 | 512 | 256 | 768 | 6.3% |
| 4 | 1024 | 256 | 1280 | 10.4% |
| 8 | 2048 | 256 | 2304 | 18.8% |

Even P=8 leaves the device at <20% DSP — far under the 80% headroom ceiling.
DSP is **not** the constraint.

### 3.3 BRAM / URAM read-bandwidth (this is the real cost of (b))

- **Weights:** P-wide reduction needs the 256-OC weight words for **P
  consecutive `k`** every cycle = **P × 1024 bits/cycle** = P × 256 INT4.
  Current banking: `num_banks=8`, `weights_per_bank_per_cycle=32` →
  256 weights/cycle. To feed P lanes:
  - **Option W1 (widen the URAM word):** reorder the weight ROM so one physical
    read returns `P·256` weights (P consecutive k for all 256 OC). URAM72 is
    72b×4096-deep; INT4 ⇒ 18 weights/primitive-line. `P·256` INT4 = `P·1024` bits
    ⇒ `ceil(P·1024/72)` URAM primitives per read = 15 (P=1) → 29 (P=2) → 57
    (P=4) → 114 (P=8) primitives **wide**, with depth `K_TOTAL/P` shorter.
    **URAM tile count is ≈ constant** (same total bits, just wider×shallower) —
    this is the cheapest option. Requires re-running `build_weight_memory_map.py`
    with a `--k-parallel P` knob that lays weights out P-consecutive-k-major.
  - **Option W2 (P read ports):** P parallel URAM read ports at the current width.
    URAM is true-dual-port; P>2 needs replication (P× URAM tiles) — rejected,
    doubles/quadruples weight BRAM which is already the tightest resource on the
    ResNet sibling (BRAM 94.6% in MEMORY). **Use W1.**
- **Activations:** P bytes = P IC-channels of the **same pixel** = P byte-slices
  of the **already-latched** `act_in_rd_data` 2048-bit beat. **Zero extra BRAM
  read port** — just P parallel byte-selects (combinational mux fan-out) of the
  held word, indexed by `k, k+1, ... k+P-1`. For IC ≤ 256 the whole pixel is one
  beat; for IC > 256 (none of the 34 convs exceed IC=960, but several exceed 256)
  the existing chunk-stride logic (`address_generator.v:175-197`) already handles
  multi-beat pixels — the P-wide select must stay within a 256-channel chunk
  boundary or read the next chunk's beat (handle the chunk-cross by either
  padding P to divide 256, or pre-staging two beats; P∈{2,4,8} all divide 256 so
  a chunk holds an integer number of P-groups — clean).

### 3.4 LUT / FF delta

- Per added lane: one 8×8 mul (DSP, not LUT) + one INT32 adder feeding the
  accumulator. The reduction tree adds `(P-1)` INT≤24 adders per OC lane
  ⇒ `256 · (P-1)` adders. P=8 ⇒ ~1,792 small adders ≈ a few % of U250 LUTs
  (1.7M LUT). FFs: the accumulator count is unchanged (still 256 INT32); add P
  product pipeline regs per lane for retiming = `256·P` 16-bit regs (P=8 ⇒ 2048
  regs ≈ negligible vs 3.4M FF). LUT/FF are **not** constraining.

---

## 4. New bottleneck after K-parallel engine

After offload + K-parallel, the limiters are (in order):

1. **Stem `conv_810`** (`output/mobilenet-v2/rtl/node_conv_810.v`): 3×3 s2 p1,
   `IC=3, OC=32, IH=IW=224, OH=OW=112`, MP=4, spatial line-buffer datapath.
   Cost ≈ `npix(12,544) · ceil(OC/MP)(8) · (MP·K_TOTAL(27)+6)(114)` ≈ **11.44M
   cyc** = 17.5 fps @200MHz on its own. This becomes the global limiter once the
   engine-serial total drops below it.
   - The stem is a **standalone spatial node**, NOT engine-dispatched. Its
     `conv_datapath` is the MP=4 serialized-lane datapath
     (`compute_conv2d_latency_cycles`, `scripts/golden_impl.py:113`), where
     `per-pass = MP·K_TOTAL+6` and `oc_passes=ceil(OC/MP)`. Because
     `MP·K_TOTAL·oc_passes ≈ K_TOTAL·OC` is **independent of MP**, raising MP
     does NOT help the stem (golden_impl.py:74 note). The stem needs a genuine
     **K-parallel (reduction) or OC-parallel** datapath rework, OR
     engine-dispatch the stem too (it is 3×3, the engine supports KH/KW up to 3).
2. **BRAM activation write/read** of the shared activation buffers between engine
   passes — bridge is single-beat (`bram_to_stream_bridge.v`, "no deep FIFO"),
   but at K-parallel rates the per-pass drain (1 beat/pass) is still « MAC cost,
   so not yet binding.

**Conclusion:** K-parallelizing the engine alone caps system throughput at the
stem's ~17.5 fps (vs the current ~10 fps engine-limited). To go beyond ~17 fps the
**stem must also be parallelized** (reduction-parallel `conv_datapath`, or move
the stem onto the K-parallel engine). Recommended sequencing: do the engine
first (biggest, cleanest win to ~17 fps), then evaluate the stem as a follow-on.

---

## 5. Byte-exactness contract (MUST hold)

The integer result must be identical to today's INT8 output. Rules:

1. **Accumulator width unchanged at INT32.** `acc` is `reg signed [31:0]`
   (`mac_array.v:81`). A P-wide reduction tree sums P products
   (each INT16, `mul_q1`) then adds to the INT32 acc. INT addition is associative;
   summing `(p0+p1+...+p_{P-1})` then adding to acc gives the **same integer** as
   adding them one at a time. No overflow risk: max |acc| for MBv2 pointwise
   (IC≤960, |act·w|≤127·7=889) ≈ 960·889 ≈ 0.85M « 2^31. **Keep INT32 acc; size
   the P-input tree's intermediate sum to ≥ ceil(log2(P)) + 16 bits** (P=8 ⇒ 19b)
   so no intermediate truncation.
2. **Saturation / requant applied ONLY after the FULL reduction.** Today
   `requant_valid_in = ag_mac_done_d5` fires after the whole IC walk drains
   (skeleton:445). Keep this: requant must see the COMPLETE acc, never a partial.
   The K-parallel change only makes the walk `K_TOTAL/P` cycles; the
   `mac_done → drain → requant` sequencing is unchanged (re-tune the `d5` depth
   only if the reduction tree adds pipeline stages — see step 4 below).
3. **Same rounding.** `requant_pipeline.v:234` uses `scaled + HALF` (round-half-up
   toward +inf), per-OC `mult`/`shift` from `scale_in`. Do NOT touch this module.
4. **Tail / partial-group handling.** If `P ∤ K_TOTAL`, the final reduction group
   has `K_TOTAL mod P` real terms; the extra lanes MUST contribute **zero**
   (gate their `mul_valid`, or feed weight=0 / act=0). MBv2 pointwise IC values:
   16,24,32,64,96,144,160,192,320,576,960 — `P=2` divides all; `P=4` divides all
   except none-problematic (all even, most ÷4: 144=÷4? 144/4=36 ✓, 96 ✓, 160 ✓,
   192 ✓, 320 ✓, 576 ✓, 960 ✓; 16,24,32,64 ✓). `P=8`: 24/8=3 ✓, 144/8=18 ✓,
   160/8=20 ✓, all listed IC are ÷8 EXCEPT none — verify per layer at build time
   and zero-pad the tail group. **Tail zeroing is the single most bug-prone spot;
   add a directed Verilator test that forces a non-divisible IC.**

**Verification (per the task):**
```
npx tsx scripts/_verify_mbv2_variant.ts <rtl.v> <module> <sidecar.json>
```
This runs `run_verilator` and reports `mismatch_count` / `max_error`. The
engine is verified through the engine-isolation harness lineage
(`tb/engine_verilator_iso_tb.cpp`, `scripts/engine_sweep_driver.py`) and at the
node level via `_verify_mbv2_variant.ts`. Acceptance: `mismatch_count == 0` on a
representative engine conv (e.g. `node_conv_912`, `node_conv_840`) at the new P,
AND a forced-non-divisible-IC case to exercise tail zeroing.

---

## 6. RANKED recommendation + step-by-step edit list

### Ranked recommendation

| Rank | Axis | P | Engine-serial speedup | DSP | BRAM/URAM | Verdict |
|---|---|---|---|---|---|---|
| **1 (DO THIS)** | (b) IC reduction-tree | **4** | ~3.6× | 1,280 (10.4%) | URAM word widened ~4× wide / ÷4 deep (≈ const tiles) | Best ratio: pushes engine-serial well under the 11.4M stem with comfortable area; all MBv2 IC values ÷4. |
| 2 | (b) IC reduction-tree | 2 | ~1.9× | 768 (6.3%) | ~2× wide / ÷2 deep | Safe fallback if P=4 weight banking is tight; may not clear the stem on the heaviest convs — measure first. |
| 3 | (b) IC reduction-tree | 8 | ~6.4× | 2,304 (18.8%) | ~8× wide / ÷8 deep | Overkill: engine already « stem at P=4, so P=8 buys NO system fps until the stem is also parallelized. Only worthwhile if stem is K-parallelized in the same push. |
| — | (a) OC-parallel | — | 0× to limiter | — | — | Rejected: already 256, MBv2 OC ≤1280 ⇒ ≤5 passes, does not touch K_TOTAL. |
| — | (c) pixel-parallel | — | ~P× | P× | **P× act read ports** | Rejected: needs P independent pixel BRAM beats/cycle; (b) reuses the one already-latched beat. |

**Pick P=4, axis (b).** Projected: engine-serial total → ~1/3.6 of current →
comfortably below the stem's 11.4M ⇒ **system limiter becomes the stem at
~17.5 fps @200MHz (~13 fps @150MHz)**, up from the current ~10 fps. DSP 10.4%,
URAM ≈ flat, LUT/FF negligible. To exceed ~17 fps, follow with a stem rework
(separate task).

### Step-by-step edit list (execute later, byte-exact, then verify)

1. **Parameterize the engine for P.** `output/rtl/shared_engine_skeleton.v`:
   add `parameter integer K_PAR = 1;` to `shared_engine` (default 1 = today's
   behavior, keeps all existing goldens passing). Thread `K_PAR` into `mac_array`.
   Update the MobileNet instantiation `output/mobilenet-v2/rtl/nn2rtl_top_engine.v:2167`
   to `shared_engine #(.K_PAR(4)) u_shared_engine (...)`.

2. **`output/rtl/engine/mac_array.v` — P-wide reduction lane.** For each of the
   256 OC lanes, accept `P` weights and `P` activation bytes per cycle; compute
   `P` products (P `(* use_dsp *)` multipliers), sum them in a width-safe tree
   (`reg signed [15+ceil(log2(P)):0] psum`), then `acc <= acc + psum` gated by
   `mac_valid_q1`. Keep `acc` INT32. Widen `weight_bus` to `256*K_PAR*WGT_W` and
   add a `[8*K_PAR-1:0] act_bytes` input (P bytes). Pipeline the tree if it
   hurts Fmax (then bump the requant `ag_mac_done_dN` depth in step 5).

3. **`output/rtl/engine/address_generator.v` — advance `k` by P.** Change the
   inner-loop advance (lines 300–342) to step `ic_cnt`/`k_cnt` by `K_PAR` and run
   `K_TOTAL/K_PAR` cycles; emit P consecutive weight words (or one P-wide word —
   see step 4) and the P byte indices `ic_cnt, ic_cnt+1, ...` per cycle. Handle
   the **tail group** when `K_PAR ∤ K_TOTAL` by gating the surplus lanes to zero
   (new `mac_lane_mask` or zeroed act/weight). Recompute `mac_done` at
   `k_cnt >= K_TOTAL` (not exact equality) since k steps by P.

4. **`scripts/build_weight_memory_map.py` — P-consecutive-k weight layout.** Add a
   `--k-parallel P` knob that lays the engine weight ROM out so one read returns
   the 256 OC × P-consecutive-k block (Option W1, §3.3). Widen `URAM_DATA_W` to
   `256*K_PAR*WGT_W` in the skeleton and the top wrapper's URAM bank
   instantiation. Re-emit `weight_memory_map.json` (`weights_per_bank_per_cycle`
   ⇒ `32*K_PAR`, `bank_useful_bits` ⇒ `256*K_PAR`). **This is the regen step the
   MEMORY warns about** — re-run the full regen chain (`build_bias_memory_map`,
   `build_scale_memory_map`, engine bank repack) so goldens stay consistent.

5. **`output/rtl/shared_engine_skeleton.v` — activation P-byte select + timing.**
   Replace the single `mac_act_byte_sel` (line 389) with P parallel byte-selects
   from the held `act_in_rd_data_d` (no new BRAM port). Re-tune the requant
   trigger depth `ag_mac_done_d5` (lines 423–445) ONLY if the P-tree adds
   pipeline stages, so requant still captures the FINAL acc (byte-exact rule §5.2).

6. **`requant_pipeline.v` — DO NOT TOUCH.** It runs once per pass on the complete
   acc; unaffected by K-parallelism. Keep the `+HALF` rounding and per-OC scale.

7. **Verify byte-exact.** Build the engine-isolation harness and run, plus:
   ```
   npx tsx scripts/_verify_mbv2_variant.ts output/mobilenet-v2/rtl/node_conv_912.v node_conv_912 <sidecar>
   ```
   on `node_conv_912` and a divisibility-stress conv. Require `mismatch_count==0`,
   `max_error==0`. Then re-run the MobileNet e2e value harness
   (`tb/mbv2_top_value_tb.cpp` / `scripts/run_mbv2_top_value.ts`).

8. **Re-measure the limiter.** Recompute the engine-serial total at P=4 from the
   updated cost model; confirm it is < 11.4M (stem). If yes, the system limiter is
   the stem — open a follow-on task for stem reduction-parallelism.

### Files touched (summary)
- `output/rtl/shared_engine_skeleton.v` (params, P-byte select, timing)
- `output/rtl/engine/mac_array.v` (P-wide reduction tree)
- `output/rtl/engine/address_generator.v` (step-by-P walk, tail zeroing)
- `output/mobilenet-v2/rtl/nn2rtl_top_engine.v` (`#(.K_PAR(4))`, widen URAM bank)
- `scripts/build_weight_memory_map.py` (`--k-parallel`, re-emit map) + full regen chain
- `requant_pipeline.v`, `bram_to_stream_bridge.v`, `config_register_block.v`: **unchanged**
- Verify: `scripts/_verify_mbv2_variant.ts`, engine-iso harness, e2e value harness

### Risk notes
- **Tail group zeroing** (§5.4) is the top byte-exactness risk — directed test it.
- **Weight ROM regen** must run the FULL chain (bias/scale maps, engine bank
  repack) or stale goldens will mask wrongness (per MEMORY
  `feedback-regen-must-rebuild-engine-maps`).
- **No Vivado** until byte-exact AND e2e pass AND fit re-checked (per MEMORY
  `feedback-vivado-only-when-proven`). The URAM word widening (Option W1) keeps
  total weight bits constant, so the fit story should be ~neutral — but confirm
  with an OOC of one widened bank before any full run.

---

## Adversarial review verdict (hardened)

**Synthesis/decision agent, 2026-06-02. READ-ONLY review of 4 adversarial
refutation passes against ground-truth RTL + IR. Verdict: this section only.**

### TL;DR headline

> **NO-GO as written.** The plan's central justification ("K-parallel engine →
> stem-limited ~17.5 fps @200 MHz, up from ~10 fps") is **false on three
> independent legs**: the engine is **already a 3× sub-limiter** (engine-serial
> = 3.79 M cyc « stem 11.44 M), engine and spatial are **mutually exclusive**
> (`spatial_stall=1` in `S_WAIT_DONE`, scheduler line 1130 — so they serialize,
> they do not overlap), and the **clock is 50 MHz** (`layer_ir.json
> clock_period_ns=20`), not 200 MHz. The HONEST e2e is ~39.4 M cyc → **5.08 fps
> @200 MHz / 1.27 fps @50 MHz**, and K-parallel P=4 moves it to ~37.0 M cyc →
> **5.41 fps @200 MHz (~+6 %), NOT 1.75×**. The system is **spatial-bound** (1
> stem + 17 depthwise 3×3, 35.6 M cyc serial); the engine is the wrong thing to
> optimize first.

### (1) Which of the 4 claims survived?

| Lens | Plan's claim | Verdict | Why |
|---|---|---|---|
| **byte_exact** | "P-wide tree → identical (byte-exact) result; mac_done→drain→requant unchanged" | **BROKEN** | Value-axis is sound (INT32 acc, no mid-reduction clamp — `mac_array.v:102` is a pure `acc<=acc+mul_q1`; single clamp in `requant_pipeline.v:258-267`). But byte-exactness here is a **timing** contract: `requant_valid_in = ag_mac_done_d5` (`skeleton:445`) is a hand-tuned, **race-critical** acc-capture depth that TWO prior silent-corruption bugs calibrated (d3→d4 for WEIGHT_RD_LATENCY=2 line 354; d4→d5 for the K+4 last-accumulate race, comments lines 415-422). A P-tree adds ≥⌈log₂P⌉ pipeline stages on the acc-feeding path → the unchanged d5 captures acc too early → silently drops the last reduction group → the exact 2026-05-31 "output-biased-LOW" signature, invisible to value spot-checks. Secondary: **no per-lane valid exists** — `mac_valid_in` is a single scalar (`skeleton:381`, `mac_array` gates the whole row on `mac_valid_q1`), so tail-zeroing needs NEW RTL; and for **IC>256** the P bytes straddle a 256-ch BRAM beat (`address_generator.v:192-197`, `act_in_ic_byte_idx=ic_cnt[7:0]` line 292) so "zero extra read port" is false at P-groups crossing the chunk boundary. |
| **divisibility** | "P=4 divides all 34 K_TOTAL, no special-casing" | **SURVIVED** (conclusion correct, even understated) | Verified all 34 dispatches from `nn2rtl_scheduler_schedule.json` cross-checked to `layer_ir.json`: all are genuine **1×1** (kernel=[1,1] ⇒ K_TOTAL=IC). Distinct IC = **{16,24,32,64,96,144,160,192,320,384,576,960}** — **every one ÷2, ÷4 AND ÷8**. P∈{2,4,8} all clean, zero tail groups for real MBv2. Plan §5.4 defects (cosmetic): (a) hand-typed IC list **omits 384** (7 of 34 convs); (b) reasons about IC not K_TOTAL (harmless ONLY because all 1×1 — must be stated); (c) garbled "÷8 EXCEPT none" text. |
| **area_neutral** (P=4) | "URAM ≈ const tiles at P=4 (and §6 rank-3 says P=8 ≈ const)" | **SURVIVED at P=4, but the MECHANISM/NUMBERS are WRONG and the P=8 claim is FALSE** | Ground truth from the deployed wrapper: 8 banks, each `uram_weight_bank #(.DEPTH(13152))` with a **288-bit word** (verified: `uram_weights_bank0.mem` = 72 hex chars/row = 288 b, 13152 rows). Native URAM288 = 72b×4096 ⇒ per bank ⌈288/72⌉=4 wide × ⌈13152/4096⌉=4 deep = 16 tiles → **128 total** (matches `weight_memory_map.json total_uram_blocks_required=128`). Tile count vs P (288-word model): **P1=128, P2=128, P4=128, P8=256**. So P=4 IS neutral (depth slack absorbs the 4× width down to 1 deep tile) — **but P=8 ~DOUBLES weight URAM (+128 tiles)**. The plan's "15→29→57→114 primitives wide, ≈const tiles" is width-only and wrong; the §6 Rank-3 "P=8 ≈const tiles" claim is **false**. P=4 fits easily: 128 of 1280 URAM = 10 % (engine weight ROM alone), far under 80 %. |
| **bottleneck / fps** | "P=4 → engine below stem → stem-limited ~17.5 fps @200 MHz, up from ~10 fps" | **BROKEN** (all three legs) | (1) **Engine is NOT the limiter at any P.** Independent recompute over all 34 dispatches: engine-serial = **3.79 M (P1) / 2.19 M (P2) / 1.38 M (P4) / 0.98 M (P8)** cyc — already 3× below the 11.44 M stem at P=1. (2) **Engine & spatial are mutually exclusive**, not overlapped: `nn2rtl_scheduler.v` `S_WAIT_DONE: spatial_stall=1'b1` (line 1130) freezes the spatial chain for the engine's entire compute ⇒ e2e ≈ engine_serial + spatial_serial. (3) **Spatial is the real limiter**: 1 stem (11.44 M) + 17 depthwise 3×3 (MP=4) = **18 non-dispatched 3×3 convs, 35.6 M cyc serial** (verified against per-layer `mac_parallelism` + output_shape; the MP-independence the plan cites for the stem applies to every depthwise too). (4) **Clock is 50 MHz** (`clock_period_ns=20` for all layers; sibling ResNet only closed timing at 25-40 MHz). |

**Score: 2 of 4 broken (byte_exact, bottleneck), 2 survived (divisibility,
area@P4).** The two survivors are necessary-but-not-sufficient: the plan can be
built byte-exact and area-neutral at P=4, but it **optimizes a non-bottleneck**.

### (2) Corrected recommendation

- **Final P = 4** *if* the engine K-parallel work is done at all — it is
  byte-exact-implementable (with the hardened contract below), area-neutral
  (128 URAM tiles, ~flat), and all 34 IC are ÷4. **Reject P=8** (doubles weight
  URAM, +128 tiles, and buys zero system fps). P=2 is a pointless half-measure.
  **But the correct decision is to NOT do the engine first** — see fps below.

- **Corrected byte-exactness contract (MUST hold):**
  1. Keep `acc` INT32 and the **single post-reduction** clamp/requant
     (`requant_pipeline.v` UNTOUCHED — it already uses unconditional `+HALF`
     round-half-up, line 234; the 3→4-cycle Lever-2 split is internal and
     unaffected). Size the P-input tree intermediate to ≥⌈log₂P⌉+16 b (P=4 ⇒
     18 b) so no intermediate truncation.
  2. **Treat the drain depth as a DERIVED constant, not a re-tune.** Set
     `requant_valid_in = ag_mac_done_d{5 + TREE_STAGES}` where `TREE_STAGES` is
     the exact count of pipeline registers the P-tree inserts between `mul_q1`
     and `acc`. **Re-derive against WEIGHT_RD_LATENCY=2** and **assert it in the
     cycle-accurate engine-isolation harness BEFORE any value check** — a
     1-cycle error here is the invisible 2026-05-31 dropped-last-term bug.
  3. **Add an explicit per-tree-lane valid mask (P bits) in `mac_array`** so the
     tail group / chunk-straddle can zero individual lanes; sum only valid
     products into the width-safe intermediate. Do NOT rely on the single scalar
     `mac_valid_in`. (Tail-zero is dead code for the 34 real convs — all ÷4 —
     but is MANDATORY the moment the plan's own next step dispatches the 3×3
     stem K_TOTAL=27 / depthwise K_TOTAL=9, which are NOT ÷4.)
  4. **For IC>256, constrain the P-group to never cross a 256-channel chunk
     boundary** (all IC are ÷4 and 256 is ÷4, so a chunk holds an integer number
     of P-groups — clean by construction at P=4) — and correct §3.3 to state the
     activation-port cost is only zero **because** P|256, not in general.
  5. Acceptance: `mismatch_count==0 AND max_error==0` on the cycle-accurate
     engine-iso harness for `node_conv_912` AND `node_conv_840` AND a forced
     non-divisible case **including K_TOTAL∈{27,9}**, not just node-level checks.

- **Corrected area cost:** P=4 weight URAM = **128 tiles (10 % of 1280)** —
  area-neutral, FITS « 80 %. The plan's "≈const tiles" is true at P=4 only
  because depth collapses to 1 tile; **P=8 = 256 tiles (+100 %)** — strike the
  §6 "P=8 ≈const" claim. Confirm with an OOC of one widened bank before any run.

- **Corrected fps + does the stem need parallelizing:** **YES — the stem AND the
  17 depthwise 3×3 must be parallelized first; the engine must not be touched
  first.** Honest numbers (engine+spatial serialize):
  - P=1 e2e ≈ **39.4 M cyc** → 5.08 fps @200 MHz / **1.27 fps @50 MHz**.
  - P=4 e2e ≈ **37.0 M cyc** → 5.41 fps @200 MHz / **1.35 fps @50 MHz** (≈ +6 %).
  - The plan's "17.5 fps" = 200e6/11.44e6 = stem-alone at a **fabricated** clock.
  The **first** throughput lever must be the SPATIAL path (35.6 M serial), via a
  reduction-/OC-parallel `conv_datapath` for the 3×3s **and/or** removing
  `spatial_stall=1` in `S_WAIT_DONE` so the engine's 34 dispatches overlap the
  spatial chain (that single change likely beats any engine K-parallelism).

### (3) Clean, ordered, EXECUTABLE edit list (run when `nn2rtl_top_engine.v` is free)

**Priority A — SYSTEM throughput (do FIRST; the engine is not the limiter):**

  A1. **Overlap engine & spatial.** `output/mobilenet-v2/rtl/nn2rtl_scheduler.v`
      line 1130: investigate setting `S_WAIT_DONE: spatial_stall = 1'b0` (mirror
      `S_WAIT_DRAIN`) so the spatial chain runs during engine compute. Requires
      verifying no input-loader BRAM write/read collision; gate behind the e2e
      value harness. This is the highest-leverage change and touches no engine
      arithmetic (byte-exact by construction if no collision).
  A2. **Parallelize the spatial 3×3 datapath** (stem + 17 depthwise, 35.6 M cyc):
      a genuine reduction- or OC-parallel `conv_datapath` rework (note
      `scripts/golden_impl.py:74` MP-independence — raising MP does NOT help), OR
      engine-dispatch the 3×3s (engine supports KH/KW≤3) — which then makes the
      tail-zero mask (B-track step) MANDATORY for K_TOTAL∈{27,9}. Atomic change:
      RTL + `compute_conv2d_latency_cycles` + patterns + goldens together
      (MEMORY `feedback-atomic-arch-changes`).

**Priority B — engine K-parallel P=4 (LOW priority follow-on; only after A):**

  B1. `output/rtl/shared_engine_skeleton.v`: add `parameter integer K_PAR=1`
      (default 1 keeps all goldens), thread into `mac_array`; set
      `#(.K_PAR(4))` at the MobileNet instantiation in
      `output/mobilenet-v2/rtl/nn2rtl_top_engine.v` (the `shared_engine` inst).
  B2. `output/rtl/engine/mac_array.v`: per OC lane accept P weights + P act
      bytes, P `(* use_dsp *)` muls, **per-tree-lane valid mask (P bits)**, sum
      valid products in a ≥18-b tree, `acc <= acc + psum` gated by
      `mac_valid_q1`. Widen `weight_bus` to `256*K_PAR*WGT_W`; add
      `act_bytes[8*K_PAR-1:0]`. Record exact `TREE_STAGES` added.
  B3. `output/rtl/engine/address_generator.v`: step `ic_cnt`/`k_cnt` by `K_PAR`,
      run `⌈K_TOTAL/K_PAR⌉` cycles, emit P byte indices, **constrain P-group to
      not cross the 256-ch chunk boundary** (clean at P=4 since 256∣…), gate
      surplus lanes to zero on the tail, recompute `mac_done` at `k_cnt≥K_TOTAL`.
  B4. `output/rtl/shared_engine_skeleton.v`: P parallel byte-selects from the
      held `act_in_rd_data_d` (line 389-390); **set
      `requant_valid_in = ag_mac_done_d{5+TREE_STAGES}`** (extend the
      `ag_mac_done_dN` chain, lines 423-445) — DERIVED from B2's TREE_STAGES,
      re-asserted against WEIGHT_RD_LATENCY=2 (line 354).
  B5. `scripts/build_weight_memory_map.py`: `--k-parallel P` knob laying weights
      P-consecutive-k-major; widen `URAM_DATA_W` and the `uram_weight_bank`
      `.DEPTH(13152)`→`⌈13152/4⌉=3288` / `READ_DATA_WIDTH` ×4 in the wrapper.
      **Re-run the FULL regen chain** (`build_bias_memory_map`,
      `build_scale_memory_map`, engine bank repack, `refresh_final_golden`,
      `rebuild_contract_goldens`) — MEMORY `feedback-regen-must-rebuild-engine-maps`.
  B6. **DO NOT TOUCH** `requant_pipeline.v`, `config_register_block.v`,
      `bram_to_stream_bridge.v`.
  B7. Verify: cycle-accurate engine-iso harness (`tb/engine_verilator_iso_tb.cpp`)
      asserting the drain depth FIRST; then `node_conv_912`, `node_conv_840`, and
      forced K_TOTAL∈{27,9} — require `mismatch_count==0 AND max_error==0`; then
      full e2e (`tb/mbv2_top_value_tb.cpp` / `scripts/run_mbv2_top_value.ts`).

**Prerequisite for ALL fps claims:** re-baseline against a **COMPLETED** MBv2
e2e — current logs are TIMEOUT/unfinished or ResNet (`decisive_e2e.log` =
ResNet conv_196). Quote fps at the design's real **50 MHz** (or gate any
200 MHz figure on an actual timing run; sibling closed only at 25-40 MHz).

### Decision

**GO/NO-GO = NO-GO on the plan as written** (engine-first, "17.5 fps @200 MHz").
**Conditional GO** for the engine K-parallel **P=4** work as a **low-priority
follow-on AFTER the spatial path is parallelized**, and ONLY with the hardened
byte-exact contract above (derived drain depth + per-lane valid mask +
chunk-boundary constraint). Surviving risks: (i) drain-depth off-by-one
re-introducing the silent dropped-last-term corruption; (ii) the spatial rework
(A2) is the actual hard, atomic, byte-exact-entangled task the plan defers.
**No Vivado** until byte-exact + completed-e2e + fit re-checked (MEMORY
`feedback-vivado-only-when-proven`).
