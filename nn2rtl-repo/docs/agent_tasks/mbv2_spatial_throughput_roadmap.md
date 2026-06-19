# MobileNetV2 — Spatial Throughput Roadmap (A1 + A2)

**Status:** READ-ONLY investigation / executable byte-exact plan. **No RTL was
edited, no Vivado was run, no long sim was run** in producing this document. All
cycle counts below are recomputed bottom-up from
`output/mobilenet-v2/layer_ir.json` (the authoritative geometry) and the on-disk
RTL latency contracts; fps is quoted at the design's **real 50 MHz**
(`clock_period_ns = 20` for every layer) with any 200 MHz figure explicitly
labelled *hypothetical / timing-gated*.

This roadmap supersedes the engine-K-parallel plan
(`mbv2_kparallel_plan.md`) as the throughput priority: that plan's adversarial
review proved the engine is **not** the limiter. The limiter is the spatial 3×3
path. This doc plans the two levers that actually move it.

---

## 0. The numbers this plan starts from (recomputed, not inherited)

Bottom-up frame-cycle recompute over `layer_ir.json` (full output frame =
`npix · OC_PASSES · per_pass`, the throughput cost, not first-valid latency):

| Block | per-pass | cyc/frame | note |
|---|---|---:|---|
| Stem `conv_810` (group=1, 3×3 s2 p1, IC=3 OC=32, 112×112) | MP·K_TOTAL+6 = 4·27+6 = 114 | **11,440,128** | `conv_datapath`, MP=4, K_TOTAL=27 |
| 17 depthwise 3×3 (inline DW datapath, MP=4, K_TOTAL=9) | 4·9+6 = 42 | **24,169,152** | per-channel 9-tap; see §A2 |
| **Spatial 3×3 total (serial)** | | **35,609,280** | the limiter |
| Engine-serial (34 pointwise) | — | **3,790,000** | already 3× under the stem |

Per-depthwise breakdown (the 17): conv_812=4.21M, 818=3.16M, 824=4.74M,
830=1.19M, 836/842=1.58M each, 848=0.40M, 854/860/866/872=0.79M each,
878/884=1.19M each, 890=0.30M, 896/902/908=0.49M each.

**Engine and spatial SERIALIZE** (proven below in §A1):
`nn2rtl_scheduler.v:1130` holds `spatial_stall = 1'b1` for the whole engine
compute window. So:

```
e2e ≈ spatial_serial + engine_serial ≈ 35.61M + 3.79M ≈ 39.40M cyc
```

| Clock | e2e fps (P=1 baseline) |
|---|---|
| **50 MHz (real)** | **1.27 fps** |
| 200 MHz (hypothetical, timing-gated) | 5.08 fps |

The sibling ResNet design only closed timing at 25–40 MHz, so even the 50 MHz
figure is optimistic until an MBv2 timing run exists. Treat 200 MHz as a ceiling,
not a deliverable.

---

## A1 — Overlap engine & spatial (cheap, conditionally byte-exact)

### What the RTL actually does today

The top builds `spatial_run` and gates **every** spatial node's `valid_in` and
every input-loader's `in_valid` with it:

`output/mobilenet-v2/rtl/nn2rtl_top_engine.v:443`
```verilog
wire spatial_throttle = engine_busy | sched_spatial_stall;
wire spatial_run      = ~spatial_throttle;
```

`engine_busy` comes straight from the engine; `sched_spatial_stall` comes from
the scheduler FSM. The scheduler asserts the stall for the entire engine-compute
phase:

`output/mobilenet-v2/rtl/nn2rtl_scheduler.v:1129`
```verilog
S_WAIT_DONE: begin
    spatial_stall = 1'b1;
end
```

versus the drain phase, which already lets the chain run:

`nn2rtl_scheduler.v:1132`
```verilog
S_WAIT_DRAIN: begin
    // Fix 14: chain MUST run so the bridge can drain the
    // engine_output_fifo and feed downstream relu/add layers.
    spatial_stall = 1'b0;
end
```

Note: even with `sched_spatial_stall=0`, `engine_busy=1` *alone* drives
`spatial_run=0` (the OR at line 443). So **A1 requires dropping the
`engine_busy` term from `spatial_throttle` too**, not only the scheduler stall —
this is the part the bare "mirror S_WAIT_DRAIN" framing misses.

### The proposed edit (two coordinated changes)

1. `nn2rtl_scheduler.v:1130` — `S_WAIT_DONE: spatial_stall = 1'b0;` (mirror
   `S_WAIT_DRAIN`).
2. `nn2rtl_top_engine.v:443` — `wire spatial_throttle = sched_spatial_stall;`
   (drop the `engine_busy | ` term), so the chain is gated only by the
   scheduler's explicit phases (`S_WRITE`/`S_WRITE_RESP`/`S_NEXT_STEP`/
   `S_PULSE_START`/`S_NEXT_DISP` still assert `spatial_stall=1`), not by the
   engine simply being busy.

### Collision analysis — does the concurrently-running spatial chain corrupt the engine's activation BRAM?

There is **one** shared activation BRAM, simple-dual-port (independent R/W):

`nn2rtl_top_engine.v:2073`
```verilog
act_unified_mem #( .DEPTH(24576), .ADDR_W(15) ) u_act_mem (
    .clk(clk),
    .rd_addr(engine_act_in_rd_addr[14:0]), .rd_en(engine_act_in_rd_en),
    .rd_data(engine_act_in_rd_data),
    .wr_addr(act_wr_addr_final), .wr_en(act_wr_en_final),
    .wr_data(act_wr_data_final) );
```

The **write port is already arbitrated engine-first** across the engine output
and all 34 input loaders (`nn2rtl_top_engine.v:2032`):
```verilog
// Priority: engine > ldr0 > ldr1 > ... > ldr33.
assign ldr0_wr_grant = ldr0_wr_req & ~(engine_act_out_wr_en);
assign ldr1_wr_grant = ldr1_wr_req & ~(engine_act_out_wr_en | ldr0_wr_req);
...
```
and the loaders are themselves fed by spatial-chain outputs gated by
`spatial_run` (e.g. `nn2rtl_top_engine.v:1367`,
`.in_valid(n4_2_valid_out & spatial_run)`).

So the structural facts are:
- **Write-port arbitration already exists and is engine-priority.** A loader and
  the engine can never both write the BRAM in the same cycle — the loader's grant
  is suppressed whenever `engine_act_out_wr_en` is high (line 2032). Dropping the
  spatial stall just lets a loader *attempt* writes during the engine window; the
  arbiter denies any that collide, and the un-granted beat stays in the loader's
  upstream because the spatial handshake backpressures it.
- **Read/write are different ports.** The engine read (`rd_addr`) and any loader
  write (`wr_addr`) are physically independent ports of the dual-port memory, so
  there is no port contention between an engine read and a loader write.
- **The genuine, un-arbitrated risk = read-during-write to the SAME ADDRESS.**
  The engine reads its *current* dispatch's input region while a loader fills a
  *future* dispatch's region. Loader base addresses are NOT all disjoint
  (`u_ldr_node_conv_814 BRAM_BASE_ADDR=0`, `u_ldr_node_conv_816 BRAM_BASE_ADDR=4096`,
  `u_ldr_node_conv_820 BRAM_BASE_ADDR=4096` — line 1363/1383/1403 show address
  windows that reuse the 24576-deep space across dispatches). The
  `current_loaded` interlock (`nn2rtl_top_engine.v:2161`,
  `current_loaded = all_loaded[sched_dispatch_idx]`) guarantees the engine never
  *starts* a dispatch until that dispatch's loader has finished, but it does NOT
  prove that a *different, later* loader, running ahead during the engine window,
  cannot overwrite a region the in-flight engine dispatch is still reading. The
  current serialization (`spatial_stall=1`) makes this impossible by construction;
  removing it removes that guarantee.

### A1 verdict

**Conditionally SAFE, NOT byte-exact-by-construction.** The write-port collision
(simultaneous writes) is already handled by the existing engine-priority arbiter,
so the simple corruption mode is covered. The residual risk is a
**read-during-write address aliasing** hazard: a future-dispatch loader running
ahead during the engine compute window could touch the activation region the
in-flight dispatch is still reading, because loader address windows are reused
across the 24576-word space and only the *consumed* dispatch is interlocked by
`current_loaded`. This cannot be discharged by inspection — it depends on the
exact per-dispatch address windows vs the engine's read schedule.

- **Speedup if safe:** removes the engine-serial term from the critical path.
  e2e `39.40M → max(35.61M spatial, 3.79M engine) ≈ 35.61M` cyc, i.e.
  **−3.79M (≈ −9.6%)**, giving **1.40 fps @50 MHz** (5.62 fps @200 MHz hypo).
  Modest on its own — A1 is only a ~10% win because the engine is already 3× under
  the spatial path. **A1's real value is as a multiplier on A2:** once A2 cuts the
  spatial path below the 3.79M engine cost, A1 is what hides the engine entirely
  (see §Combined).

### A1 required verification (e2e-gated, no Vivado)

Byte-exactness here is a **dataflow/timing** property, not an arithmetic one, so
node-level checks cannot cover it. Required gate:

1. Apply both edits (scheduler line 1130 + top line 443) on a branch; keep the
   `nn2rtl_top.v` patches intact (per memory
   `project_top_v_is_patched_not_regenerated` — never regenerate the top).
2. Run the **full MBv2 e2e value harness** to completion:
   `tb/mbv2_top_value_tb.cpp` via `scripts/run_mbv2_top_value.ts` (24-bit RGB in,
   8000-byte logits out), Verilator `--x-initial 0`, against a FRESH golden.
   Require `mismatch == 0`.
3. If e2e shows any mismatch, the read-during-write aliasing hazard is real →
   fall back to a **bank-reservation gate**: keep `spatial_stall=0` in
   `S_WAIT_DONE` but additionally suppress the specific loader(s) whose
   `BRAM_BASE_ADDR` window overlaps the in-flight dispatch's read window (the
   scheduler already carries `skip_bank_reserved_mask` /
   `input_bank_sel` / `output_bank_sel` ROMs, currently unused —
   `nn2rtl_top_engine.v:2087` ties them off; they are the intended hook for
   exactly this). That restores byte-exactness while still overlapping the
   non-conflicting loaders.

---

## A2 — Parallelize the spatial 3×3 datapath (THE real lever, 35.6M cyc)

This is an atomic, byte-exact-entangled change: RTL + `compute_conv2d_latency_cycles`
+ pattern docs + goldens must move together (memory `feedback_atomic_arch_changes`).

### Decisive structural fact: the depthwise datapath is per-channel, tap-serial AND lane-serial

The 17 depthwise convs do **not** instantiate `conv_datapath`/`conv_datapath_mp_k`
as a sub-block; they carry an **inlined** depthwise MAC (see
`output/mobilenet-v2/rtl/node_conv_812.v:155`+, header *"inline depthwise
datapath that REPLACES conv_datapath's cross-channel adder tree with a
per-channel 9-tap dot product (no IC-axis reduction)"*). The inner loop is:

`node_conv_812.v:289` (ST_MAC) walks `k_counter` 0..8 (the 9 taps) and
`lane_counter` 0..3 (MP=4 output channels), **one multiply per cycle**:
```verilog
if (lane_counter == 2'd3) begin
    lane_counter <= 2'd0;
    if (k_counter == 4'd8) mac_done_issuing <= 1'b1;
    else                   k_counter <= k_counter + 4'd1;
end else lane_counter <= lane_counter + 2'd1;
```
giving `per_pass = MP·K_TOTAL + 6 = 4·9 + 6 = 42`, `OC_PASSES = ceil(C/4)`
(`node_conv_812.v:163`). The single product is at line 264:
`mul_q <= $signed(weight_q) * $signed(tap_q);`.

So for a depthwise channel `c`, the output is
`out[c] = sat(requant_c(bias[c] + Σ_{k=0..8} w[c,k]·tap[c,k]))` — a **9-tap
reduction, no IC axis.** The two parallelism axes are therefore:
- **tap-parallel (MP_K)** — compute the 9-tap reduction tree P_K-wide → `ceil(9/P_K)`
  cycles instead of 9. Directly attacks the `K_TOTAL` serial term.
- **lane-parallel (MP)** — compute more output channels per pass → fewer
  `OC_PASSES`. `per_pass ∝ MP` and `OC_PASSES ∝ 1/MP`, so for the *tap-serial*
  baseline the product `MP·K_TOTAL·OC_PASSES ≈ K_TOTAL·C` is **MP-independent**
  (memory `golden_impl.py:74`; this is exactly why raising MP alone never helped).
  MP only helps once the tap axis is already parallel (it then trims the residual
  `+6` overhead amortization and the `ceil` rounding).

### Option 1 — tap-parallel (MP_K) reduction tree. **THE EXISTING, PROVEN MECHANISM.**

`rtl_library/conv_datapath_mp_k.v` **already implements exactly this** and is
byte-exact-proven on the ResNet spatial convs (header lines 1–25; the Phase-2
ResNet work shipped it). It does `MP × MP_K` multipliers per cycle, tree-sums the
`MP_K` products per lane into a width-safe `TREE_W = PROD_W + clog2(MP_K)`
intermediate (`conv_datapath_mp_k.v:65`), accumulates into `ACC_W` (line 66),
and requants ONCE after the full reduction. Per-pass cost
`MP·(K_TOTAL/MP_K) + 6` (line 4, constraint `K_TOTAL % MP_K == 0`).

The MBv2 depthwise just needs the *same* transform applied to its **inline**
per-channel datapath (it can't drop in `conv_datapath_mp_k` verbatim because that
module does the cross-IC window layout; the depthwise reads `chan_window_flat`,
one channel per cycle). But the arithmetic pattern, width sizing, and the
byte-exact argument transfer 1:1.

**Cycle counts (recomputed):**

| config | stem cyc | depthwise cyc | spatial cyc | speedup | DSP/module |
|---|---:|---:|---:|---:|---:|
| MP=4, MP_K=1 (today) | 11,440,128 | 24,169,152 | 35,609,280 | 1.00× | 4 |
| MP=4, **MP_K=3** | 4,214,784 | 10,358,208 | 14,572,992 | 2.44× | 12 |
| MP=4, **MP_K=9** | 1,806,336 | 5,754,560 | 7,560,896 | **4.71×** | 36 |

For the depthwise K_TOTAL=9, MP_K=9 fully unrolls the 9-tap reduction → 1 cycle
per OC-pass-step group. For the stem K_TOTAL=27, MP_K=9 gives `ceil(27/9)=3`
group cycles (27 is ÷9, clean). **MP_K=9 divides both** (9 and 27) — no tail mask
needed for the spatial path. This is the natural P for 3×3 (the kernel is
exactly 9 taps).

Optionally also raise lanes once tap-parallel:

| config | spatial cyc | speedup | DSP/module |
|---|---:|---:|---:|
| MP=4, MP_K=9 | 7,560,896 | 4.71× | 36 |
| MP=8, MP_K=9 | 5,533,472 | 6.44× | 72 |
| MP=16, MP_K=9 | 4,519,760 | 7.88× | 144 |

### Option 2 — engine-dispatch the 3×3s onto the shared 256-OC engine

**NO-GO for the 17 depthwise (the 24.17M bulk). GO only for the stem.**

The engine's `mac_array` broadcasts **one** activation byte to all 256 lanes:
`output/rtl/engine/mac_array.v:40` (`input wire [7:0] act_byte`) and line 84
(`assign a_byte = $signed(act_byte);` — the SAME byte in every one of the 256
lanes, line 77 `for (lane=0; lane<256...)`). That is correct for a **group=1**
conv (all output channels reduce over the same input-channel walk), but a
**depthwise** output channel `c` must read **only** input channel `c` — i.e. the
256 lanes need 256 *different* activation bytes in the same cycle. The engine
cannot supply that without a structural rewrite of its activation feed (256-wide
distinct-byte fan-in instead of a single broadcast byte). So depthwise on the
engine is not a parameter change; it is a new datapath, defeating the "merge into
the shared engine" premise.

The **stem** (group=1, 3×3, IC=3 OC=32) *does* map cleanly. Hypothetical engine
cost (per-pixel = `OC_PASSES·(ceil(K_TOTAL/P)+OVH)`, OVH≈12, K_TOTAL=27):

| | engine-add cyc | new engine-serial total | tail mask? |
|---|---:|---:|---|
| stem only, engine P=1 | 489,216 | 3.79M + 0.49M = 4.28M | K_TOTAL=27 not ÷ K_PAR=4 → **mask MANDATORY** |
| stem only, engine P=4 | 238,336 | 3.79M + 0.24M = 4.03M | same |

Even dispatching the stem keeps engine-serial (≈4.0–4.3M) below A2-Option-1's
spatial result (≈7.56M at MP_K=9), so it would not be the limiter — but it
**forces** the per-lane tail-zero valid mask (K_TOTAL=27 ∤ 4) into the engine,
which the K-parallel hardened contract flags as the top corruption risk. Net: the
stem-on-engine is only attractive *if* the engine is already getting the K-parallel
rework anyway; on its own it is not worth the mandatory mask. The depthwise cannot
go on the engine at all.

### A2 RECOMMENDATION: **Option 1, MP_K = 9** (tap-parallel reduction tree on the inline depthwise datapath AND the stem `conv_datapath`)

- 4.71× on the whole spatial path (35.61M → 7.56M) with MP kept at 4.
- DSP cost: `MP·MP_K = 36` multipliers per spatial module × 18 (17 DW + stem) =
  **648 MAC DSP**. Add the existing engine 512 → ~1,160 spatial+engine DSP. The
  fit projection (`mbv2_u250_fit_projection.md`) lists the whole design at
  **~1,345 DSP = 10.9%** with the spatial DW currently at 1 DSP each (17 total)
  and stem 1; Option 1 adds `648 − 18 = +630` DSP → design total
  **≈ 1,975 DSP = 16.1% of 12,288 — still far under 80%.** DSP headroom is huge
  (the whole point: the limiter is cycles, not multipliers).
- BRAM/URAM: **neutral.** The weight ROM is re-laid-out into `MP·MP_K`-wide words
  (same total weight bits, wider×shallower) exactly as `conv_datapath_mp_k`'s
  `weights_wide[NUM_WIDE_WORDS]` (line 92), and the line buffers are unchanged.
  No change to the §2 BRAM/URAM projection (1,013 RAMB36 / 128 URAM).
- LUT/FF: the reduction tree adds `(MP_K−1)` small adders per lane = `MP·(MP_K−1)`
  ≈ 32 adders/module — negligible vs the device 1.7M LUT.

If more is needed after MP_K=9, raise MP to 16 (MP_K=9) for 7.88× at 144 DSP/module
(2,592 spatial DSP, design ≈ 21% DSP — still under 80%). Recommended sequencing:
land MP_K=9 first (clean, kernel-exact, no tail mask), measure, then MP if the
target fps demands it.

### A2 byte-exactness contract (MUST hold — mirrors the proven `conv_datapath_mp_k`)

1. **Accumulator width unchanged in semantics.** Keep the per-lane INT accumulator
   wide enough: `TREE_W = PROD_W + clog2(MP_K)` for the 9-product tree
   (`conv_datapath_mp_k.v:65`), `ACC_W = TREE_W + clog2(K_GROUPS)`
   (line 66). For depthwise K_GROUPS = 9/9 = 1, so `ACC_W = 16 + clog2(9) = 20`
   bits — covers `9·(127·127) ≈ 145k` with margin. **Integer addition is
   associative**: the tree-sum of the 9 products equals the serial sum bit-for-bit
   (no intermediate truncation because `prod_w` is a `signed [PROD_W-1:0]` typed
   reg per `conv_datapath_mp_k.v:185-191` — do NOT wrap the multiply in an outer
   `$signed()`, that self-determines to 8-bit and truncates; this exact bug is
   documented at lines 178-183).
2. **Requant applied ONCE, after the full reduction.** The depthwise already does
   BIAS → SCALE → OUTPUT (`node_conv_812.v` ST_BIAS/ST_SCALE/ST_OUTPUT). Keep that
   sequence; only the ST_MAC issue count changes (9 → 1 group cycle). The
   `(scaled + SCALE_ROUND_BIAS) >>> SCALE_SHIFT` round-half-up + INT8 saturate
   (`node_conv_812.v:327` ST_SCALE, and the v_tmp clamp in ST_OUTPUT) is
   **byte-identical and must not be touched**.
3. **Latency contract moves atomically.** New depthwise per-pass = `MP·1 + 6`
   (MP_K=9 → 1 group cycle) = 10; stem per-pass = `MP·3 + 6` = 18. Update
   `scripts/golden_impl.py::compute_conv2d_latency_cycles` to the
   `MP·ceil(K_TOTAL/MP_K) + CONV_PIPELINE_STAGES` form (the formula already has
   the spatial window-fill term; only the `pass_cycles` line at
   `golden_impl.py:168` changes from `mp*k_total+6` to
   `mp*ceil(k_total/mp_k)+6`). The window-fill prefix is unchanged.
4. **No tail mask needed for MP_K=9** on the spatial path (9÷9, 27÷9 both clean).
   Add the directed non-divisible test anyway, because the moment any future 5×5 /
   7×7 / non-9 kernel appears the tail-zero of the surplus tree lanes becomes
   mandatory.

### A2 atomic change set (cite the regen chain — memory `feedback_regen_must_rebuild_engine_maps`)

**RTL (move together):**
- `rtl_library/conv_datapath_mp_k.v` — already the reference; reuse its
  tree-sum/width pattern. (No edit needed if the depthwise inlines the same
  pattern; if instead the depthwise is refactored to *instantiate* a per-channel
  variant, that variant lives here.)
- The 17 depthwise node wrappers `node_conv_{812,818,824,830,836,842,848,854,
  860,866,872,878,884,890,896,902,908}.v` — replace the inline tap-serial ST_MAC
  loop with the MP_K=9 tree-sum (per-channel; reads the same `chan_window_flat`
  9 bytes at once instead of one tap/cycle). Weight ROM re-laid-out MP_K-wide.
- `output/mobilenet-v2/rtl/node_conv_810.v` (stem) — switch its `conv_datapath`
  instance to `conv_datapath_mp_k` with `MP_K=9` (it already exposes the full
  cross-IC window; `conv_datapath_mp_k` consumes exactly that `window_flat`).
- Write all `.v` via the `write_verilog` MCP tool (never write `.v` directly —
  CLAUDE.md core rule).

**Python latency formula:**
- `scripts/golden_impl.py` — `compute_conv2d_latency_cycles` `pass_cycles`
  (line 168) → `mp*ceil(k_total/mp_k)+CONV_PIPELINE_STAGES`; thread an `mp_k`
  arg (default 1 = backward-compatible). The depthwise per-channel variant uses
  `k_total = KH*KW`, `mp_k = 9`.

**Pattern docs:**
- `knowledge/patterns/03_conv3x3_pad1.md` (and the depthwise pattern doc, if
  distinct) — document the MP_K reduction-tree contract + the
  `signed prod_w`-not-outer-`$signed()` invariant.

**Weight repack:**
- `scripts/repack_weights_wide.py --mp-k 9` per affected conv (the depthwise C-major
  per-channel 9-wide layout; the stem MP·9 layout). The `--batch` path skips the
  hand-managed spatial convs — repack each affected node individually with
  `--mp-k 9` and matching `--wgt-bits` (8 for MBv2 INT8).

**Goldens / regen chain (run ALL — skipping any silently corrupts via stale
goldens, memory `feedback_regen_must_rebuild_engine_maps`):**
1. `generate_golden` (if weights regenerated).
2. `build_bias_memory_map.py --network mobilenet-v2` (engine bias — unchanged by
   A2 but re-run to be safe).
3. `build_scale_memory_map.py` / `build_spatial_scale_mems.py` (spatial scale.mem
   per conv).
4. Per-conv `repack_weights_wide.py --mp-k 9` for the 18 spatial convs.
5. `refresh_final_golden.py` + `npx tsx scripts/rebuild_contract_goldens.ts`.
6. Verify each map's mtime > the `generate_golden` mtime before trusting any e2e.

**Verification (no Vivado):**
- Per-module byte-exact: `npx tsx scripts/_verify_mbv2_variant.ts
  output/mobilenet-v2/rtl/node_conv_896.v node_conv_896 <sidecar>` (widest C=960)
  and `node_conv_812.v` (narrow C=32) and the stem `node_conv_810.v` — require
  `mismatch_count == 0`, `max_error == 0`, AND `timing_pass == true` (the new
  pipeline_latency_cycles must match the rewritten datapath).
- Then full e2e: `tb/mbv2_top_value_tb.cpp` / `scripts/run_mbv2_top_value.ts`,
  `mismatch == 0` vs a FRESH golden.

---

## Combined A1 + A2 projection

With **A2 Option 1 (MP_K=9, MP=4)** the spatial path drops to **7.56M cyc** —
now *above* the 3.79M engine, so the engine still matters but is close. Then:

| config | spatial | engine | e2e model | e2e cyc | **fps @50 MHz** | fps @200 MHz (hypo) |
|---|---:|---:|---|---:|---:|---:|
| baseline (P=1, serialized) | 35.61M | 3.79M | serial sum | 39.40M | **1.27** | 5.08 |
| A2 only (MP_K=9, engine serialized) | 7.56M | 3.79M | serial sum | 11.35M | **4.40** | 17.6 |
| **A1 + A2 (MP_K=9, engine overlapped)** | 7.56M | 3.79M | `max(spatial, engine)` | **7.56M** | **6.61** | **26.5** |
| A1 + A2 (MP_K=9, MP=16) | 4.52M | 3.79M | `max(spatial, engine)` | 4.52M | 11.06 | 44.3 |

- A2 alone is the dominant lever (1.27 → 4.40 fps @50 MHz, **3.5×**).
- A1 on top of A2 hides the 3.79M engine entirely (4.40 → **6.61 fps @50 MHz**,
  a further 1.5×) — this is where A1 finally pays off, because A2 has pulled the
  spatial path down to the engine's neighborhood.
- The `max(spatial, engine)` model assumes A1 achieves *full* overlap. In
  practice the overlap is partial (scheduler still serializes the AXI-write /
  load / pulse phases), so treat the A1+A2 fps as an upper bound; the realized
  number lands between the "A2 only" and "A1+A2" rows. The honest, committed
  target is **≈ 4.4–6.6 fps @50 MHz**.

### Area cross-check (must stay < 80% on all 6 resources, vs `mbv2_u250_fit_projection.md`)

| Resource | Fit-projection baseline | A2 Opt1 (MP_K=9, MP=4) delta | new total | % U250 | < 80%? |
|---|---:|---|---:|---:|---|
| LUT | ~1,064,000 (61.6%) | +`MP·(MP_K−1)`·18 adders ≈ a few k | ~1.07M | ~62% | yes |
| FF | ~759,000 (22.0%) | +tree pipeline regs, small | ~0.77M | ~22% | yes |
| DSP48E2 | ~1,345 (10.9%) | +630 (648 spatial − 18 today) | **~1,975** | **16.1%** | yes |
| RAMB36 | ~1,013 (37.7%) | neutral (weight ROM wider×shallower, same bits) | ~1,013 | 37.7% | yes |
| URAM288 | 128 (10.0%) | neutral | 128 | 10.0% | yes |

All six stay well under 80%. The depthwise LUT figure in the fit projection is
already a pessimistic linear-by-C upper bound; the reduction tree adds adder LUTs
but the FSM/scheduler/window logic (the bulk of the per-module LUT) is unchanged,
so the relative LUT growth is small. At MP=16,MP_K=9 the DSP would be ~21% — still
the binding cross-check is comfortable.

---

## Summary verdicts

**A1 (overlap):** Conditionally safe, **not** byte-exact-by-construction. The
simultaneous-write collision is *already* handled by the existing engine-priority
write arbiter (`nn2rtl_top_engine.v:2032`); the residual risk is a
read-during-write *address-aliasing* hazard from future-dispatch loaders running
ahead during the engine window (loader address windows are reused across the
24576-word BRAM and only the in-flight dispatch is interlocked by
`current_loaded`). Requires the **two** edits (scheduler line 1130 `spatial_stall=0`
AND top line 443 dropping the `engine_busy` term) and a **completed e2e value
check** to discharge the hazard; fall back to a bank-reservation gate (the unused
`skip_bank_reserved_mask` ROM is the hook) if e2e mismatches. Speedup alone ≈ +10%
(1.27 → 1.40 fps @50 MHz); its real value is hiding the engine after A2.

**A2 (parallelize spatial):** **Option 1, tap-parallel MP_K = 9** — reuse the
already-byte-exact-proven `conv_datapath_mp_k` reduction-tree pattern on the inline
depthwise datapath and the stem. **4.71× on the spatial path** (35.61M → 7.56M),
DSP **16.1%** (640 added), BRAM/URAM/LUT/FF all neutral-to-negligible. **Reject
Option 2 (engine-dispatch)** for the 17 depthwise: the engine broadcasts ONE
activation byte to all 256 lanes (`mac_array.v:40,84`), structurally incompatible
with per-channel depthwise; only the stem maps, and even that forces the
mandatory K_TOTAL=27∤4 tail mask for no system win.

**Combined A1 + A2 (MP_K=9):** **≈ 4.4 fps @50 MHz** (A2 only) to **≈ 6.6 fps
@50 MHz** (full A1 overlap), up from 1.27 — a **3.5–5.2× system speedup**. At
200 MHz (hypothetical, timing-gated): ≈ 17.6–26.5 fps. Realized number lands
between the two depending on overlap quality. All six resources stay < 80%.

**Doc written:** `docs/agent_tasks/mbv2_spatial_throughput_roadmap.md`.
**No RTL edited, no Vivado run, no long sim run.**

---

## A2 adversarial review verdict (hardened)

Read-only review (3 refutation passes, independently re-verified against the
on-disk RTL, `layer_ir.json`, and `golden_impl.py`). **No RTL edited, no Vivado,
no long sim.** All numbers below were reproduced bottom-up, not inherited.

### 1. Which claims survived / broke

- **`depthwise_compat` (claim_holds=false): SURVIVES on its core point, but its
  latency-formula sub-claim is itself REFUTED.**
  - SURVIVES: `conv_datapath_mp_k.v` is a DENSE (group=1) datapath. Its
    accumulator (one lane-acc over the full `K_TOTAL=IC*KH*KW` window, line 228),
    its `tap_at` IC-axis indexer (lines 142-152, `ic_idx=k_lin/(KH*KW)`), and its
    `weights_wide` addressing (line 137, full IC depth) all hardwire a cross-IC
    reduction. It consumes the WIDE `window_flat`; the 17 MBv2 depthwise wrappers
    consume the NARROW `chan_window_flat` (72 bits, one channel, `line_buf_window`
    `EXPOSE_FULL_WINDOW(0)`, `node_conv_812.v:69,134,148`). **You cannot
    instantiate `conv_datapath_mp_k` verbatim for the 17 depthwise** — doing so
    computes `out[c]=Σ over ALL channels` (catastrophically wrong). Confirmed by
    direct read of `node_conv_812.v` (inline DW datapath, `K_TOTAL=KH*KW=9`,
    line 34) and `node_conv_810.v` (genuine `conv_datapath` instance, group=1,
    `K_TOTAL=IC*KH*KW=27`, line 25).
  - **REFUTED sub-claim:** the pass asserts `golden_impl.py:152`
    (`k_total=ic_i*kh_i*kw_i`) models `64*9` taps for depthwise and therefore
    needs a depthwise-specific `k_total=KH*KW` branch. **This is wrong.** The
    depthwise ONNX weight tensor is `[C,1,3,3]` (verified in `layer_ir.json`:
    `node_conv_812` wshape `[32,1,3,3]`, groups=32), so `ic_i=1` in the weight
    tensor and the formula ALREADY computes `k_total=1*3*3=9`. The formula
    reproduces every IR `pipeline_latency_cycles` EXACTLY as-is (verified:
    `node_conv_810`→1140, `node_conv_812`→452, `node_conv_896`→10091). **No
    line-152 depthwise branch is needed; the plan's single-line edit at line 168
    (`mp*ceil(k_total/mp_k)+6`) IS sufficient** — provided `mp_k` is threaded from
    the frontend.

- **`byte_exact_mpk9` (claim_holds=true): SURVIVES.** Tree-sum == serial sum
  (integer-add associativity), single requant after full reduction, `ACC_W=24`
  headroom (9·127² = 145k ≪ 2²³), per-product `signed prod_w` (NOT outer
  `$signed(a*b)` — the documented 8-bit-truncation trap, `conv_datapath_mp_k.v`
  :176-183). Stem 27=3×9 cross-IC grouping is correct (3 group-cycles in ONE
  OC pass, not 3 passes). The three hardening hazards are REAL and promoted to
  MUST items below: (A) pipeline drain-depth re-derivation for the hand-written
  DW tree, (B) the `$signed()` truncation trap, (C) `WGT_BITS(8)` stem override
  (default is 4 = INT4 → half-width garbage).

- **`numbers_regen` (claim_holds=true): SURVIVES, with two real regen defects
  CONFIRMED.** All cycles/DSP/fps reproduced to the digit (below). The two
  defects: (1) `refresh_final_golden.py` defaults to `node_relu_48` (ResNet),
  verified at `refresh_final_golden.py:51`; MBv2 final = `node_linear` (gemm,
  8000-bit logits), verified as `layer_ir.json` last layer → must call
  `refresh_final_golden.py node_linear 32`. (2) `onnx_frontend.py` is the IR
  producer that bakes `pipeline_latency_cycles` via `compute_conv2d_latency_cycles`
  (verified `onnx_frontend.py:1429,1719,1755-1763`) → it MUST be in the atomic
  change set to thread `mp_k` per spatial layer, else the baked plc in
  `layer_ir.json` will not match the rewritten datapath (`timing_pass` fails).

### 2. GO / NO-GO + mechanism

**GO — but the mechanism is SPLIT, not the single "reuse conv_datapath_mp_k" the
plan's headers imply:**

- **Stem `node_conv_810` (group=1): `conv_datapath_mp_k` reuse** (drop-in proven
  module). Switch the `conv_datapath` instance to `conv_datapath_mp_k` with
  **explicit** `.MP_K(9), .WGT_BITS(8), .SCALE_MULT(18143), .SCALE_SHIFT(23),
  .SCALE_PATH("")`. Low risk.
- **17 depthwise `node_conv_{812..908}`: inline-MAC MP_K (hand-written tree),
  NOT module reuse.** Replace each wrapper's tap-serial `ST_MAC` k-loop (0..8,
  one multiply/cycle) with a 9-wide combinational multiplier + per-lane tree-sum
  reading all 9 bytes of `chan_window_flat` at once. The LANE axis stays serial
  (`channel_select` still cycles 0..3 per group, one channel-window per cycle);
  only the TAP axis parallelizes. `K_GROUPS = 9/9 = 1` → no k-group accumulate
  loop (simpler than the dense module). Medium risk (hand-written reduction
  replacing a hand-tuned 3-stage drain pipeline).

The plan's Summary/§A2-recommendation wording ("reuse the already-byte-exact
`conv_datapath_mp_k`") OVERSTATES the depthwise case — the plan BODY (lines
245-249) already concedes "can't drop in verbatim". Promote that caveat into the
headers so "reuse" is not read as drop-in instantiation for the 17 DW.

### 3. Corrected byte-exact contract + corrected numbers

**Byte-exact contract (MUST hold):**
1. **Stem (module reuse):** explicit `.WGT_BITS(8)` — the `conv_datapath_mp_k`
   default is 4 (INT4); without the override the stem reads half-width garbage.
   Per-tensor scale via `SCALE_MULT/SCALE_SHIFT` + `SCALE_PATH("")` (the legacy
   fallback at `conv_datapath_mp_k.v:101-103`).
2. **Depthwise (hand-written tree):** (a) per-product `signed [PROD_W-1:0]` reg,
   NEVER outer `$signed(a*b)`; (b) re-derive `mac_done_issuing` + the q1/q2 drain
   to the new 1-group-cycle-per-channel depth (the current trigger fires at
   `k_counter==8`, `node_conv_812.v:303` — that disappears); (c) keep
   `channel_select` cycling 0..3 within a group, reading all 9 `chan_window_flat`
   bytes per cycle; (d) `ACC_W≥20` (24 today is fine); (e) `ST_BIAS/ST_SCALE/
   ST_OUTPUT` byte-identical (round-half-up `(scaled+ROUND)>>>SHIFT` + INT8 sat).
3. **Latency:** `golden_impl.py:168` → `mp*ceil(k_total/mp_k)+CONV_PIPELINE_STAGES`,
   thread `mp_k` (default 1). NO line-152 depthwise branch needed (weight tensor
   already gives `ic=1`→`k_total=9` for DW). Atomic with the RTL.
4. **No tail mask for MP_K=9** (9÷9, 27÷9 clean). Add a directed non-divisible
   (e.g. 5×5) test to lock the contract for future kernels.

**Corrected numbers (reproduced bottom-up — all MATCH the plan):**

| | stem cyc | DW cyc | spatial cyc | speedup |
|---|---:|---:|---:|---:|
| MP=4, MP_K=1 (today) | 11,440,128 | 24,169,152 | 35,609,280 | 1.00× |
| MP=4, **MP_K=9** | 1,806,336 | 5,754,560 | **7,560,896** | **4.71×** |

- **fps @50 MHz (real clock):** baseline 1.27 → **A2-only 4.40** → **A1+A2 6.61**
  → A1+A2(MP=16) 11.06. (200 MHz hypothetical/timing-gated: 5.08 → 17.6 → 26.5 →
  44.3.) Honest committed target **≈ 4.4–6.6 fps @50 MHz**.
- **DSP:** MP_K=9 = 36 MAC/module × 18 = 648; delta = 648−18 = **+630**; design
  total **1,975 = 16.1%** of 12,288. MP=16,MP_K=9 = **3,919 = 31.9%** (plan said
  ~21% — the plan UNDERSTATES this; recompute: 144×18=2592, +2574, 1345+2574=3919
  = 31.9%, still well under 80%).
- **all_under_80 = TRUE** (MP_K=9, MP=4): LUT ~60%, FF ~22%, DSP 16.1%, RAMB36
  37.7%, URAM 10.0%. Also TRUE at MP=16,MP_K=9 (DSP 31.9%).
- BRAM is **net-favorable, NOT byte-neutral per-module** (the `numbers_regen`
  pass's finding): width-bound 288b-wide shallow ROMs inflate small DW (1→4
  tiles) and shrink large DW (17→4), aggregate ~−63 RAMB36. Restate the plan's
  "neutral" as "net-favorable, per-module non-uniform" AND add a synth-gated
  check that the 288b shallow ROMs map to BRAM not LUT-RAM (the documented
  scale_rom×45 LUTRAM blowup risk) — UNVERIFIED by synth, flag it.

### COMPLETE atomic edit + regen list (corrected, with the missed steps)

**RTL (via `write_verilog` MCP only):**
- 17 DW wrappers `node_conv_{812,818,824,830,836,842,848,854,860,866,872,878,884,
  890,896,902,908}.v` — inline MP_K=9 tree (per contract item 2).
- Stem `node_conv_810.v` — `conv_datapath`→`conv_datapath_mp_k` with explicit
  `.MP_K(9), .WGT_BITS(8), .SCALE_MULT(18143), .SCALE_SHIFT(23), .SCALE_PATH("")`.

**Python (atomic with RTL — `feedback_atomic_arch_changes`):**
- `scripts/golden_impl.py:168` — `mp*ceil(k_total/mp_k)+CONV_PIPELINE_STAGES`,
  thread `mp_k` (default 1).
- **`scripts/onnx_frontend.py` — MISSED by the plan. Thread `mp_k=9` per spatial
  layer into the `compute_conv2d_latency_cycles` call (lines 1429-1434, 1755-1763)
  so the baked `pipeline_latency_cycles` matches the new datapath.**

**Weight repack (per-conv, NOT `--batch`):**
- `repack_weights_wide.py --mp-k 9 --wgt-bits 8 --k-total 9` for each of the 17
  DW (C-major per-channel layout; the existing oc_group-major packer at lines
  82-95 produces the correct `weight_word[(lane*9+kpos)*8]` layout the inline
  tree reads). `--mp-k 9 --wgt-bits 8 --k-total 27` for the stem.

**Goldens / regen chain (run ALL — `feedback_regen_must_rebuild_engine_maps`):**
1. `generate_golden` (if weights regenerated).
2. `build_bias_memory_map.py --network mobilenet-v2` (re-run to be safe).
3. `build_scale_memory_map.py` / `build_spatial_scale_mems.py`.
4. Per-conv `repack_weights_wide.py --mp-k 9 --wgt-bits 8` for the 18 convs.
5. **`refresh_final_golden.py node_linear 32` — MISSED by the plan (bare
   `refresh_final_golden.py` refreshes the ResNet `node_relu_48` default → stale
   MBv2 final golden, silent e2e fail).** Then
   `npx tsx scripts/rebuild_contract_goldens.ts`.
6. Verify each map's mtime > `generate_golden` mtime before any e2e.

**Verification (no Vivado):** per-module byte-exact (`mismatch==0`, `max_error==0`,
`timing_pass==true`) on `node_conv_812` (C=32 narrow), `node_conv_896` (C=960
widest), `node_conv_810` (stem) BEFORE e2e; then full e2e via
`scripts/run_mbv2_top_value.ts` (`--x-initial 0`) vs a FRESH golden, `mismatch==0`.

### 4. Sequencing — A2 MUST run AFTER engine P=1 verify + clean e2e land

**CONFIRMED.** A2 rewrites the spatial datapath AND triggers the full regen chain
(generate_golden → bias/scale maps → repack → refresh_final_golden →
rebuild_contract_goldens). Per `project_top_v_is_patched_not_regenerated` and the
standing "make MBv2 fully Vivado-ready, e2e may be the only failing thing"
directive, the golden state is IN-FLIGHT. Running A2's regen now would overwrite
the goldens the engine-P=1 e2e is being validated against, destroying the ability
to attribute any new mismatch. **Gate A2 behind: (i) engine P=1 verify GREEN, and
(ii) a clean baseline e2e (`mismatch==0` vs FRESH golden) LANDED and backed up.**
A2 is a parallel-prep / staged-branch activity until both gates are green — do NOT
run its regen chain against the live goldens.

**Net verdict:** GO. Mechanism = SPLIT (stem: `conv_datapath_mp_k` reuse; 17 DW:
inline-MAC MP_K hand-written tree). 4.71× spatial, 16.1% DSP, all resources <80%.
Two regen steps were missing from the plan (`onnx_frontend.py` in the atomic set;
`refresh_final_golden.py node_linear 32`). The plan's "needs a depthwise k_total
branch in golden_impl.py:152" is itself wrong — the weight tensor's `ic=1` makes
the existing formula correct for DW. Sequence AFTER engine-P=1 + clean e2e.

---

## A2 EXECUTION (staged, greenlight-ready)

**Status 2026-06-02:** PoC PROVEN byte-exact; full execution STAGED and
one-command-ready. **NOT applied** — the all-spatial e2e is RUNNING on the live
depthwise wrappers + their goldens, and A2's regen chain mutates those shared
goldens. Apply only after that e2e lands clean and is backed up (per
`feedback_regen_must_rebuild_engine_maps` / `project_top_v_is_patched_not_regenerated`).

### PoC outcome — mechanism proven byte-exact (no live file touched)

The MP_K=9 tap-parallel transform was applied to a SCRATCH copy of the C=32
depthwise reference (`scratch/node_conv_812_mpk9.v`) and verified against the SAME
live golden the baseline uses (`output/mobilenet-v2/goldens/node_conv_812.gold{in,out}`)
via `scripts/_verify_mbv2_variant.ts`:

```
status=pass  mismatch_count=0  max_error=0  exact_match_count=3,211,264/3,211,264
timing_pass=true  timing_actual_cycles=196 == expected 196
```

- **Mechanism (17 DW):** replace the tap-serial `ST_MAC` k-loop (k_counter 0..8,
  one multiply/cycle) with 9 parallel taps — `MP_K=9`, `K_GROUPS=K_TOTAL/MP_K=1`,
  so a single issue cycle per lane (no k-group accumulate loop). Each cycle reads
  all 9 `chan_window_flat` bytes + all 9 contiguous weights for the current
  channel, computes 9 products in per-tap `signed [PROD_W-1:0]` regs (NOT outer
  `$signed(a*b)` — the 8-bit-truncation trap is avoided), and a COMBINATIONAL
  tree-sum feeds the existing stage-3 acc add. The q1→q2 valid/lane/oc shift is
  kept at EXACTLY 2 stages (products registered where the baseline registered
  `mul_q`; tree-sum done combinationally in the accumulate stage) so lane
  alignment is bit-for-bit identical. `ACC_W=24` unchanged (TREE_W=20 fits).
  ST_BIAS/ST_SCALE/ST_OUTPUT requant BYTE-IDENTICAL. Byte-exactness rests on
  integer-add associativity: tree-sum of 9 products == serial sum.
- **WEIGHT REORDER for the 17 DW: NONE.** The inline tree reads
  `weights[current_global_oc*K_TOTAL + kk]` for kk=0..8 — exactly the contiguous
  addresses the baseline read serially — from the SAME existing row-major
  `node_conv_*_weights.hex`. `chan_window_flat` already exposes all 9 tap bytes in
  one cycle (`line_buf_window EXPOSE_FULL_WINDOW(0)`). **No `repack_weights_wide`
  is needed for the depthwise.** (This supersedes the review's "repack the 18"
  line — empirically settled by the PoC; only the STEM needs repack, see below.)
- **Per-pass:** DW 42→**10** (`MP*ceil(9/9)+6 = 4*1+6`); first-valid 452→196.
  Throughput (cyc/frame) DW 24,169,152→5,754,560 = **4.20× per-pass**; the
  combined spatial path is **4.71×** (the stem gains more: 27 taps → 3 group
  cycles). First-valid latency speedup is a smaller 2.31× (the 115-cycle
  window-fill prefix is fixed and prefix-dominated for a single output) — but
  throughput / fps follows the 4.20–4.71× ratio.

Artifacts: `scratch/node_conv_812_mpk9.{v,sidecar.json,results.json}`.

### The one-command apply sequence (greenlight order)

> Run from repo root `nn2rtl-repo`. `python=/c/Python313/python`,
> `PATH=/c/Users/User/oss-cad-suite/bin:/c/Users/User/w64devkit/bin`. NO Vivado.

**Step 0 — verify the staged transform without writing (already green):**
```
python scripts/apply_mpk9_depthwise.py --self-check   # transform(812) == proven PoC
python scripts/apply_mpk9_depthwise.py --dry-run       # reports 17 DW + 1 stem = 18 targets
```

**Step 1 — APPLY the RTL transform (backs up all 18 to backups/mpk9_<ts>/ first):**
```
python scripts/apply_mpk9_depthwise.py
```
This rewrites the 17 DW wrappers (inline MP_K=9 tree) and switches the stem
`node_conv_810.v` to `conv_datapath_mp_k` (`.MP_K(9), .WGT_BITS(8),
.SCALE_MULT/.SCALE_SHIFT, .SCALE_PATH(""), WEIGHTS_PATH -> *_weights_mp_k_9.hex`).
It touches NOTHING else (no goldens, no `nn2rtl_top*.v`, no engine, no `n4_*.v`).

**Step 2 — latency formula (atomic with the RTL):** apply
`scratch/golden_impl_mpk.diff` to `scripts/golden_impl.py`
(`compute_conv2d_latency_cycles`: add `mp_k=None` arg; `pass_cycles =
mp*ceil(k_total/mp_k)+CONV_PIPELINE_STAGES`, ceil via `-(-a//b)`, default
mp_k=1 = backward-compatible — VALIDATED to reproduce 810→1140, 812→452,
896→10091 at mp_k=1 and 812→196 at mp_k=9). Then thread `mp_k=9` for the 18
spatial 3x3 convs in `scripts/onnx_frontend.py` so the baked
`pipeline_latency_cycles` matches the rewritten datapath:

```python
# scripts/onnx_frontend.py — _pipeline_latency(), the conv2d branch (~line 1429)
def _conv_mp_k(spec) -> int:
    # 3x3 spatial path uses the MP_K=9 tap-parallel datapath; everything else
    # is the legacy tap-serial datapath (mp_k=1). MBv2's only 3x3 convs are the
    # stem + 17 depthwise — exactly the 18 spatial wrappers apply_mpk9 rewrites.
    if spec.weight is not None and len(spec.weight.shape) >= 4:
        kh, kw = int(spec.weight.shape[2]), int(spec.weight.shape[3])
        if kh == 3 and kw == 3:
            return 9
    return 1
...
    return compute_conv2d_latency_cycles(
        weight_shape, input_shape=spec.input_shape, stride=spec.stride,
        padding=spec.padding, mac_parallelism=_conv_mac_parallelism(spec),
        mp_k=_conv_mp_k(spec),                    # <-- add
    )
```

**Step 3 — STEM weight repack (the ONLY repack; the 17 DW need none):**
```
python scripts/repack_weights_wide.py \
  --input  output/mobilenet-v2/weights/node_conv_810_weights.hex \
  --output output/mobilenet-v2/weights/node_conv_810_weights_mp_k_9.hex \
  --oc 32 --k-total 27 --mp 4 --mp-k 9 --wgt-bits 8 \
  --output-suffix _weights_mp_k_9.hex
```
(The 17 depthwise hex files are consumed unchanged — see PoC "WEIGHT REORDER:
NONE".)

**Step 4 — FULL regen chain, in order (skipping any silently corrupts via stale
goldens — `feedback_regen_must_rebuild_engine_maps`):**
1. `python scripts/generate_golden.py <mbv2 checkpoint>` — only if weights were
   regenerated; A2 does NOT regenerate weights, so this is normally skipped.
2. `python scripts/build_bias_memory_map.py --network mobilenet-v2` (engine bias —
   unchanged by A2 but re-run to be safe).
3. `python scripts/build_scale_memory_map.py --network mobilenet-v2` then
   `python scripts/build_spatial_scale_mems.py` (spatial per-conv scale.mem; A2
   leaves scales unchanged but the chain must re-emit them).
4. STEM repack (Step 3 above) — already done; listed here for ordering.
5. `python scripts/refresh_final_golden.py node_linear 32` (MBv2 final = `node_linear`
   gemm, 8000-bit logits; the bare default `node_relu_48` is ResNet) then
   `npx tsx scripts/rebuild_contract_goldens.ts`.
6. Verify each map's mtime > the `generate_golden` mtime before trusting any e2e.

### Byte-exact verification command list (no Vivado)

Per-module FIRST (`mismatch_count==0`, `max_error==0`, `timing_pass==true`),
then full e2e. For the 17 DW + stem, point the verifier at the now-live wrappers
with their existing sidecars:

```
# 17 depthwise (narrow C=32 .. widest C=960) + stem:
for n in 812 818 824 830 836 842 848 854 860 866 872 878 884 890 896 902 908 ; do
  npx tsx scripts/_verify_mbv2_variant.ts \
    output/mobilenet-v2/rtl/node_conv_${n}.v node_conv_${n} \
    output/mobilenet-v2/tb/node_conv_${n}.sidecar.json
done
npx tsx scripts/_verify_mbv2_variant.ts \
  output/mobilenet-v2/rtl/node_conv_810.v node_conv_810 \
  output/mobilenet-v2/tb/node_conv_810.sidecar.json
```
The per-module sidecars carry `pipeline_latency_cycles`; after Step 4 they reflect
the new datapath (812→196, 810→372, 896→2411 — `timing_pass` must hold). Widest
(C=960, `node_conv_896`) and narrowest (C=32, `node_conv_812`) + the stem are the
mandatory coverage; the rest are the same transform at intermediate C.

Then the full e2e (the gate that discharges any dataflow regression):
```
npx tsx scripts/run_mbv2_top_value.ts     # Verilator --x-initial 0, vs a FRESH golden, require mismatch==0
```

### Expected result

- **4.71× on the spatial path** (35.61M → 7.56M cyc): stem 11.44M→1.81M,
  17 DW 24.17M→5.75M.
- **fps @50 MHz (real clock):** A2-only **4.40** (engine still serialized),
  A1+A2 **6.61** (engine overlapped) — committed target **≈ 4.4–6.6 fps@50 MHz**,
  up from 1.27. (200 MHz hypothetical/timing-gated: 17.6–26.5.)
- **DSP:** +630 (648 spatial MAC − 18 today) → design total **≈1,975 = 16.1%** of
  12,288. **LUT ~62%, FF ~22%, RAMB36 37.7%, URAM 10.0% — all six resources
  < 80%.** BRAM is net-favorable-to-neutral (weight ROM wider×shallower, same bits).

### Self-check / dry-run evidence captured 2026-06-02

```
[self-check] PASS: transform(node_conv_812.v) == scratch PoC (code-identical, comment-insensitive).
[apply_mpk9] targets: 17 depthwise + 1 stem = 18 total ... dryrun_targets = 18
```
All 18 transforms additionally pass `iverilog -g2012 -t null` elaboration
(strict elaborator) on scratch copies — including the `$clog2(C)`-width wrapper
(node_conv_854), the folded-tap wrapper (node_conv_866), the widest C=960
(node_conv_896), and the stem→`conv_datapath_mp_k`. No live file was written.

**Staged deliverables (NOT applied):**
- `scripts/apply_mpk9_depthwise.py` — deterministic patch generator (--dry-run,
  --self-check, auto-backup). Self-checked byte-identical to the proven PoC.
- `scratch/golden_impl_mpk.diff` — the exact `golden_impl.py` edit (validated).
- This section — the one-command apply sequence + regen chain + verify list.
