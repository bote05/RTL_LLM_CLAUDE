# MobileNetV2 engine-top e2e — DEFINITIVE blocker roadmap

**Artifact:** `output/mobilenet-v2/rtl/nn2rtl_top_engine.v` (the REAL `shared_engine` +
scheduler integration — the **<80%-fit deployment top**, as opposed to the
all-spatial `nn2rtl_top.v` which is faster e2e but ~84% LUT, over the 80% bar).

**Goal of this doc:** the complete, ordered list of everything still standing
between today's state and a *passing byte-exact engine-top e2e*, each with
mechanism, the specific fix (file:line), a risk class, an effort estimate, and a
clear SAFE-to-auto-apply vs NEEDS-USER-DECISION verdict. Plus an honest
bottom-line and an all-spatial-as-interim assessment.

Date: 2026-06-02 (overnight autonomous run). NO Vivado. NO unsupervised
engine-pipeline change (safety rule).

---

## Status snapshot (what is already GREEN)

- **Engine datapath PROVEN byte-exact: 34/34 dispatches, mismatch=0, max|err|=0**
  at the deployment 2-cycle URAM weight latency (WLAT=2), through the REAL
  `shared_engine` + real weight/bias/scale mems. Harness:
  `tb/engine_iso_wrap_mbv2.v` + `tb/engine_iso_wrap_mbv2_tb.cpp`. The engine
  MAC / requant / address pipeline is **not** to be re-litigated.
- **Engine-top integration config-correct + elaborate-clean** (`verilator
  --lint-only --top-module nn2rtl_top` exit-0). BUG 1 fixed: `shared_engine`
  instance now overrides `WGT_W=8 / URAM_DATA_W=2048 / MAX_I*=112` (was silently
  inheriting ResNet INT4 `WGT_W=4 / URAM_DATA_W=1024`, truncating the 2048-bit
  weight bus). See deliverable §4c.
- **BLOCKER #1 FIXED + VERIFIED (loader sizing, surgical-wrapper):**
  `stream_to_act_bram_bridge.TOTAL_BRAM_WORDS` was populated with a count of
  *predecessor output beats* but the bridge counts *2048-bit BRAM words*
  (off by `2048/BUS_W`). `scripts/apply_loader_word_resize.py` rewrote 22 of 33
  literals on-disk (ldr0 12544→1568, etc.; backups
  `backups/loader_word_resize_20260602_0407*`). Generator one-liner also
  identified (`scripts/build_top_wrapper.ts:1026`). **Result:** original
  deadlock CLEARED — `LDR0_LOADED @cyc 11,541,666`, `engine_start` pulsed,
  `ag_mac_done=12544` (full 112×112 frame), scheduler advanced
  S_WAIT_LOAD→S_PULSE_START→S_WAIT_DONE→S_WAIT_DRAIN. Byte-exact-irrelevant
  (a capacity constant), as documented. Full diagnosis:
  `docs/agent_tasks/mbv2_engine_top_deadlock.md`.

The chain now reaches dispatch-0 **drain** and freezes there. The remaining
blockers, in the order they will be hit:

---

## BLOCKER #2 (FIXED + VERIFIED 2026-06-02) — engine-output FIFO overflow / no engine backpressure

**STATUS: FIXED + byte-exact-VERIFIED via engine-iso (Verilator, WLAT=2). KEEP.**
The engine-output backpressure primitive is applied and proven byte-safe:
- **RTL** (param-gated, DEFAULT-OFF = byte-identical legacy):
  - `output/rtl/engine/bram_to_stream_bridge.v` — new `input wire out_ready`;
    write half HOLDS the beat (`act_out_wr_en && !out_ready` → keep en=1, freeze
    data) instead of clobbering. With `out_ready==1` the hold branch is dead =
    original 1-cycle-pulse behavior.
  - `output/rtl/shared_engine_skeleton.v` — new `parameter
    ENABLE_OUTPUT_BACKPRESSURE=0` + `input out_ready`; `eff_out_ready =
    (ENABLE..!=0)?out_ready:1'b1`; sticky `req_done_pending`; FSM `ST_REQUANT`
    arc holds the next oc_pass when `!eff_out_ready` (last pass → `ST_DRAIN`,
    whose `!bridge_busy` wait covers the held write). `.out_ready(eff_out_ready)`
    threaded to the bridge.
  - `output/mobilenet-v2/rtl/nn2rtl_top_engine.v` — shared_engine instance now
    `.ENABLE_OUTPUT_BACKPRESSURE(1)` + `.out_ready(eofifo_in_ready)` (line ~2179,
    ~2199); `_unused_eofifo_in_ready` removed.
  - `docs/agent_tasks/00_engine_skeleton_spec_PORTS.md` — `out_ready` added to
    the LOCKED bram_to_stream_bridge port table.
- **Byte-exact verification (independent re-run 2026-06-02):**
  - NO-STALL regression (ENABLE..=0, out_ready undriven → PINMISSING → eff=1'b1),
    WLAT=2, all mismatch=0: `node_conv_814` (dispatch-0, 12544-beat overflow case;
    551938 cyc, identical to legacy), `node_conv_898` (IC=960 4-chunk straddle),
    `node_conv_912` (final pointwise, oc_passes=5 — exercises the intermediate-pass
    arc).
  - BACKPRESSURE STRESS (ENABLE..=1, `tb/engine_iso_wrap_mbv2_bp.*`, commit-on-
    accept act-mem + hold-violation assertion), node_conv_814, all mismatch=0:
    mode 0 (ready high), mode 1 (periodic 3/8), mode 2 (LFSR ~50%, 281961 low cyc),
    mode 3 (adversarial 64/64 bursts, 401407 low cyc, +250k real stall cyc). No
    `[BP-ASSERT]` hold violation in any run.
  - NEGATIVE CONTROL (BROKEN non-holding bridge `tb/bram_to_stream_bridge_BROKEN.v`,
    same mode-2 stress): MISMATCH 98366/200704, max|err|=128 — proves the harness
    detects silent drops, so the holding-bridge mismatch=0 is load-bearing.
- **Datapath untouched:** MAC/requant/`ag_mac_done_d5` arithmetic unchanged; the
  prior 34/34 WGT_W=8/URAM_DATA_W=2048/WLAT=2 proof still holds; ResNet + every
  legacy engine-iso harness leave `out_ready` unconnected (eff_out_ready=1) and
  are byte-identical.
- **Backup of pre-change RTL:** `backups/engine_bp2_20260602_113204/`.

**CAVEAT (downstream, NOT this fix):** this converts the old silent FIFO-overflow
DROP into a coherent STALL. The `engine_output_fifo` does not drain while
`engine_busy=1` (spatial_run gated by engine_busy), so for dispatch-0 (12544 beats
> 4096 FIFO) the engine will now STALL once the FIFO fills rather than lose beats.
Making the engine-top e2e actually COMPLETE through dispatch-0 still requires the
separate concurrent-drain fix or a larger FIFO (a follow-on / part of getting past
the no-drain window). This change is the correct, byte-safe backpressure PRIMITIVE.

### (Original analysis below, retained for reference)

**Risk class: `engine-pipeline-change`. SAFE TO AUTO-APPLY: NO — needs user decision.**

### Mechanism (proven by probe, not inferred)
A **zero-concurrency drain window**, NOT an engine-outpaces-bridge rate problem.
Two coupled facts:

1. **The engine output write side has NO backpressure input.**
   `shared_engine.act_out_wr_en` / `act_out_wr_data` are produced by
   `bram_to_stream_bridge` (`output/rtl/engine/bram_to_stream_bridge.v:83-91`):
   `act_out_wr_en` is a pure 1-cycle pulse = registered `requant_valid`, with no
   `out_ready`/`in_ready` input ("Out of scope: deep FIFO buffering"). The
   wrapper drives `eofifo_in_ready` nowhere
   (`nn2rtl_top_engine.v:2221` — `wire _unused_eofifo_in_ready = eofifo_in_ready;`).
   `engine_output_fifo` (`nn2rtl_top_engine.v:2208`, `DEPTH=4096`) is
   **store-and-drop**: `wr_fire = in_valid && !fifo_full`, `in_ready=!fifo_full`
   ignored → writes silently lost when full.

2. **Both the bridge drain AND its consumer are gated by `engine_busy`.**
   `spatial_throttle = engine_busy | sched_spatial_stall; spatial_run =
   ~spatial_throttle` (`nn2rtl_top_engine.v:443-444`). The dispatch-0 bridge
   `u_engine_out_node_conv_814.ready_out = spatial_run`, and its consumer ldr1
   `u_ldr_node_conv_816.in_valid = node_conv_814_valid_out & spatial_run`.
   `engine_busy=1` for the *entire* engine run ⇒ `spatial_run=0` ⇒ the bridge
   drains **0 beats while the engine writes**.

**Probe (`probe_drain_16M.log`):** during `S_WAIT_DONE` the engine wrote
`ewr=12544` beats but the FIFO capped at `wr=4098 / fill=4096` (beats
4099..12544 **dropped**) while the bridge drained nothing (`tiles=1/200704`,
`run=0`). After `engine_done`→`S_WAIT_DRAIN` (`run=1`) the bridge drained the
4098 survivors = `4098*16 = 65568` tiles then froze. `EXPECTED_TILES =
EXPECTED_BEATS(12544) * TILES_PER_BEAT(2048/128=16) = 200704` is unreachable ⇒
`drain_complete[0]` can never assert ⇒ scheduler stuck in `S_WAIT_DRAIN`
forever. **8446 of 12544 output beats are permanently lost.**

### Rate is NOT the problem (it would keep up if allowed)
Engine wrote 12544 beats over ~552k cyc = **~44 cyc/beat**. The bridge drained
4098 beats / 65568 tiles over ~159k cyc = **~39 cyc/beat** — and the consumer
ldr1 kept up fully (`ldr1_wc` reached its patched target 784, `spatial_run=1`
the whole drain window). So drain rate (~39) ≥ write rate (~44): **if the bridge
were allowed to drain concurrently, the 4096-deep FIFO would never overflow.**
The overflow is 100% caused by the bridge being *forbidden* to drain while
`engine_busy=1`.

### The CORRECT minimal fix (engine-pipeline; user-gated)
Add engine output backpressure: make the engine honor `eofifo_in_ready` so the
output beat (and the upstream requant-write advance) STALLS instead of writing
into a full FIFO. Concretely:
- Route `eofifo_in_ready` (currently `_unused`, `nn2rtl_top_engine.v:2221`) into
  `shared_engine` as an `out_ready`.
- Gate the `bram_to_stream_bridge` write half + hold the requant/MAC advance
  (`bridge_busy` hold-off, `bram_to_stream_bridge.v:99-101`) until `in_ready`
  is high — so no output beat is ever produced into a full FIFO.

This makes the engine write rate irrelevant (the FIFO never overflows; the
comparable-rate bridge drains it during `S_WAIT_DRAIN`, and even a small FIFO
suffices).

### Why this is `engine-pipeline-change` (NOT auto-applyable)
It touches the `shared_engine` / `bram_to_stream_bridge` **requant-write
pipeline** (the `requant_valid → act_out_wr_en` path + the FSM `bridge_busy`
hold-off). That is exactly the byte-exact-critical pipeline the safety rule
protects: `shared_engine` is SHARED with ResNet, proven byte-exact (mbv2
engine-iso 34/34 + ResNet engine 14/14), and has a documented history of
timing-critical silent-corruption pipeline bugs (the `ag_mac_done_d5` drain
depth). Per the safety rule, this is **DOCUMENT + RECOMMEND only** tonight; it
must be a SUPERVISED change, then re-verify engine-iso byte-exact (mbv2 34/34 +
ResNet) BEFORE re-running the engine-top e2e.

### The two wrapper-only alternatives are REJECTED (not byte-exact-irrelevant)
- **(A) Ungate the bridge/consumer from `engine_busy`** (drop the `engine_busy`
  term for the engine-output path): NOT safe. It creates a real activation-BRAM
  **write-write hazard** — there is ONE shared act-BRAM write port
  (`act_wr_addr_final`, `nn2rtl_top_engine.v:2069`). The engine writes dispatch-0
  output to `act_out_base_word=4096` (bank 1) while ldr1 (`u_ldr_node_conv_816`,
  `BRAM_BASE_ADDR=4096`) would write the drained dispatch-0 output back into the
  SAME bank-1/base-4096 region the engine is still producing into → corrupts
  results. `engine_busy` gating exists precisely to prevent this.
- **(B) Enlarge `engine_output_fifo` DEPTH to the worst-case dispatch frame**
  (dispatch-0 = 12544 beats × 2048b = **25.7 Mbit per FIFO**): infeasible
  on-chip, and would have to cover the largest of all 34 dispatches. Not a
  loader-style sizing tweak.

There is **no byte-exact-irrelevant TOP-wrapper constant/wiring change** that
fixes #2 (unlike blocker #1). The consumer (ldr1) is not mis-sized — it kept up.

**Effort:** MEDIUM RTL change (route one `in_ready`, gate one write-enable +
one FSM hold) **but HIGH risk** (shared byte-exact engine pipeline). Plus a full
engine-iso re-verification pass (mbv2 34/34 + ResNet) before trusting it.

---

## BLOCKER #3 (DOWNSTREAM, after #2) — final-stage tiled↔packed CONTRACT MISMATCH

**Risk class: `contract-regen`. SAFE TO AUTO-APPLY: NO — needs user decision (major phase).**

### Mechanism (REAL dataflow break — will corrupt and/or stall the final output)
Tiled↔packed contract mismatch across the last ~5 inverted-residual blocks
(96ch, 160ch) and the classifier head (1280ch). The engine emits
`node_conv_880/886/898/904/892` (96ch/160ch projection-expand convs) and the
head as `io_mode=channel_tiled` at 256 bits/beat (32 ch/beat, multi-beat per
pixel), but every CONSUMING module is `io_mode=packed_full` expecting the FULL
channel vector in ONE beat:
- `node_add_828/900` expect 768-bit operands (96ch),
- `node_add_1038/1110` expect 1280-bit (160ch),
- `node_mean` expects 10240-bit (1280ch) per spatial beat,
- `node_conv_908` (depthwise 960ch) expects 4096-bit.

`build_top_wrapper` bridged them with raw bit-slices (`[767:0]`/`[1279:0]`) or
width-mismatched port connections that zero-extend the upper bits, never
reconciling the beat geometry. Concretely the add latches one beat
(`input_buf<=data_in` in `ST_IDLE`) then internally iterates `ch_idx 0..95` over
`input_buf`, so only the first 32 channels are real and ch 32..95 read all-zero
→ corrupt sum; additionally the producer streams 3 (or 5) 256-bit tiles per
pixel while the packed add consumes 1 beat/pixel → beat-cadence desync that can
also stall via the `ready_in`/`valid_out` handshake.

**SELRANGE-flagged sites** (`nn2rtl_top_engine.v` current line numbers):
- `node_add_828` @893
- `node_add_900` @934
- `node_add_1038` @1006
- `node_add_1110` @1047
- skip_fifo `u_skip_node_add_1038` @1205 (`WIDTH=1280` from `node_conv_892[1279:0]` 256-bit)

**Two MORE silent breaks (NO SELRANGE — declared-wide port on narrow net, legal
implicit zero-extend):**
- `node_conv_908` `data_in[4095:0]` ← `n4_33` 256-bit AND `data_out[4095:0]` →
  `n4_34` 256-bit (256↔4096),
- `node_mean` `data_in[10239:0]` ← `n4_35` 256-bit (256→10240) feeding
  `node_linear[10239:0]`.

NOTE the EARLIER adds (198/336/408/546/618/690) are CLEAN because their
channel_tiled conv producers' tile width happens to equal the add per-operand
width (e.g. add_546 ow=512 from conv_850 ow=512). The break appears only where
`channel_count*8 > 256` (96ch=768, 160ch=1280, 960ch=4096, 1280ch=10240).

### SUB-ITEM (folds into #3) — engine-output-bridge EXPECTED_BEATS inconsistency
The engine-output bridge `EXPECTED_BEATS` constants are internally inconsistent
for the channel_tiled final-stage slots and will mis-drain even after the #2
FIFO fix. **VERIFIED in RTL:** `u_engine_out_node_conv_880` has
`EXPECTED_BEATS=196` (`nn2rtl_top_engine.v:2676`) while the same-shape
`u_engine_out_node_conv_882` has `EXPECTED_BEATS=588` (`:2696`). For a
`[96,14,14]` channel_tiled tensor the true count is 588 256-bit beats, so 880's
196 is wrong — a direct artifact of the tiled↔packed contract confusion. These
constants must be regenerated together with the final-stage contracts.
fix_class = `contract-regen` (low effort once contracts settle).

### Fix (contract-regen; user-gated, MAJOR PHASE)
Regenerate the final-stage layer contracts so producers/consumers AGREE on
tiling — EITHER make the projection/expand/depthwise/pool/gemm consumers accept
`channel_tiled`, OR make the engine emit `packed_full` for these slots — and
regenerate the bridge `EXPECTED_BEATS` consistently with the chosen geometry.
This is entangled with the bit-exact module contracts (the same final-stage
contract issue project memory flags for the all-spatial top as "needs
CONTRACT-LEVEL final-stage regeneration, NOT a patch"). HIGH effort, major
phase, NOT a surgical patch.

**Effort:** HIGH. Touches generator (`build_top_wrapper.ts`) + per-layer
contracts + the affected RTL modules + bridge constants, regenerated together,
then re-verify the affected modules byte-exact and re-run the engine-top e2e.

---

## NON-BLOCKERS (verified, cleared — do not chase)

- **`node_conv_876` / `all_loaded[21]=1'b1` (`nn2rtl_top_engine.v:2118`):**
  CORRECT pre-resident ping-pong, NOT a stale/garbage hazard. node_conv_876
  (dispatch 21) has no input loader; its input is the immediately-preceding
  dispatch 20 = node_conv_874 (384→96 project). Scheduler ROM: dispatch 20
  `act_out_base_word=4096`, dispatch 21 `act_in_base_word=4096` — producer writes
  bank 4096, consumer reads 4096; dispatches are strictly serialized 0..33 with
  `S_WAIT_DRAIN` between, and node_conv_874 also fans out to the skip FIFO + the
  spatial stream so the BRAM copy is not clobbered before dispatch 21 reads it.
  The hardwire just skips the (nonexistent) external-loader wait. Reads the
  correct activation, will not stall on load. fix_class = none. (Caveat: it does
  inherit blocker #2's FIFO overflow on its own output, but that is #2, not a
  876-specific defect.)
- **`all_loaded[34..63]` / `all_drain[34..63] = 1'b1`** (`:2131` etc.): benign
  vector padding past the 34 real dispatches (`dispatch_idx` only spans 0..33).
  fix_class = none.
- **`node_mean → node_linear` flat bus** is internally consistent
  (10240→10240→8000); its only defect is being starved by the 256-bit
  channel_tiled feed — that is part of blocker #3, not a separate bus bug.
- No additional mis-sized loaders found beyond the documented #1 set.

---

## ORDERED REMAINING-BLOCKER LIST

| # | Blocker | Risk class | Auto-apply? | Effort |
|---|---|---|---|---|
| ~~1~~ | ~~Loader `TOTAL_BRAM_WORDS` units~~ | surgical-wrapper | DONE (applied+verified) | low (done) |
| **2** | Engine-output FIFO overflow / no engine backpressure (GATING) | **engine-pipeline-change** | **NO — user decision** | medium RTL + HIGH risk + full engine-iso re-verify |
| **3** | Final-stage tiled↔packed contract mismatch (+ EXPECTED_BEATS sub-item) | **contract-regen** | **NO — user decision** | HIGH, major phase |

**Total remaining blockers: 2** (both need user decision; zero remaining that
are safe to auto-apply).

---

## Verification path (after BOTH user-gated fixes land)
1. Re-verify engine-iso byte-exact: mbv2 34/34 (`tb/engine_iso_wrap_mbv2*`) +
   ResNet engine — confirms the #2 backpressure change did not corrupt the
   shared engine pipeline.
2. Re-verify the regenerated final-stage modules byte-exact (Verilator
   mismatch=0) — confirms the #3 contract regen.
3. Re-run the engine-top e2e value sim (~11 min/50M cyc) in the private
   `obj_dir_engine_value`; expect dispatch_idx to advance 0..33, `node_mean`/
   `node_linear` to produce logits, and m_axis byte-exact vs golden.

Probes/logs (all in
`output/mobilenet-v2/reports/verilator_mbv2_top_engine_value/`):
`probe_loaderfix.exe`/`probe_loaderfix_50M.log` (load+dispatch trace),
`engine_probe_drain.cpp`/`probe_drain.exe`/`probe_drain_16M.log` (drain trace).

---

## BOTTOM LINE

**The engine-top is correctness-proven where it matters (datapath 34/34
byte-exact) and config-correct, but it is NOT close to a passing e2e tonight, and
the remaining work is NOT auto-applyable.** Of the two remaining blockers, one is
an `engine-pipeline-change` to the shared byte-exact engine (protected by the
safety rule) and the other is a `contract-regen` major phase. Both need a user
decision; neither is a loader-style capacity tweak.

Realistic effort to a passing engine-top e2e:
- **#2** = supervised medium RTL change (route `eofifo_in_ready` into the engine
  as `out_ready`, stall the requant-write advance + bridge write half on a full
  FIFO) + a full engine-iso re-verification (mbv2 34/34 + ResNet). High risk
  because it edits the exact pipeline with a documented silent-corruption
  history.
- **#3** = a HIGH-effort, major contract-regeneration phase (reconcile
  channel_tiled vs packed_full across the last ~5 blocks + head + the
  EXPECTED_BEATS bridge constants), entangled with the bit-exact module
  contracts.

So: **2 blockers, both engine-deep / contract-deep, both user-gated.** This is
days-of-supervised-work, not a tonight patch.

### All-spatial top as a lower-risk INTERIM deployment path — ASSESSMENT
**Mixed.** The all-spatial top (`nn2rtl_top.v`) is datapath-simpler for the
*engine-integration* blockers — it has no engine activation-BRAM staging, so
blocker #1 (loader units) and blocker #2 (engine-output FIFO/backpressure)
**cannot occur there at all** (engine-specific). It is also FASTER e2e (~20M cyc
vs the engine-top's serialized ~39.4M). So for the two engine-class blockers it
is strictly lower-risk.

**BUT it is NOT a free interim win, for two reasons:**
1. **It does NOT meet the <80% fit bar:** all-spatial LUT ~84% — over the 80%
   target the engine-top was built to satisfy. The engine-top exists *because*
   the all-spatial top is over-budget. As a "fits-under-80%" deployment, the
   engine-top is the only candidate.
2. **It shares blocker #3 in essence:** the same final-stage tiled↔packed
   contract mismatch is the documented all-spatial e2e blocker too (project
   memory: final-stage 768→256/256→4096/1280→256 width mismatches that
   `build_top_wrapper` never reconciled, "needs CONTRACT-LEVEL final-stage
   regeneration"). So `contract-regen` is unavoidable on EITHER top for a full
   e2e.

**Verdict:** if the immediate goal is a *working e2e demonstrator* and the 80%
fit bar can be relaxed for an interim, the all-spatial top is the lower-risk path
— it avoids the engine-pipeline change (#2) entirely and needs only the
contract-regen (#3) that is shared. If the goal is the *<80% fit deployment
artifact specifically*, there is no shortcut: the engine-top needs the supervised
#2 backpressure change AND the #3 contract-regen. The contract-regen (#3) is the
common critical-path work for either top; doing it once on the all-spatial top
de-risks it before applying to the engine-top, and is the recommended sequencing
if the user wants forward progress without the engine-pipeline gamble tonight.
