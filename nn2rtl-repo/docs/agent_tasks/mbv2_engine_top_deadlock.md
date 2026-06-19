# MobileNetV2 engine-top e2e DEADLOCK — root cause + fix

Status: ROOT-CAUSED. Surgical generator fix. Not the documented final-stage
contract mismatch (that was a red herring for THIS deadlock — the chain dies
at the FIRST dispatch and never reaches the final stage).

## Stall signature (from probe, 50M-cycle run)

- Top under test: `output/mobilenet-v2/rtl/nn2rtl_top_engine.v` (REAL engine +
  scheduler, the <80%-fit deployment artifact).
- Symptom: `in=50176/50176` (full input frame consumed by ~cyc 12,000,000),
  `out=0/1`, `m_axis_tvalid` never asserts; sim hits the 50,000,000-cycle cap.
- Scheduler: parked in `S_WAIT_LOAD` (state=9) for the ENTIRE run.
  `next_state` stays 9->9. `dispatch_idx = 0` and NEVER advances
  (max_disp=0; only transition was reset 255->0 at cyc 17).
- Engine: idle the entire run. `u_shared_engine.state = 0` (IDLE),
  `engine_start` pulses = 0, `ag_mac_done` total = 0 across all 50M cycles.
  Engine is NOT stuck busy — it is never triggered.
- Waiting on: `current_loaded = all_loaded[dispatch_idx=0] = ldr0_loaded`,
  the dispatch-0 input loader `u_ldr_node_conv_814`. `ldr0_loaded` NEVER asserts.
- Loader internal: `word_count` climbs linearly with consumed input
  (53@0.5M, 326@2.5M, 939@7M, 1485@11M) and PLATEAUS at exactly **1568**
  once the full 50176-beat frame is consumed, then freezes (req=0, grant=0,
  beat_idx=0) forever. `1568 = 12544 / 8` exactly.

Probe: `output/mobilenet-v2/reports/verilator_mbv2_top_engine_value/engine_probe_deadlock.cpp`
Log:   `output/mobilenet-v2/reports/verilator_mbv2_top_engine_value/probe_deadlock_50M.log`
(links against the prebuilt `Vnn2rtl_top__ALL.a` + `verilated.o`; NO re-verilate.)

## Root cause — UNITS MISMATCH in input-loader sizing (LOADER class)

`stream_to_act_bram_bridge` (`nn2rtl_top_engine.v:3117`) counts **2048-bit
BRAM words** in `word_count`, and asserts `loaded` when
`next_word_count == TOTAL_BRAM_WORDS`. This is true in all three width
branches:
- `g_w_eq` (BUS_W==2048): word_count++ per beat — `nn2rtl_top_engine.v:3175`
- `g_w_lt` (BUS_W<2048):  word_count++ per packed word — `nn2rtl_top_engine.v:3223-3224`
  (`BEATS_PER_WORD = 2048/BUS_W` beats accumulated per word)
- `g_w_gt` (BUS_W>2048):  word_count++ per sliced word — `nn2rtl_top_engine.v:3279-3280`
  (`WORDS_PER_BEAT = BUS_W/2048` words emitted per beat)

But the generator populates `TOTAL_BRAM_WORDS` with a count of **predecessor
output BEATS**, not 2048-bit words:

  `scripts/build_top_wrapper.ts:1026`
    `const totalBramWords = d.input_hw[0] * d.input_hw[1] * icChunks;`
    (= number of `predBusOut`-bit beats the predecessor streams in one frame)

For dispatch 0 = `node_conv_814` (predecessor `n4_2`, BUS_W=256):
- BEATS_PER_WORD = 2048/256 = 8
- TOTAL_BRAM_WORDS is set to 12544 (= 112*112*ceil(32/256) = 112*112*1, a BEAT count)
- The stem streams 12544 beats -> only 12544/8 = **1568** BRAM words.
- word_count tops out at 1568, never reaches 12544, `loaded` never asserts.
- Scheduler waits forever in S_WAIT_LOAD on dispatch 0 -> total deadlock.

This is a count-of-256-bit-beats stored where the bridge expects a
count-of-2048-bit-words. Off by `2048/BUS_W`.

### Scope: ALL non-2048-bit loaders are mis-sized the same way
- BUS_W < 2048 (256/128/768/192/1152/1536/512/1280): OVER-sized by `2048/BUS_W`
  (e.g. ldr0 BUS_W=256 -> 8x; ldr1 node_conv_816 BUS_W=128 -> 16x).
  These deadlock (loaded never reaches the inflated target).
- BUS_W > 2048 (3072): UNDER-sized by `BUS_W/2048 = 1.5x`. These would assert
  `loaded` EARLY (after only 2/3 of the frame), feeding the engine a partial /
  wrong-shaped activation tile — a correctness bug, not a deadlock.
- BUS_W == 2048: correct by coincidence (1 beat == 1 word).

Dispatch 0 (ldr0, BUS_W=256) is processed first, so it is where the chain
visibly deadlocks. The all_loaded[21]=1 (node_conv_876 pre-resident) hardwire
and the final-stage flat-bus/tiled-streaming contract mismatch + SELRANGE
`data_out[1279:0]`-from-256-bit warnings are NEVER reached — the chain dies at
dispatch 0. They remain open downstream suspects but are NOT this deadlock.

## Fix — SURGICAL (generator one-liner + regenerate, OR on-disk patch)

The universally-correct sizing in 2048-bit words is:

  `totalBramWords = ceil(totalBeats * predBusOut / 2048)`

where `totalBeats = d.input_hw[0] * d.input_hw[1] * icChunks` (the current value)
and `predBusOut = predMeta.busOutBits` (already available as `b.predBusOut`).

`scripts/build_top_wrapper.ts:1026`, change:

    const totalBramWords = d.input_hw[0] * d.input_hw[1] * icChunks;

to:

    const totalBeats = d.input_hw[0] * d.input_hw[1] * icChunks;
    const totalBramWords = Math.ceil((totalBeats * predMeta.busOutBits) / 2048);

(`predMeta` is already in scope at that line; `b.predBusOut` mirrors it.)

NOTE — per project memory (`project_top_v_is_patched_not_regenerated`), the
deployed `nn2rtl_top_engine.v` is PATCHED, not regenerated; blind regen destroys
~1205 handshake patches and re-deadlocks. So the practical fix is a SURGICAL
edit of the on-disk `TOTAL_BRAM_WORDS(...)` constants for every non-2048-bit
loader: replace each `TOTAL_BRAM_WORDS(N)` with `N * BUS_W / 2048` (integer,
round up if not divisible). For ldr0: `TOTAL_BRAM_WORDS(12544)` -> `1568`.
Also fix the generator line so future regens are correct. A small
`apply_loader_word_resize.py` (parse each `stream_to_act_bram_bridge` instance's
BUS_W + TOTAL_BRAM_WORDS, rewrite the constant) is the clean way to patch all
~30 loaders deterministically.

This is a wiring/sizing constant fix, NOT the deep contract-level final-stage
regeneration the memory flags as a major phase.

## Verification path after fix
Re-run the engine-top value sim (~11 min to 50M cyc). Expect: ldr0_loaded
asserts after 1568 words; scheduler leaves S_WAIT_LOAD; engine_start pulses;
dispatch_idx advances 0->1->...; eventually reaches dispatch 21 (all_loaded[21]
hardwire) and the final stage — at which point the documented final-stage
contract mismatch / SELRANGE issue may surface as the NEXT blocker. Re-probe
then.

## Does it block the ALL-SPATIAL top?
NO. `stream_to_act_bram_bridge` and `TOTAL_BRAM_WORDS` exist ONLY because of the
engine's activation-BRAM staging (engine-top integration). The all-spatial top
streams layer-to-layer and has no input-loader bridges, so this units bug
cannot occur there. Engine-specific.

## OUTCOME (2026-06-02) — loader-word-resize patch VERIFIED; NEXT blocker found

Patch applied: `scripts/apply_loader_word_resize.py` rewrote 22 of 33
`TOTAL_BRAM_WORDS` literals in `output/mobilenet-v2/rtl/nn2rtl_top_engine.v`
(ldr0 12544->1568, etc.; diff = exactly 22 lines, zero collateral; backups in
`backups/loader_word_resize_20260602_0407*`). Engine-top exe rebuilt
(verilate OK 122 modules + make OK) in the PRIVATE
`obj_dir_engine_value`. Deadlock probe rebuilt against the fresh objects
(`probe_loaderfix.exe`) and run 50M cyc.

ORIGINAL DEADLOCK CLEARED (all three milestones hit):
- `LDR0_LOADED = YES @cyc 11,541,666` (word_count plateaus at the PATCHED 1568
  and now equals TOTAL_BRAM_WORDS -> `loaded` asserts). Was NEVER pre-patch.
- `engine_start pulses = 1 @cyc 11,541,667`; engine FSM ran (37636 state
  changes, max_estate 5), `ag_mac_done = 12544` (a full 112x112 output frame).
- Scheduler left S_WAIT_LOAD: S_WAIT_LOAD(9) -> S_PULSE_START(5) ->
  S_WAIT_DONE(6) -> S_WAIT_DRAIN(10).
So `dispatch0_loads=true`, `engine_starts=true`. The loader-word-resize fix is
correct and sufficient for its scope (byte-exact-irrelevant, as documented).

NEXT BLOCKER (NOT the loader patch, NOT the final-stage contract mismatch):
engine-output **FIFO OVERFLOW / missing engine backpressure**. The chain now
dies at dispatch 0's DRAIN, parked in S_WAIT_DRAIN forever:
- `current_drain_complete = all_drain[0] = u_engine_out_node_conv_814.drain_complete`
  NEVER asserts -> `max_disp = 0`, scheduler never reaches S_NEXT_DISP.
- Drain probe (`engine_probe_drain.cpp` -> `probe_drain.exe`,
  `probe_drain_16M.log`) shows the exact reason:
  - engine asserted `act_out_wr_en` 12544 times (`ewr=12544`) — it produced a
    full frame of output beats.
  - BUT `engine_output_fifo` (DEPTH=4096) only accepted **4098** beats
    (`wr_ptr=4098`); `in_ready = !fifo_full` and `wr_fire = in_valid &&
    !fifo_full` SILENTLY DROP every write while full. The engine FSM does NOT
    stall on `in_ready` — it writes at MAC pace regardless -> beats 4099..12544
    are LOST.
  - The bridge drained every tile that actually entered the FIFO:
    `tiles_emitted` plateaus at **65568 = 4098 beats * 16 tiles/beat**, then
    `eofifo_out_valid=0` / `buf_valid=0` forever (FIFO empty, fill=0).
  - But `EXPECTED_TILES = EXPECTED_BEATS(12544) * TILES_PER_BEAT(2048/128=16) =
    200704`. The bridge needs 200704 tiles to assert `drain_complete`; only
    65568 ever arrive -> drain_complete impossible -> permanent S_WAIT_DRAIN.
  - It is NOT a spatial_run/throttle gate: `spatial_throttle=0` (spatial_run=1)
    for the entire 3.9M-cycle drain window; the dispatch-1 input loader
    (`u_ldr_node_conv_816`) DID accept the drained tiles (`ldr1_wc` climbed to
    its patched 784) — the drain machinery works, it is starved of data.

Root mechanism (RTL): `engine_output_fifo` (nn2rtl_top_engine.v:3340) is a
4096-deep store-and-drop FIFO; the `shared_engine` write side
(`act_out_wr_en`/`act_out_wr_data`, engine instance ~2190) ignores
`eofifo_in_ready` (the wrapper even names it `_unused_eofifo_in_ready`,
line 2221). For any dispatch whose output frame > 4096 beats (dispatch 0 is
12544), the FIFO overflows and the bridge can never reach EXPECTED_TILES.

This is a REAL architecture/backpressure bug, NOT a constant-resize patch:
fixing it needs ONE of (a) engine output write-side honoring `in_ready`
(stall MACs / hold the output beat when FIFO full — the proper fix; needs
shared_engine RTL change + may interact with the documented weight-read-latency
pipelining), (b) a FIFO deep enough for the largest output frame (12544 beats
* 2048b = 25.7 Mbit — infeasible on-chip), or (c) restructure so the bridge
drains a beat for every beat the engine writes (lock-step), which the current
spatial_run-gated 1-tile/cycle bridge cannot guarantee. Recommend (a).

Probes/logs (all in
output/mobilenet-v2/reports/verilator_mbv2_top_engine_value/):
- probe_loaderfix.exe / probe_loaderfix_50M.log  (load+dispatch trace)
- engine_probe_drain.cpp / probe_drain.exe / probe_drain_16M.log (drain trace)
- tb_run_loaderfix.log (TB result: deadlock, out=0/1, no m_axis output)
TB result: result=DEADLOCK/timeout, no m_axis output, byte_exact_mismatch n/a
(-1) — chain stalls at dispatch 0 drain, never produces logits.
