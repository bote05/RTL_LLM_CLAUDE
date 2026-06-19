# MobileNetV2 engine-top HIGH-OC fix — Path B execution spec (2026-06-03)

**Goal:** make dispatches 13–33 (OC>256: 384/576/960/1280) byte-exact so the engine-top
e2e produces byte-exact `node_linear`. Low-OC (0–12) + chain-advance 0→21 already done.

**Approach (RECOMMENDED): coherent on-disk patching, block-by-block.** The high-OC MODULES
are sound (depthwise emit flat at correct width; relus are tiled-correct; loaders are a generic
parameterized module). The mis-contracting is in WIRING + PARAMS + missing retile bridges — all
on-disk fixable. Do NOT regenerate (would destroy the 61-module backpressure + add-join +
drain/overlap + loader-flush patches). See [[project-mbv2-e2e-backpressure]].

## START STATE
- On-disk top = clean verified-working (`backups/lowoc_working_20260602/nn2rtl_top_engine.v.reconfirmed`):
  legacy `TILES_PER_BEAT=1` bridge, low-OC byte-correct, chain runs 0→21.
- 3-mode engine bridge infra (STEP 1+2, ready) = `backups/bridge_tpb_fix_20260602/nn2rtl_top_engine.v.step1_2_enginebridge`.
- Tooling: `scripts/apply_highoc_bridge_params.py` (sets OC/OUT_KIND/POSITIONS on 11 slots),
  `scripts/apply_mbv2_wave2_bridges_engine.py` (BRIDGES filtered to 13; NEEDS more rework below),
  `scripts/analyze_loader_partial_words.py`, plan `docs/agent_tasks/mbv2_highoc_contract_plan.json`.
- Prepared infra: `rtl_library/retile_bridge.v` (ping-pong flat<->tiled adapter, deadlock-free).

## THE 3-MODE ENGINE BRIDGE (already built, in step1_2 backup)
`engine_output_bridge` rewritten as `generate`: OUT_KIND 0=legacy(low-OC, byte-identical),
1=tiled-256 (8 tiles/full beat + ceil(LAST_OC/32) for the partial last beat of each position;
beat_in_pos tracks ceil(OC/256) beats/pos), 2=flat-gather (gather ceil(OC/256) beats -> 1 OC*8b
contiguous beat). Verilator names regs `...__DOT__g_legacy/g_tiled/g_flat__DOT__*`.

## KEY LEARNINGS (why naive per-stage patching fails)
1. **Downstream stages are tuned to the OLD LOSSY upstream counts.** Fixing SLOT13 (OUT_KIND=2,
   correct 196 beats) without fixing ldr14 (sized 392 for the lossy count) -> dispatch-14
   S_WAIT_LOAD wedge. => fix ALL stages of a block TOGETHER.
2. **384 block is all-flat** (n4_15/depthwise854/n4_16 all 3072b, fits <4096b cap). Needs only
   engine OUT_KIND=2 + loader fix. BUT ldr14 BUS_W=3072 is a NON-MULTIPLE of 2048 -> the loader
   g_w_gt branch (WORDS_PER_BEAT=3072/2048=1) DROPS 1024b/beat. The loader MODULE needs a
   non-multiple-BUS_W fix (like the partial-word flush), OR a retile to a clean width.
3. **576/960/1280 blocks are tiled-256** (relus n4_* 256b/N-beats) with FLAT depthwise -> need the
   retile gather (tiled->flat into depthwise) + scatter (flat->tiled into relu). These are the
   wave-2 bridges. ldr22 etc. are BUS_W=256 (clean) -> just size fixes.

## WAVE-2 SCRIPT ENGINE-TOP ADAPTATION (the `_engine.py` copy needs ALL of these)
The script is baseline-coupled via 5 structures + 2 fns. For the engine top:
- `BRIDGES`: keep 13 spatial = gathers br_878/884/890/896/902/908 + scatters
  br_n4_24/26/28/30/32/34 + br_mean. DROP 5 pointwise scatters (br_876/882/888/900c/906) + 4 add
  gathers (br_828m/900m/1038m/1110m) + br_1038s. (filter done in `_engine.py`.)
- `REPOINT`: keep ONLY the 12 = (node_conv_878/884/890/896/902/908 <- br_878..) +
  (n4_24/26/28/30/32/34 <- br_n4_24..). DROP the 5 pointwise (node_conv_876.. engine-dispatched)
  + the 5 passthroughs (n4_23/25/27/31/33 — already fed by the engine bridge in the engine top).
- `ADD_MAIN = []` (engine-top adds are low-OC, already add-join-gated).
- `SKIP_FIFO_1038`: skip main() step 5 (make patch_skip_fifo a no-op).
- `UNGATE_FINAL_BRIDGELESS`: DROP node_conv_880/886/892/894/898/904/910/912 (engine-dispatched,
  no spatial instance to strip). Start with [] (or just terminal n4_35/node_linear); ITERATE —
  this is the drain-side lost-beat fix, the documented hard gating part.
- `patch_spatial_throttle`: engine top has `wire spatial_throttle = sched_spatial_stall;` (OVERLAP
  fix dropped engine_busy). Override the regex to `spatial_throttle = sched_spatial_stall | any_retile_stall`.

## EXECUTION ORDER (each step = re-verilate(~14min, build-only `npx tsx scripts/run_mbv2_top_engine_value.ts 0`) + probe(scratch/probe_build/probe_overlap.exe, watch dispatch advance))
0. Restore step1_2 infra; revert 384 slots (852/858/864/870) to OUT_KIND=0 (keep advancing-lossy);
   probe -> confirm chain still 0→21 (no-regression). [DONE once; reproducible.]
1. Finish the `_engine.py` adaptations above; dry-run on a COPY until it APPLIES clean (no
   "instance not found"). Then apply to the top. Probe.
2. STEP 4 loaders: resize ldr22/24/26/28/30/32 to positions*ceil(IC/256) words (BUS_W stays 256
   tiled; or feed from a loader-gather per the plan). ldr33 redundant (engine writes bank direct).
3. Probe -> chain should pass 22→...; iterate UNGATE/gating on any new wedge.
4. Then the 384 blocks: OUT_KIND=2 + fix ldr14/16/18/20 (the BUS_W=3072 non-multiple loader-module fix).
5. Full e2e (`NN2RTL_ENGINE_VALUE_RUN=1`) -> byte-exact node_linear. Iterate any value mismatch.

## PROBE NOTE
After the engine-bridge rewrite, bridge regs moved into generate blocks -> the probe's
`u_engine_out_node_conv_*__DOT__tiles_emitted` refs need `g_legacy/g_tiled/g_flat__DOT__` prefixes
(already fixed in scratch/engine_probe_downstream_stall_overlap.cpp). Kill stale probe_overlap.exe
(taskkill //F //IM) before re-linking or ld fails on the locked file.
