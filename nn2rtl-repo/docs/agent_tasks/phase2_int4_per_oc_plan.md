# Phase 2 execution plan — INT4 nibble-pack + per-output-channel requant

Prereq DONE (Phase 1, branch `int4-imagenet-gptq`, commits b1dd85b + 232410c):
goldens regenerated at INT4-GPTQ per-OC (ref ~78% top-1). `output/layer_ir.json`
now has, per conv2d, `weight_bits:4`, `scale_factor_per_oc:[OC]`,
`weight_scale_per_oc:[OC]`. Weight hex values ∈ [-8,7]. RTL is still INT8/per-tensor
→ does NOT match the new goldens until this phase lands. Backups: `backups/phase2_*`
(RTL), `backups/phase1_*` (pre-regen INT8 goldens).

The two changes are ORTHOGONAL — can be done/verified independently:

## Change B — INT4 nibble-packing (weight width 8→4)
- `scripts/repack_weights_wide.py`: add `--bit-width 4`; pack 2 INT4/byte (hi/lo
  nibble) so the on-chip weight bus halves but addressing is unchanged.
- `scripts/build_weight_memory_map.py`: `WGT_W=8→4`, `BANK_USEFUL_BITS 256→128`.
- `output/rtl/shared_engine_skeleton.v`: `WGT_W 8→4`, `URAM_DATA_W 2048→1024`.
- `output/rtl/engine/mac_array.v`: `weight_bus [2047:0]→[1023:0]`; extract 4-bit
  nibble per lane + sign-extend to the multiplier (`{{4{n[3]}},n}`).
- spatial datapaths `rtl_library/conv_datapath_parallel.v` / `conv_datapath_mp_k.v`:
  weight ROM width + per-lane nibble slicing.
- Regenerate weight hex/banks. Engine sweep + per-module byte-exact.

## Change A — per-output-channel requant (the bigger one)
Today: ONE (scale_mult,scale_shift) per layer/dispatch, shared across 256 lanes.
- ENGINE: scheduler `scale_*_rom` (1/dispatch) → AXI 0x24/0x28 → config_register_block
  → `requant_pipeline.v` (shared `scale_mult`/`scale_shift` inputs, applied to all
  lanes at lines ~241/224). Need 256 per-OC scales per dispatch.
  - Recommended (Option 1): add a scale ROM read path mirroring the bias path —
    new engine ports (scale_mult/shift rom addr/en/data), scheduler stores a per-
    dispatch ROM BASE (new AXI regs 0x3C/0x40 in config_register_block),
    `address_generator.v` pulses the read per OC-pass, `requant_pipeline.v` uses a
    per-lane `scale_mult_lane`/`scale_shift_lane` instead of the shared ones.
  - The 256 per-OC (mult,shift) come from `compute_scale_approx(scale_factor_per_oc[oc])`.
- SPATIAL `node_conv_*.v`: today `localparam SCALE_MULT/SCALE_SHIFT` (per-tensor),
  passed to `conv_datapath_*`. For per-OC: per-OC scale ROM ($readmemh) in the
  datapath, indexed by output-channel; emit the per-OC scale hex from layer_ir.

## Open design decisions (confirm before coding)
1. **Per-OC scale delivery** = Option 1 (engine scale ROM + scheduler base addr).
   Adds a small BRAM/URAM ROM; revisit in Phase 3 fit. OK?
2. **RTL persistence**: per-layer `node_conv_*.v` go via `write_verilog` MCP
   (writes `output/rtl/<id>.v`). But engine INFRA (`requant_pipeline.v`,
   `mac_array.v`, `shared_engine_skeleton.v`, `nn2rtl_scheduler.v`,
   `config_register_block.v`, `address_generator.v`) are NOT per-layer modules —
   confirm how to edit those (the debug workflow used direct edits, handoff-sanctioned).
3. **Verification gate**: engine in-chain ±1 is still CONFOUNDED (isolation-pass vs
   in-chain-fail). De-confound (clean Verilator engine-isolation run) before trusting
   e2e; the INT4 design inherits whatever it is.

## Order
1. Change B (INT4 pack) — engine sweep byte-exact at INT4 (per-tensor scale still,
   on the per-tensor INT8 goldens? NO — goldens are per-OC now). So Change A and B
   must both land before any byte-exact check against the new goldens.
2. Do A+B together on the ENGINE first → engine sweep 14/14 byte-exact vs new goldens.
3. Then spatial convs (A+B) → per-module byte-exact.
4. e2e value-match (reframed gate: spatial Verilator byte-exact + engine per-module).

---

## STATUS 2026-05-28 (branch int4-imagenet-gptq) — full detail in memory project-int4-fit-analysis

KEY SIMPLIFICATIONS discovered: (i) **nibble-packing is byte-transparent** (engine MAC does
`$signed(weight[7:0])*act`, so INT4-in-int8-byte == nibble) → DEFERRED to Phase 3/fit; Phase 2's
only correctness change is **per-OC requant**. (ii) **scale ROM base_words == bias base_words** →
engine reads scale at the SAME address as bias (no scheduler/addr_gen change).

DONE + VERIFIED:
- Engine weight-latency BUG (1-cyc pipeline vs 2-cyc deployment URAM) root-caused + fixed
  (shared_engine_skeleton.v WEIGHT_RD_LATENCY=2). Commit 8677bc0.
- Engine per-OC requant (requant_pipeline.v per-lane scale_in[8191:0]; skeleton scale read @ bias
  addr; build_scale_memory_map.py → scale.mem). **BYTE-EXACT conv_246(1pass)+conv_250(4pass) on
  Verilator-iso AND iverilog.** Commit 4d7daa3 + TB scale_mem.
- VERIFICATION UNBLOCKED: full INT4-regen recipe (see memory) — patch_layerir_to_tiled (widths→256
  ABI), build_weight/bias/scale_memory_map, build_spatial_scale_mems, apply_spatial_scale_path,
  rebuild_contract_goldens (orchestrate.ts materializeContractGoldens exported). All 119 INT4
  contract goldens regenerated. Commits 2cc5f8f, 7b49c5b.
- Spatial datapath per-OC (conv_datapath_mp_k.v runtime scale ROM + per-tensor fallback, lint-clean,
  commit 3c59d10) + 39 spatial scale.mem + .SCALE_PATH wired (absolute paths).

RESOLVED (the conv_198 equiv_one "max75" was a standalone-TB output-tiling artifact, NOT an RTL
bug — DBG dump proved conv_datapath_mp_k per-OC byte-exact: oc0=-21, oc32=0 match golden. In-chain
probe is the real verification path).

E2E IN-CHAIN PROGRESS 2026-05-28 (full detail in memory project-e2e-value-verification):
- build_top_wrapper.ts engine scale_mem: DONE but BEWARE — regenerating the top from the generator
  WIPES the post-gen handshake/FIFO patches (113 vs 1205 markers) → e2e DEADLOCKS. FIX: restored the
  patched on-disk top + added scale wiring SURGICALLY (see memory project-top-v-is-patched-not-
  regenerated). Dataflow now COMPLETES: 13,352,707 cyc (= known-good), 3136/3136 beats.
- conv_datapath_parallel per-OC: added for lib consistency but NOT on active path (all 59 spatial
  convs incl. conv_196 instantiate conv_datapath_mp_k).
- e2e VALUE root-cause: per-tensor engine → 11793 mismatch; per-OC engine → 8464 (mixed ±, mean|err|
  35). LOCALIZED to **stale residual-ADD fusion constants**: all 16 node_add_*.v had LHS/RHS_FUSED_
  MULT from a PRIOR calibration (INT4-GPTQ+ImageNet regen updated convs+goldens, NOT adds). Golden
  Int8Add = round_half_up((lhs*ls+rhs*rs)/os). scripts/apply_add_rescale.py recomputes from current
  layer_ir, exhaustively validates all 65536 int8 pairs vs golden-float byte-exact, patches all 16.
  NOTE: the probe-vs-contract-goldout localizer is ABI-broken for intermediate taps (all show ~95%
  mismatch even when fine; only relu_48 final-output tiling coincides) — use analyze_value_mismatch.py
  on the relu_48 capture for the trustworthy signal.

REMAINING: rebuild+re-run e2e with fixed adds → expect byte-exact (or localize engine in-chain
residual next). Then commit Phase 2. Then Phase 3 (nibble-pack for fit + biases→LUTRAM), Phase 4
(50k accuracy), Phase 5 (cycle opt), Phase 6 (Vivado).
