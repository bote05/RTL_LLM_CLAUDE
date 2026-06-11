# RESNET FINAL-SYNTH BUNDLE — ENG-PIPE + KPAR8 + WEIGHT-ADDR FANOUT FIX

**Date:** 2026-06-11 · **Base:** 2e639df (ResNet 5,664,715 cyc vec0+vec1
exact, routed 67.15 MHz @16ns; MBV2 1,184,731 cyc 8/8, routed 86.67 MHz)
**Goal:** everything relevant rides the LAST ResNet synth.

**Appliers (in order, each independently bisectable):**
1. `scripts/apply_resnet_engpipe.py` (.preengpiper backups)
2. `scripts/apply_resnet_kpar8.py` + `scripts/repack_resnet_kpar8_banks.py`
   (.prekp8r backups; gitignored `_kp8.mem` banks regenerated with proofs)
3. `scripts/apply_resnet_waddr_rep.py` (.prewrep backups — SHARED files)
4. `scripts/apply_resnet_fanout_hints.py` (.prefoh backups)

**Gate runners:** `scripts/run_resnet_bundle_lint.sh`,
`scripts/run_resnet_engine_iso_kpar8.sh`,
`scripts/run_nn2rtl_top_value.ts` (vec0 + vec1 RUNONLY),
`scripts/run_mbv2_e2e_parallel.sh`.
**Gate logs:** `output/reports_integrated/resnet_final_bundle/`.

---

## STEP 1 — ENG-PIPE enable (ResNet)  [COMMITTED b3ae73a]

`output/rtl/nn2rtl_top.v` shared_engine instantiation gains `.ENG_PIPE(1)`.
The machinery (skeleton `g_ep`: ST_GAP issue pipelining + per-pass capture
registers + event-driven retire, bubble 12/10 → 3) was built and proven on
MBV2 (`docs/agent_tasks/ENG_PIPE_ANALYSIS.md`, 21/21 ISO incl. throttled
backpressure). ResNet specifics:

* ENABLE_OUTPUT_BACKPRESSURE stays 0 → `eff_out_ready==1`: the bridge
  write drains in one cycle, the FIRE gate never sees a held beat.
* ResNet engine output NEVER touches the spatial stream directly — it goes
  through the 4096-deep `engine_output_fifo` + per-dispatch bridges. The
  MBV2 ADD-JOIN class (engine write cadence exposing an accept-vs-pop race
  at residual joins) has no ResNet analog on this path; max dispatch
  output = 784 beats << 4096 (FIFO cannot overflow).

**Measured:** vec0 AND vec1 PASS 0/100352, `e2e_cycles = 5,654,052`
(was 5,664,715 → **−10,663, −0.19%**). The expected −0.3-0.5M did NOT
materialize because post-OVERLAP the frame is max(spatial, engine) per
region and the KPAR4 engine was already non-binding nearly everywhere —
the bubble savings are shadowed by spatial work. KEPT anyway: it is
value-neutral, makes the engine strictly faster (more shadow margin for
any future spatial speedup), and costs nothing.

## STEP 2 — KPAR8 (ResNet)

Shared core already carries the complete `K_PAR==8` elaboration
(`g_p8`/`g_walk_kpar8`/`g_waddr_kpar8`/`g_ktap_kpar8`, shipped for MBV2 —
`docs/agent_tasks/KPAR8_ANALYSIS.md`) INCLUDING the ResNet pos-major
dense-KxK fast walk (`g_walk_kpar8` ic-wrap at `ic_cnt == IC-8`). The
applier touches ONLY the ResNet top:

* `ENGINE_K_PAR` 4 → 8 (`ENGINE_WBUS_W` auto-scales 3072 → 6144).
* Banks 384b×16768 (`_kp4`) → 768b×8384 (`_kp8`), ADDR_W 15 → 14;
  bits/bank IDENTICAL (6,438,912) — BRAM-neutral (67072 % 8 == 0, 0 pad).
* `weight_bank_rd_addr = engine_weight_rd_addr[13:0]` (GROUP addr, old>>3).

**ELIGIBILITY AUDIT (proof P0, asserted on every repack run):** all 17
dispatch bases {0, 2304, 4352, 6656, 8960, 9984, 12288, 14592, 16896,
18944, 28160, 32256, 36352, 45568, 49664, 53760, 62976} are %8==0 and all
IC ∈ {256,512,1024,2048} are %8==0 → **no relocation pad needed** (unlike
MBV2's FC base 13413 → 13416 [FC-PAD]). The 17 regions tile [0, 67072)
exactly; 9 dense-3x3 regions transposed pos-major (same permutation as the
KPAR4 lineage — only the packing width changed), 8 dense-1x1 identity.
An 8-aligned group never crosses a (kh,kw) position (IC%8==0 ⇒ k%8==ic%8)
⇒ one act word, one chunk, one in_bounds decision per group.

**Repack proofs:** P0 (tiling+eligibility), P1 (bijectivity + full line
re-expansion), P2 (4096-sample fast-vs-legacy walk equivalence at 3-bit
lane granularity), P3 (aligned tap-slice identity), P4 (1x1 identity) —
ALL PASS (`repack_resnet_kpar8_banks.py`, abort = no writes).

**ISO gate** (`run_resnet_engine_iso_kpar8.sh`, WLAT=2): conv_246 (3x3
IC=256, transposed walk), conv_250 (1x1 IC=512/OC=1024, chunk rotation),
conv_284 (3x3 IC=512 **stride-2**, 2-chunk), vec0+vec1, three builds:
LEGACY serial reference, KP8, KP8+ENG_PIPE (the deployed config) — gate =
output bytes IDENTICAL to LEGACY (raw-dump cmp; contract goldens are stale
2026-05-30 vintage, INFO only — see `run_resnet_engine_iso_kpar.sh`).

**Measured [COMMITTED 94e3c9e]:**
* Repack: P0-P4 ALL PASS, bits/bank 6,438,912 -> 6,438,912 (neutral).
* ISO: **12/12 IDENTICAL** — {kp8, kp8+engpipe} × {246, 250, 284} ×
  {vec0, vec1} all byte-equal to the LEGACY serial build. Walk cycles:
  246: 453,938 → 58,802 (kp8) → 57,048 (kp8+ep); 250: 409,642 → 58,410 →
  52,540; 284: 452,664 → 57,528 → 56,754 (~7.7-7.9x). The per-case
  contract-golden mismatch counts are the documented stale-golden artifact
  (identical counts in all three builds — the load-bearing fact).
* e2e: vec0 AND vec1 **PASS 0/100352, e2e_cycles = 5,299,588**
  (5,654,052 → **−354,464, −6.3%**; cumulative vs base **−365,127,
  −6.4%**). The remaining frame is spatial-chain-bound: KPAR8 halved the
  engine walk but only the regions where the engine was still exposed
  paid out (same shadowing as step 1).

## STEP 3 — WEIGHT-ADDR FANOUT FIX (the route-data fix, cycle-neutral)

### The forensics (first_light_postroute_timing_kp4mp32_c16.rpt, 40 paths)

| class | paths in top-20 SETUP | worst slack | route % | fix |
|---|---|---|---|---|
| `g_walk_kpar.weight_rd_addr_reg[*]` → `u_uram_weight_bank{4,5,6}` BRAM `CASDOMUXA`/`ADDRARDADDR` | 6 | **+0.102** | 98.9-99.3% | **[WADDR-REP]** structural per-bank register replication |
| `u_scheduler/FSM_onehot_state_reg[5]` → `u_node_conv_276/out_pix_reg[*]/CE` (BUFGCE in path) | 13 | +0.240 | 94.5% | **[FO-HINT]** max_fanout on the scheduler state reg |
| `u_node_conv_288/in_beat_idx_reg[0]` → `in_lo_reg[1945]/D` | 1 | +0.213 | 99.0% | **[FO-HINT]** max_fanout on the gather beat counters |
| 20 HOLD paths, all +0.010 (dp→pend_pix, lbw window shifts, relu→skid LUTRAM, requant lane→bridge, ldr acc→wr_data) | — | +0.010 hold | — | NONE: router-managed hold fixing; no RTL leverage (per-brief: skip) |

Notably `weight_rd_addr_reg[8]_replica_1` appears IN the worst paths —
Vivado's own late (post-placement) cloning came too late to help; the fix
must exist at synthesis time.

### [WADDR-REP] structural replication (apply_resnet_waddr_rep.py)

One 22b address register fans to 8 banks × ~200 cascaded RAMB36 address/
cascade-select pins scattered across the die (94.6% BRAM packing). Fix =
8 (* dont_touch *) register copies fed by the SAME D / same `run_active`
enable / same reset — cycle-IDENTICAL by construction, 1/8 fanout each,
each placeable next to its bank column:

* `output/rtl/engine/address_generator.v` (SHARED): parameter `WADDR_REP`
  (default 1) + output `weight_rd_addr_rep` (REP×22b). Each walk branch
  gains a replication generate; **WADDR_REP==1 elaborates a passthrough
  assign of the original register — ZERO new FFs**, so MBV2 and every iso
  harness stay bit- and FF-identical (MBV2 8/8 inertness gate run).
  The replica always-block mirrors the original's exact update protocol
  (reset 0; `kpar_fast ? weight_addr_next_fast : weight_addr_next` under
  `run_active`; hold otherwise).
* `output/rtl/shared_engine_skeleton.v` (SHARED): forwards the parameter,
  exports the replicas group-shifted per K_PAR exactly like the scalar
  (`>>3` at K_PAR=8). The stub address_generator (standalone-parse path)
  mirrors the port. The ORIGINAL register keeps only the 3-bit serial
  subword pipe load (`wsub_d1 <= ag_weight_rd_addr[2:0]`).
* `output/rtl/nn2rtl_top.v` (ResNet-own): `WADDR_REP=8`; bank b's
  `rd_addr = engine_weight_rd_addr_rep[b*22 +: 14]`; the shared
  `weight_bank_rd_addr` wire is retired.

Escalation documented (NOT taken — needs Vivado evidence first): per-bank-
HALF replicas (WADDR_REP=16) if a single bank's ~200-BRAM spread is still
route-binding, or `phys_opt_design -force_replication_on_nets` on the
replica nets.

### [FO-HINT] synthesis attributes (apply_resnet_fanout_hints.py)

* `nn2rtl_scheduler.v` (ResNet-own): `(* max_fanout = 16 *)` on the 4-bit
  FSM `state` register (decl split from the combinational `next_state`).
  The top's `spatial_run`/`spatial_throttle` already carry
  `(* max_fanout = 32 *)` [FMAX-FANOUT], so the driver LUT is replicated —
  but every replica's input converged on the single state register; this
  lets synthesis clone the state bits per region.
* 38 `node_conv_*.v` (ResNet-own): `(* max_fanout = 8 *)` on the
  tiled-streaming gather counter `in_beat_idx` (uniform class fix: the
  counter decodes into per-slice write selects of the physically spread
  multi-thousand-bit `in_lo` register; synthesis only replicates where
  fanout actually exceeds the bound).

Both sim-inert (attributes are comments to Verilator/iverilog) — gated by
the same byte- AND cycle-exact e2e anyway.

**Measured:**
* lint 4/4 PASS (0 errors, 0 warnings; leg/kp4/kp8/kp8+ep) — the
  WADDR_REP=8 replica generate additionally lint-verified standalone.
* ResNet e2e vec0 AND vec1: **PASS 0/100352 @ EXACTLY 5,299,588** —
  byte- AND cycle-exact as required (the replicas are write-only copies).
* MBV2 inertness: **8/8 PASS, mismatch 0, e2e_cycles == 1,184,731 EXACTLY
  on all 8 vectors** (WADDR_REP defaults to 1 → passthrough assign, zero
  new FFs; the only shared-text change MBV2 re-elaborates).
* Encoding note: 12 generated node_conv files carry cp1252 em-dashes —
  apply_resnet_fanout_hints.py reads/writes them latin-1 (byte-preserving).

## GATE MATRIX

| gate | step 1 (ENG-PIPE) | step 2 (KPAR8) | step 3 (WADDR-REP + FO-HINT) |
|---|---|---|---|
| lint (4 configs: leg/kp4/kp8/kp8+ep) | PASS 0/0 | PASS (engine configs unchanged; top covered by e2e verilation) | PASS 0/0 (re-run, shared files changed) |
| ISO A/B (WLAT=2) | n/a (MBV2 21/21 inherited) | **PASS 12/12 IDENTICAL** (kp8 + kp8+ep vs legacy) | covered by cycle-exact e2e |
| ResNet e2e vec0 | PASS 0/100352 @ **5,654,052** | PASS 0/100352 @ **5,299,588** | PASS 0/100352 @ **5,299,588 EXACT** |
| ResNet e2e vec1 | PASS 0/100352 @ **5,654,052** | PASS 0/100352 @ **5,299,588** | PASS 0/100352 @ **5,299,588 EXACT** |
| MBV2 8/8 @ 1,184,731 EXACT | not needed (no shared file) | not needed (no shared file) | **PASS 8/8, mismatch 0, 1,184,731 EXACT (all 8 vecs)** |

**FINAL FRAME: 5,664,715 → 5,299,588 (−365,127, −6.4%), byte-exact on
vec0+vec1**, plus the route-class fixes riding into the final synth.

## PROMOTION CHECKLIST

1. On the target tree (2e639df lineage), in order:
   `python scripts/apply_resnet_engpipe.py` →
   `python scripts/apply_resnet_kpar8.py` →
   `python scripts/repack_resnet_kpar8_banks.py` (gitignored `_kp8.mem`
   must be regenerated in the target checkout; proofs P0-P4 rerun) →
   `python scripts/apply_resnet_waddr_rep.py` →
   `python scripts/apply_resnet_fanout_hints.py`.
2. Re-gate: `bash scripts/run_resnet_bundle_lint.sh`; ResNet e2e vec0+vec1
   (expect PASS 0/100352 @ 5,299,588); MBV2 8/8 (expect
   PASS @ 1,184,731 EXACTLY — WADDR-REP touches shared files).
3. REGEN NOTE (per feedback_regen_must_rebuild_engine_maps): any future
   `generate_golden`/bank rebuild must re-run `dedup_engine_banks_k5.py`
   THEN `repack_resnet_kpar8_banks.py` (and MBV2's kp8 repack) — the
   `_kp8` banks are derived artifacts; P0's tiling assert catches a stale
   dispatch table immediately. The `_kp4` ResNet banks are now UNUSED.
4. Vivado notes for the final synth:
   * Bank ROMs are 768b×8384 inferred block-RAM (same
     `ram_style="block", cascade_height=8`); bit-count unchanged.
     8384 deep ⇒ shallower than the 16768 kp4 shape — expect the same or
     fewer cascade levels (CASDOMUXA depth shrinks: depth 8384 / (4096
     rows per RAMB36E2 in 9b-wide mode) — verify tile count in the synth
     report).
   * DSP: mac lanes 1024 → 2048 product DSPs (+1024). U250 has 12,288;
     routed baseline used 60-67% — budget fine, but watch the engine
     column congestion (16384b → wait, ResNet: 6144b weight bus, half of
     MBV2's 16384b — milder than the MBV2 KPAR8 wave).
   * Fmax risk (inherited from MBV2 KPAR8 analysis): the stage-2
     accumulate is now a 9-operand combinational sum per lane. If a synth
     wave flags it, the mitigation is TREE_STAGES=1 (registered 8:1 tree,
     d5→d6 capture) — needs its own byte-exact re-gate.
   * The (* dont_touch *) WADDR-REP replicas must survive synthesis: check
     8 copies of `g_wrep.g_r[*].waddr_rep_q` in the synthesized netlist.

## SIDE OBSERVATION (MBV2 maintenance debt, NOT this bundle's scope)

`scripts/repack_mbv2_kpar8_banks.py` P0 now FAILS on the 2e639df scheduler
("scheduler has 51 base rows != 47"): the DW-QUARTET/DW-EXT/812-PAIR waves
grew the MBV2 dispatch table past the script's expectations. The DEPLOYED
`_kp8.mem` banks are still the proven ones (this gate re-verified 8/8 @
1,184,731 with them), but the regen path is broken — the next MBV2 wave
that needs a bank rebuild must first update the repack script's dispatch
parsing (P0 will keep refusing to write until then, which is the safe
failure mode).
