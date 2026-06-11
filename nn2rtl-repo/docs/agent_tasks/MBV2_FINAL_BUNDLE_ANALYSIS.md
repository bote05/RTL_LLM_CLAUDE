# MBV2 FINAL-SYNTH BUNDLE — route-forensics fanout fixes + pblock retirement

**Date:** 2026-06-11 · **Base:** 2e639df (MBV2 = 1,184,731 cyc, 8/8 byte-exact;
new netlist LUT 18.9% / BRAM 67.3%; routed 86.67 MHz @8ns, WNS −3.538)
**Applier:** `scripts/apply_mbv2_final_bundle.py` (idempotent, anchor-asserted,
`.prefinalbundle` backups) · **Scope:** `output/mobilenet-v2/**` only — zero
shared-engine-file changes.

## 1. Route forensics (what the report actually says)

Source: `output/mobilenet-v2/reports/synth/checkpoints/mbv2_route_postroute_timing_new_c8b.rpt`
(280,978 failing endpoints @8ns, worst −4.019 ns; hold/PW clean). Every top
path is ~89–98% ROUTE delay — the netlist's logic is fast (0.19–1.35 ns); the
wall is wire distance driven by high-fanout select/enable/broadcast nets. Top-10
path classes:

| # | Slack | Source → Destination | Killer segment | Class |
|---|-------|----------------------|----------------|-------|
| 1,2,5 | −4.019/−4.017/−3.975 | `u_engine_out_node_conv_876/g_tiled.tile_idx_reg[1]` → `g_tiled.data_out_reg[82/91/100]` | `tile_idx[1]` **fo=518**, 11.5 ns route (98.4% of path) | A: bridge tile-mux select fanout |
| 3 | −4.010 | `u_scheduler/FSM_onehot_state_reg[8]_replica` → `u_engine_out_fifo/mem_reg_uram_18/EN_A` | `engine_output_ready_repN` **fo=271** (1.57 ns) → loader FSM fo=64 (2.51 ns) → conv_836 `g_legacy.data_out[1535]_i_1` CE **fo=1573** (0.68 ns) → `mem_reg_uram_0_i_1` EN **fo=53** (2.11 ns); 10 logic levels of cross-module ready resolution | B: scheduler-ready broadcast / CE / URAM-EN chain |
| 4,6 | −4.003/−3.963 | `u_engine_out_fifo/mem_reg_uram_5(0)/CLK` → `conv_850(832) g_legacy.beat_buf_reg` | **0 logic levels**, fifo `out_data` per-bit **fo=57/63**, 10.8/10.6 ns pure route | C: FIFO out_data absorbed into URAM OREG → 51 scattered bridges |
| 7..~34 | −3.955 | `u_shared_engine/ag_act_in_ic_byte_idx_d2_reg[1]` → `u_mac_array/g_p8.g_mac[112].acc0` DSP | byte-idx mux chain → `mac_act_bytes_ext[55]` **fo=2816**, 6.82 ns | D: engine act-byte broadcast — **OUT OF SCOPE** (shared file) |

## 2. Fixes shipped (all cycle-neutral)

All in `output/mobilenet-v2/rtl/nn2rtl_top_engine.v` + `nn2rtl_scheduler.v`,
applied by `scripts/apply_mbv2_final_bundle.py`.

### 2.1 CLASS A — g_tiled tile-mux select replication (STRUCTURAL)

`engine_output_bridge` OUT_KIND=1 (`g_tiled`): the monolithic
`data_out <= beat_buf[tile_idx*256 +: 256]` (256b 8:1 mux, select fo=518) is
split into 32b slices, each loaded through its own
`(* dont_touch *) reg [6:0] tile_idx_rep` shadow register (`g_tidx_rep[ts]`).
Each replica has the **identical** reset + update logic as the master
(pull-clear textually after emit-increment, same priority), so
`tile_idx_rep == tile_idx` on every cycle by induction → **byte- and
cycle-exact by construction**. Fanout per replica ≤ ~72 (32 mux bits × 2 LUT
levels + incrementer), and each replica can be placed beside its 32 slice FFs.
`emit_ready` (now the per-slice CE) additionally capped `max_fanout=128`.

Benefits all **13** OUT_KIND=1 instances (conv_876/878/882/884/888/890/894/
896/900/902/906/908/912 — every DW-on-engine + the final 1280-OC pointwise).
Cost: 8×7 = 56 extra FFs per bridge (728 total) — noise at FF 37.8%.

NOT done: a pipeline stage on the mux. Replication provably covers the
distance class (the select net is the only >1 ns segment), and a latency
change would re-gate the 1,184,731-cycle count for 13 bridges.

### 2.2 CLASS B — broadcast/CE/EN `max_fanout` caps (ATTRIBUTE-ONLY, c232a20 precedent)

| Net (RTL signal) | Report fanout | Cap | Where |
|---|---|---|---|
| `nn2rtl_scheduler.engine_output_ready` (51-bridge `start` broadcast) | 271 (post Vivado FSM-replica) | 64 | scheduler port decl |
| `engine_output_bridge` `emit_ready` (data_out CE; g_legacy/g_flat/g_tiled) | up to 1573 (conv_836, DATA_W=1536); g_flat reaches DATA_W=8000 (node_linear) | 128 | all 3 generate branches |
| `engine_output_fifo.load_skid` (URAM read/skid EN) | 53 | 16 | fifo body |

Attributes are invisible to Verilator/iverilog (sim-identical) and only let
synthesis replicate the driving LUT so copies place near their load clusters.

### 2.3 CLASS C — FIFO `out_data` OREG escape (ATTRIBUTE-ONLY)

`engine_output_fifo.out_data` was absorbed into the URAM output register
(report source = `mem_reg_uram_*/CLK`, 0 logic levels), so a single URAM-column
pin drove all 51 bridge `beat_buf` D-inputs across the die.
`(* max_fanout = 16 *)` on the port asks synthesis to keep/replicate the skid
register in fabric (~4 copies/bit, ≈6K FFs), each placeable near a bridge
cluster, and replaces the URAM CLK→Q launch with an FDRE launch. Same RTL
register either way → cycle count unchanged. If synthesis declines the
attribute it is a no-op (today's behavior, no risk). No register stage was
added — that would be +1 latency on every engine drain (NOT cycle-neutral).

### 2.4 CLASS D — shared-engine act broadcast: DOCUMENTED, NOT TOUCHED

~28 of the top-30 paths are
`u_shared_engine/ag_act_in_ic_byte_idx_d2_reg → mac_array DSP B-input`
(−3.955 ns): the act-byte select mux output `mac_act_bytes_ext[*]` broadcasts
with **fo=2816** (6.82 ns) to all 128 MAC DSPs. The fix is the same class-A
replication (per-MAC-group shadow of `ag_act_in_ic_byte_idx_d2` and/or
`max_fanout` on `mac_act_bytes_ext`) but lives in
`output/rtl/engine/address_generator.v` + `mac_array.v` /
`output/rtl/shared_engine_skeleton.v` — **sibling-owned; stopped per scope
rule.** After classes A–C this is the residual WNS owner: expect post-bundle
WNS to converge toward ~−3.9 ns unless the sibling ships the engine-side fix.

### 2.5 Stale pblock RETIRED

`output/mobilenet-v2/reports/synth/mbv2_fmax_pblock.xdc` → comment-only
tombstone. The old SLR floorplan constrained `u_node_conv_854/860/866/872/
878/884/890/896/902/908`, `u_br_ldr28/30/32` — **none exist** in the new
netlist (DW-ENGINE-EXT/DW-QUARTET/FC-ENGINE moved them onto the engine; only
`u_node_conv_810/812` remain spatial) — and loading it crashed `place_design`
(EXCEPTION_ACCESS_VIOLATION). The c8b route closes with `--no-pblock`, so per
the do-not-gamble rule it is retired, with a written recipe for a minimal
1-pblock replacement (engine + out-fifo + act loaders in the URAM SLR) should
the class-C distance ever need it — to be rebuilt from LIVE netlist cell names
and place-verified first.

## 3. Gates

| Gate | Result |
|------|--------|
| (a) Verilator lint, harness flag set (`--lint-only -Wno-fatal` + e2e waivers, full engine-top file list) | **0 errors**; 8 warnings, identical pre-existing classes/counts to the pair812 baseline (1 DEFOVERRIDE + 7 TIMESCALEMOD on untouched `rtl_library/` files) — `output/mobilenet-v2/reports/final_bundle/lint_final_bundle.log` |
| (b) MBV2 e2e 8/8 (`scripts/run_mbv2_e2e_parallel.sh`) | **PASS — 8/8 byte-exact, TOTAL mismatch = 0** — `output/mobilenet-v2/reports/final_bundle/e2e_final_bundle.log` |
| (c) Cycle-neutrality | **`e2e_cycles=1184731` EXACTLY on all 8 vectors** (== 2e639df baseline). The per-vector logs are byte-IDENTICAL to the committed baseline `reports/e2e_par/vec*.log` (git shows them unmodified after the run) — every progress line matches cycle-for-cycle. `final_bundle/e2e_result.txt` |
| (d) Applier idempotency | re-run = "already applied — no-op" ✔; partial-apply detection in place |

### 3.1 e2e result

```
[par-e2e] TOTAL mismatch (8 vecs) = 0
[par-e2e] RESULT: PASS (8/8 byte-exact)
vec0..vec7: result=PASS mismatch_bytes=0 beats_seen=32/1 in_beats=50176/50176
vec0..vec7: e2e_cycles=1184731   (gate: must equal 1,184,731 -- MET, all 8)
```
Fresh-build proven: `Vnn2rtl_top.exe` rebuilt 2026-06-11 09:32:17, vec logs
written 09:32:48 (this run), against the tracked goldens.

## 4. Promotion notes (for the LAST MBV2 synth)

- Synth flow (`scripts/run_mbv2_synth.ts`) picks up both files automatically;
  `.prefinalbundle` / `*.bak*` backups are excluded by `isExcludedRtl`.
- Run route **without** `mbv2_fmax_pblock.xdc` (it is comment-only now, so
  passing it is also harmless).
- Post-route, check the same report path: the conv_876 tile-mux class must be
  gone from the top paths; the expected new WNS owner is the shared-engine
  class-D path (~−3.9 ns @8ns) unless the sibling's engine bundle lands too.
- `dont_touch` on `g_tidx_rep[*].tile_idx_rep` keeps Vivado from re-merging
  the replicas; do not add `-flatten_hierarchy none`-dependent assumptions.
- If a later wave retunes clocks: at WNS −3.5 the honest routed ceiling was
  86.67 MHz; every ns recovered on the route-dominated paths is ~6–8 MHz.
