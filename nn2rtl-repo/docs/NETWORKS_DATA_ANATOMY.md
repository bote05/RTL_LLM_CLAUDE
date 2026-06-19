# Networks Data Anatomy — ResNet-50 & MobileNetV2 on Alveo U250 (as of 2026-06-12)

> **STATUS SNAPSHOT (verified on disk 2026-06-12 ~12:50 local; repo `D:/RTL_LLM_CLAUDE/nn2rtl-repo`, branch `int4-imagenet-gptq`, HEAD `6fad9d7`, netlists sealed `50c3054`)**
> - **ResNet-50 (5,299,588 cyc/frame, vec0+vec1 byte-exact 0/100352):** final-netlist SYNTH banked (`first_light_synth_final.dcp`, 1.29 GB, Jun 11 15:46; LUT 70.0% / BRAM36 98.3% / DSP 65.2%). Latest ROUTE attempt `_final_c14` **FAILED — congestion-infeasible, adversarially verified NOT kill-caused** (3 independent skeptics, "kill caused it" REFUTED): the router's congestion surrender (`WARNING [Route 35-447]`) fired ~03:19, **2 h 43 m before** the 12 h kill (06:02:23); the kill only terminated the `cmd.exe` wrapper, vivado.exe routed on unharmed for **85.6 min** and failed organically: `ERROR [Route 35-2]` 22,199 node overlaps. **No routed numbers exist for this netlist.**
> - ResNet last **measured-routed** = PREVIOUS netlist `kp4mp32_c16` (5,664,715 cyc): on-disk signoff says **timing MET at 12.000 ns, WNS +0.102, hold +0.010** → 83.33 MHz guaranteed / 84.05 MHz implied = **14.7 fps** (`checkpoints/first_light_postroute_timing_kp4mp32_c16.rpt`). The project-memory "67.15 MHz @16 ns" is contradicted by disk (clock-flag bug, see §2.4).
> - **MobileNetV2 (1,184,731 cyc/frame, 8/8 byte-exact):** final-netlist SYNTH banked (LUT 19.1% / BRAM 67.4% / DSP 27.2%). Route `final_c8` Jun 12: WNS −2.199 @8 ns → 98.05 MHz = 82.8 fps, hold MET. **[UPDATE 2026-06-16] A NEWER route `physopt_aggr_c7` (Jun 15 01:56) at 7 ns supersedes it: setup WNS −2.017 → Fmax 110.90 MHz = 93.61 fps, hold WHS +0.004 MET, same sealed netlist (`checkpoints/mbv2_route_postroute_timing_physopt_aggr_c7.rpt`). 110.90 MHz is the citable MBV2 routed Fmax; this snapshot's 98.05 below is the prior rung.** (`new_c8b` Jun 11 = 83.20 MHz; the "86.67 MHz" was a parser artifact.)
> - **Vivado in-flight note (updated 13:05):** the `final_c8` Vivado (pid 7284) **has exited** (~12:50); `mbv2_route_final_c8.json` (success=true, elapsed 24,236.8 s) + all post-route rpts + `mbv2_route_routed_final_c8.dcp` (362 MB) are on disk. **A NEW Vivado (pid 6004) launched 12:52: ResNet route-only RETRY** — `route_design -directive AggressiveExplore` from `first_light_physopt_final_c14.dcp` (same 12 ns placement), will write `checkpoints/first_light_routed.dcp` + `first_light_postroute_{util,timing,power}.rpt`; live log `C:/Users/User/AppData/Local/Temp/nn2rtl-routeonly-duJCDT/vivado.log`.

Conventions used throughout: **[ROUTED]** = measured post-route signoff on disk · **[SYNTH-EST]** = post-synth utilization/estimate · **[PLACE-EST]** = pre-route placement/phys_opt estimate · **[PENDING]** = not yet produced · **[MEM]** = project-memory-export only (no surviving disk artifact). Formulas, shown once and used everywhere: **Fmax = 1000 / (period_ns − WNS_ns)** · **fps = Fmax[MHz] × 10⁶ / cycles_per_frame** · **% chip = used / U250 total × 100**.

---

## 1. Device budget (the denominators)

AMD Alveo U250, part `xcu250-figd2104-2L-e` (UltraScale+ VU13P, **4 SLRs**). Every percentage in this document divides by these totals:

| Resource | U250 total |
|---|---:|
| CLB LUTs | 1,728,000 |
| CLB Registers (FF) | 3,456,000 |
| DSP48E2 | 12,288 |
| BRAM36 tiles | 2,688 (= 99.09 Mbit) |
| BRAM18 equivalents | 5,376 |
| URAM288 | 1,280 |
| LUTRAM-capable LUTs | 791,040 |
| SLL budget per SLR boundary | 23,040 |

---

## 2. ResNet-50 (INT4/INT3 Config-B, 5,299,588-cycle netlist)

Quantization: GPTQ Config-B, **18 conv layers INT3 + 35 conv layers INT4**, top-1 **77.60%** (`docs/agent_tasks/autonomous_night_log.md:763,708`; the precision mix is NOT visible in synth artifacts — 0 hits for "int3/int4" in the 7.47 MB synth log; provenance = Config-B sweep `output/reports_integrated/configb_acc_bram_sweep.json`, Jun 4 + project memory). Engine layers run INT3 (`ENGINE_WGT_W=3`, `output/rtl/nn2rtl_top.v:447`). Byte-exact gate: `output/reports_integrated/resnet_final_bundle/e2e_waddr_rep_vec{0,1}.log` — `e2e_cycles=5299588`, `total mismatching bytes = 0 / 100352`, `result=PASS`.

### 2.1 Latest artifacts & paths

All under `D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/reports_integrated/` unless absolute.

| Artifact | Path | Size | Timestamp | Status |
|---|---|---:|---|---|
| Final-netlist synth report | `first_light_synth.json` (+`first_light_synth.log`) | 7,623,895 B | Jun 11 15:47 | [SYNTH-EST], success=true |
| Final-netlist synth DCP | `checkpoints/first_light_synth_final.dcp` | 1,287,489,262 B | Jun 11 15:46 | banked, resume-ready |
| Failed-route report | `resume_from_synth.json` / `.log` | 88,111 / 85,316 B | Jun 12 06:03/06:02 | success=false (12 h kill; capture truncated) |
| Full failure log (Vivado session) | `failed_route_final_c14/vivado_full.log` (**preserved** Jun 12 13:00 from `Temp/nn2rtl-resume-YIHf1a/vivado.log`) | 93,686 B | Jun 12 07:27 | continues 1.4 h past kill; durable copy in repo |
| c14 opt DCP | `checkpoints/first_light_opt_final_c14.dcp` | 266,669,569 B | Jun 11 18:24 | resume point (12 ns baked clock) |
| c14 placed DCP | `checkpoints/first_light_placed_final_c14.dcp` | 793,848,285 B | Jun 12 00:07 | resume point (12 ns-driven placement) |
| c14 physopt DCP | `checkpoints/first_light_physopt_final_c14.dcp` | 793,857,011 B | Jun 12 00:26 | resume point |
| c14 routed DCP | — | — | — | **DOES NOT EXIST** (route failed before write) |
| Last good routed DCP (prev. netlist) | `checkpoints/first_light_routed_kp4mp32_c16.dcp` | 1,020,962,421 B | Jun 11 02:32 | [ROUTED], superseded netlist |
| kp4mp32 post-route rpts | `checkpoints/first_light_postroute_{timing,util,power}_kp4mp32_c16.rpt` | — | Jun 11 02:34–02:42 | [ROUTED] signoff, superseded netlist |
| Launcher summary | `docs/agent_tasks/13_integration_first_light_REPORT.md` | — | Jun 11 15:47 | clock 12 ns, elapsed 6,939.1 s |

Synth run identity (`first_light_synth.json`): Vivado v2025.2 (6299465), `synth_design -top nn2rtl_top -part xcu250-figd2104-2L-e -flatten_hierarchy rebuilt -verilog_define NN2RTL_SYNTHESIS=1`, 16 threads, synth_design wall 1:45:56, peak RAM 20.9 GB, total elapsed 6,939.089 s (1h 55m 39s), 0 critical warnings / 0 errors. **Clock constraint baked at synth: 12 ns** (launcher report; this matters — see §2.4). No timing was captured at synth (json wns=null, fmax=0, timing rpt section empty).

### 2.2 Post-synth utilization (final netlist) — [SYNTH-EST]

Source: `first_light_util.rpt` table embedded in `first_light_synth.json` (Design State: Synthesized, Jun 11 15:38).

| Resource | Used | % U250 |
|---|---:|---:|
| CLB LUTs | **1,209,699** | **70.01%** |
| — LUT as Logic | 1,182,027 | 68.40% |
| — LUT as Memory (distributed RAM 27,660 + SRL 12) | 27,672 | 3.50% of 791,040 |
| CLB Registers (all FF, 0 latches) | **1,215,675** | **35.18%** |
| CARRY8 | 61,590 | 28.51% |
| F7 / F8 muxes | 89,243 / 29,534 | 10.33% / 6.84% |
| Block RAM tiles | **2,656** | **98.81%** ← binding resource |
| — RAMB36E2 / RAMB18E2 | 2,642 / 28 | 98.29% / 0.52% |
| URAM288 | **662** | **51.72%** |
| DSP48E2 | **8,007** | **65.16%** |
| Bonded IOB | 328 | 48.52% |
| Unique control sets | 19,263 | — |

Register mix: FDRE 1,126,096 (92.6%) / FDCE 63,113 / FDSE 26,334 / FDPE 132 — the sync-vs-async ratio is the K1 FDCE→FDRE recode signature. LUT primitives: LUT6 553,399 / LUT2 292,086 / LUT3 234,947 / LUT5 223,089 / LUT4 100,903 / LUT1 39,096.

Cell-count breakdown (Report Instance Areas in the synth log — **cells, not LUTs**; total 2,876,941 cells, 258 top instances): spatial conv datapaths 1,929,897 (67.1%, 36 inst) · residual adds 589,485 (20.5%, 16) · `u_shared_engine` 110,117 (3.8%) · loader bridges 99,539 (17) · relus 67,030 (49) · engine-out bridges 53,489 (17). Largest singles: `u_node_conv_288` 192,536 · `u_shared_engine` 110,117 · `u_node_conv_268` 101,347. No hierarchical LUT util exists for this netlist era (`hier_util.rpt` = May 29/30, superseded old netlist).

### 2.3 Memory anatomy — what the BRAM/URAM/LUTRAM is made of

**BRAM (2,642 RAMB36, 98.3% — the chip-filler).** No "Block RAM: Final Mapping" section exists in the synth log; composition from the ROM Preliminary Mapping Report (pre-optimization) in `first_light_synth.json`:

| Consumer | Geometry | Notes |
|---|---|---|
| 8× engine weight banks `u_uram_weight_bank0..7` | inferred ROM 16384×768 each (used depth 8,384 lines × 768 b = "67072/8 wide lines", `nn2rtl_top.v:3155-3158`) | KPAR8 tap-major banks; module named "uram" but mapped to **Block RAM** under `NN2RTL_SYNTHESIS`; init `output/weights/uram_weights_bank{0-7}_kp8.mem` |
| 1× `conv_datapath_mp_k` `weight_word_q_reg` | 8192×768 | spatial full-window weight word |
| Top-level weight-word ROMs | 512×512, 1024×1024 | |
| Everything else (act BRAMs, spatial weight packings that didn't fit LUT) | — | the other 1,211 inferred ROMs (mostly spatial `mp_k` weight packings) went to **LUT**, not BRAM |

Exact per-module BRAM split would need `report_utilization -hierarchical` on `first_light_synth_final.dcp` (not run).

**URAM (662, 51.7%).** Ultra RAM Final Mapping Report itemizes 524 (remaining 138 = Vivado's "multiple instantiated RAMs reported once" dedup): `u_act_mem/mem_reg` activation memory **174** · line-buffer slots (`lbw/gen_slot[*].gen_mem_ultra`, 227×24 b) **145** · `u_engine_out_fifo` **29** · ~176 itemized across skid/skip FIFOs (`u_skip_node_add_*`, `u_skid_node_relu_*`, GCB blocks). Note: per `project_uram_no_init` [MEM], URAM cannot be bitstream-initialized on U250 — all URAM here is runtime-written; all init-bearing weight ROMs live in BRAM/LUT.

**LUTRAM (27,672).** Dominated by relu `beat_buf_reg` (RAM32M16 ×1,748 + RAM64M8 ×222) and skid LUT-FIFOs (RAM32M16 ×1,197 + RAM64M8 ×370); primitives total RAM32M16 2,790 / RAM64M8 588 / RAM32M 155 / RAM64M 4 / SRL16E 12.

**Weight bit budget (computed from the hex/mem files referenced by the deployed top):**

| Store | Bits | % of chip BRAM (99.09 Mbit) |
|---|---:|---:|
| Spatial conv ROMs (36 convs: 11× mp32_k8 = 16.78 Mbit; mp_k_7/8/9 = 6.26 Mbit) | 23.04 Mbit | 23.3% |
| Engine banks (17 dispatched convs, INT3/INT4 nibble-packed, 8 × 768 b × 8,384 lines) | 51.51 Mbit | 52.0% |
| **Total weights** | **74.55 Mbit** | **75.2%** |

(All-INT4 reference point would be 93.8 Mbit ≈ 94.6% [MEM: int4_fit_analysis] — Config-B's 18 INT3 layers are what make it fit with margin.)

### 2.4 Route history & the latest FAILED attempt

**⚠ Cross-cutting clock-flag bug (affects every ResNet "rung"):** `scripts/run_resume_from_synth.ts:91` re-applies the resume clock via `set_property -quiet PERIOD <ns> [get_clocks clk]`, which **silently does nothing** — proven by the XDC embedded in the DCPs (`nn2rtl_top_late.xdc :: create_clock -period 12.000 -name clk`, present in `first_light_synth_final.dcp`, `first_light_physopt_final_c14.dcp` AND `first_light_routed_kp4mp32_c16.dcp`) and by the c16 in-session Clock Summary (`clk 12.000 ns`). **Both the "c16" and "c14" routes actually ran at the synth-baked 12 ns**; the flags only changed labels and the JSON's Fmax arithmetic. Fix: unconditional `create_clock` in the script.

**FAILED attempt `_final_c14` (the latest ResNet event; final 5,299,588-cyc netlist).** Sources: `resume_from_synth.{json,log}` + the surviving full Vivado log `C:/Users/User/AppData/Local/Temp/nn2rtl-resume-YIHf1a/vivado.log`.

Timeline (session start Jun 11 18:02:23, PID 23276, 16 threads, `place/route -directive Explore`):

| Stage | Stage elapsed | Done (wall) | Peak RAM |
|---|---|---|---|
| open_checkpoint (synth_final.dcp) | 0:06:00 | ~18:08 | 12.9 GB |
| opt_design | 0:14:05 | ~18:23 → dcp 18:24 | 35.3 GB |
| place_design Explore | **5:37:13** | Jun 12 00:03 → dcp 00:07 | 44.0 GB |
| phys_opt_design | 0:12:55 | dcp 00:26 | 44.0 GB |
| route_design Explore | **FAILED after 6:51:16** | error ~07:17, Vivado exit 07:27:59 | 45.8 GB |

- Placer warned `[Place 46-14] design is highly congested and may have difficulty routing`. Post-place/phys_opt estimate: **WNS +0.423, TNS 0.000** [PLACE-EST] — the JSON's `setup_wns_ns=0.423 / hold −0.189 / fmax 73.65 / timing_met=true` is THIS estimate, with Fmax computed as 1000/(14−0.423) against the **never-applied** 14 ns flag; against the real 12 ns constraint the same estimate implies 86.4 MHz. **It is not a routed result.**
- Router forensics: pre-route "≥686 CLBs have high pin utilization"; SLL assignment **SLR0↔SLR1 columns oversubscribed at 114% and 104%** (demand 9,504/23,040 = 41.25% total); initial estimated congestion **Global/Short level 6 (64×64), Timing level 7 (128×128)**, boxes clustered in the mid-die SLR1 URAM/DSP column band (e.g. LONG NORTH INT_X65Y284→INT_X128Y443 anchored at URAM_URAM_FT_X64Y270).
- Phase 5.1 Global Iter 0: overlaps 1,883,230 → **26,135** (intermediate route WNS −1.122, TNS −116.4). The decisive event came at **~03:19** — `[Route 35-447] Congestion is preventing the router…prioritize completion over timing` is Vivado's **congestion surrender** (gives up timing, switches to completion-mode), **2 h 43 m before the kill**; congestion 6/7 warnings had fired at ~00:33. Phase 5.2 Global Iter 1 opened at 1,415,548 overlaps — the **normal start-of-iteration rip-up count, not thrash** — and ground down monotonically to 78,890 at the 43,200 s orchestrator kill (06:02:23), then continued improving post-kill (59,877 → 49,991), finalized at 22,199, pushed Phases 6–8, then **Phase 9 verification FAILED canonically: `[Route 35-162]` 24,830 signals failed to route; `ERROR [Route 35-2]` 22,199 node overlaps** → `route_design failed`, clean DRC, license release, byte-clean `Exiting Vivado` at 07:27:59.
- **Verdict: NOT a timeout death — adversarially verified** (3 independent skeptic passes tasked to prove "the kill caused the failure"; all returned REFUTED, high confidence). Kill mechanism: Node `execFile`'s built-in timeout `TerminateProcess`'es only the direct child (`cmd.exe` running vivado.bat) — no tree-kill in that code path (`taskkill /T` exists only in the RAM-watchdog branch, which did not fire) — so **vivado.exe was orphaned, unharmed, and routed on for 85.6 minutes** to its organic failure. The catastrophe predates the kill; the route is **congestion-infeasible at the 12 ns-driven placement** and a longer timeout would have changed nothing. What the kill DID corrupt is the **JSON run record**: the severed stdout pipe froze the capture mid-Phase-5.2, leaving the stale place-est `wns +0.423 / timing_met=true` fields that originally misread as "timed out while passing". (Two suspicious leads cleared: the orchestrator's post-kill recursive delete backed off on locked files — zero deletions, whole dir leaked intact; the garbled negative CPU times right after the kill are a >24 h formatter wrap, all values unwrap by exactly +48:00:00 into a monotonic series — cosmetic.) Top contended nodes: **8 of 10 are `u_uram_weight_bank{1,2,6,7}/weight_bus[*]`** vs shared-engine loader nets (`act_in_rd_data_d[*]`, requant-pipeline lane regs, `u_ldr_node_conv_246/278`, `u_node_conv_208` window nets) — the engine weight-bank broadcast buses crossing the SLR1 URAM/DSP band are the epicenter.

**Last SUCCESSFUL route `kp4mp32_c16` (PREVIOUS netlist, 5,664,715 cyc) — superseded netlist, but the only measured-routed ResNet.** Its JSON/log were overwritten by the c14 run; the in-session signoff reports + DCP survive in `checkpoints/`:

| Metric | On-disk value [ROUTED] | Memory claim — verdict |
|---|---|---|
| Timing | **clock 12.000 ns, setup WNS +0.102, TNS 0.000, "All user specified timing constraints are met"** (3,150,220 endpoints) — `first_light_postroute_timing_kp4mp32_c16.rpt`, Design State: Physopt postRoute | "67.15 MHz MET @16 ns WNS +1.109" — **CONTRADICTED BY DISK**; 67.15 = 1000/(16−1.109) computed against the never-applied 16 ns flag ([MEM]-only artifact, propagated to `THESIS_FINN_HLS4ML_COMPARISON.md:12` — needs correction) |
| Hold | **WHS +0.010, THS 0.000 — hold MET** (in-flow post-route phys_opt fixed it) | "hold −0.178" — superseded by disk (pre-physopt intermediate) |
| Fmax / fps | **83.33 MHz guaranteed (12 ns met) / 84.05 MHz implied** → 14.71 / 14.84 fps @ 5,664,715 cyc | memory's 11.9 fps understates the routed result |
| Utilization | LUT 1,196,343 (69.23%) / regs 1,189,379 / BRAM tiles 2,656 (98.81%, RAMB36 2,642) / URAM 662 (51.72%) / DSP 6,983 (56.83%) — `…postroute_util_kp4mp32_c16.rpt` | LUT/BRAM/DSP figures disk-verified ✓ |
| DCP | 1,020,962,421 B, Jun 11 02:32, embedded `create_clock -period 12.000` | size/date verified ✓ |

**All ResNet route attempts:**

| Tag | Netlist / cycles | Clock flag → actual | Result | WNS (status) | Fmax |
|---|---|---|---|---|---|
| chanwindow2 (Jun 4 era) | old, 13,548,787-class [MEM] | 40 ns | ROUTED, 0 overlaps — **superseded history** | +11.7 @40 ns | ~35.35 MHz routed [MEM]; dcp `first_light_routed_chanwindow2.dcp` survives |
| `kp4mp32_c16` | previous, 5,664,715 | 16 ns flag → **12 ns actual** | **ROUTED, MET** (superseded netlist) | **+0.102 setup / +0.010 hold [ROUTED]** | **83.33 guaranteed / 84.05 implied MHz** |
| `_final_c14` | **final, 5,299,588** | 14 ns flag → **12 ns actual** | **FAILED** — 22,199 node overlaps, congestion-infeasible | +0.423 [PLACE-EST only]; route intermediate −1.122 | none — **no routed Fmax exists for the final netlist** |
| **route-only RETRY (in flight)** | **final, 5,299,588** | 12 ns (inherited from physopt dcp) | **RUNNING since Jun 12 12:52** (pid 6004) — `route_design -directive AggressiveExplore` from `first_light_physopt_final_c14.dcp`; same placement, stronger router directive | [PENDING] | [PENDING] → `checkpoints/first_light_routed.dcp` + `first_light_postroute_timing.rpt` |

### 2.5 Throughput

Cycles/frame = **5,299,588** (sealed `50c3054`; gate `resnet_final_bundle/e2e_waddr_rep_vec{0,1}.log`, both PASS 0/100352; commit `8c2166e` message confirms cycle-exact).

| Clock scenario | fps = f×10⁶/5,299,588 | Status |
|---|---:|---|
| 86.36 MHz (placement est. WNS +0.423 vs real 12 ns) | 16.30 | [PLACE-EST — route then FAILED; do not cite as achievable] |
| 83.33 MHz (12 ns — the clock the previous netlist closed) | 15.72 | [TARGET] |
| 73.65 MHz (the JSON's 14 ns-assumed arithmetic) | 13.90 | [PLACE-EST, mislabeled clock] |
| 71.43 MHz (true 14 ns) | 13.48 | [TARGET] |
| 62.50 MHz (16 ns fallback) | 11.79 | [TARGET] |
| — previous netlist for reference: 83.33 MHz × 5,664,715 | **14.71** | **[ROUTED]** (best measured ResNet to date) |
| — campaign start: 35.35 MHz × 13.54 M | 2.61 | [MEM], superseded history |

**Cycle composition:** spatial-streaming-bound; the shared engine is **fully shadowed by spatial work** post-OVERLAP (commit `b3ae73a`: ENG-PIPE=1 saved only −10,663 cycles because "engine fully shadowed by spatial post-OVERLAP"; `docs/agent_tasks/RESNET_FINAL_BUNDLE_ANALYSIS.md:42`). Engine duty was 93.3% at the OVERLAP commit (9.62 M-cyc era) [MEM]. Parked levers: FRAME-PIPE, stem rework, conv_288→engine [MEM].

### 2.6 Interface anatomy

Top: `output/rtl/nn2rtl_top.v` (module `nn2rtl_top`, line 15). Single clock domain. Citations are `file:line`.

**Clock / reset**

| Signal | Width | Dir | Meaning |
|---|---|---|---|
| `clk` | 1 | in | single clock for the entire design (`nn2rtl_top.v:16`) |
| `rst_n` | 1 | in | active-low synchronous-deassert reset (`:17`) |

**`s_axis` — AXI4-Stream slave, input image (50,176 beats/frame)**

| Signal | Width | Dir | Meaning |
|---|---|---|---|
| `s_axis_tdata` | **256** | in | one input pixel per beat; **RGB packed in [23:0]**, upper 232 bits ignored (`PIXEL_IN_data = s_axis_tdata` `:52` → `node_conv_196.data_in[23:0]`, truncation at `:575`) |
| `s_axis_tvalid` / `s_axis_tready` | 1/1 | in/out | stream handshake (`:20-21`) |
| `s_axis_tlast` | 1 | in | asserted on beat 50,175 — **50,176 beats = 224×224 pixels** (TB `tb/nn2rtl_top_value_tb.cpp:51`) |

**`m_axis` — AXI4-Stream master, output feature map (3,136 beats/frame)**

| Signal | Width | Dir | Meaning |
|---|---|---|---|
| `m_axis_tdata` | **256** | out | final ReLU feature map (`= node_relu_48_data_out` `:4601`) = 7×7×2048 INT8 tensor; **32 INT8 channels/beat, 64 beats/pixel × 49 pixels = 3,136 beats** (TB map `tb:354-357`). GAP/FC are NOT on-chip — the chip emits the pre-pool feature map |
| `m_axis_tvalid` / `m_axis_tready` | 1/1 | out/in | handshake (`:25-26`) |
| `m_axis_tlast` | 1 | out | beat 3,135 (0-indexed) (`:4614`) |

**`s_axil` — AXI4-Lite control slave**

| Signal group | Width | Meaning |
|---|---|---|
| `s_axil_awaddr/araddr` | 32 | forwarded to the **shared_engine config register block** (`:31` comment; wired to scheduler-driven `sched_axil_*` `:526`); register map = `docs/agent_tasks/10_engine_config_register_block.md` |
| `s_axil_wdata/rdata` | 32 | config data |
| `s_axil_wstrb` | 4 | byte strobes |
| `s_axil_bresp/rresp` | 2 | responses |
| valid/ready pairs (aw/w/b/ar/r) | 1 each | standard AXI4-Lite handshakes |

### 2.7 Netlist statistics

- **Layer modules: 119** = 36 spatial convs + 17 engine-dispatched convs + 49 ReLUs + 16 residual adds + 1 maxpool (instantiation grep over `nn2rtl_top.v`; 53 `node_conv_*.v` files on disk, the 17 engine ones not instantiated — "data_out driven by shared_engine", e.g. `:1803`). ⚠ File header `:7` still says "spatial: 105, engine: 14" — **stale pre-K5 header** (top is patched, never regenerated); the scheduler is authoritative.
- **Engine dispatches: 17** (`output/rtl/nn2rtl_scheduler.v:3`, `LAST_DISPATCH=5'd16`; `nn2rtl_scheduler_schedule.json` num_dispatches=17): **9× 3×3** (conv_246/254/260/266/272/278/284/292/298 — incl. the K5 monster trio 284/292/298) + **8× 1×1** (conv_250/264/282/286/290/294/296/300).
- **Engine config:** `K_PAR=8` (`nn2rtl_top.v:456,:3741`) · `ENG_PIPE=1` (`:3746`) · `WADDR_REP=8` per-bank read-addr replicas (`:463,:3161`) · `ENGINE_WGT_W=3` (INT3 engine weights, `:447`; lane group 32×3 b = 96 b).
- **Weight banks:** 8 banks, **768-bit lines (8 taps × 96 b tap-major) × depth 8,384**, weight bus 6,144 b (`:3155-3236,:457`).
- **Spatial parallelism (36 convs):** stem conv_196 MP=8 (48-cyc shift-reg wrapper) · 17× MP=16/MP_K=8 · 7× MP=16/MP_K=9 · 11× MP=32/MP_K=8 (the kp4mp32-heritage MP=32 class).
- Precision mix: 18 INT3 + 35 INT4 conv layers (Config-B, §2 header; [MEM]+night-log provenance).

---

## 3. MobileNetV2 (INT8, 1,184,731-cycle netlist)

Quantization: INT8 per-OC (per-channel) requant, ~3.47 M weights; deployed accuracy 71.27% top-1 bit-identical at every gate [MEM]. Byte-exact gate: `output/mobilenet-v2/reports/final_bundle/e2e_result.txt` — `RESULT: PASS (8/8 byte-exact, TOTAL mismatch = 0)`, all 8 vectors `e2e_cycles=1184731 out_beats=32`. Inertness cross-gate: `output/reports_integrated/resnet_final_bundle/mbv2_inertness_8of8.log`.

### 3.1 Latest artifacts & paths

All under `D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/reports/synth/` unless noted.

| Artifact | Path | Size | Timestamp | Status |
|---|---|---:|---|---|
| Final-netlist synth report | `mbv2_synth.json` | — | Jun 11 16:23 | [SYNTH-EST], success=true, elapsed 2,071.0 s |
| Synth util (flat + hier depth-3) | `mbv2_util.rpt.synth` / `.synth.hier` | — | Jun 11 16:21 | [SYNTH-EST] |
| Synth DCP | `checkpoints/mbv2_post_synth.dcp` | 428,912,511 B | Jun 11 16:23 | banked, resume-ready |
| **Route final_c8 report** | `mbv2_route_final_c8.json` (+`.log`) | — | **Jun 12 12:50** | **[ROUTED], success=true, elapsed 24,236.8 s** — but read the rpt, not the json WNS (§3.4) |
| **Routed signoff rpts** | `checkpoints/mbv2_route_postroute_{timing,util,power}_final_c8.rpt` | 457,427 / 19,917 / 18,635 B | Jun 12 12:46–12:50 | **[ROUTED] — authoritative** |
| **Routed DCP** | `checkpoints/mbv2_route_routed_final_c8.dcp` | 362,101,558 B | Jun 12 12:45 | [ROUTED] (post-route-phys_opt overwrite of the 12:15 write) |
| final_c8 resume points | `checkpoints/mbv2_route_{opt,placed,physopt}_final_c8.dcp` | 94.4/273.9/271.9 MB | Jun 12 06:17/07:48/07:57 | banked |
| Post-place congestion rpt | `checkpoints/mbv2_route_congestion_final_c8.rpt` | 6,746 B | Jun 12 07:49 | [PLACE-EST] |
| Previous route (pre-seal netlist) | `mbv2_route_new_c8b.json` + `checkpoints/mbv2_route_postroute_{timing,util,power}_new_c8b.rpt` + `mbv2_route_routed_new_c8b.dcp` (358,737,570 B) | — | Jun 11 08:32–08:36 | [ROUTED], superseded by final_c8 |

Synth run identity (`mbv2_synth.json`): Vivado v2025.2, `synth_design -top nn2rtl_top -part xcu250-figd2104-2L-e -flatten_hierarchy rebuilt -verilog_define NN2RTL_ENGINE_SUBBLOCKS_PROVIDED=1 -verilog_define NN2RTL_SYNTHESIS=1`, 16 threads, synth_design 32 m 05 s, peak RAM 9.6 GB, session 34 m. Top source = `output/mobilenet-v2/rtl/nn2rtl_top_engine.v`. Clock 10 ns applied post-synth for reporting only; no synth timing captured.

### 3.2 Post-synth utilization (final netlist) — [SYNTH-EST]

Source: `mbv2_util.rpt.synth` (Design State: Synthesized). Routed deltas in §3.4.

| Resource | Used | % U250 |
|---|---:|---:|
| CLB LUTs | **329,371** | **19.06%** |
| — LUT as Logic | 315,339 | 18.25% |
| — LUT as Memory (all distributed RAM, 0 SRL) | 14,032 | 1.77% of 791,040 |
| CLB Registers (all FF, 0 latches) | **437,671** | **12.66%** |
| — FDRE 421,799 / FDCE 14,064 / FDSE 1,792 / FDPE 16 | | |
| CARRY8 | 8,921 | 4.13% |
| F7 / F8 muxes | 22,700 / 2,042 | 2.63% / 0.47% |
| Block RAM tiles | **1,812.5** | **67.43%** |
| — RAMB36E2 / RAMB18E2 | **1,809** / 7 | 67.30% / 0.13% |
| URAM288 | **235** | **18.36%** |
| DSP48E2 | **3,345** | **27.22%** |
| Bonded IOB | 328 | 48.52% |
| Unique control sets | 2,491 | — |

### 3.3 Memory anatomy

**BRAM36 ledger — 1,809, fully attributed (sums exactly; hier rpt + mapping reports in `mbv2_synth.json`):**

| Class | RAMB36 | RAMB18 | Detail |
|---|---:|---:|---|
| 8× engine KPAR8 weight banks `u_uram_weight_bank0..7` | **1,376** (172 each = 76.1% of all BRAM36) | 0 | XPM sprom ROM **2,324 × 2,304 b** each (8 × 288 b tap-major; "(18533+3 FC pad+54 DW+2 tail)/8 wide lines"); init `output/mobilenet-v2/weights/uram_weights_bank{N}_kp8.mem` (verified 2,324 lines × 2,304 b = 5.35 Mbit each); `ram_style=ultra` fell back to BRAM (`Synth 8-10226` ×8 — URAM can't be initialized) |
| `u_bias_mem` (engine per-oc_pass bias) | **114** | 0 | 256×8192 b shape, `bias.mem` 97 lines × 8192 b; derived via hier-gap arithmetic + log cell names `u_bias_mem/rd_data_reg_0..113` (exact) |
| `u_scale_mem` (per-OC requant {shift,mult}) | **114** | 0 | same module parameterization, `scale.mem` 97×8192 b |
| 15× skip/LHS residual FIFOs (`u_skip_/u_lhs_node_add_*`) | **193** | **7** | largest: add_198 pair 21+21 (4K×192 b each); 1038/1110 quads 18×4; 828/900 quads 11×4 |
| `u_node_conv_812` line buffer (3 slots 113×256 b) | **12** | 0 | |
| **Total** | **1,809** ✓ | **7** ✓ | |

**URAM ledger — 235, fully attributed; zero are weight memories** (all runtime-written, consistent with the URAM-no-init rule):

| Instance | URAM | Geometry |
|---|---:|---|
| `u_act_mem` (unified activation memory) | **203** (7×29 matrix) | 25,600 × 2,048 b |
| `u_engine_out_fifo` | **29** | 4,096 × 2,048 b |
| `u_node_conv_810/lbw` (stem line buffer) | **3** | 3 slots × 225 × 24 b (explicit ram_style=ultra) |

**LUT decomposition by module class** (hier rpt depth-1; sums to 329,369 + 2 in dissolved GCB logic): engine-out bridges ×51 **73,115** · act loader bridges ×50 **62,217** · n4 requant-tail nodes ×34 **53,753** (incl. 4,144 LUTRAM) · `u_shared_engine` **46,090** (DSP 3,075 = mac_array 2,048 + requant_pipeline 1,024 + addr_gen 3) · retile gather bridges ×4 36,067 · `u_node_mean` GAP 24,210 (9,888 LUTRAM, 16 DSP) · `u_engine_out_fifo` 9,263 · `u_node_conv_812` 5,579 (82 DSP) · node_add ×10 4,724 · stem `u_node_conv_810` 4,313 (144 DSP). DSP cross-check 3,075+144+82+16+28 = 3,345 ✓; LUTRAM 9,888+4,144 = 14,032 ✓. Hier-table caveat: `flatten_hierarchy rebuilt` dissolved 78 top instances into GCB blocks — bias/scale/act mems are invisible in the hier table (gap = exactly 228 B36 and 203 URAM).

**Weight bit budget:**

| Store | Bits | % of chip BRAM |
|---|---:|---:|
| Engine KPAR8 banks (34 dense + 16 DW + FC) | **42.84 Mbit allocated** (8 × 2,304 b × 2,324 lines; INT8 content ≈ 27.8 Mbit = 28.1%, rest tile padding; 9 b/lane × 32 lanes per 288 b tap word) | 43.2% |
| Spatial remnants (stem 810 + DW 812 — only 2 spatial convs left) | 9.2 Kbit (`node_conv_810_weights_mp_k_9.hex` 6.9 Kb + `node_conv_812_weights.hex` 2.3 Kb) | ~0.01% |
| Engine bias/scale mems | 1.59 Mbit (`bias.mem` + `scale.mem`, 0.795 each) | 1.6% |

### 3.4 Route history: c8b routed + final_c8 ROUTED today

Both routes: 1,184,731-cyc netlist family, clock 8 ns, **--no-pblock** (the old SLR pblock crashed the placer on the quarter-size netlist and is retired [MEM, 5c2d571]), `place/route -directive Explore`, resumed from `mbv2_post_synth.dcp`, 16 threads.

**`new_c8b` (Jun 11 05:11→08:36) — pre-seal 1,184,731-cyc netlist (launched before the 10:27 seal; lacks the tile_idx replication + final fanout caps).** [ROUTED, superseded by final_c8]

- **Headline correction:** the recorded "86.67 MHz" is **NOT routed** — the json top-level `setup_wns_ns=−3.538/fmax=86.67/hold=−0.173` latched the *post-place phys_opt estimate* (a parser artifact; −3.538 appears exactly once in the log, in the placement-stage Estimated Timing Summary). The signoff `mbv2_route_postroute_timing_new_c8b.rpt` says: **setup WNS −4.019 @ 8 ns → Fmax 83.20 MHz**, TNS −293,318.8 (280,978/1,303,145 failing endpoints), **hold WHS +0.005 MET**. timing_met=false at 8 ns. fps = 83.20×10⁶/1,184,731 = **70.2**.
- Critical path (measured): `u_engine_out_node_conv_876/g_tiled.tile_idx_reg[1]` → `data_out_reg[82]`, 11.756 ns of which **route 11.563 ns = 98.36%** (2 logic levels) — exactly the **tile_idx bridge class** the sealed netlist then fixed by ×13 tile_idx replication.
- Post-route util [ROUTED]: LUT 319,079 (18.47%) / FF 437,555 / BRAM 1,812.5 / URAM 235 / DSP 3,345. Session 12,296.9 s = 3 h 25 m; route_design 1:20:43, Global Iters 0–4, congestion global/short level 6, timing level 7, peak RAM 24.5 GB.

**`final_c8` (Jun 12 06:06→12:50) — SEALED final netlist (50c3054). COMPLETED; Vivado pid 7284 exited.** [ROUTED — current best]

| Phase | Elapsed | Artifact time |
|---|---|---|
| opt_design | 6:26 | dcp 06:17 |
| place_design Explore | 1:30:34 | dcp 07:48 |
| pre-route phys_opt | 6:03 | dcp 07:57 — placement est. WNS −1.899 [PLACE-EST] |
| route_design Explore | **4:17:27** (15,402.7 s internal; Global Iters 0–8, overlaps 488,940 → 0; **no** "congestion preventing router" message this time) | routed dcp 12:15 |
| post-route phys_opt + rpts + json | ~35 min | dcp overwrite 12:45, rpts 12:46–12:50, json 12:50 |

- **Signoff (authoritative, `mbv2_route_postroute_timing_final_c8.rpt`, Design State: Physopt postRoute, clock 8.000 ns): setup WNS −2.199, TNS −81,281.7 (137,944/1,303,732 failing endpoints); hold WHS +0.006, THS 0.000 — hold MET.** → **Fmax = 1000/10.199 = 98.05 MHz → 82.76 fps.** timing_met=false at the 8 ns constraint itself (98 MHz operation is the supported claim, not 125 MHz).
- ⚠ Same parser artifact as c8b: `mbv2_route_final_c8.json` top-level carries `setup_wns_ns=−2.370 / fmax 96.43 / hold −0.177` — an earlier in-flow intermediate. **The rpt wins.** Routed util from `…postroute_util_final_c8.rpt`: LUT **322,628** (18.67%) / FF 438,155 / BRAM 1,812.5 / URAM 235 / DSP 3,345.
- **Netlist-fix verification:** +1.82 ns WNS gain over c8b (−4.019 → −2.199) on the same clock = ~+14.9 MHz / +12.5 fps; the tile_idx class is **gone** from the final critical set (post-route phys_opt was working `u_shared_engine/u_bram_to_stream_bridge/wr_data[*]`, `u_act_mem/rd_data[*]`, loader wr_req nets instead) — the ×13 tile_idx replication + fanout caps did their job.

**All MBV2 route attempts:**

| Tag | Netlist / cycles | Clock | Result | WNS setup/hold (status) | Fmax / fps |
|---|---|---|---|---|---|
| `_c8` (Jun 8) | old 7.59 M netlist | 8 ns | superseded history | — | — |
| `_c8_pblock` route_only (Jun 10) | old 7,592,966 | 8 ns | routed (old netlist; dcp not kept) — superseded history | −2.708 [ROUTED, `route_only_synth.json` fmax 93.388] | 93.39 MHz / 12.30 fps |
| `new_c8` (Jun 11 05:09) | pre-seal 1,184,731 | 8 ns | aborted/short predecessor — history | — | — |
| `new_c8b` | pre-seal 1,184,731 | 8 ns | ROUTED, superseded by final_c8 | **−4.019 / +0.005 [ROUTED]** (json's −3.538 = place-est artifact) | 83.20 MHz / 70.2 fps |
| **`final_c8`** | **SEALED 1,184,731** | 8 ns | **ROUTED Jun 12 12:47 — current best** | **−2.199 / +0.006 [ROUTED]** | **98.05 MHz / 82.76 fps** |

### 3.5 Throughput

Cycles/frame = **1,184,731** (sealed `50c3054`; gates `final_bundle/e2e_result.txt` 8/8 PASS; commits `8c2166e`, `5c2d571` cycle-exact).

| Clock scenario | fps = f×10⁶/1,184,731 | Status |
|---|---:|---|
| **98.05 MHz** (WNS −2.199 @ 8 ns, hold MET) | **82.76** | **[ROUTED — final_c8, Jun 12]** |
| 86.67 MHz | 73.16 | [PLACE-EST artifact of c8b — retired number] |
| 83.20 MHz (c8b signoff) | 70.23 | [ROUTED, superseded] |
| 125 MHz (the 8 ns constraint itself) | 105.51 | [TARGET, not met] |
| — campaign start: 93.39 MHz × 7,592,966 (old netlist) | 12.30 | [ROUTED, superseded history] |

**Cycle composition:** the front zone is paced by the single remaining spatial DW conv `node_conv_812` ("the ONLY spatial depthwise conv left and paces the entire FRONT zone, with the engine 100% idle under it" — `docs/agent_tasks/PAIR812_ANALYSIS.md:26-27`; hence the 812-PAIR 2 ch/cyc lever, `2e639df`). Engine carries 51 dispatches incl. 16/17 DW + FC. Remaining headroom: FRAME-PIPE ~30% throughput, parked [MEM]; KPAR8 9-operand adder tree TREE_STAGES=1 documented as the timing lever toward 100+ MHz [MEM].

### 3.6 Interface anatomy

Top: `output/mobilenet-v2/rtl/nn2rtl_top_engine.v` (module `nn2rtl_top`, line 15; the sibling `nn2rtl_top.v` in the same dir is the pre-engine variant — do not use). Single clock domain.

**Clock / reset**

| Signal | Width | Dir | Meaning |
|---|---|---|---|
| `clk` / `rst_n` | 1 / 1 | in | single clock, active-low reset (`nn2rtl_top_engine.v:16-17`) |

**`s_axis` — AXI4-Stream slave, input image (50,176 beats/frame)**

| Signal | Width | Dir | Meaning |
|---|---|---|---|
| `s_axis_tdata` | **24** | in | one **24-bit RGB pixel per beat** — no padding, unlike ResNet's 256 b beat (`:18`) |
| `s_axis_tvalid` / `s_axis_tready` | 1/1 | in/out | handshake (`:20-21`) |
| `s_axis_tlast` | 1 | in | beat 50,175 — 224×224 pixels (TB `tb/mbv2_top_value_tb.cpp:71`) |

**`m_axis` — AXI4-Stream master, classification logits (32 beats/frame)**

| Signal | Width | Dir | Meaning |
|---|---|---|---|
| `m_axis_tdata` | **256** | out | logits: on-engine `node_linear` FC emits one 8,000-bit word = **1,000 INT8 logits** (`wire [7999:0] node_linear_data_out` `:301`), resliced byte-exact by `output_serializer #(.W_IN(8000), .BEATW(256))` (`:4130`) into **32 beats = ceil(8000/256)**; 32 logits/beat, last beat = logits 992–999 in low 64 b + zero pad (`:4126-4128`; TB `tb:252,445-448`) |
| `m_axis_tvalid` / `m_axis_tready` | 1/1 | out/in | handshake |
| `m_axis_tlast` | 1 | out | final logit beat (serializer `last_out` `:4137`) |

**`s_axil` — AXI4-Lite control slave**: identical pinout and register-block layout to ResNet (addr/data 32, wstrb 4, resp 2; `:31-48`) → shared_engine config block (`docs/agent_tasks/10_engine_config_register_block.md`). The only interface deltas between the two networks: `s_axis_tdata` width (256 vs 24) and output semantics (feature map 3,136 beats vs logits 32 beats).

### 3.7 Netlist statistics

- **Layer modules: 99 — header is CURRENT** ("Layers total: 99, spatial: 48, engine-dispatched: 51, residual adds: 10, projection convs: 11", `nn2rtl_top_engine.v:7`). Spatial 48 verified by instantiation count: 35 ReLU6 requant-tail modules (`n4`, `n4_2..35`, mostly DSP-free ROM-requant) + stem `node_conv_810` (3×3 s2) + DW `node_conv_812` + 10 `node_add_*` + 1 `node_mean` (GAP).
- **Engine dispatches: 51** (`output/mobilenet-v2/rtl/nn2rtl_scheduler.v:3`, `LAST_DISPATCH=6'd50`): **34 dense 1×1 pointwise** (conv_814…912) + **16 depthwise** (12 stride-1: conv_824/836/842/854/860/866/872/878/884/896/902/908; 4 stride-2 quartet-fill: conv_818/830/848/890) + **1 FC** (`node_linear` @ dispatch 50). Per-dispatch DEPTHWISE flag table → engine cfg reg 0x3C (`nn2rtl_scheduler.v:1355-1359`).
- **DW split: 16/17 on engine**; the one spatial DW is `node_conv_812` (C=32, 112×112) with the **812-PAIR** 2-channels/cycle MAC walk (`node_conv_812.v:6`).
- **Engine config:** `K_PAR=8` (`nn2rtl_top_engine.v:2896`) · `ENG_PIPE=1` (FSM bubble 12/10→3, `:2897`) · `ENABLE_DEPTHWISE=1` (`:2909`).
- **Weight banks:** 8 banks, **2,304-bit lines (8 × 288 b tap-major) × depth 2,317 group-addressed wide lines** (`.WORD_W(2304)` `:1433-1510`; mem files padded to 2,324 lines); 9 b/lane × 32 lanes per tap word.
- ⚠ Stale artifacts: `nn2rtl_scheduler_schedule.json` still says num_dispatches=34 (predates the DW/FC/quartet chain — scheduler .v is authoritative); top-level comments label the FC bridge "SLOT 46" (pre-quartet numbering, not the dispatch index).

---

## 4. Head-to-head summary

| Metric | ResNet-50 (final netlist) | MobileNetV2 (final netlist) |
|---|---|---|
| Cycles/frame (byte-exact sealed, 50c3054) | 5,299,588 (vec0+vec1, 0/100352) | 1,184,731 (8/8 vectors) |
| Last-routed Fmax **of this netlist** | **none — c14 route FAILED** (prev. netlist: 83.33 MHz MET @12 ns [ROUTED]) | **98.05 MHz** (WNS −2.199 @ 8 ns, hold MET) [ROUTED Jun 12] |
| fps @ last-routed | — (prev. netlist: 14.71 fps @ 5,664,715 cyc) | **82.76 fps** |
| Clock target / constraint in effect | 12 ns synth-baked (flags 14/16 ns never applied) | 8 ns (not met; 98 MHz supported) |
| Synth LUT | 1,209,699 = 70.01% | 329,371 = 19.06% (routed 322,628 = 18.67%) |
| Synth BRAM36 tiles | 2,656 = 98.81% (binding) | 1,812.5 = 67.43% |
| Synth DSP | 8,007 = 65.16% | 3,345 = 27.22% |
| URAM | 662 = 51.72% | 235 = 18.36% |
| Weights footprint | 74.55 Mbit = 75.2% chip BRAM (INT3/INT4 Config-B) | 42.85 Mbit alloc = 43.2% (INT8 content 27.8 Mbit) |
| Engine dispatches / spatial layers | 17 (9×3×3 + 8×1×1) / 36 spatial convs | 51 (34 pw + 16 DW + FC) / 2 spatial convs |
| Engine config | K_PAR=8, ENG_PIPE=1, WADDR_REP=8, INT3 weights | K_PAR=8, ENG_PIPE=1, DEPTHWISE=1, INT8 |
| Route status | **FAILED** `_final_c14` (22,199 overlaps, congestion-infeasible @12 ns placement); 3 resume DCPs banked; **AggressiveExplore route-only RETRY running since Jun 12 12:52** | **ROUTED & signed off** `final_c8`; dcp + rpts + json on disk |
| Accuracy | 77.60% top-1 (Config-B) [night log :708] | 71.27% top-1 [MEM] |

---

## 5. Known unknowns / pending

1. **ResNet final-netlist routed Fmax is UNKNOWN** — the only completed attempt (c14) failed organically (adversarially verified; not a timeout artifact). **A route-only RETRY is RUNNING (since Jun 12 12:52): `route_design -directive AggressiveExplore` from the same `physopt_final_c14.dcp`** — same 12 ns placement, stronger router, **no kill risk this time** (healthy at 13:26: 34.9 GB, pre-route WNS +0.563, Phase 5.1). Decision tree: if it closes → c14 was marginal and the directive rescues it; if it fails the same way with no timeout in play → final nail in the placement, and the **weight_bus/SLR1 epicenter becomes the fix target for the third-run synth branch**. Next options after that, in order of evidence: (a) **fix the clock bug first** — `run_resume_from_synth.ts:91` `set_property PERIOD` → unconditional `create_clock`; a genuinely relaxed 14–16 ns flow is **untested territory** and needs re-place, not just re-route, to benefit; (b) attack the proven epicenter — `u_uram_weight_bank*/weight_bus[*]` broadcast through the SLR1 URAM/DSP band (SLR0↔1 SLLs at 114%/104%): bank output pipelining/replication, or conv_288→engine (−285 BRAM relief [MEM]); (c) the placed/physopt c14 DCPs are valid resume points but carry the 12 ns placement.
2. **Durability: RESOLVED** — the c14 failure log is preserved at `output/reports_integrated/failed_route_final_c14/vivado_full.log` (copied Jun 12 13:00; original in `Temp/nn2rtl-resume-YIHf1a/`).
3. **kp4mp32 67.15 MHz vs 84.05 MHz:** the on-disk signoff rpt (12 ns MET, WNS +0.102, hold +0.010) contradicts the memory/thesis number. One-line `open_checkpoint first_light_routed_kp4mp32_c16.dcp; report_timing_summary` settles it definitively; until then the disk rpt governs. **`docs/agent_tasks/THESIS_FINN_HLS4ML_COMPARISON.md:12` ("11.9 fps / 67.15 MHz") and both MEMORY entries understate the routed result and need correction** — same for the MBV2 "86.67 MHz / 73.2 fps" memory claims (real c8b = 83.20 / 70.2; now moot, final_c8 = 98.05 / 82.76).
4. **MBV2 final_c8 report has landed** (the "pending" status in earlier notes is resolved) — but note `timing_met=false` at 8 ns: 98.05 MHz is the timing-signoff Fmax, not a met constraint. A clean "met" claim needs either a ~10 ns re-route or the documented TREE_STAGES=1 KPAR8 adder-tree lever (toward 100+ MHz) [MEM].
5. **Hold-fix phys_opt: RESOLVED on disk for both routed DCPs** — kp4mp32_c16 WHS +0.010 and final_c8 WHS +0.006, both MET (the in-flow post-route phys_opt already cleaned them). The memory item "hold −0.17..−0.18 needs cleanup" is overtaken by disk evidence.
6. **JSON parser artifact class (systemic):** top-level `setup_wns_ns/fmax_mhz/hold_wns_ns` in the route JSONs can latch pre-route or intermediate estimates (proven on c8b −3.538 and final_c8 −2.370), and a timeout kill **severs the stdout pipe and freezes the JSON mid-run** (proven on c14: stale place-est fields recorded as the run's "result"). **Always read `…postroute_timing_*.rpt`**; consider fixing the parser to grab the last Design Timing Summary. Related tooling note: the `execFile` timeout path has **no tree-kill** (only the RAM-watchdog branch does `taskkill /T`), so a "killed" run's vivado.exe is actually orphaned and keeps running — here that accident preserved the forensics, but it also means a timeout never frees the license/RAM and a relaunch could collide with the orphan; decide deliberately whether to add tree-kill or keep orphan-and-log.
7. ResNet Config-B per-layer precision mix (18 INT3 / 35 INT4) is not recoverable from synth artifacts — provenance is the night log + `configb_acc_bram_sweep.json` only; worth snapshotting the per-layer table into the repo docs for the thesis.
8. Stale metadata to not trip over: ResNet top header "spatial 105/engine 14" (pre-K5), MBV2 `nn2rtl_scheduler_schedule.json` num_dispatches=34 (pre-DW/FC/quartet), MBV2 FC bridge "SLOT 46" comments. The generated schedulers are authoritative in all three cases.
9. Parked performance levers (both nets): FRAME-PIPE (~30% MBV2 throughput, deadlock-adjacent), ResNet stem rework + ENG-PIPE-for-throughput iteration, post-fit OOC/incremental synth recipes [MEM].
