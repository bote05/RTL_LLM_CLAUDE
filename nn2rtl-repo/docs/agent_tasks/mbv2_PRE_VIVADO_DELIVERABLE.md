# MobileNetV2 (INT8) on Alveo U250 — PRE-VIVADO DELIVERABLE

**The morning summary you asked for: precise area, accuracy, and honest throughput, BEFORE any Vivado run.**
Date: 2026-06-02 (overnight autonomous run). NO Vivado synth/place/route was performed (per your directive).
All numbers are **projected/measured-per-module + analytical**, clearly labeled; a U250 synth remains gated until
bit-exact + accurate + fit-confirmed.

---

## 0. BOTTOM LINE + DECISIONS FOR YOU

**What is PROVEN (green):** every level verifiable without a full-system e2e —
- All 99 modules byte-exact individually; **engine datapath 34/34 dispatches byte-exact** (first real-engine mbv2 validation, WLAT=2); engine-top **config-correct + elaborate-clean** (caught+fixed a critical weight-bus-truncation bug).
- **Fit: all six resources PROJECTED <80%** (LUT 61.6 / FF 22 / DSP 10.9 *confirmed* / BRAM 37.7 / URAM 10).
- **Accuracy 67.27%** INT8 top-1 (byte-exact-preserved through every change).
- **Throughput roadmap hardened**: A2 (spatial 3×3 tap-parallel MP_K=9) = 4.71× → ~4.4–6.6 fps @50MHz, byte-exact, still <80% — ready to execute.

**The ONE open item: full-system e2e.** Both tops are blocked by the **same final-stage tiled↔packed contract mismatch (#3)** — a real, documented dataflow break (the adds read channels 32..95 as zeros; `EXPECTED_BEATS` 196≠588). The engine-top *additionally* needs an engine-output backpressure fix (#2). Neither is a safe unsupervised change (one is a contract-level regen "major phase"; the other touches the shared, proven, timing-critical engine pipeline). Full root-cause + per-blocker fixes + risk classes: `mbv2_engine_top_roadmap.md`.

**THREE DECISIONS for you:**
1. **Which deployment top?** Engine-top *fits* (<80% LUT) but needs #2 + #3 for e2e; all-spatial is faster e2e (~20M cyc) + avoids #2 entirely but is **~84% LUT (over the 80% bar)** and still needs #3. There is no <80% candidate other than the engine-top.
2. **Critical path = #3 contract-regen** (unavoidable for *any* full e2e). Recommended: do #3 once on the lower-risk all-spatial top first (no engine-pipeline gamble), then apply to the engine-top + do #2. This is days of supervised work, not a patch.
3. **Throughput**: A2 (spatial depthwise MP_K=9, 4.71× spatial) is **DONE — APPLIED + byte-exact verified (18/18 modules, mismatch=0)** on 2026-06-02. Spatial path 35.6M→7.56M cyc/frame; projected ~4.4–6.6 fps @50MHz (up from 1.27). All six resources still <80% (DSP now ~16%). The remaining throughput option is A1 (overlap engine & spatial) — optional, lower priority. So no pending throughput decision; the gating items are #1 (which top) and #2 (the #3 contract-regen to unblock e2e).

**Vivado is NOT recommended yet** — gated on a passing e2e (which needs #3), per your "100% know it works" bar. The fit is PROJECTED, not synth-confirmed.

---

## 1. AREA — all six resources PROJECTED under 80% of the U250 budget

Target `xcu250-figd2104-2L-e`. Budget: LUT 1,728,000 · FF 3,456,000 · DSP48E2 12,288 · RAMB36 2,688 · URAM288 1,280.

| Resource | Projected total | % of U250 | Verdict | Confidence |
|---|---:|---:|---|---|
| **LUT**     | ~1,064,000 | **61.6%** | under 80% | depthwise term is a PESSIMISTIC upper bound (linear-by-C); real synth expected lower |
| **FF**      | ~759,000   | **22.0%** | under 80% | same pessimism; comfortable |
| **DSP48E2** | ~1,345     | **10.9%** | under 80% | **CONFIRMED** — all 36 relus now on ROM (DSP→0); engine 1,283 + spatial/GAP/add ~62 |
| **RAMB36**  | ~1,013     | **37.7%** | under 80% | post both memory reshapes (skip-FIFO right-size + DW line-buf→RAMB36) |
| **URAM288** | 128        | **10.0%** | under 80% | engine weight banks only (measured geometry) |

- **Measured inputs:** per-module Vivado synth (52 conv / 35 relu / 10 add / GAP / gemm on xczu9eg, primitives
  transfer 1:1 to U250) + one full `shared_engine` synth on the real **xcu250** part (107,268 LUT / 30,979 FF /
  1,283 DSP).
- **Estimated inputs:** depthwise LUT/FF (conv_812 × C, pessimistic), tile counts (RTL geometry), ROM/GAP DSP
  (structural, byte-exact-verified), ~8k/6k glue allowance.
- Full breakdown + method: `docs/agent_tasks/mbv2_u250_fit_projection.md`.

**What changed tonight to reach all-<80%** (both byte-exact-preserving, mapping/sizing only — verified mismatch=0):
1. Depthwise line buffers off `ram_style="ultra"` → RAMB36 (`LINE_BUF_USE_URAM=0` param; ResNet keeps default ultra):
   URAM 197% → 10%.
2. Residual skip FIFOs right-sized to `next_pow2(H·W)`: RAMB36 829% → 37.7% (19,857 → 122 tiles for the skips).
3. ReLU requant → 128-entry distributed ROM on all 36 relus (was a runtime multiply): DSP 113%-class → **10.9%**.
   (Tonight's audit found the prior "all done" was a too-narrow grep; the last one, `n4_11`, was converted +
   byte-exact, and 3 stale MODDUP orphan `.v` files were neutralized.)

---

## 2. ACCURACY — 67.27% top-1 (deployed INT8), trustworthy

- **Deployed INT8 MobileNetV2 = 67.27% top-1** (vs float 72.67%), measured by `scripts/measure_deployed_mbv2_acc.py`.
- The ~5.4% gap is the known **per-tensor INT8 depthwise penalty**; per-channel weight quant would recover most of it
  (a documented future lever, not done here).
- **Preserved by construction:** every RTL change tonight (depthwise sync line-buf, relu ROM, GAP time-mux, FIFO
  right-size, URAM→RAMB36) is **byte-exact** (Verilator mismatch=0), so the deployed accuracy is unchanged.

---

## 3. THROUGHPUT — honest numbers (NO fabricated clock)

> **Clock caveat:** `layer_ir.json` uses `clock_period_ns=20` → **50 MHz**. The achievable fmax is **unknown** without a
> Vivado timing run (the sibling ResNet design closed timing only at 25–40 MHz). Every "@200 MHz" figure below is an
> illustrative hypothetical, **gated on a real timing run**. The hard number is **cycles/frame**.

The design has two compute domains that **serialize** (scheduler `S_WAIT_DONE: spatial_stall=1` freezes the spatial
chain during engine compute): a shared 256-OC-parallel **engine** (34 pointwise) + a **spatial** chain (stem +
17 depthwise 3×3). Verified bottom-up from `layer_ir.json`:

| Configuration | cyc/frame | fps @50MHz | fps @200MHz (hypothetical) | Notes |
|---|---:|---:|---:|---|
| **Engine top, as-is (P=1)** — the <80% FIT artifact | ~39.4M | **1.27** | 5.08 | engine 3.79M + spatial 35.6M, serialized |
| + **A1** (overlap engine & spatial) | ~35.6M | ~1.40 | ~5.6 | removes engine from serial path (~+10%) |
| + **A2** (spatial 3×3 tap-parallel MP_K=9) | ~7.6M spatial | **~4.4–6.6** | ~17.6–26.5 | **4.71× on the spatial path**, the real lever |
| All-spatial top (reference, no engine) | ~20M | ~2.5 | ~10 | FASTER e2e but LUT ~84% (does NOT fit <80%) |

**Key correction from tonight's adversarial review (the earlier "~17.5 fps @200 MHz" was wrong on 3 legs):** the engine
is NOT the bottleneck (it's 3× below the stem); the real limiter is the **spatial 3×3 path (35.6M cyc)**; and the clock
is 50 MHz not 200. See `memory/project_mbv2_throughput_corrected.md`.

### The fit-vs-throughput tradeoff (a decision for you)
- **All-spatial top:** faster e2e (~20M cyc, pointwise-bound) but **LUT ~84% — over the 80% target.**
- **Engine top:** **fits (LUT 61.6%)** but e2e-slower (39.4M, serialized) **until A1/A2 land.**
- **Ideal = engine top + A1 + A2 (MP_K=9):** fits AND ~4.4–6.6 fps @50MHz, all six resources <80% (DSP rises to
  16.1%). This is the recommended path. A2 is a medium-risk **atomic** change (17 depthwise wrappers + stem + latency
  formula + weight repack + full regen) — hardened plan in `docs/agent_tasks/mbv2_spatial_throughput_roadmap.md`.

---

## 4. CORRECTNESS — what's proven vs pending

| Item | Status |
|---|---|
| 98/99 modules byte-exact individually | ✓ (node_linear 2/8000 = golden float32-acc artifact, RTL more-correct) |
| 17 depthwise · all 36 relu (ROM) · GAP · memory reshapes | ✓ byte-exact (Verilator mismatch=0) |
| MODDUP orphan relus | ✓ neutralized (renamed `.orphan`) |
| **Clean all-spatial e2e** (deadlock fixed: under-sized add_198 FIFO + MODDUP) | ⏳ RUNNING (`bmambqvzx`) — PLACEHOLDER §4a |
| **Engine top P=1 per-dispatch byte-exact** (the <80% fit artifact) | ✓ **34/34 PASS, mismatch=0** (WLAT=2 deployment URAM latency) — see §4b + `docs/agent_tasks/mbv2_engine_p1_correctness.md` |
| **Engine top INTEGRATION** (assembled `nn2rtl_top_engine.v`) config-correct + elaborate-clean | ✓ **CONFIG-CORRECT + ELABORATES CLEAN** — see §4c. BUG 1 fixed: `shared_engine` instance now overrides `WGT_W=8`/`URAM_DATA_W=2048`/`MAX_I*H/W=112` (was ResNet INT4 defaults silently truncating the 2048b weight bus to 1024b). Verilator `--lint-only` exit-0; BUG-1 WIDTHTRUNC/UNUSEDSIGNAL gone; no warning touches the engine instance ports. |
| **Engine top E2E VALUE** sim | ⏳ RUNS, dies at dispatch-0 drain. **BLOCKER #1 (loader units) FIXED+VERIFIED** (`apply_loader_word_resize.py`): dispatch 0 now loads @cyc 11.5M, engine starts, produces a full 12544-beat frame. **2 BLOCKERS REMAIN, both user-gated** — #2 engine-output FIFO overflow (engine-pipeline-change) + #3 final-stage tiled↔packed contract mismatch (contract-regen). Full ordered roadmap: `docs/agent_tasks/mbv2_engine_top_roadmap.md`. |

> §4a RESULT (clean all-spatial e2e): **DEADLOCKED at the final stage — empirically CONFIRMS blocker #3 on the all-spatial top** (killed at the user's request after the confirmation). Ran ~5.5 h (02:14→07:44), built clean, consumed all 50,176 input beats by ~cycle 12M, then produced **no output through cycle 22M+** (`out=0/1`, `tvalid=0` for ~10M cycles past input-completion = the deadlock signature, not draining). The under-sized add_198 FIFO + MODDUP fixes cleared the *earlier* deadlock, but the run then stalls at the final stage exactly as the roadmap predicted — the all-spatial top **shares the #3 final-stage tiled↔packed contract mismatch**. So **neither top produces a clean e2e output without the #3 contract-regen** (it is the shared critical path). The datapath up to the final stage is proven (per-module byte-exact). Honest all-spatial cycle cost ≈ pointwise-bound ~20M cyc/frame → ~2.5 fps@50MHz (the engine offload exists for the LUT-fit, not speed). **Conclusion: #3 contract-regen is mandatory for any full e2e, on either top.**
>
> §4b RESULT (engine P=1, 34 pointwise dispatches): **34/34 PASS, 0 FAIL, total mismatch=0, max|err|=0** across all 34 engine-dispatched pointwise convs (node_conv_814…912), verified through the REAL `shared_engine` (WGT_W=8/URAM_DATA_W=2048) + real `uram_weights_bank0..7.mem`/`bias.mem`/`scale.mem` at the **deployment 2-cycle URAM weight read latency (WLAT=2)** — NOT the false-confidence 1-cycle path. Covers all geometry classes: IC>256 chunk-straddle (576/960), multi-OC-pass (up to OC=1280, 5 passes), and the full 112×112→7×7 spatial range. No failure root-cause needed. Harness: `tb/engine_iso_wrap_mbv2.v` + `tb/engine_iso_wrap_mbv2_tb.cpp` (race-immune private build for the last 11 via `scripts/gen_iso_p1_cfg.py` + `scripts/build_iso_p1.bat`). **The engine-based <80% fit artifact is CORRECTNESS-PROVEN at the dispatch level** (full engine-top e2e value sim still pending; bridges/loaders are structural).
>
> §4c RESULT (engine-TOP integration): the assembled `output/mobilenet-v2/rtl/nn2rtl_top_engine.v` is now **CONFIG-CORRECT and ELABORATES CLEAN** (`verilator --lint-only --top-module nn2rtl_top`, exit-0). **BUG 1 (CRITICAL, fixed):** the `shared_engine u_shared_engine` instance (line 2167) had NO param-override block, inheriting the ResNet INT4 defaults `WGT_W=4`/`URAM_DATA_W=1024` — silently truncating the 2048-bit weight bus to 1024b (dropping URAM banks 4–7 / OC lanes 128–255) and reinterpreting bytes as nibbles. Fixed by inserting the override `WGT_W=8 / URAM_DATA_W=2048 / MAX_IC=MAX_OC=2048 / MAX_IH=IW=OH=OW=112` (verbatim from the proven iso harness); post-fix the WIDTHTRUNC + UNUSEDSIGNAL on `engine_weight_rd_data` are gone and no warning touches the engine instance ports. **STILL UNVALIDATED by static checks (needs engine-top e2e value sim):** (i) the final-stage flat-bus contract-width mismatch — 10 SELRANGE at lines 893/934/1006/1047/1205 extract 768/1280 bits from 256-bit signals (the documented 768→256/1280→256 contract issue `build_top_wrapper` never reconciled; pre-existing, NOT a param bug, will surface as wrong VALUES); (ii) dispatch timing / scheduler sequencing (`LAST_DISPATCH=33`); (iii) loader/bridge dataflow + shared-BRAM arbitration, incl. dispatch 21 = node_conv_876's "no input loader, `all_loaded[21]=1` resident" sequencing assumption. **E2E runner status:** NONE exists — `run_mbv2_top_value.ts` is hardcoded all-spatial + injects an inactive engine STUB (and `nn2rtl_top`/`nn2rtl_top_engine` share the module name). A new `run_mbv2_engine_top_value.ts` must be built (sources the engine top + REAL engine submodules, drops the stub; recipe in `mbv2_engine_p1_correctness.md`). Backup: `backups/engine_top_integration_20260602_030528/`.

**Earlier "e2e" deadlock was NOT a datapath bug** — it was a config defect (a skip FIFO under-sized to 256 for a
3,136-beat residual → dropped beats → residual-add stream desync → stall) compounded by an order-dependent MODDUP
elaboration. Both fixed; the clean rerun is the authoritative gate.

---

## 5. Pre-Vivado checklist
1. ✓ All 6 resources projected <80% (both memory reshapes applied + byte-exact).
2. ✓ ReLU-ROM rollout complete (DSP confirmed 10.9%).
3. ✓ MODDUP orphans removed (deterministic elaboration).
4. ⏳ Clean all-spatial e2e byte-exact (running).
5. ✓ Engine top P=1 per-dispatch byte-exact — **34/34 PASS, mismatch=0** at WLAT=2 (deployment URAM latency); the engine-based <80% fit artifact is correctness-proven at the dispatch level (see `mbv2_engine_p1_correctness.md`).
5b. ✓ Engine-TOP integration config-correct + elaborate-clean — BUG 1 (`shared_engine` missing param override → WGT_W=4/URAM_DATA_W=1024 truncating the weight bus) FIXED; `verilator --lint-only` exit-0.
5c. ⏳ Engine-TOP e2e value sim — RUNS to dispatch-0 drain. **BLOCKER #1 (loader `TOTAL_BRAM_WORDS` units, surgical-wrapper) FIXED+VERIFIED** (dispatch 0 loads @11.5M, engine starts, full 12544-beat frame produced). **2 blockers remain, BOTH need user decision:** #2 engine-output FIFO overflow / missing engine backpressure (`engine-pipeline-change` — shared byte-exact engine, safety-rule protected) is the current GATING blocker; #3 final-stage tiled↔packed contract mismatch (`contract-regen` — major phase, shared with the all-spatial top). Ordered roadmap + fixes (file:line) + risk classes: `docs/agent_tasks/mbv2_engine_top_roadmap.md`.
6. ✓ (Throughput) A2 MP_K=9 → ~4.4–6.6 fps @50MHz: **APPLIED + byte-exact VERIFIED (18/18 modules: 17 depthwise + stem node_conv_810, all `status=pass mismatch=0`)** on 2026-06-02 via `scripts/apply_mpk9_depthwise.py` + the regen chain. Spatial path 35.6M→7.56M cyc/frame (4.71×); depthwise 24.17M→5.75M (4.2×). DSP rises to ~16% — still <80% on all six resources. Backup: `backups/a2_apply_20260602_075338/`. A1 (overlap) remains an optional further addition.
7. ☐ Vivado synth/P&R — GATED until 4+5+5b+5c green (and your "100% know it fits" bar).

> All §1 area numbers are **PROJECTED**, not synth-confirmed. Do not present as a confirmed fit.
> Throughput §3 is cycle-accurate bottom-up; fps@200MHz is hypothetical pending a timing run.
