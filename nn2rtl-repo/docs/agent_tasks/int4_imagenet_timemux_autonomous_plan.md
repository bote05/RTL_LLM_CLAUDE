# Full Autonomous Plan: ImageNet Goldens + INT4 + Engine Time-Multiplexing → Fit + Vivado

You are taking the working INT8 design (e2e verified, ~13.35M cycles) and converting it to an **INT4, ImageNet-calibrated, engine-time-multiplexed** design that fits all-on-chip in **bitstream-initializable BRAM**, then measuring it in Vivado. Run as autonomously as possible. Stop only on the hard conditions in each phase.

---

## PRE-COMMIT FINDINGS (2026-05-28, analysis-only — done before the rework)

Phase 1.5 and Phase 3.2 were run as pure analysis (no build) to decide whether INT4 is worth the rework. Results, in short — **yes, commit to INT4 / Scheme A:**

- **FIT IS RESOLVED — INT4 fits BRAM.** Hard per-array BRAM36 count (from actual wide-ROM hex geometry, optimal aspect-ratio packing): spatial 1528 + engine shared-store 1131 + biases 57 = **2716 BRAM36 = 101% of 2688 (over by just 28).** The overage is pure width-rounding waste in ~20 tiny shallow 1×1 ROMs. **Moving biases (57 BRAM36) to LUTRAM → 98.9% → fits**, with more headroom from narrow-repacking the shallow 1×1s. Raw bits are only 95.5%. (script: `uram_init_test/bram36_count_int4.py`)
- **PRECISION = Scheme A (INT4 weights / INT8 activations).** Decisive on all 3 axes: identical fit (full-INT4 buys nothing on the binding BRAM constraint — weight storage is the same 93.8 Mbit either way), ~9 shared files to edit vs **60+** for full-INT4 (all 53 `node_conv` files), and better accuracy. **`requant_pipeline.v` is untouched under Scheme A** (operates on the 32-bit accumulator) → major Phase 2 de-risk.
- **CORRECTION to the Phase 3 premise: engine time-mux gives ZERO BRAM saving here.** With no DRAM and URAM-can't-init, the non-active engine weights have nowhere to be evicted to → all weights sit in BRAM regardless. The "resident peak = spatial + largest engine" model does NOT apply. The fit is driven by INT4 weights *alone*. (Time-mux may still be worth doing for the engine's hot-weight bandwidth, but not for fit.)
- **Runtime buffers → URAM zero-init.** FIFOs (~26.8) + engine activation (~12.2) + line buffers (~1.5) = ~40 Mbit are runtime data (no init needed) → map to URAM zero-init (~11% of URAM). They do NOT compete for the 99.1 Mbit init-capable BRAM. Only constants (weights + biases) do.
- **Accuracy expectation:** W4A8 PTQ ~74–75.5% literature; **but this quantizer is per-tensor scalar**, which penalizes INT4 more → realistic ~72–75% top-1. Per-channel weight scales would help but add requant complexity.
- **DSP will NOT auto-halve** at INT4. Packing 2 MACs/DSP (the 4×8 SIMD trick) is an explicit RTL rewrite; without it the design still maps ~1 DSP/lane. Don't expect a free DSP win in Phase 6.
- **Weight `.hex` packing** switches from 1 byte/weight to nibble-packing (2 weights/byte) in `repack_weights_wide.py` + `build_weight_memory_map.py`. Keep accumulators at 32-bit (both schemes) to avoid requant rework.

See per-phase **[FINDING]** callouts below for detail. Full record in memory: `project_int4_fit_analysis.md`.

---

## Global operating rules

- **Backup before each phase.** Snapshot `output/rtl/` (and `output/weights/`, `output/tb/` where touched) to `backups/<phase>_<timestamp>/`. Verify the backup exists before proceeding.
- **Git commit after each phase that passes its gate**, with a descriptive message. This is the revert anchor.
- **Byte-exact verification is the gate.** Every module that changes re-runs `equiv_one.ts`. The e2e value-match (Phase 0 harness) is the integration gate.
- **Build/run:** always `cd /c/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo &&` first. `taskkill //F //IM Vnn2rtl_top.exe 2>/dev/null` before each rebuild. Background long runs; read the redirected log.
- **Evidence before diagnosis.** On any unexpected result, add probes, get numbers, then hypothesize. Never pattern-match to a plausible cause and act.
- **Honest logging.** Log failures as failures in `project_e2e_sim_debug.md`. Never reframe a failure as a discovery.
- **Self-recover only within proven pattern families** (handshake skid, ready_out gate, spatial_run gate, weight repack). Anything outside that → hard stop and report.
- **No autonomous wakeups.**

## Global hard-stop conditions

- A per-module byte-exact test fails after a change and the cause isn't an obvious repack/width fix
- e2e value-match shows a mismatch you can't localize and fix within 2 iterations
- e2e produces <3136 beats or cycle count regresses unexplained
- An arithmetic/datapath change produces results you can't make byte-exact (especially INT4 phase)
- BRAM/URAM/LUT exceeds device capacity and can't be resolved by mapping
- Any change requires an architectural decision not derivable from local evidence (e.g. touching `spatial_run` correctness)

## Known facts to use (don't re-derive)

- **URAM CANNOT be bitstream-initialized to non-zero on this device** (xcu250/VU13P), proven 3 ways on Vivado 2025.2 (`Synth 8-10226`, `Synth 8-12183`, URAM288 primitive has no INIT params). The whole point of INT4 is to make weights fit in BRAM (which CAN be init'd) so URAM is not needed. **Do not attempt URAM `$readmemh` init** — it silently falls back to BRAM.
- **ImageNet val set is already downloaded locally** (see Phase 1.2). Do not try to download it — HuggingFace is not in the network allowlist and the dataset is gated.
- **BRAM bitstream-initializable capacity on U250 ≈ 99.1 Mbit** (2688 BRAM36).

---

## Phase 0 — Build the e2e value-verification harness (prerequisite for everything)

Build it against the current INT8 design first so you have a known-good baseline.

- **0.1.** Backup.
- **0.2.** Adapt `tb/static_verilator_tb.cpp`'s NN2V parser to drive the whole `nn2rtl_top`: feed `conv_196.goldin` (vector 0) to `s_axis` (`s_axis_tready = node_conv_196_ready_in`; conv_196 IC=3), capture `m_axis`, compare byte-exact to `node_relu_48.goldout` (vector 0).
- **0.3.** Validate the harness detects mismatches: run it once with a deliberately corrupted golden (flip a few bytes) and confirm it reports the mismatch at the right location. **A false MATCH is the dangerous failure** — prove the harness can fail before trusting a pass.
- **0.4.** Run on the current INT8 design. This establishes the INT8 e2e baseline. Handle the known conv_202/conv_216 ±1 stale samples: either regenerate those goldens or trace any ±1 to the documented stale source. Report exact mismatch count.

**Gate:** harness proven to detect mismatches (0.3) AND current INT8 e2e result recorded (0.4, byte-exact or with documented ±1 trace).

---

## Phase 1 — Regenerate the reference: ImageNet calibration + INT4, together

Do ImageNet recalibration and INT4 quantization in ONE regeneration pass to avoid re-verifying twice.

- **1.1.** Backup `output/weights/`, `output/tb/` (goldens/sidecars).
- **1.2.** **ImageNet data — ALREADY DOWNLOADED.** Location:

  ```
  C:\Users\User\Desktop\RTL_LLM_CLAUDE\imagenet-val\data\validation-*.parquet
  ```

  - 14 parquet files, full ILSVRC2012 validation split, 50,000 images.
  - Format: parquet, each row = image (JPEG bytes) + label (int32, 0-999) + label_name (string). **NOT raw JPEG folders** — decode the image bytes with pyarrow/datasets/PIL.
  - Sanity-check before use: read the parquet files, decode a few images, confirm total = 50,000 rows and labels are in 0-999. Report the count.
  - Do NOT attempt to re-download — it's gated and HuggingFace isn't reachable from the agent's network.

- **1.3.** **Label-order alignment check (CRITICAL — the #1 source of false accuracy failure):**
  Before any accuracy use, confirm the parquet integer label ordering matches torchvision ResNet-50's output class order. Verify on a few clearly-classifiable images (run them through torchvision pretrained ResNet-50 in float and confirm the predicted class index matches the parquet label). If a later accuracy run (Phase 4) comes out near 0.1% (chance level), suspect label misalignment FIRST, not a hardware/model bug. Document the confirmed mapping.
- **1.4.** Modify the reference frontend (`onnx_frontend.py`):
  - Replace the random calibration feed (line ~1416, `rng.integers(-128,128)`) with real preprocessed ImageNet calibration images: ~256-1024 images from the parquet set, preprocessed resize 256 → center-crop 224 → normalize (ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`).
  - Change the quantization target from INT8 to INT4 for weights (decide scheme — see 1.5).
  - Regenerate per-layer goldens (`.goldin`/`.goldout`) and sidecars from the recalibrated INT4 model.
- **1.5.** **Precision decision (report before committing):** assess INT4-weight/INT8-activation vs full-INT4. Report for each: the fit math (does resident set fit BRAM — see Phase 3), the datapath blast radius, and the published accuracy range. Recommend one. If the trade-off isn't clear-cut, **STOP and ask.** Default unless analysis says otherwise: **INT4 weights, INT8 activations** (best fit/accuracy balance, smaller datapath disruption).
  - **[FINDING 2026-05-28 — DONE, decided: Scheme A (INT4 w / INT8 act).]** Blast radius from reading the RTL: **Scheme A ≈ 9 shared files** (`mac_array.v`, `shared_engine_skeleton.v` params, `address_generator.v` packing-only, `conv_datapath_mp_k.v` / `conv_datapath_parallel.v` / `conv_datapath.v`, `repack_weights_wide.py`, `build_weight_memory_map.py`, golden gen) with **zero per-layer `node_conv` edits** and **`requant_pipeline.v` untouched** (32-bit accumulator, width-independent). **Scheme B (full INT4) ≈ 60+ files** — adds the activation path across all 53 `node_conv` files + `bram_to_stream_bridge` + requant output + `line_buf_window`/`coord_scheduler`. Full-INT4 buys **nothing** on the binding BRAM fit (weight storage identical), so it trades accuracy + 6× the files for no benefit. Accuracy: W4A8 ~74–75.5% lit, realistically ~72–75% here (per-tensor scalar quant). **Decision: Scheme A.** Keep accumulators at 32-bit; weights nibble-packed (2/byte).
- **1.6.** Regenerate all goldens with the chosen scheme. Verify the reference model runs end-to-end in software and produces sane outputs (not all-zero, not NaN), and that it actually classifies the calibration images sensibly (a quick top-1 spot-check on ~50 images — if it's at chance, the calibration or label mapping is wrong, fix before proceeding).

**Gate:** ImageNet calibration data loaded + label order confirmed (1.3), reference quantized to chosen INT4 scheme, all goldens regenerated, reference runs clean and classifies sensibly. STOP if precision decision unclear or label mapping can't be confirmed.

---

## Phase 2 — INT4 datapath rework (HIGHEST RISK — expect human-in-loop)

The RTL multiplier/accumulator/requant widths change for INT4. NOT mechanical — most likely phase to need intervention.

> **[FINDING 2026-05-28 — blast radius mapped, Scheme A is smaller than feared.]** Under Scheme A the change is confined to ~9 SHARED files (no per-layer `node_conv` edits — they only encode activation width, which stays INT8). **`requant_pipeline.v` needs no arithmetic change** (scale_mult operates on the 32-bit accumulator regardless of operand width); only its *output* byte width would change, and only under Scheme B. Keep `acc`/`acc_out` at 32 bits everywhere to avoid requant rework (products shrink 16→12b but 32b is safe). Weight memory/bus: engine `MAC_COUNT*WGT_W` 2048→1024 b/cyc; spatial `MP*MP_K*8`→`*4`. `$readmemh`/.hex packing switches to nibble-pack (2 weights/byte) in `repack_weights_wide.py` + `build_weight_memory_map.py`. DSP: a 4×8 multiply *can* pack 2/DSP but only via an explicit MAC-packing rewrite — not automatic.

- **2.1.** Backup.
- **2.2.** **Identify the blast radius:** list every module and shared file that encodes weight bit-width — `mac_array.v`, every spatial conv datapath (`conv_datapath_mp_k`), the engine compute, requant logic, weight packing in `repack_weights_wide.py`. Report the list before changing anything.
- **2.3.** **Change weight packing first:** update `repack_weights_wide.py` to pack INT4 weights. Regenerate all `_weights_*.hex` at INT4.
- **2.4.** **Datapath width changes:** update the MAC datapath for INT4 weights (INT4×INT8 multiply if mixed, or INT4×INT4 if full). Update accumulator widths and requant. Do this to the shared library modules first (one datapath, many users).
- **2.5.** **Verify per-module, incrementally:** re-run `equiv_one.ts` for each conv against its new INT4 golden, in dependency order. Each must be byte-exact (max_error=0, accounting for documented rounding tolerance) before moving on. If a module can't be made byte-exact, **STOP** — that's a real datapath bug, not a tolerance issue.
- **2.6.** **Engine convs:** verify via the engine sweep (`run_engine_sweep_all.sh`) against new INT4 goldens. 14/14 byte-exact.
- **2.7.** **e2e value-match** (Phase 0 harness, now against INT4 goldens). Must match byte-exact (or documented bounded rounding). Localize any divergence via intermediate goldens.

**Gate:** all 119 modules byte-exact at INT4 + engine sweep 14/14 + e2e value-match passes. Do not paper over a non-byte-exact module as "tolerance" without tracing it.

---

## Phase 3 — Engine weight time-multiplexing + fit analysis

Make the engine hold only the current dispatch's weights, and confirm the INT4 design fits BRAM.

> **Note from prior analysis:** at INT8 the binding constraint was 4 heavy SPATIAL convs ~73 Mbit that can't be time-muxed; at INT4 those halve to ~37 Mbit — this is exactly what should bring the resident set under BRAM. Verify with real numbers.

- **3.1.** Backup.
- **3.2.** **Fit math at INT4 (report first):**
  - Spatial weights (39 concurrent convs) at INT4 = total Mbit (must be resident)
  - Engine: total at INT4, and largest single engine layer at INT4
  - Resident peak = spatial + largest engine layer
  - Add FIFOs (right-sized — already done this session, ~728 BRAM36; confirm still applied), engine activation BRAM, biases, line buffers
  - Does the full resident set fit in 99.1 Mbit bitstream-initializable BRAM? Report the breakdown.
- **3.3.** **If it fits BRAM:** map all resident weights to BRAM with `$readmemh` init (bitstream-initializable — no URAM, no runtime loader, no URAM-init problem). **This is the goal — it dissolves the entire URAM problem.**
- **3.4.** **Engine weight time-multiplexing:** change the engine weight memory to hold only the current dispatch's weights, reloading between dispatches from the BRAM-resident weight store. Verify:
  - Engine sweep still 14/14 byte-exact
  - e2e value-match passes
  - Report cycle impact of per-dispatch weight reload (may add cycles — measure it)
- **3.5.** **If it does NOT fit BRAM even at INT4:** STOP and report the overage. Fall back options to present: full INT4 (if you did mixed), or document as a finding. **Do NOT silently fall back to a URAM design** — URAM can't be bitstream-init'd (proven), so a URAM weight design requires a runtime loader, which is an architectural decision for the user, not an autonomous choice.

> **[FINDING 2026-05-28 — fit math + hard BRAM36 count DONE.]**
> Weights at INT4 (decimal Mbit): spatial 53.4, engine 40.4 (largest single conv_286 = 4.19), total **93.8**.
> **Hard per-array BRAM36 count** (actual ROM geometry, optimal aspect packing): spatial **1528** + engine shared-store **1131** (1024b × 39 424) + biases **57** = **2716 BRAM36 = 101% of 2688 → over by 28.**
> **FITS with one cheap lever:** the overage is pure width-rounding waste in ~20 tiny shallow 1×1 ROMs (conv_198 = 8 BRAM36 for 0.4 of data = 94% waste; conv_202/204/206/210/212 ≈ 78% each). **Biases→LUTRAM alone → 98.9% → fits**; narrow-repacking the shallow 1×1s frees more. LUTRAM is also bitstream-init-capable, so it sidesteps both BRAM pressure and the URAM-init problem.
> **Phase 3.4 caveat — time-mux ≠ BRAM saving:** with no DRAM and URAM-can't-init, all weights are resident in BRAM regardless (nowhere to evict the non-active engine weights). So 3.4's time-mux does NOT reduce the BRAM fit — the fit is driven by INT4 weights alone (above). Do 3.4 only if it helps the engine's hot-weight bandwidth, not for fit. **Engine on-disk banks are URAM-oriented (288-bit lines) and must be re-laid-out for an efficient BRAM store** (part of this phase). Runtime buffers (FIFO/act/line-buf, ~40 Mbit) → URAM zero-init, don't compete for BRAM.

**Gate:** fit confirmed (ideally BRAM-only, bitstream-initializable), engine time-mux byte-exact + e2e passes, cycle impact measured.

---

## Phase 4 — Accuracy measurement (now meaningful, ImageNet-calibrated)

- **4.1.** Run the INT4 ImageNet-calibrated reference model on the  ImageNet validation set (50,000 images from the parquet files at the Phase 1.2 path) in software (PyTorch — fast, NOT Verilator). Compute top-1/top-5. ( i guess not the full thing it will take weeks , a normal amount of time i guess )
- **4.2.** Use the confirmed label mapping from 1.3. If top-1 comes out near 0.1% (chance), **STOP and re-check label alignment** before reporting — that's almost certainly a label-order bug, not a real result.
- **4.3.** Report: top-1/top-5 of the INT4 reference + the byte-exact chain ("RTL byte-exact to INT4 ImageNet-calibrated reference per-module + e2e → reference accuracy = hardware accuracy").
- **4.4.** Note the accuracy vs INT8 (the expected INT4 drop). This is now a measured result, not a cited literature value.

**Gate:** none — accuracy is a reported number. Flag if surprisingly low (label bug or quantization bug worth investigating before trusting it).

---

## Phase 5 — Cycle optimization (Levers 1 & 2)

On the INT4 design. Re-run e2e value-match after each.

- **5.1.** **Lever 1 — conv_196 MP=32.** One-off verify (feed goldin, compare to golden), confirm on current MP first, bump to MP=32, re-verify, e2e value-match. Revert if can't be made byte-exact. Target ~2M cycles.
- **5.2.** **Lever 2 — engine K-parallelism.** Backup engine. Baseline sweep. K_PAR=1 identity → K_PAR=2 (widen feed, more parallel banks) → sweep 14/14 → e2e value-match → K_PAR=4 if 2 passes. Revert on any sweep/e2e failure. Target ~2-3M cycles.

**Gate:** each lever byte-exact + e2e passes + cycles reduced, else revert that lever and continue.

---

## Phase 6 — Vivado timing run (final INT4 design)

- **6.1.** Backup + git commit ("pre-Vivado: INT4, ImageNet, engine-time-mux, optimized").
- **6.2.** **Weight mapping:**
  - **If Phase 3 confirmed BRAM fit (the goal):** map weights to BRAM with `$readmemh` init. This is the clean all-on-chip, bitstream-initializable case — no URAM, no URAM-init problem, no runtime loader. **This is the expected path at INT4.**
  - **If Phase 3 needed URAM (overage even at INT4):** do NOT use `$readmemh` on URAM (it silently falls back to BRAM and overflows — proven). For timing measurement only, map the over-budget weights to URAM as **zero-initialized** (no init file), which maps cleanly to URAM and gives valid compute-path timing (timing is weight-value-independent). Document explicitly that deployment requires a runtime weight loader since URAM can't be bitstream-init'd. **This is a measurement workaround, not a deployable config — flag it.**
- **6.3.** `synth_design` → report LUT/FF/DSP/BRAM/URAM.
- **6.4.** Place & route → routed Fmax (slow corner), final utilization, failing paths.
- **6.5.** If whole-design P&R won't route → per-module/OOC synth for engine + representative conv → per-module Fmax.

**Gate:** report synth results even if P&R fails — over-utilization is itself a result.

---

## Phase 7 — Final report

- **Precision:** INT4 scheme used (weight/activation widths)
- **Verification:** ImageNet-calibrated goldens, label-order confirmed, per-module byte-exact, e2e value-match confirmed
- **Accuracy:** measured top-1/top-5 on ImageNet (Phase 4), vs INT8 baseline
- **Fit:** BRAM-only all-on-chip (if achieved — the goal), or the precise overage + resolution
- **Cycle progression:** INT8 baseline (13,348,787) → INT4 → +engine-time-mux → +Lever 1 → +Lever 2
- **Real fps:** at measured Fmax, projections at 150/200/300 MHz
- **Resources:** Vivado-measured LUT/FF/DSP/BRAM/URAM
- **Engine time-mux finding:** weight savings + cycle cost trade-off
- **URAM finding:** bitstream-init impossible (proven), and how INT4 sidesteps it by fitting BRAM
- **Backups list** (every phase revertible)
- **Honest assessment:** what worked, what didn't, what surprised you

---

## Phase ordering summary

1. **Build e2e value harness** (on INT8 baseline) — prerequisite
2. **Regenerate goldens:** ImageNet calibration (data at `C:\Users\User\Desktop\RTL_LLM_CLAUDE\imagenet-val\data\`) + INT4 together — one pass, confirm label order
3. **INT4 datapath rework** — highest risk, expect human-in-loop
4. **Engine time-mux + fit analysis** — the payoff (BRAM fit at INT4?)
5. **Accuracy on ImageNet** — now meaningful, use confirmed label mapping
6. **Cycle optimization** (Levers 1, 2)
7. **Vivado timing** — BRAM `$readmemh` if it fits (expected at INT4); URAM zero-init only as measurement workaround
8. **Final report**
