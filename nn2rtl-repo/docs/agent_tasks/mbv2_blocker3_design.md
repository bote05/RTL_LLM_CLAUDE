# MobileNet-V2 BLOCKER #3 — Byte-Exact Implementation Plan (all-spatial top)

**Target file:** `output/mobilenet-v2/rtl/nn2rtl_top.v` (the ALL-SPATIAL top, NOT `nn2rtl_top_engine.v`)
**Status of this doc:** read-only investigation complete; this is the implementation plan. NO RTL/sim run this phase.
**Date:** 2026-06-02

---

## 0. TL;DR

Blocker #3 (tiled<->packed contract mismatch) was *architecturally* solved on the all-spatial top by the
**wave-2 retile bridge mesh** (`rtl_library/retile_bridge.v` + `scripts/apply_mbv2_wave2_bridges.py`, 23 bridges
already wired into the live top). **retile_bridge.v IS the reusable byte-exact tiled<->packed adapter** the task
asked about — Approach **C** (insert retile bridges) is the correct, cleanest, least-invasive, byte-exact-preservable
choice, and it is already chosen and built.

**But there is a live, proven BUG that keeps the e2e deadlocked:** a per-bridge **PARAMETER mismatch**. Four of the
six final-stage depthwise convs were re-architected to **single-beat full-width** input/output, but their gather/scatter
bridges are still parameterized for the OLD **2-beat 4096b** depthwise contract. This is both a **width truncation**
(channels >512 silently read as zero via implicit zero-extend) and a **beat-cadence desync** (2 beats fed into a 1-beat
consumer). It is the highest-confidence cause of the e2e stall (`result.json` = TIMEOUT, full input consumed, 0 output).

**The fix is an 8-parameter edit in `apply_mbv2_wave2_bridges.py` + idempotent re-apply.** No RTL module edits, no
golden/contract regen, no `build_top_wrapper` regen. retile_bridge.v already supports the corrected params generically
(`br_mean` already runs OUT_W=10240/OUT_BEATS=1).

---

## 1. RECONCILIATION APPROACH — **C (retile bridges)**

| Approach | Verdict | Why |
|---|---|---|
| **A** — make consumers accept channel_tiled | ❌ rejected | Deep atomic-arch change: would re-contract 4 adds + GAP + Gemm + 6 depthwise to multi-beat tiled, moving RTL + `compute_conv2d_latency_cycles` + patterns + per-module goldens + sidecars TOGETHER (per `feedback_atomic_arch_changes`). Maximally invasive; destroys the 98/99 byte-exact module verdicts. |
| **B** — make producers emit packed_full | ❌ impossible | 576/960/1280 channels = 4608/7680/10240 bits > the **4096-bit flat-bus cap**. The tiled contract exists *because of* this cap. B violates the cap. |
| **C** — insert tiled<->packed retile bridges | ✅ **CHOSEN + ALREADY BUILT** | Only option that respects the 4096b bus cap. Localized to inserted glue (`retile_bridge.v`), leaves every per-module contract / golden / sidecar / latency formula **byte-exact-INVARIANT**. Surgical top-wrapper wiring (consistent with `project_top_v_is_patched_not_regenerated`). |

**Why retile_bridge.v is byte-exact (verified):** it is a pure PING-PONG gather/scatter with NO channel reordering.
The load-bearing invariant (header lines 96–115) is that both sides pack channels contiguously INT8, LSB=lowest channel,
so `full[k*256 +: 256] == tiled_beat_k`. Verified against the actual consumers: `node_add_828.v` reads `data_in[c*8 +: 8]`
over c=0..95; `n4_24.v` is a 256b tiled consumer (`data_in [255:0]`). Layout matches with no permutation. Handshake is the
drain==latch invariant (drain xfer and consumer latch share the identical per-bridge `spatial_run_drain_<i>` gate);
always-accept intake (`do_write = valid_in & wsel_empty`) accommodates the free-running MobileNet producers.

---

## 2. THE LIVE BUG (root-caused, verified on-disk)

### 2.1 Evidence

Verified **actual module port widths** (`output/mobilenet-v2/rtl/node_conv_*.v`):

| Module | `data_in` / `data_out` width | Beats/pixel | C |
|---|---|---|---|
| `node_conv_878` | **[4607:0] = 4608b** | **SINGLE** (feeds `line_buf_window` directly, no assembler) | 576 |
| `node_conv_884` | [4095:0] = 4096b | **2-beat** (has explicit `lo_latch`/HI assembler, header: "TWO 4096b beats per pixel") | 576 |
| `node_conv_890` | **[4607:0] = 4608b** | **SINGLE** (header: "single-beat, == conv_818") | 576 |
| `node_conv_896` | **[7679:0] = 7680b** | **SINGLE** (header: "SINGLE beat per pixel") | 960 |
| `node_conv_902` | **[7679:0] = 7680b** | **SINGLE** | 960 |
| `node_conv_908` | [4095:0] = 4096b | **2-beat** (has `lo_hold` assembler, header: "2 beats/pixel ... INPUT ASSEMBLER reconstructs 7680b") | 960 |

But **every** gather/scatter bridge for these edges is parameterized `OUT_W=4096/OUT_BEATS=2` (gather) and
`IN_W=4096/IN_BEATS=2` (scatter), verified at `nn2rtl_top.v:672,684,696,708,720,732` (gather) and `:744,756,768,780,792,804` (scatter).

### 2.2 Why it breaks (two coupled failures)

1. **Width truncation:** `br_878.data_out` is 4096b but `node_conv_878.data_in` is 4608b. Connecting a 4096b net to a
   4608b port zero-extends → channels 512..575 read as **zeros** → corrupt pixel.
2. **Cadence desync:** the gather emits 2 beats/pixel (`OUT_BEATS=2`); `node_conv_878` latches ONE `valid_in` beat as a
   whole pixel and advances → beat1 arrives as a spurious second "pixel"; meanwhile the output scatter `br_n4_24`
   (`IN_BEATS=2`) waits for a 2nd output beat the single-beat conv never emits → **terminal stall**. Matches the symptom
   exactly: input fully consumed (`in_beats_seen=50176/50176`), `tvalid=0` forever, `result=TIMEOUT`.

### 2.3 The exact 8 wrong params (and the 4 correct ones to LEAVE alone)

`FULL_W = N_TILES * TILE_W` is already correct on every bridge (18*256=4608, 30*256=7680). **Only `OUT_W/OUT_BEATS`
(gather) and `IN_W/IN_BEATS` (scatter) are wrong** on the 4 single-beat depthwise edges:

| Bridge (script `BRIDGES` entry) | kind | conv | CURRENT (wrong) | CORRECT |
|---|---|---|---|---|
| `br_878` | gather→878 | 4608b single | `full_beat_w=4096, full_beats=2` | `full_beat_w=4608, full_beats=1` |
| `br_n4_24` | scatter 878→relu | 4608b single | `full_beat_w=4096, full_beats=2` | `full_beat_w=4608, full_beats=1` |
| `br_890` | gather→890 | 4608b single | `full_beat_w=4096, full_beats=2` | `full_beat_w=4608, full_beats=1` |
| `br_n4_28` | scatter 890→relu | 4608b single | `full_beat_w=4096, full_beats=2` | `full_beat_w=4608, full_beats=1` |
| `br_896` | gather→896 | 7680b single | `full_beat_w=4096, full_beats=2` | `full_beat_w=7680, full_beats=1` |
| `br_n4_30` | scatter 896→relu | 7680b single | `full_beat_w=4096, full_beats=2` | `full_beat_w=7680, full_beats=1` |
| `br_902` | gather→902 | 7680b single | `full_beat_w=4096, full_beats=2` | `full_beat_w=7680, full_beats=1` |
| `br_n4_32` | scatter 902→relu | 7680b single | `full_beat_w=4096, full_beats=2` | `full_beat_w=7680, full_beats=1` |

**LEAVE UNCHANGED (genuinely 2-beat 4096b):** `br_884`, `br_n4_26` (conv_884), `br_908`, `br_n4_34` (conv_908). Their
2-beat assemblers are byte-exact at module level (mismatch=0) and the 2-beat partial-hi-channel layout matches the bridge
emit_chunk zero-extend (verified: 884 FULL_W=4608 → beat1 carries ch512..575 in low 512b; 908 FULL_W=7680 → beat1 carries
ch512..959 in low 3584b).

**No `retile_bridge.v` RTL edit is needed:** the gather `emit_chunk` loop and scatter `wbuf_next` loop are fully
parameterized in `OUT_W/IN_W` and `OUT_BEATS/IN_BEATS`. `br_mean` already runs at `OUT_W=10240, OUT_BEATS=1` in the live
top, proving single-beat wide output works. SPATIAL params are documentation-only and already correct (878=196, 896=49).

---

## 3. WHY the other lens's "skip-FIFO write gate" theory does NOT apply here

A competing analysis localized the deadlock to the residual skip-FIFO write gates (`u_skip_node_add_828/900/1110`
writing with bare `& spatial_run` on a single-cycle producer pulse). That analysis describes an **older top revision**
predating the wave-2 bridges. On the **current** top (re-patched 2026-06-02 01:36):
- The add main operands are now fed by gather bridges with per-bridge `spatial_run_drain_*` gates (drain==latch invariant).
- `apply_mbv2_wave2_bridges.py` step 6 (`UNGATE_FINAL_BRIDGELESS`) already STRIPPED the bare `& spatial_run` from the
  bridgeless free-running hops, and the add skip FIFOs are fed/gated through the per-bridge drain path.
- The depthwise param mismatch (Section 2) is a hard width+cadence break that stalls regardless of the skip gates, and is
  present and verifiable in the current file. It is the correct first fix. (If a residual skip-gate hazard re-surfaces
  after the param fix, it is caught by the same e2e/probe gate in Section 5.)

---

## 4. ORDERED ATOMIC CHANGE SET (surgical, no blind regen)

**Modules changed: 0.** **Bridges added: 0** (23 already present; 8 params corrected on existing bridges).

1. **Edit `scripts/apply_mbv2_wave2_bridges.py` `BRIDGES` list (8 dict fields):** set `full_beat_w/full_beats` to the
   CORRECT column of the Section 2.3 table for `br_878, br_890, br_896, br_902` (gather) and `br_n4_24, br_n4_28,
   br_n4_30, br_n4_32` (scatter). This is the ONLY source edit.
2. **Re-apply the patch (idempotent):** `/c/Python313/python scripts/apply_mbv2_wave2_bridges.py`. It detects the existing
   `// ===== WAVE-2 RETILE BRIDGES =====` block and REPLACES it in place (regex on `MARK_BEGIN..MARK_END`), preserving all
   other top patches. Do NOT run `build_top_wrapper` (would destroy ~1205 handshake patches per
   `project_top_v_is_patched_not_regenerated`).
3. **(Optional, recommended for uniformity, separate sub-task):** make `node_conv_884` and `node_conv_908` single-beat
   full-width too (they are the lone 2-beat outliers; 908 is also flagged historically wrong in
   `project_mobilenet_u250_status`). This would let ALL six depthwise bridges share one single-beat param shape. **This IS
   an atomic-arch change** (module RTL + its 2-beat external contract + sidecar + golden retile + latency formula) and
   should be staged separately, NOT bundled into the minimal unblock. The minimal fix (steps 1–2) keeps 884/908 at 4096/2.
4. **`compute_conv2d_latency_cycles` / patterns:** **NO change.** The fix is a top-wrapper bridge-param edit; per-module
   bus contracts, beats-per-sample, and latency are unchanged. (Only step 3, if taken, would touch the latency formula for
   884/908.)
5. **EXPECTED_BEATS fix:** **N/A for the all-spatial top.** `EXPECTED_BEATS` belongs to the engine-output-bridge, which
   does not exist on the all-spatial datapath (engine stubbed, `engine_busy=0`). The roadmap's claim that
   `conv_880`(196) vs `conv_882`(588) is "inconsistent" is FALSE — they are different-shape tensors ([96,14,14] vs
   [576,14,14]); 196 and 588 are both correct engine-beat counts (`oh*ow*ceil(oc/256)`). Do NOT "fix" EXPECTED_BEATS;
   doing so would INTRODUCE a bug. This is purely an engine-top concern, deferred to the transfer (Section 7).
6. **Regen chain:** **NONE required for this fix.** All per-module goldens, sidecars, `layer_ir` contract fields, and
   weight/bias/scale maps are byte-exact-INVARIANT under a bridge-param wiring change. The standing chain from
   `feedback_regen_must_rebuild_engine_maps` (generate_golden → build_bias/scale/spatial maps → single-file repack →
   engine banks → refresh_final_golden → rebuild_contract_goldens) applies ONLY if a weight/quant change is independently
   triggered, which this is not. Do NOT run it for the #3 fix; conflating the two would also destroy the top patches.

---

## 5. VERIFICATION GATES (byte-exact)

**Tier 1 — elaboration (seconds–minutes, no sim):** `verilator --lint-only --top-module nn2rtl_top` over the full source
set incl. `rtl_library/retile_bridge.v`. After the param fix, the `br_878/890/896/902` `data_out` widths (4608/7680) must
match the `node_conv_878/890/896/902` `data_in` ports exactly → expect ZERO width/SELRANGE warnings. (Before the fix the
4096↔4608 connection silently zero-extends — lint may or may not warn; the fix makes them bit-equal.)

**Tier 1b — per-module byte-exact (already GREEN; re-confirm if any module touched):** `_verify_mbv2_variant.ts <mod.v>
<mod> <sidecar.json>`. Current state verified this phase: `node_conv_878/884/890/896/902/908`, `n4_24/28/30/32`,
`node_mean` all **mismatch=0**; `node_linear` = 2/8000 (documented float32-acc golden artifact, RTL more correct). The
bridges themselves have no standalone golden — they are exercised only at Tier 2/3. (Module-level byte-exactness is
necessary, NOT sufficient: it cannot catch the inter-module bridge cadence bug — that is what the param fix + e2e covers.)

**Tier 2 — focused localization (recommended ONLY if Tier 3 still stalls):** a probe-instrumented run that links the
already-built `obj_dir_value/Vnn2rtl_top__ALL.a` (NO re-verilate) and taps each final-stage bridge `{full0,full1,stall_out,
valid_out}` + each terminal-hop `{valid_in,ready_in,valid_out}` every ~1M cyc. The first bridge/module that freezes with a
non-advancing counter localizes any residual stall. ~35 min/run, no rebuild.

**Tier 3 — e2e byte-exact (the gate of record):** `npx tsx scripts/run_mbv2_top_value.ts 0` (vec0). The harness already
lists `retile_bridge.v` (line 226) and compares `m_axis` to the LOGICAL `node_linear.goldout` (8×1000B). PASS criteria:
`result=PASS`, `out_beats=1/1`, `mismatch_bytes=0`. Then `MBV2_ALL_VECS=1` for all 8 vectors. **Only this Tier closes
byte-exactness end-to-end** — the design currently produces NO output, so byte-exactness is UNPROVEN until a clean run
completes. Use Verilator `--x-initial 0` (hardware-faithful).

**Backup / revert plan:** `apply_mbv2_wave2_bridges.py` is idempotent and self-replacing, so revert = restore the prior
`BRIDGES` params and re-run, OR `git stash`/`git checkout` the top + script (top is untracked-patched; copy
`nn2rtl_top.v` to `backups/mbv2_blocker3_<date>/` before the edit). A byte-exact baseline of the WHOLE design lives at
`backups/allint4_byteexact/` (ResNet) — for mbv2, snapshot the top + script before editing since the top is not in git.

---

## 6. RISK + EFFORT

- **Riskiest step:** **Tier 3 e2e re-run after the param fix** — there may be a SECOND latent issue (the 2026-06-01
  per-bridge `drain_en` rewrite was never e2e-verified; `conv_908` is independently flagged max_error-20 in memory,
  though it now reports module-level mismatch=0). If the param fix still stalls, the distributed handshake invariant across
  23 ping-pong bridges sharing one global `any_retile_stall` is the next suspect (whack-a-mole risk; the bug class already
  bit twice). The param fix is *necessary*; sufficiency is unproven until the e2e passes.
- **Achievable in one pass?** **The fix edit + re-apply: YES** (8 params, idempotent). **A guaranteed-passing byte-exact
  e2e: NO** — it needs the localize→fix→re-test loop (param fix, then re-sim; if residual stall, Tier-2 probe → surgical
  hop/ready_down fix → re-sim). Realistically 1 fix-and-verify pass if the param mismatch is the sole bug; staged
  otherwise.
- **Effort:** param edit ~minutes; re-apply ~seconds; Tier-1 lint ~1–2 min; Tier-3 build+run ~35+35 min/vector
  (8 vectors ≈ overnight serialized). Total to first e2e verdict ≈ 1.5 h; to full confidence 1–2 supervised days if a
  residual stall needs the probe loop.

---

## 7. TRANSFER TO THE ENGINE TOP

The fix is **shared/transferable**. `nn2rtl_top_engine.v` currently has **0 retile bridges** (verified). The same
`apply_mbv2_wave2_bridges.py` retargets it via `--top` / `NN2RTL_TOP` (`resolve_top`, lines 631–639), and `retile_bridge.v`
is identical. **BUT the transfer is gated:** (1) BLOCKER #2 (engine output backpressure / engine_busy drain interaction)
must land first, and the engine must be re-verified byte-exact (mbv2 34/34, ResNet 14/14) before #3 bridges are reachable
there; (2) on the engine top the engine-output-bridge `EXPECTED_BEATS`/pad-tile suppression (96-of-256 real channels) must
be re-validated — it is correct per-tensor today and needs NO change, but the pad-tile vs wave-2 consumer geometry must be
re-checked once bridges are wired. The all-spatial top deliberately isolates #3 from #2, which is why it is the correct
target for THIS phase.

---

## 8. DECISIONS NEEDING THE USER

- **None blocking the minimal fix** (Section 4 steps 1–2): it is a surgical, idempotent, byte-exact-preservable bridge-param
  edit on the patched top, fully within the sanctioned path.
- **One judgment call (optional, staged):** whether to ALSO unify `conv_884`/`conv_908` to single-beat (Section 4 step 3).
  That is an atomic-arch change (RTL + contract + golden + latency) and should be a separate user-approved sub-task — NOT
  bundled into the unblock. Recommendation: do the minimal param fix first, prove e2e byte-exact, then decide on unification
  for cleanliness.
- **No Vivado / long-sim** is requested by this fix; Tier-3 e2e is a Verilator value sim (not gated by the Vivado rule).
