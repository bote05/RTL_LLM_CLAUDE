# PLAN D — Overlap SPATIAL (depthwise 3x3) with ENGINE (1x1 pointwise) in the MobileNetV2 top scheduler

READ-ONLY analysis. No .v/.mem/.json/.py modified. All facts cite live code at
`output/mobilenet-v2/rtl/nn2rtl_top_engine.v` (4097 lines) and
`output/mobilenet-v2/rtl/nn2rtl_scheduler.v` (1153 lines), plus the bridge bodies
emitted by `scripts/build_top_wrapper.ts`.

================================================================================
## TL;DR / RISK VERDICT (read this first)

**The overlap is ALREADY APPLIED, ACTIVE, and BYTE-EXACT in the committed design.**
It is not a future change. The memory note "engine-top e2e still TIMEOUTs" is STALE.

Evidence:
- `output/mobilenet-v2/reports/e2e_mp8_allvecs.log`: all 8 vectors
  `result=PASS mismatch_bytes=0`, `e2e_cycles=7742779`.
- `output/mobilenet-v2/reports/e2e_878_native_vec0.log`: `result=PASS
  mismatch_bytes=0`, `e2e_cycles=7744752` (post native-tiled re-arch, current HEAD).
- HEAD commit lineage `e76217e` = "clean 8/8 e2e", and the overlap markers
  (`[OVERLAP-FIX]`, `[DRAIN-FIX2]`, `[DRAIN-CONCURRENCY]`) are present in the
  committed top.

The three mechanisms the prompt asks to "add" are all present and live:
1. `spatial_throttle` (top:450) already DROPPED `engine_busy` — only
   `sched_spatial_stall | any_retile_stall` gates spatial now.
2. `S_WAIT_DONE` (scheduler:1130) already sets `spatial_stall = 1'b0`.
3. `engine_drain_run = engine_busy | ~sched_spatial_stall` (top:457) lets the
   engine-output FIFO drain concurrently with engine compute.

**So the question is not "how do I add overlap" — it is "is the EXISTING overlap
the maximal safe overlap, and is there a remaining serialization that is still
safely removable?" The answer below: the existing overlap already removes the
biggest serialization safely. The REMAINING serialization (dispatch N+1's input
loader cannot fill while dispatch N's engine computes) is the residual ~46%, and
removing it is the RISKY part. Verdict: NOT safely doable in one shot as a pure
top edit; it requires per-dispatch activation-bank double-buffering that the
current 2-bank ping-pong scheme cannot express. Safest incremental win is in §6.**

================================================================================
## 1. EXACT CURRENT SERIALIZATION MECHANISM (file:line)

### 1a. The two coupling signals (top)
- `output/mobilenet-v2/rtl/nn2rtl_top_engine.v:450`
  ```
  (* max_fanout = 32 *) wire spatial_throttle = sched_spatial_stall | any_retile_stall;
  // [OVERLAP-FIX] dropped engine_busy
  ```
  `spatial_run = ~spatial_throttle` (top:451) broadcasts to ~260 ready/valid
  gates on the spatial chain. Historically this term was `... | engine_busy`,
  which forced `spatial_run=0` for the WHOLE engine compute window = the original
  XOR serialization. That XOR is GONE.

- `output/mobilenet-v2/rtl/nn2rtl_top_engine.v:457`
  ```
  wire engine_drain_run = engine_busy | ~sched_spatial_stall;  // [DRAIN-FIX2]
  ```
  Used only by the SLOT-0 output bridge (top:2666) whose 12544-beat output is
  larger than the 4096 engine_output_fifo, so it must drain DURING engine_busy.

### 1b. The scheduler FSM (the real arbiter)
`nn2rtl_scheduler.v` drives one output, `spatial_stall`, that the top turns into
`spatial_run`. The FSM is a strict per-dispatch sequencer (scheduler:1005-1037,
output map scheduler:1080-1147). Per dispatch it walks:

```
S_WRITE / S_WRITE_RESP / S_NEXT_STEP   (13 AXI config writes)  spatial_stall=1
S_WAIT_LOAD   (wait current_loaded)                            spatial_stall=0
S_PULSE_START (engine_start=1)                                 spatial_stall=1
S_WAIT_DONE   (wait engine_done)                               spatial_stall=0  <-- OVERLAP WINDOW
S_WAIT_DRAIN  (wait current_drain_complete)                    spatial_stall=0
S_NEXT_DISP   (engine_output_ready=1, dispatch_idx++)          spatial_stall=1
```

The serialization that REMAINS is structural, not a single XOR:

1. **One engine, 34 dispatches, in program order.** `dispatch_idx` is a single
   counter (scheduler:68) advanced only in `S_NEXT_DISP` (scheduler:1062). The
   engine processes exactly one 1x1 conv at a time; `engine_busy = (state !=
   ST_IDLE)` in `shared_engine_skeleton.v:319` spans the whole MAC+requant+drain.
   This is inherent to a SHARED engine and is NOT what Plan D targets.

2. **Dispatch N+1's input loader cannot start until N's output exists.** The
   per-dispatch input loaders (`stream_to_act_bram_bridge`, top:1726+) are gated
   by `& spatial_run` AND fed by the spatial chain, whose data for dispatch N+1
   is produced by N's output (a relu/add chain hanging off N's engine output
   bridge). Because the engine writes N's output to act-BRAM and the FIFO only
   drains while N is the `active_slot` (bridge `active_slot = (dispatch_count ==
   SLOT)`, build_top_wrapper.ts:1931), N+1's depthwise/relu/add cannot run to
   completion and fill N+1's loader until N has fully drained. The `S_WAIT_DRAIN`
   state (scheduler:1132) is exactly this serialization point: the FSM holds
   before `S_NEXT_DISP` until `current_drain_complete`.

3. **`current_loaded` is a single muxed scalar** (top:2580):
   `current_loaded = all_loaded[sched_dispatch_idx]`. `S_WAIT_LOAD`
   (scheduler:1023) blocks `engine_start` until the CURRENT dispatch's loader is
   full. There is no "prefetch dispatch N+1's loader while N computes" — the loop
   is load(N) -> compute(N) -> drain(N) -> load(N+1) -> ...

So: the engine compute window IS overlapped with spatial drain (the win already
banked, ~ the 46% the prompt cites was the pre-overlap XOR). The residual serial
cost is the **load <-> compute <-> drain dependency chain per dispatch**, which is
a data dependency, not an artificial mutual-exclusion. Breaking it needs
pipelining ACROSS dispatches (prefetch N+1 inputs during N compute), which needs
more activation banks than the current 2-bank ping-pong provides.

================================================================================
## 2. STATE OF THE PARTIAL-OVERLAP REMNANTS (verify vs live code)

| Mechanism | Location | Status |
|---|---|---|
| `spatial_throttle` drops `engine_busy` | top:450 | ACTIVE |
| `S_WAIT_DONE: spatial_stall=0` | scheduler:1130 | ACTIVE |
| `S_WAIT_DRAIN: spatial_stall=0` | scheduler:1135 | ACTIVE |
| `S_WAIT_LOAD: spatial_stall=0` | scheduler:1119 | ACTIVE |
| `engine_drain_run = engine_busy \| ~stall` | top:457 | ACTIVE (SLOT-0 only) |
| per-bridge `spatial_run_drain_br_*` | top:480-486 | ACTIVE (7 retile bridges) |
| act-BRAM arbiter: engine priority > loaders | top:2451-2489 | ACTIVE |
| engine-output backpressure (ENABLE_OUTPUT_BACKPRESSURE=1) | top:2598, eofifo:2631 | ACTIVE |

Conclusion: the prior "spatial_throttle drops engine_busy + S_WAIT_DONE
spatial_stall=0" attempt is NOT reverted and NOT partial — it is the live,
byte-exact, 8/8-passing design.

================================================================================
## 3. RESIDUAL-ADD JOINS — full enumeration + de-sync analysis

10 residual adds. Each pairs a MAIN arm (lhs) with a SKIP arm (rhs, via a
`skip_fifo`). The hazard class (the conv_202-style de-sync the memory warns about)
is: if the two arms have different latency and only ONE arm is buffered, a cadence
change on the faster arm de-syncs the pair -> deadlock/corruption.

| Add | top line | LHS (main) source | LHS buffered? | RHS skip src | RHS FIFO |
|---|---|---|---|---|---|
| node_add_198 | 750 | conv_826 (engine) | YES `u_lhs_node_add_198` (192b/4096) top:1410 | conv_820 | top:1393 |
| node_add_336 | 829 | conv_838 (engine, direct) | NO (direct `node_conv_838_data_out`) | add_336 chain | top:1421 |
| node_add_408 | 874 | conv_844 (engine, direct) | NO | add_336 | top:1432 |
| node_add_546 | 953 | conv_856 (engine, direct) | NO | conv_850 | top:1443 |
| node_add_618 | 998 | conv_862 (engine, direct) | NO | add_546 | top:1454 |
| node_add_690 | 1043 | conv_868 (engine, direct) | NO | add_618 | top:1465 |
| node_add_828 | 1127 | conv_880 (engine) | YES `u_lhs_node_add_828` (768b/256) top:1492 | conv_874 | top:1476 |
| node_add_900 | 1177 | conv_886 (engine) | YES `u_lhs_node_add_900` (768b/256) top:1520 | add_828 | top:1504 |
| node_add_1038 | 1264 | conv_898 (engine) | YES `u_lhs_node_add_1038` (1280b/64) top:1548 | conv_892 | top:1531 |
| node_add_1110 | 1304 | conv_904 (engine) | YES `u_lhs_node_add_1110` (1280b/64) top:1576 | add_1038 | top:1560 |

### 3a. Why the existing overlap does NOT de-sync these
Every add uses ENABLE_BACKPRESSURE(1) and a `valid_in = lhs_valid & skip_valid &
spatial_run` gate (e.g. top:752, 831, 1129) — i.e. it is **rate-independent**: the
add only fires when BOTH operands are present, and pops both in lock-step. The
skip arm always has a `skip_fifo` (depth = full residual frame: 4096/1024/256/64
words, right-sized at top:1393 etc.). So the skip arm can absorb arbitrary
main-arm latency. This is exactly the elastic property that makes the EXISTING
overlap byte-exact: overlap changes WHEN the main arm produces, not the pairing.

### 3b. The 5 adds with NO explicit LHS skid (336/408/546/618/690)
These pair an engine-direct LHS (`node_conv_8XX_data_out`, no buffer) with a
skip FIFO RHS. They are still safe under the CURRENT overlap because:
- The LHS producer is the engine output bridge, which is itself elastic
  (`active_slot`-gated, valid/ready) and its `ready_out` is gated by the add's
  readiness AND the skip's validity (e.g. SLOT-8 conv_838 ready_out =
  `node_add_336_ready_in & node_add_336_skip_valid & spatial_run`, top:2826).
  So the engine bridge will not over-produce the LHS past the skip arm — the skip
  FIFO's presence on the consumer's ready path back-pressures the LHS bridge.

**De-sync RISK if Plan D goes further (cross-dispatch prefetch):** these 5 are the
weakest links. If a future change lets dispatch N+1's engine bridge produce its
LHS while dispatch N is still draining (true cross-dispatch overlap), the
engine-direct LHS of an add could arrive before its skip FIFO has been filled by
the older spatial path — the add would then pair (lhs[N+1], skip[N]) = CORRUPTION,
not deadlock (the skip FIFO is non-empty so valid_in asserts with the WRONG skip
beat). **Guard required:** add a symmetric `skip_fifo` LHS skid to all 5
(mirroring the 5 that already have `u_lhs_*`), so both arms are FIFO-ordered and
the add pairs by frame, not by cycle. Cost: ~5 BRAM FIFOs (256-1024 deep).

================================================================================
## 4. THE CONCRETE FURTHER-OVERLAP DESIGN (what it would take to remove §1b.2/3)

To convert the load->compute->drain serial loop into a software-pipeline
(prefetch N+1 inputs during N compute), the following must ALL hold. Each is an
EXACT edit; together they are NOT a one-shot byte-exact change (see §7).

### 4a. Scheduler: decouple load(N+1) from drain(N)
Add a parallel "loader pointer" `ld_idx` distinct from `dispatch_idx`, advanced
as soon as a dispatch's input bank is FREE (not when the prior dispatch drains).
- New reg in `nn2rtl_scheduler.v` near scheduler:68:
  `reg [5:0] ld_idx;` exported alongside `dispatch_idx_out`.
- New state or parallel handshake so the top's loaders are selected by `ld_idx`
  while the engine config/compute is selected by `dispatch_idx`.
- `S_WAIT_LOAD` (scheduler:1023) must wait on `all_loaded[dispatch_idx]` (compute
  pointer) while the loaders run ahead on `ld_idx`.
This is a structural FSM rewrite, NOT a few-line edit.

### 4b. Activation-bank triple-buffer (the blocker)
Current scheme is a 2-bank ping-pong: `input_bank_rom`/`output_bank_rom`
(scheduler:771-851) alternate bank 0/1 per dispatch; `act_unified_mem` is one
flat 25600x2048 URAM (top:2492) partitioned by `act_in_base_word_rom` /
`act_out_base_word_rom` (scheduler:894-974). With only 2 banks you cannot hold
{N reading, N writing, N+1 prefetch-filling} simultaneously — N+1's prefetch
target collides with N's output bank. **You need a 3rd bank window** (or per-
dispatch disjoint base addresses), which changes the base-word ROMs and the
act-mem depth/address map = a regen-pipeline change (build_scheduler.py +
build_top_wrapper.ts + weight/act memory maps), NOT a top patch.

### 4c. The 5 missing LHS skids (§3b) become MANDATORY
As argued in §3b, cross-dispatch overlap can pair the wrong skip beat at adds
336/408/546/618/690. Add `skip_fifo #(.WIDTH(<lhs_w>), .DEPTH(<frame>))` LHS skids
mirroring top:1410/1492/1520/1548/1576, and rewire each add's `valid_in`/`data_in`
to the buffered lhs (exactly as node_add_198 was done at top:752-754).

### 4d. Output-bridge `active_slot` must allow N and N+1 concurrently
`active_slot = (dispatch_count == SLOT)` (build_top_wrapper.ts:1931) means only
ONE bridge drains at a time. Cross-dispatch overlap needs N's bridge to keep
draining while N+1's bridge starts. Requires a 2-entry active window
(`dispatch_count-1` OR `dispatch_count`) and a 2-deep engine_output_fifo tag, so
two dispatches' outputs don't interleave in the single shared eofifo. This is the
deepest change and the one most likely to break byte-exactness (the eofifo is a
single shared stream; two producers' beats would interleave unless tagged).

================================================================================
## 5. DEADLOCK / DE-SYNC SAFETY ARGUMENT

### For the EXISTING (already-shipped) overlap — PROVEN SAFE
- **No beat drops:** every spatial stage is ENABLE_BACKPRESSURE(1) elastic
  (advance-iff-latch); `s_axis_tready = node_conv_810_ready_in & spatial_run`
  (top:459) back-pressures the host. The engine writes act-BRAM at top arbiter
  priority (top:2451), loaders are deferred not dropped. The eofifo has
  ENABLE_OUTPUT_BACKPRESSURE so the engine stalls on a full FIFO (top:2598,2618).
- **No deadlock:** the only place the engine could wedge (12544-beat SLOT-0 >
  4096 FIFO) is handled by `engine_drain_run` including `engine_busy` (top:457,
  comment [DRAIN-FIX2]) so the FIFO drains concurrently.
- **Add joins safe:** §3a — all 10 adds are valid/ready elastic with full-frame
  skip FIFOs; overlap changes cadence, not pairing.
- **Empirical:** 8/8 byte-exact at 7.74M cyc (e2e_mp8_allvecs.log).

### For the FURTHER overlap (§4) — NOT proven safe; specific hazards
1. **Wrong-skip-beat corruption at adds 336/408/546/618/690** (§3b) -> guard 4c.
2. **eofifo beat interleave** when two bridges drain (4d) -> needs per-dispatch
   tagging or a second FIFO; without it, byte-exactness breaks silently.
3. **Bank collision** N-output vs N+1-prefetch (4b) -> needs a 3rd bank.
4. **`current_loaded`/`current_drain_complete` are scalar muxes** on a single
   `sched_dispatch_idx` (top:2580, 3418) -> must split into ld/compute pointers.

================================================================================
## 6. SAFEST PARTIAL-OVERLAP THAT STILL NETS CYCLES (recommended)

Because the big XOR is already removed, the cheap remaining win is **input-loader
prefetch for the SUBSET of dispatches whose input bank does NOT collide with the
prior dispatch's output bank** — i.e. dispatches where `input_bank_rom[N+1] !=
output_bank_rom[N]`. From scheduler:771-851, the banks already alternate, so for
many adjacent pairs the prefetch target is the bank the engine is NOT writing.

Minimal, lower-risk staged plan (each independently e2e-gateable):

**Stage 1 (lowest risk, pure guard, NO behavior change):** add the 5 missing LHS
skids (§4c) to adds 336/408/546/618/690. This is byte-exact on its own (the adds
already pair by valid/ready; a transparent in-order FIFO only delays the lhs pop)
and is the prerequisite that makes ANY further overlap safe. Verify 8/8 first.

**Stage 2 (medium risk):** allow `ld_idx = dispatch_idx + 1` prefetch ONLY when
`input_bank_rom[dispatch_idx+1] != output_bank_rom[dispatch_idx]` (a static
per-dispatch enable ROM). Gate the prefetch loader's `in_valid` by this enable so
it can fill during `S_WAIT_DONE`. No 3rd bank needed (only non-colliding pairs),
no eofifo change (still one active drain slot). This nets the load-latency of the
non-colliding dispatches without the deep hazards of §4d.

**Stage 3 (high risk, defer):** full triple-buffer + 2-slot drain (§4b/4d). This
is the multi-day, regen-pipeline change. Do NOT attempt as a top patch.

================================================================================
## 7. RISK VERDICT

- **Existing overlap:** DONE, ACTIVE, byte-exact 8/8 @ 7.74M cyc. The single
  largest serialization (engine_busy XOR spatial) is ALREADY removed safely.
- **Further overlap (cross-dispatch software-pipeline):** NOT safely doable in
  one shot and NOT a pure top edit. It requires (a) a 3rd activation bank /
  disjoint base-word map (regen-pipeline: build_scheduler.py + build_top_wrapper.ts
  + memory maps), (b) split ld/compute pointers in the FSM, (c) per-dispatch
  eofifo tagging for 2-slot drain, and (d) 5 new LHS skids. Each of (a)-(c) can
  silently break byte-exactness; this is multi-day work that MUST be staged and
  e2e-gated per stage.
- **Recommended now:** Stage 1 (5 LHS skids; byte-exact prerequisite) + Stage 2
  (non-colliding-pair input prefetch). These net real cycles without touching the
  eofifo/bank map. Defer Stage 3.

**Do NOT claim a further-overlap speedup is byte-exact until the full 8-vector
e2e (run_mbv2_top_value.ts, single-thread) reports mismatch_bytes=0 on all 8.**
The existing overlap already passes that gate; any new stage must re-pass it.
