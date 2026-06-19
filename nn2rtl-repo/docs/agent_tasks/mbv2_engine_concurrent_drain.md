# MBV2 engine-top: concurrent-drain blocker — verdict + minimal byte-exact fix

**Scope:** read-only RTL ground-truthing of the "engine_output_fifo cannot drain
while `engine_busy=1`" deadlock that follows the (already-applied, byte-exact)
blocker #2 backpressure primitive. NO RTL edited here. All line numbers verified
against the on-disk RTL on 2026-06-02.

Files (absolute):
- `c:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/rtl/nn2rtl_top_engine.v`
- `c:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/rtl/nn2rtl_scheduler.v`
- `c:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/rtl/shared_engine_skeleton.v`
- `c:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/rtl/engine/bram_to_stream_bridge.v`
- `c:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/rtl/engine/address_generator.v`

---

## 1. Is the act-BRAM WRITE-WRITE HAZARD real? — NO. The roadmap got it wrong.

The roadmap (blocker #2, "alternative (A) REJECTED", roadmap lines 166-174) claims
that ungating the drain from `engine_busy` creates an act-BRAM write-write hazard:
"the engine writes dispatch-0 output to `act_out_base_word=4096` (bank 1) while
ldr1 (`u_ldr_node_conv_816`, `BRAM_BASE_ADDR=4096`) would write the drained
dispatch-0 output back into the SAME bank-1/base-4096 region the engine is still
producing into → corrupts results."

**Ground truth from the RTL refutes the corruption claim on three independent
counts:**

### (a) The engine is ALREADY a direct, top-priority act-BRAM writer during compute.
A single signal `engine_act_out_wr_en` / `engine_act_out_wr_data` /
`engine_act_out_wr_addr` is FORKED to two destinations on the same cycle:
- The shared act-BRAM write arbiter, at **unconditional top priority**:
  `act_wr_en_final = engine_act_out_wr_en | ldr0_wr_req | ...` (line 2068);
  `act_wr_addr_final = engine_act_out_wr_en ? engine_act_out_wr_addr[14:0] : ...`
  (line 2069); same for data (line 2070). The single write port of
  `u_act_mem` (`act_unified_mem`, DEPTH=24576, lines 2073-2084) is driven by these.
- The `engine_output_fifo` `.in_valid(engine_act_out_wr_en)` /
  `.in_data(engine_act_out_wr_data)` (lines 2222-2223).

The engine output address = `cfg_act_out_bram_base + pixel_index*oc_passes + oc_pass_idx`
(`address_generator.v` line 203-204); for dispatch-0 `cfg_act_out_bram_base=4096`.
So **the engine writes bank-1 words 4096..4096+12543 DIRECTLY during compute**,
on the same cycles it pushes each beat into the FIFO. The framing that the engine
output "goes only to the FIFO" is false: the FIFO/bridge/ldr1 path is a
**parallel, redundant second copy** of data the engine has already committed to
bank 1.

### (b) The two bank-1 writers carry IDENTICAL data to IDENTICAL addresses.
- ldr0 = `u_ldr_node_conv_814` (`BRAM_BASE_ADDR=0`, bank 0) is fed by the SPATIAL
  chain (`n4_2_valid_out`, line 1367) — it loads the engine's **input** (bank 0).
- ldr1 = `u_ldr_node_conv_816` (`BRAM_BASE_ADDR=4096`, bank 1) is fed by
  `node_conv_814_data_out` (line 1388) — the drained engine **output** straight
  out of the dispatch-0 `engine_output_bridge` (SLOT 0), with **no intervening
  transform** between the bridge `data_out` and `ldr1.in_data`. The bridge merely
  slices each 2048b FIFO beat into 16×128b tiles (DATA_W=128) and ldr1 re-packs
  them back into 2048b bank-1 words. So ldr1 writes the SAME values to the SAME
  bank-1 addresses the engine already wrote directly. The two writers are
  content-redundant, not conflicting.

### (c) Even concurrent same-port requests are arbitrated atomically, no corruption.
The grant cascade masks every loader when the engine writes:
`ldr1_wr_grant = ldr1_wr_req & ~(engine_act_out_wr_en | ldr0_wr_req)` (line 2033).
The engine wins every collision; ldr1's `stream_to_act_bram_bridge` only advances
its word_count on `wr_req && wr_grant`, so it stalls (backpressures) rather than
corrupts. Two writers to one port is a structural fact handled by the arbiter; it
is NOT a data-corruption hazard, and here the two writers carry identical data
anyway.

### (d) dispatch-1's input does not even depend on the drain.
Dispatch-0 writes bank 1 (`act_out_base_word_rom[0]=4096`, line 938); dispatch-1
reads bank 1 (`act_in_base_word_rom[1]=4096`, line 898). Because the engine's
DIRECT write already populated bank 1, dispatch-1's input is resident with or
without the drain — the same proven pre-resident ping-pong that dispatch 20→21
already relies on (`all_loaded[21]=1'b1` hardwire, line 2118, no loader). The
FIFO/bridge/ldr1 path exists to feed the downstream SPATIAL stream
(`node_conv_814_valid_out` → relu/add consumers), NOT to populate dispatch-1's
BRAM input.

**VERDICT (1): the act-BRAM write-write HAZARD is NOT REAL.** The roadmap's
rejection of (A) conflated "two writers to one port" (handled by arbitration)
with "two writers of DIFFERENT data" (not the case — they are byte-identical),
and it overlooked that the engine itself is the primary bank-1 writer during
compute. The genuine bug is a **drain-concurrency deadlock**, not a hazard.

---

## The actual blocker (confirmed): drain-gated-behind-engine_done deadlock

- The dispatch-0 bridge `u_engine_out_node_conv_814` (SLOT 0,
  EXPECTED_BEATS=12544, line 2235-2251) has `.ready_out(spatial_run)` (line 2247).
- `spatial_run = ~(engine_busy | sched_spatial_stall)` (lines 443-444). During
  compute (`S_PULSE_START`/`S_WAIT_DONE`) the scheduler sets `spatial_stall=1`
  (scheduler lines 1127/1130) and `engine_busy=1`, so `spatial_run=0` → the
  bridge `ready_out=0` → it drains 0 beats.
- The engine forks every beat into the 4096-deep FIFO. With
  `ENABLE_OUTPUT_BACKPRESSURE(1)` (line 2179) and `.out_ready(eofifo_in_ready)`
  (line 2199), once the FIFO hits 4096 entries (`in_ready=0`) the engine STALLS in
  `ST_REQUANT` (`req_done_pending`, `shared_engine_skeleton.v` lines 267-306) and
  never reaches `engine_done`.
- The only state that opens the drain is `S_WAIT_DRAIN` (`spatial_stall=0`,
  scheduler line 1135), reachable only AFTER `engine_done` (`S_WAIT_DONE →
  S_WAIT_DRAIN` arc, scheduler line 1031). Classic deadlock: drain gated behind
  `engine_done`, `engine_done` gated behind a drain that cannot happen.

**Net concurrent-drain rate (re-derived from the roadmap probe):** engine writes
~44 cyc/beat, bridge+ldr1 drain ~39 cyc/beat (FASTER). Net fill = 1/44 − 1/39 =
−0.0029 beat/cyc → if drain is allowed concurrently, FIFO occupancy stays O(1).
The overflow is 100% caused by FORBIDDING drain during `engine_busy`, not by any
rate mismatch — so the existing 4096-deep FIFO is already ~256× oversized for
concurrent operation.

---

## 2. Recommended minimal byte-exact fix — (A) UNGATE the engine-output drain

The smallest, safest fix is to drop the `engine_busy` term from the
engine-output drain path **only** (keep `sched_spatial_stall`), so the SLOT-0
bridge drains and ldr1 writes concurrently with `engine_busy=1`. This is a
**TOP-WRAPPER-only** change — it touches NO engine pipeline
(`shared_engine`/`bram_to_stream_bridge`), so it is OUTSIDE the
`engine-pipeline-change` safety rule the roadmap invoked.

### Exact edits (top-wrapper only, `nn2rtl_top_engine.v`)

Introduce a drain-only run signal that ignores `engine_busy` but still honors the
scheduler stall, then use it for the SLOT-0 engine-output bridge `ready_out` and
the ldr1 input gate. (Edits NOT applied here — read-only task.)

1. After line 444 add:
   ```
   // Engine-output DRAIN may proceed concurrently with engine_busy: the engine
   // already commits its output to act-BRAM directly (top-priority port), so the
   // FIFO/bridge/ldr1 path is a redundant copy; draining it during compute keeps
   // the 4096 FIFO at O(1) occupancy and cannot corrupt bank 1 (identical data,
   // arbiter gives the engine priority). Still gate on sched_spatial_stall so the
   // drain pauses during AXI-config / dispatch-boundary windows.
   wire engine_drain_run = ~sched_spatial_stall;
   ```

2. Line 2247 — SLOT-0 bridge `.ready_out(spatial_run)` → `.ready_out(engine_drain_run)`.

3. Line 1387 — ldr1 `.in_valid(node_conv_814_valid_out & spatial_run)` →
   `.in_valid(node_conv_814_valid_out & engine_drain_run)`.

Leave SLOT-1 (`u_engine_out_node_conv_816`, line 2267, `ready_out =
(n4_3_ready_in & spatial_run)`) and all other spatial gates UNCHANGED unless the
e2e shows the same overflow on dispatch-1 (also 12544 beats, line 2259) — in which
case apply the identical 2-line treatment to SLOT-1 and its consumer loader.
**Total edit: 1 wire + 2 single-term swaps (≤4 lines).**

### Byte-exactness argument
- The engine's DIRECT bank-1 write (top-priority port) is the authoritative copy
  of dispatch-0 output and is UNAFFECTED by ungating — the engine always wins the
  arbiter (`act_wr_*_final = engine_act_out_wr_en ? engine... : ldr...`, lines
  2068-2070), so ldr1 can never displace an engine write.
- ldr1's writes are byte-identical content to the same bank-1 addresses, and ldr1
  honors `wr_grant` backpressure (it advances only on `wr_req && wr_grant`), so a
  concurrent request is deferred a cycle, never dropped, never reordered into a
  wrong final value. Bank-1 contents CONVERGE to the correct word set.
- dispatch-1 only begins reading bank 1 after `S_WAIT_DRAIN → S_NEXT_DISP →
  S_WRITE` (engine fully done + drain_complete), by which point both writers have
  long finished. There is no read-during-write window into bank 1.
- The backpressure hold in `bram_to_stream_bridge.v` (lines 105-113) is idempotent
  under ungating: when the FIFO is full it re-asserts the SAME `act_out_wr_en`/data
  for the SAME engine output address (`wr_fire = in_valid && !fifo_full` prevents
  double-push), so even repeated direct writes of one beat are harmless.
- Downstream SPATIAL correctness: `node_conv_814_valid_out` now flows during
  compute instead of only after. The data values are unchanged; only the cadence
  shifts earlier. The bridge's tile/beat ordering is FIFO-preserved, so the
  spatial relu/add consumers see the identical byte stream in the identical order.

The ONE residual risk is purely cadence at the spatial-chain join (e.g. n4_3
consuming `node_conv_816` downstream) — a handshake/timing concern, NOT a datapath
value risk. The engine-top e2e covers it directly.

### How to verify (engine-top e2e, ~11 min)
1. Apply the ≤4-line edit (supervised), keep a backup of `nn2rtl_top_engine.v`.
2. Run the engine-top e2e harness for MobileNetV2 (the ~11-min value run). Expect:
   - dispatch-0 now COMPLETES (no `S_WAIT_DRAIN` freeze; `drain_complete[0]`
     asserts because all 12544 beats drain).
   - Final logits byte-exact vs the golden (mismatch=0).
3. Sanity probe (optional): assert FIFO occupancy stays well below 4096 during
   `engine_busy` (expected O(1) given the −0.0029 beat/cyc net fill).
4. NO engine-iso re-verification is required — the engine pipeline
   (`shared_engine`/`bram_to_stream_bridge`) is untouched, so the existing
   mbv2 34/34 + ResNet 14/14 byte-exact proofs still hold.

### Fallback if e2e reveals a genuine different-data corruption (it should not)
Fall back to **(B) deepen the FIFO** to the worst case (see §3): a top-wrapper
param change `DEPTH 4096→16384, ADDR_W 12→14` (2 params), capacity-only and
byte-exact like blocker #1. Per-FIFO 12544×2048b = 25.69 Mbit = 88 URAM288 — the
roadmap's "infeasible on-chip" is OVERSTATED for U250 (1280 URAM288 total → 88 =
6.9%, and routed baseline URAM is at 16%). Feasible but the least elegant (8.4×
the working set). Prefer (A); (B) only as a fallback.

**Rejected alternatives:** (C) narrow/variable-width FIFO — the FIFO is shared
across 34 dispatches with OC 16..1280, so a narrow width either re-bloats for the
1280-ch head or needs a width-strip stage; higher invasiveness, no advantage over
(A). (D) ping-pong/different drain bank — the bank ROMs are fixed in the
scheduler; re-banking desyncs dispatch-1's read and adds a third copy. Both
REJECTED.

---

## 3. FIFO-sizing worst case (max output beats over all 34 dispatches)

`max_output_beats_dispatch = 12544`, shared by SLOT 0 (`node_conv_814`) and
SLOT 1 (`node_conv_816`) — both `.EXPECTED_BEATS(12544)` (lines 2239, 2259), the
112×112 stem-adjacent layers (112*112 = 12544 spatial positions, 1 beat each).
Full EXPECTED_BEATS distribution across the 34 drain bridges:
**12544 ×2, 3136 ×4, 784 ×6, 588 ×3, 392 ×4, 245 ×1, 196 ×10, 98 ×1, 49 ×3.**
12544 is decisively the sizing worst case. Relevant ONLY if a sizing fix (B) is
chosen; the recommended fix (A) keeps the existing 4096-deep FIFO (O(1)
occupancy under concurrent drain).

---

## 4. Surgical vs architecture change + effort

**SURGICAL.** The recommended fix (A) is a top-wrapper-only edit of ≤4 lines
(1 new wire + 2 single-term gate swaps). It does NOT touch the byte-exact engine
pipeline, does NOT change the scheduler FSM or bank ROMs, and does NOT require
engine-iso re-verification. This is the opposite of an architecture change.

Effort: minutes to edit + one ~11-min engine-top e2e to confirm. The roadmap's
framing of #2's fix as a HIGH-risk `engine-pipeline-change` applied to the
already-DONE backpressure primitive; the REMAINING concurrent-drain step is a
small wrapper gate change, not an engine-pipeline change and not an architecture
change.
