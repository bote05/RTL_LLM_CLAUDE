#!/usr/bin/env python3
"""check_act_region_hazards.py — static act-BRAM hazard prover for the
engine/spatial OVERLAP lever (must PASS before running the e2e gate).

Model (all derived from the ACTUAL artifacts, asserted here):
  * act mem = flat 24576 x 2048b; ONE read port (engine act_in), ONE write
    port (arbiter: loaders [+ engine, unless the overlap patch removed it]).
  * Every dispatch d's engine act_in read region must EXACTLY equal its
    loader's fill region (the current_loaded gate then guarantees the
    region is fully written before engine_start).
  * Under overlap, while dispatch d runs, the spatial chain processes d's
    output stream; the ONLY loader(s) that can receive beats during d's
    run are the SUCCESSOR dispatch's loader (chain topology: the stream
    stops at the next engine conv because its output bridge slot is not
    active). Successor map below was derived from nn2rtl_top.v wiring
    (loader in_valid producers) and is asserted against the schedule.

Checks:
  C1  loader region == engine read region, for all 17 dispatches.
  C2  engine act-BRAM writes are removed from the arbiter (else: report
      every dispatch whose act_out region overlaps a LIVE loader region
      that fills concurrently -> FAIL).
  C3  for every dispatch d and every loader L filling during d's run:
      region(L) does not overlap read_region(d).
  C4  loader lifetime safety: for any two loaders with overlapping
      regions, the later fill must start at-or-after the earlier loader's
      consuming run (fill window of loader(s) = run of dispatch s-1, or
      the pre-engine phase for d0/d1) — i.e. no fill clobbers parked,
      not-yet-consumed input data.
  C5  every region within [0, 24576); remapped (>=12288) regions mutually
      disjoint and inside the schedule-empty banks 3-5.

Exit 0 = PASS, exit 1 = FAIL (with the violating pair printed).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOP = REPO / "output" / "rtl" / "nn2rtl_top.v"
SCHED = REPO / "output" / "rtl" / "nn2rtl_scheduler.v"
SCHEDULE = REPO / "output" / "rtl" / "nn2rtl_scheduler_schedule.json"

# dispatch -> loader instance (from nn2rtl_top.v all_loaded mux + instances)
DISPATCH_LOADER = {
    0: "u_ldr_node_conv_246", 1: "u_ldr_node_conv_250", 2: "u_ldr_node_conv_254",
    3: "u_ldr_node_conv_260", 4: "u_ldr_node_conv_264", 5: "u_ldr_node_conv_266",
    6: "u_ldr_node_conv_272", 7: "u_ldr_node_conv_278", 8: "u_ldr_node_conv_282",
    9: "u_ldr_node_conv_284", 10: "u_ldr_node_conv_286", 11: "u_ldr_node_conv_290",
    12: "u_ldr_node_conv_292", 13: "u_ldr_node_conv_294", 14: "u_ldr_node_conv_296",
    15: "u_ldr_node_conv_298", 16: "u_ldr_node_conv_300",
}

# Expected dispatch module order — the successor map below is ONLY valid for
# this exact schedule. If the schedule changes, re-derive (see analysis doc).
EXPECTED_MODULES = [
    "node_conv_246", "node_conv_250", "node_conv_254", "node_conv_260",
    "node_conv_264", "node_conv_266", "node_conv_272", "node_conv_278",
    "node_conv_282", "node_conv_284", "node_conv_286", "node_conv_290",
    "node_conv_292", "node_conv_294", "node_conv_296", "node_conv_298",
    "node_conv_300",
]

# Which loaders can RECEIVE stream beats while dispatch d's engine run is in
# flight (overlap ungated). Derived from the spatial wiring between each
# dispatch's output bridge and the next loader's producer relu:
#   d0 -> none   (relu_23 -> conv_248 parks into add_7's skip fifo; add_7
#                 stalls until d1's bridge; ldr1 was filled pre-engine-phase
#                 by relu_21)
#   d1 -> ldr(d2)  via add_7 -> relu_24 -> conv_252 -> relu_25
#   d2 -> ldr(d3)  via relu_26 -> conv_256 -> add_8 -> relu_27 -> conv_258 -> relu_28
#   d3 -> ldr(d4)  via relu_29 -> conv_262 -> add_9 -> relu_30
#   d4 -> ldr(d5)  via relu_31
#   d5 -> ldr(d6)  via relu_32 -> conv_268 -> add_10 -> relu_33 -> conv_270 -> relu_34
#   d6 -> ldr(d7)  via relu_35 -> conv_274 -> add_11 -> relu_36 -> conv_276 -> relu_37
#   d7 -> ldr(d8)  via relu_38 -> conv_280 -> add_12 -> relu_39
#   d8 -> ldr(d9)  via relu_40
#   d9 -> ldr(d10) via relu_41
#   d10 -> ldr(d11) via add_13 -> relu_42 -> skid_ldr_290
#   d11 -> ldr(d12) via relu_43
#   d12 -> ldr(d13) via relu_44
#   d13 -> ldr(d14) via add_14 -> relu_45 -> skid_ldr_296
#   d14 -> ldr(d15) via relu_46
#   d15 -> ldr(d16) via relu_47
#   d16 -> none (relu_48 = network output)
FILLS_DURING_RUN = {0: []} | {d: [d + 1] for d in range(1, 16)} | {16: []}

# Fill window of each loader, expressed as "the run of dispatch w" (or PRE):
# loader(d) fills during run(d-1) for d>=2; ldr0/ldr1 fill in the pre-engine
# phase (producers relu_22/relu_21 are upstream of conv_246).
PRE_PHASE = -1


def fill_window(d: int) -> int:
    return PRE_PHASE if d <= 1 else d - 1


def fail(msg: str) -> None:
    print(f"[hazard-check] FAIL: {msg}")
    sys.exit(1)


def words(ch: int, h: int, w: int) -> int:
    return h * w * ((ch + 255) // 256)


def main() -> None:
    sched_text = SCHED.read_text(encoding="utf-8")
    top_text = TOP.read_text(encoding="utf-8")
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))

    dispatches = schedule["dispatches"]
    if [d["module_id"] for d in dispatches] != EXPECTED_MODULES:
        fail("schedule dispatch order changed — successor map must be re-derived "
             "(see ENGINE_OVERLAP_ANALYSIS.md)")

    # empty banks per schedule
    empty_banks = {b["bank_id"] for b in schedule["banks"]
                   if b["max_bytes_used"] == 0 and not b["module_owners"]}
    if not {3, 4, 5} <= empty_banks:
        fail(f"banks 3-5 not all empty in schedule (empty={sorted(empty_banks)})")

    # --- parse scheduler act_in_base ROM ---
    m = re.search(r"reg \[15:0\] act_in_base_word_rom;.*?endcase", sched_text, re.S)
    if not m:
        fail("scheduler act_in_base_word_rom not found")
    act_in_base = {int(a): int(b) for a, b in
                   re.findall(r"5'd(\d+): act_in_base_word_rom = 16'd(\d+);", m.group(0))}

    m = re.search(r"reg \[15:0\] act_out_base_word_rom;.*?endcase", sched_text, re.S)
    if not m:
        fail("scheduler act_out_base_word_rom not found")
    act_out_base = {int(a): int(b) for a, b in
                    re.findall(r"5'd(\d+): act_out_base_word_rom = 16'd(\d+);", m.group(0))}

    # --- parse top loader params ---
    loader_region: dict[int, tuple[int, int]] = {}
    for d, inst in DISPATCH_LOADER.items():
        idx = top_text.find(f") {inst} (")
        if idx < 0:
            fail(f"top: loader {inst} not found")
        seg = top_text[top_text.rfind("stream_to_act_bram_bridge #(", 0, idx):idx]
        mb = re.search(r"\.BRAM_BASE_ADDR\((\d+)\)", seg)
        mw = re.search(r"\.TOTAL_BRAM_WORDS\((\d+)\)", seg)
        if not (mb and mw):
            fail(f"top: loader {inst} params not parseable")
        loader_region[d] = (int(mb.group(1)), int(mw.group(1)))

    # --- engine read/write regions from schedule geometry ---
    read_region: dict[int, tuple[int, int]] = {}
    write_region: dict[int, tuple[int, int]] = {}
    for d in dispatches:
        i = d["dispatch_index"]
        ih, iw = d["input_hw"]
        oh, ow = d["output_hw"]
        read_region[i] = (act_in_base[i], words(d["channel_in"], ih, iw))
        write_region[i] = (act_out_base[i], words(d["channel_out"], oh, ow))

    def overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return a[0] < b[0] + b[1] and b[0] < a[0] + a[1]

    failures: list[str] = []

    # C1: loader region == read region
    for i in range(17):
        if loader_region[i] != read_region[i]:
            failures.append(
                f"C1 d{i}: loader region {loader_region[i]} != engine read region {read_region[i]}")

    # C2: engine removed from the act-write arbiter?
    engine_in_arbiter = ("act_wr_en_final   = engine_act_out_wr_en" in top_text
                         or "engine_act_out_wr_en | ldr0_wr_req" in top_text)
    if engine_in_arbiter:
        # engine still writes BRAM: its write region must not touch any loader
        # region that is live (filled, not yet consumed) or filling during the run
        for i in range(17):
            for s, (base, n) in loader_region.items():
                if s == i:
                    continue
                # live span of loader s's data: fill_window(s) .. run s
                if fill_window(s) <= i <= s and overlap(write_region[i], (base, n)):
                    failures.append(
                        f"C2 d{i}: engine act_out {write_region[i]} overlaps live loader "
                        f"region of d{s} {(base, n)} (engine still in arbiter)")
    else:
        print("[hazard-check] C2: engine act-BRAM write removed from arbiter (writes are FIFO-only) — OK")

    # C3: concurrent fill vs engine read
    for i in range(17):
        for s in FILLS_DURING_RUN[i]:
            if overlap(loader_region[s], read_region[i]):
                failures.append(
                    f"C3 d{i}: loader of d{s} fills {loader_region[s]} DURING d{i}'s run, "
                    f"overlapping d{i}'s read region {read_region[i]}")

    # C4: fill must not clobber parked, not-yet-consumed loader data
    for a in range(17):
        for b in range(17):
            if a == b or not overlap(loader_region[a], loader_region[b]):
                continue
            if a < b and not (fill_window(b) >= a):
                failures.append(
                    f"C4 d{b}: fill (window run d{fill_window(b)}) precedes d{a}'s "
                    f"consumption but regions overlap")

    # C5: bounds + remap disjointness
    for i, (base, n) in loader_region.items():
        if not (0 <= base and base + n <= 24576):
            failures.append(f"C5 d{i}: region {(base, n)} out of act-mem bounds")
    remapped = [(i, r) for i, r in loader_region.items() if r[0] >= 12288]
    for x in range(len(remapped)):
        for y in range(x + 1, len(remapped)):
            if overlap(remapped[x][1], remapped[y][1]):
                failures.append(f"C5: remapped regions overlap: d{remapped[x][0]} d{remapped[y][0]}")
        i, (base, n) = remapped[x]
        if base + n > 16384 and not engine_in_arbiter:
            pass  # banks 4-5 also empty; only flag if engine writes there
        if base + n > 24576:
            failures.append(f"C5 d{i}: remapped region exceeds act mem")

    # report
    print("[hazard-check] dispatch table:")
    print("  d  module          read[base,+n)      loader[base,+n)    fill-during-run-of")
    for d in dispatches:
        i = d["dispatch_index"]
        fw = fill_window(i)
        print(f"  {i:>2} {d['module_id']:<15} {str(read_region[i]):<18} "
              f"{str(loader_region[i]):<18} {'PRE-PHASE' if fw == PRE_PHASE else 'd' + str(fw)}")

    if failures:
        for f_ in failures:
            print(f"[hazard-check] {f_}")
        print(f"[hazard-check] RESULT: FAIL ({len(failures)} violations)")
        sys.exit(1)
    print("[hazard-check] RESULT: PASS (C1-C5 clean across all 17 dispatches)")


if __name__ == "__main__":
    main()
