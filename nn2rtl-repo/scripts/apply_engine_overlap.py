#!/usr/bin/env python3
"""apply_engine_overlap.py — engine/spatial OVERLAP lever (ResNet-50 top).

Lets the spatial chain keep streaming while the shared engine runs its 17
dispatches (scheduler S_WAIT_DONE previously froze the chain: 6.118M cycles
= 47.75% of the 12,813,738-cycle frame). Verified hazard analysis:
docs/agent_tasks/ENGINE_OVERLAP_ANALYSIS.md.

Three coordinated, byte-exact-by-construction edits:

  (A) INPUT-REGION REMAP — 5 dispatches whose loader fills DURING the
      previous dispatch's engine run land in the very region that run is
      reading (all bank-2 loaders share base 8192 today):
          d3  (conv_260, 196w)  8192 -> 12288   protects d2's read
          d6  (conv_272, 196w)  8192 -> 12544   protects d5's read
          d10 (conv_286,  98w)  8192 -> 12800   protects d9's read
          d13 (conv_294,  98w)  8192 -> 12928   protects d12's read
          d16 (conv_300,  98w)  8192 -> 13056   protects d15's read
      Banks 3-5 (words 12288..24575) are EMPTY in the schedule (zero
      module_owners), so the new regions are virgin storage. Patches BOTH
      the scheduler act_in_base_word_rom AND the matching loader
      BRAM_BASE_ADDR (they must change together).

  (B) ENGINE ACT-BRAM WRITE REMOVAL — the engine's act_out writes into the
      activation BRAM are DEAD for all 17 dispatches (proven: the act mem's
      only read port is the engine's act_in; every dispatch input region is
      loader-filled-complete before engine_start (current_loaded gate);
      residual/skip data rides skip_fifo's, never the act BRAM; the REAL
      outputs ride the engine_output_fifo, tapped directly off
      engine_act_out_wr_en/data — untouched here). Dropping the engine from
      the act-write arbiter (instead of remapping its base) also removes
      write-port contention with loader fills during runs (the engine had
      absolute priority and would starve loader grants -> 1-deep-skid
      overflow -> B20-class beat loss).

  (C) THE UNGATE — scheduler S_WAIT_DONE drives spatial_stall=0, and the
      top's spatial_throttle drops engine_busy (MBV2 overlap precedent).

Usage:
  python scripts/apply_engine_overlap.py [--dry-run]

Idempotent: re-running on a patched tree is a no-op (reports 'already
applied'). Backups: <file>.pre_overlap saved once (first apply).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOP = REPO / "output" / "rtl" / "nn2rtl_top.v"
SCHED = REPO / "output" / "rtl" / "nn2rtl_scheduler.v"

MARKER = "[OVERLAP]"

# (dispatch_idx, loader_instance, total_words, old_base, new_base)
REMAPS = [
    (3,  "u_ldr_node_conv_260", 196, 8192, 12288),
    (6,  "u_ldr_node_conv_272", 196, 8192, 12544),
    (10, "u_ldr_node_conv_286",  98, 8192, 12800),
    (13, "u_ldr_node_conv_294",  98, 8192, 12928),
    (16, "u_ldr_node_conv_300",  98, 8192, 13056),
]


def fail(msg: str) -> None:
    print(f"[apply_engine_overlap] FAIL: {msg}")
    sys.exit(1)


def patch_scheduler(text: str) -> tuple[str, list[str]]:
    notes: list[str] = []

    # ---- (A) act_in_base_word_rom remaps -------------------------------
    m = re.search(
        r"reg \[15:0\] act_in_base_word_rom;.*?endcase", text, re.S)
    if not m:
        fail("scheduler: act_in_base_word_rom block not found")
    block = m.group(0)
    new_block = block
    for d, inst, words, old, new in REMAPS:
        pat_old = f"5'd{d}: act_in_base_word_rom = 16'd{old};"
        pat_new = (f"5'd{d}: act_in_base_word_rom = 16'd{new};"
                   f"  // {MARKER} remap (was {old}); see ENGINE_OVERLAP_ANALYSIS.md")
        if pat_new in new_block:
            notes.append(f"sched act_in d{d}: already applied")
            continue
        if pat_old not in new_block:
            fail(f"scheduler: anchor not found: {pat_old!r}")
        new_block = new_block.replace(pat_old, pat_new, 1)
        notes.append(f"sched act_in d{d}: {old} -> {new}")
    text = text.replace(block, new_block, 1)

    # ---- (C) S_WAIT_DONE ungate ----------------------------------------
    anchor_old = ("            S_WAIT_DONE: begin\n"
                  "                spatial_stall = 1'b1;\n"
                  "            end")
    anchor_new = ("            S_WAIT_DONE: begin\n"
                  f"                // {MARKER} engine/spatial overlap: the chain keeps streaming\n"
                  "                // while the engine runs. Safe because (1) every dispatch\n"
                  "                // input region is loader-private (5 remapped to bank 3),\n"
                  "                // (2) the engine no longer writes the act BRAM (FIFO-only\n"
                  "                // output), (3) skip FIFOs bound at full-map < DEPTH.\n"
                  "                spatial_stall = 1'b0;\n"
                  "            end")
    if anchor_new in text:
        notes.append("sched S_WAIT_DONE: already applied")
    else:
        if anchor_old not in text:
            fail("scheduler: S_WAIT_DONE output-block anchor not found")
        text = text.replace(anchor_old, anchor_new, 1)
        notes.append("sched S_WAIT_DONE: spatial_stall 1 -> 0")
    return text, notes


def patch_top(text: str) -> tuple[str, list[str]]:
    notes: list[str] = []

    # ---- (A) loader BRAM_BASE_ADDR remaps ------------------------------
    for d, inst, words, old, new in REMAPS:
        # Find the instance, then the param block immediately before it.
        idx = text.find(f") {inst} (")
        if idx < 0:
            fail(f"top: loader instance {inst} not found")
        start = text.rfind("stream_to_act_bram_bridge #(", 0, idx)
        if start < 0:
            fail(f"top: param block for {inst} not found")
        seg = text[start:idx]
        if f".BRAM_BASE_ADDR({new})" in seg:
            notes.append(f"top {inst}: already applied")
            continue
        if f".BRAM_BASE_ADDR({old})" not in seg:
            fail(f"top {inst}: expected .BRAM_BASE_ADDR({old}) not found")
        if f".TOTAL_BRAM_WORDS({words})" not in seg:
            fail(f"top {inst}: expected .TOTAL_BRAM_WORDS({words}) not found")
        new_seg = seg.replace(
            f".BRAM_BASE_ADDR({old})",
            f".BRAM_BASE_ADDR({new})  /* {MARKER} was {old} */", 1)
        text = text[:start] + new_seg + text[idx:]
        notes.append(f"top {inst}: BRAM_BASE_ADDR {old} -> {new}")

    # ---- (B) drop engine from the act-write arbiter --------------------
    if f"{MARKER} engine act-BRAM write removed" in text:
        notes.append("top arbiter: already applied")
    else:
        # grant lines: ldr0 special-case, ldr1..ldr16 share the same shape.
        g0_old = "assign ldr0_wr_grant = ldr0_wr_req & ~(engine_act_out_wr_en);"
        g0_new = (f"// {MARKER} engine act-BRAM write removed: act_out is DEAD in BRAM\n"
                  "    // (sole act-mem read port = engine act_in; all inputs loader-filled;\n"
                  "    // skips ride skip_fifo's; real outputs ride engine_output_fifo which\n"
                  "    // taps engine_act_out_wr_en/data directly). Loaders now never lose\n"
                  "    // grants to the engine during overlapped runs.\n"
                  "    assign ldr0_wr_grant = ldr0_wr_req;")
        if g0_old not in text:
            fail("top: ldr0 grant anchor not found")
        text = text.replace(g0_old, g0_new, 1)
        n = text.count("~(engine_act_out_wr_en | ")
        if n != 16:
            fail(f"top: expected 16 ldrN grant terms, found {n}")
        text = text.replace("~(engine_act_out_wr_en | ", "~(", 16)

        en_old = "assign act_wr_en_final   = engine_act_out_wr_en | ldr0_wr_req"
        if en_old not in text:
            fail("top: act_wr_en_final anchor not found")
        text = text.replace(
            en_old, "assign act_wr_en_final   = ldr0_wr_req", 1)

        addr_old = ("assign act_wr_addr_final = engine_act_out_wr_en ? "
                    "engine_act_out_wr_addr[14:0] : ldr0_wr_req ? ldr0_wr_addr")
        if addr_old not in text:
            fail("top: act_wr_addr_final anchor not found")
        text = text.replace(
            addr_old,
            "assign act_wr_addr_final = ldr0_wr_req ? ldr0_wr_addr", 1)

        data_old = ("assign act_wr_data_final = engine_act_out_wr_en ? "
                    "engine_act_out_wr_data : ldr0_wr_req ? ldr0_wr_data")
        if data_old not in text:
            fail("top: act_wr_data_final anchor not found")
        text = text.replace(
            data_old,
            "assign act_wr_data_final = ldr0_wr_req ? ldr0_wr_data", 1)

        # engine_act_out_wr_addr is now unused below bit 15; tie it off.
        tie_anchor = "wire _unused_act_out_addr_hi = |engine_act_out_wr_addr[15:15];"
        if tie_anchor not in text:
            fail("top: act_out addr tie-off anchor not found")
        text = text.replace(
            tie_anchor,
            f"wire _unused_act_out_addr_full = |engine_act_out_wr_addr;  // {MARKER} BRAM write dropped",
            1)
        notes.append("top arbiter: engine act-BRAM write removed (17 grants + en/addr/data)")

    # ---- (C) spatial_throttle drops engine_busy ------------------------
    thr_old = ("(* max_fanout = 32 *) wire spatial_throttle = "
               "engine_busy | sched_spatial_stall;")
    thr_new = (f"// {MARKER} overlap: engine_busy no longer freezes the chain (the\n"
               "    // scheduler's spatial_stall covers config-write/start windows only).\n"
               "    (* max_fanout = 32 *) wire spatial_throttle = sched_spatial_stall;\n"
               "    wire _unused_engine_busy_throttle = engine_busy;")
    if thr_new.splitlines()[2].strip() in text:
        notes.append("top throttle: already applied")
    else:
        if thr_old not in text:
            fail("top: spatial_throttle anchor not found")
        text = text.replace(thr_old, thr_new, 1)
        notes.append("top throttle: engine_busy dropped")

    return text, notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for f in (TOP, SCHED):
        if not f.exists():
            fail(f"missing {f}")

    # Process LF-normalized; restore each file's original ending on write.
    def read_norm(f: Path) -> tuple[str, str]:
        raw = f.read_bytes().decode("utf-8")
        eol = "\r\n" if "\r\n" in raw else "\n"
        return raw.replace("\r\n", "\n"), eol

    sched_text, sched_eol = read_norm(SCHED)
    top_text, top_eol = read_norm(TOP)

    new_sched, sched_notes = patch_scheduler(sched_text)
    new_top, top_notes = patch_top(top_text)

    for n in sched_notes + top_notes:
        print(f"[apply_engine_overlap] {n}")

    changed = (new_sched != sched_text) or (new_top != top_text)
    if args.dry_run:
        print(f"[apply_engine_overlap] DRY-RUN ok; would_change={changed}")
        return
    if not changed:
        print("[apply_engine_overlap] no changes (already fully applied)")
        return

    for f, old, new, eol in ((SCHED, sched_text, new_sched, sched_eol),
                             (TOP, top_text, new_top, top_eol)):
        bak = f.with_suffix(f.suffix + ".pre_overlap")
        if old != new:
            if not bak.exists():
                shutil.copyfile(f, bak)
                print(f"[apply_engine_overlap] backup -> {bak.name}")
            f.write_bytes(new.replace("\n", eol).encode("utf-8"))
            print(f"[apply_engine_overlap] wrote {f}")
    print("[apply_engine_overlap] DONE — now run scripts/check_act_region_hazards.py, then the e2e gate")


if __name__ == "__main__":
    main()
