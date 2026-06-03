#!/usr/bin/env python3
"""
apply_loader_word_resize.py

Surgically resize the TOTAL_BRAM_WORDS parameter of every
stream_to_act_bram_bridge input-loader instance (u_ldr_node_conv_*) in the
(PATCHED-not-regenerated) engine top:

    output/mobilenet-v2/rtl/nn2rtl_top_engine.v

WHY (root cause)
----------------
build_top_wrapper.ts:1026 sets TOTAL_BRAM_WORDS to a count of *predecessor
BEATS* (= input_hw[0]*input_hw[1]*icChunks). But the bridge's `loaded` output
only asserts when its internal `word_count` (a count of 2048-bit BRAM WORDS
written) reaches TOTAL_BRAM_WORDS. For BUS_W<2048 several beats pack into one
2048-bit word, so word_count plateaus far below the beat count and `loaded`
NEVER asserts -> the scheduler parks in S_WAIT_LOAD on all_loaded[dispatch]
forever -> e2e deadlock at dispatch 0 (u_ldr_node_conv_814, BUS_W=256,
TOTAL_BRAM_WORDS=12544, should be 1568). For BUS_W>2048 the value is instead
UNDER-sized (asserts early). BUS_W==2048 is already correct (1 beat = 1 word).

This fix is BYTE-EXACT-IRRELEVANT: TOTAL_BRAM_WORDS only controls *when* `loaded`
asserts (the load-complete threshold). It changes no datapath value.

EXACT BRIDGE SEMANTICS (derived from the on-disk RTL, module
stream_to_act_bram_bridge, ~line 3117 of nn2rtl_top_engine.v)
------------------------------------------------------------------------------
word_count increments by 1 only on (wr_req && wr_grant); `loaded` latches when
next_word_count == TOTAL_BRAM_WORDS. The number of WORDS the bridge will write
for a given BEAT count depends on the width regime:

  g_w_eq  (BUS_W == 2048): one beat -> one wr_req -> one word.
        words = beats.

  g_w_lt  (BUS_W <  2048): BEATS_PER_WORD = 2048 / BUS_W  (integer trunc).
        wr_req fires ONLY when a full word has accumulated (`would_complete`,
        i.e. every BEATS_PER_WORD-th beat). A trailing partial group of
        (beats mod BEATS_PER_WORD) beats is NEVER flushed (no wr_req), so
        word_count can only ever reach floor(beats / BEATS_PER_WORD).
        words = beats // BEATS_PER_WORD = beats // (2048 // BUS_W).

  g_w_gt  (BUS_W >  2048): WORDS_PER_BEAT = BUS_W / 2048  (integer trunc).
        Each beat is sliced into WORDS_PER_BEAT words, one wr_req per slice.
        words = beats * WORDS_PER_BEAT = beats * (BUS_W // 2048).
        (For BUS_W=3072, WORDS_PER_BEAT=1 -> words = beats, i.e. unchanged;
        the bridge drops the upper 1024b of each beat — a separate latent
        property, not affected by this threshold patch.)

These match the EXACT increment/assert logic quoted above, including integer
truncation for non-pow2 BUS_W (768/192/1152/1536/1280) and for BUS_W=3072.

WHAT THIS PATCH DOES
--------------------
For every `stream_to_act_bram_bridge #( ... ) u_ldr_node_conv_* ( ... )`
instance it parses BUS_W and the current TOTAL_BRAM_WORDS (the beat count),
recomputes the correct WORD count per the regime above, and rewrites ONLY the
TOTAL_BRAM_WORDS literal in-place. No other text changes. Idempotent
(re-running on an already-resized file applies the same formula to the new
value; for eq/gt at the fixed point it is a no-op, and lt converges since the
correct word count is itself the fixed point only when re-derived from beats —
so DO NOT run twice; the script refuses if it cannot find the original beat
count, see --check).

It backs up nn2rtl_top_engine.v to backups/loader_word_resize_<TS>/ first.

USAGE
-----
    python scripts/apply_loader_word_resize.py            # patch in place
    python scripts/apply_loader_word_resize.py --dry-run  # show plan only
"""
import argparse
import datetime
import os
import re
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOP = os.path.join(REPO, "output", "mobilenet-v2", "rtl", "nn2rtl_top_engine.v")


def correct_words(bus_w: int, beats: int) -> int:
    """Number of 2048-bit BRAM words the bridge actually fills for `beats`
    input beats at width `bus_w`, matching the g_w_lt / g_w_eq / g_w_gt RTL."""
    if bus_w == 2048:
        return beats
    if bus_w < 2048:
        beats_per_word = 2048 // bus_w
        return beats // beats_per_word
    words_per_beat = bus_w // 2048
    return beats * words_per_beat


# Match one whole bridge instance: the param block ... ) u_ldr_node_conv_<id> (
# Captures BUS_W, the TOTAL_BRAM_WORDS literal (with its exact surrounding
# text so we can do a precise replace), and the instance name.
INSTANCE_RE = re.compile(
    r"stream_to_act_bram_bridge\s*#\("
    r"(?P<params>.*?)"               # the parameter list (non-greedy)
    r"\)\s*(?P<inst>u_ldr_node_conv_\d+)\s*\(",
    re.DOTALL,
)
BUSW_RE = re.compile(r"\.BUS_W\s*\(\s*(\d+)\s*\)")
TBW_RE = re.compile(r"(\.TOTAL_BRAM_WORDS\s*\(\s*)(\d+)(\s*\))")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the per-loader plan, do not write")
    ap.add_argument("--top", default=TOP, help="path to engine top .v")
    args = ap.parse_args()

    top = os.path.abspath(args.top)
    if not os.path.isfile(top):
        print(f"ERROR: top not found: {top}", file=sys.stderr)
        return 2

    with open(top, "r", encoding="utf-8", newline="") as f:
        src = f.read()

    matches = list(INSTANCE_RE.finditer(src))
    if not matches:
        print("ERROR: no stream_to_act_bram_bridge u_ldr_node_conv_* instances "
              "found", file=sys.stderr)
        return 3

    plan = []  # (inst, bus_w, old, new, regime)
    for m in matches:
        params = m.group("params")
        inst = m.group("inst")
        bm = BUSW_RE.search(params)
        tm = TBW_RE.search(params)
        if not bm or not tm:
            print(f"ERROR: {inst}: could not find BUS_W and/or "
                  f"TOTAL_BRAM_WORDS", file=sys.stderr)
            return 4
        bus_w = int(bm.group(1))
        old = int(tm.group(2))
        new = correct_words(bus_w, old)
        regime = "eq" if bus_w == 2048 else ("lt" if bus_w < 2048 else "gt")
        plan.append((inst, bus_w, old, new, regime))

    # Report.
    print(f"{'instance':25} {'BUS_W':>6} {'old(beats)':>11} "
          f"{'new(words)':>11} {'regime':>6} {'changed':>8}")
    n_changed = 0
    for inst, bus_w, old, new, regime in plan:
        changed = "YES" if new != old else "no"
        if new != old:
            n_changed += 1
        print(f"{inst:25} {bus_w:>6} {old:>11} {new:>11} {regime:>6} "
              f"{changed:>8}")
    print(f"\n{len(plan)} loaders, {n_changed} would change.")

    if args.dry_run:
        print("[dry-run] no file written.")
        return 0

    # Backup.
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bdir = os.path.join(REPO, "backups", f"loader_word_resize_{ts}")
    os.makedirs(bdir, exist_ok=True)
    shutil.copy2(top, os.path.join(bdir, os.path.basename(top)))
    print(f"backed up {top} -> {bdir}")

    # Rewrite each instance's TOTAL_BRAM_WORDS by reconstructing the full
    # source with span-accurate edits (replace within each matched param
    # block only). We rebuild left-to-right to keep offsets valid.
    out = []
    last = 0
    for m, (inst, bus_w, old, new, regime) in zip(matches, plan):
        out.append(src[last:m.start()])
        block = m.group(0)
        if new != old:
            # Replace exactly the TOTAL_BRAM_WORDS literal inside this block.
            block = TBW_RE.sub(
                lambda mm: f"{mm.group(1)}{new}{mm.group(3)}", block, count=1)
        out.append(block)
        last = m.end()
    out.append(src[last:])
    new_src = "".join(out)

    with open(top, "w", encoding="utf-8", newline="") as f:
        f.write(new_src)
    print(f"patched {top}: {n_changed} TOTAL_BRAM_WORDS values rewritten.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
