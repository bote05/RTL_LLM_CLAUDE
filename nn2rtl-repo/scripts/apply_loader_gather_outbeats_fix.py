#!/usr/bin/env python3
"""FIX: the 6 loader-feed retile_gathers (br_ldr22..br_ldr32) had OUT_BEATS set to
the loader's WORDS-PER-BEAT (BUS_W/2048 = 3 for 6144, 4 for 8192). But OUT_BEATS is
the number of OUTPUT BEATS the gather emits per gathered buffer (= per position). The
loader (stream_to_act_bram_bridge, g_w_gt) ALREADY splits each single BUS_W-bit beat
into BUS_W/2048 act-BRAM words. So the gather must emit exactly ONE beat per position:
OUT_BEATS=1, OUT_W=6144/8192 (= FULL_W zero-padded to the loader bus width).

With the buggy OUT_BEATS=3/4 the gather emitted 3/4 beats/position (beat0 = real
4608/7680 bits zero-padded, the rest ALL-ZERO), so the loader wrote 3x/4x too many
words -> filled its TOTAL_BRAM_WORDS cap at ~1/3 of the positions with mostly-zero
garbage -> 'loaded' asserted early (wc overshoot 588->594) -> downstream backpressure
-> the 576/960 engine output bridge wedged in S_WAIT_DRAIN. Root cause of the
dispatch-21 deadlock.

Scoped to 'u_br_ldr' instance lines only. Idempotent.
"""
import re, pathlib
TOP = pathlib.Path("output/mobilenet-v2/rtl/nn2rtl_top_engine.v")

def fix(s):
    n = 0
    out = []
    for line in s.splitlines(keepends=True):
        if "u_br_ldr" in line and "retile_gather" in line and ".OUT_BEATS(" in line:
            new = re.sub(r"\.OUT_BEATS\(\d+\)", ".OUT_BEATS(1)", line, count=1)
            if new != line:
                n += 1
                # show before/after instance + old beats
                ob = re.search(r"\.OUT_BEATS\((\d+)\)", line).group(1)
                inst = re.search(r"u_br_ldr\d+", line).group(0)
                print("  %s: OUT_BEATS %s -> 1" % (inst, ob))
            out.append(new)
        else:
            out.append(line)
    return "".join(out), n

def main():
    s = TOP.read_text()
    s2, n = fix(s)
    assert n == 6, "expected 6 loader gathers, fixed %d (already patched? check)" % n
    TOP.write_text(s2)
    print("\nFixed %d loader-feed gathers to OUT_BEATS=1." % n)

if __name__ == "__main__":
    main()
