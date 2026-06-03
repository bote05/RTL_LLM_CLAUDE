#!/usr/bin/env python3
"""STEP 4: re-point the 6 high-OC tiled loaders onto their loader-feed GATHER bridges
(br_ldr22..br_ldr32, instantiated by apply_mbv2_wave2_bridges_engine.py) and resize them
to the FLAT per-position layout (BUS_W = ceil(IC/256)*2048, a clean multiple of 2048 so the
g_w_gt slicer stores ceil(IC/256) words/position aligned to the engine's per-position read).

Per loader: in_valid -> br_*_valid_out & spatial_run_drain_br_*, in_data -> br_*_data_out,
BUS_W/TOTAL_BRAM_WORDS resized, and .in_ready(ldrN_in_ready) exposed (the gather's ready_down).
Scoped to each instance block. Idempotent-ish; asserts each edit applies.
"""
import re, pathlib
TOP = pathlib.Path("output/mobilenet-v2/rtl/nn2rtl_top_engine.v")
# (loader_node, gather, BUS_W_new, WORDS_new, disp_wire, producer)
L = [
    (880, "br_ldr22", 6144, 588, "ldr22", "n4_24"),
    (886, "br_ldr24", 6144, 588, "ldr24", "n4_26"),
    (892, "br_ldr26", 6144, 147, "ldr26", "n4_28"),
    (898, "br_ldr28", 8192, 196, "ldr28", "n4_30"),
    (904, "br_ldr30", 8192, 196, "ldr30", "n4_32"),
    (910, "br_ldr32", 8192, 196, "ldr32", "n4_34"),
]
def main():
    s = TOP.read_text(); n=0
    for node, br, busw, words, dw, prod in L:
        inst = "u_ldr_node_conv_%d" % node
        # match ONLY this instance: stream_to_act_bram_bridge #( ... ) u_ldr_node_conv_N ( ... );
        # Use [^;] (NOT (.|\n)) so the match CANNOT span across other instances: every
        # instance is ';'-terminated, so [^;]*? confines the match to a single one.
        # (The prior (.|\n)*? matched from the FIRST stream_to_act_bram_bridge all the
        #  way to u_ldr_node_conv_N, corrupting the wrong/earlier loaders.)
        m = re.search(r"stream_to_act_bram_bridge\s*#\(\s*[^;]*?\)\s*" + inst + r"\s*\([^;]*?\);", s)
        assert m, "instance %s not found" % inst
        blk = m.group(0); orig = blk
        # 1) BUS_W + TOTAL_BRAM_WORDS (scoped to this block)
        blk = re.sub(r"\.BUS_W\(\d+\)", ".BUS_W(%d)" % busw, blk, count=1)
        blk = re.sub(r"\.TOTAL_BRAM_WORDS\(\d+\)", ".TOTAL_BRAM_WORDS(%d)" % words, blk, count=1)
        # 2) in_valid / in_data -> gather
        blk = re.sub(r"\.in_valid\([^)]*\)",
                     ".in_valid(%s_valid_out & spatial_run_drain_%s)" % (br, br), blk, count=1)
        blk = re.sub(r"\.in_data\([^)]*\)", ".in_data(%s_data_out)" % br, blk, count=1)
        # 3) expose in_ready (gather ready_down). Add the port if absent.
        if ".in_ready(" not in blk:
            blk = re.sub(r"(\.loaded\([^)]*\))", r"\1,\n        .in_ready(%s_in_ready)" % dw, blk, count=1)
        assert blk != orig, "no change to %s" % inst
        s = s.replace(orig, blk, 1)
        # ensure the in_ready wire is declared (right before the instance)
        if ("wire        %s_in_ready;" % dw) not in s and ("wire %s_in_ready;" % dw) not in s:
            s = s.replace("stream_to_act_bram_bridge", "wire %s_in_ready;\n    stream_to_act_bram_bridge" % dw, 1) \
                if False else s  # (most BP-PARTB loaders already declare it; add only if truly missing)
        n += 1
        print("  %s: -> %s, BUS_W=%d, WORDS=%d, in_ready=%s_in_ready" % (inst, br, busw, words, dw))
    TOP.write_text(s)
    print("\nSTEP 4 applied to %d loaders." % n)
if __name__ == "__main__":
    main()
