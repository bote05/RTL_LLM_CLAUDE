#!/usr/bin/env python3
"""384-block byte-exact fix, LOADER half (the bridge half = OUT_KIND 0->2 done via Edit).

After OUT_KIND=2 (g_flat) the 4 expand bridges (852/858/864/870) emit ONE contiguous
3072-bit beat/pixel (EXPECTED_TILES=POSITIONS=196), matching node_conv_854/860/... depthwise
contract ("3072b in/out, 1 beat per pixel"). The depthwise outputs 196 beats/frame; the
post-DW relu (n4_16/18/20/22) feeds the project's input loader (ldr 856/862/868/874).

OUT_KIND=2 HALVES the beats reaching the loader vs the old g_legacy (which emitted 392
zero-padded beats). So the loaders must change from BUS_W=3072 (WORDS_PER_BEAT=3072/2048=1,
lossy: writes only low 2048b = 256ch, drops 128ch; AND would under-fill at 196<392 -> wedge)
to BUS_W=4096 (WORDS_PER_BEAT=2): 196 beats x 2 words = 392 = TOTAL. in_data is zero-padded
3072->4096 so the 2 act-BRAM words are word0=ch0-255, word1={1024'b0, ch256-383} -- exactly
the engine's ceil(384/256)=2-word/pixel read layout. TOTAL_BRAM_WORDS stays 392.

On-disk surgical patch (never regenerate). Idempotent.
"""
import re, pathlib
TOP = pathlib.Path("output/mobilenet-v2/rtl/nn2rtl_top_engine.v")
LOADERS = [(856,"n4_16"),(862,"n4_18"),(868,"n4_20"),(874,"n4_22")]

def main():
    s = TOP.read_text(); n=0
    for node, prod in LOADERS:
        idx = s.find("u_ldr_node_conv_%d (" % node)
        assert idx>0, "loader %d not found" % node
        start = s.rfind("stream_to_act_bram_bridge", 0, idx)
        end = s.find(");", idx)+2
        blk = s[start:end]; orig = blk
        # BUS_W 3072 -> 4096
        blk = blk.replace(".BUS_W(3072)", ".BUS_W(4096)", 1)
        # in_data(n4_XX_data_out) -> in_data({1024'b0, n4_XX_data_out})  (zero-pad 3072->4096)
        pad = "{1024'b0, %s_data_out}" % prod
        if pad not in blk:
            blk = re.sub(r"\.in_data\(\s*%s_data_out\s*\)" % re.escape(prod),
                         ".in_data(%s)" % pad, blk, count=1)
        assert blk != orig, "no change to loader %d (already patched?)" % node
        assert ".BUS_W(4096)" in blk and pad in blk, "incomplete patch on %d" % node
        s = s[:start] + blk + s[end:]
        n += 1
        print("  u_ldr_node_conv_%d: BUS_W 3072->4096, in_data zero-padded ({1024'b0, %s_data_out})" % (node, prod))
    TOP.write_text(s)
    print("DONE: %d 384-block loaders fixed." % n)

if __name__ == "__main__":
    main()
