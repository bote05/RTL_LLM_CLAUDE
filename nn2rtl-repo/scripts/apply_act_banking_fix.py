#!/usr/bin/env python3
"""Act-banking fix (workflow wf_794c851a, adversarial 3/3 upheld) — numeric edits.
The g_w_lt branch rewrite (1 beat -> 1 zero-extended word) is applied separately via Edit.
Here: (1) set TOTAL_BRAM_WORDS = H*W (beat count) for the 14 BPW>1 g_w_lt loaders,
(2) ldr1 (conv_816) BRAM_BASE_ADDR 4096->12544 (disjoint 1-px/word window for D1),
(3) act_unified_mem DEPTH 24576->25600 (D0 in[0..12543] + D1 in[12544..25087]),
(4) scheduler act_in D1 4096->12544 and act_out D0 4096->12544 (D0-out region == D1-in,
    so the engine's direct act_out write and ldr1's write coincide; D1 only reads its low
    IC=16 bytes which are bit-identical; D1 act_out=0 overwrites the consumed D0-in region).
On-disk, never regenerate. Idempotent-ish (asserts).
"""
import re, pathlib
TOP = pathlib.Path("output/mobilenet-v2/rtl/nn2rtl_top_engine.v")
SCH = pathlib.Path("output/mobilenet-v2/rtl/nn2rtl_scheduler.v")

# node -> new TOTAL_BRAM_WORDS (= H*W)
TOTAL = {814:12544, 816:12544, 820:3136, 822:3136, 828:3136,
         834:784, 840:784, 846:784, 852:196, 858:196, 864:196, 870:196, 882:196, 888:196}

def main():
    s = TOP.read_text(); n=0
    for node, newtot in TOTAL.items():
        idx = s.find("u_ldr_node_conv_%d (" % node)
        assert idx>0, "loader %d not found"%node
        st = s.rfind("stream_to_act_bram_bridge", 0, idx); en = s.find(");", idx)+2
        blk = s[st:en]; orig = blk
        blk = re.sub(r"\.TOTAL_BRAM_WORDS\(\d+\)", ".TOTAL_BRAM_WORDS(%d)"%newtot, blk, count=1)
        if node == 816:
            blk = re.sub(r"\.BRAM_BASE_ADDR\(\d+\)", ".BRAM_BASE_ADDR(12544)", blk, count=1)
        assert blk != orig, "no change to loader %d"%node
        s = s[:st]+blk+s[en:]; n+=1
        extra = " + BASE->12544" if node==816 else ""
        print("  u_ldr_node_conv_%d: TOTAL_BRAM_WORDS->%d%s" % (node, newtot, extra))
    # act mem DEPTH
    assert s.count(".DEPTH(24576)")==1, "expected 1 .DEPTH(24576), got %d"%s.count(".DEPTH(24576)")
    s = s.replace(".DEPTH(24576)", ".DEPTH(25600)", 1)
    print("  act_unified_mem DEPTH 24576->25600")
    TOP.write_text(s)

    # scheduler: act_in D1 4096->12544, act_out D0 4096->12544
    c = SCH.read_text()
    # act_in_base_word_rom 6'd1: 4096 -> 12544
    c2, k1 = re.subn(r"(act_in_base_word_rom\s*=\s*16'd)4096(;\s*//[^\n]*)?(?=\n)", r"\g<1>4096\2", c)  # placeholder no-op
    # do scoped replacements by line content
    def repl_rom(text, rom, idx, newval):
        # find the rom's case line: "6'd<idx>: <rom> = 16'd<old>;"
        pat = re.compile(r"(6'd%d:\s*%s\s*=\s*16'd)(\d+)(;)" % (idx, rom))
        m = pat.search(text); assert m, "%s 6'd%d not found"%(rom,idx)
        old = m.group(2)
        return text[:m.start()] + m.group(1) + str(newval) + m.group(3) + text[m.end():], old
    c, oin = repl_rom(c, "act_in_base_word_rom", 1, 12544)
    print("  scheduler act_in_base_word_rom[D1] %s->12544"%oin)
    c, oout = repl_rom(c, "act_out_base_word_rom", 0, 12544)
    print("  scheduler act_out_base_word_rom[D0] %s->12544"%oout)
    SCH.write_text(c)
    print("DONE: %d loaders + DEPTH + 2 scheduler ROMs."%n)

if __name__=="__main__":
    main()
