#!/usr/bin/env python3
"""BYTE-EXACT fix for the high-OC residual-add + tail (workflow wf_d91d632f, adversarially verified).

ROOT CAUSE: 5 project-conv engine_output_bridge instances feeding high-OC residual adds were left at
the default DATA_W=256 (OUT_KIND=0 g_slice -> emits only the first 32 channels/position); the downstream
add reads [767:0]/[1279:0] from a [255:0] wire -> channels 32..OC-1 ZEROED -> corrupt residual stream ->
propagates to GAP/Gemm -> 991/1000 logit mismatch. Plus the coupled tail (conv_910 OC=320 bridge still
OUT_KIND=0 DATA_W=256, and ldr27/ldr33 input loaders under-sized BUS_W=256). The byte-correct contract
(used by every OC<=256 low-OC conv) is DATA_W=OC*8; for OC>256 use OUT_KIND=2 g_flat + a 2-word loader.

8 EDITS (all on-disk, never regenerate the top):
 1-5. project bridges DATA_W 256->OC*8 + widen their data_out wire to [OC*8-1:0]:
        880,886 OC=96 ->768 ; 892,898,904 OC=160 ->1280.   (OUT_KIND stays 0: OC*8<2048, 1 beat/pos.)
 6.   conv_910 (OC=320) output bridge: DATA_W 256->2560, ADD .OC(320)/.OUT_KIND(2)/.POSITIONS(49)
        (g_flat gathers 2 engine beats -> 1 contiguous 2560b/pos) + wire node_conv_910_data_out ->[2559:0].
 7.   ldr27 (u_ldr_node_conv_894, conv_894 IC=160 input): BUS_W 256->1280, TOTAL 6->49,
        in_data node_conv_892_data_out -> node_conv_892_data_out[1279:0] (mirrors conv_900/906 loaders).
 8.   ldr33 (u_ldr_node_conv_912, conv_912 IC=320 input): BUS_W 256->4096, TOTAL 12->98,
        in_data node_conv_910_data_out -> {1536'b0, node_conv_910_data_out[2559:0]} (g_w_gt 2 words/pos:
        word0=ch0-255, word1={1536'b0,ch256-319}; mirrors the IC=384 conv_856 zero-pad precedent).
Idempotent; asserts each edit.
"""
import re, pathlib
TOP = pathlib.Path("output/mobilenet-v2/rtl/nn2rtl_top_engine.v")

BR = [(880,96),(886,96),(892,160),(898,160),(904,160)]   # (node, OC) project bridges feeding adds

def edit_bridge_dataw(s, node, oc):
    """Scoped: in u_engine_out_node_conv_<node> param block, DATA_W(256)->DATA_W(oc*8)."""
    idx = s.find("u_engine_out_node_conv_%d (" % node)
    assert idx>0, "bridge %d not found"%node
    bstart = s.rfind("engine_output_bridge", 0, idx)
    blk = s[bstart:idx]; orig=blk
    blk = blk.replace(".DATA_W(256)", ".DATA_W(%d)" % (oc*8), 1)
    assert blk!=orig, "no DATA_W(256) in bridge %d"%node
    return s[:bstart]+blk+s[idx:]

def edit_wire(s, node, width):
    """wire [255:0] node_conv_<node>_data_out -> [width-1:0]."""
    pat = "wire [255:0] node_conv_%d_data_out" % node
    assert pat in s, "wire for %d not [255:0] (already patched?)"%node
    return s.replace(pat, "wire [%d:0] node_conv_%d_data_out" % (width-1, node), 1)

def main():
    s = TOP.read_text()
    # 1-5: project bridges + wires
    for node,oc in BR:
        s = edit_bridge_dataw(s, node, oc)
        s = edit_wire(s, node, oc*8)
        print("  bridge %d: DATA_W->%d, wire->[%d:0]" % (node, oc*8, oc*8-1))
    # 6: conv_910 bridge DATA_W->2560 + add OC/OUT_KIND/POSITIONS + wire
    idx=s.find("u_engine_out_node_conv_910 (")
    bstart=s.rfind("engine_output_bridge",0,idx); blk=s[bstart:idx]; orig=blk
    blk=blk.replace(".DATA_W(256)", ".DATA_W(2560)",1)
    assert ".OC(" not in blk, "conv_910 already has OC?"
    # append OC/OUT_KIND/POSITIONS after NUM_DISPATCHES(34)
    blk=re.sub(r"(\.NUM_DISPATCHES\(34\))", r"\1,\n        .OC(320), .OUT_KIND(2), .POSITIONS(49)", blk, count=1)
    assert blk!=orig and ".OC(320)" in blk, "conv_910 param edit failed"
    s=s[:bstart]+blk+s[idx:]
    s=edit_wire(s, 910, 2560)
    print("  bridge 910: DATA_W->2560, +OC(320)/OUT_KIND(2)/POSITIONS(49), wire->[2559:0]")
    # 7: ldr27 (894)
    def edit_loader(s, node, busw, tot, indata_new):
        idx=s.find("u_ldr_node_conv_%d ("%node)
        lstart=s.rfind("stream_to_act_bram_bridge",0,idx); lend=s.find(");",idx)+2
        blk=s[lstart:lend]; orig=blk
        blk=blk.replace(".BUS_W(256)", ".BUS_W(%d)"%busw,1)
        blk=re.sub(r"\.TOTAL_BRAM_WORDS\(\d+\)", ".TOTAL_BRAM_WORDS(%d)"%tot, blk,1)
        blk=re.sub(r"\.in_data\([^)]*\)", ".in_data(%s)"%indata_new, blk,1)
        assert blk!=orig, "loader %d unchanged"%node
        return s[:lstart]+blk+s[lend:]
    s=edit_loader(s, 894, 1280, 49, "node_conv_892_data_out[1279:0]")
    print("  ldr27(894): BUS_W->1280, TOTAL->49, in_data->node_conv_892_data_out[1279:0]")
    # 8: ldr33 (912)
    s=edit_loader(s, 912, 4096, 98, "{1536'b0, node_conv_910_data_out[2559:0]}")
    print("  ldr33(912): BUS_W->4096, TOTAL->98, in_data->{1536'b0, node_conv_910_data_out[2559:0]}")
    TOP.write_text(s)
    print("DONE: 8 high-OC byte-exact edits applied.")

if __name__=="__main__":
    main()
