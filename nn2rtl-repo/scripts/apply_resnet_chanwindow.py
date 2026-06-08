#!/usr/bin/env python3
"""apply_resnet_chanwindow.py (2026-06-03)

ResNet-50 route-congestion fix (workflow wqvnlwjfk): switch the deep 3x3 convs
node_conv_284/292/298 (IC=512, MP_K=9) from the wide 36864-bit combinational
window_flat mux to line_buf_window's narrow per-channel chan_window_flat path.

Byte-exact for MP_K==KH*KW: conv_datapath_mp_k tap_at(k_group*9+i) ==
window_flat[(i*IC+k_group)*8] == chan_window_flat[i*8] with channel_select=k_group;
chan_window_flat is combinational (zero latency change). The wide mux (the LUT-
saturated route-congestion epicenter) is then never built (lbw EXPOSE_FULL_WINDOW=0).

Edits (idempotent, atomic-per-file, backed up):
  rtl_library/conv_datapath_mp_k.v  : +USE_CHAN_WINDOW param, +chan_window_flat input,
       +channel_select output, +CSEL_W/assign, generate-gated Stage-1 tap load + guard.
  output/rtl/node_conv_{284,292,298}.v : narrow wires + lbw EXPOSE_FULL_WINDOW(0) +
       channel_select/chan_window_flat wiring + dp USE_CHAN_WINDOW(1) + narrow ports.

USE_CHAN_WINDOW defaults 0 -> all ~38 other conv_datapath_mp_k instances byte-identical.
"""
import sys, os, time, shutil

REPO = r"C:\Users\User\Desktop\RTL_LLM_CLAUDE\nn2rtl-repo"
DP = os.path.join(REPO, "rtl_library", "conv_datapath_mp_k.v")
NODES = [os.path.join(REPO, "output", "rtl", f"node_conv_{n}.v") for n in (284, 292, 298)]

# ---- conv_datapath_mp_k.v edits ----
DP_EDITS = [
("    parameter integer WGT_BITS     = 4\n) (",
 "    parameter integer WGT_BITS     = 4,\n"
 "    // [RESNET-CHANWINDOW] 1 => read narrow per-channel chan_window_flat (one channel/cycle via\n"
 "    // channel_select=k_group) instead of slicing wide window_flat. Byte-identical for MP_K==KH*KW\n"
 "    // (compile guard below). Eliminates the KH*KW*IC*8 combinational window mux. Default 0 = legacy.\n"
 "    parameter integer USE_CHAN_WINDOW = 0\n) ("),

("    input  wire [KH*KW*IC*8-1:0]              window_flat,\n    input  wire                               start_mac,",
 "    input  wire [KH*KW*IC*8-1:0]              window_flat,\n"
 "    // [RESNET-CHANWINDOW] narrow per-channel window (used iff USE_CHAN_WINDOW); wide window_flat\n"
 "    // is then constant-0 (lbw EXPOSE_FULL_WINDOW=0) and unread.\n"
 "    input  wire [KH*KW*8-1:0]                 chan_window_flat,\n"
 "    output wire [((IC>1)?$clog2(IC):1)-1:0]   channel_select,\n"
 "    input  wire                               start_mac,"),

("    wire [$clog2(NUM_WIDE_WORDS+1)-1:0] weight_read_addr = oc_group * K_GROUPS + k_group;",
 "    wire [$clog2(NUM_WIDE_WORDS+1)-1:0] weight_read_addr = oc_group * K_GROUPS + k_group;\n"
 "    // [RESNET-CHANWINDOW] drive lbw channel_select = current k_group (== input-channel index when\n"
 "    // MP_K==KH*KW); combinational fanout of the same register that feeds tap_at.\n"
 "    localparam integer CSEL_W = (IC > 1) ? $clog2(IC) : 1;\n"
 "    // meaningful only when USE_CHAN_WINDOW=1 (MP_K==KH*KW => K_GROUPS==IC => k_group is exactly\n"
 "    // CSEL_W bits, in-range). Other instances (CSEL_W>k_group width) tie 0 to avoid SELRANGE.\n"
 "    generate\n"
 "    if (USE_CHAN_WINDOW != 0) begin : g_csel\n"
 "        assign channel_select = k_group[CSEL_W-1:0];\n"
 "    end else begin : g_csel_off\n"
 "        assign channel_select = {CSEL_W{1'b0}};\n"
 "    end\n"
 "    endgenerate"),

("    always @(posedge clk) begin\n"
 "        weight_word_q <= weights_wide[weight_read_addr];\n"
 "        for (ld_i = 0; ld_i < MP_K; ld_i = ld_i + 1)\n"
 "            tap_q[ld_i] <= $signed(tap_at(k_group * MP_K + ld_i));\n"
 "    end",
 "    // [RESNET-CHANWINDOW] Stage-1 tap source. USE_CHAN_WINDOW=0 (default, all other instances):\n"
 "    // unchanged tap_at(window_flat) slice. =1 (convs 284/292/298): read chan_window_flat[i] =\n"
 "    // window_flat[(i*IC+k_group)*8] (byte-identical for MP_K==KH*KW) so the wide mux is never built.\n"
 "    // Same FF stage -> zero latency change -> residual-add joins stay balanced.\n"
 "    generate\n"
 "    if (USE_CHAN_WINDOW != 0) begin : g_chan_window\n"
 "        initial if (MP_K != KH*KW) $fatal(1, \"conv_datapath_mp_k: USE_CHAN_WINDOW=1 requires MP_K==KH*KW\");\n"
 "        always @(posedge clk) begin\n"
 "            weight_word_q <= weights_wide[weight_read_addr];\n"
 "            for (ld_i = 0; ld_i < MP_K; ld_i = ld_i + 1)\n"
 "                tap_q[ld_i] <= $signed(chan_window_flat[ld_i*8 +: 8]);\n"
 "        end\n"
 "    end else begin : g_full_window\n"
 "        always @(posedge clk) begin\n"
 "            weight_word_q <= weights_wide[weight_read_addr];\n"
 "            for (ld_i = 0; ld_i < MP_K; ld_i = ld_i + 1)\n"
 "                tap_q[ld_i] <= $signed(tap_at(k_group * MP_K + ld_i));\n"
 "        end\n"
 "    end\n"
 "    endgenerate"),
]

# ---- node_conv_{284,292,298}.v edits (identical anchors across all three) ----
NODE_EDITS = [
("    wire [KH*KW*IC*8-1:0] window_flat;",
 "    wire [KH*KW*IC*8-1:0] window_flat;\n"
 "    // [RESNET-CHANWINDOW] narrow per-channel window path (collapses the 36864-bit mux)\n"
 "    wire [KH*KW*8-1:0] chan_window_flat_w;\n"
 "    wire [((IC>1)?$clog2(IC):1)-1:0] dp_channel_select;"),

("    line_buf_window #(.IC(IC),.IW(IW),.IH(IH),.KH(KH),.KW(KW),.PW(PW),.PH(PH)) lbw (",
 "    line_buf_window #(.IC(IC),.IW(IW),.IH(IH),.KH(KH),.KW(KW),.PW(PW),.PH(PH),.EXPOSE_FULL_WINDOW(0)) lbw ("),

("        .valid_in(lib_valid_in_w),.data_in(lib_data_in_w),.window_flat(window_flat));",
 "        .valid_in(lib_valid_in_w),.data_in(lib_data_in_w),.window_flat(window_flat),\n"
 "        .channel_select(dp_channel_select),.chan_window_flat(chan_window_flat_w));"),

("    conv_datapath_mp_k #(.IC(IC),.OC(OC),.KH(KH),.KW(KW),.K_TOTAL(K_TOTAL),.MP(MP),.WGT_BITS(3),",
 "    conv_datapath_mp_k #(.IC(IC),.OC(OC),.KH(KH),.KW(KW),.K_TOTAL(K_TOTAL),.MP(MP),.WGT_BITS(3),.USE_CHAN_WINDOW(1),"),

("        .clk(clk),.rst_n(rst_n),.window_flat(window_flat),\n        .start_mac(sched_output_fires),",
 "        .clk(clk),.rst_n(rst_n),.window_flat(window_flat),\n"
 "        .chan_window_flat(chan_window_flat_w),.channel_select(dp_channel_select),\n"
 "        .start_mac(sched_output_fires),"),
]


def apply_file(path, edits, bk):
    with open(path, "rb") as f:
        raw = f.read()
    crlf = b"\r\n" in raw
    text = raw.decode("utf-8")
    if crlf:
        text = text.replace("\r\n", "\n")
    # idempotency: if the marker is already present, skip
    if "[RESNET-CHANWINDOW]" in text:
        print(f"SKIP {os.path.basename(path)}: already has [RESNET-CHANWINDOW] marker")
        return True
    work = text
    for i, (old, new) in enumerate(edits):
        c = work.count(old)
        if c != 1:
            print(f"FAIL {os.path.basename(path)} edit#{i}: anchor count={c} (need 1): {old[:60]!r}")
            return False
        work = work.replace(old, new, 1)
    shutil.copy2(path, os.path.join(bk, os.path.basename(path)))
    out = work.replace("\n", "\r\n") if crlf else work
    with open(path, "wb") as f:
        f.write(out.encode("utf-8"))
    print(f"OK   {os.path.basename(path)}: applied {len(edits)} edits")
    return True


def main():
    bk = os.path.join(REPO, "backups", f"resnet_chanwindow_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(bk, exist_ok=True)
    ok = True
    ok &= apply_file(DP, DP_EDITS, bk)
    for n in NODES:
        ok &= apply_file(n, NODE_EDITS, bk)
    print("-" * 50)
    print("ALL OK" if ok else "FAILED (see above) - no partial files written for failed ones")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
