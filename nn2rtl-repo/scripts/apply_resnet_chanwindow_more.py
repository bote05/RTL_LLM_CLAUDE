#!/usr/bin/env python3
"""apply_resnet_chanwindow_more.py (2026-06-04)

Extend the proven chan_window mux-collapse (apply_resnet_chanwindow.py, which did the
IC=512 convs 284/292/298) to EVERY remaining eligible 3x3 spatial conv, to attack the
slice-saturation route congestion globally (the route is logic-bound, not BRAM-bound;
removing combinational window-mux logic frees slices device-wide).

Eligible = 3x3 (KH*KW=9), MP_K==9 (=> byte-exact collapse), AND instantiated standalone
in nn2rtl_top.v (the engine-dispatched 3x3s 246/254/260/266/272/278 have in_top=0, so
their files aren't synthesized -> collapsing them is a no-op; EXCLUDED). That leaves:
  stage1 IC=64 : node_conv_200, 208, 214
  stage2 IC=128: node_conv_220, 228, 234, 240   (incl. the literal hotspot convs 220/240)
=> 7 convs. All INT4 (no explicit .WGT_BITS -> module default 4).

Difference vs the INT3 template (284/292/298): those carry .WGT_BITS(3) on the dp
instantiation and inserted .USE_CHAN_WINDOW(1) right after it. INT4 convs OMIT .WGT_BITS,
so the dp instantiation line ends at .MP(MP), -> we insert .USE_CHAN_WINDOW(1) there.
The other 4 edits (narrow wires, lbw EXPOSE_FULL_WINDOW(0)+ports, dp ports) are identical.

Byte-exact for MP_K==KH*KW==9: chan_window_flat[i] = window_flat[(i*IC+k_group)*8] with
channel_select=k_group; same FF stage => ZERO latency change => residual-add joins stay
balanced. The conv_datapath_mp_k.v USE_CHAN_WINDOW param + guard are already present
(applied by apply_resnet_chanwindow.py); this script only wires the 7 node files.

Idempotent (skips files already carrying the [RESNET-CHANWINDOW] marker), atomic-per-file,
backed up to backups/resnet_chanwindow_more_<ts>/. Verify with the full e2e value harness
(npx tsx scripts/run_nn2rtl_top_value.ts 0 -> expect PASS mismatch=0) BEFORE any synth.
"""
import sys, os, time, shutil

REPO = r"C:\Users\User\Desktop\RTL_LLM_CLAUDE\nn2rtl-repo"
CONVS = (200, 208, 214, 220, 228, 234, 240)
NODES = [os.path.join(REPO, "output", "rtl", f"node_conv_{n}.v") for n in CONVS]

# INT4 node edits: identical to apply_resnet_chanwindow.py NODE_EDITS except edit #4
# (insert .USE_CHAN_WINDOW(1) after .MP(MP), since INT4 has no .WGT_BITS on the line).
NODE_EDITS = [
("    wire [KH*KW*IC*8-1:0] window_flat;",
 "    wire [KH*KW*IC*8-1:0] window_flat;\n"
 "    // [RESNET-CHANWINDOW] narrow per-channel window path (collapses the KH*KW*IC*8 mux)\n"
 "    wire [KH*KW*8-1:0] chan_window_flat_w;\n"
 "    wire [((IC>1)?$clog2(IC):1)-1:0] dp_channel_select;"),

("    line_buf_window #(.IC(IC),.IW(IW),.IH(IH),.KH(KH),.KW(KW),.PW(PW),.PH(PH)) lbw (",
 "    line_buf_window #(.IC(IC),.IW(IW),.IH(IH),.KH(KH),.KW(KW),.PW(PW),.PH(PH),.EXPOSE_FULL_WINDOW(0)) lbw ("),

("        .valid_in(lib_valid_in_w),.data_in(lib_data_in_w),.window_flat(window_flat));",
 "        .valid_in(lib_valid_in_w),.data_in(lib_data_in_w),.window_flat(window_flat),\n"
 "        .channel_select(dp_channel_select),.chan_window_flat(chan_window_flat_w));"),

# INT4 variant: line ends at .MP(MP), (no .WGT_BITS). Insert .USE_CHAN_WINDOW(1) here.
("    conv_datapath_mp_k #(.IC(IC),.OC(OC),.KH(KH),.KW(KW),.K_TOTAL(K_TOTAL),.MP(MP),",
 "    conv_datapath_mp_k #(.IC(IC),.OC(OC),.KH(KH),.KW(KW),.K_TOTAL(K_TOTAL),.MP(MP),.USE_CHAN_WINDOW(1),"),

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
    bk = os.path.join(REPO, "backups", f"resnet_chanwindow_more_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(bk, exist_ok=True)
    ok = True
    for n in NODES:
        ok &= apply_file(n, NODE_EDITS, bk)
    print("-" * 50)
    print(f"backups -> {bk}")
    print("ALL OK" if ok else "FAILED (see above) - failed files left unmodified")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
