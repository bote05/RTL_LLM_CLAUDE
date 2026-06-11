// tb_equiv.sv -- TAIL_PIPE equivalence harness (2026-06-09).
// Drives TWO conv_datapath_mp_k instances (TAIL_PIPE=0 reference vs TAIL_PIPE=1
// DUT) with identical random windows + real node_conv_244 weight/bias/scale
// mems (IC=512, OC=256, MP=16, MP_K=8, DSP_INPUT_PIPE=1 -- the live config).
// Checks per pixel:
//   1. data_out byte-identical at the two valid_out pulses,
//   2. DUT valid_out lands exactly 2*OC_PASSES cycles after the reference
//      (the documented +2/oc_pass cost; OC_PASSES=16 -> +32).
// Run (cwd = nn2rtl-repo):
//   <oss-cad-suite>/bin/verilator_bin.exe --binary --timing -Wno-fatal
//     -Wno-WIDTH --x-initial 0 verify/tail_pipe2_equiv/tb_equiv.sv
//     rtl_library/conv_datapath_mp_k.v -o tb_equiv
`timescale 1ns/1ps
module tb_equiv #(parameter integer TB_PIXELS = 64);
    localparam integer IC = 512, OC = 256, KH = 1, KW = 1;
    localparam integer MP = 16, MP_K = 8;
    localparam integer K_TOTAL   = IC*KH*KW;
    localparam integer OC_PASSES = (OC + MP - 1) / MP;
    localparam integer EXP_DELTA = 2 * OC_PASSES;   // +2 cycles per oc_pass

    reg clk = 0, rst_n = 0, start_mac = 0;
    always #5 clk = ~clk;

    reg  [KH*KW*IC*8-1:0] window_flat;
    wire [KH*KW*8-1:0]    chan_window_flat = {KH*KW*8{1'b0}};

    wire vo_ref, vo_dut, busy_ref, busy_dut;
    wire [OC*8-1:0] do_ref, do_dut;

    conv_datapath_mp_k #(
        .IC(IC), .OC(OC), .KH(KH), .KW(KW), .K_TOTAL(K_TOTAL),
        .MP(MP), .MP_K(MP_K), .DSP_INPUT_PIPE(1), .TAIL_PIPE(0),
        .WEIGHTS_PATH("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_244_weights_mp_k_8.hex"),
        .BIAS_PATH   ("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_244_bias.hex"),
        .SCALE_PATH  ("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_244_scale.mem")
    ) u_ref (.clk(clk), .rst_n(rst_n), .window_flat(window_flat),
             .chan_window_flat(chan_window_flat), .channel_select(),
             .start_mac(start_mac), .valid_out(vo_ref), .data_out(do_ref),
             .mac_busy(busy_ref));

    conv_datapath_mp_k #(
        .IC(IC), .OC(OC), .KH(KH), .KW(KW), .K_TOTAL(K_TOTAL),
        .MP(MP), .MP_K(MP_K), .DSP_INPUT_PIPE(1), .TAIL_PIPE(1),
        .WEIGHTS_PATH("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_244_weights_mp_k_8.hex"),
        .BIAS_PATH   ("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_244_bias.hex"),
        .SCALE_PATH  ("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_244_scale.mem")
    ) u_dut (.clk(clk), .rst_n(rst_n), .window_flat(window_flat),
             .chan_window_flat(chan_window_flat), .channel_select(),
             .start_mac(start_mac), .valid_out(vo_dut), .data_out(do_dut),
             .mac_busy(busy_dut));

    integer px, b, mism, t_ref, t_dut, cyc, bad_delta;
    integer seed;
    reg [OC*8-1:0] cap_ref, cap_dut;
    reg got_ref, got_dut;

    always @(posedge clk) cyc <= cyc + 1;

    initial begin
        seed = 32'hC0FFEE42; mism = 0; bad_delta = 0; cyc = 0;
        repeat (4) @(negedge clk);
        rst_n = 1;
        repeat (2) @(negedge clk);

        for (px = 0; px < TB_PIXELS; px = px + 1) begin
            for (b = 0; b < KH*KW*IC*8; b = b + 32)
                window_flat[b +: 32] = $random(seed);
            @(negedge clk); start_mac = 1;
            @(negedge clk); start_mac = 0;
            got_ref = 0; got_dut = 0; t_ref = 0; t_dut = 0;
            while (!(got_ref && got_dut)) begin
                @(negedge clk);
                if (vo_ref && !got_ref) begin cap_ref = do_ref; t_ref = cyc; got_ref = 1; end
                if (vo_dut && !got_dut) begin cap_dut = do_dut; t_dut = cyc; got_dut = 1; end
            end
            if (cap_ref !== cap_dut) begin
                mism = mism + 1;
                $display("[tb] px=%0d DATA MISMATCH", px);
            end
            if (t_dut - t_ref != EXP_DELTA) begin
                bad_delta = bad_delta + 1;
                $display("[tb] px=%0d latency delta=%0d (expected %0d)", px, t_dut - t_ref, EXP_DELTA);
            end
            // let both fully drain before the next pixel
            @(negedge clk);
            while (busy_ref || busy_dut) @(negedge clk);
        end
        $display("[tb][equiv][summary] pixels=%0d mismatch_pixels=%0d bad_latency_deltas=%0d expected_delta=%0d result=%s",
                 TB_PIXELS, mism, bad_delta, EXP_DELTA,
                 (mism == 0 && bad_delta == 0) ? "PASS" : "FAIL");
        $finish;
    end
endmodule
