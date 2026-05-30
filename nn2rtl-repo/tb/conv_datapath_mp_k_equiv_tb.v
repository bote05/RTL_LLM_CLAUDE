// Equivalence TB: conv_datapath (original, serialized MAC) vs
// conv_datapath_mp_k (parallel MP and parallel MP_K).
//
// Drives the same window_flat into both DUTs. Compares output values
// pixel-by-pixel (modulo the latency offset; new DUT is much faster).
// Override shape params with -D{IC,OC,KH,KW,MP,MP_K,SCALE_MULT,SCALE_SHIFT}_OVR.

`timescale 1ns / 1ps

module conv_datapath_mp_k_equiv_tb;

`ifdef IC_OVR
    localparam integer IC = `IC_OVR;
`else
    localparam integer IC = 64;
`endif
`ifdef OC_OVR
    localparam integer OC = `OC_OVR;
`else
    localparam integer OC = 64;
`endif
`ifdef KH_OVR
    localparam integer KH = `KH_OVR;
`else
    localparam integer KH = 3;
`endif
`ifdef KW_OVR
    localparam integer KW = `KW_OVR;
`else
    localparam integer KW = 3;
`endif
`ifdef MP_OVR
    localparam integer MP = `MP_OVR;
`else
    localparam integer MP = 4;
`endif
`ifdef MP_K_OVR
    localparam integer MP_K = `MP_K_OVR;
`else
    localparam integer MP_K = 9;
`endif
`ifdef SCALE_MULT_OVR
    localparam integer SCALE_MULT = `SCALE_MULT_OVR;
`else
    localparam integer SCALE_MULT = 23777;
`endif
`ifdef SCALE_SHIFT_OVR
    localparam integer SCALE_SHIFT = `SCALE_SHIFT_OVR;
`else
    localparam integer SCALE_SHIFT = 21;
`endif
    localparam integer K_TOTAL    = IC * KH * KW;
    localparam integer MAX_PIXELS = 8;

    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0;

    reg start_mac_n = 0;
    reg start_mac_w = 0;
    reg [KH*KW*IC*8-1:0] window_flat;

    wire           narrow_valid_out;
    wire [OC*8-1:0] narrow_data_out;
    wire           mpk_valid_out;
    wire [OC*8-1:0] mpk_data_out;

    reg [8*256-1:0] weights_narrow_path;
    reg [8*256-1:0] weights_mpk_path;
    reg [8*256-1:0] bias_path;

    conv_datapath #(
        .IC(IC), .OC(OC), .KH(KH), .KW(KW), .K_TOTAL(K_TOTAL), .MP(MP),
        .SCALE_MULT(SCALE_MULT), .SCALE_SHIFT(SCALE_SHIFT)
    ) dut_n (
        .clk(clk), .rst_n(rst_n),
        .window_flat(window_flat),
        .start_mac(start_mac_n),
        .valid_out(narrow_valid_out),
        .data_out(narrow_data_out),
        .mac_busy()
    );

    conv_datapath_mp_k #(
        .IC(IC), .OC(OC), .KH(KH), .KW(KW), .K_TOTAL(K_TOTAL),
        .MP(MP), .MP_K(MP_K),
        .SCALE_MULT(SCALE_MULT), .SCALE_SHIFT(SCALE_SHIFT)
    ) dut_w (
        .clk(clk), .rst_n(rst_n),
        .window_flat(window_flat),
        .start_mac(start_mac_w),
        .valid_out(mpk_valid_out),
        .data_out(mpk_data_out),
        .mac_busy()
    );

    initial begin
        if (!$value$plusargs("WEIGHTS_NARROW=%s", weights_narrow_path)) begin
            $display("[mpk] FATAL: +WEIGHTS_NARROW=<path> required");
            $finish;
        end
        if (!$value$plusargs("WEIGHTS_MPK=%s", weights_mpk_path)) begin
            $display("[mpk] FATAL: +WEIGHTS_MPK=<path> required");
            $finish;
        end
        if (!$value$plusargs("BIAS=%s", bias_path)) begin
            $display("[mpk] FATAL: +BIAS=<path> required");
            $finish;
        end
        $display("[mpk] narrow weights: %0s", weights_narrow_path);
        $display("[mpk] mp_k weights  : %0s", weights_mpk_path);
        $display("[mpk] bias          : %0s", bias_path);
        $readmemh(weights_narrow_path, dut_n.weights);
        $readmemh(weights_mpk_path,    dut_w.weights_wide);
        $readmemh(bias_path,           dut_n.biases);
        $readmemh(bias_path,           dut_w.biases);
    end

    reg [OC*8-1:0] narrow_outputs [0:MAX_PIXELS-1];
    reg [OC*8-1:0] mpk_outputs    [0:MAX_PIXELS-1];
    integer narrow_count = 0;
    integer mpk_count    = 0;

    always @(posedge clk) begin
        if (narrow_valid_out && narrow_count < MAX_PIXELS) begin
            narrow_outputs[narrow_count] <= narrow_data_out;
            narrow_count <= narrow_count + 1;
        end
        if (mpk_valid_out && mpk_count < MAX_PIXELS) begin
            mpk_outputs[mpk_count] <= mpk_data_out;
            mpk_count <= mpk_count + 1;
        end
    end

    integer test_seed = 32'hC0FFEE;
    integer pix_i, byte_i;
    integer mismatches = 0;
    integer iter;

    initial begin
        rst_n = 0;
        window_flat = 0;
        @(posedge clk); @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        $display("[mpk] starting %0d-pixel equivalence run (MP=%0d, MP_K=%0d)", MAX_PIXELS, MP, MP_K);

        for (pix_i = 0; pix_i < MAX_PIXELS; pix_i = pix_i + 1) begin
            for (byte_i = 0; byte_i < KH*KW*IC; byte_i = byte_i + 1) begin
                test_seed = (test_seed * 32'd1103515245 + 32'd12345) & 32'h7fffffff;
                window_flat[byte_i*8 +: 8] = test_seed[7:0];
            end

            @(posedge clk);
            start_mac_n <= 1; start_mac_w <= 1;
            @(posedge clk);
            start_mac_n <= 0; start_mac_w <= 0;

            iter = 0;
            while ((narrow_count <= pix_i || mpk_count <= pix_i)
                   && iter < 1000000) begin
                @(posedge clk);
                iter = iter + 1;
            end
            if (narrow_count <= pix_i || mpk_count <= pix_i) begin
                $display("[mpk] TIMEOUT at pixel %0d: narrow=%0d mpk=%0d",
                         pix_i, narrow_count, mpk_count);
                $finish;
            end
        end

        $display("[mpk] all %0d pixels captured; comparing", MAX_PIXELS);
        for (pix_i = 0; pix_i < MAX_PIXELS; pix_i = pix_i + 1) begin
            if (narrow_outputs[pix_i] !== mpk_outputs[pix_i]) begin
                mismatches = mismatches + 1;
                if (mismatches <= 4) begin
                    $display("[mpk] MISMATCH pixel %0d:", pix_i);
                    $display("    narrow = %h", narrow_outputs[pix_i]);
                    $display("    mp_k   = %h", mpk_outputs[pix_i]);
                end
            end
        end

        if (mismatches == 0) begin
            $display("[mpk] PASS: %0d/%0d pixels byte-equal", MAX_PIXELS, MAX_PIXELS);
        end else begin
            $display("[mpk] FAIL: %0d/%0d pixels mismatched", mismatches, MAX_PIXELS);
        end
        $finish;
    end

    initial begin
        #500000000;
        $display("[mpk] WATCHDOG at sim time %0t", $time);
        $finish;
    end

endmodule
