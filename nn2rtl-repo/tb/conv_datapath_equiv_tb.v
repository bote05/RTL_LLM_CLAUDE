// conv_datapath equivalence TB
//
// Drives the SAME window_flat into both conv_datapath (original) and
// conv_datapath_parallel (new). Captures each valid_out cycle's data_out.
// Asserts byte-equality between the two captured streams.
//
// The original and parallel datapaths produce outputs at different cycles
// (parallel is ~MP× faster), so we compare on the N-th valid_out pulse
// of each, not at fixed cycles.
//
// Build:
//   iverilog -g2012 -o /tmp/equiv_tb \
//     tb/conv_datapath_equiv_tb.v \
//     rtl_library/conv_datapath.v \
//     rtl_library/conv_datapath_parallel.v
//   vvp /tmp/equiv_tb +WEIGHTS_NARROW=<path> +WEIGHTS_WIDE=<path> +BIAS=<path>

`timescale 1ns / 1ps

module conv_datapath_equiv_tb;

    // Layer params; defaults to node_conv_196 (7x7 stem). Override via
    // `define IC_OVR / OC_OVR / KH_OVR / KW_OVR / MP_OVR / SCALE_MULT_OVR /
    // SCALE_SHIFT_OVR at compile time for different shapes.
`ifdef IC_OVR
    localparam integer IC          = `IC_OVR;
`else
    localparam integer IC          = 3;
`endif
`ifdef OC_OVR
    localparam integer OC          = `OC_OVR;
`else
    localparam integer OC          = 64;
`endif
`ifdef KH_OVR
    localparam integer KH          = `KH_OVR;
`else
    localparam integer KH          = 7;
`endif
`ifdef KW_OVR
    localparam integer KW          = `KW_OVR;
`else
    localparam integer KW          = 7;
`endif
`ifdef MP_OVR
    localparam integer MP          = `MP_OVR;
`else
    localparam integer MP          = 8;
`endif
`ifdef SCALE_MULT_OVR
    localparam integer SCALE_MULT  = `SCALE_MULT_OVR;
`else
    localparam integer SCALE_MULT  = 11709;
`endif
`ifdef SCALE_SHIFT_OVR
    localparam integer SCALE_SHIFT = `SCALE_SHIFT_OVR;
`else
    localparam integer SCALE_SHIFT = 23;
`endif
    localparam integer K_TOTAL     = IC * KH * KW;
    localparam integer OC_PASSES   = (OC + MP - 1) / MP;
    localparam integer MAX_PIXELS  = 16;   // enough to exercise multiple frames

    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0;

    reg start_mac_n = 0;
    reg start_mac_w = 0;
    reg [KH*KW*IC*8-1:0] window_flat;

    wire           narrow_valid_out;
    wire [OC*8-1:0] narrow_data_out;
    wire           parallel_valid_out;
    wire [OC*8-1:0] parallel_data_out;

    // Read CLI plusargs for weight paths.
    reg [8*256-1:0] weights_narrow_path;
    reg [8*256-1:0] weights_wide_path;
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

    conv_datapath_parallel #(
        .IC(IC), .OC(OC), .KH(KH), .KW(KW), .K_TOTAL(K_TOTAL), .MP(MP),
        .SCALE_MULT(SCALE_MULT), .SCALE_SHIFT(SCALE_SHIFT)
    ) dut_w (
        .clk(clk), .rst_n(rst_n),
        .window_flat(window_flat),
        .start_mac(start_mac_w),
        .valid_out(parallel_valid_out),
        .data_out(parallel_data_out),
        .mac_busy()
    );

    // Load weights into both DUTs via $readmemh on cli plusargs.
    initial begin
        if (!$value$plusargs("WEIGHTS_NARROW=%s", weights_narrow_path)) begin
            $display("[equiv] FATAL: +WEIGHTS_NARROW=<path> required");
            $finish;
        end
        if (!$value$plusargs("WEIGHTS_WIDE=%s", weights_wide_path)) begin
            $display("[equiv] FATAL: +WEIGHTS_WIDE=<path> required");
            $finish;
        end
        if (!$value$plusargs("BIAS=%s", bias_path)) begin
            $display("[equiv] FATAL: +BIAS=<path> required");
            $finish;
        end
        $display("[equiv] loading narrow weights: %0s", weights_narrow_path);
        $display("[equiv] loading wide weights  : %0s", weights_wide_path);
        $display("[equiv] loading bias          : %0s", bias_path);
        // Hierarchical $readmemh into each DUT's internal storage.
        $readmemh(weights_narrow_path, dut_n.weights);
        $readmemh(weights_wide_path,   dut_w.weights_wide);
        $readmemh(bias_path,           dut_n.biases);
        $readmemh(bias_path,           dut_w.biases);
    end

    // Output capture buffers (one entry per output pixel).
    reg [OC*8-1:0] narrow_outputs   [0:MAX_PIXELS-1];
    reg [OC*8-1:0] parallel_outputs [0:MAX_PIXELS-1];
    integer narrow_count   = 0;
    integer parallel_count = 0;

    always @(posedge clk) begin
        if (narrow_valid_out && narrow_count < MAX_PIXELS) begin
            narrow_outputs[narrow_count] <= narrow_data_out;
            narrow_count <= narrow_count + 1;
        end
        if (parallel_valid_out && parallel_count < MAX_PIXELS) begin
            parallel_outputs[parallel_count] <= parallel_data_out;
            parallel_count <= parallel_count + 1;
        end
    end

    integer test_seed = 32'hC0FFEE;
    integer pix_i, byte_i;
    integer mismatches = 0;
    integer iter;

    // Drive a deterministic random window_flat per pixel, fire both DUTs in
    // parallel, wait until BOTH have emitted one output, advance to next.
    initial begin
        rst_n = 0;
        window_flat = 0;
        @(posedge clk); @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        $display("[equiv] starting %0d-pixel equivalence run", MAX_PIXELS);

        for (pix_i = 0; pix_i < MAX_PIXELS; pix_i = pix_i + 1) begin
            // Deterministic pseudo-random window for this pixel.
            // Use a simple LCG so the byte stream is reproducible.
            for (byte_i = 0; byte_i < KH*KW*IC; byte_i = byte_i + 1) begin
                test_seed = (test_seed * 32'd1103515245 + 32'd12345) & 32'h7fffffff;
                window_flat[byte_i*8 +: 8] = test_seed[7:0];
            end

            // Fire both DUTs at the same time.
            @(posedge clk);
            start_mac_n <= 1; start_mac_w <= 1;
            @(posedge clk);
            start_mac_n <= 0; start_mac_w <= 0;

            // Wait for both to emit one more output.
            iter = 0;
            while ((narrow_count <= pix_i || parallel_count <= pix_i)
                   && iter < 200000) begin
                @(posedge clk);
                iter = iter + 1;
            end
            if (narrow_count <= pix_i || parallel_count <= pix_i) begin
                $display("[equiv] TIMEOUT at pixel %0d: narrow_count=%0d parallel_count=%0d",
                         pix_i, narrow_count, parallel_count);
                $finish;
            end
        end

        // Now compare the captured outputs byte-by-byte.
        $display("[equiv] all %0d pixels captured; comparing outputs", MAX_PIXELS);
        for (pix_i = 0; pix_i < MAX_PIXELS; pix_i = pix_i + 1) begin
            if (narrow_outputs[pix_i] !== parallel_outputs[pix_i]) begin
                mismatches = mismatches + 1;
                if (mismatches <= 4) begin
                    $display("[equiv] MISMATCH pixel %0d:", pix_i);
                    $display("    narrow  = %h", narrow_outputs[pix_i]);
                    $display("    parallel= %h", parallel_outputs[pix_i]);
                end
            end
        end

        if (mismatches == 0) begin
            $display("[equiv] PASS: %0d/%0d pixels byte-equal", MAX_PIXELS, MAX_PIXELS);
        end else begin
            $display("[equiv] FAIL: %0d/%0d pixels mismatched", mismatches, MAX_PIXELS);
        end
        $finish;
    end

    initial begin
        // Global watchdog so a deadlocked DUT doesn't hang the test.
        #200000000;
        $display("[equiv] WATCHDOG hit at sim time %0t", $time);
        $finish;
    end

endmodule
