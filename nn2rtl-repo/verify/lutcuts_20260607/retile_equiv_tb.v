`timescale 1ns/1ps
`default_nettype none
// ============================================================================
// retile_equiv_tb.v -- equivalence TB for retile_bridge SYNTH_FIXED_MUX 0 vs 1.
//
// For each distinct parameterization used in nn2rtl_top_engine.v (including
// parameterizations with a PARTIAL last beat where FULL_W is NOT a clean multiple
// of OUT_W / IN_W -- the case that caused [Synth 8-524]), this TB instantiates
// the bridge TWICE: one with SYNTH_FIXED_MUX(0) (original variable barrel ops)
// and one with SYNTH_FIXED_MUX(1) (fixed mux + clamp).  Both get IDENTICAL random
// data driven through every gather index and every drain (emit) index, including
// the final partial beat.  Every cycle, data_out / valid_out / ready_out /
// stall_out of the two instances must be bit-identical (mismatch=0).
// ============================================================================

module retile_equiv_tb;
    reg clk = 0;
    reg rst_n = 0;
    always #5 clk = ~clk;

    integer total_mismatch = 0;
    integer total_compares = 0;

    // ------------------------------------------------------------------------
    // GATHER harness: drive N_TILES tiled beats in, drain OUT_BEATS wide beats.
    // ------------------------------------------------------------------------
`define GATHER_PAIR(NM, P_TILE, P_NT, P_OW, P_OB, P_SP) \
    wire                 NM``_ready0, NM``_ready1, NM``_valid0, NM``_valid1; \
    wire                 NM``_stall0, NM``_stall1, NM``_wracc0, NM``_wracc1; \
    wire [P_OW-1:0]      NM``_dout0,  NM``_dout1; \
    reg                  NM``_vin = 0; reg [P_TILE-1:0] NM``_din = 0; \
    reg                  NM``_rdwn = 0; reg NM``_den = 0; \
    retile_gather #(.TILE_W(P_TILE), .N_TILES(P_NT), .OUT_W(P_OW), .OUT_BEATS(P_OB), .SPATIAL(P_SP), .SYNTH_FIXED_MUX(0)) NM``_u0 ( \
        .clk(clk), .rst_n(rst_n), .valid_in(NM``_vin), .ready_out(NM``_ready0), .data_in(NM``_din), \
        .valid_out(NM``_valid0), .ready_down(NM``_rdwn), .drain_en(NM``_den), .data_out(NM``_dout0), \
        .wr_accept(NM``_wracc0), .stall_out(NM``_stall0)); \
    retile_gather #(.TILE_W(P_TILE), .N_TILES(P_NT), .OUT_W(P_OW), .OUT_BEATS(P_OB), .SPATIAL(P_SP), .SYNTH_FIXED_MUX(1)) NM``_u1 ( \
        .clk(clk), .rst_n(rst_n), .valid_in(NM``_vin), .ready_out(NM``_ready1), .data_in(NM``_din), \
        .valid_out(NM``_valid1), .ready_down(NM``_rdwn), .drain_en(NM``_den), .data_out(NM``_dout1), \
        .wr_accept(NM``_wracc1), .stall_out(NM``_stall1));

`define SCATTER_PAIR(NM, P_TILE, P_NT, P_IW, P_IB, P_SP) \
    wire                 NM``_ready0, NM``_ready1, NM``_valid0, NM``_valid1; \
    wire                 NM``_stall0, NM``_stall1, NM``_wracc0, NM``_wracc1; \
    wire [P_TILE-1:0]    NM``_dout0,  NM``_dout1; \
    reg                  NM``_vin = 0; reg [P_IW-1:0] NM``_din = 0; \
    reg                  NM``_rdwn = 0; reg NM``_den = 0; \
    retile_scatter #(.TILE_W(P_TILE), .N_TILES(P_NT), .IN_W(P_IW), .IN_BEATS(P_IB), .SPATIAL(P_SP), .SYNTH_FIXED_MUX(0)) NM``_u0 ( \
        .clk(clk), .rst_n(rst_n), .valid_in(NM``_vin), .ready_out(NM``_ready0), .data_in(NM``_din), \
        .valid_out(NM``_valid0), .ready_down(NM``_rdwn), .drain_en(NM``_den), .data_out(NM``_dout0), \
        .wr_accept(NM``_wracc0), .stall_out(NM``_stall0)); \
    retile_scatter #(.TILE_W(P_TILE), .N_TILES(P_NT), .IN_W(P_IW), .IN_BEATS(P_IB), .SPATIAL(P_SP), .SYNTH_FIXED_MUX(1)) NM``_u1 ( \
        .clk(clk), .rst_n(rst_n), .valid_in(NM``_vin), .ready_out(NM``_ready1), .data_in(NM``_din), \
        .valid_out(NM``_valid1), .ready_down(NM``_rdwn), .drain_en(NM``_den), .data_out(NM``_dout1), \
        .wr_accept(NM``_wracc1), .stall_out(NM``_stall1));

    // ---- GATHER combos (cover partial + exact last beats) ----
    `GATHER_PAIR(g_a, 256, 18, 4096, 2, 196)   // FULL_W=4608, beat1 partial (4096+4096>4608) -> the 8-524 case
    `GATHER_PAIR(g_b, 256, 30, 4096, 2, 49)    // FULL_W=7680, beat1 partial
    `GATHER_PAIR(g_c, 256, 40, 2048, 5, 49)    // FULL_W=10240, exact (no partial)
    `GATHER_PAIR(g_d, 256, 18, 2048, 3, 196)   // FULL_W=4608, beat2 partial (4096+2048>4608)
    `GATHER_PAIR(g_e, 256, 30, 2048, 4, 49)    // FULL_W=7680, beat3 partial (6144+2048>7680)

    // ---- SCATTER combos (insert side partial) ----
    `SCATTER_PAIR(s_a, 256, 18, 4096, 2, 196)  // FULL_W=4608, insert beat1 partial
    `SCATTER_PAIR(s_b, 256, 30, 4096, 2, 49)   // FULL_W=7680, insert beat1 partial

    // ------------------------------------------------------------------------
    // Compare both instances every cycle (after reset).
    // ------------------------------------------------------------------------
    task chk_g;
        input [255:0] name; input v0; input v1; input r0; input r1;
        input st0; input st1; input [4095:0] d0; input [4095:0] d1;
        begin
            total_compares = total_compares + 1;
            if (v0 !== v1 || r0 !== r1 || st0 !== st1) begin
                total_mismatch = total_mismatch + 1;
                $display("MISMATCH ctrl %0s @%0t v %b/%b r %b/%b st %b/%b", name, $time, v0, v1, r0, r1, st0, st1);
            end
            // compare data ONLY when the FIXED variant declares the beat valid
            if (v1 && (d0 !== d1)) begin
                total_mismatch = total_mismatch + 1;
                $display("MISMATCH data %0s @%0t", name, $time);
            end
        end
    endtask
    task chk_s;
        input [255:0] name; input v0; input v1; input r0; input r1;
        input st0; input st1; input [255:0] d0; input [255:0] d1;
        begin
            total_compares = total_compares + 1;
            if (v0 !== v1 || r0 !== r1 || st0 !== st1) begin
                total_mismatch = total_mismatch + 1;
                $display("MISMATCH ctrl %0s @%0t v %b/%b r %b/%b st %b/%b", name, $time, v0, v1, r0, r1, st0, st1);
            end
            if (v1 && (d0 !== d1)) begin
                total_mismatch = total_mismatch + 1;
                $display("MISMATCH data %0s @%0t", name, $time);
            end
        end
    endtask

    // Per-cycle comparison (sampled just before posedge effects settle in next).
    always @(negedge clk) if (rst_n) begin
        chk_g("g_a", g_a_valid0, g_a_valid1, g_a_ready0, g_a_ready1, g_a_stall0, g_a_stall1, {3072'b0, g_a_dout0}, {3072'b0, g_a_dout1});
        chk_g("g_b", g_b_valid0, g_b_valid1, g_b_ready0, g_b_ready1, g_b_stall0, g_b_stall1, {3072'b0, g_b_dout0}, {3072'b0, g_b_dout1});
        chk_g("g_c", g_c_valid0, g_c_valid1, g_c_ready0, g_c_ready1, g_c_stall0, g_c_stall1, {6144'b0, g_c_dout0}, {6144'b0, g_c_dout1});
        chk_g("g_d", g_d_valid0, g_d_valid1, g_d_ready0, g_d_ready1, g_d_stall0, g_d_stall1, {6144'b0, g_d_dout0}, {6144'b0, g_d_dout1});
        chk_g("g_e", g_e_valid0, g_e_valid1, g_e_ready0, g_e_ready1, g_e_stall0, g_e_stall1, {6144'b0, g_e_dout0}, {6144'b0, g_e_dout1});
        chk_s("s_a", s_a_valid0, s_a_valid1, s_a_ready0, s_a_ready1, s_a_stall0, s_a_stall1, s_a_dout0, s_a_dout1);
        chk_s("s_b", s_b_valid0, s_b_valid1, s_b_ready0, s_b_ready1, s_b_stall0, s_b_stall1, s_b_dout0, s_b_dout1);
    end

    // ------------------------------------------------------------------------
    // Stimulus.  Random gather pixels with random drain throttling; we keep
    // ready_down & drain_en randomly toggling so emit indices step through ALL
    // beats incl the partial last one many times.
    // ------------------------------------------------------------------------
    integer i;
    reg [255:0] rnd_tile;
    reg [4095:0] rnd_in;

    // gather driver: push a tile beat when ready_out (use u0's ready, identical to u1)
    task step_gather; begin
        rnd_tile = {$random, $random, $random, $random, $random, $random, $random, $random};
        g_a_din  = rnd_tile; g_b_din = rnd_tile; g_c_din = rnd_tile; g_d_din = rnd_tile; g_e_din = rnd_tile;
        // valid_in asserted randomly
        g_a_vin = $random; g_b_vin = $random; g_c_vin = $random; g_d_vin = $random; g_e_vin = $random;
        // drain randomly enabled
        g_a_rdwn = $random; g_a_den = $random;
        g_b_rdwn = $random; g_b_den = $random;
        g_c_rdwn = $random; g_c_den = $random;
        g_d_rdwn = $random; g_d_den = $random;
        g_e_rdwn = $random; g_e_den = $random;
    end endtask

    task step_scatter; begin
        rnd_in = {$random,$random,$random,$random,$random,$random,$random,$random,
                  $random,$random,$random,$random,$random,$random,$random,$random,
                  $random,$random,$random,$random,$random,$random,$random,$random,
                  $random,$random,$random,$random,$random,$random,$random,$random,
                  $random,$random,$random,$random,$random,$random,$random,$random,
                  $random,$random,$random,$random,$random,$random,$random,$random,
                  $random,$random,$random,$random,$random,$random,$random,$random,
                  $random,$random,$random,$random,$random,$random,$random,$random,
                  $random,$random,$random,$random,$random,$random,$random,$random,
                  $random,$random,$random,$random,$random,$random,$random,$random,
                  $random,$random,$random,$random,$random,$random,$random,$random,
                  $random,$random,$random,$random,$random,$random,$random,$random,
                  $random,$random,$random,$random,$random,$random,$random,$random,
                  $random,$random,$random,$random,$random,$random,$random,$random,
                  $random,$random,$random,$random,$random,$random,$random,$random,
                  $random,$random,$random,$random,$random,$random,$random,$random};
        s_a_din = rnd_in[4095:0]; s_b_din = rnd_in[4095:0];
        s_a_vin = $random; s_b_vin = $random;
        s_a_rdwn = $random; s_a_den = $random;
        s_b_rdwn = $random; s_b_den = $random;
    end endtask

    initial begin
        rst_n = 0;
        repeat (5) @(negedge clk);
        rst_n = 1;
        for (i = 0; i < 20000; i = i + 1) begin
            @(negedge clk);
            step_gather;
            step_scatter;
        end
        // a flush phase: stop new intake, keep draining to step through all emit beats
        g_a_vin=0; g_b_vin=0; g_c_vin=0; g_d_vin=0; g_e_vin=0; s_a_vin=0; s_b_vin=0;
        g_a_rdwn=1; g_a_den=1; g_b_rdwn=1; g_b_den=1; g_c_rdwn=1; g_c_den=1;
        g_d_rdwn=1; g_d_den=1; g_e_rdwn=1; g_e_den=1; s_a_rdwn=1; s_a_den=1; s_b_rdwn=1; s_b_den=1;
        repeat (200) @(negedge clk);

        $display("=== RETILE EQUIV TB DONE ===");
        $display("compares=%0d mismatch=%0d", total_compares, total_mismatch);
        if (total_mismatch == 0) $display("RESULT: PASS mismatch=0");
        else                     $display("RESULT: FAIL mismatch=%0d", total_mismatch);
        $finish;
    end
endmodule
`default_nettype wire
