`timescale 1ns/1ps
`default_nettype none
// ============================================================================
// retile_bridge_equiv_tb.v
// Equivalence TB for the [LUT-CUT 2026-06-07] SYNTH_FIXED_MUX path of
// retile_gather / retile_scatter.  For each instantiated parameter combo we
// build TWO copies driven by IDENTICAL stimulus -- one with SYNTH_FIXED_MUX(0)
// (the barrel-shift reference) and one with SYNTH_FIXED_MUX(1) (the fixed-mux
// candidate) -- and assert their FULL output port set (valid_out, data_out,
// ready_out, stall_out, wr_accept) is bit-identical every cycle.  Random data,
// random valid_in / ready_down bubbles so every g_idx (gather) and e_idx
// (scatter emit / partial-final beat) value is walked, incl the partial final
// beat where the 4096b chunk straddles FULL_W.
//
// PASS criterion: mismatch == 0 across all tested cycles for ALL combos.
//
// Build: see the task command (verilator --binary --timing with the Wno set).
// ============================================================================

module retile_bridge_equiv_tb;
    reg clk = 0;
    reg rst_n = 0;
    always #5 clk = ~clk;

    integer total_mismatch = 0;
    integer cyc = 0;

    // ---- shared random-stimulus driver bits ----
    reg        v_in;       // common valid_in
    reg        rd_down;    // common ready_down
    reg [4095:0] din_wide; // wide random data (used for IN_W up to 4096)
    reg [255:0]  din_tile; // tile random data (256b producer side for gather)

    // simple xorshift PRNG
    reg [63:0] rng = 64'hdeadbeefcafef00d;
    task step_rng; begin
        rng = rng ^ (rng << 13);
        rng = rng ^ (rng >> 7);
        rng = rng ^ (rng << 17);
    end endtask

    // ------------------------------------------------------------------
    // GATHER combos under test (TILE_W=256 producer):
    //   G1: N_TILES=18, OUT_W=4096, OUT_BEATS=2   (partial final beat)
    //   G2: N_TILES=30, OUT_W=4096, OUT_BEATS=2   (partial final beat)
    //   G3: N_TILES=40, OUT_W=2048, OUT_BEATS=5   (mean: exact, 5 arms)
    //   G4: N_TILES=18, OUT_W=2048, OUT_BEATS=3   (loader: partial final)
    //   G5: N_TILES=30, OUT_W=2048, OUT_BEATS=4   (loader: partial final)
    // SCATTER combos (TILE_W=256 consumer):
    //   S1: N_TILES=18, IN_W=4096, IN_BEATS=2     (partial final insert)
    //   S2: N_TILES=30, IN_W=4096, IN_BEATS=2     (partial final insert)
    // ------------------------------------------------------------------

    // ===================== GATHER G1 =====================
    wire g1a_vo, g1b_vo, g1a_ro, g1b_ro, g1a_so, g1b_so, g1a_wa, g1b_wa;
    wire [4095:0] g1a_do, g1b_do;
    retile_gather #(.TILE_W(256),.N_TILES(18),.OUT_W(4096),.OUT_BEATS(2),.SPATIAL(196),.SYNTH_FIXED_MUX(0)) g1a (
        .clk(clk),.rst_n(rst_n),.valid_in(v_in),.ready_out(g1a_ro),.data_in(din_tile[255:0]),
        .valid_out(g1a_vo),.ready_down(rd_down),.drain_en(1'b1),.data_out(g1a_do),.wr_accept(g1a_wa),.stall_out(g1a_so));
    retile_gather #(.TILE_W(256),.N_TILES(18),.OUT_W(4096),.OUT_BEATS(2),.SPATIAL(196),.SYNTH_FIXED_MUX(1)) g1b (
        .clk(clk),.rst_n(rst_n),.valid_in(v_in),.ready_out(g1b_ro),.data_in(din_tile[255:0]),
        .valid_out(g1b_vo),.ready_down(rd_down),.drain_en(1'b1),.data_out(g1b_do),.wr_accept(g1b_wa),.stall_out(g1b_so));

    // ===================== GATHER G2 =====================
    wire g2a_vo, g2b_vo, g2a_ro, g2b_ro, g2a_so, g2b_so, g2a_wa, g2b_wa;
    wire [4095:0] g2a_do, g2b_do;
    retile_gather #(.TILE_W(256),.N_TILES(30),.OUT_W(4096),.OUT_BEATS(2),.SPATIAL(49),.SYNTH_FIXED_MUX(0)) g2a (
        .clk(clk),.rst_n(rst_n),.valid_in(v_in),.ready_out(g2a_ro),.data_in(din_tile[255:0]),
        .valid_out(g2a_vo),.ready_down(rd_down),.drain_en(1'b1),.data_out(g2a_do),.wr_accept(g2a_wa),.stall_out(g2a_so));
    retile_gather #(.TILE_W(256),.N_TILES(30),.OUT_W(4096),.OUT_BEATS(2),.SPATIAL(49),.SYNTH_FIXED_MUX(1)) g2b (
        .clk(clk),.rst_n(rst_n),.valid_in(v_in),.ready_out(g2b_ro),.data_in(din_tile[255:0]),
        .valid_out(g2b_vo),.ready_down(rd_down),.drain_en(1'b1),.data_out(g2b_do),.wr_accept(g2b_wa),.stall_out(g2b_so));

    // ===================== GATHER G3 (mean, OUT_BEATS=5) =====================
    wire g3a_vo, g3b_vo, g3a_ro, g3b_ro, g3a_so, g3b_so, g3a_wa, g3b_wa;
    wire [2047:0] g3a_do, g3b_do;
    retile_gather #(.TILE_W(256),.N_TILES(40),.OUT_W(2048),.OUT_BEATS(5),.SPATIAL(49),.SYNTH_FIXED_MUX(0)) g3a (
        .clk(clk),.rst_n(rst_n),.valid_in(v_in),.ready_out(g3a_ro),.data_in(din_tile[255:0]),
        .valid_out(g3a_vo),.ready_down(rd_down),.drain_en(1'b1),.data_out(g3a_do),.wr_accept(g3a_wa),.stall_out(g3a_so));
    retile_gather #(.TILE_W(256),.N_TILES(40),.OUT_W(2048),.OUT_BEATS(5),.SPATIAL(49),.SYNTH_FIXED_MUX(1)) g3b (
        .clk(clk),.rst_n(rst_n),.valid_in(v_in),.ready_out(g3b_ro),.data_in(din_tile[255:0]),
        .valid_out(g3b_vo),.ready_down(rd_down),.drain_en(1'b1),.data_out(g3b_do),.wr_accept(g3b_wa),.stall_out(g3b_so));

    // ===================== GATHER G4 (loader, OUT_BEATS=3 partial) =====================
    wire g4a_vo, g4b_vo, g4a_ro, g4b_ro, g4a_so, g4b_so, g4a_wa, g4b_wa;
    wire [2047:0] g4a_do, g4b_do;
    retile_gather #(.TILE_W(256),.N_TILES(18),.OUT_W(2048),.OUT_BEATS(3),.SPATIAL(196),.SYNTH_FIXED_MUX(0)) g4a (
        .clk(clk),.rst_n(rst_n),.valid_in(v_in),.ready_out(g4a_ro),.data_in(din_tile[255:0]),
        .valid_out(g4a_vo),.ready_down(rd_down),.drain_en(1'b1),.data_out(g4a_do),.wr_accept(g4a_wa),.stall_out(g4a_so));
    retile_gather #(.TILE_W(256),.N_TILES(18),.OUT_W(2048),.OUT_BEATS(3),.SPATIAL(196),.SYNTH_FIXED_MUX(1)) g4b (
        .clk(clk),.rst_n(rst_n),.valid_in(v_in),.ready_out(g4b_ro),.data_in(din_tile[255:0]),
        .valid_out(g4b_vo),.ready_down(rd_down),.drain_en(1'b1),.data_out(g4b_do),.wr_accept(g4b_wa),.stall_out(g4b_so));

    // ===================== GATHER G5 (loader, OUT_BEATS=4 partial) =====================
    wire g5a_vo, g5b_vo, g5a_ro, g5b_ro, g5a_so, g5b_so, g5a_wa, g5b_wa;
    wire [2047:0] g5a_do, g5b_do;
    retile_gather #(.TILE_W(256),.N_TILES(30),.OUT_W(2048),.OUT_BEATS(4),.SPATIAL(49),.SYNTH_FIXED_MUX(0)) g5a (
        .clk(clk),.rst_n(rst_n),.valid_in(v_in),.ready_out(g5a_ro),.data_in(din_tile[255:0]),
        .valid_out(g5a_vo),.ready_down(rd_down),.drain_en(1'b1),.data_out(g5a_do),.wr_accept(g5a_wa),.stall_out(g5a_so));
    retile_gather #(.TILE_W(256),.N_TILES(30),.OUT_W(2048),.OUT_BEATS(4),.SPATIAL(49),.SYNTH_FIXED_MUX(1)) g5b (
        .clk(clk),.rst_n(rst_n),.valid_in(v_in),.ready_out(g5b_ro),.data_in(din_tile[255:0]),
        .valid_out(g5b_vo),.ready_down(rd_down),.drain_en(1'b1),.data_out(g5b_do),.wr_accept(g5b_wa),.stall_out(g5b_so));

    // ===================== SCATTER S1 =====================
    wire s1a_vo, s1b_vo, s1a_ro, s1b_ro, s1a_so, s1b_so, s1a_wa, s1b_wa;
    wire [255:0] s1a_do, s1b_do;
    retile_scatter #(.TILE_W(256),.N_TILES(18),.IN_W(4096),.IN_BEATS(2),.SPATIAL(196),.SYNTH_FIXED_MUX(0)) s1a (
        .clk(clk),.rst_n(rst_n),.valid_in(v_in),.ready_out(s1a_ro),.data_in(din_wide[4095:0]),
        .valid_out(s1a_vo),.ready_down(rd_down),.drain_en(1'b1),.data_out(s1a_do),.wr_accept(s1a_wa),.stall_out(s1a_so));
    retile_scatter #(.TILE_W(256),.N_TILES(18),.IN_W(4096),.IN_BEATS(2),.SPATIAL(196),.SYNTH_FIXED_MUX(1)) s1b (
        .clk(clk),.rst_n(rst_n),.valid_in(v_in),.ready_out(s1b_ro),.data_in(din_wide[4095:0]),
        .valid_out(s1b_vo),.ready_down(rd_down),.drain_en(1'b1),.data_out(s1b_do),.wr_accept(s1b_wa),.stall_out(s1b_so));

    // ===================== SCATTER S2 =====================
    wire s2a_vo, s2b_vo, s2a_ro, s2b_ro, s2a_so, s2b_so, s2a_wa, s2b_wa;
    wire [255:0] s2a_do, s2b_do;
    retile_scatter #(.TILE_W(256),.N_TILES(30),.IN_W(4096),.IN_BEATS(2),.SPATIAL(49),.SYNTH_FIXED_MUX(0)) s2a (
        .clk(clk),.rst_n(rst_n),.valid_in(v_in),.ready_out(s2a_ro),.data_in(din_wide[4095:0]),
        .valid_out(s2a_vo),.ready_down(rd_down),.drain_en(1'b1),.data_out(s2a_do),.wr_accept(s2a_wa),.stall_out(s2a_so));
    retile_scatter #(.TILE_W(256),.N_TILES(30),.IN_W(4096),.IN_BEATS(2),.SPATIAL(49),.SYNTH_FIXED_MUX(1)) s2b (
        .clk(clk),.rst_n(rst_n),.valid_in(v_in),.ready_out(s2b_ro),.data_in(din_wide[4095:0]),
        .valid_out(s2b_vo),.ready_down(rd_down),.drain_en(1'b1),.data_out(s2b_do),.wr_accept(s2b_wa),.stall_out(s2b_so));

    // ---- per-cycle comparison ----
    task chk;
        input [255:0] tag;
        input a; input b;
        begin
            if (a !== b) begin
                total_mismatch = total_mismatch + 1;
                if (total_mismatch <= 40)
                    $display("MISMATCH cyc=%0d %s : a=%b b=%b", cyc, tag, a, b);
            end
        end
    endtask

    task chk_data4096;
        input [255:0] tag;
        input [4095:0] a; input [4095:0] b;
        begin
            if (a !== b) begin
                total_mismatch = total_mismatch + 1;
                if (total_mismatch <= 40)
                    $display("MISMATCH cyc=%0d %s DATA", cyc, tag);
            end
        end
    endtask
    task chk_data2048;
        input [255:0] tag;
        input [2047:0] a; input [2047:0] b;
        begin
            if (a !== b) begin
                total_mismatch = total_mismatch + 1;
                if (total_mismatch <= 40)
                    $display("MISMATCH cyc=%0d %s DATA", cyc, tag);
            end
        end
    endtask
    task chk_data256;
        input [255:0] tag;
        input [255:0] a; input [255:0] b;
        begin
            if (a !== b) begin
                total_mismatch = total_mismatch + 1;
                if (total_mismatch <= 40)
                    $display("MISMATCH cyc=%0d %s DATA", cyc, tag);
            end
        end
    endtask

    // Drive inputs REGISTERED on posedge -> stable for the whole cycle the DUTs
    // sample.  Both copies of every combo see the IDENTICAL inputs.
    integer bb;
    always @(posedge clk) begin
        if (!rst_n) begin
            v_in <= 0; rd_down <= 0; din_wide <= 0; din_tile <= 0;
        end else begin
            step_rng; v_in    <= rng[0] | rng[5];   // mostly valid, some bubbles
            step_rng; rd_down <= rng[1] | rng[9];   // mostly ready, some bubbles
            for (bb = 0; bb < 256;  bb = bb + 32) begin step_rng; din_tile[bb +: 32] <= rng[31:0]; end
            for (bb = 0; bb < 4096; bb = bb + 32) begin step_rng; din_wide[bb +: 32] <= rng[31:0]; end
        end
    end

    // Compare registered DUT state + outputs at posedge (same edge that updated
    // buffers); inputs are stable so combinational outputs are valid.
    always @(posedge clk) begin
        if (rst_n) begin
            cyc = cyc + 1;
            // G1
            chk("G1 vo", g1a_vo, g1b_vo); chk("G1 ro", g1a_ro, g1b_ro);
            chk("G1 so", g1a_so, g1b_so); chk("G1 wa", g1a_wa, g1b_wa);
            if (g1a_vo) chk_data4096("G1", g1a_do, g1b_do);
            // G2
            chk("G2 vo", g2a_vo, g2b_vo); chk("G2 ro", g2a_ro, g2b_ro);
            chk("G2 so", g2a_so, g2b_so); chk("G2 wa", g2a_wa, g2b_wa);
            if (g2a_vo) chk_data4096("G2", g2a_do, g2b_do);
            // G3
            chk("G3 vo", g3a_vo, g3b_vo); chk("G3 ro", g3a_ro, g3b_ro);
            chk("G3 so", g3a_so, g3b_so); chk("G3 wa", g3a_wa, g3b_wa);
            if (g3a_vo) chk_data2048("G3", g3a_do, g3b_do);
            // G4
            chk("G4 vo", g4a_vo, g4b_vo); chk("G4 ro", g4a_ro, g4b_ro);
            chk("G4 so", g4a_so, g4b_so); chk("G4 wa", g4a_wa, g4b_wa);
            if (g4a_vo) chk_data2048("G4", g4a_do, g4b_do);
            // G5
            chk("G5 vo", g5a_vo, g5b_vo); chk("G5 ro", g5a_ro, g5b_ro);
            chk("G5 so", g5a_so, g5b_so); chk("G5 wa", g5a_wa, g5b_wa);
            if (g5a_vo) chk_data2048("G5", g5a_do, g5b_do);
            // S1
            chk("S1 vo", s1a_vo, s1b_vo); chk("S1 ro", s1a_ro, s1b_ro);
            chk("S1 so", s1a_so, s1b_so); chk("S1 wa", s1a_wa, s1b_wa);
            if (s1a_vo) chk_data256("S1", s1a_do, s1b_do);
            // S2
            chk("S2 vo", s2a_vo, s2b_vo); chk("S2 ro", s2a_ro, s2b_ro);
            chk("S2 so", s2a_so, s2b_so); chk("S2 wa", s2a_wa, s2b_wa);
            if (s2a_vo) chk_data256("S2", s2a_do, s2b_do);
        end
    end

    initial begin
        v_in = 0; rd_down = 0; din_wide = 0; din_tile = 0;
        rst_n = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        // run 60000 cycles -> walks all g_idx/e_idx incl partial finals, many
        // fill/drain phases, random bubbles on both handshake sides.
        repeat (60000) @(posedge clk);
        @(posedge clk);
        if (total_mismatch == 0)
            $display("RESULT: PASS mismatch=0 cycles=%0d", cyc);
        else
            $display("RESULT: FAIL mismatch=%0d cycles=%0d", total_mismatch, cyc);
        $finish;
    end

    // safety timeout
    initial begin
        #20000000;
        $display("RESULT: TIMEOUT mismatch=%0d", total_mismatch);
        $finish;
    end
endmodule
`default_nettype wire
