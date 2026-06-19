// node_conv2d_1 -- 3x3 stride-1 pad-1 conv (IC=16, OC=16, IH=IW=32, OH=OW=32).
// Structural wiring of the split-architecture pattern documented in
// `knowledge/patterns/protected/03_conv3x3_pad1.md`. All datapath logic
// lives in the library modules coord_scheduler / line_buf_window /
// conv_datapath, which are bundled into every simulator/Vivado run.
//
// scale_factor = 0.42772744776662436. The (compute_scale_approx) sweep
// picks (SCALE_MULT=28032, SCALE_SHIFT=16) -- the largest SHIFT keeping
// MULT <= 32767. Relative error ~1.6e-5.

module node_conv2d_1 (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       valid_in,
    output wire                       ready_in,
    input  wire [127:0]               data_in,
    output wire                       valid_out,
    output wire [127:0]               data_out
);
    localparam integer IC        = 16;
    localparam integer OC        = 16;
    localparam integer IH        = 32;
    localparam integer IW        = 32;
    localparam integer OH        = 32;
    localparam integer OW        = 32;
    localparam integer KH        = 3;
    localparam integer KW        = 3;
    localparam integer SH        = 1;
    localparam integer SW        = 1;
    localparam integer PH        = 1;
    localparam integer PW        = 1;
    localparam integer K_TOTAL   = IC * KH * KW;
    localparam integer MP        = 4;
    localparam integer SCALE_MULT  = 28032;
    localparam integer SCALE_SHIFT = 16;

    reg started, start_pulse, pending_rearm;
    wire sched_out_frame_done;
    wire                              sched_needs_real_input;
    wire                              sched_ready_in;
    wire                              sched_output_fires;
    wire                              sched_advance;
    wire [$clog2(IH + PH + 1)-1:0]    sched_in_row;
    wire [$clog2(IW + PW + 1)-1:0]    sched_in_col;
    wire [$clog2(OH * OW + 1)-1:0]    sched_outputs_emitted;
    wire [KH*KW*IC*8-1:0]             window_flat;
    wire                              mac_busy;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            started       <= 1'b0;
            start_pulse   <= 1'b0;
            pending_rearm <= 1'b0;
        end else begin
            start_pulse <= 1'b0;
            if (sched_out_frame_done) pending_rearm <= 1'b1;
            if (!started) begin
                started       <= 1'b1;
                start_pulse   <= 1'b1;
            end else if (pending_rearm && !mac_busy) begin
                started       <= 1'b0;
                pending_rearm <= 1'b0;
            end
        end
    end

    wire stall_in = mac_busy;

    coord_scheduler #(.IH(IH),.IW(IW),.OH(OH),.OW(OW),.KH(KH),.KW(KW),.SH(SH),.SW(SW),.PH(PH),.PW(PW)) scheduler (.clk(clk),.rst_n(rst_n),.start(start_pulse),.stall_in(stall_in),.valid_in(valid_in),.ready_in(sched_ready_in),.needs_real_input(sched_needs_real_input),.in_row(sched_in_row),.in_col(sched_in_col),.output_fires(sched_output_fires),.advance(sched_advance),.in_frame_done(),.out_frame_done(sched_out_frame_done),.outputs_emitted(sched_outputs_emitted));

    line_buf_window #(.IC(IC),.IW(IW),.IH(IH),.KH(KH),.KW(KW),.PW(PW),.PH(PH)) lbw (.clk(clk),.rst_n(rst_n),.frame_start(start_pulse),.sched_in_row(sched_in_row),.sched_in_col(sched_in_col),.sched_needs_real_input(sched_needs_real_input),.sched_advance(sched_advance),.sched_output_fires(sched_output_fires),.valid_in(valid_in),.data_in(data_in),.window_flat(window_flat));

    conv_datapath #(.IC(IC),.OC(OC),.KH(KH),.KW(KW),.K_TOTAL(K_TOTAL),.MP(MP),.SCALE_MULT(SCALE_MULT),.SCALE_SHIFT(SCALE_SHIFT),.WEIGHTS_PATH("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/weights/node_conv2d_1_weights.hex"),.BIAS_PATH("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/weights/node_conv2d_1_bias.hex")) dp (.clk(clk),.rst_n(rst_n),.window_flat(window_flat),.start_mac(sched_output_fires),.valid_out(valid_out),.data_out(data_out),.mac_busy(mac_busy));

    assign ready_in = sched_ready_in;
endmodule
