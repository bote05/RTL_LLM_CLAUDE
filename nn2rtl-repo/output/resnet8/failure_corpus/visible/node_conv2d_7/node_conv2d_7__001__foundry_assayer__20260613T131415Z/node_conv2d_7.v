// node_conv2d_7 -- 4x4 stride-2 pad-1 conv (IC=32, OC=64, IH=IW=16, OH=OW=8).
// Split architecture: coord_scheduler + line_buf_window + conv_datapath.
// MP=4, OC_PASSES=16, K_TOTAL=512, pass_cycles=MP*K_TOTAL+6=2054, per-pixel=32864.
// setup = (KH-1-PH)*(IW+PW) + (KW-PW) = 2*17 + 3 = 37.
// pipeline_latency_cycles = setup + 1 + per_pixel = 37 + 1 + 32864 = 32902.
// scale_factor=1.2727433159114343 -> SCALE_MULT=10426, SCALE_SHIFT=13
//   (10426/8192 = 1.272705..., rel err ~3.0e-5).

module node_conv2d_7 (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       valid_in,
    output wire                       ready_in,
    input  wire [255:0]               data_in,
    output wire                       valid_out,
    output wire [511:0]               data_out
);
    localparam integer IC          = 32;
    localparam integer OC          = 64;
    localparam integer IH          = 16;
    localparam integer IW          = 16;
    localparam integer OH          = 8;
    localparam integer OW          = 8;
    localparam integer KH          = 4;
    localparam integer KW          = 4;
    localparam integer SH          = 2;
    localparam integer SW          = 2;
    localparam integer PH          = 1;
    localparam integer PW          = 1;
    localparam integer K_TOTAL     = IC * KH * KW; // 512
    localparam integer MP          = 4;

    localparam integer SCALE_MULT  = 10426;
    localparam integer SCALE_SHIFT = 13;

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
            if (sched_out_frame_done) begin
                pending_rearm <= 1'b1;
            end
            if (!started) begin
                started     <= 1'b1;
                start_pulse <= 1'b1;
            end else if (pending_rearm && !mac_busy) begin
                started       <= 1'b0;
                pending_rearm <= 1'b0;
            end
        end
    end

    wire stall_in = mac_busy;

    coord_scheduler #(
        .IH(IH), .IW(IW), .OH(OH), .OW(OW),
        .KH(KH), .KW(KW), .SH(SH), .SW(SW),
        .PH(PH), .PW(PW)
    ) scheduler (
        .clk(clk), .rst_n(rst_n),
        .start(start_pulse),
        .stall_in(stall_in),
        .valid_in(valid_in),
        .ready_in(sched_ready_in),
        .needs_real_input(sched_needs_real_input),
        .in_row(sched_in_row),
        .in_col(sched_in_col),
        .output_fires(sched_output_fires),
        .advance(sched_advance),
        .in_frame_done(),
        .out_frame_done(sched_out_frame_done),
        .outputs_emitted(sched_outputs_emitted)
    );

    line_buf_window #(
        .IC(IC), .IW(IW), .IH(IH),
        .KH(KH), .KW(KW), .PW(PW), .PH(PH)
    ) lbw (
        .clk(clk), .rst_n(rst_n),
        .frame_start(start_pulse),
        .sched_in_row(sched_in_row),
        .sched_in_col(sched_in_col),
        .sched_needs_real_input(sched_needs_real_input),
        .sched_advance(sched_advance),
        .sched_output_fires(sched_output_fires),
        .valid_in(valid_in),
        .data_in(data_in),
        .window_flat(window_flat)
    );

    wire         dp_valid_out;
    wire [511:0] dp_data_out;

    conv_datapath #(
        .IC(IC), .OC(OC), .KH(KH), .KW(KW),
        .K_TOTAL(K_TOTAL), .MP(MP),
        .SCALE_MULT(SCALE_MULT), .SCALE_SHIFT(SCALE_SHIFT),
        .WEIGHTS_PATH("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/weights/node_conv2d_7_weights.hex"),
        .BIAS_PATH("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/weights/node_conv2d_7_bias.hex")
    ) dp (
        .clk(clk), .rst_n(rst_n),
        .window_flat(window_flat),
        .start_mac(sched_output_fires),
        .valid_out(dp_valid_out), // [INVARIANT:VALID_OUT_LATENCY]
        .data_out(dp_data_out),
        .mac_busy(mac_busy)
    );

    assign valid_out = dp_valid_out;
    assign data_out  = dp_data_out;
    assign ready_in  = sched_ready_in; // [INVARIANT:READY_IN_GATING]

endmodule
