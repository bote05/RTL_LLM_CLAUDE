// node_conv_810 -- 3x3 stride-2 pad-1 conv, IC=3, OC=32, IH=IW=224, OH=OW=112.
// Split-architecture pattern per knowledge/patterns/protected/03_conv3x3_pad1.md.
//
// PARAM-GATED ELASTIC BACKPRESSURE (ENABLE_BACKPRESSURE, default 0):
//   * ==0 (default): bit/cycle-IDENTICAL to the legacy module. out_ready_in is
//     IGNORED; skid_block is a constant 0 (scheduler/rearm never freeze); the
//     external valid_out/data_out come DIRECTLY from the datapath, so the
//     1-cycle output skid is bypassed. The per-module verify TB (param=0) is
//     byte-exact with NO harness change.
//   * ==1: 1-deep output skid (out_full/out_data) captures the datapath's
//     1-cycle valid_out pulse, and skid_block = out_full && !out_ready_in feeds
//     stall_in + blocks the frame rearm so the scheduler/datapath FREEZE while a
//     beat is parked and the downstream is not ready. The datapath arithmetic is
//     unchanged; only the emit *timing* changes (per scratch/node_conv_810_bp.v).

module node_conv_810 #(
    parameter ENABLE_BACKPRESSURE = 0
)(
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       valid_in,
    output wire                       ready_in,
    input  wire [23:0]                data_in,
    input  wire                       out_ready_in,   // NEW: downstream-ready (ignored when ENABLE_BACKPRESSURE==0)
    output wire                       valid_out,
    output wire [255:0]               data_out
);
    localparam integer IC        = 3;
    localparam integer OC        = 32;
    localparam integer IH        = 224;
    localparam integer IW        = 224;
    localparam integer OH        = 112;
    localparam integer OW        = 112;
    localparam integer KH        = 3;
    localparam integer KW        = 3;
    localparam integer SH        = 2;
    localparam integer SW        = 2;
    localparam integer PH        = 1;
    localparam integer PW        = 1;
    localparam integer K_TOTAL   = IC * KH * KW;
    localparam integer MP        = 16;

    localparam integer SCALE_MULT  = 18143;
    localparam integer SCALE_SHIFT = 23;

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

    // ---- datapath valid_out pulse + stable data ----
    wire                              dp_valid_out;
    wire [255:0]                      dp_data_out;

    // ---- 1-deep output skid (only meaningful when ENABLE_BACKPRESSURE==1) ----
    reg          out_full;
    reg  [255:0] out_data;
    // skid_block freezes the scheduler + rearm while a beat is parked and the
    // downstream cannot take it. With ENABLE_BACKPRESSURE==0 it is a constant 0,
    // so the scheduler/rearm/datapath behave exactly as the legacy module.
    wire skid_block = (ENABLE_BACKPRESSURE != 0) && out_full && !out_ready_in;

    generate
    if (ENABLE_BACKPRESSURE == 0) begin : g_out_legacy
        // Direct passthrough: cycle-identical to the original module.
        assign valid_out = dp_valid_out;
        assign data_out  = dp_data_out;
    end else begin : g_out_bp
        assign valid_out = out_full;
        assign data_out  = out_data;
    end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_full <= 1'b0;
            out_data <= 256'd0;
        end else begin
            if (out_full && out_ready_in)
                out_full <= 1'b0;
            // Capture the datapath's 1-cycle valid_out pulse. By construction
            // (skid_block freezes the scheduler) the datapath never pulses
            // dp_valid_out while out_full is set and not being taken.
            if (dp_valid_out) begin
                out_data <= dp_data_out;
                out_full <= 1'b1;
            end
        end
    end

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
                started       <= 1'b1;
                start_pulse   <= 1'b1;
            end else if (pending_rearm && !mac_busy && !skid_block) begin
                started       <= 1'b0;
                pending_rearm <= 1'b0;
            end
        end
    end

    wire stall_in = mac_busy || skid_block;

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

    conv_datapath_mp_k #(
        .IC(IC), .OC(OC), .KH(KH), .KW(KW),
        .K_TOTAL(K_TOTAL), .MP(MP),
        .MP_K(9),                 // tap-parallel; K_TOTAL=27 = 3*9 -> 3 group cycles
        .WGT_BITS(8),               // MBv2 INT8 (mp_k default is 4=INT4!)
        .SCALE_MULT(SCALE_MULT), .SCALE_SHIFT(SCALE_SHIFT), .SCALE_PATH(""),
        .WEIGHTS_PATH("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_810_weights_mp_k_9.hex"),
        .BIAS_PATH("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_810_bias.hex")
    ) dp (
        .clk(clk), .rst_n(rst_n),
        .window_flat(window_flat),
        .start_mac(sched_output_fires),
        .valid_out(dp_valid_out),
        .data_out(dp_data_out),
        .mac_busy(mac_busy)
    );

    assign ready_in = sched_ready_in;

endmodule
