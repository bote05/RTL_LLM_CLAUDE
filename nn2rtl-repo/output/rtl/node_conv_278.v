// node_conv_278 -- tiled-streaming 3x3 conv2d.
//   IC=256, OC=256, IH=IW=14, stride=1, padding=1, MP=4.
//   channel_tile=32 -> IN_BEATS=8, OUT_BEATS=8 per pixel.
//
// Split-architecture (coord_scheduler + line_buf_window + conv_datapath)
// wrapped with a beat-aggregator + beat-splitter. The library still sees
// the full IC*8 / OC*8 packed pixel; the wrapper serializes that pixel
// across CHANNEL_TILE-wide beats on the external bus.

module node_conv_278 (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       valid_in,
    output wire                       ready_in,
    input  wire [255:0]               data_in,
    output wire                       valid_out,
    output wire [255:0]               data_out
);
    localparam integer IC             = 256;
    localparam integer OC             = 256;
    localparam integer IH             = 14;
    localparam integer IW             = 14;
    localparam integer OH             = 14;
    localparam integer OW             = 14;
    localparam integer KH             = 3;
    localparam integer KW             = 3;
    localparam integer SH             = 1;
    localparam integer SW             = 1;
    localparam integer PH             = 1;
    localparam integer PW             = 1;
    localparam integer K_TOTAL        = IC * KH * KW;
    localparam integer MP             = 4;

    localparam integer SCALE_MULT     = 18129;
    localparam integer SCALE_SHIFT    = 22;

    localparam integer CHANNEL_TILE   = 32;
    localparam integer TILE_BITS      = CHANNEL_TILE * 8;
    localparam integer IN_PIXEL_BITS  = IC * 8;
    localparam integer OUT_PIXEL_BITS = OC * 8;
    localparam integer IN_BEATS       = IC / CHANNEL_TILE;
    localparam integer OUT_BEATS      = OC / CHANNEL_TILE;
    localparam integer LAST_IN_BEAT   = IN_BEATS - 1;
    localparam integer LAST_OUT_BEAT  = OUT_BEATS - 1;

    wire                                sched_needs_real_input;
    wire                                sched_ready_in;
    wire                                sched_output_fires;
    wire                                sched_advance;
    wire [$clog2(IH + PH + 1)-1:0]      sched_in_row;
    wire [$clog2(IW + PW + 1)-1:0]      sched_in_col;
    wire [$clog2(OH * OW + 1)-1:0]      sched_outputs_emitted;
    wire                                sched_out_frame_done;

    wire [KH*KW*IC*8-1:0]               window_flat;
    wire                                mac_busy;
    wire                                lib_valid_out_w;
    wire [OUT_PIXEL_BITS-1:0]           lib_data_out_w;

    localparam ST_ARM  = 2'd0;
    localparam ST_RUN  = 2'd1;
    localparam ST_WAIT = 2'd2;

    reg [1:0] frame_state;
    reg       start_pulse;

    reg  [2:0]                          in_beat_idx;
    reg  [IN_PIXEL_BITS-TILE_BITS-1:0]  pixel_low_r;

    wire                                last_beat_now;
    wire                                early_beat_now;

    // [INVARIANT:READY_IN_GATING]
    assign ready_in       = (in_beat_idx == LAST_IN_BEAT) ? sched_ready_in : 1'b1;
    assign early_beat_now = valid_in && (in_beat_idx != LAST_IN_BEAT);
    assign last_beat_now  = valid_in && (in_beat_idx == LAST_IN_BEAT) && sched_ready_in;

    wire                                lib_valid_in_w = last_beat_now;
    wire [IN_PIXEL_BITS-1:0]            lib_data_in_w  = {data_in, pixel_low_r};

    reg  [OUT_PIXEL_BITS-TILE_BITS-1:0] out_pixel_high_r;
    reg                                 out_streaming;
    reg  [2:0]                          out_beat_idx;
    reg  [TILE_BITS-1:0]                streaming_tile_w;

    // [INVARIANT:VALID_OUT_LATENCY]
    assign valid_out = lib_valid_out_w || out_streaming;
    assign data_out  = out_streaming ? streaming_tile_w : lib_data_out_w[TILE_BITS-1:0];

    always @(*) begin
        case (out_beat_idx)
            3'd1: streaming_tile_w = out_pixel_high_r[0*TILE_BITS +: TILE_BITS];
            3'd2: streaming_tile_w = out_pixel_high_r[1*TILE_BITS +: TILE_BITS];
            3'd3: streaming_tile_w = out_pixel_high_r[2*TILE_BITS +: TILE_BITS];
            3'd4: streaming_tile_w = out_pixel_high_r[3*TILE_BITS +: TILE_BITS];
            3'd5: streaming_tile_w = out_pixel_high_r[4*TILE_BITS +: TILE_BITS];
            3'd6: streaming_tile_w = out_pixel_high_r[5*TILE_BITS +: TILE_BITS];
            3'd7: streaming_tile_w = out_pixel_high_r[6*TILE_BITS +: TILE_BITS];
            default: streaming_tile_w = {TILE_BITS{1'b0}};
        endcase
    end

    // Dynamically-indexed slice write into pixel_low_r lives in a sync-only
    // always block. The async-reset block below carries everything else
    // (FSM state, counters, output-streaming regs) so Vivado can infer
    // distributed RAM for this beat-staging memory.
    always @(posedge clk) begin
        if (early_beat_now)
            pixel_low_r[in_beat_idx*TILE_BITS +: TILE_BITS] <= data_in;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            frame_state      <= ST_ARM;
            start_pulse      <= 1'b0;
            in_beat_idx      <= 3'd0;
            out_pixel_high_r <= {(OUT_PIXEL_BITS-TILE_BITS){1'b0}};
            out_streaming    <= 1'b0;
            out_beat_idx     <= 3'd0;
        end else begin
            start_pulse <= 1'b0;

            case (frame_state)
                ST_ARM: begin
                    start_pulse <= 1'b1;
                    frame_state <= ST_RUN;
                end
                ST_RUN: begin
                    if (sched_out_frame_done) frame_state <= ST_WAIT;
                end
                ST_WAIT: begin
                    if (!mac_busy && !out_streaming) frame_state <= ST_ARM;
                end
                default: frame_state <= ST_ARM;
            endcase

            if (early_beat_now) begin
                in_beat_idx <= in_beat_idx + 3'd1;
            end else if (last_beat_now) begin
                in_beat_idx <= 3'd0;
            end

            if (lib_valid_out_w) begin
                out_pixel_high_r <= lib_data_out_w[OUT_PIXEL_BITS-1:TILE_BITS];
                out_streaming    <= 1'b1;
                out_beat_idx     <= 3'd1;
            end else if (out_streaming) begin
                if (out_beat_idx == LAST_OUT_BEAT[2:0]) begin
                    out_streaming <= 1'b0;
                    out_beat_idx  <= 3'd0;
                end else begin
                    out_beat_idx <= out_beat_idx + 3'd1;
                end
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
        .valid_in(lib_valid_in_w),
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
        .valid_in(lib_valid_in_w),
        .data_in(lib_data_in_w),
        .window_flat(window_flat)
    );

    conv_datapath_mp_k #(.DSP_INPUT_PIPE(1),
        .IC(IC), .OC(OC), .KH(KH), .KW(KW),
        .K_TOTAL(K_TOTAL), .MP(MP),
        .MP_K(9),
        .SCALE_MULT(SCALE_MULT), .SCALE_SHIFT(SCALE_SHIFT),
        .WEIGHTS_PATH("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_278_weights_mp_k_9.hex"),
        .BIAS_PATH("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_278_bias.hex")
    ) dp (
        .clk(clk), .rst_n(rst_n),
        .window_flat(window_flat),
        .start_mac(sched_output_fires),
        .valid_out(lib_valid_out_w),
        .data_out(lib_data_out_w),
        .mac_busy(mac_busy)
    );

endmodule
