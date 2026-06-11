// node_conv_260 -- tiled-streaming 3x3 conv2d.
//   IC=256, OC=256, IH=IW=14, stride=1, padding=1, MP=4.
//   channel_tile=32 -> IN_BEATS=8, OUT_BEATS=8 per pixel.
//
// Split-architecture (coord_scheduler + line_buf_window + conv_datapath)
// wrapped with an N-beat aggregator (collect IN_BEATS tiles, present one
// IC*8-bit pixel) and an N-beat splitter (one OC*8-bit pixel emitted as
// OUT_BEATS tiles).
//
// The indexed slice write to the input-tile shift register lives in a
// dedicated `always @(posedge clk)` block with no reset clause so the
// Vivado structural preflight (activation_memory_in_async_reset_block)
// does not flag it. The control regs (FSM, beat indices, start pulse)
// keep their async-reset semantics in a sibling block.

module node_conv_260 (
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

    localparam integer SCALE_MULT     = 15643;
    localparam integer SCALE_SHIFT    = 23;

    localparam integer CHANNEL_TILE   = 32;
    localparam integer TILE_BITS      = CHANNEL_TILE * 8;
    localparam integer IN_BEATS       = (IC + CHANNEL_TILE - 1) / CHANNEL_TILE;
    localparam integer OUT_BEATS      = (OC + CHANNEL_TILE - 1) / CHANNEL_TILE;
    localparam integer BEAT_IDX_W     = 3;
    localparam integer IN_PIXEL_BITS  = IC * 8;
    localparam integer OUT_PIXEL_BITS = OC * 8;

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

    reg [BEAT_IDX_W-1:0]                in_beat_idx;
    reg [(IN_BEATS-1)*TILE_BITS-1:0]    tile_low_q;

    wire                                last_beat_phase;
    wire                                beat_early_now;
    wire                                beat_last_now;

    assign last_beat_phase = (in_beat_idx == IN_BEATS-1);
    assign ready_in        = last_beat_phase ? sched_ready_in : 1'b1;
    assign beat_early_now  = valid_in && !last_beat_phase;
    assign beat_last_now   = valid_in &&  last_beat_phase && sched_ready_in;

    wire                                lib_valid_in_w = beat_last_now;
    wire [IN_PIXEL_BITS-1:0]            lib_data_in_w  = {data_in, tile_low_q};

    reg [BEAT_IDX_W-1:0]                out_beat_idx;
    reg [(OUT_BEATS-1)*TILE_BITS-1:0]   out_pack_high;

    wire                                streaming_high_beats = (out_beat_idx != {BEAT_IDX_W{1'b0}});

    assign valid_out = lib_valid_out_w || streaming_high_beats;
    assign data_out  = streaming_high_beats
                         ? out_pack_high[(out_beat_idx - 1'b1)*TILE_BITS +: TILE_BITS]
                         : lib_data_out_w[TILE_BITS-1:0];

    always @(posedge clk) begin
        if (beat_early_now) begin
            tile_low_q[in_beat_idx*TILE_BITS +: TILE_BITS] <= data_in;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            frame_state    <= ST_ARM;
            start_pulse    <= 1'b0;
            in_beat_idx    <= {BEAT_IDX_W{1'b0}};
            out_beat_idx   <= {BEAT_IDX_W{1'b0}};
            out_pack_high  <= {((OUT_BEATS-1)*TILE_BITS){1'b0}};
        end else begin
            start_pulse <= 1'b0;
            case (frame_state)
                ST_ARM: begin start_pulse <= 1'b1; frame_state <= ST_RUN; end
                ST_RUN: begin if (sched_out_frame_done) frame_state <= ST_WAIT; end
                ST_WAIT: begin if (!mac_busy && !streaming_high_beats && !lib_valid_out_w) frame_state <= ST_ARM; end
                default: frame_state <= ST_ARM;
            endcase
            if (beat_early_now) in_beat_idx <= in_beat_idx + 1'b1;
            else if (beat_last_now) in_beat_idx <= {BEAT_IDX_W{1'b0}};
            if (lib_valid_out_w) begin
                out_pack_high <= lib_data_out_w[OUT_PIXEL_BITS-1:TILE_BITS];
                out_beat_idx  <= 3'd1;
            end else if (streaming_high_beats) begin
                if (out_beat_idx == OUT_BEATS-1) out_beat_idx <= {BEAT_IDX_W{1'b0}};
                else out_beat_idx <= out_beat_idx + 1'b1;
            end
        end
    end

    wire stall_in = mac_busy;

    coord_scheduler #(.IH(IH), .IW(IW), .OH(OH), .OW(OW), .KH(KH), .KW(KW), .SH(SH), .SW(SW), .PH(PH), .PW(PW)) scheduler (
        .clk(clk), .rst_n(rst_n), .start(start_pulse), .stall_in(stall_in),
        .valid_in(lib_valid_in_w), .ready_in(sched_ready_in),
        .needs_real_input(sched_needs_real_input),
        .in_row(sched_in_row), .in_col(sched_in_col),
        .output_fires(sched_output_fires), .advance(sched_advance),
        .in_frame_done(), .out_frame_done(sched_out_frame_done),
        .outputs_emitted(sched_outputs_emitted)
    );

    line_buf_window #(.IC(IC), .IW(IW), .IH(IH), .KH(KH), .KW(KW), .PW(PW), .PH(PH)) lbw (
        .clk(clk), .rst_n(rst_n), .frame_start(start_pulse),
        .sched_in_row(sched_in_row), .sched_in_col(sched_in_col),
        .sched_needs_real_input(sched_needs_real_input),
        .sched_advance(sched_advance), .sched_output_fires(sched_output_fires),
        .valid_in(lib_valid_in_w), .data_in(lib_data_in_w), .window_flat(window_flat)
    );

    conv_datapath_mp_k #(.DSP_INPUT_PIPE(1),.IC(IC), .OC(OC), .KH(KH), .KW(KW), .K_TOTAL(K_TOTAL), .MP(MP),
        .MP_K(9),
        .SCALE_MULT(SCALE_MULT), .SCALE_SHIFT(SCALE_SHIFT),
        .WEIGHTS_PATH("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_260_weights_mp_k_9.hex"),
        .BIAS_PATH("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_260_bias.hex")
    ) dp (
        .clk(clk), .rst_n(rst_n), .window_flat(window_flat),
        .start_mac(sched_output_fires),
        .valid_out(lib_valid_out_w), .data_out(lib_data_out_w), .mac_busy(mac_busy)
    );

endmodule
