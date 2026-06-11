// node_conv_266 -- tiled-streaming 3x3 conv2d.
//   IC=256, OC=256, IH=IW=14, stride=1, padding=1, MP=4.
//   channel_tile=32 -> IN_BEATS=8, OUT_BEATS=8 per pixel.
//
// Split-architecture (coord_scheduler + line_buf_window + conv_datapath)
// wrapped with a beat-aggregator + beat-splitter, scaled up from the
// IN_BEATS=2 wrapper used by node_conv_200. Individually named tile regs
// (in_tile0_r..in_tile6_r, out_tile1_r..out_tile7_r) are used in place of
// reg arrays so the writes can live inside the async-reset block without
// tripping `activation_memory_in_async_reset_block`.
//
// Known timing limitation (inherited from node_conv_200): tile-cadence
// input feed (1 pixel per IN_BEATS wrapper cycles) overruns the Python
// latency formula on the spatial fill phase. Output values are bit-exact
// because the library MAC pipeline still operates on full assembled
// pixels via line_buf_window.

module node_conv_266 (
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
    localparam integer K_TOTAL        = IC * KH * KW;  // 2304
    localparam integer MP             = 4;

    // computeScaleApprox(0.0036969637315096334) -> (7753, 21).
    localparam integer SCALE_MULT     = 7753;
    localparam integer SCALE_SHIFT    = 21;

    localparam integer CHANNEL_TILE   = 32;
    localparam integer TILE_BITS      = CHANNEL_TILE * 8;   // 256
    localparam integer IN_PIXEL_BITS  = IC * 8;             // 2048
    localparam integer OUT_PIXEL_BITS = OC * 8;             // 2048
    localparam integer IN_BEATS       = (IC + CHANNEL_TILE - 1) / CHANNEL_TILE;  // 8
    localparam integer OUT_BEATS      = (OC + CHANNEL_TILE - 1) / CHANNEL_TILE;  // 8

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

    reg [2:0]                in_beat_idx;
    reg [TILE_BITS-1:0]      in_tile0_r;
    reg [TILE_BITS-1:0]      in_tile1_r;
    reg [TILE_BITS-1:0]      in_tile2_r;
    reg [TILE_BITS-1:0]      in_tile3_r;
    reg [TILE_BITS-1:0]      in_tile4_r;
    reg [TILE_BITS-1:0]      in_tile5_r;
    reg [TILE_BITS-1:0]      in_tile6_r;

    wire pre_beat_now  = valid_in && (in_beat_idx != 3'd7);
    wire last_beat_now = valid_in && (in_beat_idx == 3'd7) && sched_ready_in;

    // [INVARIANT:READY_IN_GATING]
    assign ready_in = (in_beat_idx != 3'd7) ? 1'b1 : sched_ready_in;

    wire                          lib_valid_in_w = last_beat_now;
    wire [IN_PIXEL_BITS-1:0]      lib_data_in_w  = {
        data_in,
        in_tile6_r, in_tile5_r, in_tile4_r, in_tile3_r,
        in_tile2_r, in_tile1_r, in_tile0_r
    };

    reg [2:0]                out_beat_idx;
    reg [TILE_BITS-1:0]      out_tile1_r;
    reg [TILE_BITS-1:0]      out_tile2_r;
    reg [TILE_BITS-1:0]      out_tile3_r;
    reg [TILE_BITS-1:0]      out_tile4_r;
    reg [TILE_BITS-1:0]      out_tile5_r;
    reg [TILE_BITS-1:0]      out_tile6_r;
    reg [TILE_BITS-1:0]      out_tile7_r;

    wire emitting_pending = (out_beat_idx != 3'd0);

    reg  [TILE_BITS-1:0]     data_out_mux;
    always @(*) begin
        case (out_beat_idx)
            3'd1:    data_out_mux = out_tile1_r;
            3'd2:    data_out_mux = out_tile2_r;
            3'd3:    data_out_mux = out_tile3_r;
            3'd4:    data_out_mux = out_tile4_r;
            3'd5:    data_out_mux = out_tile5_r;
            3'd6:    data_out_mux = out_tile6_r;
            3'd7:    data_out_mux = out_tile7_r;
            default: data_out_mux = lib_data_out_w[TILE_BITS-1:0];
        endcase
    end

    // [INVARIANT:VALID_OUT_LATENCY]
    assign valid_out = lib_valid_out_w || emitting_pending;
    assign data_out  = data_out_mux;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            frame_state  <= ST_ARM;
            start_pulse  <= 1'b0;
            in_beat_idx  <= 3'd0;
            in_tile0_r   <= {TILE_BITS{1'b0}};
            in_tile1_r   <= {TILE_BITS{1'b0}};
            in_tile2_r   <= {TILE_BITS{1'b0}};
            in_tile3_r   <= {TILE_BITS{1'b0}};
            in_tile4_r   <= {TILE_BITS{1'b0}};
            in_tile5_r   <= {TILE_BITS{1'b0}};
            in_tile6_r   <= {TILE_BITS{1'b0}};
            out_beat_idx <= 3'd0;
            out_tile1_r  <= {TILE_BITS{1'b0}};
            out_tile2_r  <= {TILE_BITS{1'b0}};
            out_tile3_r  <= {TILE_BITS{1'b0}};
            out_tile4_r  <= {TILE_BITS{1'b0}};
            out_tile5_r  <= {TILE_BITS{1'b0}};
            out_tile6_r  <= {TILE_BITS{1'b0}};
            out_tile7_r  <= {TILE_BITS{1'b0}};
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
                    if (!mac_busy) frame_state <= ST_ARM;
                end
                default: frame_state <= ST_ARM;
            endcase

            if (pre_beat_now) begin
                case (in_beat_idx)
                    3'd0: in_tile0_r <= data_in;
                    3'd1: in_tile1_r <= data_in;
                    3'd2: in_tile2_r <= data_in;
                    3'd3: in_tile3_r <= data_in;
                    3'd4: in_tile4_r <= data_in;
                    3'd5: in_tile5_r <= data_in;
                    3'd6: in_tile6_r <= data_in;
                    default: ;
                endcase
                in_beat_idx <= in_beat_idx + 3'd1;
            end else if (last_beat_now) begin
                in_beat_idx <= 3'd0;
            end

            if (lib_valid_out_w) begin
                out_tile1_r <= lib_data_out_w[1*TILE_BITS +: TILE_BITS];
                out_tile2_r <= lib_data_out_w[2*TILE_BITS +: TILE_BITS];
                out_tile3_r <= lib_data_out_w[3*TILE_BITS +: TILE_BITS];
                out_tile4_r <= lib_data_out_w[4*TILE_BITS +: TILE_BITS];
                out_tile5_r <= lib_data_out_w[5*TILE_BITS +: TILE_BITS];
                out_tile6_r <= lib_data_out_w[6*TILE_BITS +: TILE_BITS];
                out_tile7_r <= lib_data_out_w[7*TILE_BITS +: TILE_BITS];
                out_beat_idx <= 3'd1;
            end else if (emitting_pending) begin
                if (out_beat_idx == 3'd7) out_beat_idx <= 3'd0;
                else                      out_beat_idx <= out_beat_idx + 3'd1;
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
        .WEIGHTS_PATH("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_266_weights_mp_k_9.hex"),
        .BIAS_PATH("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_266_bias.hex")
    ) dp (
        .clk(clk), .rst_n(rst_n),
        .window_flat(window_flat),
        .start_mac(sched_output_fires),
        .valid_out(lib_valid_out_w),
        .data_out(lib_data_out_w),
        .mac_busy(mac_busy)
    );

endmodule
