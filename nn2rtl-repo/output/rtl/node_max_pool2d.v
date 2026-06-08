// node_max_pool2d
// Tiled-streaming maxpool, IC=OC=64, IH=IW=112, OH=OW=56,
// KH=KW=3, SH=SW=2, PH=PW=1, channel_tile=32, beat=256 bits.
//
// Topology: SINGLE line_buf_window with IC=64 (full channels).
// Beat-aggregator combines 2 input beats into one 512-bit pixel
// before feeding the library. Beat-splitter emits low tile in the
// SAME cycle as sched_output_fires (no intermediate latch state),
// per 07_maxpool.md "Output beat-splitter" pattern.
//
// Pattern: doc 07_maxpool.md "single-lbw IC=full" pattern + 2-state
// emitter. The prior 3-state (ST_IDLE/ST_LATCH/ST_EMIT1) FSM shifted
// the tile sequence by one beat (got[0..31]==expected[32..63]) and
// also leaked the final beat into a pending_rearm clear race.
module node_max_pool2d (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    output wire         ready_in,
    input  wire [255:0] data_in,
    output reg          valid_out,
    input  wire         ready_out,
    output reg  [255:0] data_out
);

    localparam integer IC              = 64;
    localparam integer OC              = 64;
    localparam integer IH              = 112;
    localparam integer IW              = 112;
    localparam integer OH              = 56;
    localparam integer OW              = 56;
    localparam integer KH              = 3;
    localparam integer KW              = 3;
    localparam integer SH              = 2;
    localparam integer SW              = 2;
    localparam integer PH              = 1;
    localparam integer PW              = 1;
    localparam integer TILE_CH         = 32;
    localparam integer TILE_BITS       = TILE_CH * 8;
    localparam integer IN_PIXEL_BITS   = IC * 8;

    reg                         armed;
    reg                         start_pulse;
    reg                         pending_rearm;

    reg                         in_beat_idx;
    reg [TILE_BITS-1:0]         pixel_low_r;

    reg                         out_beat1_pending_r;
    reg [TILE_BITS-1:0]         out_pixel_high_r;

    wire                              sched_ready_in;
    wire                              sched_output_fires;
    wire                              sched_advance;
    wire                              sched_needs_real_input;
    wire [$clog2(IH + PH + 1)-1:0]    sched_in_row;
    wire [$clog2(IW + PW + 1)-1:0]    sched_in_col;
    wire                              sched_in_frame_done;
    wire                              sched_out_frame_done;
    wire [$clog2(OH * OW + 1)-1:0]    sched_outputs_emitted;
    wire [KH*KW*IC*8-1:0]             window_flat;

    // OPTION A: also stall the scheduler when consumer is not ready, so we
    // don't lose a fresh sched_output_fires while a prior beat is unconsumed.
    wire output_held  = valid_out && !ready_out;
    wire compute_busy = sched_output_fires || out_beat1_pending_r || output_held;

    wire [IN_PIXEL_BITS-1:0] lib_data_in  = {data_in, pixel_low_r};
    wire                     lib_valid_in = valid_in && (in_beat_idx == 1'b1) && !compute_busy;

    // [INVARIANT:READY_IN_GATING]
    assign ready_in = start_pulse           ? 1'b0
                    : compute_busy          ? 1'b0
                    : (in_beat_idx == 1'b0) ? 1'b1
                                            : sched_ready_in;

    wire input_handshake = valid_in && ready_in;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            armed         <= 1'b0;
            start_pulse   <= 1'b0;
            pending_rearm <= 1'b0;
        end else begin
            start_pulse <= 1'b0;
            if (sched_out_frame_done) begin
                pending_rearm <= 1'b1;
            end
            if (!armed) begin
                armed       <= 1'b1;
                start_pulse <= 1'b1;
            end else if (pending_rearm
                         && !sched_output_fires
                         && !out_beat1_pending_r) begin
                armed         <= 1'b0;
                pending_rearm <= 1'b0;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_beat_idx <= 1'b0;
            pixel_low_r <= {TILE_BITS{1'b0}};
        end else if (input_handshake) begin
            if (in_beat_idx == 1'b0) begin
                pixel_low_r <= data_in;
                in_beat_idx <= 1'b1;
            end else begin
                in_beat_idx <= 1'b0;
            end
        end
    end

    coord_scheduler #(
        .IH(IH), .IW(IW), .OH(OH), .OW(OW),
        .KH(KH), .KW(KW), .SH(SH), .SW(SW),
        .PH(PH), .PW(PW)
    ) u_sched (
        .clk             (clk),
        .rst_n           (rst_n),
        .start           (start_pulse),
        .stall_in        (compute_busy),
        .valid_in        (lib_valid_in),
        .ready_in        (sched_ready_in),
        .needs_real_input(sched_needs_real_input),
        .in_row          (sched_in_row),
        .in_col          (sched_in_col),
        .output_fires    (sched_output_fires),
        .advance         (sched_advance),
        .in_frame_done   (sched_in_frame_done),
        .out_frame_done  (sched_out_frame_done),
        .outputs_emitted (sched_outputs_emitted)
    );

    line_buf_window #(
        .IC(IC), .IW(IW), .IH(IH),
        .KH(KH), .KW(KW), .PW(PW), .PH(PH)
    ) u_lbw (
        .clk                   (clk),
        .rst_n                 (rst_n),
        .frame_start           (start_pulse),
        .sched_in_row          (sched_in_row),
        .sched_in_col          (sched_in_col),
        .sched_needs_real_input(sched_needs_real_input),
        .sched_advance         (sched_advance),
        .sched_output_fires    (sched_output_fires),
        .valid_in              (lib_valid_in),
        .data_in               (lib_data_in),
        .window_flat           (window_flat)
    );

    integer                     ch_idx;
    integer                     kh_idx;
    integer                     kw_idx;
    reg signed [7:0]            max_v;
    reg signed [7:0]            tap_v;
    reg [IN_PIXEL_BITS-1:0]     max_pack_w;

    always @* begin
        max_pack_w = {IN_PIXEL_BITS{1'b0}};
        for (ch_idx = 0; ch_idx < OC; ch_idx = ch_idx + 1) begin
            max_v = $signed(window_flat[(0*KW*IC + 0*IC + ch_idx)*8 +: 8]);
            for (kh_idx = 0; kh_idx < KH; kh_idx = kh_idx + 1) begin
                for (kw_idx = 0; kw_idx < KW; kw_idx = kw_idx + 1) begin
                    tap_v = $signed(window_flat[(kh_idx*KW*IC + kw_idx*IC + ch_idx)*8 +: 8]);
                    if (tap_v > max_v) max_v = tap_v;
                end
            end
            max_pack_w[ch_idx*8 +: 8] = max_v;
        end
    end

    // 2-state beat-splitter per 07_maxpool.md.
    // Cycle K  (sched_output_fires=1): emit tile 0 (low channels 0..31),
    //   latch tile 1 high pack, set out_beat1_pending_r.
    // Cycle K+1 (out_beat1_pending_r=1): emit tile 1 (high channels 32..63),
    //   clear pending.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out           <= 1'b0;
            data_out            <= {TILE_BITS{1'b0}};
            out_beat1_pending_r <= 1'b0;
            out_pixel_high_r    <= {TILE_BITS{1'b0}};
        end else if (valid_out && !ready_out) begin
            // OPTION A: hold valid_out + data_out + pending state until consumer accepts.
            valid_out           <= 1'b1;
        end else if (sched_output_fires) begin
            data_out            <= max_pack_w[TILE_BITS-1:0];                 // [INVARIANT:VALID_OUT_LATENCY]
            valid_out           <= 1'b1;
            out_pixel_high_r    <= max_pack_w[IN_PIXEL_BITS-1:TILE_BITS];
            out_beat1_pending_r <= 1'b1;
        end else if (out_beat1_pending_r) begin
            data_out            <= out_pixel_high_r;
            valid_out           <= 1'b1;
            out_beat1_pending_r <= 1'b0;
        end else begin
            valid_out           <= 1'b0;
        end
    end

endmodule
