// node_add — tiled-streaming INT8 residual add
//   contract_id : tiled-streaming
//   channel_tile=32, beats_per_pixel=8
//   data_in [255:0]   = lhs tile (32 INT8 channels)
//   data_in [511:256] = rhs tile (32 INT8 channels)
//   data_out[255:0]   = one 32-channel INT8 output tile per beat
//
//   Gather (8 beats)  -> Compute (serialized OC=256, 4-stage pipe)
//                     -> Stream  (8 output beats)
//   Beat counters: in_beat_count tracks the input tile index inside a
//   pixel, out_beat_count tracks the output tile index, cur_beat_stream
//   mirrors out_beat_count for downstream observers.

module node_add (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    output reg          ready_in,
    input  wire [511:0] data_in,
    output reg          valid_out,
    input  wire         ready_out,   // [BP-FIX] downstream-ready: stall stream when low (was missing -> dropped beats)
    output reg  [255:0] data_out
);

    localparam integer OC               = 256;
    localparam integer CHANNEL_TILE     = 32;
    localparam integer BEATS_PER_PIXEL  = 8;

    localparam integer        FUSED_SHIFT      = 22;
    localparam signed [24:0]  LHS_FUSED_MULT   = 34'sd1449288;
    localparam signed [24:0]  RHS_FUSED_MULT   = 34'sd4267903;
    localparam signed [34:0]  FUSED_ROUND_BIAS = 34'sd2097152;

    localparam [9:0]  LAST_COMPUTE_IDX = 10'd265;
    localparam [3:0]  LAST_GATHER_IDX  = 4'd7;
    localparam [3:0]  BPP_4            = 4'd8;

    localparam [1:0]  ST_IDLE    = 2'd0;
    localparam [1:0]  ST_GATHER  = 2'd1;
    localparam [1:0]  ST_COMPUTE = 2'd2;
    localparam [1:0]  ST_STREAM  = 2'd3;

    (* ram_style = "block" *) reg signed [7:0] lhs_buf [0:OC-1];
    (* ram_style = "block" *) reg signed [7:0] rhs_buf [0:OC-1];
    (* ram_style = "block" *) reg signed [7:0] out_buf [0:OC-1];

    reg [1:0]  state;
    reg [3:0]  in_beat_count;
    reg [9:0]  compute_cycle;
    reg [3:0]  out_beat_count;
    reg [3:0]  cur_beat_stream;

    wire [9:0] ch_in       = compute_cycle;
    wire       ch_in_valid = (state == ST_COMPUTE) && (compute_cycle < 10'd256);
    wire accept_beat = ((state == ST_IDLE) || (state == ST_GATHER)) && valid_in && ready_in;
    wire [8:0] gather_oc_base = {in_beat_count, 5'b0};
    wire [8:0] stream_oc_base = {out_beat_count, 5'b0};

    reg signed [7:0]   op_lhs_a, op_rhs_a;
    reg        [9:0]   ch_a;
    reg                vld_a;
    (* use_dsp = "yes" *) reg signed [32:0] lhs_term_b;
    (* use_dsp = "yes" *) reg signed [32:0] rhs_term_b;
    reg        [9:0]   ch_b;
    reg                vld_b;
    reg signed [34:0]  sum_term_c;
    reg        [9:0]   ch_c;
    reg                vld_c;
    reg signed [34:0]  scaled_d;
    reg        [9:0]   ch_d;
    reg                vld_d;

    integer ti;
    integer si;

    always @(posedge clk) begin
        if (accept_beat) begin
            for (ti = 0; ti < CHANNEL_TILE; ti = ti + 1) begin
                lhs_buf[gather_oc_base + ti] <= data_in[ti*8 +: 8];
                rhs_buf[gather_oc_base + ti] <= data_in[256 + ti*8 +: 8];
            end
        end
        if (vld_d) begin
            if (scaled_d > 35'sd127)        out_buf[ch_d] <= 8'sd127;
            else if (scaled_d < -35'sd128)  out_buf[ch_d] <= 8'h80;
            else                            out_buf[ch_d] <= scaled_d[7:0];
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            ready_in <= 1'b1;           // [INVARIANT:READY_IN_GATING]
            valid_out <= 1'b0;
            data_out <= 256'd0;
            in_beat_count <= 4'd0; compute_cycle <= 10'd0;
            out_beat_count <= 4'd0; cur_beat_stream <= 4'd0;
            op_lhs_a <= 8'sd0; op_rhs_a <= 8'sd0; ch_a <= 10'd0; vld_a <= 1'b0;
            lhs_term_b <= 33'sd0; rhs_term_b <= 33'sd0; ch_b <= 10'd0; vld_b <= 1'b0;
            sum_term_c <= 35'sd0; ch_c <= 10'd0; vld_c <= 1'b0;
            scaled_d <= 35'sd0; ch_d <= 10'd0; vld_d <= 1'b0;
        end else begin
            op_lhs_a <= lhs_buf[ch_in];
            op_rhs_a <= rhs_buf[ch_in];
            ch_a <= ch_in; vld_a <= ch_in_valid;
            lhs_term_b <= $signed(op_lhs_a) * LHS_FUSED_MULT;
            rhs_term_b <= $signed(op_rhs_a) * RHS_FUSED_MULT;
            ch_b <= ch_a; vld_b <= vld_a;
            sum_term_c <= $signed(lhs_term_b) + $signed(rhs_term_b) + FUSED_ROUND_BIAS;  // [INVARIANT:ROUNDING]
            ch_c <= ch_b; vld_c <= vld_b;
            scaled_d <= sum_term_c >>> FUSED_SHIFT;
            ch_d <= ch_c; vld_d <= vld_c;
            case (state)
                ST_IDLE: begin
                    valid_out <= 1'b0;
                    ready_in <= 1'b1;   // [INVARIANT:READY_IN_GATING]
                    if (valid_in) begin in_beat_count <= 4'd1; state <= ST_GATHER; end
                end
                ST_GATHER: begin
                    if (valid_in) begin
                        if (in_beat_count == LAST_GATHER_IDX) begin
                            ready_in <= 1'b0;   // [INVARIANT:READY_IN_GATING]
                            state <= ST_COMPUTE; compute_cycle <= 10'd0; in_beat_count <= 4'd0;
                        end else in_beat_count <= in_beat_count + 4'd1;
                    end
                end
                ST_COMPUTE: begin
                    if (compute_cycle == LAST_COMPUTE_IDX) begin
                        state <= ST_STREAM;
                        valid_out <= 1'b1;   // [INVARIANT:VALID_OUT_LATENCY]
                        for (si = 0; si < CHANNEL_TILE; si = si + 1)
                            data_out[si*8 +: 8] <= out_buf[si];
                        out_beat_count <= 4'd1;
                        cur_beat_stream <= 4'd1;
                    end else compute_cycle <= compute_cycle + 10'd1;
                end
                ST_STREAM: begin
                    // [BP-FIX] Only advance when the downstream ACCEPTS the currently
                    // presented beat (valid_out & ready_out). When ready_out is low,
                    // HOLD valid_out + data_out + out_beat_count (no drop). Beat 0 was
                    // presented at the COMPUTE->STREAM transition with out_beat_count=1.
                    if (ready_out) begin
                        if (out_beat_count < BPP_4) begin
                            valid_out <= 1'b1;   // [INVARIANT:VALID_OUT_LATENCY]
                            for (si = 0; si < CHANNEL_TILE; si = si + 1)
                                data_out[si*8 +: 8] <= out_buf[stream_oc_base + si];
                            out_beat_count <= out_beat_count + 4'd1;
                            cur_beat_stream <= out_beat_count + 4'd1;
                        end else begin
                            state <= ST_IDLE; valid_out <= 1'b0; ready_in <= 1'b1;
                            out_beat_count <= 4'd0; cur_beat_stream <= 4'd0;
                        end
                    end
                end
                default: state <= ST_IDLE;
            endcase
        end
    end
endmodule
