// node_add_3 -- INT8 residual add, contract: tiled-streaming
// channel_tile = 32, BEATS_PER_PIXEL = 16, OC = 512.
// Public bus: data_in[511:0] = {rhs_tile[511:256], lhs_tile[255:0]};
//             data_out[255:0] = one 32-channel output tile beat.
// 4-state FSM (IDLE / GATHER / COMPUTE / STREAM) per 05_add_quantized.md
// and the probationary tiled-add pattern doc.
//
// Quantisation: out = saturate(round((lhs * (lhs_sf/out_sf)
//                                   + rhs * (rhs_sf/out_sf)))).
// Fused multipliers (round(r * 2^FUSED_SHIFT)) -- normalisation by out_scale
// is folded into the constants. Using raw lhs_sf / rhs_sf alone would
// over-scale every output by out_scale (~50% mismatch on prior attempts).
// Rounding bias is unconditional +HALF -- sign-aware (HALF / HALF-1) ties
// diverge from round_half_up_toward_pos_inf and produced the ~22% mismatch
// regression on previous node_add_* attempts.

module node_add_3 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    output reg          ready_in,
    input  wire [511:0] data_in,
    output reg          valid_out,
    input  wire         ready_out,   // [BP-FIX] downstream-ready: stall stream when low (was missing -> dropped beats)
    output reg  [255:0] data_out
);

    localparam integer OC              = 512;
    localparam integer CHANNEL_TILE    = 32;
    localparam integer BEATS_PER_PIXEL = OC / CHANNEL_TILE; // 16
    localparam integer FUSED_SHIFT     = 20;

    localparam [4:0] BEATS_PER_PIXEL_5 = 5'd16;

    localparam signed [23:0] LHS_FUSED_MULT   = 34'sd916207;
    localparam signed [23:0] RHS_FUSED_MULT   = 34'sd487759;
    localparam signed [33:0] FUSED_ROUND_BIAS = 34'sd524288;

    localparam integer COMPUTE_DURATION = 530;

    localparam [1:0] S_IDLE    = 2'd0;
    localparam [1:0] S_GATHER  = 2'd1;
    localparam [1:0] S_COMPUTE = 2'd2;
    localparam [1:0] S_STREAM  = 2'd3;

    reg [1:0] state;

    reg [4:0]  in_beat_count;
    reg [4:0]  out_beat_count;
    reg [9:0]  cur_beat_stream;
    reg [9:0]  compute_cycle;

    reg signed [7:0] lhs_buf   [0:OC-1];
    reg signed [7:0] rhs_buf   [0:OC-1];
    reg signed [7:0] out_bytes [0:OC-1];

    reg                s0_valid, s1_valid, s2_valid;
    reg [9:0]          s0_ch, s1_ch, s2_ch;
    reg signed [7:0]   lhs_s0, rhs_s0;
    (* use_dsp = "yes" *) reg signed [33:0] lhs_term;
    (* use_dsp = "yes" *) reg signed [33:0] rhs_term;
    reg signed [34:0]  sum_term;

    reg signed [34:0]  rounded;
    reg signed [7:0]   sat_byte;

    integer i;
    integer j;

    always @* begin
        rounded = sum_term >>> FUSED_SHIFT;
        if (rounded > 35'sd127) begin
            sat_byte = 8'sd127;
        end else if (rounded < -35'sd128) begin
            sat_byte = -8'sd128;
        end else begin
            sat_byte = rounded[7:0];
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            ready_in        <= 1'b1;
            valid_out       <= 1'b0;
            data_out        <= 256'd0;
            in_beat_count   <= 5'd0;
            out_beat_count  <= 5'd0;
            cur_beat_stream <= 10'd0;
            compute_cycle   <= 10'd0;
            s0_valid        <= 1'b0;
            s1_valid        <= 1'b0;
            s2_valid        <= 1'b0;
            s0_ch           <= 10'd0;
            s1_ch           <= 10'd0;
            s2_ch           <= 10'd0;
            lhs_s0          <= 8'sd0;
            rhs_s0          <= 8'sd0;
            lhs_term        <= 34'sd0;
            rhs_term        <= 34'sd0;
            sum_term        <= 35'sd0;
        end else begin
            case (state)
                S_IDLE: begin
                    valid_out       <= 1'b0;
                    in_beat_count   <= 5'd0;
                    out_beat_count  <= 5'd0;
                    cur_beat_stream <= 10'd0;
                    compute_cycle   <= 10'd0;
                    s0_valid        <= 1'b0;
                    s1_valid        <= 1'b0;
                    s2_valid        <= 1'b0;
                    ready_in        <= 1'b1;
                    if (valid_in && ready_in) begin
                        in_beat_count <= 5'd1;
                        state         <= S_GATHER;
                    end
                end

                S_GATHER: begin
                    if (valid_in && ready_in) begin
                        if (in_beat_count == BEATS_PER_PIXEL - 1) begin
                            ready_in      <= 1'b0;
                            in_beat_count <= 5'd0;
                            compute_cycle <= 10'd0;
                            state         <= S_COMPUTE;
                        end else begin
                            in_beat_count <= in_beat_count + 5'd1;
                        end
                    end
                end

                S_COMPUTE: begin
                    if (compute_cycle < OC) begin
                        s0_valid <= 1'b1;
                        s0_ch    <= compute_cycle;
                        lhs_s0   <= lhs_buf[compute_cycle];
                        rhs_s0   <= rhs_buf[compute_cycle];
                    end else begin
                        s0_valid <= 1'b0;
                    end

                    s1_valid <= s0_valid;
                    s1_ch    <= s0_ch;
                    lhs_term <= $signed(lhs_s0) * $signed(LHS_FUSED_MULT);
                    rhs_term <= $signed(rhs_s0) * $signed(RHS_FUSED_MULT);

                    s2_valid <= s1_valid;
                    s2_ch    <= s1_ch;
                    sum_term <= lhs_term + rhs_term + FUSED_ROUND_BIAS;

                    if (compute_cycle == COMPUTE_DURATION) begin
                        state           <= S_STREAM;
                        valid_out       <= 1'b1;
                        out_beat_count  <= 5'd1;
                        cur_beat_stream <= 10'd1;
                        compute_cycle   <= 10'd0;
                        s0_valid        <= 1'b0;
                        s1_valid        <= 1'b0;
                        s2_valid        <= 1'b0;
                        for (i = 0; i < CHANNEL_TILE; i = i + 1) begin
                            data_out[i*8 +: 8] <= out_bytes[i];
                        end
                    end else begin
                        compute_cycle <= compute_cycle + 10'd1;
                    end
                end

                S_STREAM: begin
                    // [BP-FIX] Only advance when the downstream ACCEPTS the currently
                    // presented beat (valid_out & ready_out). When ready_out is low,
                    // HOLD valid_out + data_out + out_beat_count (no drop). Beat 0 was
                    // presented at the COMPUTE->STREAM transition with out_beat_count=1.
                    if (ready_out) begin
                        if (out_beat_count == BEATS_PER_PIXEL_5) begin
                            valid_out       <= 1'b0;
                            ready_in        <= 1'b1;
                            out_beat_count  <= 5'd0;
                            cur_beat_stream <= 10'd0;
                            state           <= S_IDLE;
                        end else begin
                            valid_out <= 1'b1;
                            for (i = 0; i < CHANNEL_TILE; i = i + 1) begin
                                data_out[i*8 +: 8] <= out_bytes[out_beat_count * CHANNEL_TILE + i];
                            end
                            out_beat_count  <= out_beat_count + 5'd1;
                            cur_beat_stream <= cur_beat_stream + 10'd1;
                        end
                    end
                    // else: hold (no change) -- valid_out stays 1, data_out & counters frozen
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    always @(posedge clk) begin
        if ((state == S_IDLE || state == S_GATHER) && valid_in && ready_in) begin
            for (j = 0; j < CHANNEL_TILE; j = j + 1) begin
                lhs_buf[in_beat_count * CHANNEL_TILE + j] <= data_in[j*8 +: 8];
                rhs_buf[in_beat_count * CHANNEL_TILE + j] <= data_in[(256 + j*8) +: 8];
            end
        end
        if (state == S_COMPUTE && s2_valid) begin
            out_bytes[s2_ch] <= sat_byte;
        end
    end

endmodule
