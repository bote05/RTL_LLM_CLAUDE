// node_add_1: tiled-streaming INT8 residual add.
// LayerIR contract:
//   - input_width_bits = 512 (lhs tile [255:0] + rhs tile [511:256])
//   - output_width_bits = 256 (one 32-channel tile beat)
//   - OC = 256, channel_tile = 32, BEATS_PER_PIXEL = 8
//   - latency: first valid_out exactly 274 cycles after first valid_in
//   - lhs_scale=15.149..., rhs_scale=13.237..., out_scale=15.149...
//     r_lhs=1.0, r_rhs~0.87374186, FUSED_SHIFT=13:
//        LHS_FUSED_MULT=262144, RHS_FUSED_MULT=229046, HALF=131072
// FSM: IDLE -> GATHER (8 input beats) -> COMPUTE (6-cycle align +
//      256 channels through a 3-stage MAC pipeline) -> STREAM (8 output beats).
module node_add_1 (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             valid_in,
    output reg              ready_in,
    input  wire [511:0]     data_in,
    output reg              valid_out,
    input  wire             ready_out,   // [BP-FIX] downstream-ready: stall stream when low (was missing -> dropped beats)
    output reg  [255:0]     data_out
);

    // ----------------------------------------------------------------
    // Parameters
    // ----------------------------------------------------------------
    localparam integer OC              = 256;
    localparam integer CHANNEL_TILE    = 32;
    localparam integer BEATS_PER_PIXEL = 8;            // OC / CHANNEL_TILE
    localparam integer ALIGN_CYCLES    = 6;            // tunes latency to 274

    localparam integer FUSED_SHIFT                = 13;
    localparam signed [33:0] LHS_FUSED_MULT       = 34'sd8192;
    localparam signed [33:0] RHS_FUSED_MULT       = 34'sd4065;
    localparam signed [33:0] FUSED_ROUND_BIAS     = 34'sd4096;  // 1<<17

    localparam [1:0] S_IDLE    = 2'd0;
    localparam [1:0] S_GATHER  = 2'd1;
    localparam [1:0] S_COMPUTE = 2'd2;
    localparam [1:0] S_STREAM  = 2'd3;

    // ----------------------------------------------------------------
    // Storage
    // ----------------------------------------------------------------
    reg signed [7:0]  lhs_buf  [0:OC-1];
    reg signed [7:0]  rhs_buf  [0:OC-1];
    reg [255:0]       out_beats [0:BEATS_PER_PIXEL-1];

    // ----------------------------------------------------------------
    // Control regs
    // ----------------------------------------------------------------
    reg [1:0]  state;
    reg [3:0]  in_beat_count;
    reg [3:0]  out_beat_count;
    reg [3:0]  cur_beat_stream;
    reg [3:0]  align_count;
    reg [9:0]  ch_counter;

    // ----------------------------------------------------------------
    // 3-stage MAC pipeline regs
    //   stage 1: latch lhs/rhs from buffers
    //   stage 2: signed multiplies (DSP48E2)
    //   stage 3: sum + round bias (unconditional +HALF per add contract)
    // ----------------------------------------------------------------
    reg                 s1_valid, s2_valid, s3_valid;
    reg signed [7:0]    s1_lhs, s1_rhs;
    reg [9:0]           s1_ch, s2_ch, s3_ch;
    (* use_dsp = "yes" *) reg signed [33:0] s2_lhs_term;
    (* use_dsp = "yes" *) reg signed [33:0] s2_rhs_term;
    reg signed [33:0]   s3_sum_term;

    // ----------------------------------------------------------------
    // Combinational saturate of stage-3 result
    // ----------------------------------------------------------------
    wire signed [33:0] shifted_val;
    wire signed [7:0]  sat_val;

    assign shifted_val = s3_sum_term >>> FUSED_SHIFT;
    assign sat_val     = (shifted_val > 34'sd127)
                            ? 8'sd127
                            : (shifted_val < -34'sd128)
                                ? 8'sh80
                                : shifted_val[7:0];

    integer i;

    // ----------------------------------------------------------------
    // Block A : array writes (sync-only -- keeps lhs_buf / rhs_buf /
    // out_beats out of the async-reset block so Vivado can infer
    // distributed RAM cleanly).
    // ----------------------------------------------------------------
    always @(posedge clk) begin
        if (valid_in && ready_in) begin
            for (i = 0; i < CHANNEL_TILE; i = i + 1) begin
                lhs_buf[in_beat_count[2:0] * CHANNEL_TILE + i] <= data_in[i*8 +: 8];
                rhs_buf[in_beat_count[2:0] * CHANNEL_TILE + i] <= data_in[256 + i*8 +: 8];
            end
        end
        if (s3_valid) begin
            out_beats[s3_ch[7:5]][s3_ch[4:0]*8 +: 8] <= sat_val;
        end
    end

    // ----------------------------------------------------------------
    // Block B : control FSM, pipeline propagation, and output drives.
    // ----------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            ready_in        <= 1'b1;
            valid_out       <= 1'b0;
            data_out        <= 256'd0;
            in_beat_count   <= 4'd0;
            out_beat_count  <= 4'd0;
            cur_beat_stream <= 4'd0;
            align_count     <= 4'd0;
            ch_counter      <= 10'd0;
            s1_valid        <= 1'b0;
            s2_valid        <= 1'b0;
            s3_valid        <= 1'b0;
            s1_lhs          <= 8'sd0;
            s1_rhs          <= 8'sd0;
            s1_ch           <= 10'd0;
            s2_ch           <= 10'd0;
            s3_ch           <= 10'd0;
            s2_lhs_term     <= 34'sd0;
            s2_rhs_term     <= 34'sd0;
            s3_sum_term     <= 34'sd0;
        end else begin
            // Default pipeline propagation each cycle.
            s2_valid    <= s1_valid;
            s2_ch       <= s1_ch;
            s2_lhs_term <= s1_lhs * LHS_FUSED_MULT;
            s2_rhs_term <= s1_rhs * RHS_FUSED_MULT;

            s3_valid    <= s2_valid;
            s3_ch       <= s2_ch;
            s3_sum_term <= s2_lhs_term + s2_rhs_term + FUSED_ROUND_BIAS;  // [INVARIANT:ROUNDING]

            // valid_out is a per-beat pulse driven only by S_STREAM.
            valid_out <= 1'b0;

            case (state)
                S_IDLE: begin
                    ready_in <= 1'b1;     // [INVARIANT:READY_IN_GATING]
                    if (valid_in) begin
                        in_beat_count <= 4'd1;
                        state         <= S_GATHER;
                    end
                end

                S_GATHER: begin
                    if (valid_in) begin
                        if (in_beat_count == BEATS_PER_PIXEL - 1) begin
                            ready_in       <= 1'b0;       // [INVARIANT:READY_IN_GATING]
                            in_beat_count  <= 4'd0;
                            align_count    <= 4'd0;
                            ch_counter     <= 10'd0;
                            state          <= S_COMPUTE;
                        end else begin
                            in_beat_count  <= in_beat_count + 4'd1;
                        end
                    end
                end

                S_COMPUTE: begin
                    if (align_count < ALIGN_CYCLES) begin
                        align_count <= align_count + 4'd1;
                        s1_valid    <= 1'b0;
                    end else if (ch_counter < OC) begin
                        s1_valid   <= 1'b1;
                        s1_lhs     <= lhs_buf[ch_counter[7:0]];
                        s1_rhs     <= rhs_buf[ch_counter[7:0]];
                        s1_ch      <= ch_counter;
                        ch_counter <= ch_counter + 10'd1;
                    end else begin
                        s1_valid <= 1'b0;
                    end

                    if (s3_valid && (s3_ch == OC - 1)) begin
                        // [BP-FIX] Present beat 0 at the COMPUTE->STREAM transition (matches
                        // the proven node_add pattern). out_beat_count/cur_beat_stream point
                        // at the NEXT beat to present (1). The presented beat 0 is held until
                        // ready_out accepts it (the S_STREAM advance is gated on ready_out).
                        state            <= S_STREAM;
                        valid_out        <= 1'b1;          // [INVARIANT:VALID_OUT_LATENCY]
                        data_out         <= out_beats[0];
                        cur_beat_stream  <= 4'd1;
                        out_beat_count   <= 4'd1;
                    end
                end

                S_STREAM: begin
                    // [BP-FIX] Only advance / present a new beat when the downstream
                    // ACCEPTS the currently presented one (valid_out & ready_out). When
                    // ready_out is LOW, HOLD: re-assert valid_out (to override the global
                    // valid_out<=1'b0 default above) and leave data_out + counters
                    // untouched so the current beat persists -- no beat is dropped.
                    if (ready_out) begin
                        if (out_beat_count < BEATS_PER_PIXEL) begin
                            valid_out        <= 1'b1;      // [INVARIANT:VALID_OUT_LATENCY]
                            data_out         <= out_beats[out_beat_count[2:0]];
                            cur_beat_stream  <= out_beat_count + 4'd1;
                            out_beat_count   <= out_beat_count + 4'd1;
                        end else begin
                            // All BEATS_PER_PIXEL beats accepted -> return to IDLE.
                            // (valid_out drops via the global default this cycle.)
                            cur_beat_stream <= 4'd0;
                            out_beat_count  <= 4'd0;
                            ready_in        <= 1'b1;       // [INVARIANT:READY_IN_GATING]
                            state           <= S_IDLE;
                        end
                    end else begin
                        valid_out <= 1'b1;   // [INVARIANT:VALID_OUT_LATENCY] HOLD current beat
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
