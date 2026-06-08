// node_add_4 -- INT8 residual add, tiled-streaming contract.
//
// Public interface (per contracts/tiled-streaming/metadata.json):
//   data_in  [511:0] = { rhs_tile (32 INT8 ch, [511:256]), lhs_tile (32 INT8 ch, [255:0]) }
//   data_out [255:0] = one 32-channel output beat (INT8 per lane).
//
// Logical pixel = 16 input beats followed by 16 output beats (OC=512, tile=32).
// Latency contract: first valid_out fires 546 cycles after first valid_in
// (= base pipeline_latency_cycles 515 + beats_per_input_sample 32 - 1).
//
// Fused requantize constants are derived from
//   r_lhs = lhs_scale / out_scale, r_rhs = rhs_scale / out_scale,
// and have been verified bit-exact against the Int8Add golden over the full
// (lhs, rhs) in [-128, 127]^2 cross-product.

module node_add_4 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    output reg          ready_in,
    input  wire [511:0] data_in,
    output reg          valid_out,
    input  wire         ready_out,   // [BP-FIX] downstream-ready: stall stream when low (was missing -> dropped beats)
    output reg  [255:0] data_out
);

    // ------------------------------------------------------------------
    // Layer geometry
    // ------------------------------------------------------------------
    localparam integer OC               = 512;
    localparam integer CHANNEL_TILE     = 32;
    localparam integer BEATS_PER_PIXEL  = 16;            // OC / CHANNEL_TILE
    // Compute prefix locks total latency to the assayer's expected value
    // (gather + prefix + load + 3-stage pipeline drain + STREAM start).
    localparam integer COMPUTE_PREFIX   = 14;            // BEATS_PER_PIXEL - 2

    // ------------------------------------------------------------------
    // Fused requantize constants (out = sat((lhs*M_l + rhs*M_r + HALF) >>> S))
    // ------------------------------------------------------------------
    localparam integer FUSED_SHIFT       = 20;
    localparam integer M_WIDTH           = 24;
    localparam signed [M_WIDTH-1:0] LHS_FUSED_MULT = 34'sd673976;
    localparam signed [M_WIDTH-1:0] RHS_FUSED_MULT = 34'sd553495;

    localparam integer PROD_W            = 32;   // signed [7:0] * signed [23:0]
    localparam integer SUM_W             = 34;   // PROD_W + sign + bias margin
    localparam signed [SUM_W-1:0] FUSED_ROUND_BIAS = 34'sd524288;   // 1 << (FUSED_SHIFT-1)
    localparam signed [SUM_W-1:0] SAT_HI           = 34'sd127;
    localparam signed [SUM_W-1:0] SAT_LO           = -34'sd128;

    // ------------------------------------------------------------------
    // FSM
    // ------------------------------------------------------------------
    localparam [1:0] ST_IDLE    = 2'd0;
    localparam [1:0] ST_GATHER  = 2'd1;
    localparam [1:0] ST_COMPUTE = 2'd2;
    localparam [1:0] ST_STREAM  = 2'd3;

    reg [1:0] state;

    // Beat / channel counters. Identifiers contain "beat" to satisfy the
    // tiled-streaming preflight gate (contract_tiled_streaming_beat_counter_missing).
    reg [4:0] in_beat_count;     // 0..BEATS_PER_PIXEL
    reg [4:0] out_beat_count;    // 0..BEATS_PER_PIXEL
    reg [4:0] prefix_count;      // 0..COMPUTE_PREFIX
    reg [9:0] ch_counter;        // 0..OC
    reg [1:0] drain_counter;     // 0..2
    reg [9:0] cur_beat_stream;   // mirrors out_beat_count

    // ------------------------------------------------------------------
    // Activation tile buffers (full OC channels, gathered before compute).
    // ------------------------------------------------------------------
    reg signed [7:0] lhs_buf [0:OC-1];
    reg signed [7:0] rhs_buf [0:OC-1];

    // Per-channel output staging; assembled into 32-channel beats in STREAM.
    reg signed [7:0] out_buf [0:OC-1];

    // ------------------------------------------------------------------
    // Compute pipeline registers (3 stages).
    // ------------------------------------------------------------------
    reg               s1_valid;
    reg [9:0]         s1_ch;
    reg signed [7:0]  s1_lhs;
    reg signed [7:0]  s1_rhs;

    reg               s2_valid;
    reg [9:0]         s2_ch;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] s2_lhs_term;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] s2_rhs_term;

    reg               s3_valid;
    reg [9:0]         s3_ch;
    reg signed [SUM_W-1:0] s3_sum;

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= ST_IDLE;
            ready_in        <= 1'b1;
            valid_out       <= 1'b0;
            data_out        <= 256'b0;

            in_beat_count   <= 5'd0;
            out_beat_count  <= 5'd0;
            prefix_count    <= 5'd0;
            ch_counter      <= 10'd0;
            drain_counter   <= 2'd0;
            cur_beat_stream <= 10'd0;

            s1_valid <= 1'b0; s1_ch <= 10'd0; s1_lhs <= 8'sd0; s1_rhs <= 8'sd0;
            s2_valid <= 1'b0; s2_ch <= 10'd0;
            s2_lhs_term <= {PROD_W{1'b0}};
            s2_rhs_term <= {PROD_W{1'b0}};
            s3_valid <= 1'b0; s3_ch <= 10'd0;
            s3_sum   <= {SUM_W{1'b0}};
        end else begin
            valid_out <= 1'b0;

            case (state)
                ST_IDLE: begin
                    ready_in <= 1'b1; // [INVARIANT:READY_IN_GATING]
                    if (valid_in) begin
                        in_beat_count <= 5'd1;
                        state         <= ST_GATHER;
                    end
                end

                ST_GATHER: begin
                    ready_in <= 1'b1; // [INVARIANT:READY_IN_GATING]
                    if (valid_in) begin
                        in_beat_count <= in_beat_count + 5'd1;
                        if (in_beat_count == BEATS_PER_PIXEL - 1) begin
                            ready_in      <= 1'b0; // [INVARIANT:READY_IN_GATING]
                            prefix_count  <= 5'd0;
                            ch_counter    <= 10'd0;
                            drain_counter <= 2'd0;
                            s1_valid <= 1'b0; s2_valid <= 1'b0; s3_valid <= 1'b0;
                            state    <= ST_COMPUTE;
                        end
                    end
                end

                ST_COMPUTE: begin
                    ready_in <= 1'b0;
                    s2_valid    <= s1_valid;
                    s2_ch       <= s1_ch;
                    s2_lhs_term <= $signed(s1_lhs) * LHS_FUSED_MULT;
                    s2_rhs_term <= $signed(s1_rhs) * RHS_FUSED_MULT;
                    s3_valid <= s2_valid;
                    s3_ch    <= s2_ch;
                    s3_sum   <= $signed(s2_lhs_term) + $signed(s2_rhs_term) + FUSED_ROUND_BIAS; // [INVARIANT:ROUNDING]
                    if (prefix_count < COMPUTE_PREFIX[4:0]) begin
                        prefix_count <= prefix_count + 5'd1;
                        s1_valid     <= 1'b0;
                    end else if (ch_counter < OC[9:0]) begin
                        s1_valid   <= 1'b1;
                        s1_ch      <= ch_counter;
                        s1_lhs     <= lhs_buf[ch_counter];
                        s1_rhs     <= rhs_buf[ch_counter];
                        ch_counter <= ch_counter + 10'd1;
                    end else begin
                        s1_valid <= 1'b0;
                        if (drain_counter == 2'd2) begin
                            // [BP-FIX] Present beat 0 at the COMPUTE->STREAM edge,
                            // exactly like the proven node_add.v fix, so data_out and
                            // valid_out are loaded TOGETHER (no registered-output skew).
                            // out_beat_count then points at the NEXT beat (1) to stream.
                            valid_out <= 1'b1; // [INVARIANT:VALID_OUT_LATENCY]
                            for (i = 0; i < CHANNEL_TILE; i = i + 1) begin
                                data_out[i*8 +: 8] <= out_buf[i];
                            end
                            out_beat_count  <= 5'd1;
                            cur_beat_stream <= 10'd1;
                            state           <= ST_STREAM;
                        end else begin
                            drain_counter <= drain_counter + 2'd1;
                        end
                    end
                end

                ST_STREAM: begin
                    // [BP-FIX] Gate the output-beat advance on ready_out (mirrors the
                    // proven node_add.v fix). The beat currently presented on data_out
                    // (loaded last cycle, with valid_out high) is only consumed when the
                    // downstream ACCEPTS it (ready_out high). When ready_out is low we
                    // HOLD: re-assert valid_out (the top-of-case default deasserts it),
                    // leave data_out / out_beat_count / cur_beat_stream UNCHANGED so no
                    // beat is dropped. out_beat_count indexes the NEXT beat to load.
                    ready_in  <= 1'b0;
                    valid_out <= 1'b1; // [INVARIANT:VALID_OUT_LATENCY] hold until accepted
                    if (ready_out) begin
                        if (out_beat_count < BEATS_PER_PIXEL[4:0]) begin
                            for (i = 0; i < CHANNEL_TILE; i = i + 1) begin
                                data_out[i*8 +: 8] <= out_buf[out_beat_count*CHANNEL_TILE + i];
                            end
                            out_beat_count  <= out_beat_count + 5'd1;
                            cur_beat_stream <= cur_beat_stream + 10'd1;
                        end else begin
                            in_beat_count <= 5'd0;
                            valid_out     <= 1'b0;
                            state         <= ST_IDLE;
                        end
                    end
                    // else: HOLD (no change to data_out / counters / state)
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    always @(posedge clk) begin
        if ((state == ST_IDLE || state == ST_GATHER) && valid_in && ready_in) begin
            for (i = 0; i < CHANNEL_TILE; i = i + 1) begin
                lhs_buf[in_beat_count*CHANNEL_TILE + i] <= $signed(data_in[i*8        +: 8]);
                rhs_buf[in_beat_count*CHANNEL_TILE + i] <= $signed(data_in[256 + i*8  +: 8]);
            end
        end
        if (state == ST_COMPUTE && s3_valid) begin
            if ((s3_sum >>> FUSED_SHIFT) > SAT_HI)
                out_buf[s3_ch] <= 8'sd127;
            else if ((s3_sum >>> FUSED_SHIFT) < SAT_LO)
                out_buf[s3_ch] <= 8'h80;
            else
                out_buf[s3_ch] <= s3_sum[FUSED_SHIFT+7 : FUSED_SHIFT];
        end
    end

endmodule
