// node_add_2 -- tiled-streaming INT8 residual add
//
// Public bus contract (tiled-streaming, channel_tile=32):
//   data_in [511:0]  = { rhs_tile[255:0], lhs_tile[255:0] }
//   data_out[255:0]  = one 32-channel output tile beat
//   BEATS_PER_PIXEL  = OC / CHANNEL_TILE = 256 / 32 = 8
//
// FSM: IDLE -> GATHER -> COMPUTE -> STREAM -> IDLE
//   - IDLE/GATHER: collect BEATS_PER_PIXEL beats into per-channel lhs_buf/rhs_buf
//   - COMPUTE: 3-stage signed MAC pipeline, one channel per cycle. Fused scale
//     multipliers are normalised by out_scale (r_lhs = lhs_scale/out_scale,
//     r_rhs = rhs_scale/out_scale), then quantised at FUSED_SHIFT = 13.
//     Unconditional +HALF rounding matches Int8Add round_half_up_toward_pos_inf.
//   - STREAM: emit BEATS_PER_PIXEL output beats from out_buf.
//
// Activation memories live in a sync-only always block so Vivado can infer
// BRAM/LUTRAM. The async-reset block carries scalar control + pipeline regs only.

module node_add_2 (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 valid_in,
    output reg                  ready_in,
    input  wire [511:0]         data_in,
    output reg                  valid_out,
    input  wire                 ready_out,   // [BP-FIX] downstream-ready: stall stream when low (was missing -> dropped beats)
    output reg  [255:0]         data_out
);

    // ------------------------------------------------------------------
    // ABI constants
    // ------------------------------------------------------------------
    localparam integer OC               = 256;
    localparam integer CHANNEL_TILE     = 32;
    localparam integer BEATS_PER_PIXEL  = 8;

    // COMPUTE phase runs OC + 8 cycles so first valid_out fires exactly
    // pipeline_latency_cycles + (beats_per_input_sample - 1) = 259 + 15 = 274
    // cycles after first valid_in, matching the orchestrator's
    // tiled-streaming latency contract for this layer.
    localparam integer COMP_TRIGGER     = OC + 8;  // = 264

    // ------------------------------------------------------------------
    // Fused scale constants (per protected/01_context + probationary doc)
    //   r_lhs = lhs_scale_factor / scale_factor ~= 0.96965170516952879
    //   r_rhs = rhs_scale_factor / scale_factor ~= 0.10803908186480508
    //   FUSED_SHIFT chosen jointly so both multipliers fit in <2^23.
    // ------------------------------------------------------------------
    localparam integer FUSED_SHIFT             = 13;
    localparam signed [23:0] LHS_FUSED_MULT    = 34'sd8192;
    localparam signed [23:0] RHS_FUSED_MULT    = 34'sd2405;
    localparam signed [33:0] FUSED_ROUND_BIAS  = 34'sd4096; // 1 << (FUSED_SHIFT - 1)

    // ------------------------------------------------------------------
    // FSM state encoding
    // ------------------------------------------------------------------
    localparam [1:0] IDLE    = 2'd0;
    localparam [1:0] GATHER  = 2'd1;
    localparam [1:0] COMPUTE = 2'd2;
    localparam [1:0] STREAM  = 2'd3;

    // ------------------------------------------------------------------
    // Control regs
    // ------------------------------------------------------------------
    reg [1:0] state;
    reg [3:0] in_beat_count;     // 0..BEATS_PER_PIXEL-1
    reg [3:0] out_beat_count;    // 0..BEATS_PER_PIXEL-1
    reg [3:0] cur_beat_stream;   // mirrors out_beat_count
    reg [8:0] comp_count;        // 0..COMP_TRIGGER

    // ------------------------------------------------------------------
    // Activation buffers and output staging (true memories -- sync-only writes)
    // ------------------------------------------------------------------
    (* ram_style = "block" *) reg signed [7:0] lhs_buf [0:OC-1];
    (* ram_style = "block" *) reg signed [7:0] rhs_buf [0:OC-1];
    (* ram_style = "distributed" *) reg signed [7:0] out_buf [0:OC-1];

    // ------------------------------------------------------------------
    // MAC pipeline scalar regs (3 stages)
    // ------------------------------------------------------------------
    reg               s1_valid;
    reg signed [7:0]  s1_lhs;
    reg signed [7:0]  s1_rhs;
    reg [7:0]         s1_ch;

    reg               s2_valid;
    reg [7:0]         s2_ch;
    (* use_dsp = "yes" *) reg signed [33:0] s2_lhs_term;
    (* use_dsp = "yes" *) reg signed [33:0] s2_rhs_term;

    reg               s3_valid;
    reg [7:0]         s3_ch;
    reg signed [33:0] s3_sum;

    // ------------------------------------------------------------------
    // Module-scope loop temporaries (Verilog-2001 strict scoping)
    // ------------------------------------------------------------------
    integer i;
    integer beat_offset;

    // ------------------------------------------------------------------
    // Combinational round + shift target for the requantize writeback
    // ------------------------------------------------------------------
    wire signed [33:0] s3_sum_term      = s3_sum + FUSED_ROUND_BIAS;
    wire signed [33:0] s3_round_shifted = s3_sum_term >>> FUSED_SHIFT;

    // ==================================================================
    // Sync-only block: writes to memory-style arrays (BRAM/LUTRAM-safe).
    // ==================================================================
    always @(posedge clk) begin
        // Activation gather: pack 32 lhs and 32 rhs channels per beat.
        if ((state == IDLE || state == GATHER) && valid_in && ready_in) begin
            beat_offset = in_beat_count * CHANNEL_TILE;
            for (i = 0; i < CHANNEL_TILE; i = i + 1) begin
                lhs_buf[beat_offset + i] <= $signed(data_in[i*8 +: 8]);
                rhs_buf[beat_offset + i] <= $signed(data_in[256 + i*8 +: 8]);
            end
        end

        // Stage-3 writeback: requantize + saturate into out_buf.
        // s3_round_shifted = (s3_sum + FUSED_ROUND_BIAS) >>> FUSED_SHIFT
        // is the unconditional +HALF round-half-up form. [INVARIANT:ROUNDING]
        if (state == COMPUTE && s3_valid) begin
            if (s3_round_shifted > 34'sd127)
                out_buf[s3_ch] <= 8'sd127;
            else if (s3_round_shifted < -34'sd128)
                out_buf[s3_ch] <= 8'h80;
            else
                out_buf[s3_ch] <= s3_round_shifted[7:0];
        end
    end

    // ==================================================================
    // Async-reset block: scalar control + MAC pipeline registers
    // ==================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= IDLE;
            ready_in        <= 1'b1;
            valid_out       <= 1'b0;
            data_out        <= 256'd0;
            in_beat_count   <= 4'd0;
            out_beat_count  <= 4'd0;
            cur_beat_stream <= 4'd0;
            comp_count      <= 9'd0;
            s1_valid        <= 1'b0;
            s1_lhs          <= 8'sd0;
            s1_rhs          <= 8'sd0;
            s1_ch           <= 8'd0;
            s2_valid        <= 1'b0;
            s2_ch           <= 8'd0;
            s2_lhs_term     <= 34'sd0;
            s2_rhs_term     <= 34'sd0;
            s3_valid        <= 1'b0;
            s3_ch           <= 8'd0;
            s3_sum          <= 34'sd0;
        end else begin
            valid_out <= 1'b0;

            case (state)
                IDLE: begin
                    ready_in <= 1'b1; // [INVARIANT:READY_IN_GATING]
                    if (valid_in) begin
                        in_beat_count <= 4'd1;
                        state         <= GATHER;
                    end
                end

                GATHER: begin
                    if (valid_in) begin
                        if (in_beat_count == BEATS_PER_PIXEL - 1) begin
                            ready_in      <= 1'b0; // [INVARIANT:READY_IN_GATING]
                            in_beat_count <= 4'd0;
                            comp_count    <= 9'd0;
                            s1_valid      <= 1'b0;
                            s2_valid      <= 1'b0;
                            s3_valid      <= 1'b0;
                            state         <= COMPUTE;
                        end else begin
                            in_beat_count <= in_beat_count + 4'd1;
                        end
                    end
                end

                COMPUTE: begin
                    comp_count <= comp_count + 9'd1;

                    if (comp_count < OC[8:0]) begin
                        s1_valid <= 1'b1;
                        s1_ch    <= comp_count[7:0];
                        s1_lhs   <= lhs_buf[comp_count];
                        s1_rhs   <= rhs_buf[comp_count];
                    end else begin
                        s1_valid <= 1'b0;
                    end

                    s2_valid    <= s1_valid;
                    s2_ch       <= s1_ch;
                    s2_lhs_term <= $signed(s1_lhs) * LHS_FUSED_MULT;
                    s2_rhs_term <= $signed(s1_rhs) * RHS_FUSED_MULT;

                    s3_valid <= s2_valid;
                    s3_ch    <= s2_ch;
                    s3_sum   <= s2_lhs_term + s2_rhs_term;

                    if (comp_count == COMP_TRIGGER[8:0]) begin
                        // [BP-FIX] Present beat 0 AT the COMPUTE->STREAM transition (like
                        // node_add.v) so STREAM is always entered with a real beat already
                        // on data_out. This prevents a spurious valid_out over stale
                        // data_out=0 when ready_out is low on the very first STREAM cycle.
                        state           <= STREAM;
                        valid_out       <= 1'b1; // [INVARIANT:VALID_OUT_LATENCY]
                        for (i = 0; i < CHANNEL_TILE; i = i + 1) begin
                            data_out[i*8 +: 8] <= out_buf[i]; // beat 0 (oc base 0)
                        end
                        out_beat_count  <= 4'd1;
                        cur_beat_stream <= 4'd1;
                    end
                end

                STREAM: begin
                    // [BP-FIX] Only advance the output beat when the downstream
                    // ACCEPTS the currently presented beat (valid_out & ready_out).
                    // When ready_out is LOW, HOLD valid_out + data_out + the beat
                    // counter so no beat is dropped. The terminal beat (== BPP) must
                    // DEASSERT valid_out on the IDLE transition so the held last beat
                    // is not re-captured into the next pixel.
                    if (ready_out) begin
                        if (out_beat_count < BEATS_PER_PIXEL[3:0]) begin
                            valid_out <= 1'b1; // [INVARIANT:VALID_OUT_LATENCY]
                            for (i = 0; i < CHANNEL_TILE; i = i + 1) begin
                                data_out[i*8 +: 8] <=
                                    out_buf[out_beat_count * CHANNEL_TILE + i];
                            end
                            out_beat_count  <= out_beat_count  + 4'd1;
                            cur_beat_stream <= out_beat_count + 4'd1;
                        end else begin
                            state           <= IDLE;
                            valid_out       <= 1'b0;
                            ready_in        <= 1'b1; // [INVARIANT:READY_IN_GATING]
                            out_beat_count  <= 4'd0;
                            cur_beat_stream <= 4'd0;
                        end
                    end else begin
                        // HOLD: re-assert valid_out + keep data_out + counter unchanged,
                        // but ONLY while a real beat is being presented. At the terminal
                        // index (== BPP) there is no beat to hold, so leave valid_out low.
                        if (out_beat_count < BEATS_PER_PIXEL[3:0])
                            valid_out <= 1'b1; // [INVARIANT:VALID_OUT_LATENCY]
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
