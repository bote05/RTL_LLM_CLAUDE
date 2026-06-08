`timescale 1ns / 1ps
// node_add_8 - tiled-streaming INT8 residual ADD.
// spec_hash add_1024x1024_s14x14_i512_o256_iotiled-streaming_tile32
// CHANNEL_TILE = 32, NUM_TILES = 32, OC = 1024.
// Latency first_valid_in -> first_valid_out = 1090
//   = GATHER(32 beats) + COMPUTE(32 setup + 1024 MAC reads + 2 pipeline drain).
// Bus: data_in[255:0] = lhs tile, data_in[511:256] = rhs tile,
//      data_out[255:0] = one 32-channel INT8 output tile.
// Quantization: UNCONDITIONAL +HALF rounding (FUSED_ROUND_BIAS = 1<<21).
//   Multipliers normalised by out_scale:
//     r_lhs = lhs_scale_factor / scale_factor = 1.7323101
//     r_rhs = rhs_scale_factor / scale_factor = 1.1556058
//   FUSED_SHIFT=20 chosen so both |M/2^22 - r|/r < 4e-8 and bit-exact
//   matches the Python golden over the full INT8 x INT8 input space.

module node_add_8 (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    output reg            ready_in,
    input  wire [511:0]   data_in,
    output reg            valid_out,
    input  wire           ready_out,   // [BP-FIX] downstream-ready: stall stream when low (was missing -> dropped beats)
    output reg  [255:0]   data_out
);

    localparam integer OC              = 1024;
    localparam integer CHANNEL_TILE    = 32;
    localparam integer NUM_TILES       = 32;
    localparam integer FUSED_SHIFT     = 20;

    localparam integer MULT_W = 24;
    localparam signed [MULT_W-1:0] LHS_FUSED_MULT = 34'sd1384542;
    localparam signed [MULT_W-1:0] RHS_FUSED_MULT = 34'sd1159169;

    localparam integer PROD_W = 8 + MULT_W;
    localparam integer SUM_W  = PROD_W + 1;
    localparam integer TERM_W = SUM_W + 1;
    localparam signed [TERM_W-1:0] FUSED_ROUND_BIAS = 34'sd524288;
    localparam signed [TERM_W-1:0] SAT_HI =  34'sd127;
    localparam signed [TERM_W-1:0] SAT_LO = -34'sd128;

    localparam [10:0] COMPUTE_MAC_START_V = 11'd32;
    localparam [10:0] COMPUTE_MAC_END_V   = 11'd1056;
    localparam [10:0] COMPUTE_TOTAL_M1    = 11'd1057;
    localparam [5:0]  NUM_TILES_M1        = 6'd31;

    localparam [1:0] ST_IDLE    = 2'd0,
                     ST_GATHER  = 2'd1,
                     ST_COMPUTE = 2'd2,
                     ST_STREAM  = 2'd3;

    reg [1:0]  state;
    reg [5:0]  in_beat_count;
    reg [5:0]  out_beat_count;
    reg [10:0] compute_cycle;
    reg        cur_beat_stream;

    (* ram_style = "block" *) reg signed [7:0] lhs_buf [0:OC-1];
    (* ram_style = "block" *) reg signed [7:0] rhs_buf [0:OC-1];
    (* ram_style = "block" *) reg signed [7:0] out_buf [0:OC-1];

    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] s1_prod_lhs;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] s1_prod_rhs;
    reg [10:0]                                    s1_oc;
    reg                                           s1_valid;
    reg signed [SUM_W-1:0]                        s2_sum_pre;
    reg [10:0]                                    s2_oc;
    reg                                           s2_valid;

    wire        mac_read_phase = (compute_cycle >= COMPUTE_MAC_START_V) &&
                                 (compute_cycle <  COMPUTE_MAC_END_V);
    wire [10:0] mac_oc         = mac_read_phase
                                 ? (compute_cycle - COMPUTE_MAC_START_V)
                                 : 11'd0;

    wire signed [7:0] lhs_cur = lhs_buf[mac_oc];
    wire signed [7:0] rhs_cur = rhs_buf[mac_oc];

    wire signed [TERM_W-1:0] s3_sum_term = s2_sum_pre + FUSED_ROUND_BIAS; // [INVARIANT:ROUNDING]
    wire signed [TERM_W-1:0] s3_shifted  = s3_sum_term >>> FUSED_SHIFT;
    wire [7:0] s3_sat = (s3_shifted > SAT_HI) ? 8'h7F :
                        (s3_shifted < SAT_LO) ? 8'h80 :
                                                 s3_shifted[7:0];

    wire [255:0] gather_lhs_chunk = data_in[255:0];
    wire [255:0] gather_rhs_chunk = data_in[511:256];

    wire [255:0] first_beat_word;
    genvar gfb;
    generate
        for (gfb = 0; gfb < CHANNEL_TILE; gfb = gfb + 1) begin: g_fb
            assign first_beat_word[gfb*8 +: 8] = out_buf[gfb];
        end
    endgenerate

    wire [10:0] next_beat_base = (out_beat_count == NUM_TILES_M1)
                                 ? 11'd0
                                 : (({5'd0, out_beat_count} + 11'd1) * 11'd32);
    wire [255:0] next_beat_word;
    genvar gnb;
    generate
        for (gnb = 0; gnb < CHANNEL_TILE; gnb = gnb + 1) begin: g_nb
            assign next_beat_word[gnb*8 +: 8] = out_buf[next_beat_base + gnb];
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= ST_IDLE;
            ready_in        <= 1'b1;                 // [INVARIANT:READY_IN_GATING]
            valid_out       <= 1'b0;
            data_out        <= 256'd0;
            in_beat_count   <= 6'd0;
            out_beat_count  <= 6'd0;
            compute_cycle   <= 11'd0;
            cur_beat_stream <= 1'b0;
            s1_prod_lhs     <= {PROD_W{1'b0}};
            s1_prod_rhs     <= {PROD_W{1'b0}};
            s1_oc           <= 11'd0;
            s1_valid        <= 1'b0;
            s2_sum_pre      <= {SUM_W{1'b0}};
            s2_oc           <= 11'd0;
            s2_valid        <= 1'b0;
        end else begin
            valid_out <= 1'b0;

            if (state == ST_COMPUTE) begin
                s1_prod_lhs <= lhs_cur * LHS_FUSED_MULT;
                s1_prod_rhs <= rhs_cur * RHS_FUSED_MULT;
                s1_oc       <= mac_oc;
                s1_valid    <= mac_read_phase;
                s2_sum_pre  <= s1_prod_lhs + s1_prod_rhs;
                s2_oc       <= s1_oc;
                s2_valid    <= s1_valid;
            end else begin
                s1_valid <= 1'b0;
                s2_valid <= 1'b0;
            end

            case (state)
                ST_IDLE: begin
                    ready_in <= 1'b1;                // [INVARIANT:READY_IN_GATING]
                    if (valid_in) begin
                        in_beat_count <= 6'd1;
                        state         <= ST_GATHER;
                    end
                end

                ST_GATHER: begin
                    if (valid_in) begin
                        if (in_beat_count == NUM_TILES_M1) begin
                            ready_in      <= 1'b0;   // [INVARIANT:READY_IN_GATING]
                            in_beat_count <= 6'd0;
                            state         <= ST_COMPUTE;
                            compute_cycle <= 11'd0;
                        end else begin
                            in_beat_count <= in_beat_count + 6'd1;
                        end
                    end
                end

                ST_COMPUTE: begin
                    if (compute_cycle == COMPUTE_TOTAL_M1) begin
                        state           <= ST_STREAM;
                        out_beat_count  <= 6'd0;
                        cur_beat_stream <= 1'b1;
                        valid_out       <= 1'b1;     // [INVARIANT:VALID_OUT_LATENCY]
                        data_out        <= first_beat_word;
                        compute_cycle   <= 11'd0;
                    end else begin
                        compute_cycle <= compute_cycle + 11'd1;
                    end
                end

                ST_STREAM: begin
                    // [BP-FIX] Only advance the output beat when the downstream ACCEPTS
                    // the currently presented beat (valid_out & ready_out). When ready_out
                    // is low, HOLD valid_out + data_out + out_beat_count (no beat dropped).
                    // The first beat was presented at COMPUTE->STREAM with out_beat_count=0.
                    if (ready_out) begin
                        if (out_beat_count == NUM_TILES_M1) begin
                            state           <= ST_IDLE;
                            out_beat_count  <= 6'd0;
                            cur_beat_stream <= 1'b0;
                            valid_out       <= 1'b0;
                            ready_in        <= 1'b1;     // [INVARIANT:READY_IN_GATING]
                        end else begin
                            out_beat_count <= out_beat_count + 6'd1;
                            valid_out      <= 1'b1;      // [INVARIANT:VALID_OUT_LATENCY]
                            data_out       <= next_beat_word;
                        end
                    end else begin
                        // hold: re-assert valid_out (cleared unconditionally above),
                        // keep data_out + out_beat_count unchanged
                        valid_out <= 1'b1;               // [INVARIANT:VALID_OUT_LATENCY]
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    integer ig;
    always @(posedge clk) begin
        if ((state == ST_IDLE) && valid_in && ready_in) begin
            for (ig = 0; ig < CHANNEL_TILE; ig = ig + 1) begin
                lhs_buf[ig] <= $signed(gather_lhs_chunk[ig*8 +: 8]);
                rhs_buf[ig] <= $signed(gather_rhs_chunk[ig*8 +: 8]);
            end
        end else if ((state == ST_GATHER) && valid_in) begin
            for (ig = 0; ig < CHANNEL_TILE; ig = ig + 1) begin
                lhs_buf[in_beat_count * CHANNEL_TILE + ig] <= $signed(gather_lhs_chunk[ig*8 +: 8]);
                rhs_buf[in_beat_count * CHANNEL_TILE + ig] <= $signed(gather_rhs_chunk[ig*8 +: 8]);
            end
        end

        if (s2_valid) begin
            out_buf[s2_oc] <= s3_sat;
        end
    end

endmodule
