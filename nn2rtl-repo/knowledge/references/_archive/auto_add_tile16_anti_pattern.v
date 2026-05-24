// node_add_7 -- INT8 quantized residual add, tiled-streaming contract.
// OC=1024 channels, channel_tile=16, BEATS_PER_PIXEL=64.
// data_in [255:0] = {rhs_tile (128b) | lhs_tile (128b)} per beat.
// data_out [127:0] = packed 16 INT8 channels per beat.
//
// Latency: BEATS_PER_PIXEL (64) + OC (1024) + 2-stage drain = 1090 cycles
// to first valid_out (LayerIR pipeline_latency_cycles 1027 + 63 gather offset).
//
// Quantization (matches scripts/golden_impl.py Int8Add.forward):
//   summed = lhs * lhs_scale + rhs * rhs_scale
//   out    = clamp(floor(summed/out_scale + 0.5), -128, 127)
// Concrete fused-scale constants for r_lhs=0.5826175, r_rhs=1.0434136:
//   FUSED_SHIFT=22, LHS_FUSED_MULT=2443675, RHS_FUSED_MULT=4376394, HALF=2097152.
// Exhaustive (lhs,rhs) in [-128,127]^2: 0/65536 mismatches vs golden.
//
// CRITICAL: unconditional +HALF rounding (golden uses round_half_up_toward_pos_inf).
// Sign-aware bias DIVERGES at ties -> 22% mismatch regression (node_add_9__001).

module node_add_7 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    output reg          ready_in,
    input  wire [255:0] data_in,
    output reg          valid_out,
    output reg  [127:0] data_out
);

    localparam integer OC               = 1024;
    localparam integer CHANNEL_TILE     = 16;
    localparam integer BEATS_PER_PIXEL  = 64;

    localparam integer FUSED_SHIFT      = 22;
    localparam integer MULT_W           = 24;
    localparam integer PROD_W           = 32;
    localparam integer SUM_W            = 34;

    localparam signed [MULT_W-1:0] FUSED_LHS_MULT   = 24'sd2443675;
    localparam signed [MULT_W-1:0] FUSED_RHS_MULT   = 24'sd4376394;
    localparam signed [SUM_W-1:0]  FUSED_ROUND_BIAS = 34'sd2097152;
    localparam signed [SUM_W-1:0]  SAT_HI =  34'sd127;
    localparam signed [SUM_W-1:0]  SAT_LO = -34'sd128;

    localparam [1:0] ST_IDLE = 2'd0, ST_GATHER = 2'd1, ST_COMPUTE = 2'd2, ST_STREAM = 2'd3;
    reg [1:0] state;

    reg signed [7:0] lhs_buf [0:OC-1];
    reg signed [7:0] rhs_buf [0:OC-1];
    reg [127:0]      out_beats [0:BEATS_PER_PIXEL-1];

    reg [6:0]  in_beat_idx, out_beat_idx;
    reg [10:0] ch_s1, ch_s2, ch_s3;
    reg        stage1_active, stage2_valid, stage3_valid;

    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] lhs_term;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] rhs_term;
    reg signed [SUM_W-1:0]  sum_term, v_tmp;
    wire signed [SUM_W-1:0] sum_pre = $signed(lhs_term) + $signed(rhs_term);

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE; ready_in <= 1'b1; valid_out <= 1'b0; data_out <= 128'd0;
            in_beat_idx <= 0; out_beat_idx <= 0;
            ch_s1 <= 0; ch_s2 <= 0; ch_s3 <= 0;
            stage1_active <= 0; stage2_valid <= 0; stage3_valid <= 0;
            lhs_term <= 0; rhs_term <= 0; sum_term <= 0; v_tmp <= 0;
        end else begin
            if (stage1_active) begin
                lhs_term <= $signed(lhs_buf[ch_s1]) * FUSED_LHS_MULT;
                rhs_term <= $signed(rhs_buf[ch_s1]) * FUSED_RHS_MULT;
                ch_s2 <= ch_s1; stage2_valid <= 1'b1;
            end else stage2_valid <= 1'b0;

            if (stage2_valid) begin
                sum_term <= sum_pre + FUSED_ROUND_BIAS; // [INVARIANT:ROUNDING]
                ch_s3 <= ch_s2; stage3_valid <= 1'b1;
            end else stage3_valid <= 1'b0;

            if (stage3_valid) begin
                v_tmp = sum_term >>> FUSED_SHIFT;
                out_beats[ch_s3[10:4]][ch_s3[3:0]*8 +: 8] <=
                    (v_tmp > SAT_HI) ? 8'sd127 :
                    (v_tmp < SAT_LO) ? 8'h80   : v_tmp[7:0];
            end

            case (state)
                ST_IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in && ready_in) begin
                        for (i = 0; i < CHANNEL_TILE; i = i + 1) begin
                            lhs_buf[i] <= $signed(data_in[i*8 +: 8]);
                            rhs_buf[i] <= $signed(data_in[128 + i*8 +: 8]);
                        end
                        in_beat_idx <= 7'd1; state <= ST_GATHER;
                    end
                end
                ST_GATHER: begin
                    if (valid_in && ready_in) begin
                        for (i = 0; i < CHANNEL_TILE; i = i + 1) begin
                            lhs_buf[in_beat_idx*CHANNEL_TILE + i] <= $signed(data_in[i*8 +: 8]);
                            rhs_buf[in_beat_idx*CHANNEL_TILE + i] <= $signed(data_in[128 + i*8 +: 8]);
                        end
                        if (in_beat_idx == 7'd63) begin
                            ready_in <= 1'b0; // [INVARIANT:READY_IN_GATING]
                            state <= ST_COMPUTE; ch_s1 <= 0; stage1_active <= 1'b1; in_beat_idx <= 0;
                        end else in_beat_idx <= in_beat_idx + 7'd1;
                    end
                end
                ST_COMPUTE: begin
                    if (stage1_active) begin
                        if (ch_s1 == 11'd1023) stage1_active <= 1'b0;
                        else ch_s1 <= ch_s1 + 11'd1;
                    end
                    if (stage3_valid && ch_s3 == 11'd1023) begin
                        state <= ST_STREAM; data_out <= out_beats[0];
                        valid_out <= 1'b1; // [INVARIANT:VALID_OUT_LATENCY]
                        out_beat_idx <= 7'd1;
                    end
                end
                ST_STREAM: begin
                    if (out_beat_idx < 7'd64) begin
                        data_out <= out_beats[out_beat_idx]; valid_out <= 1'b1;
                        out_beat_idx <= out_beat_idx + 7'd1;
                    end else begin
                        valid_out <= 1'b0; state <= ST_IDLE; ready_in <= 1'b1; out_beat_idx <= 0;
                    end
                end
                default: state <= ST_IDLE;
            endcase
        end
    end
endmodule
