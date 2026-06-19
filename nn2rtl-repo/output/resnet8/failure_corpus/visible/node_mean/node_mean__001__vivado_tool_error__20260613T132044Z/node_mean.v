// node_mean — global_avg_pool, 64ch x 8x8 -> 64ch x 1x1, INT8 per-tensor.
// Contract: flat-bus, packed_full. Bus: 512b in / 512b out (64 INT8 channels per beat).
// scale_factor = 0.0874008134 = (input_scale / output_scale) / (H*W).
// SCALE_MULT=22910, SCALE_SHIFT=18 (relative err ~ 4.3e-6).
// Latency contract: 67 cycles between first valid_in and first valid_out.
//   ACC: 64 beats (HW=64) -> BIASED latch -> SCALE multiply -> OUT round/clamp/emit.

module node_mean (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    output reg          ready_in,
    input  wire [511:0] data_in,
    output reg          valid_out,
    output reg  [511:0] data_out
);

    localparam integer C  = 64;
    localparam integer HW = 64;

    localparam integer SCALE_MULT  = 22910;
    localparam integer SCALE_SHIFT = 18;

    localparam integer ACC_W    = 16;
    localparam integer SCALE_W  = 16;
    localparam integer SCALED_W = 32;

    localparam signed [SCALE_W-1:0] SCALE_MULT_CONST = SCALE_MULT;
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};

    reg signed [ACC_W-1:0]    acc        [0:C-1];
    reg signed [ACC_W-1:0]    biased_reg [0:C-1];
    reg signed [SCALED_W-1:0] scaled     [0:C-1];

    localparam [2:0] ST_IDLE   = 3'd0;
    localparam [2:0] ST_ACC    = 3'd1;
    localparam [2:0] ST_BIASED = 3'd2;
    localparam [2:0] ST_SCALE  = 3'd3;
    localparam [2:0] ST_OUT    = 3'd4;

    reg [2:0] state;
    reg [6:0] beat_cnt;

    integer i;
    reg signed [SCALED_W-1:0] v_tmp;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= ST_IDLE;
            beat_cnt  <= 7'd0;
            ready_in  <= 1'b1; // [INVARIANT:READY_IN_GATING]
            valid_out <= 1'b0;
            data_out  <= 512'd0;
            for (i = 0; i < C; i = i + 1) begin
                acc[i]        <= {ACC_W{1'b0}};
                biased_reg[i] <= {ACC_W{1'b0}};
                scaled[i]     <= {SCALED_W{1'b0}};
            end
        end else begin
            valid_out <= 1'b0;
            case (state)
                ST_IDLE: begin
                    if (valid_in) begin
                        for (i = 0; i < C; i = i + 1) begin
                            acc[i] <= $signed(data_in[i*8 +: 8]);
                        end
                        beat_cnt <= 7'd1;
                        state    <= ST_ACC;
                    end
                end
                ST_ACC: begin
                    if (valid_in) begin
                        for (i = 0; i < C; i = i + 1) begin
                            acc[i] <= acc[i] + $signed(data_in[i*8 +: 8]);
                        end
                        if (beat_cnt == HW - 1) begin
                            state    <= ST_BIASED;
                            ready_in <= 1'b0; // [INVARIANT:READY_IN_GATING]
                        end
                        beat_cnt <= beat_cnt + 7'd1;
                    end
                end
                ST_BIASED: begin
                    for (i = 0; i < C; i = i + 1) begin
                        biased_reg[i] <= acc[i];
                    end
                    state <= ST_SCALE;
                end
                ST_SCALE: begin
                    for (i = 0; i < C; i = i + 1) begin
                        scaled[i] <= $signed(biased_reg[i]) * $signed(SCALE_MULT_CONST);
                    end
                    state <= ST_OUT;
                end
                ST_OUT: begin
                    for (i = 0; i < C; i = i + 1) begin
                        v_tmp = (scaled[i] +
                                 (scaled[i][SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                                        : SCALE_ROUND_HALF)
                                ) >>> SCALE_SHIFT; // [INVARIANT:ROUNDING]
                        data_out[i*8 +: 8] <= (v_tmp > 127)  ?  8'sd127 :
                                              (v_tmp < -128) ? -8'sd128 :
                                              v_tmp[7:0];
                    end
                    valid_out <= 1'b1; // [INVARIANT:VALID_OUT_LATENCY]
                    ready_in  <= 1'b1; // [INVARIANT:READY_IN_GATING]
                    beat_cnt  <= 7'd0;
                    state     <= ST_IDLE;
                end
                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
