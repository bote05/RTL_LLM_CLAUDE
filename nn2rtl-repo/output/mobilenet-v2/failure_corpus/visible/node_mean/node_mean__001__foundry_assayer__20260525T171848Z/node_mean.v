// node_mean - INT8 global average pool, flat-bus contract
// op_type:        global_avg_pool
// input shape:    [1, 1280, 7, 7]   (49 spatial beats, 1280 channels packed per beat)
// output shape:   [1, 1280, 1, 1]   (1 beat, 1280 channels packed)
// bus widths:     in=10240 bits, out=10240 bits (1280 * INT8)
// pipeline:       49 accumulate cycles + 3 tail stages (SCALE, ROUND, PACK) = 52 cycles
// scale_factor:   0.029064472933370725  ->  MULT=7619, SHIFT=18  (the 1/H*W divisor is folded in)

module node_mean (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                valid_in,
    output reg                 ready_in,
    input  wire [10239:0]      data_in,
    output reg                 valid_out,
    output reg  [10239:0]      data_out
);

    localparam integer C             = 1280;
    localparam integer HW            = 49;
    localparam integer ACC_W         = 16;
    localparam integer SCALE_MULT    = 7619;
    localparam integer SCALE_SHIFT   = 18;
    localparam integer SCALE_CONST_W = 14;
    localparam integer SCALED_W      = ACC_W + SCALE_CONST_W;
    localparam integer ROUNDED_W     = 16;

    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST = SCALE_MULT;

    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};

    localparam [1:0] ST_ACCUM = 2'd0;
    localparam [1:0] ST_SCALE = 2'd1;
    localparam [1:0] ST_ROUND = 2'd2;
    localparam [1:0] ST_PACK  = 2'd3;

    reg [1:0] state;
    reg [6:0] cell_count;

    reg signed [ACC_W-1:0]     acc     [0:C-1];
    reg signed [SCALED_W-1:0]  scaled  [0:C-1];
    reg signed [ROUNDED_W-1:0] rounded [0:C-1];
    reg signed [SCALED_W-1:0]  v_tmp;

    integer i;

    always @(posedge clk) begin
        if (state == ST_ACCUM && valid_in && ready_in) begin
            if (cell_count == 7'd0) begin
                for (i = 0; i < C; i = i + 1)
                    acc[i] <= $signed(data_in[i*8 +: 8]);
            end else begin
                for (i = 0; i < C; i = i + 1)
                    acc[i] <= acc[i] + $signed(data_in[i*8 +: 8]);
            end
        end

        if (state == ST_SCALE) begin
            for (i = 0; i < C; i = i + 1)
                scaled[i] <= $signed(acc[i]) * $signed(SCALE_MULT_CONST);
        end

        if (state == ST_ROUND) begin
            for (i = 0; i < C; i = i + 1) begin
                v_tmp = (scaled[i] +
                         (scaled[i][SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                                : SCALE_ROUND_HALF)
                        ) >>> SCALE_SHIFT; // [INVARIANT:ROUNDING]
                rounded[i] <= v_tmp[ROUNDED_W-1:0];
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= ST_ACCUM;
            cell_count <= 7'd0;
            ready_in   <= 1'b1;
            valid_out  <= 1'b0;
            data_out   <= {10240{1'b0}};
        end else begin
            valid_out <= 1'b0;
            case (state)
                ST_ACCUM: begin
                    if (valid_in && ready_in) begin
                        if (cell_count == HW - 1) begin
                            cell_count <= 7'd0;
                            ready_in   <= 1'b0; // [INVARIANT:READY_IN_GATING]
                            state      <= ST_SCALE;
                        end else begin
                            cell_count <= cell_count + 7'd1;
                        end
                    end
                end

                ST_SCALE: state <= ST_ROUND;
                ST_ROUND: state <= ST_PACK;

                ST_PACK: begin
                    for (i = 0; i < C; i = i + 1) begin
                        data_out[i*8 +: 8] <= (rounded[i] > 16'sd127)  ?  8'sd127 :
                                              (rounded[i] < -16'sd128) ? -8'sd128 :
                                                                          rounded[i][7:0];
                    end
                    valid_out <= 1'b1; // [INVARIANT:VALID_OUT_LATENCY]
                    ready_in  <= 1'b1; // [INVARIANT:READY_IN_GATING]
                    state     <= ST_ACCUM;
                end

                default: state <= ST_ACCUM;
            endcase
        end
    end

endmodule
