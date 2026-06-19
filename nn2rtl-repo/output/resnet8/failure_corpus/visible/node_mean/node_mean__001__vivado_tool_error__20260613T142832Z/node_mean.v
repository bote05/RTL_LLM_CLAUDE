// node_mean — global_avg_pool, 64ch x 8x8 -> 64ch, flat-bus, INT8.
// scale_factor=0.08740081342990223 -> SCALE_MULT=22913, SCALE_SHIFT=18.
// Latency = 1 (first beat) + 63 (remaining) + 3 (SCALE/ROUND/OUTPUT) = 67.
module node_mean #(
    parameter integer IC                = 64,
    parameter integer OC                = 64,
    parameter integer HW_TOTAL          = 64,
    parameter integer INPUT_WIDTH_BITS  = 512,
    parameter integer OUTPUT_WIDTH_BITS = 512,
    parameter integer SCALE_MULT        = 22913,
    parameter integer SCALE_SHIFT       = 18
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              valid_in,
    output reg                               ready_in,
    input  wire [INPUT_WIDTH_BITS-1:0]       data_in,
    output reg                               valid_out,
    output reg  [OUTPUT_WIDTH_BITS-1:0]      data_out
);

    localparam integer ACC_W         = 16;
    localparam integer SCALE_CONST_W = 16;
    localparam integer SCALED_W      = 32;

    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST = SCALE_MULT;
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);

    localparam [2:0] ST_ACCUM  = 3'd0;
    localparam [2:0] ST_SCALE  = 3'd1;
    localparam [2:0] ST_ROUND  = 3'd2;
    localparam [2:0] ST_OUTPUT = 3'd3;

    reg [2:0] state;
    reg [6:0] cnt;

    reg signed [ACC_W-1:0]    acc      [0:OC-1];
    reg signed [SCALED_W-1:0] scaled   [0:OC-1];
    reg signed [7:0]          out_byte [0:OC-1];

    wire signed [SCALED_W-1:0] rounded  [0:OC-1];
    wire signed [7:0]          clamped  [0:OC-1];

    integer c;
    integer i;
    genvar  gv;

    generate
        for (gv = 0; gv < OC; gv = gv + 1) begin: requantize
            // [INVARIANT:ROUNDING]
            assign rounded[gv] = (scaled[gv] +
                                  (scaled[gv][SCALED_W-1] ? (SCALE_ROUND_HALF - 1)
                                                          : SCALE_ROUND_HALF)
                                 ) >>> SCALE_SHIFT;
            assign clamped[gv] = (rounded[gv] >  32'sd127) ?  8'sd127 :
                                 (rounded[gv] < -32'sd128) ? -8'sd128 :
                                                              rounded[gv][7:0];
        end
    endgenerate

    always @(posedge clk) begin
        if (state == ST_ACCUM && valid_in) begin
            for (c = 0; c < OC; c = c + 1) begin
                if (cnt == 7'd0)
                    acc[c] <= $signed(data_in[c*8 +: 8]);
                else
                    acc[c] <= acc[c] + $signed(data_in[c*8 +: 8]);
            end
        end
        if (state == ST_SCALE) begin
            for (c = 0; c < OC; c = c + 1)
                scaled[c] <= $signed(acc[c]) * $signed(SCALE_MULT_CONST);
        end
        if (state == ST_ROUND) begin
            for (c = 0; c < OC; c = c + 1)
                out_byte[c] <= clamped[c];
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= ST_ACCUM;
            cnt       <= 7'd0;
            ready_in  <= 1'b1;
            valid_out <= 1'b0;
            data_out  <= {OUTPUT_WIDTH_BITS{1'b0}};
        end else begin
            valid_out <= 1'b0;
            case (state)
                ST_ACCUM: begin
                    ready_in <= 1'b1;
                    if (valid_in) begin
                        if (cnt == HW_TOTAL - 1) begin
                            cnt      <= 7'd0;
                            state    <= ST_SCALE;
                            ready_in <= 1'b0;
                        end else begin
                            cnt <= cnt + 7'd1;
                        end
                    end
                end
                ST_SCALE:  state <= ST_ROUND;
                ST_ROUND:  state <= ST_OUTPUT;
                ST_OUTPUT: begin
                    for (i = 0; i < OC; i = i + 1)
                        data_out[i*8 +: 8] <= out_byte[i];
                    valid_out <= 1'b1;
                    ready_in  <= 1'b1;
                    state     <= ST_ACCUM;
                end
                default: state <= ST_ACCUM;
            endcase
        end
    end

endmodule
