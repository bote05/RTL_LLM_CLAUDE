// [FREE-RUNNING PARALLEL ADD] rewritten by apply_resnet8_parallel_adds.py
// node_add_56 -- INT8 residual add, flat-bus, OC=32.
// FREE-RUNNING fully-parallel rewrite (byte-identical arithmetic to the
// generated serial FSM; see scripts/apply_resnet8_parallel_adds.py).
//   data_in[255:0]      = lhs (32 ch * 8b)
//   data_in[511:256]  = rhs (32 ch * 8b)
//   data_out[255:0]     = saturated INT8 sum, 32 channels packed
//   ready_in = 1; latency 3 cycles; throughput 1 beat/cycle.

module node_add_56 (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 valid_in,
    output wire                 ready_in,
    input  wire [511:0] data_in,
    output reg                  valid_out,
    output reg  [255:0] data_out
);

    localparam integer OC          = 32;
    localparam integer FUSED_SHIFT = 22;
    localparam integer MULT_W      = 24;
    localparam integer PROD_W      = 32;  // 8 + MULT_W
    localparam integer SUM_W       = 34;  // PROD_W + 2

    localparam signed [MULT_W-1:0] FUSED_LHS_MULT   = 24'sd1490108;
    localparam signed [MULT_W-1:0] FUSED_RHS_MULT   = 24'sd3570959;
    localparam signed [SUM_W-1:0]  FUSED_ROUND_BIAS = 34'sd2097152;
    localparam signed [SUM_W-1:0]  SAT_HI           = 34'sd127;
    localparam signed [SUM_W-1:0]  SAT_LO           = -34'sd128;

    // free-running: never stall an un-throttleable systolic producer
    assign ready_in = 1'b1;  // [INVARIANT:READY_IN_GATING]

    // ---- stage 1: per-channel products (all OC in parallel) ----
    (* use_dsp = "no" *) reg signed [PROD_W-1:0] lhs_term [0:OC-1];
    (* use_dsp = "no" *) reg signed [PROD_W-1:0] rhs_term [0:OC-1];
    reg v1;
    // ---- stage 2: per-channel rounded sums ----
    reg signed [SUM_W-1:0] sum_term [0:OC-1];
    reg v2;

    integer i;
    reg signed [SUM_W-1:0] v_tmp;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1        <= 1'b0;
            v2        <= 1'b0;
            valid_out <= 1'b0;
            data_out  <= 256'd0;
            for (i = 0; i < OC; i = i + 1) begin
                lhs_term[i] <= {PROD_W{1'b0}};
                rhs_term[i] <= {PROD_W{1'b0}};
                sum_term[i] <= {SUM_W{1'b0}};
            end
        end else begin
            // stage 1
            for (i = 0; i < OC; i = i + 1) begin
                lhs_term[i] <= $signed(data_in[i*8 +: 8])           * FUSED_LHS_MULT;
                rhs_term[i] <= $signed(data_in[256 + i*8 +: 8]) * FUSED_RHS_MULT;
            end
            v1 <= valid_in;

            // stage 2: lhs+rhs+round  [INVARIANT:ROUNDING]
            for (i = 0; i < OC; i = i + 1)
                sum_term[i] <= $signed(lhs_term[i]) + $signed(rhs_term[i]) + FUSED_ROUND_BIAS;
            v2 <= v1;

            // stage 3: arithmetic shift + saturate
            for (i = 0; i < OC; i = i + 1) begin
                v_tmp = sum_term[i] >>> FUSED_SHIFT;
                data_out[i*8 +: 8] <= (v_tmp > SAT_HI) ? 8'sd127 :
                                      (v_tmp < SAT_LO) ? 8'h80   : v_tmp[7:0];
            end
            valid_out <= v2;  // [INVARIANT:VALID_OUT_LATENCY]
        end
    end

endmodule
