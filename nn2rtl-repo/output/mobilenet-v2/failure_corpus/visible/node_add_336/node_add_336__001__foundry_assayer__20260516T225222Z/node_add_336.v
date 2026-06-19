`default_nettype none

module node_add_336 (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 valid_in,
    output reg                  ready_in,
    input  wire [511:0]         data_in,
    output reg                  valid_out,
    output reg  [255:0]         data_out
);

    localparam integer OC            = 32;
    localparam integer W_OUT         = 256;
    localparam integer CH_IDX_W      = 6;

    localparam integer SCALE_CONST_W = 16;
    localparam integer FUSED_SHIFT   = 17;
    localparam signed [SCALE_CONST_W-1:0] LHS_FUSED_MULT = 16'sd10197;
    localparam signed [SCALE_CONST_W-1:0] RHS_FUSED_MULT = 16'sd7761;

    localparam integer PROD_W = 8 + SCALE_CONST_W;
    localparam integer SUM_W  = PROD_W + 2;

    localparam signed [SUM_W-1:0] FUSED_HALF    = 26'sd65536;
    localparam signed [SUM_W-1:0] FUSED_HALF_M1 = 26'sd65535;
    localparam signed [SUM_W-1:0] SAT_HI        =  26'sd127;
    localparam signed [SUM_W-1:0] SAT_LO        = -26'sd128;

    localparam ST_IDLE = 1'b0;
    localparam ST_RUN  = 1'b1;
    reg                       state;

    reg [511:0]               input_buf;
    reg [CH_IDX_W-1:0]        ch_idx;
    reg [CH_IDX_W-1:0]        stage1_idx;
    reg [CH_IDX_W-1:0]        stage2_idx;
    reg                       stage1_valid;
    reg                       stage2_valid;

    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] lhs_term;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] rhs_term;
    reg signed [SUM_W-1:0]    sum_term;

    wire signed [7:0]         cur_lhs;
    wire signed [7:0]         cur_rhs;
    wire signed [SUM_W-1:0]   sum_pre;
    wire signed [SUM_W-1:0]   shifted;
    wire signed [7:0]         sat_byte;

    assign cur_lhs  = $signed(input_buf[ch_idx*8 +: 8]);
    assign cur_rhs  = $signed(input_buf[W_OUT + ch_idx*8 +: 8]);
    assign sum_pre  = lhs_term + rhs_term;
    assign shifted  = sum_term >>> FUSED_SHIFT;
    assign sat_byte = (shifted > SAT_HI) ?  8'sd127 :
                      (shifted < SAT_LO) ? -8'sd128 :
                       shifted[7:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= ST_IDLE;
            ready_in     <= 1'b1;
            valid_out    <= 1'b0;
            data_out     <= 256'd0;
            input_buf    <= 512'd0;
            ch_idx       <= {CH_IDX_W{1'b0}};
            stage1_idx   <= {CH_IDX_W{1'b0}};
            stage2_idx   <= {CH_IDX_W{1'b0}};
            stage1_valid <= 1'b0;
            stage2_valid <= 1'b0;
            lhs_term     <= {PROD_W{1'b0}};
            rhs_term     <= {PROD_W{1'b0}};
            sum_term     <= {SUM_W{1'b0}};
        end else begin
            valid_out <= 1'b0;

            if (state == ST_RUN && ch_idx < 6'd32) begin
                lhs_term     <= cur_lhs * LHS_FUSED_MULT;
                rhs_term     <= cur_rhs * RHS_FUSED_MULT;
                stage1_idx   <= ch_idx;
                stage1_valid <= 1'b1;
                ch_idx       <= ch_idx + 6'd1;
            end else begin
                stage1_valid <= 1'b0;
            end

            if (stage1_valid) begin
                sum_term     <= sum_pre + (sum_pre[SUM_W-1] ? FUSED_HALF_M1 : FUSED_HALF); // [INVARIANT:ROUNDING]
                stage2_idx   <= stage1_idx;
                stage2_valid <= 1'b1;
            end else begin
                stage2_valid <= 1'b0;
            end

            if (stage2_valid) begin
                data_out[stage2_idx*8 +: 8] <= sat_byte;
                if (stage2_idx == 6'd31) begin
                    valid_out <= 1'b1;             // [INVARIANT:VALID_OUT_LATENCY]
                    state     <= ST_IDLE;
                    ready_in  <= 1'b1;             // [INVARIANT:READY_IN_GATING]
                end
            end

            if (state == ST_IDLE && valid_in) begin
                input_buf    <= data_in;
                state        <= ST_RUN;
                ready_in     <= 1'b0;              // [INVARIANT:READY_IN_GATING]
                ch_idx       <= {CH_IDX_W{1'b0}};
                stage1_valid <= 1'b0;
                stage2_valid <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire
