`default_nettype none

module node_add_828 (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    output reg            ready_in,
    input  wire [1535:0]  data_in,
    output reg            valid_out,
    output reg  [767:0]   data_out
);

    localparam integer OC            = 96;
    localparam integer INPUT_WIDTH   = 1536;
    localparam integer OUTPUT_WIDTH  = 768;
    localparam integer RHS_BASE      = OUTPUT_WIDTH;

    localparam integer FUSED_SHIFT      = 15;
    localparam integer SCALE_CONST_W    = 16;
    localparam signed [SCALE_CONST_W-1:0] LHS_FUSED_MULT = 16'sd27077;
    localparam signed [SCALE_CONST_W-1:0] RHS_FUSED_MULT = 16'sd20771;

    localparam integer PROD_W           = 8 + SCALE_CONST_W;
    localparam integer SUM_W            = PROD_W + 2;

    localparam signed [SUM_W-1:0] FUSED_HALF    = 26'sd16384;
    localparam signed [SUM_W-1:0] FUSED_HALF_M1 = 26'sd16383;
    localparam signed [SUM_W-1:0] SAT_HI        =  26'sd127;
    localparam signed [SUM_W-1:0] SAT_LO        = -26'sd128;

    localparam integer CH_IDX_W      = 7;

    localparam [1:0] ST_IDLE = 2'd0,
                     ST_LOAD = 2'd1,
                     ST_RUN  = 2'd2;
    reg [1:0]              state;

    reg [INPUT_WIDTH-1:0]  input_buf;
    reg [CH_IDX_W:0]       ch_idx;
    reg [CH_IDX_W-1:0]     stage1_idx;
    reg [CH_IDX_W-1:0]     stage2_idx;
    reg [CH_IDX_W-1:0]     stage3_idx;
    reg                    stage1_valid;
    reg                    stage2_valid;
    reg                    stage3_valid;

    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] lhs_term;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] rhs_term;
    reg signed [SUM_W-1:0] sum_term;

    wire signed [7:0]       cur_lhs;
    wire signed [7:0]       cur_rhs;
    wire signed [SUM_W-1:0] sum_pre;
    wire signed [SUM_W-1:0] shifted;
    wire signed [7:0]       sat_byte;

    assign cur_lhs  = $signed(input_buf[ch_idx*8 +: 8]);
    assign cur_rhs  = $signed(input_buf[RHS_BASE + ch_idx*8 +: 8]);
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
            data_out     <= {OUTPUT_WIDTH{1'b0}};
            input_buf    <= {INPUT_WIDTH{1'b0}};
            ch_idx       <= {(CH_IDX_W+1){1'b0}};
            stage1_idx   <= {CH_IDX_W{1'b0}};
            stage2_idx   <= {CH_IDX_W{1'b0}};
            stage3_idx   <= {CH_IDX_W{1'b0}};
            stage1_valid <= 1'b0;
            stage2_valid <= 1'b0;
            stage3_valid <= 1'b0;
            lhs_term     <= {PROD_W{1'b0}};
            rhs_term     <= {PROD_W{1'b0}};
            sum_term     <= {SUM_W{1'b0}};
        end else begin
            valid_out <= 1'b0;

            if (state == ST_RUN && ch_idx < OC) begin
                lhs_term     <= cur_lhs * LHS_FUSED_MULT;
                rhs_term     <= cur_rhs * RHS_FUSED_MULT;
                stage1_idx   <= ch_idx[CH_IDX_W-1:0];
                stage1_valid <= 1'b1;
                ch_idx       <= ch_idx + 1'b1;
            end else begin
                stage1_valid <= 1'b0;
            end

            if (stage1_valid) begin
                sum_term     <= sum_pre +
                                (sum_pre[SUM_W-1] ? FUSED_HALF_M1 : FUSED_HALF);
                stage2_idx   <= stage1_idx;
                stage2_valid <= 1'b1;
            end else begin
                stage2_valid <= 1'b0;
            end

            stage3_valid <= stage2_valid;
            stage3_idx   <= stage2_idx;
            if (stage2_valid) begin
                data_out[stage2_idx*8 +: 8] <= sat_byte;
            end

            case (state)
                ST_IDLE: begin
                    if (valid_in) begin
                        input_buf <= data_in;
                        state     <= ST_LOAD;
                        ready_in  <= 1'b0;
                    end
                end
                ST_LOAD: begin
                    ch_idx <= {(CH_IDX_W+1){1'b0}};
                    state  <= ST_RUN;
                end
                ST_RUN: begin
                    if (stage2_valid && stage2_idx == (OC - 1)) begin
                        valid_out <= 1'b1;
                        ready_in  <= 1'b1;
                        state     <= ST_IDLE;
                    end
                end
                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
