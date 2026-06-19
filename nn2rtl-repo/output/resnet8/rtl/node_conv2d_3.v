// node_conv2d_3 - pointwise 1x1 conv2d stride 2x2, IC=16, OC=32, IH=IW=32, OH=OW=16.
// RE-PARALLELIZED 1x1: MP=16 lanes x K_PAR=8 [LUT] taps = 128 INT8 multiplies/cycle.
// ST_RUNNING = K_GROUPS(2) * MP(16) cycles/pass; OC_PASSES=2 passes/pixel.
// Byte-exact vs the serial MP=4 1x1 FSM (same products, same accumulation order,
// same per-tensor SCALE_MULT=19187/SCALE_SHIFT=21, same sign-dependent
// round bias + saturate, same stride-2 gating + inter-frame reset). Weights repacked
// WIDE (MP*K_PAR bytes/word) read one word/cycle.

module node_conv2d_3 (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              valid_in,
    output reg               ready_in,
    input  wire [127:0]      data_in,
    output reg               valid_out,
    output reg  [255:0]      data_out
);
    localparam IC        = 16;
    localparam OC        = 32;
    localparam IH        = 32;
    localparam IW        = 32;
    localparam OH        = 16;
    localparam OW        = 16;
    localparam OH_OW     = OH * OW;
    localparam SH        = 2;
    localparam SW        = 2;
    localparam KH        = 1;
    localparam KW        = 1;
    localparam K_TOTAL   = IC * KH * KW;       // 16
    localparam MP        = 16;
    localparam K_PAR     = 8;
    localparam K_GROUPS  = K_TOTAL / K_PAR;    // 2
    localparam OC_PASSES = OC / MP;            // 2
    localparam NUM_WIDE  = OC_PASSES * K_GROUPS; // 4
    localparam WIDE_W    = MP * K_PAR * 8;     // 1024

    localparam SCALE_MULT  = 19187;
    localparam SCALE_SHIFT = 21;

    localparam integer PROD_W        = 16;
    localparam integer TREE_W        = PROD_W + $clog2(K_PAR);
    localparam integer ACC_W         = TREE_W + $clog2(K_GROUPS > 1 ? K_GROUPS : 2);
    localparam integer BIAS_W        = 32;
    localparam integer BIASED_W      = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1;
    localparam integer SCALE_MAG_W   = $clog2(SCALE_MULT + 1);
    localparam integer SCALE_CONST_W = SCALE_MAG_W + 1;
    localparam integer SCALED_W      = BIASED_W + SCALE_CONST_W;
    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST = SCALE_MULT;
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);

    localparam KGROUP_W   = (K_GROUPS <= 1) ? 1 : $clog2(K_GROUPS);
    localparam OC_GROUP_W = (OC_PASSES <= 1) ? 1 : $clog2(OC_PASSES);

    localparam ST_STREAM  = 3'd0;
    localparam ST_RUNNING = 3'd1;
    localparam ST_BIAS    = 3'd2;
    localparam ST_SCALE   = 3'd3;
    localparam ST_OUTPUT  = 3'd4;

    (* rom_style = "block", ram_style = "block" *) reg [WIDE_W-1:0]  weights_wide [0:NUM_WIDE-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases       [0:OC-1];
    initial begin
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/weights/node_conv2d_3_weights_wide1x1_mp16_kp8.hex", weights_wide);
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/weights/node_conv2d_3_bias.hex", biases);
    end

    reg signed [7:0] in_latch [0:IC-1];

    reg signed [ACC_W-1:0]    acc    [0:MP-1];
    reg signed [BIASED_W-1:0] biased [0:MP-1];
    reg signed [SCALED_W-1:0] scaled [0:MP-1];
    reg signed [SCALED_W-1:0] v_tmp;

    reg [KGROUP_W-1:0]   k_group;
    reg [OC_GROUP_W-1:0] oc_group;
    reg [2:0] state;

    reg [$clog2(IH)-1:0] in_row;
    reg [$clog2(IW)-1:0] in_col;
    reg [$clog2(OH_OW+1)-1:0] out_count;

    wire [$clog2(NUM_WIDE+1)-1:0] weight_read_addr = oc_group * K_GROUPS + k_group;

    // Stage 1: register wide weight word + K_PAR taps (taps from in_latch).
    reg [WIDE_W-1:0] weight_word_q;
    reg signed [7:0] tap_q [0:K_PAR-1];
    integer ld_i;
    always @(posedge clk) begin
        weight_word_q <= weights_wide[weight_read_addr];
        for (ld_i = 0; ld_i < K_PAR; ld_i = ld_i + 1)
            tap_q[ld_i] <= in_latch[k_group * K_PAR + ld_i];
    end

    // Stage 2: MP*K_PAR multipliers, tree-sum per lane (combinational).
    reg signed [TREE_W-1:0] partial_q [0:MP-1];
    reg signed [TREE_W-1:0] sum_lane_w [0:MP-1];
    reg signed [PROD_W-1:0] prod_w;
    integer cs_lane, cs_kpos;
    always @* begin
        for (cs_lane = 0; cs_lane < MP; cs_lane = cs_lane + 1) begin
            sum_lane_w[cs_lane] = {TREE_W{1'b0}};
            for (cs_kpos = 0; cs_kpos < K_PAR; cs_kpos = cs_kpos + 1) begin
                prod_w = $signed(weight_word_q[(cs_lane * K_PAR + cs_kpos) * 8 +: 8]) *
                         $signed(tap_q[cs_kpos]);
                sum_lane_w[cs_lane] = sum_lane_w[cs_lane] + prod_w;
            end
        end
    end

    reg                  mac_valid_q1;
    reg                  mac_valid_q2;
    reg                  mac_done_issuing;
    integer i, lane, p_i;
    integer bias_oc, out_oc;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= ST_STREAM;
            ready_in         <= 1'b1;
            valid_out        <= 1'b0;
            k_group          <= 0;
            oc_group         <= 0;
            in_row           <= 0;
            in_col           <= 0;
            out_count        <= 0;
            data_out         <= {(OC*8){1'b0}};
            mac_valid_q1     <= 1'b0;
            mac_valid_q2     <= 1'b0;
            mac_done_issuing <= 1'b0;
            for (i = 0; i < IC; i = i + 1)
                in_latch[i] <= 8'sd0;
            for (lane = 0; lane < MP; lane = lane + 1) begin
                acc   [lane] <= 0;
                biased[lane] <= 0;
                scaled[lane] <= 0;
                partial_q[lane] <= 0;
            end
        end else begin
            // Stage 2 register + Stage 3 accumulate.
            for (p_i = 0; p_i < MP; p_i = p_i + 1)
                partial_q[p_i] <= sum_lane_w[p_i];
            mac_valid_q2 <= mac_valid_q1;
            if (mac_valid_q2) begin
                for (p_i = 0; p_i < MP; p_i = p_i + 1)
                    acc[p_i] <= acc[p_i] + $signed(partial_q[p_i]);
            end

            case (state)

            ST_STREAM: begin
                valid_out    <= 1'b0;
                mac_valid_q1 <= 1'b0;
                mac_valid_q2 <= 1'b0;
                if (valid_in) begin
                    if (in_col == IW - 1) begin
                        in_col <= 0;
                        if (in_row == IH - 1) in_row <= 0;
                        else                  in_row <= in_row + 1;
                    end else begin
                        in_col <= in_col + 1;
                    end

                    if ((in_row[0] == 1'b0) && (in_col[0] == 1'b0)) begin
                        for (i = 0; i < IC; i = i + 1)
                            in_latch[i] <= $signed(data_in[i*8 +: 8]);
                        ready_in         <= 1'b0;
                        k_group          <= 0;
                        oc_group         <= 0;
                        mac_done_issuing <= 1'b0;
                        for (lane = 0; lane < MP; lane = lane + 1)
                            acc[lane] <= 0;
                        state <= ST_RUNNING;
                    end
                end
            end

            ST_RUNNING: begin
                if (mac_done_issuing) begin
                    mac_valid_q1 <= 1'b0;
                    if (!mac_valid_q1 && !mac_valid_q2) begin
                        mac_done_issuing <= 1'b0;
                        state            <= ST_BIAS;
                    end
                end else begin
                    mac_valid_q1 <= 1'b1;
                    if (k_group == K_GROUPS - 1) begin
                        mac_done_issuing <= 1'b1;
                    end else begin
                        k_group <= k_group + 1;
                    end
                end
            end

            ST_BIAS: begin
                for (lane = 0; lane < MP; lane = lane + 1) begin
                    bias_oc = oc_group * MP + lane;
                    if (bias_oc < OC)
                        biased[lane] <= $signed(acc[lane]) + $signed(biases[bias_oc]);
                    else
                        biased[lane] <= 0;
                end
                state <= ST_SCALE;
            end

            ST_SCALE: begin
                for (lane = 0; lane < MP; lane = lane + 1)
                    scaled[lane] <= $signed(biased[lane]) * $signed(SCALE_MULT_CONST);
                state <= ST_OUTPUT;
            end

            ST_OUTPUT: begin
                for (lane = 0; lane < MP; lane = lane + 1) begin
                    out_oc = oc_group * MP + lane;
                    if (out_oc < OC) begin
                        v_tmp = (scaled[lane] +
                                 (scaled[lane][SCALED_W-1] ? (SCALE_ROUND_HALF - 1)
                                                           : SCALE_ROUND_HALF)
                                ) >>> SCALE_SHIFT;
                        data_out[out_oc*8 +: 8] <= (v_tmp >  127) ?  8'sd127 :
                                                   (v_tmp < -128) ? -8'sd128 :
                                                                    v_tmp[7:0];
                    end
                end

                if (oc_group < OC_PASSES - 1) begin
                    for (lane = 0; lane < MP; lane = lane + 1) acc[lane] <= 0;
                    k_group          <= 0;
                    oc_group         <= oc_group + 1;
                    mac_valid_q1     <= 1'b0;
                    mac_valid_q2     <= 1'b0;
                    mac_done_issuing <= 1'b0;
                    state            <= ST_RUNNING;
                end else begin
                    valid_out <= 1'b1;
                    ready_in  <= 1'b1;
                    oc_group  <= 0;
                    state     <= ST_STREAM;
                    if (out_count == OH_OW - 1) begin
                        out_count <= 0;
                        in_row    <= 0;
                        in_col    <= 0;
                    end else begin
                        out_count <= out_count + 1;
                    end
                end
            end

            default: state <= ST_STREAM;
            endcase
        end
    end
endmodule
