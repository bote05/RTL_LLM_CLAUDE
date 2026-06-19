`timescale 1ns / 1ps
`default_nettype none

// ---------------------------------------------------------------------------
// node_conv_830 -- MobileNet-v2 depthwise 3x3 stride-2 padding-1 conv
//   C   = 144  (groups == in_channels == out_channels)
//   IH  = IW  = 56, OH = OW = 28
//   KH  = KW  = 3, PH = PW = 1, stride 2
//   MAC parallelism = 4 lanes (sequential one-MAC-per-cycle, lane-then-tap)
//   OC_PASSES = ceil(C/MP) = 36; pass = MP * K_TOTAL + 6 = 4*9+6 = 42 cycles
//   Pipeline latency = COMPUTE_START + OC_PASSES * PASS_CYCLES + 2
//                    = 58 + 1512 + 2 = 1572 cycles (LayerIR authoritative)
//   Contract: depthwise-conv with channel_tile = 144 = C, 1 beat per pixel
//   Bus  : 1152b in / 1152b out (144 INT8 channels per beat)
//   Scale: 6715 / 2^20 ~= 0.006403978 (target 0.006403977, |err| ~ 1e-5)
//   Depthwise: each output channel reads only its own input channel; no IC
//   reduction inside K_TOTAL. Weight memory is [C * K_TOTAL] = 1296 entries.
// ---------------------------------------------------------------------------

module node_conv_830 #(
    parameter WEIGHTS_PATH = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_830_weights.hex",
    parameter BIAS_PATH    = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_830_bias.hex"
) (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    output wire           ready_in,
    input  wire [1151:0]  data_in,
    output reg            valid_out,
    output reg  [1151:0]  data_out
);

    localparam integer C             = 144;
    localparam integer IH            = 56;
    localparam integer IW            = 56;
    localparam integer OH            = 28;
    localparam integer OW            = 28;
    localparam integer KH            = 3;
    localparam integer KW            = 3;
    localparam integer MP            = 4;
    localparam integer K_TOTAL       = 9;
    localparam integer OC_PASSES     = 36;
    localparam integer PASS_CYCLES   = 42;
    localparam integer COMPUTE_START = 58;
    localparam integer FIRST_OUT_CYC = 1572;
    localparam integer N_PIX_IN      = 3136;
    localparam integer N_PIX_OUT     = 784;
    localparam integer BUS_W         = 1152;
    localparam integer SCALE_SHIFT   = 20;
    localparam signed [63:0] SCALE_MULT_64 = 64'sd6715;

    (* ram_style = "block" *) reg [BUS_W-1:0] line_buf [0:N_PIX_IN-1];
    (* ram_style = "block" *) reg [BUS_W-1:0] out_buf  [0:N_PIX_OUT-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [7:0]  weights [0:C*K_TOTAL-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases  [0:C-1];

    initial begin
        $readmemh(WEIGHTS_PATH, weights);
        $readmemh(BIAS_PATH,    biases);
    end

    reg        run;
    reg [15:0] cyc_cnt;
    reg        vector_done;

    reg [11:0] in_pix;
    reg        input_done;

    // [INVARIANT:READY_IN_GATING]
    assign ready_in = !input_done;

    reg                 cmp_active;
    reg [10:0]          cmp_pix;
    reg [4:0]           cmp_oh;
    reg [4:0]           cmp_ow;
    reg [7:0]           cmp_pass;
    reg [5:0]           cmp_step;
    reg signed [31:0]   acc [0:MP-1];
    reg [N_PIX_OUT-1:0] pix_done;
    reg [10:0]          outputs_emitted;

    wire is_issue     = cmp_active && (cmp_step < 6'd36);
    wire is_writeback = cmp_active && (cmp_step == 6'd38);
    wire is_pass_end  = cmp_active && (cmp_step == 6'd41);

    wire [1:0] cur_lane =
        (cmp_step < 6'd9)  ? 2'd0 :
        (cmp_step < 6'd18) ? 2'd1 :
        (cmp_step < 6'd27) ? 2'd2 : 2'd3;

    wire [5:0] cur_k_wide =
        (cmp_step < 6'd9)  ? cmp_step             :
        (cmp_step < 6'd18) ? (cmp_step - 6'd9)    :
        (cmp_step < 6'd27) ? (cmp_step - 6'd18)   :
                             (cmp_step - 6'd27);
    wire [3:0] cur_k = is_issue ? cur_k_wide[3:0] : 4'd0;

    wire [9:0] base_ch = {cmp_pass[7:0], 2'b00};
    wire [9:0] cur_ch  = base_ch + {8'd0, cur_lane};

    wire [1:0] kh = (cur_k >= 4'd6) ? 2'd2 :
                    (cur_k >= 4'd3) ? 2'd1 : 2'd0;
    wire [3:0] kw_calc = (kh == 2'd0) ? cur_k        :
                         (kh == 2'd1) ? (cur_k - 4'd3) :
                                        (cur_k - 4'd6);
    wire [1:0] kw = kw_calc[1:0];

    wire signed [8:0] cmp_oh_s = $signed({4'd0, cmp_oh});
    wire signed [8:0] cmp_ow_s = $signed({4'd0, cmp_ow});
    wire signed [8:0] kh_s     = $signed({7'd0, kh});
    wire signed [8:0] kw_s     = $signed({7'd0, kw});
    wire signed [8:0] in_r     = (cmp_oh_s <<< 1) + kh_s - 9'sd1;
    wire signed [8:0] in_c     = (cmp_ow_s <<< 1) + kw_s - 9'sd1;
    wire              in_bounds = (in_r >= 9'sd0) && (in_r < 9'sd56) &&
                                  (in_c >= 9'sd0) && (in_c < 9'sd56);

    wire [5:0]  in_r_u = in_r[5:0];
    wire [5:0]  in_c_u = in_c[5:0];
    wire [11:0] in_pix_idx = in_bounds
        ? (({6'd0, in_r_u} * 12'd56) + {6'd0, in_c_u})
        : 12'd0;

    wire [BUS_W-1:0]  act_pix  = line_buf[in_pix_idx];
    wire signed [7:0] act_byte = !in_bounds ? 8'sd0 : $signed(act_pix[cur_ch*8 +: 8]);

    wire [13:0]        w_addr   = {4'd0, cur_ch} * 14'd9 + {10'd0, cur_k};
    wire signed [7:0]  w_byte   = weights[w_addr];
    wire signed [15:0] mac_prod = act_byte * w_byte;

    integer            i;
    reg signed [63:0]  sum_wb;
    reg signed [63:0]  prod_wb;
    reg signed [63:0]  round_wb;
    reg signed [63:0]  scaled_wb;
    reg signed [7:0]   sat_byte;
    reg [9:0]          wb_ch;

    always @(posedge clk) begin
        if (!rst_n || vector_done) begin
            run     <= 1'b0;
            cyc_cnt <= 16'd0;
        end else if (!run) begin
            if (valid_in && ready_in) begin
                run     <= 1'b1;
                cyc_cnt <= 16'd1;
            end
        end else if (cyc_cnt != 16'hFFFF) begin
            cyc_cnt <= cyc_cnt + 16'd1;
        end
    end

    always @(posedge clk) begin
        if (!rst_n || vector_done) begin
            in_pix     <= 12'd0;
            input_done <= 1'b0;
        end else if (valid_in && ready_in) begin
            line_buf[in_pix] <= data_in;
            if (in_pix == 12'd3135) input_done <= 1'b1;
            in_pix <= in_pix + 12'd1;
        end
    end

    always @(posedge clk) begin
        if (!rst_n || vector_done) begin
            cmp_active      <= 1'b0;
            cmp_pix         <= 11'd0;
            cmp_oh          <= 5'd0;
            cmp_ow          <= 5'd0;
            cmp_pass        <= 8'd0;
            cmp_step        <= 6'd0;
            pix_done        <= {N_PIX_OUT{1'b0}};
            outputs_emitted <= 11'd0;
            acc[0]          <= 32'sd0;
            acc[1]          <= 32'sd0;
            acc[2]          <= 32'sd0;
            acc[3]          <= 32'sd0;
        end else begin
            if (!cmp_active && run && cyc_cnt >= 16'd58) begin
                cmp_active <= 1'b1;
                cmp_step   <= 6'd0;
                cmp_pix    <= 11'd0;
                cmp_oh     <= 5'd0;
                cmp_ow     <= 5'd0;
                cmp_pass   <= 8'd0;
                acc[0]     <= 32'sd0;
                acc[1]     <= 32'sd0;
                acc[2]     <= 32'sd0;
                acc[3]     <= 32'sd0;
            end else if (cmp_active) begin
                if (is_issue) begin
                    acc[cur_lane] <= acc[cur_lane] + mac_prod;
                end

                if (is_writeback) begin
                    for (i = 0; i < MP; i = i + 1) begin
                        wb_ch    = base_ch + i[9:0];
                        sum_wb   = acc[i] + biases[wb_ch];
                        prod_wb  = sum_wb * SCALE_MULT_64;
                        // [INVARIANT:ROUNDING]
                        if (sum_wb >= 64'sd0)
                            round_wb = prod_wb + (64'sd1 <<< (SCALE_SHIFT - 1));
                        else
                            round_wb = prod_wb + (64'sd1 <<< (SCALE_SHIFT - 1)) - 64'sd1;
                        scaled_wb = round_wb >>> SCALE_SHIFT;
                        if (scaled_wb > 64'sd127)
                            sat_byte = 8'sd127;
                        else if (scaled_wb < -64'sd128)
                            sat_byte = -8'sd128;
                        else
                            sat_byte = scaled_wb[7:0];
                        out_buf[cmp_pix][wb_ch*8 +: 8] <= sat_byte;
                    end
                end

                if (is_pass_end) begin
                    cmp_step <= 6'd0;
                    acc[0]   <= 32'sd0;
                    acc[1]   <= 32'sd0;
                    acc[2]   <= 32'sd0;
                    acc[3]   <= 32'sd0;
                    if (cmp_pass == 8'd35) begin
                        cmp_pass          <= 8'd0;
                        pix_done[cmp_pix] <= 1'b1;
                        outputs_emitted   <= outputs_emitted + 11'd1;
                        if (cmp_pix == 11'd783) begin
                            cmp_active <= 1'b0;
                        end else begin
                            cmp_pix <= cmp_pix + 11'd1;
                            if (cmp_ow == 5'd27) begin
                                cmp_ow <= 5'd0;
                                cmp_oh <= cmp_oh + 5'd1;
                            end else begin
                                cmp_ow <= cmp_ow + 5'd1;
                            end
                        end
                    end else begin
                        cmp_pass <= cmp_pass + 8'd1;
                    end
                end else begin
                    cmp_step <= cmp_step + 6'd1;
                end
            end
        end
    end

    reg [10:0] em_pix;

    // [INVARIANT:VALID_OUT_LATENCY]
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out   <= 1'b0;
            data_out    <= {BUS_W{1'b0}};
            em_pix      <= 11'd0;
            vector_done <= 1'b0;
        end else begin
            vector_done <= 1'b0;
            if (!vector_done && run &&
                (cyc_cnt >= 16'd1571) &&
                (em_pix < 11'd784) &&
                pix_done[em_pix]) begin
                valid_out <= 1'b1;
                data_out  <= out_buf[em_pix];
                if (em_pix == 11'd783) begin
                    em_pix      <= 11'd0;
                    vector_done <= 1'b1;
                end else begin
                    em_pix <= em_pix + 11'd1;
                end
            end else begin
                valid_out <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire
