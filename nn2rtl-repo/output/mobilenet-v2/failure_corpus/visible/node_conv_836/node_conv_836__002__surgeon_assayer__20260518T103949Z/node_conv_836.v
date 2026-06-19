`timescale 1ns / 1ps
`default_nettype none

// ---------------------------------------------------------------------------
// node_conv_836 -- MobileNet-v2 depthwise 3x3 stride-1 padding-1 conv
//   C   = 192  (groups == in_channels == out_channels)
//   IH  = IW  = 28,  OH = OW = 28  (stride 1, same)
//   KH  = KW  = 3,   PH = PW = 1
//   MAC parallelism = 4 lanes; OC_PASSES = ceil(C/MP) = 48
//   Pass duration = MP * K_TOTAL + 6 = 4*9 + 6 = 42 cycles
//   Pipeline latency (LayerIR-authoritative) = 2048 cycles
//   COMPUTE_START = 31 -> pix_done[0] set at edge 2047 -> valid_out at 2048
//   Contract: depthwise-conv (single beat per pixel, channel_tile = 192 = C)
//   Bus  : 1536 b in/out, 1 beat per pixel (192 channels packed INT8)
//   Scale: SCALE_MULT/2^SCALE_SHIFT = 21656 / 2^19 ~= 0.04130554
//          (target 0.04130484, |rel err| ~ 1.7e-5)
//   No cross-channel reduction -- each lane reads its own channel-of-interest
//   from the same (kh, kw) tap, no IC iteration inside K_TOTAL.
//
//   N_IN_PIX = N_OUT_PIX = 784. line_buf and out_buf banked into 2 unpacked
//   memories of depth 512 each (786432 bits per bank, under Vivado's ~900k
//   per-variable cap). Bank select = pix_idx[9]; bank offset = pix_idx[8:0].
// ---------------------------------------------------------------------------

module node_conv_836 #(
    parameter WEIGHTS_PATH = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_836_weights.hex",
    parameter BIAS_PATH    = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_836_bias.hex"
) (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    output wire           ready_in,
    input  wire [1535:0]  data_in,
    output reg            valid_out,
    output reg  [1535:0]  data_out
);

    localparam integer C             = 192;
    localparam integer IH            = 28;
    localparam integer IW            = 28;
    localparam integer OH            = 28;
    localparam integer OW            = 28;
    localparam integer KH            = 3;
    localparam integer KW            = 3;
    localparam integer SH            = 1;
    localparam integer SW            = 1;
    localparam integer PH            = 1;
    localparam integer PW            = 1;
    localparam integer MP            = 4;
    localparam integer K_TOTAL       = 9;
    localparam integer OC_PASSES     = 48;
    localparam integer FIRST_OUT_CYC = 2048;
    localparam integer COMPUTE_START = 31;
    localparam integer N_IN_PIX      = 784;
    localparam integer N_OUT_PIX     = 784;
    localparam integer BEAT_W        = 1536;
    localparam integer SCALE_SHIFT   = 19;
    localparam signed [63:0] SCALE_MULT_64 = 64'sd21656;

    (* ram_style = "block" *) reg [BEAT_W-1:0] line_buf_b0 [0:511];
    (* ram_style = "block" *) reg [BEAT_W-1:0] line_buf_b1 [0:511];
    (* ram_style = "block" *) reg [BEAT_W-1:0] out_buf_b0  [0:511];
    (* ram_style = "block" *) reg [BEAT_W-1:0] out_buf_b1  [0:511];
    (* rom_style = "block", ram_style = "block" *) reg signed [7:0]  weights [0:C*K_TOTAL-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases  [0:C-1];

    initial begin
        $readmemh(WEIGHTS_PATH, weights);
        $readmemh(BIAS_PATH, biases);
    end

    reg         run;
    reg [15:0]  cyc_cnt;
    reg         vector_done;

    reg [10:0]  in_beat;
    reg         input_done;

    // [INVARIANT:READY_IN_GATING]
    assign ready_in = !input_done;

    reg                cmp_active;
    reg [9:0]          cmp_pix;
    reg [4:0]          cmp_oh;
    reg [4:0]          cmp_ow;
    reg [5:0]          cmp_pass;
    reg [5:0]          cmp_step;
    reg signed [31:0]  acc [0:MP-1];
    reg [N_OUT_PIX-1:0] pix_done;
    reg [10:0]         outputs_emitted;

    reg signed [7:0]   window [0:K_TOTAL-1][0:MP-1];

    wire is_issue     = cmp_active && (cmp_step < 6'd36);
    wire is_writeback = cmp_active && (cmp_step == 6'd38);
    wire is_pass_end  = cmp_active && (cmp_step == 6'd41);

    wire [1:0] cur_lane = cmp_step[1:0];
    wire [3:0] cur_k    = is_issue ? cmp_step[5:2] : 4'd0;

    wire [7:0] base_ch = {cmp_pass, 2'b00};
    wire [7:0] cur_ch  = base_ch + {6'd0, cur_lane};

    wire [1:0] kh = (cur_k >= 4'd6) ? 2'd2 :
                    (cur_k >= 4'd3) ? 2'd1 : 2'd0;
    wire [3:0] kw_calc = (kh == 2'd0) ? cur_k         :
                         (kh == 2'd1) ? (cur_k - 4'd3) :
                                        (cur_k - 4'd6);
    wire [1:0] kw = kw_calc[1:0];

    wire signed [6:0] oh_s = $signed({2'b00, cmp_oh});
    wire signed [6:0] ow_s = $signed({2'b00, cmp_ow});
    wire signed [6:0] in_r = oh_s + $signed({5'b00000, kh}) - 7'sd1;
    wire signed [6:0] in_c = ow_s + $signed({5'b00000, kw}) - 7'sd1;
    wire              in_bounds = (in_r >= 7'sd0) && (in_r < 7'sd28) &&
                                  (in_c >= 7'sd0) && (in_c < 7'sd28);

    wire [4:0] in_r_u = in_r[4:0];
    wire [4:0] in_c_u = in_c[4:0];
    wire [9:0] in_pix_idx = in_bounds
        ? (({5'd0, in_r_u} * 10'd28) + {5'd0, in_c_u})
        : 10'd0;

    wire       rd_bank = in_pix_idx[9];
    wire [8:0] rd_off  = in_pix_idx[8:0];

    reg [BEAT_W-1:0] line_buf_word;
    always @(*) begin
        case (rd_bank)
            1'd0: line_buf_word = line_buf_b0[rd_off];
            1'd1: line_buf_word = line_buf_b1[rd_off];
            default: line_buf_word = {BEAT_W{1'b0}};
        endcase
    end

    wire signed [7:0] act_byte = in_bounds
        ? $signed(line_buf_word[cur_ch*8 +: 8])
        : 8'sd0;

    wire [11:0]        w_addr   = {4'd0, cur_ch} * 12'd9 + {8'd0, cur_k};
    wire signed [7:0]  w_byte   = weights[w_addr];
    wire signed [15:0] mac_prod = act_byte * w_byte;

    integer            i;
    integer            wj;
    reg signed [63:0]  sum_wb;
    reg signed [63:0]  prod_wb;
    reg signed [63:0]  round_wb;
    reg signed [63:0]  scaled_wb;
    reg signed [7:0]   sat_byte;
    reg [7:0]          wb_ch;

    wire                                     cs_needs_real_input;
    wire                                     cs_ready_in;
    wire                                     cs_advance;
    wire                                     cs_in_frame_done;
    wire                                     cs_out_frame_done;
    wire [$clog2(IH + 1 + 1)-1:0]            cs_in_row;
    wire [$clog2(IW + 1 + 1)-1:0]            cs_in_col;
    wire                                     cs_output_fires;
    wire [$clog2(OH * OW + 1)-1:0]           cs_outputs_emitted;
    reg                                      cs_start;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cs_start <= 1'b0;
        else
            cs_start <= (!run && valid_in && ready_in);
    end

    coord_scheduler #(
        .IH(IH), .IW(IW), .OH(OH), .OW(OW),
        .KH(KH), .KW(KW), .SH(SH), .SW(SW), .PH(PH), .PW(PW)
    ) u_coord_scheduler (
        .clk             (clk),
        .rst_n           (rst_n),
        .start           (cs_start),
        .stall_in        (1'b0),
        .valid_in        (1'b0),
        .ready_in        (cs_ready_in),
        .needs_real_input(cs_needs_real_input),
        .in_row          (cs_in_row),
        .in_col          (cs_in_col),
        .output_fires    (cs_output_fires),
        .advance         (cs_advance),
        .in_frame_done   (cs_in_frame_done),
        .out_frame_done  (cs_out_frame_done),
        .outputs_emitted (cs_outputs_emitted)
    );

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
            in_beat    <= 11'd0;
            input_done <= 1'b0;
        end else if (valid_in && ready_in) begin
            case (in_beat[9])
                1'd0: line_buf_b0[in_beat[8:0]] <= data_in;
                1'd1: line_buf_b1[in_beat[8:0]] <= data_in;
                default: ;
            endcase
            if (in_beat == N_IN_PIX - 1)
                input_done <= 1'b1;
            in_beat <= in_beat + 11'd1;
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            for (wj = 0; wj < K_TOTAL; wj = wj + 1) begin
                window[wj][0] <= 8'sd0;
                window[wj][1] <= 8'sd0;
                window[wj][2] <= 8'sd0;
                window[wj][3] <= 8'sd0;
            end
        end else if (is_issue) begin
            window[cur_k][cur_lane] <= act_byte;
        end
    end

    always @(posedge clk) begin
        if (!rst_n || vector_done) begin
            cmp_active      <= 1'b0;
            cmp_pix         <= 10'd0;
            cmp_oh          <= 5'd0;
            cmp_ow          <= 5'd0;
            cmp_pass        <= 6'd0;
            cmp_step        <= 6'd0;
            pix_done        <= {N_OUT_PIX{1'b0}};
            outputs_emitted <= 11'd0;
            acc[0] <= 32'sd0;
            acc[1] <= 32'sd0;
            acc[2] <= 32'sd0;
            acc[3] <= 32'sd0;
        end else begin
            if (!cmp_active && run && cyc_cnt == COMPUTE_START - 1) begin
                cmp_active <= 1'b1;
                cmp_pix    <= 10'd0;
                cmp_oh     <= 5'd0;
                cmp_ow     <= 5'd0;
                cmp_pass   <= 6'd0;
                cmp_step   <= 6'd0;
                acc[0] <= 32'sd0;
                acc[1] <= 32'sd0;
                acc[2] <= 32'sd0;
                acc[3] <= 32'sd0;
            end else if (cmp_active) begin
                if (is_issue) begin
                    acc[cur_lane] <= acc[cur_lane] + {{16{mac_prod[15]}}, mac_prod};
                end

                if (is_writeback) begin
                    for (i = 0; i < MP; i = i + 1) begin
                        wb_ch    = base_ch + i[7:0];
                        sum_wb   = {{32{acc[i][31]}}, acc[i]}
                                 + {{32{biases[wb_ch][31]}}, biases[wb_ch]};
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
                        if (cmp_pix[9] == 1'b0)
                            out_buf_b0[cmp_pix[8:0]][wb_ch*8 +: 8] <= sat_byte;
                        else
                            out_buf_b1[cmp_pix[8:0]][wb_ch*8 +: 8] <= sat_byte;
                    end
                end

                if (is_pass_end) begin
                    cmp_step <= 6'd0;
                    acc[0] <= 32'sd0;
                    acc[1] <= 32'sd0;
                    acc[2] <= 32'sd0;
                    acc[3] <= 32'sd0;
                    if (cmp_pass == OC_PASSES - 1) begin
                        cmp_pass          <= 6'd0;
                        pix_done[cmp_pix] <= 1'b1;
                        outputs_emitted   <= outputs_emitted + 11'd1;
                        if (cmp_pix == N_OUT_PIX - 1) begin
                            cmp_active <= 1'b0;
                        end else begin
                            cmp_pix <= cmp_pix + 10'd1;
                            if (cmp_ow == OW - 1) begin
                                cmp_ow <= 5'd0;
                                cmp_oh <= cmp_oh + 5'd1;
                            end else begin
                                cmp_ow <= cmp_ow + 5'd1;
                            end
                        end
                    end else begin
                        cmp_pass <= cmp_pass + 6'd1;
                    end
                end else begin
                    cmp_step <= cmp_step + 6'd1;
                end
            end
        end
    end

    reg [9:0] em_pix;

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out   <= 1'b0;
            data_out    <= {BEAT_W{1'b0}};
            em_pix      <= 10'd0;
            vector_done <= 1'b0;
        end else begin
            vector_done <= 1'b0;
            if (!vector_done && run &&
                (cyc_cnt >= FIRST_OUT_CYC - 1) &&
                (em_pix < N_OUT_PIX[9:0]) &&
                pix_done[em_pix]) begin
                // [INVARIANT:VALID_OUT_LATENCY]
                valid_out <= 1'b1;
                if (em_pix[9] == 1'b0)
                    data_out <= out_buf_b0[em_pix[8:0]];
                else
                    data_out <= out_buf_b1[em_pix[8:0]];
                if (em_pix == N_OUT_PIX[9:0] - 10'd1) begin
                    em_pix      <= 10'd0;
                    vector_done <= 1'b1;
                end else begin
                    em_pix <= em_pix + 10'd1;
                end
            end else begin
                valid_out <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire
