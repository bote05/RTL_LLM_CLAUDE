`timescale 1ns / 1ps
`default_nettype none

// ---------------------------------------------------------------------------
// node_conv_818 -- MobileNet-v2 depthwise 3x3 stride-2 padding-1 conv
//   C   = 96   (groups == in_channels == out_channels)
//   IH  = IW  = 112,  OH = OW = 56
//   KH  = KW  = 3,    PH = PW = 1, stride 2
//   MAC parallelism = 4 lanes; OC_PASSES = ceil(C/MP) = 24
//   Pass duration = MP * K_TOTAL + 6 = 4*9 + 6 = 42 cycles
//   Pipeline latency (LayerIR-authoritative) = 1124 cycles
//   COMPUTE_START = 115 -> pix_done[0] at edge T+1123 -> valid_out at cyc=1124
//   Contract: depthwise-conv (channel_tile = 96 = C, single beat per pixel)
//   Bus  : 768b in/out, 1 beat per pixel (96 channels packed INT8)
//   Scale: SCALE_MULT/2^SCALE_SHIFT = 16815 / 2^22 ~= 0.00400895
//          (target 0.00400894, |rel err| ~ 2.3e-5)
//   line_buf banked 13 x depth-1024 (each 786,432 bits < ~900k cap) for the
//   12544 input pixels. out_buf banked 4 x depth-1024 for 3136 output pixels.
//   No cross-channel reduction -- each lane reads its own channel-of-interest
//   from the same (kh, kw) tap, no IC iteration inside K_TOTAL.
// ---------------------------------------------------------------------------

module node_conv_818 #(
    parameter WEIGHTS_PATH = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_818_weights.hex",
    parameter BIAS_PATH    = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_818_bias.hex"
) (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    output wire           ready_in,
    input  wire [767:0]   data_in,
    output reg            valid_out,
    output reg  [767:0]   data_out
);

    localparam integer C             = 96;
    localparam integer IH            = 112;
    localparam integer IW            = 112;
    localparam integer OH            = 56;
    localparam integer OW            = 56;
    localparam integer KH            = 3;
    localparam integer KW            = 3;
    localparam integer SH            = 2;
    localparam integer SW            = 2;
    localparam integer PH            = 1;
    localparam integer PW            = 1;
    localparam integer MP            = 4;
    localparam integer K_TOTAL       = 9;
    localparam integer OC_PASSES     = 24;
    localparam integer FIRST_OUT_CYC = 1124;
    localparam integer COMPUTE_START = 115;
    localparam integer N_IN_PIX      = 12544;
    localparam integer N_OUT_PIX     = 3136;
    localparam integer BEAT_W        = 768;
    localparam integer SCALE_SHIFT   = 22;
    localparam signed [63:0] SCALE_MULT_64 = 64'sd16815;

    (* ram_style = "block" *) reg [BEAT_W-1:0] line_buf_b0  [0:1023];
    (* ram_style = "block" *) reg [BEAT_W-1:0] line_buf_b1  [0:1023];
    (* ram_style = "block" *) reg [BEAT_W-1:0] line_buf_b2  [0:1023];
    (* ram_style = "block" *) reg [BEAT_W-1:0] line_buf_b3  [0:1023];
    (* ram_style = "block" *) reg [BEAT_W-1:0] line_buf_b4  [0:1023];
    (* ram_style = "block" *) reg [BEAT_W-1:0] line_buf_b5  [0:1023];
    (* ram_style = "block" *) reg [BEAT_W-1:0] line_buf_b6  [0:1023];
    (* ram_style = "block" *) reg [BEAT_W-1:0] line_buf_b7  [0:1023];
    (* ram_style = "block" *) reg [BEAT_W-1:0] line_buf_b8  [0:1023];
    (* ram_style = "block" *) reg [BEAT_W-1:0] line_buf_b9  [0:1023];
    (* ram_style = "block" *) reg [BEAT_W-1:0] line_buf_b10 [0:1023];
    (* ram_style = "block" *) reg [BEAT_W-1:0] line_buf_b11 [0:1023];
    (* ram_style = "block" *) reg [BEAT_W-1:0] line_buf_b12 [0:1023];
    (* ram_style = "block" *) reg [BEAT_W-1:0] out_buf_b0   [0:1023];
    (* ram_style = "block" *) reg [BEAT_W-1:0] out_buf_b1   [0:1023];
    (* ram_style = "block" *) reg [BEAT_W-1:0] out_buf_b2   [0:1023];
    (* ram_style = "block" *) reg [BEAT_W-1:0] out_buf_b3   [0:1023];
    (* rom_style = "block", ram_style = "block" *) reg signed [7:0]  weights [0:C*K_TOTAL-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases  [0:C-1];

    initial begin
        $readmemh(WEIGHTS_PATH, weights);
        $readmemh(BIAS_PATH, biases);
    end

    reg         run;
    reg [15:0]  cyc_cnt;
    reg         vector_done;

    reg [13:0]  in_beat;
    reg         input_done;

    // [INVARIANT:READY_IN_GATING]
    assign ready_in = !input_done;

    reg                cmp_active;
    reg [11:0]         cmp_pix;
    reg [5:0]          cmp_oh;
    reg [5:0]          cmp_ow;
    reg [4:0]          cmp_pass;
    reg [5:0]          cmp_step;
    reg signed [31:0]  acc [0:MP-1];
    reg [N_OUT_PIX-1:0] pix_done;
    reg [12:0]         outputs_emitted;

    reg signed [7:0]   window [0:K_TOTAL-1][0:MP-1];

    wire is_issue     = cmp_active && (cmp_step < 6'd36);
    wire is_writeback = cmp_active && (cmp_step == 6'd38);
    wire is_pass_end  = cmp_active && (cmp_step == 6'd41);

    wire [1:0] cur_lane = cmp_step[1:0];
    wire [3:0] cur_k    = is_issue ? cmp_step[5:2] : 4'd0;
    wire [6:0] base_ch  = {cmp_pass, 2'b00};
    wire [6:0] cur_ch   = base_ch + {5'd0, cur_lane};

    wire [1:0] kh = (cur_k >= 4'd6) ? 2'd2 : (cur_k >= 4'd3) ? 2'd1 : 2'd0;
    wire [3:0] kw_calc = (kh == 2'd0) ? cur_k : (kh == 2'd1) ? (cur_k - 4'd3) : (cur_k - 4'd6);
    wire [1:0] kw = kw_calc[1:0];

    wire signed [8:0] oh_s = $signed({3'b000, cmp_oh});
    wire signed [8:0] ow_s = $signed({3'b000, cmp_ow});
    wire signed [8:0] in_r = (oh_s <<< 1) + $signed({7'd0, kh}) - 9'sd1;
    wire signed [8:0] in_c = (ow_s <<< 1) + $signed({7'd0, kw}) - 9'sd1;
    wire              in_bounds = (in_r >= 9'sd0) && (in_r < 9'sd112) &&
                                  (in_c >= 9'sd0) && (in_c < 9'sd112);

    wire [6:0]  in_r_u = in_r[6:0];
    wire [6:0]  in_c_u = in_c[6:0];
    wire [13:0] in_pix_idx = in_bounds
        ? (({7'd0, in_r_u} * 14'd112) + {7'd0, in_c_u})
        : 14'd0;
    wire [3:0]  rd_bank = in_pix_idx[13:10];
    wire [9:0]  rd_off  = in_pix_idx[9:0];

    reg [BEAT_W-1:0] line_buf_word;
    always @(*) begin
        case (rd_bank)
            4'd0:  line_buf_word = line_buf_b0[rd_off];
            4'd1:  line_buf_word = line_buf_b1[rd_off];
            4'd2:  line_buf_word = line_buf_b2[rd_off];
            4'd3:  line_buf_word = line_buf_b3[rd_off];
            4'd4:  line_buf_word = line_buf_b4[rd_off];
            4'd5:  line_buf_word = line_buf_b5[rd_off];
            4'd6:  line_buf_word = line_buf_b6[rd_off];
            4'd7:  line_buf_word = line_buf_b7[rd_off];
            4'd8:  line_buf_word = line_buf_b8[rd_off];
            4'd9:  line_buf_word = line_buf_b9[rd_off];
            4'd10: line_buf_word = line_buf_b10[rd_off];
            4'd11: line_buf_word = line_buf_b11[rd_off];
            4'd12: line_buf_word = line_buf_b12[rd_off];
            default: line_buf_word = {BEAT_W{1'b0}};
        endcase
    end

    wire signed [7:0]  act_byte = in_bounds ? $signed(line_buf_word[cur_ch*8 +: 8]) : 8'sd0;
    wire [9:0]         w_addr   = {3'd0, cur_ch} * 10'd9 + {6'd0, cur_k};
    wire signed [7:0]  w_byte   = weights[w_addr];
    wire signed [15:0] mac_prod = act_byte * w_byte;

    integer            i;
    integer            wj;
    reg signed [63:0]  sum_wb;
    reg signed [63:0]  prod_wb;
    reg signed [63:0]  round_wb;
    reg signed [63:0]  scaled_wb;
    reg signed [7:0]   sat_byte;
    reg [6:0]          wb_ch;

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
            in_beat    <= 14'd0;
            input_done <= 1'b0;
        end else if (valid_in && ready_in) begin
            case (in_beat[13:10])
                4'd0:  line_buf_b0[in_beat[9:0]]  <= data_in;
                4'd1:  line_buf_b1[in_beat[9:0]]  <= data_in;
                4'd2:  line_buf_b2[in_beat[9:0]]  <= data_in;
                4'd3:  line_buf_b3[in_beat[9:0]]  <= data_in;
                4'd4:  line_buf_b4[in_beat[9:0]]  <= data_in;
                4'd5:  line_buf_b5[in_beat[9:0]]  <= data_in;
                4'd6:  line_buf_b6[in_beat[9:0]]  <= data_in;
                4'd7:  line_buf_b7[in_beat[9:0]]  <= data_in;
                4'd8:  line_buf_b8[in_beat[9:0]]  <= data_in;
                4'd9:  line_buf_b9[in_beat[9:0]]  <= data_in;
                4'd10: line_buf_b10[in_beat[9:0]] <= data_in;
                4'd11: line_buf_b11[in_beat[9:0]] <= data_in;
                4'd12: line_buf_b12[in_beat[9:0]] <= data_in;
                default: ;
            endcase
            if (in_beat == N_IN_PIX - 1)
                input_done <= 1'b1;
            in_beat <= in_beat + 14'd1;
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
            cmp_pix         <= 12'd0;
            cmp_oh          <= 6'd0;
            cmp_ow          <= 6'd0;
            cmp_pass        <= 5'd0;
            cmp_step        <= 6'd0;
            pix_done        <= {N_OUT_PIX{1'b0}};
            outputs_emitted <= 13'd0;
            acc[0] <= 32'sd0;
            acc[1] <= 32'sd0;
            acc[2] <= 32'sd0;
            acc[3] <= 32'sd0;
        end else begin
            if (!cmp_active && run && cyc_cnt == COMPUTE_START - 1) begin
                cmp_active <= 1'b1;
                cmp_pix    <= 12'd0;
                cmp_oh     <= 6'd0;
                cmp_ow     <= 6'd0;
                cmp_pass   <= 5'd0;
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
                        wb_ch   = base_ch + i[6:0];
                        sum_wb  = {{32{acc[i][31]}}, acc[i]}
                                + {{32{biases[wb_ch][31]}}, biases[wb_ch]};
                        prod_wb = sum_wb * SCALE_MULT_64;
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
                        case (cmp_pix[11:10])
                            2'd0: out_buf_b0[cmp_pix[9:0]][wb_ch*8 +: 8] <= sat_byte;
                            2'd1: out_buf_b1[cmp_pix[9:0]][wb_ch*8 +: 8] <= sat_byte;
                            2'd2: out_buf_b2[cmp_pix[9:0]][wb_ch*8 +: 8] <= sat_byte;
                            2'd3: out_buf_b3[cmp_pix[9:0]][wb_ch*8 +: 8] <= sat_byte;
                            default: ;
                        endcase
                    end
                end

                if (is_pass_end) begin
                    cmp_step <= 6'd0;
                    acc[0] <= 32'sd0;
                    acc[1] <= 32'sd0;
                    acc[2] <= 32'sd0;
                    acc[3] <= 32'sd0;
                    if (cmp_pass == OC_PASSES - 1) begin
                        cmp_pass          <= 5'd0;
                        pix_done[cmp_pix] <= 1'b1;
                        outputs_emitted   <= outputs_emitted + 13'd1;
                        if (cmp_pix == N_OUT_PIX - 1) begin
                            cmp_active <= 1'b0;
                        end else begin
                            cmp_pix <= cmp_pix + 12'd1;
                            if (cmp_ow == OW - 1) begin
                                cmp_ow <= 6'd0;
                                cmp_oh <= cmp_oh + 6'd1;
                            end else begin
                                cmp_ow <= cmp_ow + 6'd1;
                            end
                        end
                    end else begin
                        cmp_pass <= cmp_pass + 5'd1;
                    end
                end else begin
                    cmp_step <= cmp_step + 6'd1;
                end
            end
        end
    end

    reg [11:0] em_pix;
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out   <= 1'b0;
            data_out    <= {BEAT_W{1'b0}};
            em_pix      <= 12'd0;
            vector_done <= 1'b0;
        end else begin
            vector_done <= 1'b0;
            if (!vector_done && run &&
                (cyc_cnt >= FIRST_OUT_CYC - 1) &&
                (em_pix < N_OUT_PIX[11:0]) &&
                pix_done[em_pix]) begin
                // [INVARIANT:VALID_OUT_LATENCY]
                valid_out <= 1'b1;
                case (em_pix[11:10])
                    2'd0: data_out <= out_buf_b0[em_pix[9:0]];
                    2'd1: data_out <= out_buf_b1[em_pix[9:0]];
                    2'd2: data_out <= out_buf_b2[em_pix[9:0]];
                    2'd3: data_out <= out_buf_b3[em_pix[9:0]];
                    default: data_out <= {BEAT_W{1'b0}};
                endcase
                if (em_pix == N_OUT_PIX[11:0] - 12'd1) begin
                    em_pix      <= 12'd0;
                    vector_done <= 1'b1;
                end else begin
                    em_pix <= em_pix + 12'd1;
                end
            end else begin
                valid_out <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire
