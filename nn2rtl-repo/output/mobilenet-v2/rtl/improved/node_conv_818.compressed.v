`timescale 1ns / 1ps
`default_nettype none

// ---------------------------------------------------------------------------
// node_conv_818 -- MobileNet-v2 depthwise 3x3 stride-2 padding-1 conv
//   C   = 96  (groups == in_channels == out_channels)
//   IH  = IW  = 112,  OH = OW = 56
//   KH  = KW  = 3,    PH = PW = 1, stride 2
//   MAC parallelism = 4 lanes; OC_PASSES = ceil(C/MP) = 24
//   Pass duration = MP * K_TOTAL + 6 = 42 cycles
//   Pipeline latency (LayerIR-authoritative) = 1124 cycles
//   COMPUTE_START = 115 -> pix_done[0] at edge T+1123 -> valid_out at cyc 1124
//   Contract: depthwise-conv (single beat per pixel, channel_tile = 96 = C)
//   Bus  : 768 b in/out, 1 beat per pixel (96 channels packed INT8)
//   Scale: SCALE_MULT/2^SCALE_SHIFT = 16815 / 2^22 ~= 0.004008889
//          (target 0.004008940, |rel err| ~ 1.3e-5)
//   line_buf: 13 banks x 1024 entries (786,432 b/bank < 900k cap)
//   out_buf:  4 banks x 1024 entries
//   No cross-channel reduction -- each lane reads its own channel from the
//   same (kh, kw) tap, no IC iteration inside K_TOTAL.
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

    // [COMPRESS] out_buf restructured: 96 per-channel BRAM byte arrays
    //           (full-word aligned writes) replace 4x768b byte-written LUTRAM banks.
    (* ram_style = "block" *) reg signed [7:0] out_ch00 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch01 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch02 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch03 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch04 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch05 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch06 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch07 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch08 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch09 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch10 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch11 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch12 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch13 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch14 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch15 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch16 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch17 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch18 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch19 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch20 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch21 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch22 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch23 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch24 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch25 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch26 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch27 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch28 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch29 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch30 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch31 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch32 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch33 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch34 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch35 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch36 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch37 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch38 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch39 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch40 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch41 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch42 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch43 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch44 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch45 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch46 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch47 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch48 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch49 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch50 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch51 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch52 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch53 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch54 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch55 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch56 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch57 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch58 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch59 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch60 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch61 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch62 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch63 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch64 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch65 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch66 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch67 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch68 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch69 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch70 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch71 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch72 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch73 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch74 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch75 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch76 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch77 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch78 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch79 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch80 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch81 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch82 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch83 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch84 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch85 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch86 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch87 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch88 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch89 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch90 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch91 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch92 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch93 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch94 [0:N_OUT_PIX-1];
    (* ram_style = "block" *) reg signed [7:0] out_ch95 [0:N_OUT_PIX-1];

    (* rom_style = "block", ram_style = "block" *) reg signed [7:0]  weights [0:C*K_TOTAL-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases  [0:C-1];

    initial begin
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_818_weights.hex", weights);
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_818_bias.hex", biases);
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
    reg [5:0]          cmp_pass;
    reg [5:0]          cmp_step;
    reg signed [31:0]  acc [0:MP-1];
    reg [N_OUT_PIX-1:0] pix_done;
    reg [12:0]         outputs_emitted;

    // [COMPRESS] dead `window` array removed (written, never read).

    wire is_issue     = cmp_active && (cmp_step < 6'd36);
    wire is_writeback = cmp_active && (cmp_step == 6'd38);
    wire is_pass_end  = cmp_active && (cmp_step == 6'd41);

    wire [1:0] cur_lane = cmp_step[1:0];
    wire [3:0] cur_k    = is_issue ? cmp_step[5:2] : 4'd0;

    wire [6:0] base_ch = {cmp_pass[4:0], 2'b00};
    wire [6:0] cur_ch  = base_ch + {5'd0, cur_lane};

    wire [1:0] kh = (cur_k >= 4'd6) ? 2'd2 : (cur_k >= 4'd3) ? 2'd1 : 2'd0;
    wire [3:0] kw_calc = (kh == 2'd0) ? cur_k : (kh == 2'd1) ? (cur_k - 4'd3) : (cur_k - 4'd6);
    wire [1:0] kw = kw_calc[1:0];

    wire signed [8:0] oh_s = $signed({3'b000, cmp_oh});
    wire signed [8:0] ow_s = $signed({3'b000, cmp_ow});
    wire signed [8:0] in_r = (oh_s <<< 1) + $signed({7'd0, kh}) - 9'sd1;
    wire signed [8:0] in_c = (ow_s <<< 1) + $signed({7'd0, kw}) - 9'sd1;
    wire in_bounds = (in_r >= 9'sd0) && (in_r < 9'sd112) && (in_c >= 9'sd0) && (in_c < 9'sd112);

    wire [6:0] in_r_u = in_r[6:0];
    wire [6:0] in_c_u = in_c[6:0];
    wire [13:0] in_pix_idx = in_bounds ? (({7'd0, in_r_u} * 14'd112) + {7'd0, in_c_u}) : 14'd0;

    wire [3:0] rd_bank = in_pix_idx[13:10];
    wire [9:0] rd_off  = in_pix_idx[9:0];

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

    integer i;
    integer wj;
    reg signed [63:0] sum_wb, prod_wb, round_wb, scaled_wb;
    reg signed [7:0]  sat_byte;
    reg [6:0]         wb_ch;

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
        if (!rst_n) cs_start <= 1'b0;
        else        cs_start <= (!run && valid_in && ready_in);
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
            if (in_beat == N_IN_PIX - 1) input_done <= 1'b1;
            in_beat <= in_beat + 14'd1;
        end
    end

    // [COMPRESS] dead `window` write process removed.

    always @(posedge clk) begin
        if (!rst_n || vector_done) begin
            cmp_active      <= 1'b0;
            cmp_pix         <= 12'd0;
            cmp_oh          <= 6'd0;
            cmp_ow          <= 6'd0;
            cmp_pass        <= 6'd0;
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
                        case (wb_ch)
                            7'd0: out_ch00[cmp_pix] <= sat_byte;
                            7'd1: out_ch01[cmp_pix] <= sat_byte;
                            7'd2: out_ch02[cmp_pix] <= sat_byte;
                            7'd3: out_ch03[cmp_pix] <= sat_byte;
                            7'd4: out_ch04[cmp_pix] <= sat_byte;
                            7'd5: out_ch05[cmp_pix] <= sat_byte;
                            7'd6: out_ch06[cmp_pix] <= sat_byte;
                            7'd7: out_ch07[cmp_pix] <= sat_byte;
                            7'd8: out_ch08[cmp_pix] <= sat_byte;
                            7'd9: out_ch09[cmp_pix] <= sat_byte;
                            7'd10: out_ch10[cmp_pix] <= sat_byte;
                            7'd11: out_ch11[cmp_pix] <= sat_byte;
                            7'd12: out_ch12[cmp_pix] <= sat_byte;
                            7'd13: out_ch13[cmp_pix] <= sat_byte;
                            7'd14: out_ch14[cmp_pix] <= sat_byte;
                            7'd15: out_ch15[cmp_pix] <= sat_byte;
                            7'd16: out_ch16[cmp_pix] <= sat_byte;
                            7'd17: out_ch17[cmp_pix] <= sat_byte;
                            7'd18: out_ch18[cmp_pix] <= sat_byte;
                            7'd19: out_ch19[cmp_pix] <= sat_byte;
                            7'd20: out_ch20[cmp_pix] <= sat_byte;
                            7'd21: out_ch21[cmp_pix] <= sat_byte;
                            7'd22: out_ch22[cmp_pix] <= sat_byte;
                            7'd23: out_ch23[cmp_pix] <= sat_byte;
                            7'd24: out_ch24[cmp_pix] <= sat_byte;
                            7'd25: out_ch25[cmp_pix] <= sat_byte;
                            7'd26: out_ch26[cmp_pix] <= sat_byte;
                            7'd27: out_ch27[cmp_pix] <= sat_byte;
                            7'd28: out_ch28[cmp_pix] <= sat_byte;
                            7'd29: out_ch29[cmp_pix] <= sat_byte;
                            7'd30: out_ch30[cmp_pix] <= sat_byte;
                            7'd31: out_ch31[cmp_pix] <= sat_byte;
                            7'd32: out_ch32[cmp_pix] <= sat_byte;
                            7'd33: out_ch33[cmp_pix] <= sat_byte;
                            7'd34: out_ch34[cmp_pix] <= sat_byte;
                            7'd35: out_ch35[cmp_pix] <= sat_byte;
                            7'd36: out_ch36[cmp_pix] <= sat_byte;
                            7'd37: out_ch37[cmp_pix] <= sat_byte;
                            7'd38: out_ch38[cmp_pix] <= sat_byte;
                            7'd39: out_ch39[cmp_pix] <= sat_byte;
                            7'd40: out_ch40[cmp_pix] <= sat_byte;
                            7'd41: out_ch41[cmp_pix] <= sat_byte;
                            7'd42: out_ch42[cmp_pix] <= sat_byte;
                            7'd43: out_ch43[cmp_pix] <= sat_byte;
                            7'd44: out_ch44[cmp_pix] <= sat_byte;
                            7'd45: out_ch45[cmp_pix] <= sat_byte;
                            7'd46: out_ch46[cmp_pix] <= sat_byte;
                            7'd47: out_ch47[cmp_pix] <= sat_byte;
                            7'd48: out_ch48[cmp_pix] <= sat_byte;
                            7'd49: out_ch49[cmp_pix] <= sat_byte;
                            7'd50: out_ch50[cmp_pix] <= sat_byte;
                            7'd51: out_ch51[cmp_pix] <= sat_byte;
                            7'd52: out_ch52[cmp_pix] <= sat_byte;
                            7'd53: out_ch53[cmp_pix] <= sat_byte;
                            7'd54: out_ch54[cmp_pix] <= sat_byte;
                            7'd55: out_ch55[cmp_pix] <= sat_byte;
                            7'd56: out_ch56[cmp_pix] <= sat_byte;
                            7'd57: out_ch57[cmp_pix] <= sat_byte;
                            7'd58: out_ch58[cmp_pix] <= sat_byte;
                            7'd59: out_ch59[cmp_pix] <= sat_byte;
                            7'd60: out_ch60[cmp_pix] <= sat_byte;
                            7'd61: out_ch61[cmp_pix] <= sat_byte;
                            7'd62: out_ch62[cmp_pix] <= sat_byte;
                            7'd63: out_ch63[cmp_pix] <= sat_byte;
                            7'd64: out_ch64[cmp_pix] <= sat_byte;
                            7'd65: out_ch65[cmp_pix] <= sat_byte;
                            7'd66: out_ch66[cmp_pix] <= sat_byte;
                            7'd67: out_ch67[cmp_pix] <= sat_byte;
                            7'd68: out_ch68[cmp_pix] <= sat_byte;
                            7'd69: out_ch69[cmp_pix] <= sat_byte;
                            7'd70: out_ch70[cmp_pix] <= sat_byte;
                            7'd71: out_ch71[cmp_pix] <= sat_byte;
                            7'd72: out_ch72[cmp_pix] <= sat_byte;
                            7'd73: out_ch73[cmp_pix] <= sat_byte;
                            7'd74: out_ch74[cmp_pix] <= sat_byte;
                            7'd75: out_ch75[cmp_pix] <= sat_byte;
                            7'd76: out_ch76[cmp_pix] <= sat_byte;
                            7'd77: out_ch77[cmp_pix] <= sat_byte;
                            7'd78: out_ch78[cmp_pix] <= sat_byte;
                            7'd79: out_ch79[cmp_pix] <= sat_byte;
                            7'd80: out_ch80[cmp_pix] <= sat_byte;
                            7'd81: out_ch81[cmp_pix] <= sat_byte;
                            7'd82: out_ch82[cmp_pix] <= sat_byte;
                            7'd83: out_ch83[cmp_pix] <= sat_byte;
                            7'd84: out_ch84[cmp_pix] <= sat_byte;
                            7'd85: out_ch85[cmp_pix] <= sat_byte;
                            7'd86: out_ch86[cmp_pix] <= sat_byte;
                            7'd87: out_ch87[cmp_pix] <= sat_byte;
                            7'd88: out_ch88[cmp_pix] <= sat_byte;
                            7'd89: out_ch89[cmp_pix] <= sat_byte;
                            7'd90: out_ch90[cmp_pix] <= sat_byte;
                            7'd91: out_ch91[cmp_pix] <= sat_byte;
                            7'd92: out_ch92[cmp_pix] <= sat_byte;
                            7'd93: out_ch93[cmp_pix] <= sat_byte;
                            7'd94: out_ch94[cmp_pix] <= sat_byte;
                            7'd95: out_ch95[cmp_pix] <= sat_byte;
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
                        cmp_pass          <= 6'd0;
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
                        cmp_pass <= cmp_pass + 6'd1;
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
                data_out <= {out_ch95[em_pix], out_ch94[em_pix], out_ch93[em_pix], out_ch92[em_pix], out_ch91[em_pix], out_ch90[em_pix], out_ch89[em_pix], out_ch88[em_pix], out_ch87[em_pix], out_ch86[em_pix], out_ch85[em_pix], out_ch84[em_pix], out_ch83[em_pix], out_ch82[em_pix], out_ch81[em_pix], out_ch80[em_pix], out_ch79[em_pix], out_ch78[em_pix], out_ch77[em_pix], out_ch76[em_pix], out_ch75[em_pix], out_ch74[em_pix], out_ch73[em_pix], out_ch72[em_pix], out_ch71[em_pix], out_ch70[em_pix], out_ch69[em_pix], out_ch68[em_pix], out_ch67[em_pix], out_ch66[em_pix], out_ch65[em_pix], out_ch64[em_pix], out_ch63[em_pix], out_ch62[em_pix], out_ch61[em_pix], out_ch60[em_pix], out_ch59[em_pix], out_ch58[em_pix], out_ch57[em_pix], out_ch56[em_pix], out_ch55[em_pix], out_ch54[em_pix], out_ch53[em_pix], out_ch52[em_pix], out_ch51[em_pix], out_ch50[em_pix], out_ch49[em_pix], out_ch48[em_pix], out_ch47[em_pix], out_ch46[em_pix], out_ch45[em_pix], out_ch44[em_pix], out_ch43[em_pix], out_ch42[em_pix], out_ch41[em_pix], out_ch40[em_pix], out_ch39[em_pix], out_ch38[em_pix], out_ch37[em_pix], out_ch36[em_pix], out_ch35[em_pix], out_ch34[em_pix], out_ch33[em_pix], out_ch32[em_pix], out_ch31[em_pix], out_ch30[em_pix], out_ch29[em_pix], out_ch28[em_pix], out_ch27[em_pix], out_ch26[em_pix], out_ch25[em_pix], out_ch24[em_pix], out_ch23[em_pix], out_ch22[em_pix], out_ch21[em_pix], out_ch20[em_pix], out_ch19[em_pix], out_ch18[em_pix], out_ch17[em_pix], out_ch16[em_pix], out_ch15[em_pix], out_ch14[em_pix], out_ch13[em_pix], out_ch12[em_pix], out_ch11[em_pix], out_ch10[em_pix], out_ch09[em_pix], out_ch08[em_pix], out_ch07[em_pix], out_ch06[em_pix], out_ch05[em_pix], out_ch04[em_pix], out_ch03[em_pix], out_ch02[em_pix], out_ch01[em_pix], out_ch00[em_pix]};
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
