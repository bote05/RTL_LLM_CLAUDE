`timescale 1ns / 1ps
`default_nettype none

// ---------------------------------------------------------------------------
// node_conv_896 -- MobileNet-v2 depthwise 3x3 stride-1 padding-1 conv
//   C   = 960  (groups == in_channels == out_channels)
//   IH  = IW  = 7,   OH = OW = 7
//   KH  = KW  = 3,   PH = PW = 1, stride 1
//   MAC parallelism = 4 lanes; OC_PASSES = ceil(C/MP) = 240
//   Pass duration = MP * K_TOTAL + 6 = 4*9 + 6 = 42 cycles
//   Pipeline latency (LayerIR-authoritative) = 10091 cycles
//   Contract: depthwise-conv (tiled-streaming compatible, channel_tile = 512)
//   Bus  : 4096b in/out, 2 beats/pixel (512 ch + 448 real ch + 64 zero-pad ch)
//   Scale: SCALE_MULT/2^SCALE_SHIFT = 24546 / 2^20 ~= 0.023412705
//          (target 0.023412335, |err| ~ 1.6e-5)
//   No cross-channel reduction -- each lane reads its own channel-of-interest
//   from the same (kh, kw) tap, no IC iteration inside K_TOTAL.
// ---------------------------------------------------------------------------

module node_conv_896 #(
    parameter WEIGHTS_PATH = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_896_weights.hex",
    parameter BIAS_PATH    = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_896_bias.hex"
) (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           valid_in,
    output wire           ready_in,
    input  wire [4095:0]  data_in,
    output reg            valid_out,
    output reg  [4095:0]  data_out
);

    localparam integer C             = 960;
    localparam integer IH            = 7;
    localparam integer IW            = 7;
    localparam integer KH            = 3;
    localparam integer KW            = 3;
    localparam integer MP            = 4;
    localparam integer K_TOTAL       = 9;
    localparam integer OC_PASSES     = 240;
    localparam integer PASS_CYCLES   = 42;
    localparam integer FIRST_OUT_CYC = 10091;
    localparam integer COMPUTE_START = 10;
    localparam integer N_PIX         = 49;
    localparam integer N_BEATS       = 98;
    localparam integer BEAT_W        = 4096;
    localparam integer LO_CH         = 512;
    localparam integer HI_CH         = 448;
    localparam integer HI_W          = 3584;
    localparam integer SCALE_SHIFT   = 20;
    localparam signed [63:0] SCALE_MULT_64 = 64'sd24546;

    (* ram_style = "block" *) reg [BEAT_W-1:0] line_buf_lo [0:N_PIX-1];
    (* ram_style = "block" *) reg [HI_W-1:0]   line_buf_hi [0:N_PIX-1];
    (* ram_style = "block" *) reg [BEAT_W-1:0] out_lo      [0:N_PIX-1];
    (* ram_style = "block" *) reg [HI_W-1:0]   out_hi      [0:N_PIX-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [7:0]  weights [0:C*K_TOTAL-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases  [0:C-1];

    initial begin
        $readmemh(WEIGHTS_PATH, weights);
        $readmemh(BIAS_PATH,    biases);
    end

    reg        run;
    reg [15:0] cyc_cnt;
    reg        vector_done;

    reg [7:0]  in_beat;
    reg        input_done;

    assign ready_in = !input_done;

    reg               cmp_active;
    reg [5:0]         cmp_pix;
    reg [2:0]         cmp_oh;
    reg [2:0]         cmp_ow;
    reg [7:0]         cmp_pass;
    reg [5:0]         cmp_step;
    reg signed [31:0] acc [0:MP-1];
    reg [N_PIX-1:0]   pix_done;
    reg [10:0]        outputs_emitted;

    reg signed [7:0]  window [0:K_TOTAL-1][0:MP-1];

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

    wire [9:0] base_ch = {cmp_pass, 2'b00};
    wire [9:0] cur_ch  = base_ch + {8'd0, cur_lane};

    wire [1:0] kh = (cur_k >= 4'd6) ? 2'd2 :
                    (cur_k >= 4'd3) ? 2'd1 : 2'd0;
    wire [3:0] kw_calc = (kh == 2'd0) ? cur_k        :
                        (kh == 2'd1) ? (cur_k - 4'd3) :
                                        (cur_k - 4'd6);
    wire [1:0] kw = kw_calc[1:0];

    wire signed [4:0] oh_s = $signed({2'b00, cmp_oh});
    wire signed [4:0] ow_s = $signed({2'b00, cmp_ow});
    wire signed [4:0] in_r = oh_s + $signed({3'b000, kh}) - 5'sd1;
    wire signed [4:0] in_c = ow_s + $signed({3'b000, kw}) - 5'sd1;
    wire              in_bounds = (in_r >= 5'sd0) && (in_r < 5'sd7) &&
                                  (in_c >= 5'sd0) && (in_c < 5'sd7);

    wire [2:0] in_r_u = in_r[2:0];
    wire [2:0] in_c_u = in_c[2:0];
    wire [5:0] in_pix_idx = in_bounds
        ? (({3'd0, in_r_u} * 6'd7) + {3'd0, in_c_u})
        : 6'd0;

    wire [9:0]        hi_off_ch  = cur_ch - 10'd512;
    wire signed [7:0] tap_lo     = line_buf_lo[in_pix_idx][cur_ch*8 +: 8];
    wire signed [7:0] tap_hi     = line_buf_hi[in_pix_idx][hi_off_ch*8 +: 8];
    wire signed [7:0] act_byte   = !in_bounds         ? 8'sd0 :
                                   (cur_ch < 10'd512) ? tap_lo : tap_hi;

    wire [13:0] w_addr            = {4'd0, cur_ch} * 14'd9 + {10'd0, cur_k};
    wire signed [7:0]  w_byte     = weights[w_addr];
    wire signed [15:0] mac_prod   = act_byte * w_byte;

    integer            i;
    reg signed [63:0]  sum_wb;
    reg signed [63:0]  prod_wb;
    reg signed [63:0]  round_wb;
    reg signed [63:0]  scaled_wb;
    reg signed [7:0]   sat_byte;
    reg [9:0]          wb_ch;

    wire                                     cs_needs_real_input;
    wire                                     cs_ready_in;
    wire                                     cs_advance;
    wire                                     cs_in_frame_done;
    wire                                     cs_out_frame_done;
    wire [$clog2(IH + 1 + 1)-1:0]            cs_in_row;
    wire [$clog2(IW + 1 + 1)-1:0]            cs_in_col;
    wire                                     cs_output_fires;
    wire [$clog2(IH * IW + 1)-1:0]           cs_outputs_emitted;
    reg                                      cs_start;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cs_start <= 1'b0;
        else
            cs_start <= (!run && valid_in && ready_in);
    end

    coord_scheduler #(
        .IH(IH), .IW(IW), .OH(IH), .OW(IW),
        .KH(KH), .KW(KW), .SH(1), .SW(1), .PH(1), .PW(1)
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
            in_beat    <= 8'd0;
            input_done <= 1'b0;
        end else if (valid_in && ready_in) begin
            if (in_beat[0] == 1'b0)
                line_buf_lo[in_beat[7:1]] <= data_in;
            else
                line_buf_hi[in_beat[7:1]] <= data_in[HI_W-1:0];
            if (in_beat == 8'd97) input_done <= 1'b1;
            in_beat <= in_beat + 8'd1;
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            for (i = 0; i < K_TOTAL; i = i + 1) begin
                window[i][0] <= 8'sd0;
                window[i][1] <= 8'sd0;
                window[i][2] <= 8'sd0;
                window[i][3] <= 8'sd0;
            end
        end else if (is_issue) begin
            window[cur_k][cur_lane] <= act_byte;
        end
    end

    always @(posedge clk) begin
        if (!rst_n || vector_done) begin
            cmp_active      <= 1'b0;
            cmp_pix         <= 6'd0;
            cmp_oh          <= 3'd0;
            cmp_ow          <= 3'd0;
            cmp_pass        <= 8'd0;
            cmp_step        <= 6'd0;
            pix_done        <= {N_PIX{1'b0}};
            outputs_emitted <= 11'd0;
            acc[0] <= 32'sd0;
            acc[1] <= 32'sd0;
            acc[2] <= 32'sd0;
            acc[3] <= 32'sd0;
        end else begin
            if (!cmp_active && run && cyc_cnt == COMPUTE_START - 1) begin
                cmp_active <= 1'b1;
                cmp_pix    <= 6'd0;
                cmp_oh     <= 3'd0;
                cmp_ow     <= 3'd0;
                cmp_pass   <= 8'd0;
                cmp_step   <= 6'd0;
                acc[0] <= 32'sd0;
                acc[1] <= 32'sd0;
                acc[2] <= 32'sd0;
                acc[3] <= 32'sd0;
            end else if (cmp_active) begin
                if (is_issue) begin
                    acc[cur_lane] <= acc[cur_lane] + {{16{mac_prod[15]}}, mac_prod};
                end

                // INVARIANT: ROUNDING -- sign-aware half-up rounding before arithmetic right shift
                if (is_writeback) begin
                    for (i = 0; i < MP; i = i + 1) begin
                        wb_ch    = base_ch + i[9:0];
                        sum_wb   = {{32{acc[i][31]}}, acc[i]}
                                 + {{32{biases[wb_ch][31]}}, biases[wb_ch]};
                        prod_wb  = sum_wb * SCALE_MULT_64;
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
                        if (wb_ch < 10'd512)
                            out_lo[cmp_pix][wb_ch*8 +: 8] <= sat_byte;
                        else
                            out_hi[cmp_pix][(wb_ch - 10'd512)*8 +: 8] <= sat_byte;
                    end
                end

                if (is_pass_end) begin
                    cmp_step <= 6'd0;
                    acc[0] <= 32'sd0;
                    acc[1] <= 32'sd0;
                    acc[2] <= 32'sd0;
                    acc[3] <= 32'sd0;
                    if (cmp_pass == OC_PASSES - 1) begin
                        cmp_pass          <= 8'd0;
                        pix_done[cmp_pix] <= 1'b1;
                        outputs_emitted   <= outputs_emitted + 11'd1;
                        if (cmp_pix == N_PIX - 1) begin
                            cmp_active <= 1'b0;
                        end else begin
                            cmp_pix <= cmp_pix + 6'd1;
                            if (cmp_ow == 3'd6) begin
                                cmp_ow <= 3'd0;
                                cmp_oh <= cmp_oh + 3'd1;
                            end else begin
                                cmp_ow <= cmp_ow + 3'd1;
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

    // INVARIANT: VALID_OUT_LATENCY
    reg [5:0]  em_pix;
    reg        em_phase;

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out   <= 1'b0;
            data_out    <= {BEAT_W{1'b0}};
            em_pix      <= 6'd0;
            em_phase    <= 1'b0;
            vector_done <= 1'b0;
        end else begin
            vector_done <= 1'b0;
            if (!vector_done && run &&
                (cyc_cnt >= FIRST_OUT_CYC - 1) &&
                (em_pix < N_PIX[5:0]) &&
                pix_done[em_pix]) begin
                valid_out <= 1'b1;
                if (em_phase == 1'b0) begin
                    data_out <= out_lo[em_pix];
                    em_phase <= 1'b1;
                end else begin
                    data_out <= {{(BEAT_W - HI_W){1'b0}}, out_hi[em_pix]};
                    em_phase <= 1'b0;
                    if (em_pix == N_PIX[5:0] - 6'd1) begin
                        em_pix      <= 6'd0;
                        vector_done <= 1'b1;
                    end else begin
                        em_pix <= em_pix + 6'd1;
                    end
                end
            end else begin
                valid_out <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire
