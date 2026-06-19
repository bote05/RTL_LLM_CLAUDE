`timescale 1ns / 1ps
`default_nettype none

// ---------------------------------------------------------------------------
// node_conv_836 -- MobileNet-v2 depthwise 3x3 stride-1 padding-1 conv.
//   C  = 192 (groups == in_channels == out_channels)
//   IH = IW = 28, OH = OW = 28
//   KH = KW = 3, PH = PW = 1, stride 1
//   MAC parallelism (MP) = 4 lanes processing 4 distinct channels in parallel
//   OC_PASSES = ceil(C/MP) = 48 passes per output pixel
//   PASS_CYCLES = K_TOTAL (9 issue) + 1 writeback = 10 cycles per pass
//   Pipeline latency (LayerIR-authoritative) = 2048 cycles
//   Contract: depthwise-conv with channel_tile = 192 -> bus is single beat
//             (1536 bits = 192 channels * 8 bits).
//   No cross-channel reduction: each lane MACs its own channel's 9 taps
//   against its own 3x3 filter (weights[ch * 9 + k]).
//   SCALE_MULT / 2^SCALE_SHIFT = 2707 / 2^16 ~= 0.041305542
//     (target 0.04130484182389141, |rel err| ~ 1.7e-5)
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
    localparam integer KH            = 3;
    localparam integer KW            = 3;
    localparam integer K_TOTAL       = 9;
    localparam integer MP            = 4;
    localparam integer OC_PASSES     = 48;
    localparam integer N_PIX         = 784;
    localparam integer PASS_CYCLES   = 10;
    localparam integer COMPUTE_START = 800;
    localparam integer FIRST_OUT_CYC = 2048;
    localparam integer LO_W          = 768;
    localparam integer HALF_C        = 96;
    localparam integer SCALE_SHIFT   = 16;
    localparam signed [63:0] SCALE_MULT_64    = 64'sd2707;
    localparam signed [63:0] SCALE_ROUND_HALF = 64'sd1 <<< (SCALE_SHIFT - 1);

    (* ram_style = "block" *) reg [LO_W-1:0] act_buf_lo [0:N_PIX-1];
    (* ram_style = "block" *) reg [LO_W-1:0] act_buf_hi [0:N_PIX-1];
    (* ram_style = "block" *) reg [LO_W-1:0] out_buf_lo [0:N_PIX-1];
    (* ram_style = "block" *) reg [LO_W-1:0] out_buf_hi [0:N_PIX-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [7:0]  weights [0:C*K_TOTAL-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases  [0:C-1];

    initial begin
        $readmemh(WEIGHTS_PATH, weights);
        $readmemh(BIAS_PATH,    biases);
    end

    reg        run;
    reg [15:0] cyc_cnt;
    reg        vector_done;

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

    reg [9:0]  in_beat;
    reg        input_done;

    // [INVARIANT:READY_IN_GATING]
    assign ready_in = !input_done;

    always @(posedge clk) begin
        if (!rst_n || vector_done) begin
            in_beat    <= 10'd0;
            input_done <= 1'b0;
        end else if (valid_in && ready_in) begin
            act_buf_lo[in_beat] <= data_in[LO_W-1:0];
            act_buf_hi[in_beat] <= data_in[2*LO_W-1:LO_W];
            if (in_beat == N_PIX[9:0] - 10'd1) input_done <= 1'b1;
            in_beat <= in_beat + 10'd1;
        end
    end

    reg cs_start;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) cs_start <= 1'b0;
        else        cs_start <= (!run && valid_in && ready_in);
    end

    wire                                     cs_ready_in;
    wire                                     cs_needs_real_input;
    wire                                     cs_advance;
    wire                                     cs_in_frame_done;
    wire                                     cs_out_frame_done;
    wire                                     cs_output_fires;
    wire [$clog2(IH + 1 + 1)-1:0]            cs_in_row;
    wire [$clog2(IW + 1 + 1)-1:0]            cs_in_col;
    wire [$clog2(IH * IW + 1)-1:0]           cs_outputs_emitted;

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

    reg               cmp_active;
    reg [9:0]         cmp_pix;
    reg [4:0]         cmp_oh;
    reg [4:0]         cmp_ow;
    reg [5:0]         cmp_pass;
    reg [3:0]         cmp_step;
    reg signed [31:0] acc [0:MP-1];
    reg [N_PIX-1:0]   pix_done;

    reg [LO_W-1:0] cur_pix_lo;
    reg [LO_W-1:0] cur_pix_hi;

    wire is_issue     = cmp_active && (cmp_step <  4'd9);
    wire is_writeback = cmp_active && (cmp_step == 4'd9);
    wire [3:0] cur_k  = cmp_step;

    wire [1:0] kh_w = (cur_k >= 4'd6) ? 2'd2 :
                      (cur_k >= 4'd3) ? 2'd1 : 2'd0;
    wire [3:0] kw_calc = (kh_w == 2'd0) ? cur_k :
                         (kh_w == 2'd1) ? (cur_k - 4'd3) :
                                          (cur_k - 4'd6);
    wire [1:0] kw_w = kw_calc[1:0];

    wire signed [6:0] oh_s = $signed({2'b00, cmp_oh});
    wire signed [6:0] ow_s = $signed({2'b00, cmp_ow});
    wire signed [6:0] in_r = oh_s + $signed({5'b00000, kh_w}) - 7'sd1;
    wire signed [6:0] in_c = ow_s + $signed({5'b00000, kw_w}) - 7'sd1;
    wire              in_bounds = (in_r >= 7'sd0) && (in_r < 7'sd28) &&
                                  (in_c >= 7'sd0) && (in_c < 7'sd28);

    wire [4:0] in_r_u       = in_r[4:0];
    wire [4:0] in_c_u       = in_c[4:0];
    wire [9:0] in_pix_idx_c = ({5'd0, in_r_u} * 10'd28) + {5'd0, in_c_u};
    wire [9:0] in_pix_idx   = in_bounds ? in_pix_idx_c : 10'd0;

    wire [LO_W-1:0] pix_lo = act_buf_lo[in_pix_idx];
    wire [LO_W-1:0] pix_hi = act_buf_hi[in_pix_idx];

    wire [7:0] base_ch = {cmp_pass, 2'b00};
    wire [7:0] ch0 = base_ch + 8'd0;
    wire [7:0] ch1 = base_ch + 8'd1;
    wire [7:0] ch2 = base_ch + 8'd2;
    wire [7:0] ch3 = base_ch + 8'd3;

    wire signed [7:0] act0 = !in_bounds ? 8'sd0 :
        ((ch0 < HALF_C[7:0]) ? $signed(pix_lo[ch0*8 +: 8])
                             : $signed(pix_hi[(ch0 - HALF_C[7:0])*8 +: 8]));
    wire signed [7:0] act1 = !in_bounds ? 8'sd0 :
        ((ch1 < HALF_C[7:0]) ? $signed(pix_lo[ch1*8 +: 8])
                             : $signed(pix_hi[(ch1 - HALF_C[7:0])*8 +: 8]));
    wire signed [7:0] act2 = !in_bounds ? 8'sd0 :
        ((ch2 < HALF_C[7:0]) ? $signed(pix_lo[ch2*8 +: 8])
                             : $signed(pix_hi[(ch2 - HALF_C[7:0])*8 +: 8]));
    wire signed [7:0] act3 = !in_bounds ? 8'sd0 :
        ((ch3 < HALF_C[7:0]) ? $signed(pix_lo[ch3*8 +: 8])
                             : $signed(pix_hi[(ch3 - HALF_C[7:0])*8 +: 8]));

    wire [10:0] w_addr0 = {3'd0, ch0} * 11'd9 + {7'd0, cur_k};
    wire [10:0] w_addr1 = {3'd0, ch1} * 11'd9 + {7'd0, cur_k};
    wire [10:0] w_addr2 = {3'd0, ch2} * 11'd9 + {7'd0, cur_k};
    wire [10:0] w_addr3 = {3'd0, ch3} * 11'd9 + {7'd0, cur_k};

    wire signed [7:0] w0 = weights[w_addr0];
    wire signed [7:0] w1 = weights[w_addr1];
    wire signed [7:0] w2 = weights[w_addr2];
    wire signed [7:0] w3 = weights[w_addr3];

    wire signed [15:0] p0 = act0 * w0;
    wire signed [15:0] p1 = act1 * w1;
    wire signed [15:0] p2 = act2 * w2;
    wire signed [15:0] p3 = act3 * w3;

    integer            i;
    reg signed [63:0]  sum_wb;
    reg signed [63:0]  prod_wb;
    reg signed [63:0]  round_wb;
    reg signed [63:0]  scaled_wb;
    reg signed [7:0]   sat_byte;
    reg [7:0]          wb_ch;

    always @(posedge clk) begin
        if (!rst_n || vector_done) begin
            cmp_active <= 1'b0;
            cmp_pix    <= 10'd0;
            cmp_oh     <= 5'd0;
            cmp_ow     <= 5'd0;
            cmp_pass   <= 6'd0;
            cmp_step   <= 4'd0;
            pix_done   <= {N_PIX{1'b0}};
            acc[0]     <= 32'sd0;
            acc[1]     <= 32'sd0;
            acc[2]     <= 32'sd0;
            acc[3]     <= 32'sd0;
            cur_pix_lo <= {LO_W{1'b0}};
            cur_pix_hi <= {LO_W{1'b0}};
        end else begin
            if (!cmp_active && run && (cyc_cnt == COMPUTE_START[15:0] - 16'd1)) begin
                cmp_active <= 1'b1;
                acc[0]     <= 32'sd0;
                acc[1]     <= 32'sd0;
                acc[2]     <= 32'sd0;
                acc[3]     <= 32'sd0;
            end else if (cmp_active) begin
                if (is_issue) begin
                    acc[0] <= acc[0] + {{16{p0[15]}}, p0};
                    acc[1] <= acc[1] + {{16{p1[15]}}, p1};
                    acc[2] <= acc[2] + {{16{p2[15]}}, p2};
                    acc[3] <= acc[3] + {{16{p3[15]}}, p3};
                end

                if (is_writeback) begin
                    for (i = 0; i < MP; i = i + 1) begin
                        wb_ch    = base_ch + i[7:0];
                        sum_wb   = {{32{acc[i][31]}}, acc[i]}
                                 + {{32{biases[wb_ch][31]}}, biases[wb_ch]};
                        prod_wb  = sum_wb * SCALE_MULT_64;
                        // [INVARIANT:ROUNDING]
                        if (sum_wb >= 64'sd0)
                            round_wb = prod_wb + SCALE_ROUND_HALF;
                        else
                            round_wb = prod_wb + SCALE_ROUND_HALF - 64'sd1;
                        scaled_wb = round_wb >>> SCALE_SHIFT;
                        if (scaled_wb > 64'sd127)
                            sat_byte = 8'sd127;
                        else if (scaled_wb < -64'sd128)
                            sat_byte = -8'sd128;
                        else
                            sat_byte = scaled_wb[7:0];
                        if (wb_ch < HALF_C[7:0])
                            cur_pix_lo[wb_ch*8 +: 8] <= sat_byte;
                        else
                            cur_pix_hi[(wb_ch - HALF_C[7:0])*8 +: 8] <= sat_byte;
                    end
                end

                if (cmp_step == PASS_CYCLES[3:0] - 4'd1) begin
                    cmp_step <= 4'd0;
                    acc[0]   <= 32'sd0;
                    acc[1]   <= 32'sd0;
                    acc[2]   <= 32'sd0;
                    acc[3]   <= 32'sd0;
                    if (cmp_pass == OC_PASSES[5:0] - 6'd1) begin
                        cmp_pass          <= 6'd0;
                        pix_done[cmp_pix] <= 1'b1;
                        out_buf_lo[cmp_pix] <= cur_pix_lo;
                        out_buf_hi[cmp_pix] <= cur_pix_hi;
                        if (cmp_pix == N_PIX[9:0] - 10'd1) begin
                            cmp_active <= 1'b0;
                        end else begin
                            cmp_pix <= cmp_pix + 10'd1;
                            if (cmp_ow == 5'd27) begin
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
                    cmp_step <= cmp_step + 4'd1;
                end
            end
        end
    end

    reg [9:0] em_pix;

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out   <= 1'b0;
            data_out    <= 1536'd0;
            em_pix      <= 10'd0;
            vector_done <= 1'b0;
        end else begin
            vector_done <= 1'b0;
            if (!vector_done && run &&
                (cyc_cnt >= FIRST_OUT_CYC[15:0] - 16'd1) &&
                (em_pix < N_PIX[9:0]) &&
                pix_done[em_pix]) begin
                // [INVARIANT:VALID_OUT_LATENCY]
                valid_out <= 1'b1;
                data_out  <= {out_buf_hi[em_pix], out_buf_lo[em_pix]};
                if (em_pix == N_PIX[9:0] - 10'd1) begin
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
