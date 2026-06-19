`timescale 1ns/1ps
`default_nettype none

// ---------------------------------------------------------------------------
// node_conv_824 -- MobileNet-v2 depthwise 3x3 stride-1 padding-1 conv
//   C   = 144  (groups == in_channels == out_channels)
//   IH  = IW  = 56, OH = OW = 56
//   KH  = KW  = 3,  PH = PW = 1, stride 1
//   MAC parallelism MP = 4 lanes; OC_PASSES = ceil(C/MP) = 36
//   Pass duration = MP * K_TOTAL + 6 = 42 cycles
//   First valid_out cycle (LayerIR-authoritative) = 1572
//     = fill_rows*(IW+PW) + fill_cols + OC_PASSES*PASS_CYCLES + 1
//     = 1*57 + 2 + 36*42 + 1 = 1572
//   Contract: depthwise-conv (channel_tile == C, single-beat bus)
//   Bus  : 1152b in/out, 1 beat/pixel
//   Scale: SCALE_MULT/2^SCALE_SHIFT = 8513 / 2^20 ~= 0.008118629
//          (target 0.008118643, |err| ~ 1.4e-8 absolute, ~1.7e-6 relative)
//   No cross-channel reduction -- each lane drives one output channel that
//   reads only its own input channel and its own 9-tap filter.
// ---------------------------------------------------------------------------

module node_conv_824 #(
    parameter WEIGHTS_PATH = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_824_weights.hex",
    parameter BIAS_PATH    = "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_824_bias.hex"
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
    localparam integer KH            = 3;
    localparam integer KW            = 3;
    localparam integer MP            = 4;
    localparam integer K_TOTAL       = 9;
    localparam integer OC_PASSES     = 36;
    localparam integer PASS_CYCLES   = 42;
    localparam integer COMPUTE_START = 59;
    localparam integer N_PIX         = 3136;
    localparam integer BEAT_W        = 1152;

    localparam integer SCALE_SHIFT   = 20;
    localparam signed [63:0] SCALE_MULT_64 = 64'sd8513;

    (* ram_style = "block" *) reg [BEAT_W-1:0] line_buf [0:N_PIX-1];
    (* ram_style = "block" *) reg [BEAT_W-1:0] out_buf  [0:N_PIX-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [7:0]  weights [0:C*K_TOTAL-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases  [0:C-1];

    initial begin
        $readmemh(WEIGHTS_PATH, weights);
        $readmemh(BIAS_PATH,    biases);
    end

    reg vector_done;

    reg [11:0] in_pix;
    reg        input_done;
    // [INVARIANT:READY_IN_GATING]
    assign ready_in = !input_done;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_pix     <= 12'd0;
            input_done <= 1'b0;
        end else if (vector_done) begin
            in_pix     <= 12'd0;
            input_done <= 1'b0;
        end else if (valid_in && ready_in) begin
            line_buf[in_pix] <= data_in;
            if (in_pix == N_PIX[11:0] - 12'd1)
                input_done <= 1'b1;
            in_pix <= in_pix + 12'd1;
        end
    end

    reg        run;
    reg [23:0] cyc_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            run     <= 1'b0;
            cyc_cnt <= 24'd0;
        end else if (vector_done) begin
            run     <= 1'b0;
            cyc_cnt <= 24'd0;
        end else if (!run) begin
            if (valid_in && ready_in) begin
                run     <= 1'b1;
                cyc_cnt <= 24'd1;
            end
        end else if (cyc_cnt != 24'hFFFFFF) begin
            cyc_cnt <= cyc_cnt + 24'd1;
        end
    end

    reg        cmp_active;
    reg [11:0] cmp_pix;
    reg [5:0]  cmp_oh;
    reg [5:0]  cmp_ow;
    reg [7:0]  cmp_pass;
    reg [5:0]  cmp_step;
    reg signed [31:0] acc0;
    reg signed [31:0] acc1;
    reg signed [31:0] acc2;
    reg signed [31:0] acc3;

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
    wire [3:0] kw_calc = (kh == 2'd0) ? cur_k          :
                         (kh == 2'd1) ? (cur_k - 4'd3) :
                                        (cur_k - 4'd6);
    wire [1:0] kw = kw_calc[1:0];

    wire signed [7:0] in_r =
        $signed({2'b00, cmp_oh}) + $signed({6'b000000, kh}) - 8'sd1;
    wire signed [7:0] in_c =
        $signed({2'b00, cmp_ow}) + $signed({6'b000000, kw}) - 8'sd1;
    wire in_bounds = (in_r >= 8'sd0) && (in_r < 8'sd56) &&
                     (in_c >= 8'sd0) && (in_c < 8'sd56);

    wire [5:0]  in_r_u    = in_r[5:0];
    wire [5:0]  in_c_u    = in_c[5:0];
    wire [11:0] in_pix_idx = in_bounds
        ? ({6'd0, in_r_u} * 12'd56 + {6'd0, in_c_u})
        : 12'd0;

    wire signed [7:0] act_byte = !in_bounds ? 8'sd0
                                            : $signed(line_buf[in_pix_idx][cur_ch*8 +: 8]);

    wire [12:0] w_addr            = {3'd0, cur_ch} * 13'd9 + {9'd0, cur_k};
    wire signed [7:0]  w_byte     = weights[w_addr];
    wire signed [15:0] mac_prod   = act_byte * w_byte;

    integer            wb_i;
    reg signed [63:0]  sum_wb;
    reg signed [63:0]  acc_sel;
    reg signed [63:0]  prod_wb;
    reg signed [63:0]  round_wb;
    reg signed [63:0]  scaled_wb;
    reg signed [7:0]   sat_byte;
    reg [9:0]          wb_ch;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cmp_active <= 1'b0;
            cmp_pix    <= 12'd0;
            cmp_oh     <= 6'd0;
            cmp_ow     <= 6'd0;
            cmp_pass   <= 8'd0;
            cmp_step   <= 6'd0;
            acc0       <= 32'sd0;
            acc1       <= 32'sd0;
            acc2       <= 32'sd0;
            acc3       <= 32'sd0;
        end else if (vector_done) begin
            cmp_active <= 1'b0;
            cmp_pix    <= 12'd0;
            cmp_oh     <= 6'd0;
            cmp_ow     <= 6'd0;
            cmp_pass   <= 8'd0;
            cmp_step   <= 6'd0;
            acc0       <= 32'sd0;
            acc1       <= 32'sd0;
            acc2       <= 32'sd0;
            acc3       <= 32'sd0;
        end else begin
            if (!cmp_active && run && cyc_cnt == COMPUTE_START[23:0] - 24'd1) begin
                cmp_active <= 1'b1;
                cmp_pix    <= 12'd0;
                cmp_oh     <= 6'd0;
                cmp_ow     <= 6'd0;
                cmp_pass   <= 8'd0;
                cmp_step   <= 6'd0;
                acc0       <= 32'sd0;
                acc1       <= 32'sd0;
                acc2       <= 32'sd0;
                acc3       <= 32'sd0;
            end else if (cmp_active) begin
                if (is_issue) begin
                    case (cur_lane)
                        2'd0: acc0 <= acc0 + {{16{mac_prod[15]}}, mac_prod};
                        2'd1: acc1 <= acc1 + {{16{mac_prod[15]}}, mac_prod};
                        2'd2: acc2 <= acc2 + {{16{mac_prod[15]}}, mac_prod};
                        2'd3: acc3 <= acc3 + {{16{mac_prod[15]}}, mac_prod};
                    endcase
                end

                if (is_writeback) begin
                    for (wb_i = 0; wb_i < MP; wb_i = wb_i + 1) begin
                        wb_ch = base_ch + wb_i[9:0];
                        case (wb_i[1:0])
                            2'd0: acc_sel = {{32{acc0[31]}}, acc0};
                            2'd1: acc_sel = {{32{acc1[31]}}, acc1};
                            2'd2: acc_sel = {{32{acc2[31]}}, acc2};
                            2'd3: acc_sel = {{32{acc3[31]}}, acc3};
                        endcase
                        sum_wb   = acc_sel + {{32{biases[wb_ch][31]}}, biases[wb_ch]};
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
                    acc0     <= 32'sd0;
                    acc1     <= 32'sd0;
                    acc2     <= 32'sd0;
                    acc3     <= 32'sd0;
                    if (cmp_pass == OC_PASSES[7:0] - 8'd1) begin
                        cmp_pass <= 8'd0;
                        if (cmp_pix == N_PIX[11:0] - 12'd1) begin
                            cmp_active <= 1'b0;
                        end else begin
                            cmp_pix <= cmp_pix + 12'd1;
                            if (cmp_ow == 6'd55) begin
                                cmp_ow <= 6'd0;
                                cmp_oh <= cmp_oh + 6'd1;
                            end else begin
                                cmp_ow <= cmp_ow + 6'd1;
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

    reg [11:0] em_pix;
    reg [11:0] pixels_done;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out   <= 1'b0;
            data_out    <= {BEAT_W{1'b0}};
            em_pix      <= 12'd0;
            pixels_done <= 12'd0;
            vector_done <= 1'b0;
        end else begin
            vector_done <= 1'b0;

            if (cmp_active && is_pass_end && (cmp_pass == OC_PASSES[7:0] - 8'd1)) begin
                pixels_done <= pixels_done + 12'd1;
            end

            // [INVARIANT:VALID_OUT_LATENCY]
            if (!vector_done && run && (em_pix < N_PIX[11:0]) && (em_pix < pixels_done)) begin
                valid_out <= 1'b1;
                data_out  <= out_buf[em_pix];
                if (em_pix == N_PIX[11:0] - 12'd1) begin
                    em_pix      <= 12'd0;
                    pixels_done <= 12'd0;
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
