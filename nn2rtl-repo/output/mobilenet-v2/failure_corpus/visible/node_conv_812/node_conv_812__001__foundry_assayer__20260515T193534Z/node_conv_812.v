// node_conv_812 -- depthwise conv 3x3 stride 1 pad 1, C=32, IH=IW=112.
// Contract: depthwise-conv, packed 256-bit activation bus (32 channels x 8 bits per beat).

`timescale 1ns/1ps
`default_nettype none

module node_conv_812 #(
    parameter WEIGHTS_PATH = "output/mobilenet-v2/weights/node_conv_812_weights.hex",
    parameter BIAS_PATH    = "output/mobilenet-v2/weights/node_conv_812_bias.hex"
)(
    input  wire          clk,
    input  wire          rst_n,
    input  wire          valid_in,
    output wire          ready_in,
    input  wire [255:0]  data_in,
    output reg           valid_out,
    output reg  [255:0]  data_out
);

    // ---- Geometry ----
    localparam integer C        = 32;
    localparam integer IH       = 112;
    localparam integer IW       = 112;
    localparam integer OH       = 112;
    localparam integer OW       = 112;
    localparam integer KH       = 3;
    localparam integer KW       = 3;
    localparam integer PH       = 1;
    localparam integer PW       = 1;
    localparam integer K_TOTAL  = KH * KW;          // 9
    localparam integer NUM_PIX  = OH * OW;          // 12544

    // ---- Schedule ----
    localparam integer MP          = 4;
    localparam integer OC_PASSES   = 8;             // C / MP
    localparam integer PASS_CYCLES = 42;            // 4*9 MAC issues + 5 idle + 1 writeback
    localparam integer COMP_THRESHOLD = (IW + PW) + (KW - PW) - 1; // 114

    // ---- Quantization ----
    localparam integer SCALE_SHIFT = 22;
    localparam signed [63:0] SCALE_MULT = 64'sd20518;
    localparam signed [63:0] HALF       = 64'sd1 <<< (SCALE_SHIFT - 1);

    // ---- Weight / Bias ROMs ----
    (* rom_style = "block" *)
    reg signed [7:0]  weight_rom [0:C*K_TOTAL-1];
    (* rom_style = "block" *)
    reg signed [31:0] bias_rom   [0:C-1];

    initial begin
        $readmemh(WEIGHTS_PATH, weight_rom);
        $readmemh(BIAS_PATH,    bias_rom);
    end

    // ---- 3-row line buffer (rotating) ----
    (* ram_style = "block" *) reg [255:0] row_buf0 [0:IW-1];
    (* ram_style = "block" *) reg [255:0] row_buf1 [0:IW-1];
    (* ram_style = "block" *) reg [255:0] row_buf2 [0:IW-1];

    // ---- Input streaming state ----
    reg  [7:0]  in_row;            // 0..IH (IH means input_done)
    reg  [7:0]  in_col;            // 0..IW-1
    reg  [1:0]  in_slot;           // which row_buf to fill next
    reg  [14:0] pixels_written;
    reg         input_done;

    // ---- Compute state ----
    reg  [7:0]  oh, ow;
    reg  [1:0]  slot_minus_1, slot_zero, slot_plus_1;
    reg  [3:0]  cmp_pass;          // 0..OC_PASSES-1
    reg  [5:0]  cmp_step;          // 0..PASS_CYCLES-1
    reg         cmp_active;
    reg  [1:0]  mac_lane;          // 0..MP-1
    reg  [3:0]  mac_k;              // 0..K_TOTAL-1

    // ---- Accumulators ----
    reg signed [31:0] acc0, acc1, acc2, acc3;

    // ---- Output staging ----
    reg signed [7:0] out_bank [0:C-1];
    reg              out_pixel_ready;
    reg [14:0]       out_pix_count;
    reg              vector_done;

    // ---- read-mux temporaries (module scope) ----
    reg [1:0]  read_slot;
    reg [255:0] read_word;
    integer i;

    // ---- ready handshake (3-row rotating buffer rule) ----
    wire stall_input = (in_row > (oh + 8'd1));
    assign ready_in = !input_done && !stall_input && !vector_done;

    // ---- Start of compute ----
    wire have_enough_pixels = (pixels_written >= COMP_THRESHOLD[14:0]);
    wire start_compute = !cmp_active
                       && !out_pixel_ready
                       && have_enough_pixels
                       && !vector_done
                       && (out_pix_count < NUM_PIX[14:0]);

    // ---- MAC indexing ----
    wire [4:0] cur_ch = {cmp_pass[2:0], 2'b00} + {3'd0, mac_lane};
    wire [3:0] cur_k  = mac_k;

    wire [1:0] kh_idx = (cur_k < 4'd3) ? 2'd0 : (cur_k < 4'd6) ? 2'd1 : 2'd2;
    wire [1:0] kw_idx =
          (cur_k == 4'd0 || cur_k == 4'd3 || cur_k == 4'd6) ? 2'd0
        : (cur_k == 4'd1 || cur_k == 4'd4 || cur_k == 4'd7) ? 2'd1
        :                                                     2'd2;

    wire signed [9:0] in_r_s = $signed({2'b00, oh}) + $signed({8'd0, kh_idx}) - $signed(10'sd1);
    wire signed [9:0] in_c_s = $signed({2'b00, ow}) + $signed({8'd0, kw_idx}) - $signed(10'sd1);
    wire              in_bounds = (in_r_s >= 0) && (in_r_s < IH) && (in_c_s >= 0) && (in_c_s < IW);

    always @(*) begin
        case (kh_idx)
            2'd0:    read_slot = slot_minus_1;
            2'd1:    read_slot = slot_zero;
            2'd2:    read_slot = slot_plus_1;
            default: read_slot = 2'd0;
        endcase
    end

    wire [7:0]   read_col   = in_c_s[7:0];
    wire [255:0] read_word_0 = row_buf0[read_col];
    wire [255:0] read_word_1 = row_buf1[read_col];
    wire [255:0] read_word_2 = row_buf2[read_col];

    always @(*) begin
        case (read_slot)
            2'd0:    read_word = read_word_0;
            2'd1:    read_word = read_word_1;
            2'd2:    read_word = read_word_2;
            default: read_word = 256'd0;
        endcase
    end

    wire [7:0]        act_bits   = read_word[{cur_ch, 3'b000} +: 8];
    wire signed [7:0] act_byte   = in_bounds ? $signed(act_bits) : 8'sd0;

    wire [8:0]        w_addr     = cur_ch * 5'd9 + cur_k;
    wire signed [7:0] w_byte     = weight_rom[w_addr];

    wire signed [15:0] mac_prod     = act_byte * w_byte;
    wire signed [31:0] mac_prod_ext = mac_prod;

    wire mac_issue       = cmp_active && (cmp_step < 6'd36);
    wire mac_issue_lane0 = mac_issue && (mac_lane == 2'd0);
    wire mac_issue_lane1 = mac_issue && (mac_lane == 2'd1);
    wire mac_issue_lane2 = mac_issue && (mac_lane == 2'd2);
    wire mac_issue_lane3 = mac_issue && (mac_lane == 2'd3);

    // ---- Writeback datapath (combinational) ----
    wire [4:0] wb_ch0 = {cmp_pass[2:0], 2'b00} + 5'd0;
    wire [4:0] wb_ch1 = {cmp_pass[2:0], 2'b00} + 5'd1;
    wire [4:0] wb_ch2 = {cmp_pass[2:0], 2'b00} + 5'd2;
    wire [4:0] wb_ch3 = {cmp_pass[2:0], 2'b00} + 5'd3;

    wire signed [31:0] bias0 = bias_rom[wb_ch0];
    wire signed [31:0] bias1 = bias_rom[wb_ch1];
    wire signed [31:0] bias2 = bias_rom[wb_ch2];
    wire signed [31:0] bias3 = bias_rom[wb_ch3];

    wire signed [33:0] sum0 = $signed({{2{acc0[31]}}, acc0}) + $signed({{2{bias0[31]}}, bias0});
    wire signed [33:0] sum1 = $signed({{2{acc1[31]}}, acc1}) + $signed({{2{bias1[31]}}, bias1});
    wire signed [33:0] sum2 = $signed({{2{acc2[31]}}, acc2}) + $signed({{2{bias2[31]}}, bias2});
    wire signed [33:0] sum3 = $signed({{2{acc3[31]}}, acc3}) + $signed({{2{bias3[31]}}, bias3});

    wire signed [63:0] scaled0 = sum0 * SCALE_MULT;
    wire signed [63:0] scaled1 = sum1 * SCALE_MULT;
    wire signed [63:0] scaled2 = sum2 * SCALE_MULT;
    wire signed [63:0] scaled3 = sum3 * SCALE_MULT;

    wire signed [63:0] rounded0 = scaled0 + (scaled0[63] ? (HALF - 64'sd1) : HALF);
    wire signed [63:0] rounded1 = scaled1 + (scaled1[63] ? (HALF - 64'sd1) : HALF);
    wire signed [63:0] rounded2 = scaled2 + (scaled2[63] ? (HALF - 64'sd1) : HALF);
    wire signed [63:0] rounded3 = scaled3 + (scaled3[63] ? (HALF - 64'sd1) : HALF);

    wire signed [63:0] shifted0 = rounded0 >>> SCALE_SHIFT;
    wire signed [63:0] shifted1 = rounded1 >>> SCALE_SHIFT;
    wire signed [63:0] shifted2 = rounded2 >>> SCALE_SHIFT;
    wire signed [63:0] shifted3 = rounded3 >>> SCALE_SHIFT;

    function signed [7:0] sat8;
        input signed [63:0] x;
        begin
            if      (x >  64'sd127) sat8 =  8'sd127;
            else if (x < -64'sd128) sat8 = -8'sd128;
            else                    sat8 = x[7:0];
        end
    endfunction

    wire signed [7:0] wb_byte0 = sat8(shifted0);
    wire signed [7:0] wb_byte1 = sat8(shifted1);
    wire signed [7:0] wb_byte2 = sat8(shifted2);
    wire signed [7:0] wb_byte3 = sat8(shifted3);

    // ---- INPUT STREAMING ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_row          <= 8'd0;
            in_col          <= 8'd0;
            in_slot         <= 2'd0;
            pixels_written  <= 15'd0;
            input_done      <= 1'b0;
        end else if (vector_done) begin
            in_row          <= 8'd0;
            in_col          <= 8'd0;
            in_slot         <= 2'd0;
            pixels_written  <= 15'd0;
            input_done      <= 1'b0;
        end else if (valid_in && ready_in) begin
            pixels_written  <= pixels_written + 15'd1;
            if (in_col == IW[7:0] - 8'd1) begin
                in_col  <= 8'd0;
                if (in_row == IH[7:0] - 8'd1) begin
                    input_done <= 1'b1;
                    in_row     <= IH[7:0];
                end else begin
                    in_row  <= in_row + 8'd1;
                    in_slot <= (in_slot == 2'd2) ? 2'd0 : in_slot + 2'd1;
                end
            end else begin
                in_col <= in_col + 8'd1;
            end
        end
    end

    // ---- Row-buffer write (separate to keep BRAM inference clean) ----
    always @(posedge clk) begin
        if (valid_in && ready_in && !input_done && !vector_done) begin
            case (in_slot)
                2'd0: row_buf0[in_col] <= data_in;
                2'd1: row_buf1[in_col] <= data_in;
                2'd2: row_buf2[in_col] <= data_in;
            endcase
        end
    end

    // ---- COMPUTE / OUTPUT FSM ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cmp_active      <= 1'b0;
            cmp_step        <= 6'd0;
            cmp_pass        <= 4'd0;
            mac_lane        <= 2'd0;
            mac_k           <= 4'd0;
            oh              <= 8'd0;
            ow              <= 8'd0;
            slot_minus_1    <= 2'd2;
            slot_zero       <= 2'd0;
            slot_plus_1     <= 2'd1;
            acc0            <= 32'sd0;
            acc1            <= 32'sd0;
            acc2            <= 32'sd0;
            acc3            <= 32'sd0;
            out_pixel_ready <= 1'b0;
            out_pix_count   <= 15'd0;
            vector_done     <= 1'b0;
            for (i = 0; i < C; i = i + 1) out_bank[i] <= 8'sd0;
        end else begin
            vector_done <= 1'b0;

            if (vector_done) begin
                cmp_active      <= 1'b0;
                cmp_step        <= 6'd0;
                cmp_pass        <= 4'd0;
                mac_lane        <= 2'd0;
                mac_k           <= 4'd0;
                oh              <= 8'd0;
                ow              <= 8'd0;
                slot_minus_1    <= 2'd2;
                slot_zero       <= 2'd0;
                slot_plus_1     <= 2'd1;
                acc0            <= 32'sd0;
                acc1            <= 32'sd0;
                acc2            <= 32'sd0;
                acc3            <= 32'sd0;
                out_pixel_ready <= 1'b0;
                out_pix_count   <= 15'd0;
            end else begin
                // Accumulator update during MAC issue cycles
                if (mac_issue_lane0) acc0 <= acc0 + mac_prod_ext;
                if (mac_issue_lane1) acc1 <= acc1 + mac_prod_ext;
                if (mac_issue_lane2) acc2 <= acc2 + mac_prod_ext;
                if (mac_issue_lane3) acc3 <= acc3 + mac_prod_ext;

                if (cmp_active) begin
                    if (cmp_step < 6'd35) begin
                        cmp_step <= cmp_step + 6'd1;
                        if (mac_k == 4'd8) begin
                            mac_k    <= 4'd0;
                            mac_lane <= mac_lane + 2'd1;
                        end else begin
                            mac_k    <= mac_k + 4'd1;
                        end
                    end else if (cmp_step == 6'd35) begin
                        // Last MAC of this pass just issued (lane=3, k=8)
                        cmp_step <= 6'd36;
                        mac_lane <= 2'd0;
                        mac_k    <= 4'd0;
                    end else if (cmp_step < 6'd41) begin
                        cmp_step <= cmp_step + 6'd1;
                    end else begin
                        // cmp_step == 41 -- writeback
                        out_bank[wb_ch0] <= wb_byte0;
                        out_bank[wb_ch1] <= wb_byte1;
                        out_bank[wb_ch2] <= wb_byte2;
                        out_bank[wb_ch3] <= wb_byte3;

                        if (cmp_pass == OC_PASSES[3:0] - 4'd1) begin
                            // Pixel complete
                            cmp_active      <= 1'b0;
                            cmp_step        <= 6'd0;
                            cmp_pass        <= 4'd0;
                            mac_lane        <= 2'd0;
                            mac_k           <= 4'd0;
                            out_pixel_ready <= 1'b1;
                            acc0            <= 32'sd0;
                            acc1            <= 32'sd0;
                            acc2            <= 32'sd0;
                            acc3            <= 32'sd0;

                            // Advance (oh, ow)
                            if (ow == OW[7:0] - 8'd1) begin
                                ow <= 8'd0;
                                if (oh == OH[7:0] - 8'd1) begin
                                    oh <= 8'd0;
                                end else begin
                                    oh           <= oh + 8'd1;
                                    slot_minus_1 <= slot_zero;
                                    slot_zero    <= slot_plus_1;
                                    slot_plus_1  <= slot_minus_1;
                                end
                            end else begin
                                ow <= ow + 8'd1;
                            end
                        end else begin
                            // Next pass of same pixel
                            cmp_pass <= cmp_pass + 4'd1;
                            cmp_step <= 6'd0;
                            mac_lane <= 2'd0;
                            mac_k    <= 4'd0;
                            acc0     <= 32'sd0;
                            acc1     <= 32'sd0;
                            acc2     <= 32'sd0;
                            acc3     <= 32'sd0;
                        end
                    end
                end else if (start_compute) begin
                    cmp_active <= 1'b1;
                    cmp_step   <= 6'd0;
                    cmp_pass   <= 4'd0;
                    mac_lane   <= 2'd0;
                    mac_k      <= 4'd0;
                    acc0       <= 32'sd0;
                    acc1       <= 32'sd0;
                    acc2       <= 32'sd0;
                    acc3       <= 32'sd0;
                end

                // Output handshake: out_pixel_ready is consumed exactly one cycle after writeback
                if (out_pixel_ready) begin
                    out_pixel_ready <= 1'b0;
                    out_pix_count   <= out_pix_count + 15'd1;
                    if (out_pix_count == NUM_PIX[14:0] - 15'd1) begin
                        vector_done <= 1'b1;
                    end
                end
            end
        end
    end

    // ---- Output register stage ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            data_out  <= 256'd0;
        end else if (out_pixel_ready) begin
            valid_out <= 1'b1;
            data_out  <= {
                out_bank[31], out_bank[30], out_bank[29], out_bank[28],
                out_bank[27], out_bank[26], out_bank[25], out_bank[24],
                out_bank[23], out_bank[22], out_bank[21], out_bank[20],
                out_bank[19], out_bank[18], out_bank[17], out_bank[16],
                out_bank[15], out_bank[14], out_bank[13], out_bank[12],
                out_bank[11], out_bank[10], out_bank[9],  out_bank[8],
                out_bank[7],  out_bank[6],  out_bank[5],  out_bank[4],
                out_bank[3],  out_bank[2],  out_bank[1],  out_bank[0]
            };
        end else begin
            valid_out <= 1'b0;
            data_out  <= 256'd0;
        end
    end

endmodule

`default_nettype wire
