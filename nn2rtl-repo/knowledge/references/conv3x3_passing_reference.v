// Foundry — spatial 3x3 conv2d, padding=1, stride=1
// Representative layer: layer1_0_conv2 (IC=64 OC=64 IH=IW=112 KH=KW=3 SH=SW=1 PH=PW=1 MP=4)
//
// Structural pattern: coord_scheduler + KH-row line buffer + registered shift-register
// window + serialized MP-lane MAC loop + OC-group iteration. Mirrors conv1x1_passing_reference.v
// for the datapath stages (BIAS → SCALE → OUTPUT, round-to-nearest, INT8 saturation).
// Spatial-specific pieces: the line buffer, the window-shift discipline, and the
// coord_scheduler handshake contract from 01_context.md.
//
// Adaptable localparams for other 3×3 s1 p1 layers:
//   IC, OC, IH, IW, OH, OW, MP, SCALE_MULT, SCALE_SHIFT, weights_path, bias_path.

module layer1_0_conv2 (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              valid_in,
    output wire              ready_in,
    input  wire [511:0]      data_in,
    output reg               valid_out,
    output reg  [511:0]      data_out
);
    // ------ Geometry (from LayerIR) ----------------------------------------
    localparam IC        = 64;
    localparam OC        = 64;
    localparam IH        = 112;
    localparam IW        = 112;
    localparam OH        = 112;
    localparam OW        = 112;
    localparam KH        = 3;
    localparam KW        = 3;
    localparam SH        = 1;
    localparam SW        = 1;
    localparam PH        = 1;
    localparam PW        = 1;
    localparam K_TOTAL   = IC * KH * KW;      // 576
    localparam MP        = 4;
    localparam OC_PASSES = (OC + MP - 1) / MP; // 16
    localparam LB_ROWS   = KH;                 // 3

    // ------ Scale factor (from LayerIR scale_factor) -----------------------
    localparam SCALE_MULT  = 8490;
    localparam SCALE_SHIFT = 21;

    // ------ Derived widths -------------------------------------------------
    localparam integer PROD_W        = 16;
    localparam integer ACC_W         = PROD_W + $clog2(K_TOTAL);
    localparam integer BIAS_W        = 32;
    localparam integer BIASED_W      = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1;
    localparam integer SCALE_MAG_W   = $clog2(SCALE_MULT + 1);
    localparam integer SCALE_CONST_W = SCALE_MAG_W + 1;
    localparam integer SCALED_W      = BIASED_W + SCALE_CONST_W;
    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST = SCALE_MULT;
    localparam signed [SCALED_W-1:0] SCALE_ROUND_BIAS =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);

    // ------ FSM states -----------------------------------------------------
    localparam ST_STREAM = 3'd0;
    localparam ST_MAC    = 3'd1;
    localparam ST_BIAS   = 3'd2;
    localparam ST_SCALE  = 3'd3;
    localparam ST_OUTPUT = 3'd4;

    // ------ Weight / bias arrays ------------------------------------------
    // [INVARIANT:WEIGHT_ARRAY]
    (* ram_style = "block" *) reg signed [7:0]  weights [0:OC*K_TOTAL-1];
    // [INVARIANT:WEIGHT_ARRAY]
    (* ram_style = "block" *) reg signed [31:0] biases  [0:OC-1];
    initial begin
        // [INVARIANT:WEIGHT_ARRAY]
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/layer1_0_conv2_weights.hex", weights);
        // [INVARIANT:WEIGHT_ARRAY]
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/layer1_0_conv2_bias.hex", biases);
    end

    // ------ Line buffer + registered window -------------------------------
    // line_buf holds the last LB_ROWS = KH input rows, rotating on new-row events.
    // window[kh][kw][ic] holds the receptive field for the output pixel currently
    // being firing-gated by the scheduler.
    reg signed [7:0] line_buf [0:LB_ROWS-1][0:IW-1][0:IC-1];
    reg signed [7:0] window   [0:KH-1][0:KW-1][0:IC-1];

    // ------ MAC / pipeline registers --------------------------------------
    reg signed [ACC_W-1:0]    acc    [0:MP-1];
    reg signed [BIASED_W-1:0] biased [0:MP-1];
    reg signed [SCALED_W-1:0] scaled [0:MP-1];
    reg signed [SCALED_W-1:0] v_tmp;

    reg [$clog2(K_TOTAL+1)-1:0]   k_counter;
    reg [$clog2(MP+1)-1:0]        lane_counter;
    reg [$clog2(OC_PASSES+1)-1:0] oc_group;
    reg [2:0]                     state;
    reg                           started;
    reg                           start_pulse;

    // cur_row: slot in line_buf where the CURRENT input row writes.
    // Rotates forward on every in_col wrap. KH=3, LB_ROWS=3.
    reg [$clog2(LB_ROWS)-1:0] cur_row;

    integer i, j, c_ch, lane_i, global_oc_base;

    // ------ coord_scheduler instantiation ---------------------------------
    wire                               sched_needs_real_input;
    wire                               sched_ready_in;
    wire                               sched_output_fires;
    wire                               sched_out_frame_done;
    wire [$clog2(IH + PH + 1)-1:0]     sched_in_row;
    wire [$clog2(IW + PW + 1)-1:0]     sched_in_col;
    wire [$clog2(OH * OW + 1)-1:0]     sched_outputs_emitted;

    wire mac_busy_w = (state == ST_MAC) || (state == ST_BIAS) ||
                      (state == ST_SCALE) || (state == ST_OUTPUT);
    // stall_in is combinational per the 01_context.md contract: freezes the
    // scheduler on any firing coord (output_fires) and across the MAC pipeline.
    wire stall_in = sched_output_fires || mac_busy_w;

    coord_scheduler #(
        .IH(IH), .IW(IW), .OH(OH), .OW(OW),
        .KH(KH), .KW(KW), .SH(SH), .SW(SW),
        .PH(PH), .PW(PW)
    ) scheduler (
        .clk(clk), .rst_n(rst_n),
        .start(start_pulse),
        .stall_in(stall_in),
        .valid_in(valid_in),
        .ready_in(sched_ready_in),
        .needs_real_input(sched_needs_real_input),
        .in_row(sched_in_row), .in_col(sched_in_col),
        .output_fires(sched_output_fires),
        .in_frame_done(),
        .out_frame_done(sched_out_frame_done),
        .outputs_emitted(sched_outputs_emitted)
    );

    // Only accept upstream when the scheduler does AND we're in ST_STREAM.
    // [INVARIANT:READY_IN_GATING]
    assign ready_in = sched_ready_in && (state == ST_STREAM);

    // Padding-region classifiers (combinational on scheduler outputs).
    wire bottom_padded = (sched_in_row >= IH);
    wire right_padded  = (sched_in_col >= IW);

    // win_row[i] is the line_buf slot that holds receptive-field row i of the
    // CURRENT scheduler position. With cur_row tracking the newest written row:
    //   win_row[0] = oldest of the 3-row RF = (cur_row + 1) mod 3
    //   win_row[1] = middle               = (cur_row + 2) mod 3
    //   win_row[2] = newest (current row) = cur_row
    // LB_ROWS = KH = 3 ⇒ tiny case split instead of a modulo divider.
    wire [$clog2(LB_ROWS)-1:0] win_row_0 = (cur_row == 2'd0) ? 2'd1 :
                                           (cur_row == 2'd1) ? 2'd2 : 2'd0;
    wire [$clog2(LB_ROWS)-1:0] win_row_1 = (cur_row == 2'd0) ? 2'd2 :
                                           (cur_row == 2'd1) ? 2'd0 : 2'd1;
    wire [$clog2(LB_ROWS)-1:0] win_row_2 = cur_row;

    // Does the scheduler advance this cycle? (Same predicate as inside the
    // scheduler module; used to gate window shift + line_buf write.)
    wire handshake_real = sched_needs_real_input && valid_in && sched_ready_in;
    wire pad_step       = !sched_needs_real_input && !stall_in;
    wire advance        = handshake_real || pad_step;

    // ------ FSM / datapath ------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= ST_STREAM;
            valid_out    <= 1'b0;
            data_out     <= {512{1'b0}};
            cur_row      <= 0;
            k_counter    <= 0;
            lane_counter <= 0;
            oc_group     <= 0;
            started      <= 1'b0;
            start_pulse  <= 1'b0;
            v_tmp        <= 0;
            for (i = 0; i < MP; i = i + 1) begin
                acc[i]    <= 0;
                biased[i] <= 0;
                scaled[i] <= 0;
            end
            for (i = 0; i < KH; i = i + 1)
                for (j = 0; j < KW; j = j + 1)
                    for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                        window[i][j][c_ch] <= 8'sd0;
            for (i = 0; i < LB_ROWS; i = i + 1)
                for (j = 0; j < IW; j = j + 1)
                    for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                        line_buf[i][j][c_ch] <= 8'sd0;
        end else begin
            // Defaults each cycle.
            valid_out   <= 1'b0;
            start_pulse <= 1'b0;

            // Arm the scheduler on the first valid_in after reset; re-arm on
            // out_frame_done so back-to-back input frames work.
            if (!started && valid_in) begin
                started     <= 1'b1;
                start_pulse <= 1'b1;
            end else if (sched_out_frame_done) begin
                started <= 1'b0;
            end

            case (state)
                // ----------------------------------------------------------
                // ST_STREAM — accept pixels via scheduler handshake, shift
                // the window, write line_buf. On sched_output_fires hand off
                // to ST_MAC with the window frozen for all OC_PASSES.
                // ----------------------------------------------------------
                ST_STREAM: begin
                    if (advance) begin
                        // Shift window columns left (all rows at once).
                        for (i = 0; i < KH; i = i + 1)
                            for (j = 0; j < KW-1; j = j + 1)
                                for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                                    window[i][j][c_ch] <= window[i][j+1][c_ch];

                        // Load the new rightmost column.
                        //   Upper rows (win_row_0, win_row_1) from line_buf,
                        //   unless right-padded (load zero).
                        //   Bottom row: bypass line_buf straight from data_in
                        //   on REAL handshake; zero when padded (right or bottom).
                        for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1) begin
                            if (right_padded) begin
                                window[0][KW-1][c_ch] <= 8'sd0;
                                window[1][KW-1][c_ch] <= 8'sd0;
                                window[2][KW-1][c_ch] <= 8'sd0;
                            end else begin
                                window[0][KW-1][c_ch] <= line_buf[win_row_0][sched_in_col][c_ch];
                                window[1][KW-1][c_ch] <= line_buf[win_row_1][sched_in_col][c_ch];
                                if (bottom_padded) begin
                                    window[2][KW-1][c_ch] <= 8'sd0;
                                end else if (handshake_real) begin
                                    window[2][KW-1][c_ch] <= $signed(data_in[c_ch*8 +: 8]);
                                end
                            end
                        end

                        // Line buffer write: only on REAL-region handshake.
                        if (handshake_real) begin
                            for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                                line_buf[cur_row][sched_in_col][c_ch] <=
                                    $signed(data_in[c_ch*8 +: 8]);
                        end

                        // cur_row rotates forward on in_col wrap (end of a row).
                        // Only on REAL rows (bottom-padded rows keep cur_row
                        // advancing too so the zero writes cascade cleanly).
                        if (sched_in_col == (IW + PW - 1)) begin
                            cur_row <= (cur_row == LB_ROWS - 1) ?
                                       {$clog2(LB_ROWS){1'b0}} :
                                       cur_row + {{($clog2(LB_ROWS)-1){1'b0}}, 1'b1};
                        end
                    end

                    // Output dispatch: take the firing pixel into the MAC
                    // pipeline with the window frozen.
                    if (sched_output_fires) begin
                        state        <= ST_MAC;
                        k_counter    <= 0;
                        lane_counter <= 0;
                        oc_group     <= 0;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                            acc[lane_i] <= 0;
                    end
                end

                // ----------------------------------------------------------
                // ST_MAC — serialized lane rotation over K_TOTAL kernel taps.
                // Exactly one weight read / one multiply / one accumulate per
                // cycle (serialized weight reads, see 01_context.md). Loops
                // MP*K_TOTAL cycles, then advances to ST_BIAS.
                // ----------------------------------------------------------
                ST_MAC: begin : MAC_BLOCK
                    // integer globals to decompose the flat kernel index
                    integer kh_idx, kw_idx, ic_idx, global_oc;
                    global_oc      = oc_group * MP + lane_counter;
                    kh_idx         = (k_counter % (KH * KW)) / KW;
                    kw_idx         = k_counter % KW;
                    ic_idx         = k_counter / (KH * KW);
                    acc[lane_counter] <= acc[lane_counter] +
                        $signed(weights[global_oc * K_TOTAL + k_counter]) *
                        $signed(window[kh_idx][kw_idx][ic_idx]);

                    if (lane_counter == MP - 1) begin
                        lane_counter <= 0;
                        if (k_counter == K_TOTAL - 1) begin
                            k_counter <= 0;
                            state     <= ST_BIAS;
                        end else begin
                            k_counter <= k_counter + 1;
                        end
                    end else begin
                        lane_counter <= lane_counter + 1;
                    end
                end

                // ----------------------------------------------------------
                // ST_BIAS — add per-channel bias to each of the MP accumulators.
                // Direct signed add; both operands are `reg signed`, so the
                // destination's wider context sign-extends correctly (no
                // concatenation-based extension).
                // ----------------------------------------------------------
                ST_BIAS: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        global_oc_base = oc_group * MP + lane_i;
                        biased[lane_i] <= $signed(acc[lane_i]) + $signed(biases[global_oc_base]);
                    end
                    state <= ST_SCALE;
                end

                // ----------------------------------------------------------
                // ST_SCALE — multiply by SCALE_MULT_CONST. Full-width signed
                // multiply; SCALED_W = BIASED_W + SCALE_CONST_W absorbs the
                // growth.
                // ----------------------------------------------------------
                ST_SCALE: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                        scaled[lane_i] <= $signed(biased[lane_i]) * $signed(SCALE_MULT_CONST);
                    state <= ST_OUTPUT;
                end

                // ----------------------------------------------------------
                // ST_OUTPUT — round-to-nearest shift, INT8 saturation, pack
                // into data_out[global_oc*8 +: 8]. On the last OC group fire
                // valid_out; otherwise advance oc_group and go back to ST_MAC
                // with the same window contents (window-freeze rule).
                // ----------------------------------------------------------
                ST_OUTPUT: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        global_oc_base = oc_group * MP + lane_i;
                        // [INVARIANT:ROUNDING]
                        v_tmp = (scaled[lane_i] + SCALE_ROUND_BIAS) >>> SCALE_SHIFT;
                        data_out[global_oc_base*8 +: 8] <=
                            (v_tmp > 127)  ?  8'sd127 :
                            (v_tmp < -128) ? -8'sd128 : v_tmp[7:0];
                    end

                    if (oc_group == OC_PASSES - 1) begin
                        // [INVARIANT:VALID_OUT_LATENCY]
                        valid_out <= 1'b1;
                        state     <= ST_STREAM;
                    end else begin
                        oc_group     <= oc_group + 1;
                        k_counter    <= 0;
                        lane_counter <= 0;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                            acc[lane_i] <= 0;
                        state <= ST_MAC;
                    end
                end

                default: state <= ST_STREAM;
            endcase
        end
    end

endmodule
