`timescale 1ns/1ps

module node_conv_292 (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             valid_in,
    output reg              ready_in,
    input  wire [255:0]     data_in,
    output reg              valid_out,
    output reg  [255:0]     data_out,
    output reg              weights_arvalid,
    input  wire             weights_arready,
    output reg  [31:0]      weights_araddr,
    output reg  [7:0]       weights_arlen,
    input  wire             weights_rvalid,
    output reg              weights_rready,
    input  wire [63:0]      weights_rdata,
    input  wire             weights_rlast
);

    localparam IC              = 512;
    localparam OC              = 512;
    localparam KH              = 3;
    localparam KW              = 3;
    localparam KH_KW           = KH*KW;
    localparam K_TOTAL         = IC*KH*KW;
    localparam IH              = 7;
    localparam IW              = 7;
    localparam OH              = 7;
    localparam OW              = 7;
    localparam ACTIVE_PIXELS   = OH*OW;
    localparam PH              = 1;
    localparam PW              = 1;
    localparam MP              = 4;
    localparam OC_PASSES       = OC/MP;
    localparam CHANNEL_TILE    = 32;
    localparam IN_BEATS        = IC/CHANNEL_TILE;
    localparam OUT_BEATS       = OC/CHANNEL_TILE;
    localparam BEAT_BITS       = 256;
    localparam BYTES_PER_PASS  = MP*K_TOTAL;
    localparam BEATS_PER_PASS  = BYTES_PER_PASS/8;
    localparam BURSTS_PER_PASS = 9;
    localparam TOTAL_IN_PIXELS = IH*IW;
    localparam LBUF_DEPTH      = IN_BEATS*TOTAL_IN_PIXELS;
    localparam FILL_DELAY      = 11;

    localparam SCALE_MULT  = 24577;
    localparam SCALE_SHIFT = 22;

    localparam integer PROD_W        = 16;
    localparam integer ACC_W         = PROD_W + $clog2(K_TOTAL);
    localparam integer BIAS_W        = 32;
    localparam integer BIASED_W      = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1;
    localparam integer SCALE_MAG_W   = $clog2(SCALE_MULT + 1);
    localparam integer SCALE_CONST_W = SCALE_MAG_W + 1;
    localparam integer SCALED_W      = BIASED_W + SCALE_CONST_W;
    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST = SCALE_MULT;
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);

    parameter BIAS_PATH =
        "D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_292_bias.hex";

    (* rom_style = "block", ram_style = "block" *)
    reg signed [31:0] bias_rom [0:OC-1];
    initial begin
        $readmemh(BIAS_PATH, bias_rom);
    end

    (* ram_style = "block" *) reg [63:0] cache_a [0:BEATS_PER_PASS-1];
    (* ram_style = "block" *) reg [63:0] cache_b [0:BEATS_PER_PASS-1];
    reg cache_a_loaded;
    reg cache_b_loaded;

    (* ram_style = "block" *)
    reg [BEAT_BITS-1:0] line_buf [0:LBUF_DEPTH-1];
    reg signed [7:0]    window   [0:KH-1][0:KW-1][0:IC-1];

    reg [OC*8-1:0] out_buffer;
    reg signed [ACC_W-1:0]    acc    [0:MP-1];
    reg signed [SCALED_W-1:0] scaled [0:MP-1];
    reg signed [SCALED_W-1:0] v_tmp;

    localparam AR_IDLE      = 3'd0;
    localparam AR_ISSUE     = 3'd1;
    localparam AR_WAIT_DATA = 3'd2;
    localparam AR_DONE      = 3'd3;

    reg [2:0]  ar_state;
    reg [11:0] ar_beat_counter;
    reg [3:0]  ar_burst_index;
    reg [7:0]  ar_pass_target;
    reg        ar_target_cache;
    reg        ar_kick_pending;

    reg        cache_we;
    reg        cache_we_target;
    reg [11:0] cache_we_addr;
    reg [63:0] cache_we_data;

    reg ar_kick;
    reg ar_restart;

    always @(posedge clk) begin
        if (cache_we) begin
            if (cache_we_target == 1'b0)
                cache_a[cache_we_addr] <= cache_we_data;
            else
                cache_b[cache_we_addr] <= cache_we_data;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ar_state         <= AR_IDLE;
            ar_beat_counter  <= 12'd0;
            ar_burst_index   <= 4'd0;
            ar_pass_target   <= 8'd0;
            ar_target_cache  <= 1'b0;
            weights_arvalid  <= 1'b0;
            weights_araddr   <= 32'd0;
            weights_arlen    <= 8'd0;
            weights_rready   <= 1'b0;
            cache_we         <= 1'b0;
            cache_we_target  <= 1'b0;
            cache_we_addr    <= 12'd0;
            cache_we_data    <= 64'd0;
            ar_kick_pending  <= 1'b0;
            cache_a_loaded   <= 1'b0;
            cache_b_loaded   <= 1'b0;
        end else if (ar_restart) begin
            ar_state         <= AR_IDLE;
            ar_beat_counter  <= 12'd0;
            ar_burst_index   <= 4'd0;
            ar_pass_target   <= 8'd0;
            ar_target_cache  <= 1'b0;
            weights_arvalid  <= 1'b0;
            weights_rready   <= 1'b0;
            cache_we         <= 1'b0;
            ar_kick_pending  <= ar_kick;
            cache_a_loaded   <= 1'b0;
            cache_b_loaded   <= 1'b0;
        end else begin
            cache_we <= 1'b0;
            if (ar_kick) ar_kick_pending <= 1'b1;
            case (ar_state)
                AR_IDLE: begin
                    weights_arvalid <= 1'b0;
                    weights_rready  <= 1'b0;
                    if (ar_kick_pending && ar_pass_target < OC_PASSES) begin
                        ar_burst_index  <= 4'd0;
                        ar_beat_counter <= 12'd0;
                        ar_state        <= AR_ISSUE;
                        ar_kick_pending <= 1'b0;
                        if (ar_target_cache == 1'b0)
                            cache_a_loaded <= 1'b0;
                        else
                            cache_b_loaded <= 1'b0;
                    end
                end
                AR_ISSUE: begin
                    weights_arvalid <= 1'b1;
                    weights_arlen   <= 8'd255;
                    weights_araddr  <= (ar_pass_target * BYTES_PER_PASS) +
                                       (ar_burst_index * 32'd2048);
                    if (weights_arready && weights_arvalid) begin
                        weights_arvalid <= 1'b0;
                        weights_rready  <= 1'b1;
                        ar_state        <= AR_WAIT_DATA;
                    end
                end
                AR_WAIT_DATA: begin
                    weights_rready <= 1'b1;
                    if (weights_rvalid) begin
                        cache_we        <= 1'b1;
                        cache_we_target <= ar_target_cache;
                        cache_we_addr   <= (ar_burst_index * 12'd256) + ar_beat_counter;
                        cache_we_data   <= weights_rdata;
                        ar_beat_counter <= ar_beat_counter + 12'd1;
                        if (weights_rlast) begin
                            weights_rready  <= 1'b0;
                            ar_beat_counter <= 12'd0;
                            if (ar_burst_index + 4'd1 < BURSTS_PER_PASS) begin
                                ar_burst_index <= ar_burst_index + 4'd1;
                                ar_state       <= AR_ISSUE;
                            end else begin
                                ar_state <= AR_DONE;
                            end
                        end
                    end
                end
                AR_DONE: begin
                    if (ar_target_cache == 1'b0)
                        cache_a_loaded <= 1'b1;
                    else
                        cache_b_loaded <= 1'b1;
                    ar_pass_target  <= ar_pass_target + 8'd1;
                    ar_target_cache <= ~ar_target_cache;
                    ar_state        <= AR_IDLE;
                end
                default: ar_state <= AR_IDLE;
            endcase
        end
    end

    localparam ST_INIT_BOOT  = 4'd0;
    localparam ST_BOOT_WAIT  = 4'd1;
    localparam ST_INPUT      = 4'd2;
    localparam ST_RUNNING    = 4'd4;
    localparam ST_BIAS_SCALE = 4'd5;
    localparam ST_PACK       = 4'd6;
    localparam ST_STREAM_OUT = 4'd7;
    localparam ST_NEXT_OUT   = 4'd8;
    localparam ST_PASS_WAIT  = 4'd9;

    reg [3:0]  state;
    reg [1:0]  kh_counter;
    reg [1:0]  kw_counter;
    reg [9:0]  ic_counter;
    reg [1:0]  lane_counter;
    reg [13:0] k_counter;
    reg [7:0]  oc_pass;
    reg        active_cache_sel;
    reg [4:0]  in_beat_index;
    reg [5:0]  out_beat;
    reg [5:0]  in_pixel_counter;
    reg [6:0]  active_pixel_counter;
    reg [3:0]  fill_counter;
    reg [3:0]  out_row;
    reg [3:0]  out_col;

    wire active_cache_loaded = active_cache_sel ? cache_b_loaded : cache_a_loaded;

    reg                     mac_valid_q1;
    reg [1:0]               mac_lane_q1;
    reg                     mac_in_valid_q1;
    reg [5:0]               mac_in_pix_q1;
    reg [9:0]               mac_in_ch_q1;
    reg [14:0]              mac_weight_addr_q1;
    reg [1:0]               mac_kh_q1;
    reg [1:0]               mac_kw_q1;
    reg [9:0]               mac_ic_q1;
    reg                     mac_done_issuing;

    reg                     mac_valid_q2;
    reg [1:0]               mac_lane_q2;
    reg                     mac_in_valid_q2;
    reg [2:0]               mac_byte_in_word_q2;
    reg [1:0]               mac_kh_q2;
    reg [1:0]               mac_kw_q2;
    reg [9:0]               mac_ic_q2;

    reg                     mac_valid_q3;
    reg [1:0]               mac_lane_q3;

    reg [BEAT_BITS-1:0]     line_buf_word_q2;
    reg [4:0]               byte_in_word_q2;
    reg [63:0]              cache_word_q2;

    wire signed [7:0] weight_q2 = $signed(cache_word_q2[mac_byte_in_word_q2*8 +: 8]);
    wire signed [7:0] in_value_q2 = mac_in_valid_q2 ?
        $signed(line_buf_word_q2[byte_in_word_q2*8 +: 8]) : 8'sd0;

    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] mul_q3;

    integer i;
    integer lane;
    integer ch;
    integer bias_oc;
    integer out_oc;

    reg signed [4:0]  in_row_signed_c;
    reg signed [4:0]  in_col_signed_c;
    reg               in_valid_c;
    reg [5:0]         in_pix_c;
    reg [14:0]        weight_addr_c;
    reg [12:0]        py_k_c;

    wire        line_buf_we = valid_in && ready_in &&
                              ((state == ST_INPUT) || (state == ST_RUNNING));
    wire [9:0]  line_buf_waddr = {in_pixel_counter[5:0], in_beat_index[3:0]};
    wire [9:0]  line_buf_raddr = {mac_in_pix_q1[5:0], mac_in_ch_q1[8:5]};

    always @(posedge clk) begin
        if (line_buf_we) begin
            line_buf[line_buf_waddr] <= data_in;
        end
    end

    always @(posedge clk) begin
        if (active_cache_sel == 1'b0)
            cache_word_q2 <= cache_a[mac_weight_addr_q1[14:3]];
        else
            cache_word_q2 <= cache_b[mac_weight_addr_q1[14:3]];
        mac_byte_in_word_q2 <= mac_weight_addr_q1[2:0];
        line_buf_word_q2    <= line_buf[line_buf_raddr];
        byte_in_word_q2     <= mac_in_ch_q1[4:0];
        mac_kh_q2           <= mac_kh_q1;
        mac_kw_q2           <= mac_kw_q1;
        mac_ic_q2           <= mac_ic_q1;
    end

    always @(posedge clk) begin
        if (mac_valid_q2) begin
            window[mac_kh_q2][mac_kw_q2][mac_ic_q2] <= in_value_q2;
        end
    end

    wire        sched_ready_in_unused;
    wire        sched_needs_real_input_unused;
    wire [3:0]  sched_in_row_unused;
    wire [3:0]  sched_in_col_unused;
    wire        sched_output_fires_unused;
    wire        sched_advance_unused;
    wire        sched_in_frame_done_unused;
    wire        sched_out_frame_done_unused;
    wire [5:0]  sched_outputs_emitted_unused;

    coord_scheduler #(
        .IH(IH), .IW(IW), .OH(OH), .OW(OW),
        .KH(KH), .KW(KW), .SH(1), .SW(1), .PH(PH), .PW(PW)
    ) u_coord_scheduler (
        .clk             (clk),
        .rst_n           (rst_n),
        .start           (1'b0),
        .stall_in        (1'b1),
        .valid_in        (1'b0),
        .ready_in        (sched_ready_in_unused),
        .needs_real_input(sched_needs_real_input_unused),
        .in_row          (sched_in_row_unused),
        .in_col          (sched_in_col_unused),
        .output_fires    (sched_output_fires_unused),
        .advance         (sched_advance_unused),
        .in_frame_done   (sched_in_frame_done_unused),
        .out_frame_done  (sched_out_frame_done_unused),
        .outputs_emitted (sched_outputs_emitted_unused)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state                <= ST_INIT_BOOT;
            ready_in             <= 1'b0;
            valid_out            <= 1'b0;
            data_out             <= {BEAT_BITS{1'b0}};
            in_beat_index        <= 5'd0;
            in_pixel_counter     <= 6'd0;
            active_pixel_counter <= 7'd0;
            kh_counter           <= 2'd0;
            kw_counter           <= 2'd0;
            ic_counter           <= 10'd0;
            k_counter            <= 14'd0;
            lane_counter         <= 2'd0;
            oc_pass              <= 8'd0;
            out_beat             <= 6'd0;
            out_buffer           <= {(OC*8){1'b0}};
            active_cache_sel     <= 1'b0;
            fill_counter         <= 4'd0;
            out_row              <= 4'd0;
            out_col              <= 4'd0;
            mac_valid_q1         <= 1'b0;
            mac_lane_q1          <= 2'd0;
            mac_in_valid_q1      <= 1'b0;
            mac_in_pix_q1        <= 6'd0;
            mac_in_ch_q1         <= 10'd0;
            mac_weight_addr_q1   <= 15'd0;
            mac_kh_q1            <= 2'd0;
            mac_kw_q1            <= 2'd0;
            mac_ic_q1            <= 10'd0;
            mac_done_issuing     <= 1'b0;
            mac_valid_q2         <= 1'b0;
            mac_lane_q2          <= 2'd0;
            mac_in_valid_q2      <= 1'b0;
            mac_kh_q2            <= 2'd0;
            mac_kw_q2            <= 2'd0;
            mac_ic_q2            <= 10'd0;
            mac_valid_q3         <= 1'b0;
            mac_lane_q3          <= 2'd0;
            mul_q3               <= {PROD_W{1'b0}};
            ar_kick              <= 1'b0;
            ar_restart           <= 1'b0;
            v_tmp                <= {SCALED_W{1'b0}};
            for (lane = 0; lane < MP; lane = lane + 1) begin
                acc   [lane] <= {ACC_W{1'b0}};
                scaled[lane] <= {SCALED_W{1'b0}};
            end
        end else begin
            ar_kick      <= 1'b0;
            ar_restart   <= 1'b0;
            valid_out    <= 1'b0;
            mac_valid_q2 <= mac_valid_q1;
            mac_lane_q2  <= mac_lane_q1;
            mac_in_valid_q2 <= mac_in_valid_q1;
            mul_q3       <= weight_q2 * in_value_q2;
            mac_valid_q3 <= mac_valid_q2;
            mac_lane_q3  <= mac_lane_q2;
            if (mac_valid_q3) begin
                acc[mac_lane_q3] <= acc[mac_lane_q3] + $signed(mul_q3);
            end

            in_row_signed_c = $signed({1'b0, out_row}) + $signed({1'b0, kh_counter}) - 5'sd1;
            in_col_signed_c = $signed({1'b0, out_col}) + $signed({1'b0, kw_counter}) - 5'sd1;
            in_valid_c      = (in_row_signed_c >= 5'sd0) && (in_row_signed_c < 5'sd7) &&
                              (in_col_signed_c >= 5'sd0) && (in_col_signed_c < 5'sd7);
            in_pix_c        = in_row_signed_c[3:0] * IW + in_col_signed_c[3:0];
            py_k_c          = ic_counter * KH_KW + kh_counter * KW + kw_counter;
            weight_addr_c   = lane_counter * K_TOTAL + py_k_c;

            case (state)
            ST_INIT_BOOT: begin
                ready_in <= 1'b0;
                ar_kick  <= 1'b1;
                state    <= ST_BOOT_WAIT;
            end
            ST_BOOT_WAIT: begin
                ready_in <= 1'b0;
                if (cache_a_loaded) begin
                    ready_in         <= 1'b1;
                    in_beat_index    <= 5'd0;
                    in_pixel_counter <= 6'd0;
                    fill_counter     <= 4'd0;
                    active_cache_sel <= 1'b0;
                    out_row          <= 4'd0;
                    out_col          <= 4'd0;
                    state            <= ST_INPUT;
                end
            end
            ST_INPUT: begin
                if (valid_in && ready_in) begin
                    if (in_beat_index == IN_BEATS - 1) begin
                        in_beat_index <= 5'd0;
                        if (in_pixel_counter + 6'd1 == TOTAL_IN_PIXELS) begin
                            ready_in         <= 1'b0;
                            in_pixel_counter <= 6'd0;
                        end else begin
                            in_pixel_counter <= in_pixel_counter + 6'd1;
                        end
                    end else begin
                        in_beat_index <= in_beat_index + 5'd1;
                    end
                end
                if (fill_counter < FILL_DELAY) begin
                    fill_counter <= fill_counter + 4'd1;
                end
                if (fill_counter == FILL_DELAY - 1) begin
                    kh_counter       <= 2'd0;
                    kw_counter       <= 2'd0;
                    ic_counter       <= 10'd0;
                    k_counter        <= 14'd0;
                    lane_counter     <= 2'd0;
                    oc_pass          <= 8'd0;
                    mac_done_issuing <= 1'b0;
                    for (lane = 0; lane < MP; lane = lane + 1)
                        acc[lane] <= {ACC_W{1'b0}};
                    ar_kick <= 1'b1;
                    state   <= ST_RUNNING;
                end
            end
            ST_RUNNING: begin
                if (valid_in && ready_in) begin
                    if (in_beat_index == IN_BEATS - 1) begin
                        in_beat_index <= 5'd0;
                        if (in_pixel_counter + 6'd1 == TOTAL_IN_PIXELS) begin
                            ready_in         <= 1'b0;
                            in_pixel_counter <= 6'd0;
                        end else begin
                            in_pixel_counter <= in_pixel_counter + 6'd1;
                        end
                    end else begin
                        in_beat_index <= in_beat_index + 5'd1;
                    end
                end

                if (mac_done_issuing) begin
                    mac_valid_q1 <= 1'b0;
                    if (!mac_valid_q1 && !mac_valid_q2 && !mac_valid_q3) begin
                        mac_done_issuing <= 1'b0;
                        state            <= ST_BIAS_SCALE;
                    end
                end else if (active_cache_loaded) begin
                    mac_valid_q1       <= 1'b1;
                    mac_lane_q1        <= lane_counter;
                    mac_in_valid_q1    <= in_valid_c;
                    mac_in_pix_q1      <= in_pix_c;
                    mac_in_ch_q1       <= ic_counter;
                    mac_weight_addr_q1 <= weight_addr_c;
                    mac_kh_q1          <= kh_counter;
                    mac_kw_q1          <= kw_counter;
                    mac_ic_q1          <= ic_counter;
                    if (lane_counter == MP - 1) begin
                        lane_counter <= 2'd0;
                        if (k_counter == K_TOTAL - 1) begin
                            mac_done_issuing <= 1'b1;
                        end else begin
                            k_counter <= k_counter + 14'd1;
                            if (ic_counter == IC - 1) begin
                                ic_counter <= 10'd0;
                                if (kw_counter == KW - 1) begin
                                    kw_counter <= 2'd0;
                                    kh_counter <= kh_counter + 2'd1;
                                end else begin
                                    kw_counter <= kw_counter + 2'd1;
                                end
                            end else begin
                                ic_counter <= ic_counter + 10'd1;
                            end
                        end
                    end else begin
                        lane_counter <= lane_counter + 2'd1;
                    end
                end else begin
                    mac_valid_q1 <= 1'b0;
                end
            end
            ST_BIAS_SCALE: begin
                for (lane = 0; lane < MP; lane = lane + 1) begin
                    bias_oc = oc_pass * MP + lane;
                    if (bias_oc < OC)
                        scaled[lane] <= ($signed(acc[lane]) +
                                         $signed(bias_rom[bias_oc])) *
                                        $signed(SCALE_MULT_CONST);
                    else
                        scaled[lane] <= {SCALED_W{1'b0}};
                end
                state <= ST_PACK;
            end
            ST_PACK: begin
                for (lane = 0; lane < MP; lane = lane + 1) begin
                    out_oc = oc_pass * MP + lane;
                    v_tmp = (scaled[lane] + SCALE_ROUND_HALF) >>> SCALE_SHIFT;
                    if (out_oc < OC)
                        out_buffer[out_oc*8 +: 8] <= (v_tmp >  127) ?  8'sd127 :
                                                     (v_tmp < -128) ? -8'sd128 :
                                                                       v_tmp[7:0];
                end
                if (oc_pass == OC_PASSES - 1) begin
                    valid_out            <= 1'b1;
                    data_out             <= out_buffer[BEAT_BITS-1:0];
                    out_beat             <= 6'd0;
                    oc_pass              <= 8'd0;
                    active_pixel_counter <= active_pixel_counter + 7'd1;
                    state                <= ST_STREAM_OUT;
                end else begin
                    oc_pass          <= oc_pass + 8'd1;
                    active_cache_sel <= ~active_cache_sel;
                    kh_counter       <= 2'd0;
                    kw_counter       <= 2'd0;
                    ic_counter       <= 10'd0;
                    k_counter        <= 14'd0;
                    lane_counter     <= 2'd0;
                    for (lane = 0; lane < MP; lane = lane + 1)
                        acc[lane] <= {ACC_W{1'b0}};
                    if (oc_pass + 8'd2 <= OC_PASSES)
                        ar_kick <= 1'b1;
                    mac_done_issuing <= 1'b0;
                    state <= ST_RUNNING;
                end
            end
            ST_PASS_WAIT: begin
                if (active_cache_loaded) begin
                    if (oc_pass == 8'd0)
                        ar_kick <= 1'b1;
                    mac_done_issuing <= 1'b0;
                    state            <= ST_RUNNING;
                end
            end
            ST_STREAM_OUT: begin
                if (out_beat == OUT_BEATS - 1) begin
                    valid_out <= 1'b0;
                    out_beat  <= 6'd0;
                    state     <= ST_NEXT_OUT;
                end else begin
                    valid_out <= 1'b1;
                    data_out  <= out_buffer[(out_beat + 1) * BEAT_BITS +: BEAT_BITS];
                    out_beat  <= out_beat + 6'd1;
                end
            end
            ST_NEXT_OUT: begin
                if (active_pixel_counter == ACTIVE_PIXELS) begin
                    active_pixel_counter <= 7'd0;
                    out_row              <= 4'd0;
                    out_col              <= 4'd0;
                    in_pixel_counter     <= 6'd0;
                    in_beat_index        <= 5'd0;
                    fill_counter         <= 4'd0;
                    active_cache_sel     <= 1'b0;
                    ar_restart           <= 1'b1;
                    state                <= ST_INIT_BOOT;
                end else begin
                    if (out_col == OW - 1) begin
                        out_col <= 4'd0;
                        out_row <= out_row + 4'd1;
                    end else begin
                        out_col <= out_col + 4'd1;
                    end
                    active_cache_sel <= 1'b0;
                    ar_restart       <= 1'b1;
                    ar_kick          <= 1'b1;
                    oc_pass          <= 8'd0;
                    kh_counter       <= 2'd0;
                    kw_counter       <= 2'd0;
                    ic_counter       <= 10'd0;
                    k_counter        <= 14'd0;
                    lane_counter     <= 2'd0;
                    mac_done_issuing <= 1'b0;
                    for (lane = 0; lane < MP; lane = lane + 1)
                        acc[lane] <= {ACC_W{1'b0}};
                    state <= ST_PASS_WAIT;
                end
            end
            default: state <= ST_INIT_BOOT;
            endcase
        end
    end

endmodule
