// Pointwise (1x1) conv2d reference - serialized, Vivado/Artix-7 friendly.
//
// Current verified contract
// =========================
// MP is the number of accumulator lanes in one output-channel group. A
// lane_counter selects exactly one lane per cycle, so each ST_RUNNING cycle
// performs one synchronous ROM read, one multiply, and one accumulation into
// acc[lane_counter]. A pass therefore costs MP*K_TOTAL issue cycles plus one
// trailing consume, BIAS, SCALE, and OUTPUT: MP*K_TOTAL + 4 cycles.
//
// LayerIR.weight_bank_paths may exist, but this reference intentionally uses
// the flat weights_path because the banked-parallel datapath has a different
// latency contract. Do not switch this file to MP parallel bank reads unless
// compute_conv2d_latency_cycles() and the static testbench expectations are
// updated at the same time.

module layer1_0_conv1 (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              valid_in,
    output reg               ready_in,
    input  wire [511:0]      data_in,
    output reg               valid_out,
    output reg  [511:0]      data_out
);
    localparam IC        = 64;
    localparam OC        = 64;
    localparam IH        = 112;
    localparam IW        = 112;
    localparam OH        = 112;
    localparam OW        = 112;
    localparam KH        = 1;
    localparam KW        = 1;
    localparam SH        = 1;
    localparam SW        = 1;
    localparam PH        = 0;
    localparam PW        = 0;
    localparam K_TOTAL   = IC * KH * KW;
    localparam MP        = 4;
    localparam OC_PASSES = (OC + MP - 1) / MP;
    localparam NUM_WEIGHTS    = OC * K_TOTAL;
    localparam WEIGHT_ADDR_W  = (NUM_WEIGHTS <= 1) ? 1 : $clog2(NUM_WEIGHTS);
    localparam OC_INDEX_W     = (OC + MP <= 1) ? 1 : $clog2(OC + MP);

    localparam SCALE_MULT  = 29009;
    localparam SCALE_SHIFT = 20;

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

    localparam ST_STREAM  = 3'd0;
    localparam ST_RUNNING = 3'd1;
    localparam ST_BIAS    = 3'd2;
    localparam ST_SCALE   = 3'd3;
    localparam ST_OUTPUT  = 3'd4;

    (* rom_style = "block", ram_style = "block" *) reg signed [7:0]  weights [0:OC*K_TOTAL-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases  [0:OC-1];
    initial begin
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/layer1_0_conv1_weights.hex", weights);
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/layer1_0_conv1_bias.hex", biases);
    end

    reg signed [7:0] in_latch [0:IC-1];

    reg signed [ACC_W-1:0]    acc    [0:MP-1];
    reg signed [BIASED_W-1:0] biased [0:MP-1];
    reg signed [SCALED_W-1:0] scaled [0:MP-1];
    reg signed [SCALED_W-1:0] v_tmp;

    reg [$clog2(K_TOTAL+1)-1:0]   k_counter;
    reg [$clog2(MP+1)-1:0]        lane_counter;
    reg [$clog2(OC_PASSES+1)-1:0] oc_group;
    reg [2:0] state;

    reg  signed [7:0]              weight_q;
    wire [OC_INDEX_W-1:0]          current_global_oc;
    wire [WEIGHT_ADDR_W-1:0]       weight_read_addr;
    assign current_global_oc = oc_group * MP + lane_counter;
    assign weight_read_addr  =
        (current_global_oc < OC)
            ? (current_global_oc * K_TOTAL + k_counter)
            : {WEIGHT_ADDR_W{1'b0}};

    always @(posedge clk) begin
        weight_q <= weights[weight_read_addr];
    end

    reg                            mac_valid_q;
    reg [$clog2(MP+1)-1:0]         mac_lane_q;
    reg [$clog2(K_TOTAL+1)-1:0]    mac_k_q;
    reg [OC_INDEX_W-1:0]           mac_global_oc_q;
    reg                            mac_done_issuing;

    integer i, lane;
    integer bias_oc;
    integer out_oc;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= ST_STREAM;
            ready_in         <= 1'b1;  // [INVARIANT:READY_IN_GATING]
            valid_out        <= 1'b0;
            k_counter        <= 0;
            lane_counter     <= 0;
            oc_group         <= 0;
            data_out         <= {(OC*8){1'b0}};
            mac_valid_q      <= 1'b0;
            mac_lane_q       <= 0;
            mac_k_q          <= 0;
            mac_global_oc_q  <= 0;
            mac_done_issuing <= 1'b0;
            for (i = 0; i < IC; i = i + 1)
                in_latch[i] <= 8'sd0;
            for (lane = 0; lane < MP; lane = lane + 1) begin
                acc   [lane] <= 0;
                biased[lane] <= 0;
                scaled[lane] <= 0;
            end
        end else begin
            if (mac_valid_q && mac_global_oc_q < OC) begin
                acc[mac_lane_q] <= acc[mac_lane_q] +
                    $signed(weight_q) * $signed(in_latch[mac_k_q]);
            end

            case (state)

            ST_STREAM: begin
                valid_out   <= 1'b0;
                mac_valid_q <= 1'b0;
                if (valid_in) begin
                    for (i = 0; i < IC; i = i + 1)
                        in_latch[i] <= $signed(data_in[i*8 +: 8]);
                    ready_in         <= 1'b0;  // [INVARIANT:READY_IN_GATING]
                    k_counter        <= 0;
                    lane_counter     <= 0;
                    oc_group         <= 0;
                    mac_done_issuing <= 1'b0;
                    for (lane = 0; lane < MP; lane = lane + 1)
                        acc[lane] <= 0;
                    state <= ST_RUNNING;
                end
            end

            ST_RUNNING: begin
                if (mac_done_issuing) begin
                    mac_valid_q      <= 1'b0;
                    mac_done_issuing <= 1'b0;
                    state            <= ST_BIAS;
                end else begin
                    mac_lane_q       <= lane_counter;
                    mac_k_q          <= k_counter;
                    mac_global_oc_q  <= current_global_oc;
                    mac_valid_q      <= 1'b1;

                    if (lane_counter == MP - 1) begin
                        lane_counter <= 0;
                        if (k_counter == K_TOTAL - 1) begin
                            mac_done_issuing <= 1'b1;
                        end else begin
                            k_counter <= k_counter + 1;
                        end
                    end else begin
                        lane_counter <= lane_counter + 1;
                    end
                end
            end

            ST_BIAS: begin
                for (lane = 0; lane < MP; lane = lane + 1) begin
                    bias_oc = oc_group * MP + lane;
                    if (bias_oc < OC)
                        biased[lane] <= $signed(acc[lane]) + $signed(biases[bias_oc]);
                    else
                        biased[lane] <= 0;
                end
                state <= ST_SCALE;
            end

            ST_SCALE: begin
                for (lane = 0; lane < MP; lane = lane + 1)
                    scaled[lane] <= $signed(biased[lane]) * $signed(SCALE_MULT_CONST);
                state <= ST_OUTPUT;
            end

            ST_OUTPUT: begin
                for (lane = 0; lane < MP; lane = lane + 1) begin
                    out_oc = oc_group * MP + lane;
                    if (out_oc < OC) begin
                        // [INVARIANT:ROUNDING]
                        v_tmp = (scaled[lane] + SCALE_ROUND_BIAS) >>> SCALE_SHIFT;
                        data_out[out_oc*8 +: 8] <= (v_tmp >  127) ?  8'sd127 :
                                                   (v_tmp < -128) ? -8'sd128 :
                                                                    v_tmp[7:0];
                    end
                end

                if (oc_group < OC_PASSES - 1) begin
                    for (lane = 0; lane < MP; lane = lane + 1) acc[lane] <= 0;
                    k_counter    <= 0;
                    lane_counter <= 0;
                    oc_group     <= oc_group + 1;
                    state        <= ST_RUNNING;
                end else begin
                    valid_out <= 1'b1;  // [INVARIANT:VALID_OUT_LATENCY]
                    ready_in  <= 1'b1;  // [INVARIANT:READY_IN_GATING]
                    oc_group  <= 0;
                    state     <= ST_STREAM;
                end
            end

            default: state <= ST_STREAM;
            endcase
        end
    end
endmodule
