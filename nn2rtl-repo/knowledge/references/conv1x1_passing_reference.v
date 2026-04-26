// Pointwise (1x1) conv2d reference for Foundry/Surgeon to adapt.
//
// Vivado / Artix-7 friendly: the weight ROM is read SYNCHRONOUSLY through a
// registered `weight_q`, which lets `synth_design` infer block ROM cleanly
// instead of leaving a wide combinational mux on the BRAM output. The MAC
// adder consumes `weight_q` one cycle after the address is issued; this
// adds exactly one cycle of pipeline latency per pass, captured in the
// LayerIR via `CONV_PIPELINE_STAGES = 4`.
//
// Original parameters: IC=64 OC=64 IH=IW=112 KH=KW=1 stride=1 pad=0 MP=4.
// Adapt the localparam block (IC/OC/IH/IW/MP/SCALE_MULT/SCALE_SHIFT and the
// $readmemh paths) to a new LayerIR; do not regenerate the FSM from scratch.

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
    localparam K_TOTAL   = IC * KH * KW;  // 64
    localparam MP        = 4;
    localparam OC_PASSES = (OC + MP - 1) / MP;  // 16
    localparam NUM_WEIGHTS    = OC * K_TOTAL;
    localparam WEIGHT_ADDR_W  = (NUM_WEIGHTS <= 1) ? 1 : $clog2(NUM_WEIGHTS);

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

    // Block ROM hints for Vivado. With synchronous reads (weight_q below),
    // synth_design infers a true BRAM with a single read port.
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

    // Registered ROM read. Address is computed combinationally from the
    // counters; weight_q holds the value that arrives one clock later.
    // Declared AFTER the counter regs so Verilog-2001 elaboration sees them.
    reg  signed [7:0]              weight_q;
    wire [31:0]                    current_global_oc;
    wire [WEIGHT_ADDR_W-1:0]       weight_read_addr;
    assign current_global_oc = oc_group * MP + lane_counter;
    assign weight_read_addr  =
        (current_global_oc < OC)
            ? (current_global_oc * K_TOTAL + k_counter)
            : {WEIGHT_ADDR_W{1'b0}};

    always @(posedge clk) begin
        weight_q <= weights[weight_read_addr];
    end

    // MAC-pipeline tracking — registered metadata that pairs with weight_q.
    // mac_valid_q  : weight_q this cycle came from a valid MAC issue
    // mac_lane_q   : which `acc` lane to update on consume
    // mac_k_q      : which `in_latch` channel to multiply with weight_q
    // mac_global_oc_q : OC index — used only for the `< OC` guard so the
    //                   tail of the last OC group does not contaminate acc[]
    // mac_done_issuing: the most recent issue was the LAST one of this pass;
    //                   the next cycle drains the trailing consume and
    //                   transitions to ST_BIAS.
    reg                            mac_valid_q;
    reg [$clog2(MP+1)-1:0]         mac_lane_q;
    reg [$clog2(K_TOTAL+1)-1:0]    mac_k_q;
    reg [31:0]                     mac_global_oc_q;
    reg                            mac_done_issuing;

    integer i, lane;

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
            v_tmp            <= 0;
            for (i = 0; i < IC; i = i + 1)
                in_latch[i] <= 8'sd0;
            for (lane = 0; lane < MP; lane = lane + 1) begin
                acc   [lane] <= 0;
                biased[lane] <= 0;
                scaled[lane] <= 0;
            end
        end else begin
            // Trailing consume: every cycle, if the previous cycle's issue is
            // sitting in weight_q (mac_valid_q == 1), accumulate it. Guarded
            // by `mac_global_oc_q < OC` so the top OC group does not write
            // beyond `acc[]`.
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
                    // Final consume already happened above; advance to BIAS.
                    mac_valid_q      <= 1'b0;
                    mac_done_issuing <= 1'b0;
                    state            <= ST_BIAS;
                end else begin
                    // Issue: capture which lane / k this address is for, so the
                    // NEXT cycle's consume knows where to put weight_q.
                    mac_lane_q       <= lane_counter;
                    mac_k_q          <= k_counter;
                    mac_global_oc_q  <= current_global_oc;
                    mac_valid_q      <= 1'b1;

                    if (lane_counter == MP - 1) begin
                        lane_counter <= 0;
                        if (k_counter == K_TOTAL - 1) begin
                            // Last issue of this pass; one more cycle to drain.
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
                for (lane = 0; lane < MP; lane = lane + 1) begin : BIAS_LANE
                    integer bias_oc;
                    bias_oc = oc_group * MP + lane;
                    if (bias_oc < OC)
                        biased[lane] <= acc[lane] + biases[bias_oc];
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
                for (lane = 0; lane < MP; lane = lane + 1) begin : OUT_LANE
                    integer out_oc;
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
