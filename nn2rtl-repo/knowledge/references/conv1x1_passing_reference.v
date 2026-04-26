// Pointwise (1x1) conv2d reference - serialized, Vivado/Artix-7 friendly,
// DSP48E1-inferring.
//
// Current verified contract
// =========================
// MP is the number of accumulator lanes in one output-channel group. A
// lane_counter selects exactly one lane per cycle, so each ST_RUNNING
// "issue" cycle performs one synchronous ROM read, one multiplier-input
// register, and (one cycle later) one multiply, and (one more cycle
// later) one accumulation into acc[lane_counter].
//
// Pipeline stages from issue to acc-write:
//   stage 1 (1 cycle delay):  weight_q       <= weights[bank_addr]
//                             tap stays as the registered in_latch read
//                             mac_*_q1       <= captured issue metadata
//   stage 2 (1 more cycle):   mul_q          <= weight_q * in_latch[mac_k_q1]
//                             mac_*_q2       <= mac_*_q1
//   stage 3 (1 more cycle):   acc[mac_lane_q2] <= acc + mul_q   (DSP P-reg)
//
// Per pass (one OC group): MP*K_TOTAL issue cycles + 2 trailing drain
// cycles + ST_BIAS (1) + ST_SCALE (1) + ST_OUTPUT (1) = MP*K_TOTAL + 6.
// (compute_conv2d_latency_cycles in scripts/golden_impl.py uses
// CONV_PIPELINE_STAGES = 6 for this contract.)
//
// (* use_dsp = "yes" *) on the registered `mul_q` is the canonical Vivado
// DSP-inference pattern: a registered multiplier output (MREG=1) feeding
// an external accumulator. The MP-way mux on `acc[mac_lane_q2]` lives in
// LUT fabric; the multiplier itself maps to one DSP48E1 block.
//
// LayerIR.weight_bank_paths may exist, but this reference intentionally
// uses the flat weights_path because the banked-parallel datapath has a
// different latency contract. Do not switch this file to MP parallel
// bank reads unless compute_conv2d_latency_cycles() and the static
// testbench expectations are updated at the same time.

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

    // Stage 1: capture issue metadata. Stage 2 propagates to _q2 alongside
    // the registered multiplier output `mul_q`. The consume reads _q2.
    reg                            mac_valid_q1;
    reg [$clog2(MP+1)-1:0]         mac_lane_q1;
    reg [$clog2(K_TOTAL+1)-1:0]    mac_k_q1;
    reg [OC_INDEX_W-1:0]           mac_global_oc_q1;
    reg                            mac_done_issuing;

    reg                            mac_valid_q2;
    reg [$clog2(MP+1)-1:0]         mac_lane_q2;
    reg [OC_INDEX_W-1:0]           mac_global_oc_q2;

    // Registered multiplier output. `(* use_dsp = "yes" *)` on the reg
    // declaration forces Vivado to map this multiply into a DSP48E1's
    // M-register, instead of leaving the 8x8 signed multiply in LUT
    // fabric (which is what happened with the prior wire-based form).
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] mul_q;

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
            mac_valid_q1     <= 1'b0;
            mac_lane_q1      <= 0;
            mac_k_q1         <= 0;
            mac_global_oc_q1 <= 0;
            mac_valid_q2     <= 1'b0;
            mac_lane_q2      <= 0;
            mac_global_oc_q2 <= 0;
            mac_done_issuing <= 1'b0;
            mul_q            <= 0;
            for (i = 0; i < IC; i = i + 1)
                in_latch[i] <= 8'sd0;
            for (lane = 0; lane < MP; lane = lane + 1) begin
                acc   [lane] <= 0;
                biased[lane] <= 0;
                scaled[lane] <= 0;
            end
        end else begin
            // Stage 2: register the multiplier output. Use mac_k_q1 to
            // index in_latch so the multiplier inputs are paired with the
            // issue at the right cycle. mac_valid_q2 / mac_lane_q2 /
            // mac_global_oc_q2 forward stage-1's metadata one cycle later
            // so the consume sees the right targeting.
            mul_q            <= $signed(weight_q) * $signed(in_latch[mac_k_q1]);
            mac_valid_q2     <= mac_valid_q1;
            mac_lane_q2      <= mac_lane_q1;
            mac_global_oc_q2 <= mac_global_oc_q1;

            // Stage 3: accumulate the registered product into the lane's
            // accumulator. The MP-way mux on `acc[mac_lane_q2]` lives in
            // LUT fabric; the DSP48E1 only owns the multiplier itself.
            if (mac_valid_q2 && mac_global_oc_q2 < OC) begin
                acc[mac_lane_q2] <= acc[mac_lane_q2] + $signed(mul_q);
            end

            case (state)

            ST_STREAM: begin
                valid_out    <= 1'b0;
                mac_valid_q1 <= 1'b0;
                mac_valid_q2 <= 1'b0;
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
                    // Stop issuing. Wait two more cycles for stages 2 and
                    // 3 to drain (mul_q forwards last issue's product;
                    // consume processes it). State transitions to ST_BIAS
                    // when mac_valid_q2 has gone to 0, which means the
                    // last consume just landed.
                    mac_valid_q1 <= 1'b0;
                    if (!mac_valid_q1 && !mac_valid_q2) begin
                        // Both stages drained. Last consume has landed.
                        mac_done_issuing <= 1'b0;
                        state            <= ST_BIAS;
                    end
                end else begin
                    // Issue cycle: stage 1 captures metadata for this
                    // (lane, k). The address is already on `weight_q`'s
                    // input; weight_q will register it next cycle.
                    mac_lane_q1      <= lane_counter;
                    mac_k_q1         <= k_counter;
                    mac_global_oc_q1 <= current_global_oc;
                    mac_valid_q1     <= 1'b1;

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
