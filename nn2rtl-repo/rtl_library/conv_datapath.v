// conv_datapath - serialized MAC pipeline (no coord logic, no line buffer).
// Part of the split spatial-conv architecture (see SPLIT_ARCHITECTURE.md).
//
// Consumes a flat KH*KW*IC-element window (packed INT8) and a start_mac
// trigger; produces a scaled+saturated output byte vector on data_out, with
// valid_out asserted once per input trigger. Exposes mac_busy so the top-level
// can drive coord_scheduler.stall_in = mac_busy.
//
// Current verified contract
// =========================
// MP is the number of accumulator lanes in the current output-channel group.
// It is NOT MP cycle-parallel BRAM throughput in this library. A lane_counter
// selects one lane per cycle, so each ST_MAC cycle performs one registered ROM
// read, one multiply, and one accumulation into acc[lane_counter].
//
// Per pass (one OC group): MP*K_TOTAL issues + 1 trailing consume + BIAS +
// SCALE + OUTPUT = MP*K_TOTAL + 4 cycles.
//
// weight_bank_paths emitted by the Python frontend are reserved for a future
// banked-parallel datapath with a different latency contract. This module
// intentionally uses the flat WEIGHTS_PATH so it stays aligned with
// compute_conv2d_latency_cycles() and the static testbench.

module conv_datapath #(
    parameter integer IC          = 64,
    parameter integer OC          = 64,
    parameter integer KH          = 3,
    parameter integer KW          = 3,
    parameter integer K_TOTAL     = IC * KH * KW,
    parameter integer MP          = 4,
    parameter integer OC_PASSES   = (OC + MP - 1) / MP,
    parameter integer SCALE_MULT  = 1,
    parameter integer SCALE_SHIFT = 16,
    parameter         WEIGHTS_PATH = "",
    parameter         BIAS_PATH    = ""
) (
    input  wire                               clk,
    input  wire                               rst_n,

    input  wire [KH*KW*IC*8-1:0]              window_flat,
    input  wire                               start_mac,

    output reg                                valid_out,
    output reg  [OC*8-1:0]                    data_out,
    output wire                               mac_busy
);

    // ---------------- Derived widths ----------------------------------
    localparam integer PROD_W          = 16;
    localparam integer ACC_W           = PROD_W + $clog2(K_TOTAL);
    localparam integer BIAS_W          = 32;
    localparam integer BIASED_W        = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1;
    localparam integer SCALE_MAG_W     = $clog2(SCALE_MULT + 1);
    localparam integer SCALE_CONST_W   = SCALE_MAG_W + 1;
    localparam integer SCALED_W        = BIASED_W + SCALE_CONST_W;
    localparam integer NUM_WEIGHTS     = OC * K_TOTAL;
    localparam integer WEIGHT_ADDR_W   = (NUM_WEIGHTS <= 1) ? 1 : $clog2(NUM_WEIGHTS);
    localparam integer K_COUNTER_W     = (K_TOTAL <= 1) ? 1 : $clog2(K_TOTAL);
    localparam integer LANE_COUNTER_W  = (MP <= 1) ? 1 : $clog2(MP);
    localparam integer OC_GROUP_W      = (OC_PASSES <= 1) ? 1 : $clog2(OC_PASSES);
    localparam integer OC_INDEX_W      = (OC + MP <= 1) ? 1 : $clog2(OC + MP);
    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST = SCALE_MULT;
    localparam signed [SCALED_W-1:0] SCALE_ROUND_BIAS =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);

    // ---------------- FSM states --------------------------------------
    localparam ST_IDLE   = 3'd0;
    localparam ST_MAC    = 3'd1;
    localparam ST_BIAS   = 3'd2;
    localparam ST_SCALE  = 3'd3;
    localparam ST_OUTPUT = 3'd4;

    reg [2:0] state;

    // ---------------- Flat synchronous ROM + bias ---------------------
    (* rom_style = "block", ram_style = "block" *) reg signed [7:0] weights [0:NUM_WEIGHTS-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases [0:OC-1];
    initial begin
        if (WEIGHTS_PATH != "") $readmemh(WEIGHTS_PATH, weights);
        if (BIAS_PATH    != "") $readmemh(BIAS_PATH,    biases);
    end

    // ---------------- MAC pipeline registers --------------------------
    reg signed [ACC_W-1:0]    acc    [0:MP-1];
    reg signed [BIASED_W-1:0] biased [0:MP-1];
    reg signed [SCALED_W-1:0] scaled [0:MP-1];
    reg signed [SCALED_W-1:0] v_tmp;

    reg [K_COUNTER_W-1:0]    k_counter;
    reg [LANE_COUNTER_W-1:0] lane_counter;
    reg [OC_GROUP_W-1:0]     oc_group;

    integer i, lane_i;
    integer bias_oc;
    integer out_oc;

    assign mac_busy = (state != ST_IDLE);

    wire [OC_INDEX_W-1:0] current_global_oc = oc_group * MP + lane_counter;
    wire [WEIGHT_ADDR_W-1:0] weight_read_addr =
        (current_global_oc < OC)
            ? (current_global_oc * K_TOTAL + k_counter)
            : {WEIGHT_ADDR_W{1'b0}};

    // Window-tap indexer: kernel-index k -> flat window byte.
    //   ic = k / (KH*KW); kh = (k % (KH*KW)) / KW; kw = k % KW
    //   tap = window[kh][kw][ic] with layout (kh*KW*IC + kw*IC + ic)*8
    function [7:0] tap_at;
        input [K_COUNTER_W-1:0] k;
        integer kh_idx, kw_idx, ic_idx, flat_idx;
        begin
            kh_idx   = (k % (KH * KW)) / KW;
            kw_idx   = k % KW;
            ic_idx   = k / (KH * KW);
            flat_idx = kh_idx * KW * IC + kw_idx * IC + ic_idx;
            tap_at   = window_flat[flat_idx*8 +: 8];
        end
    endfunction

    reg signed [7:0] weight_q;
    reg signed [7:0] tap_q;
    always @(posedge clk) begin
        weight_q <= weights[weight_read_addr];
        tap_q    <= $signed(tap_at(k_counter));
    end

    (* use_dsp = "yes" *) wire signed [PROD_W-1:0] mul_q;
    assign mul_q = $signed(weight_q) * $signed(tap_q);

    reg                        mac_valid_q;
    reg [LANE_COUNTER_W-1:0]   mac_lane_q;
    reg [OC_INDEX_W-1:0]       mac_global_oc_q;
    reg                        mac_done_issuing;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= ST_IDLE;
            valid_out        <= 1'b0;
            data_out         <= {OC*8{1'b0}};
            k_counter        <= 0;
            lane_counter     <= 0;
            oc_group         <= 0;
            mac_valid_q      <= 1'b0;
            mac_lane_q       <= 0;
            mac_global_oc_q  <= 0;
            mac_done_issuing <= 1'b0;
            for (i = 0; i < MP; i = i + 1) begin
                acc[i]    <= 0;
                biased[i] <= 0;
                scaled[i] <= 0;
            end
        end else begin
            valid_out <= 1'b0;

            // Consume the registered ROM/tap pair from the previous issue.
            if (mac_valid_q && mac_global_oc_q < OC) begin
                acc[mac_lane_q] <= acc[mac_lane_q] + $signed(mul_q);
            end

            case (state)
                ST_IDLE: begin
                    if (start_mac) begin
                        state            <= ST_MAC;
                        k_counter        <= 0;
                        lane_counter     <= 0;
                        oc_group         <= 0;
                        mac_valid_q      <= 1'b0;
                        mac_done_issuing <= 1'b0;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                            acc[lane_i] <= 0;
                    end
                end

                ST_MAC: begin
                    if (mac_done_issuing) begin
                        // Final consume already happened above; advance to BIAS.
                        mac_valid_q      <= 1'b0;
                        mac_done_issuing <= 1'b0;
                        state            <= ST_BIAS;
                    end else begin
                        mac_lane_q      <= lane_counter;
                        mac_global_oc_q <= current_global_oc;
                        mac_valid_q     <= 1'b1;

                        if (lane_counter == MP - 1) begin
                            lane_counter <= 0;
                            if (k_counter == K_TOTAL - 1) begin
                                mac_done_issuing <= 1'b1;
                            end else begin
                                k_counter <= k_counter + 1'b1;
                            end
                        end else begin
                            lane_counter <= lane_counter + 1'b1;
                        end
                    end
                end

                ST_BIAS: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        bias_oc = oc_group * MP + lane_i;
                        if (bias_oc < OC)
                            biased[lane_i] <= $signed(acc[lane_i]) + $signed(biases[bias_oc]);
                        else
                            biased[lane_i] <= 0;
                    end
                    state <= ST_SCALE;
                end

                ST_SCALE: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                        scaled[lane_i] <= $signed(biased[lane_i]) *
                                          $signed(SCALE_MULT_CONST);
                    state <= ST_OUTPUT;
                end

                ST_OUTPUT: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        out_oc = oc_group * MP + lane_i;
                        if (out_oc < OC) begin
                            // [INVARIANT:ROUNDING]
                            v_tmp = (scaled[lane_i] + SCALE_ROUND_BIAS) >>> SCALE_SHIFT;
                            data_out[out_oc*8 +: 8] <=
                                (v_tmp >  127) ?  8'sd127 :
                                (v_tmp < -128) ? -8'sd128 : v_tmp[7:0];
                        end
                    end

                    if (oc_group == OC_PASSES - 1) begin
                        // [INVARIANT:VALID_OUT_LATENCY]
                        valid_out <= 1'b1;
                        state     <= ST_IDLE;
                    end else begin
                        oc_group     <= oc_group + 1'b1;
                        k_counter    <= 0;
                        lane_counter <= 0;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                            acc[lane_i] <= 0;
                        state <= ST_MAC;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
