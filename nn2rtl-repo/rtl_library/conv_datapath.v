// conv_datapath — MAC pipeline only (no coord logic, no line buffer).
// Part of the split spatial-conv architecture (see SPLIT_ARCHITECTURE.md).
//
// Consumes a flat KH*KW*IC-element window (packed INT8) and a `start_mac`
// trigger; produces a scaled+saturated output byte vector on data_out,
// with valid_out asserted once per input trigger. Exposes `mac_busy`
// so the top-level can drive scheduler's stall_in combinationally
// (stall_in = mac_busy — that's it).
//
// Drop-in for BOTH pointwise (KH=KW=1) and spatial (KH*KW > 1) convs:
// the flat window layout `window_flat[(kh*KW*IC + kw*IC + ic)*8 +: 8]`
// collapses to just `window_flat[ic*8 +: 8]` when KH=KW=1, so the MAC
// index decomposition works unchanged.
//
// Weights / biases are loaded via $readmemh inside an initial block. Weight
// reads use a registered address/data path so Vivado can infer block ROM
// instead of a wide async LUT mux.

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

    // Window input (packed). On start_mac, this must hold the receptive
    // field for the pixel to be computed. The top-level coord_scheduler
    // and line_buf_window guarantee the window is stable for the full
    // MAC duration (scheduler is frozen while mac_busy is high).
    input  wire [KH*KW*IC*8-1:0]              window_flat,

    // One-cycle pulse from the scheduler when at a firing coord and the
    // window is ready. Datapath samples window_flat and runs the MAC.
    input  wire                               start_mac,

    // Output data path.
    output reg                                valid_out,
    output reg  [OC*8-1:0]                    data_out,

    // Scheduler-side status. Top-level wires `stall_in = mac_busy` —
    // high through the entire ST_MAC → ST_OUTPUT pipeline, low in
    // ST_IDLE. That's all the coord_scheduler needs; the registered
    // output_fires pulse from the scheduler handles the ST_IDLE→ST_MAC
    // kickoff.
    output wire                               mac_busy
);

    // ---------------- Derived widths ----------------------------------
    localparam integer PROD_W        = 16;
    localparam integer ACC_W         = PROD_W + $clog2(K_TOTAL);
    localparam integer BIAS_W        = 32;
    localparam integer BIASED_W      = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1;
    localparam integer NUM_WEIGHTS   = OC * K_TOTAL;
    localparam integer WEIGHT_ADDR_W = (NUM_WEIGHTS <= 1) ? 1 : $clog2(NUM_WEIGHTS);
    localparam integer SCALE_MAG_W   = $clog2(SCALE_MULT + 1);
    localparam integer SCALE_CONST_W = SCALE_MAG_W + 1;
    localparam integer SCALED_W      = BIASED_W + SCALE_CONST_W;
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

    // ---------------- Weight / bias arrays ----------------------------
    (* rom_style = "block", ram_style = "block" *) reg signed [7:0]  weights [0:NUM_WEIGHTS-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases  [0:OC-1];
    initial begin
        if (WEIGHTS_PATH != "") $readmemh(WEIGHTS_PATH, weights);
        if (BIAS_PATH    != "") $readmemh(BIAS_PATH,    biases);
    end

    reg signed [7:0] weight_q;

    // ---------------- MAC pipeline registers --------------------------
    reg signed [ACC_W-1:0]    acc    [0:MP-1];
    reg signed [BIASED_W-1:0] biased [0:MP-1];
    reg signed [SCALED_W-1:0] scaled [0:MP-1];
    reg signed [SCALED_W-1:0] v_tmp;

    reg [$clog2(K_TOTAL+1)-1:0]   k_counter;
    reg [$clog2(MP+1)-1:0]        lane_counter;
    reg [$clog2(OC_PASSES+1)-1:0] oc_group;
    reg signed [7:0]              tap_q;
    reg [$clog2(MP+1)-1:0]        mac_lane_q;
    integer                       mac_global_oc_q;
    reg                           mac_valid_q;
    reg                           mac_done_issuing;

    integer i, lane_i, issue_global_oc;

    wire [31:0] current_global_oc;
    wire [WEIGHT_ADDR_W-1:0] weight_read_addr;

    assign current_global_oc = oc_group * MP + lane_counter;
    assign weight_read_addr =
        (current_global_oc < OC) ? (current_global_oc * K_TOTAL + k_counter) : {WEIGHT_ADDR_W{1'b0}};

    always @(posedge clk) begin
        weight_q <= weights[weight_read_addr];
    end

    // mac_busy is high whenever the pipeline is not idle.
    assign mac_busy = (state != ST_IDLE);

    // Helper: extract window tap for a given flat kernel index.
    // Index decomposition mirrors the one in 03_conv3x3_pad1.md:
    //   ic = k / (KH*KW); kh = (k % (KH*KW)) / KW; kw = k % KW
    //   tap = window[kh][kw][ic]
    // For KH=KW=1 this collapses to window[0][0][ic] = window_flat[ic*8 +: 8].
    function [7:0] tap_at;
        input [$clog2(K_TOTAL)-1:0] k;
        integer kh_idx, kw_idx, ic_idx, flat_idx;
        begin
            kh_idx   = (k % (KH * KW)) / KW;
            kw_idx   = k % KW;
            ic_idx   = k / (KH * KW);
            flat_idx = kh_idx * KW * IC + kw_idx * IC + ic_idx;
            tap_at   = window_flat[flat_idx*8 +: 8];
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= ST_IDLE;
            valid_out    <= 1'b0;
            data_out     <= {OC*8{1'b0}};
            k_counter    <= 0;
            lane_counter <= 0;
            oc_group     <= 0;
            tap_q        <= 0;
            mac_lane_q   <= 0;
            mac_global_oc_q <= 0;
            mac_valid_q  <= 1'b0;
            mac_done_issuing <= 1'b0;
            v_tmp        <= 0;
            for (i = 0; i < MP; i = i + 1) begin
                acc[i]    <= 0;
                biased[i] <= 0;
                scaled[i] <= 0;
            end
        end else begin
            // Defaults each cycle.
            valid_out <= 1'b0;

            case (state)
                // ------------------------------------------------------
                // ST_IDLE — wait for start_mac. On pulse, sample the
                // current window_flat (assumed stable) and begin MAC.
                // ------------------------------------------------------
                ST_IDLE: begin
                    if (start_mac) begin
                        state        <= ST_MAC;
                        k_counter    <= 0;
                        lane_counter <= 0;
                        oc_group     <= 0;
                        tap_q        <= 0;
                        mac_lane_q   <= 0;
                        mac_global_oc_q <= 0;
                        mac_valid_q  <= 1'b0;
                        mac_done_issuing <= 1'b0;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                            acc[lane_i] <= 0;
                    end
                end

                // ------------------------------------------------------
                // ST_MAC — serialized MP-lane rotation over K_TOTAL taps.
                // Vivado-friendly synchronous ROM read: issue address/tap in
                // one cycle, consume registered weight_q on the next. Once
                // primed, throughput is still one MAC per cycle.
                // ------------------------------------------------------
                ST_MAC: begin : MAC_BLOCK
                    if (mac_valid_q && mac_global_oc_q < OC) begin
                        acc[mac_lane_q] <= acc[mac_lane_q] +
                            $signed(weight_q) * $signed(tap_q);
                    end

                    if (mac_done_issuing) begin
                        mac_valid_q      <= 1'b0;
                        mac_done_issuing <= 1'b0;
                        k_counter        <= 0;
                        lane_counter     <= 0;
                        state            <= ST_BIAS;
                    end else begin
                        issue_global_oc = oc_group * MP + lane_counter;
                        tap_q           <= $signed(tap_at(k_counter));
                        mac_lane_q      <= lane_counter;
                        mac_global_oc_q <= issue_global_oc;
                        mac_valid_q     <= 1'b1;

                        if (lane_counter == MP - 1) begin
                            lane_counter <= 0;
                            if (k_counter == K_TOTAL - 1) begin
                                k_counter        <= 0;
                                mac_done_issuing <= 1'b1;
                            end else begin
                                k_counter <= k_counter + 1;
                            end
                        end else begin
                            lane_counter <= lane_counter + 1;
                        end
                    end
                end

                // ------------------------------------------------------
                // ST_BIAS — per-channel bias add for the current OC group.
                // Direct signed add; reg signed context sign-extends
                // automatically.
                // ------------------------------------------------------
                ST_BIAS: begin : BIAS_BLOCK
                    integer bias_oc;
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        bias_oc = oc_group * MP + lane_i;
                        if (bias_oc < OC) begin
                            biased[lane_i] <= $signed(acc[lane_i]) + $signed(biases[bias_oc]);
                        end else begin
                            biased[lane_i] <= 0;
                        end
                    end
                    state <= ST_SCALE;
                end

                // ------------------------------------------------------
                // ST_SCALE — multiply by SCALE_MULT_CONST.
                // ------------------------------------------------------
                ST_SCALE: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                        scaled[lane_i] <= $signed(biased[lane_i]) *
                                          $signed(SCALE_MULT_CONST);
                    state <= ST_OUTPUT;
                end

                // ------------------------------------------------------
                // ST_OUTPUT — round-to-nearest, INT8 saturation, packed
                // write into data_out[global_oc*8 +: 8]. On the last OC
                // group, fire valid_out and return to ST_IDLE. Scheduler
                // resumes advancing next cycle when mac_busy drops.
                // ------------------------------------------------------
                ST_OUTPUT: begin : OUT_BLOCK
                    integer out_oc;
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
                        oc_group     <= oc_group + 1;
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
