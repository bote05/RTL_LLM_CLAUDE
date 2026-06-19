// node_conv_910 - pointwise 1x1 conv2d, tiled-streaming contract.  [COMPRESSED VARIANT]
//   IC=960, OC=320, IH=IW=7 -> OH=OW=7 (stride=1), MP=4.
//   channel_tile=32 => IN_BEATS=30, OUT_BEATS=10, K_TOTAL=960.
//   OC_PASSES = ceil(OC/MP) = 80.
//   Latency = IN_BEATS + OC_PASSES * (MP*K_TOTAL + 6) = 30 + 80*3846 = 307710.
//
// COMPRESSION (use-bram + reduce-lut), ported from the proven byte-exact
// node_conv_252__reduce-lut-use-bram template:
//   * The baseline mapped the 307200x8 (Vivado-padded 524288x8) weight ROM to
//     LUT *logic* (report_ram_utilization: "weights 524288x8 LUT"), burning
//     ~46k LUT6 + 23k MUXF7 even though rom_style="block" was set. Root cause:
//     the 2.4Mbit array exceeds Vivado's default MAX_BRAM_CASCADE_HEIGHT and
//     the cheap 1-cycle read template gives the cascade no output-register
//     stage, so Vivado falls back to LUT-ROM.
//   * Fix (use-bram): cascade_height=8 on the weights array + the canonical
//     2-cycle registered BRAM read (waddr_r -> weights[] -> weight_q). This is
//     the RAMB36 DOA_REG=1 inference template Vivado actually maps to BRAM.
//   * Fix (reduce-lut): in_latch / out_pack moved to dedicated index-addressed
//     arrays (in_latch_mem / out_pack_mem) so the wide variable-select
//     out_pack[global_oc*8 +: 8] mux tree collapses out of LUT logic.
//   * The MAC pipeline grows from 2 to 3 stages to track the deeper weight
//     read; the old ST_BIAS state is folded into the final (4th) drain cycle
//     so per-pass = MP*K_TOTAL + 6 = 3846 and total latency stays 307710.
//
// IMPORTANT: K_TOTAL=960 is NOT a power of two, so the weight address is built
// ARITHMETICALLY ((oc_group*MP+lane)*K_TOTAL + k_counter), unlike the 252
// reference which could bit-concatenate (K_TOTAL=1024). All requant constants
// (SCALE_MULT/SHIFT, SCALED_W=49, clamp literal widths) are byte-identical to
// the baseline node_conv_910.v.
module node_conv_910 (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              valid_in,
    output reg               ready_in,
    input  wire [255:0]      data_in,
    output reg               valid_out,
    output reg  [255:0]      data_out
);

    localparam integer IC           = 960;
    localparam integer OC           = 320;
    localparam integer IH           = 7;
    localparam integer IW           = 7;
    localparam integer OH           = 7;
    localparam integer OW           = 7;
    localparam integer KH           = 1;
    localparam integer KW           = 1;
    localparam integer K_TOTAL      = IC*KH*KW;
    localparam integer MP           = 4;
    localparam integer OC_PASSES    = (OC + MP - 1) / MP;
    localparam integer CHANNEL_TILE = 32;
    localparam integer TILE_BITS    = CHANNEL_TILE * 8;
    localparam integer IN_BEATS     = (IC + CHANNEL_TILE - 1) / CHANNEL_TILE;
    localparam integer OUT_BEATS    = (OC + CHANNEL_TILE - 1) / CHANNEL_TILE;
    localparam integer TOTAL_PIXELS = OH * OW;

    localparam integer SCALE_MULT   = 21804;
    localparam integer SCALE_SHIFT  = 22;

    localparam integer PROD_W       = 16;
    localparam integer ACC_W        = PROD_W + 10;
    localparam integer BIAS_W       = 32;
    localparam integer BIASED_W     = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1;
    localparam integer SCALE_MAG_W  = 15;
    localparam integer SCALE_CONST_W= SCALE_MAG_W + 1;
    localparam integer SCALED_W     = BIASED_W + SCALE_CONST_W;

    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST = SCALE_MULT;
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};

    // Weight ROM forced into cascaded Block RAM (32x RAMB36 chain) via
    // cascade_height + the 2-cycle registered read pipeline below.
    (* rom_style = "block", ram_style = "block", cascade_height = 8 *)
    reg signed [7:0]  weights [0:OC*K_TOTAL-1];
    (* rom_style = "block", ram_style = "block" *)
    reg signed [31:0] biases  [0:OC-1];

    initial begin
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_910_weights.hex", weights);
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_conv_910_bias.hex", biases);
    end

    // Activation latch + output pack as dedicated index-addressed arrays so the
    // wide variable-select muxes drop out of LUT logic.
    (* ram_style = "distributed" *) reg [7:0] in_latch_mem  [0:IC-1];
    (* ram_style = "distributed" *) reg [7:0] out_pack_mem  [0:OC-1];

    localparam [2:0] ST_LOAD=3'd0, ST_RUNNING=3'd1, ST_BIAS=3'd2, ST_SCALE=3'd3, ST_OUTPUT=3'd4, ST_EMIT=3'd5;
    reg [2:0] state;

    reg [4:0]  in_beat_idx;
    reg [3:0]  out_beat_idx;
    reg [2:0]  pixel_row;
    reg [2:0]  pixel_col;
    reg [5:0]  active_emit_count;
    reg [6:0]  oc_group;
    reg [1:0]  lane_counter;
    reg [9:0]  k_counter;

    // BRAM read pipeline: registered ARITHMETIC address -> BRAM array ->
    // registered output. K_TOTAL=960 is not a power of two so the address must
    // be computed, not bit-concatenated.
    reg [18:0]       waddr_r;
    reg signed [7:0] weight_q;
    reg signed [7:0] data_q1, data_q;

    (* use_dsp = "yes" *)
    reg signed [PROD_W-1:0]       mul_q;
    reg [1:0]                     mac_lane_q1, mac_lane_q2, mac_lane_q3;
    reg                           mac_valid_q1, mac_valid_q2, mac_valid_q3;
    reg                           mac_done_issuing;
    reg [1:0]                     drain_cnt;

    reg signed [ACC_W-1:0]        acc    [0:MP-1];
    reg signed [BIASED_W-1:0]     biased [0:MP-1];
    reg signed [SCALED_W-1:0]     scaled [0:MP-1];

    integer i;
    integer lane;
    integer global_oc;

    // Combinational requant (rounding + clamp), identical arithmetic to the
    // baseline ST_OUTPUT body. Literal widths (49-bit) match SCALED_W=49.
    reg signed [SCALED_W-1:0] vtmp_comb  [0:MP-1];
    reg signed [7:0]          clamp_byte [0:MP-1];
    integer lc;
    always @* begin
        for (lc = 0; lc < MP; lc = lc + 1) begin
            // [INVARIANT:ROUNDING]
            vtmp_comb[lc] = (scaled[lc] +
                (scaled[lc][SCALED_W-1] ? SCALE_ROUND_HALF_M1 : SCALE_ROUND_HALF)
            ) >>> SCALE_SHIFT;
            if (vtmp_comb[lc] > 49'sd127)
                clamp_byte[lc] = 8'sd127;
            else if (vtmp_comb[lc] < -49'sd128)
                clamp_byte[lc] = -8'sd128;
            else
                clamp_byte[lc] = vtmp_comb[lc][7:0];
        end
    end

    // Canonical Vivado cascaded-BRAM inference template:
    //   stage 1: waddr_r registers the arithmetic address (BRAM input reg).
    //   stage 2: weight_q <= weights[waddr_r] is the BRAM read + output reg.
    // data_q1 / data_q form the matching 2-stage operand delay.
    always @(posedge clk) begin
        waddr_r  <= (oc_group * MP + lane_counter) * K_TOTAL + k_counter;
        weight_q <= weights[waddr_r];
        data_q1  <= in_latch_mem[k_counter];
        data_q   <= data_q1;
    end

    integer wi, lw;
    always @(posedge clk) begin
        if (state == ST_LOAD && valid_in && ready_in) begin
            for (wi = 0; wi < CHANNEL_TILE; wi = wi + 1) begin
                if ((in_beat_idx * CHANNEL_TILE + wi) < IC)
                    in_latch_mem[in_beat_idx*CHANNEL_TILE + wi] <= data_in[wi*8 +: 8];
            end
        end
        if (state == ST_OUTPUT) begin
            for (lw = 0; lw < MP; lw = lw + 1) begin
                if ((oc_group*MP + lw) < OC)
                    out_pack_mem[oc_group*MP + lw] <= clamp_byte[lw];
            end
        end
    end

    integer bi;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_LOAD; ready_in <= 1'b1; valid_out <= 1'b0; data_out <= 256'd0;
            in_beat_idx <= 5'd0; pixel_row <= 3'd0; pixel_col <= 3'd0; active_emit_count <= 6'd0;
            oc_group <= 7'd0; lane_counter <= 2'd0; k_counter <= 10'd0;
            out_beat_idx <= 4'd0;
            mul_q <= {PROD_W{1'b0}};
            mac_lane_q1 <= 2'd0; mac_lane_q2 <= 2'd0; mac_lane_q3 <= 2'd0;
            mac_valid_q1 <= 1'b0; mac_valid_q2 <= 1'b0; mac_valid_q3 <= 1'b0;
            mac_done_issuing <= 1'b0; drain_cnt <= 2'd0;
            for (i = 0; i < MP; i = i + 1) begin acc[i] <= {ACC_W{1'b0}}; biased[i] <= {BIASED_W{1'b0}}; scaled[i] <= {SCALED_W{1'b0}}; end
        end else begin
            // 3-stage mac-valid / mac-lane shift register to track the deeper
            // weight read pipeline. acc update fires on mac_valid_q3.
            mac_valid_q3 <= mac_valid_q2; mac_lane_q3 <= mac_lane_q2;
            mac_valid_q2 <= mac_valid_q1; mac_lane_q2 <= mac_lane_q1;
            mac_valid_q1 <= 1'b0;
            mul_q        <= $signed(weight_q) * $signed(data_q);
            if (mac_valid_q3) acc[mac_lane_q3] <= $signed(acc[mac_lane_q3]) + $signed(mul_q);
            case (state)
                ST_LOAD: begin
                    valid_out <= 1'b0;
                    if (valid_in && ready_in) begin
                        if (in_beat_idx == IN_BEATS - 1) begin
                            in_beat_idx <= 5'd0; ready_in <= 1'b0;
                            oc_group <= 7'd0; lane_counter <= 2'd0; k_counter <= 10'd0;
                            mac_done_issuing <= 1'b0; drain_cnt <= 2'd0;
                            for (i = 0; i < MP; i = i + 1) acc[i] <= {ACC_W{1'b0}};
                            state <= ST_RUNNING;
                        end else begin in_beat_idx <= in_beat_idx + 5'd1; end
                    end
                end
                ST_RUNNING: begin
                    if (!mac_done_issuing) begin
                        mac_valid_q1 <= 1'b1; mac_lane_q1 <= lane_counter;
                        if (k_counter == K_TOTAL - 1) begin
                            k_counter <= 10'd0;
                            if (lane_counter == MP - 1) mac_done_issuing <= 1'b1;
                            else lane_counter <= lane_counter + 2'd1;
                        end else k_counter <= k_counter + 10'd1;
                    end else begin
                        // Drain: 4 cycles to clear the 3-stage MAC pipeline. On
                        // the final drain cycle (drain_cnt==3) fold the old
                        // ST_BIAS work so per-pass = MP*K_TOTAL + 6 = 3846.
                        if (drain_cnt == 2'd3) begin
                            for (lane = 0; lane < MP; lane = lane + 1)
                                biased[lane] <= $signed(acc[lane]) + $signed(biases[oc_group*MP + lane]);
                            state <= ST_SCALE;
                        end else drain_cnt <= drain_cnt + 2'd1;
                    end
                end
                ST_SCALE: begin
                    for (lane = 0; lane < MP; lane = lane + 1) scaled[lane] <= $signed(biased[lane]) * $signed(SCALE_MULT_CONST);
                    state <= ST_OUTPUT;
                end
                ST_OUTPUT: begin
                    // out_pack_mem write happens in the dedicated array block above.
                    if (oc_group == OC_PASSES - 1) begin
                        valid_out <= 1'b1; // [INVARIANT:VALID_OUT_LATENCY]
                        for (bi = 0; bi < CHANNEL_TILE; bi = bi + 1)
                            data_out[bi*8 +: 8] <= out_pack_mem[bi];
                        out_beat_idx <= 4'd1; state <= ST_EMIT;
                    end else begin
                        oc_group <= oc_group + 7'd1; lane_counter <= 2'd0; k_counter <= 10'd0;
                        mac_done_issuing <= 1'b0; drain_cnt <= 2'd0;
                        for (i = 0; i < MP; i = i + 1) acc[i] <= {ACC_W{1'b0}};
                        state <= ST_RUNNING;
                    end
                end
                ST_EMIT: begin
                    valid_out <= 1'b1;
                    for (bi = 0; bi < CHANNEL_TILE; bi = bi + 1)
                        data_out[bi*8 +: 8] <= out_pack_mem[out_beat_idx*CHANNEL_TILE + bi];
                    if (out_beat_idx == OUT_BEATS - 1) begin
                        out_beat_idx <= 4'd0;
                        if (active_emit_count == TOTAL_PIXELS - 1) begin
                            active_emit_count <= 6'd0; pixel_row <= 3'd0; pixel_col <= 3'd0;
                        end else begin
                            active_emit_count <= active_emit_count + 6'd1;
                            if (pixel_col == OW - 1) begin pixel_col <= 3'd0; pixel_row <= pixel_row + 3'd1; end
                            else pixel_col <= pixel_col + 3'd1;
                        end
                        ready_in <= 1'b1; state <= ST_LOAD; // [INVARIANT:READY_IN_GATING]
                    end else out_beat_idx <= out_beat_idx + 4'd1;
                end
                default: state <= ST_LOAD;
            endcase
        end
    end
endmodule
