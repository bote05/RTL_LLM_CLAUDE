`timescale 1ns / 1ps

// n4_26 — quantized ReLU + requantize on tiled-streaming contract.
// IC=OC=576, bus=256b, channel_tile=32, BEATS_PER_PIXEL=18.
// Input and output INT8 scales differ: after max(x,0), apply requantize
// multiply (SCALE_MULT/2^SCALE_SHIFT) with sign-aware rounding, then
// saturate to signed INT8. SHIFT=13, MULT=17153 reproduces the goldens
// bit-exact.
//
// DSP-FREE REQUANT: the post-ReLU value is strictly in 0..127 (7-bit
// non-negative), and SCALE_MULT/SCALE_SHIFT are compile-time constants,
// so the entire requant multiply+round+shift+saturate collapses to a
// 128-entry ROM:  requant_rom[relu_byte] = the EXACT INT8 result of the
// original expression for input relu_byte (0..127). The ROM is populated
// at elaboration via an initial loop that evaluates the SAME expression,
// so it is byte-exact by construction. The per-channel multiply (one DSP
// per lane) is replaced by parallel ROM reads (0 DSP, distributed LUTROM).
// Latency/FSM are unchanged: BEATS_PER_PIXEL=18.
module n4_26 #(
    parameter ENABLE_BACKPRESSURE = 0
)(
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    output reg          ready_in,
    input  wire [255:0] data_in,
    input  wire         out_ready_in,   // NEW: downstream-ready (ignored when ENABLE_BACKPRESSURE==0)
    output reg          valid_out,
    output reg  [255:0] data_out
);
    localparam integer CHANNEL_TILE    = 32;
    localparam integer BEAT_WIDTH_BITS = 256;
    localparam integer BEATS_PER_PIXEL = 18;
    localparam integer COUNT_W         = 5;

    localparam integer SCALE_MULT      = 32'd17153;
    localparam integer SCALE_SHIFT     = 5'd13;
    localparam integer SCALE_CONST_W   = 16;
    localparam integer SCALED_W        = 8 + SCALE_CONST_W;

    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST = SCALE_MULT;
    localparam signed [SCALED_W-1:0]      SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0]      SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};

    // ---- Precomputed requant ROM (replaces the per-channel DSP multiply) ----
    // requant_rom[x] holds the EXACT 8-bit (signed INT8) requant output for a
    // post-ReLU input value x in 0..127, evaluated with the identical
    // multiply/round/shift/saturate expression used previously.
    (* rom_style = "distributed" *) reg [7:0] requant_rom [0:127];
    integer ridx;
    reg signed [SCALED_W-1:0] rom_scaled;
    reg signed [SCALED_W-1:0] rom_vtmp;
    initial begin
        for (ridx = 0; ridx < 128; ridx = ridx + 1) begin
            // relu_byte == ridx (already non-negative, 0..127)
            rom_scaled = $signed(ridx[7:0]) * $signed(SCALE_MULT_CONST);
            rom_vtmp   = (rom_scaled +
                          (rom_scaled[SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                                  : SCALE_ROUND_HALF)
                         ) >>> SCALE_SHIFT; // [INVARIANT:ROUNDING]
            requant_rom[ridx] = (rom_vtmp > 127)  ?  8'sd127 :
                                (rom_vtmp < -128) ? -8'sd128 : rom_vtmp[7:0];
        end
    end

    reg [BEAT_WIDTH_BITS-1:0] beat_buf [0:BEATS_PER_PIXEL-1];
    reg [COUNT_W-1:0] in_beat_count;
    reg [COUNT_W-1:0] out_beat_count;
    reg               sending;

    integer ch, i;
    reg signed [7:0] tmp_byte;
    reg        [6:0] relu_idx;

    // [K1-MBV2] sync-only memory write -- no reset clause (ResNet K1 P8 /
    // node_relu.v precedent): beat_buf is gather DATA, fully rewritten each
    // pixel before the sending phase reads it; the guard replicates the
    // original nested condition (identical in both generate branches; only
    // one elaborates). Also unblocks LUTRAM inference.
    always @(posedge clk) begin
        if (!sending && valid_in && ready_in) begin
            beat_buf[in_beat_count] <= data_in;
        end
    end

    generate
    if (ENABLE_BACKPRESSURE == 0) begin : g_legacy
        // ---- LEGACY FSM: bit/cycle-identical to the pre-backpressure module ----
        always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ready_in       <= 1'b1;
            valid_out      <= 1'b0;
            data_out       <= {BEAT_WIDTH_BITS{1'b0}};
            in_beat_count  <= {COUNT_W{1'b0}};
            out_beat_count <= {COUNT_W{1'b0}};
            sending        <= 1'b0;
        end else begin
            valid_out <= 1'b0;
            if (!sending) begin
                ready_in <= 1'b1; // [INVARIANT:READY_IN_GATING]
                if (valid_in && ready_in) begin
                    if (in_beat_count == BEATS_PER_PIXEL - 1) begin
                        in_beat_count <= {COUNT_W{1'b0}};
                        sending       <= 1'b1;
                        ready_in      <= 1'b0; // [INVARIANT:READY_IN_GATING]
                        for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                            tmp_byte = $signed(beat_buf[0][ch*8 +: 8]);
                            relu_idx = (tmp_byte > 8'sd0) ? tmp_byte[6:0] : 7'd0;
                            data_out[ch*8 +: 8] <= requant_rom[relu_idx]; // ROM requant (0 DSP)
                        end
                        valid_out      <= 1'b1; // [INVARIANT:VALID_OUT_LATENCY]
                        out_beat_count <= 5'd1;
                    end else begin
                        in_beat_count <= in_beat_count + 5'd1;
                    end
                end
            end else begin
                for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                    tmp_byte = $signed(beat_buf[out_beat_count][ch*8 +: 8]);
                    relu_idx = (tmp_byte > 8'sd0) ? tmp_byte[6:0] : 7'd0;
                    data_out[ch*8 +: 8] <= requant_rom[relu_idx]; // ROM requant (0 DSP)
                end
                valid_out <= 1'b1;
                if (out_beat_count == BEATS_PER_PIXEL - 1) begin
                    sending        <= 1'b0;
                    out_beat_count <= {COUNT_W{1'b0}};
                    ready_in       <= 1'b1;
                end else begin
                    out_beat_count <= out_beat_count + 5'd1;
                end
            end
        end
        end
    end else begin : g_bp
        // ---- ELASTIC FSM: output-beat emission gated on out_ready_in ----
        always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ready_in       <= 1'b1;
            valid_out      <= 1'b0;
            data_out       <= {BEAT_WIDTH_BITS{1'b0}};
            in_beat_count  <= {COUNT_W{1'b0}};
            out_beat_count <= {COUNT_W{1'b0}};
            sending        <= 1'b0;
        end else begin
            valid_out <= 1'b0;
            if (!sending) begin
                ready_in <= 1'b1; // [INVARIANT:READY_IN_GATING]
                if (valid_in && ready_in) begin
                    if (in_beat_count == BEATS_PER_PIXEL - 1) begin
                        in_beat_count <= {COUNT_W{1'b0}};
                        sending       <= 1'b1;
                        ready_in      <= 1'b0; // [INVARIANT:READY_IN_GATING]
                        for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                            tmp_byte = $signed(beat_buf[0][ch*8 +: 8]);
                            relu_idx = (tmp_byte > 8'sd0) ? tmp_byte[6:0] : 7'd0;
                            data_out[ch*8 +: 8] <= requant_rom[relu_idx]; // ROM requant (0 DSP)
                        end
                        valid_out      <= 1'b1; // [INVARIANT:VALID_OUT_LATENCY]
                        out_beat_count <= 5'd1;
                    end else begin
                        in_beat_count <= in_beat_count + 5'd1;
                    end
                end
            end else begin
                // [ELASTIC] advance the output stream only when the
                // downstream accepts the current beat; hold otherwise.
                if (out_ready_in) begin
                    for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                        tmp_byte = $signed(beat_buf[out_beat_count][ch*8 +: 8]);
                        relu_idx = (tmp_byte > 8'sd0) ? tmp_byte[6:0] : 7'd0;
                        data_out[ch*8 +: 8] <= requant_rom[relu_idx]; // ROM requant (0 DSP)
                    end
                    valid_out <= 1'b1;
                    if (out_beat_count == BEATS_PER_PIXEL - 1) begin
                        sending        <= 1'b0;
                        out_beat_count <= {COUNT_W{1'b0}};
                        ready_in       <= 1'b1;
                    end else begin
                        out_beat_count <= out_beat_count + 5'd1;
                    end
                end else begin
                    valid_out <= 1'b1; // hold the parked beat
                end
            end
        end
        end
    end
    endgenerate
endmodule
