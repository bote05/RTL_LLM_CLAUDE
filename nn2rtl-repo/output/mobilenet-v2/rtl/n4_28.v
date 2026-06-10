`timescale 1ns / 1ps

// n4_28 — quantized ReLU6 on tiled-streaming contract.
// IC=OC=576, bus=256b, channel_tile=32, BEATS_PER_PIXEL=18.
// clip_max=6 ⇒ this is a ReLU6 layer: the upstream conv's output_scale
// is wider than this layer's output_scale, so the INT8 stream must be
// requantized by scale_factor = input_scale/output_scale = 0.8950890700022379.
// SCALE_MULT/SCALE_SHIFT = 14665/14 ≈ 0.8950806 (rel err 9.5e-6).
// Saturation at INT8 +127 is the in-domain ReLU6 ceiling.
// Latency: first valid_out fires BEATS_PER_PIXEL=18 cycles after the
// first valid_in by overlapping the last-input-beat capture with the
// first output-beat emission in the same cycle.
module n4_28 #(
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

    // ReLU6 fixed-point requantize constants.
    localparam integer        SCALE_SHIFT      = 14;
    localparam signed [15:0]  SCALE_MULT_CONST = 16'sd14665;
    localparam signed [31:0]  SCALE_HALF       = 32'sd8192; // 1 << (SHIFT-1)

    reg [BEAT_WIDTH_BITS-1:0] beat_buf [0:BEATS_PER_PIXEL-1];
    reg [COUNT_W-1:0] in_beat_count;
    reg [COUNT_W-1:0] out_beat_count;
    reg               sending;

    integer            ch, i;
    reg signed [7:0]   in_byte;
    reg signed [7:0]   relu_byte;

    // -----------------------------------------------------------------------
    // Requant ROM. The post-ReLU input domain is exactly 0..127 (7-bit
    // non-negative INT8), and SCALE_MULT/SCALE_SHIFT are compile-time
    // constants, so the full requant (multiply + round + arithmetic shift +
    // INT8 saturation) is precomputable into a 128-entry x 8-bit table.
    // Replacing the per-channel multiply with rom[relu_byte] is byte-exact
    // and eliminates the DSP multiplier (this becomes LUT/BRAM ROM, 0 DSP).
    // The initial loop below evaluates the SAME expression that the original
    // datapath used, so equality is structural, not numeric coincidence.
    // -----------------------------------------------------------------------
    (* rom_style = "distributed" *) reg signed [7:0] requant_rom [0:127];
    integer            r;
    reg signed [31:0]  rom_scaled;
    reg signed [31:0]  rom_shifted;
    initial begin
        for (r = 0; r < 128; r = r + 1) begin
            rom_scaled  = r * SCALE_MULT_CONST;          // relu_byte * SCALE_MULT
            rom_shifted = (rom_scaled + SCALE_HALF) >>> SCALE_SHIFT;
            requant_rom[r] = (rom_shifted > 32'sd127)  ?  8'sd127 :
                             (rom_shifted < -32'sd128) ? -8'sd128 :
                             rom_shifted[7:0];
        end
    end

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
                        // Overlap last-input capture with first output beat.
                        // ROM lookup replaces the per-channel multiply/shift/sat.
                        // [INVARIANT:ROUNDING]
                        for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                            in_byte   = $signed(beat_buf[0][ch*8 +: 8]);
                            relu_byte = (in_byte > 8'sd0) ? in_byte : 8'sd0;
                            data_out[ch*8 +: 8] <= requant_rom[relu_byte[6:0]];
                        end
                        valid_out      <= 1'b1; // [INVARIANT:VALID_OUT_LATENCY]
                        out_beat_count <= 5'd1;
                    end else begin
                        in_beat_count <= in_beat_count + 5'd1;
                    end
                end
            end else begin
                // ROM lookup replaces the per-channel multiply/shift/sat.
                // [INVARIANT:ROUNDING]
                for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                    in_byte   = $signed(beat_buf[out_beat_count][ch*8 +: 8]);
                    relu_byte = (in_byte > 8'sd0) ? in_byte : 8'sd0;
                    data_out[ch*8 +: 8] <= requant_rom[relu_byte[6:0]];
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
                        // Overlap last-input capture with first output beat.
                        // ROM lookup replaces the per-channel multiply/shift/sat.
                        // [INVARIANT:ROUNDING]
                        for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                            in_byte   = $signed(beat_buf[0][ch*8 +: 8]);
                            relu_byte = (in_byte > 8'sd0) ? in_byte : 8'sd0;
                            data_out[ch*8 +: 8] <= requant_rom[relu_byte[6:0]];
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
                    // ROM lookup replaces the per-channel multiply/shift/sat.
                    // [INVARIANT:ROUNDING]
                    for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                        in_byte   = $signed(beat_buf[out_beat_count][ch*8 +: 8]);
                        relu_byte = (in_byte > 8'sd0) ? in_byte : 8'sd0;
                        data_out[ch*8 +: 8] <= requant_rom[relu_byte[6:0]];
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
