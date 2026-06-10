`timescale 1ns / 1ps

// n4_35 — ReLU6 on tiled-streaming contract.
// IC=1280, channel_tile=32, beat_width=256 bits.
// BEATS_PER_PIXEL = 1280*8/256 = 40, equals expected_latency_cycles.
// Per 06_relu.md: ReLU then requantize via SCALE_MULT/SCALE_SHIFT, clamp to INT8.
// clip_max=6 makes this a ReLU6; scale_factor=4.725078 ~= 9677/2048.
//
// DSP-elimination rewrite: the post-ReLU byte is strictly in [0,127] and the
// requant scale is a compile-time constant, so the requantize result
//   clamp( ((relu_byte*SCALE_MULT) + round) >>> SCALE_SHIFT , -128, 127 )
// is a pure function of a 7-bit input. It is precomputed once into a 128-entry
// 8-bit ROM (requant_rom) in an initial block using the EXACT same expression,
// and the per-channel multiply datapath is replaced by parallel rom[] lookups.
// 0 DSP, byte-exact, identical FSM/latency.

module n4_35 #(
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
    localparam integer BEATS_PER_PIXEL = 40;
    localparam integer COUNT_W         = 6;

    localparam integer SCALE_MULT      = 9677;
    localparam integer SCALE_SHIFT     = 11;
    localparam integer SCALE_CONST_W   = 15;
    localparam integer SCALED_W        = 32;
    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST = 15'sd9677;
    localparam signed [SCALED_W-1:0]      SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0]      SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};

    // ---------------------------------------------------------------------
    // Precomputed requant lookup. Input domain is the post-ReLU byte 0..127.
    // requant_rom[x] = clamp( ((x*SCALE_MULT)+round) >>> SCALE_SHIFT, -128,127 ).
    // Built once with the IDENTICAL expression used by the original multiply
    // datapath, so behaviour is byte-exact. distributed (LUT) ROM -> 0 DSP.
    // ---------------------------------------------------------------------
    (* rom_style = "distributed" *)
    reg [7:0] requant_rom [0:127];

    integer                   r;
    reg signed [SCALED_W-1:0] rom_scaled;
    reg signed [SCALED_W-1:0] rom_rounded;
    initial begin
        for (r = 0; r < 128; r = r + 1) begin
            rom_scaled  = $signed(r[7:0]) * SCALE_MULT_CONST;
            // [INVARIANT:ROUNDING]
            rom_rounded = (rom_scaled +
                           (rom_scaled[SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                                   : SCALE_ROUND_HALF)
                          ) >>> SCALE_SHIFT;
            requant_rom[r] = (rom_rounded > 32'sd127)  ? 8'sd127  :
                             (rom_rounded < -32'sd128) ? -8'sd128 :
                             rom_rounded[7:0];
        end
    end

    reg [BEAT_WIDTH_BITS-1:0] beat_buf [0:BEATS_PER_PIXEL-1];
    reg [COUNT_W-1:0]         in_beat_count;
    reg [COUNT_W-1:0]         out_beat_count;
    reg                       sending;

    integer                   ch, i;
    reg signed [7:0]          tmp_byte;
    reg [6:0]                 relu_idx;

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
                // [INVARIANT:READY_IN_GATING]
                ready_in <= 1'b1;
                if (valid_in && ready_in) begin
                    if (in_beat_count == BEATS_PER_PIXEL - 1) begin
                        in_beat_count <= {COUNT_W{1'b0}};
                        sending       <= 1'b1;
                        // [INVARIANT:READY_IN_GATING]
                        ready_in      <= 1'b0;
                        for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                            tmp_byte = $signed(beat_buf[0][ch*8 +: 8]);
                            // ReLU: clamp to [0,127] -> 7-bit ROM index.
                            relu_idx = (tmp_byte > 8'sd0) ? tmp_byte[6:0] : 7'd0;
                            data_out[ch*8 +: 8] <= requant_rom[relu_idx];
                        end
                        // [INVARIANT:VALID_OUT_LATENCY]
                        valid_out      <= 1'b1;
                        out_beat_count <= 6'd1;
                    end else begin
                        in_beat_count <= in_beat_count + 6'd1;
                    end
                end
            end else begin
                for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                    tmp_byte = $signed(beat_buf[out_beat_count][ch*8 +: 8]);
                    // ReLU: clamp to [0,127] -> 7-bit ROM index.
                    relu_idx = (tmp_byte > 8'sd0) ? tmp_byte[6:0] : 7'd0;
                    data_out[ch*8 +: 8] <= requant_rom[relu_idx];
                end
                // [INVARIANT:VALID_OUT_LATENCY]
                valid_out <= 1'b1;
                if (out_beat_count == BEATS_PER_PIXEL - 1) begin
                    sending        <= 1'b0;
                    out_beat_count <= {COUNT_W{1'b0}};
                    // [INVARIANT:READY_IN_GATING]
                    ready_in       <= 1'b1;
                end else begin
                    out_beat_count <= out_beat_count + 6'd1;
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
                // [INVARIANT:READY_IN_GATING]
                ready_in <= 1'b1;
                if (valid_in && ready_in) begin
                    if (in_beat_count == BEATS_PER_PIXEL - 1) begin
                        in_beat_count <= {COUNT_W{1'b0}};
                        sending       <= 1'b1;
                        // [INVARIANT:READY_IN_GATING]
                        ready_in      <= 1'b0;
                        for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                            tmp_byte = $signed(beat_buf[0][ch*8 +: 8]);
                            // ReLU: clamp to [0,127] -> 7-bit ROM index.
                            relu_idx = (tmp_byte > 8'sd0) ? tmp_byte[6:0] : 7'd0;
                            data_out[ch*8 +: 8] <= requant_rom[relu_idx];
                        end
                        // [INVARIANT:VALID_OUT_LATENCY]
                        valid_out      <= 1'b1;
                        out_beat_count <= 6'd1;
                    end else begin
                        in_beat_count <= in_beat_count + 6'd1;
                    end
                end
            end else begin
                // [ELASTIC] advance the output stream only when the
                // downstream accepts the current beat; hold otherwise.
                if (out_ready_in) begin
                    for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                        tmp_byte = $signed(beat_buf[out_beat_count][ch*8 +: 8]);
                        // ReLU: clamp to [0,127] -> 7-bit ROM index.
                        relu_idx = (tmp_byte > 8'sd0) ? tmp_byte[6:0] : 7'd0;
                        data_out[ch*8 +: 8] <= requant_rom[relu_idx];
                    end
                    // [INVARIANT:VALID_OUT_LATENCY]
                    valid_out <= 1'b1;
                    if (out_beat_count == BEATS_PER_PIXEL - 1) begin
                        sending        <= 1'b0;
                        out_beat_count <= {COUNT_W{1'b0}};
                        // [INVARIANT:READY_IN_GATING]
                        ready_in       <= 1'b1;
                    end else begin
                        out_beat_count <= out_beat_count + 6'd1;
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
