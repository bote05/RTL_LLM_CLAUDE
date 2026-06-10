`timescale 1ns / 1ps

// n4_29 — ReLU6 on tiled-streaming contract.
// IC = OC = 960, bus = 256 bits, channel_tile = 32.
// BEATS_PER_PIXEL = ceil(960 * 8 / 256) = 30.
// Latency contract: first valid_out fires BEATS_PER_PIXEL (=30) cycles
// after first valid_in. Achieved by overlapping the last input-beat
// capture with the first output-beat emission in the same posedge.
//
// Datapath per channel (per the 06_relu.md ReLU6 spec):
//   in_byte    = $signed(beat_buf[idx][ch*8 +: 8])
//   relu_byte  = max(in_byte, 0)                       // 0..127
//   scaled     = relu_byte * SCALE_MULT_CONST  (signed)
//   v_tmp      = (scaled + sign_aware_round) >>> SCALE_SHIFT
//   data_out   = clamp(v_tmp, -128, 127)
//
// scale_factor = 0.9109575748443604 -> SCALE_MULT=29850, SCALE_SHIFT=15.
//
// DSP-ELIMINATION: relu_byte is strictly in [0,127] (post-ReLU 7-bit), and
// the requant constants are compile-time. The entire multiply+round+shift+
// clamp chain is therefore a pure function of relu_byte over a 128-entry
// domain. We precompute REQUANT_ROM[relu_byte] = the EXACT final INT8 result
// of the original expression, in an initial loop that evaluates the SAME
// arithmetic. The per-channel multiply (the DSPs) is replaced by a parallel
// ROM lookup REQUANT_ROM[relu_byte]. FSM, handshake, and latency unchanged.

module n4_29 #(
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
    localparam integer BEATS_PER_PIXEL = 30;
    localparam integer COUNT_W         = 5;

    localparam integer SCALE_MULT      = 29850;
    localparam integer SCALE_SHIFT     = 15;
    localparam integer SCALE_CONST_W   = 16;
    localparam integer SCALED_W        = 8 + SCALE_CONST_W;

    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST =
        SCALE_MULT[SCALE_CONST_W-1:0];

    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};

    // ----------------------------------------------------------------------
    // Precomputed requant ROM. Index = relu_byte (0..127). Each entry holds
    // the exact clamped INT8 output of the original requant expression. This
    // replaces the per-channel signed multiply (the DSP usage) with a LUT.
    // ----------------------------------------------------------------------
    (* rom_style = "distributed" *)
    reg [7:0] REQUANT_ROM [0:127];

    integer                   r;
    reg signed [SCALED_W-1:0] rom_scaled;
    reg signed [SCALED_W-1:0] rom_v_tmp;
    initial begin
        for (r = 0; r < 128; r = r + 1) begin
            // relu_byte == r (always >= 0 over this domain).
            rom_scaled = $signed(r[7:0]) * SCALE_MULT_CONST;
            // [INVARIANT:ROUNDING]
            rom_v_tmp  = (rom_scaled +
                          (rom_scaled[SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                                  : SCALE_ROUND_HALF)
                         ) >>> SCALE_SHIFT;
            REQUANT_ROM[r] =
                (rom_v_tmp >  24'sd127)  ?  8'sd127 :
                (rom_v_tmp < -24'sd128)  ? -8'sd128 :
                rom_v_tmp[7:0];
        end
    end

    reg [BEAT_WIDTH_BITS-1:0] beat_buf [0:BEATS_PER_PIXEL-1];
    reg [COUNT_W-1:0]         in_beat_count;
    reg [COUNT_W-1:0]         out_beat_count;
    reg                       sending;

    integer                   ch;
    integer                   i;
    reg signed [7:0]          in_byte;
    reg        [6:0]          relu_byte;

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
                            in_byte   = $signed(beat_buf[0][ch*8 +: 8]);
                            relu_byte = (in_byte > 8'sd0) ? in_byte[6:0] : 7'd0;
                            // [INVARIANT:ROUNDING] requant via precomputed ROM
                            data_out[ch*8 +: 8] <= REQUANT_ROM[relu_byte];
                        end
                        // [INVARIANT:VALID_OUT_LATENCY]
                        valid_out      <= 1'b1;
                        out_beat_count <= {{(COUNT_W-1){1'b0}}, 1'b1};
                    end else begin
                        in_beat_count <= in_beat_count +
                                         {{(COUNT_W-1){1'b0}}, 1'b1};
                    end
                end
            end else begin
                for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                    in_byte   = $signed(beat_buf[out_beat_count][ch*8 +: 8]);
                    relu_byte = (in_byte > 8'sd0) ? in_byte[6:0] : 7'd0;
                    // [INVARIANT:ROUNDING] requant via precomputed ROM
                    data_out[ch*8 +: 8] <= REQUANT_ROM[relu_byte];
                end
                // [INVARIANT:VALID_OUT_LATENCY]
                valid_out <= 1'b1;
                if (out_beat_count == BEATS_PER_PIXEL - 1) begin
                    sending        <= 1'b0;
                    out_beat_count <= {COUNT_W{1'b0}};
                    // [INVARIANT:READY_IN_GATING]
                    ready_in       <= 1'b1;
                end else begin
                    out_beat_count <= out_beat_count +
                                      {{(COUNT_W-1){1'b0}}, 1'b1};
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
                            in_byte   = $signed(beat_buf[0][ch*8 +: 8]);
                            relu_byte = (in_byte > 8'sd0) ? in_byte[6:0] : 7'd0;
                            // [INVARIANT:ROUNDING] requant via precomputed ROM
                            data_out[ch*8 +: 8] <= REQUANT_ROM[relu_byte];
                        end
                        // [INVARIANT:VALID_OUT_LATENCY]
                        valid_out      <= 1'b1;
                        out_beat_count <= {{(COUNT_W-1){1'b0}}, 1'b1};
                    end else begin
                        in_beat_count <= in_beat_count +
                                         {{(COUNT_W-1){1'b0}}, 1'b1};
                    end
                end
            end else begin
                // [ELASTIC] advance the output stream only when the
                // downstream accepts the current beat; hold otherwise.
                if (out_ready_in) begin
                    for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                        in_byte   = $signed(beat_buf[out_beat_count][ch*8 +: 8]);
                        relu_byte = (in_byte > 8'sd0) ? in_byte[6:0] : 7'd0;
                        // [INVARIANT:ROUNDING] requant via precomputed ROM
                        data_out[ch*8 +: 8] <= REQUANT_ROM[relu_byte];
                    end
                    // [INVARIANT:VALID_OUT_LATENCY]
                    valid_out <= 1'b1;
                    if (out_beat_count == BEATS_PER_PIXEL - 1) begin
                        sending        <= 1'b0;
                        out_beat_count <= {COUNT_W{1'b0}};
                        // [INVARIANT:READY_IN_GATING]
                        ready_in       <= 1'b1;
                    end else begin
                        out_beat_count <= out_beat_count +
                                          {{(COUNT_W-1){1'b0}}, 1'b1};
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
