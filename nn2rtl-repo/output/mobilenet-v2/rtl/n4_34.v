`timescale 1ns / 1ps

// n4_34 — ReLU6 on tiled-streaming contract. [ROM-requant variant: 0 DSP]
// IC = OC = 960, bus = 256 bits, channel_tile = 32.
// BEATS_PER_PIXEL = ceil(960 * 8 / 256) = 30.
// Latency contract: first valid_out fires BEATS_PER_PIXEL (=30) cycles
// after first valid_in. Achieved by overlapping the last input-beat
// capture with the first output-beat emission in the same posedge.
//
// Datapath per channel (per the 06_relu.md ReLU6 spec):
//   in_byte    = $signed(beat_buf[idx][ch*8 +: 8])
//   relu_byte  = max(in_byte, 0)              // strictly 0..127
//   scaled     = relu_byte * SCALE_MULT_CONST  (signed)
//   v_tmp      = (scaled + sign_aware_round) >>> SCALE_SHIFT
//   data_out   = clamp(v_tmp, -128, 127)
//
// ROM optimization: relu_byte is always in [0,127], and SCALE_MULT/SCALE_SHIFT
// are compile-time constants, so the entire requant pipeline is a pure function
// of the 7-bit relu_byte. Precompute a 128-entry x 8-bit ROM (requant_rom)
// from the EXACT requant expression and replace the per-channel multiply with
// a parallel ROM lookup. This removes the DSP multipliers while remaining
// byte-exact. The ROM is small (128x8) and maps to LUT/distributed RAM.
//
// scale_factor = 0.9109575748443604 -> SCALE_MULT=32'd15439, SCALE_SHIFT=5'd12.

module n4_34 #(
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

    localparam integer SCALE_MULT      = 32'd15439;
    localparam integer SCALE_SHIFT     = 5'd12;
    localparam integer SCALE_CONST_W   = 16;
    localparam integer SCALED_W        = 8 + SCALE_CONST_W;

    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST =
        SCALE_MULT[SCALE_CONST_W-1:0];

    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};

    // ---------------------------------------------------------------------
    // Precomputed requant ROM. requant_rom[x] holds the EXACT output byte of
    // the original multiply/round/shift/clamp pipeline for relu_byte == x,
    // for every possible post-ReLU input x in [0,127]. 0 DSP, no multiplier.
    // (* rom_style *) hints the tool toward distributed/block ROM inference.
    // ---------------------------------------------------------------------
    (* rom_style = "distributed" *)
    reg [7:0] requant_rom [0:127];

    integer                   ri;
    reg signed [SCALED_W-1:0] rom_scaled;
    reg signed [SCALED_W-1:0] rom_vtmp;
    initial begin
        for (ri = 0; ri < 128; ri = ri + 1) begin
            // relu_byte == ri (0..127), always non-negative.
            rom_scaled = $signed(ri[7:0]) * SCALE_MULT_CONST;
            // [INVARIANT:ROUNDING]
            rom_vtmp   = (rom_scaled +
                          (rom_scaled[SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                                  : SCALE_ROUND_HALF)
                         ) >>> SCALE_SHIFT;
            requant_rom[ri] =
                (rom_vtmp >  24'sd127)  ?  8'sd127 :
                (rom_vtmp < -24'sd128)  ? -8'sd128 :
                rom_vtmp[7:0];
        end
    end

    reg [BEAT_WIDTH_BITS-1:0] beat_buf [0:BEATS_PER_PIXEL-1];
    reg [COUNT_W-1:0]         in_beat_count;
    reg [COUNT_W-1:0]         out_beat_count;
    reg                       sending;

    integer                   ch;
    integer                   i;
    reg signed [7:0]          in_byte;
    reg              [6:0]    relu_idx;

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
                            in_byte  = $signed(beat_buf[0][ch*8 +: 8]);
                            // relu(max(in,0)) -> 0..127 ROM index
                            relu_idx = (in_byte > 8'sd0) ? in_byte[6:0] : 7'd0;
                            data_out[ch*8 +: 8] <= requant_rom[relu_idx];
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
                    in_byte  = $signed(beat_buf[out_beat_count][ch*8 +: 8]);
                    // relu(max(in,0)) -> 0..127 ROM index
                    relu_idx = (in_byte > 8'sd0) ? in_byte[6:0] : 7'd0;
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
                            in_byte  = $signed(beat_buf[0][ch*8 +: 8]);
                            // relu(max(in,0)) -> 0..127 ROM index
                            relu_idx = (in_byte > 8'sd0) ? in_byte[6:0] : 7'd0;
                            data_out[ch*8 +: 8] <= requant_rom[relu_idx];
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
                        in_byte  = $signed(beat_buf[out_beat_count][ch*8 +: 8]);
                        // relu(max(in,0)) -> 0..127 ROM index
                        relu_idx = (in_byte > 8'sd0) ? in_byte[6:0] : 7'd0;
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
