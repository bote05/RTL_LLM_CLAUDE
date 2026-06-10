`timescale 1ns / 1ps

// n4_33 — ReLU6 on tiled-streaming contract.
// IC = OC = 960, bus = 256 bits, channel_tile = 32.
// BEATS_PER_PIXEL = ceil(960 * 8 / 256) = 30.
// Latency contract: first valid_out fires BEATS_PER_PIXEL (=30) cycles
// after first valid_in. Achieved by overlapping the last input-beat
// capture with the first output-beat emission in the same posedge.
//
// Datapath per channel (per the 06_relu.md ReLU6 spec):
//   in_byte    = $signed(beat_buf[idx][ch*8 +: 8])
//   relu_byte  = max(in_byte, 0)
//   scaled     = relu_byte * SCALE_MULT_CONST  (signed)
//   v_tmp      = (scaled + sign_aware_round) >>> SCALE_SHIFT
//   data_out   = clamp(v_tmp, -128, 127)
//
// scale_factor = 0.9109575748443604 -> SCALE_MULT=32'd19497, SCALE_SHIFT=5'd14.
//
// DSP-ELIMINATION (byte-exact):
//   relu_byte is strictly in 0..127 (post-ReLU of a signed INT8, max(x,0)).
//   SCALE_MULT_CONST / SCALE_SHIFT are compile-time constants, so the entire
//   requant chain (multiply + sign-aware round + arithmetic shift + clamp) is
//   a pure function of the 7-bit relu_byte.  We precompute it into a 128-entry
//   8-bit ROM (REQUANT_ROM) at elaboration time using the *identical*
//   expression, then index it with relu_byte.  This removes every per-channel
//   multiplier (0 DSP) while keeping the FSM, latency, and bit-exact result.

module n4_33 #(
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

    localparam integer SCALE_MULT      = 32'd19497;
    localparam integer SCALE_SHIFT     = 5'd14;
    localparam integer SCALE_CONST_W   = 16;
    localparam integer SCALED_W        = 8 + SCALE_CONST_W;

    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST =
        SCALE_MULT[SCALE_CONST_W-1:0];

    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};

    // ---------------------------------------------------------------------
    // Requant ROM: rom[x] = clamp( (x*SCALE_MULT_CONST + round) >>> SHIFT )
    // for x in 0..127.  relu_byte can never exceed 127 (it is max(int8,0)),
    // so a 128-entry table covers the full input domain exactly.
    // distributed -> realized in LUTs (no DSP, no BRAM), 32 parallel reads.
    // ---------------------------------------------------------------------
    (* rom_style = "distributed" *)
    reg [7:0] REQUANT_ROM [0:127];

    integer                   k;
    reg signed [SCALED_W-1:0] rom_scaled;
    reg signed [SCALED_W-1:0] rom_vtmp;
    initial begin
        for (k = 0; k < 128; k = k + 1) begin
            // relu_byte = k (k is already the post-ReLU non-negative value).
            rom_scaled = $signed(k[7:0]) * SCALE_MULT_CONST;
            // [INVARIANT:ROUNDING]
            rom_vtmp   = (rom_scaled +
                          (rom_scaled[SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                                  : SCALE_ROUND_HALF)
                         ) >>> SCALE_SHIFT;
            REQUANT_ROM[k] =
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
    reg        [6:0]          relu_idx;

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
                            // relu_byte = max(in_byte, 0); index is 0..127.
                            relu_idx  = (in_byte > 8'sd0) ? in_byte[6:0] : 7'd0;
                            data_out[ch*8 +: 8] <= REQUANT_ROM[relu_idx];
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
                    // relu_byte = max(in_byte, 0); index is 0..127.
                    relu_idx  = (in_byte > 8'sd0) ? in_byte[6:0] : 7'd0;
                    data_out[ch*8 +: 8] <= REQUANT_ROM[relu_idx];
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
                            // relu_byte = max(in_byte, 0); index is 0..127.
                            relu_idx  = (in_byte > 8'sd0) ? in_byte[6:0] : 7'd0;
                            data_out[ch*8 +: 8] <= REQUANT_ROM[relu_idx];
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
                        // relu_byte = max(in_byte, 0); index is 0..127.
                        relu_idx  = (in_byte > 8'sd0) ? in_byte[6:0] : 7'd0;
                        data_out[ch*8 +: 8] <= REQUANT_ROM[relu_idx];
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
