`timescale 1ns / 1ps

// n4_31 — ReLU6 on tiled-streaming contract.
// IC = OC = 960, bus = 256 bits, channel_tile = 32.
// BEATS_PER_PIXEL = ceil(960 * 8 / 256) = 30.
// Latency contract: first valid_out fires BEATS_PER_PIXEL (=30) cycles
// after first valid_in. Achieved by overlapping the last input-beat
// capture with the first output-beat emission in the same posedge.
//
// Datapath per channel (per the 06_relu.md ReLU6 spec):
//   in_byte    = $signed(beat_buf[idx][ch*8 +: 8])
//   relu_byte  = max(in_byte, 0)            -> always in [0, 127]
//   scaled     = relu_byte * SCALE_MULT_CONST  (signed)
//   v_tmp      = (scaled + sign_aware_round) >>> SCALE_SHIFT
//   data_out   = clamp(v_tmp, -128, 127)
//
// DSP-ELIMINATION (ROM requant):
//   relu_byte is strictly in [0, 127] (7-bit non-negative) and the requant
//   scale is a per-tensor compile-time constant, so the entire
//   multiply+round+shift+clamp pipeline collapses to a 128-entry lookup
//   requant_rom[relu_byte]. The ROM is populated in an `initial` block by
//   evaluating the SAME requant expression for every input 0..127, so the
//   result is byte-exact with the prior multiply datapath but uses ZERO DSPs.
//   The ROM is read in parallel across all CHANNEL_TILE lanes (LUT/BRAM ROM).
//
// scale: SCALE_MULT=32'd21967, SCALE_SHIFT=5'd14.

module n4_31 #(
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

    localparam integer SCALE_MULT      = 32'd21967;
    localparam integer SCALE_SHIFT     = 5'd14;
    localparam integer SCALE_CONST_W   = 16;
    localparam integer SCALED_W        = 8 + SCALE_CONST_W;

    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST =
        SCALE_MULT[SCALE_CONST_W-1:0];

    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};

    reg [BEAT_WIDTH_BITS-1:0] beat_buf [0:BEATS_PER_PIXEL-1];
    reg [COUNT_W-1:0]         in_beat_count;
    reg [COUNT_W-1:0]         out_beat_count;
    reg                       sending;

    integer                   ch;
    integer                   i;

    // ---------------------------------------------------------------------
    // Requant ROM: rom[relu_byte] = clamp((relu_byte*SCALE_MULT_CONST +
    //   sign_aware_round) >>> SCALE_SHIFT, -128, 127).  relu_byte in [0,127].
    // Populated by evaluating the IDENTICAL expression -> byte-exact, 0 DSP.
    // ---------------------------------------------------------------------
    (* rom_style = "distributed" *)
    reg [7:0] requant_rom [0:127];

    integer                   r_idx;
    reg signed [7:0]          r_relu;
    reg signed [SCALED_W-1:0] r_scaled;
    reg signed [SCALED_W-1:0] r_vtmp;
    initial begin
        for (r_idx = 0; r_idx < 128; r_idx = r_idx + 1) begin
            r_relu   = r_idx[7:0];                 // 0..127, non-negative
            r_scaled = $signed(r_relu) * SCALE_MULT_CONST;
            // [INVARIANT:ROUNDING]
            r_vtmp   = (r_scaled +
                        (r_scaled[SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                              : SCALE_ROUND_HALF)
                       ) >>> SCALE_SHIFT;
            requant_rom[r_idx] =
                (r_vtmp >  24'sd127)  ?  8'sd127 :
                (r_vtmp < -24'sd128)  ? -8'sd128 :
                r_vtmp[7:0];
        end
    end

    // Combinational ReLU + ROM lookup for one channel of a given beat.
    reg signed [7:0] in_byte;
    reg        [6:0] relu_idx;

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
            for (i = 0; i < BEATS_PER_PIXEL; i = i + 1)
                beat_buf[i] <= {BEAT_WIDTH_BITS{1'b0}};
        end else begin
            valid_out <= 1'b0;
            if (!sending) begin
                // [INVARIANT:READY_IN_GATING]
                ready_in <= 1'b1;
                if (valid_in && ready_in) begin
                    beat_buf[in_beat_count] <= data_in;
                    if (in_beat_count == BEATS_PER_PIXEL - 1) begin
                        in_beat_count <= {COUNT_W{1'b0}};
                        sending       <= 1'b1;
                        // [INVARIANT:READY_IN_GATING]
                        ready_in      <= 1'b0;
                        for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                            in_byte  = $signed(beat_buf[0][ch*8 +: 8]);
                            // relu: max(in_byte, 0) -> 7-bit ROM index
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
            for (i = 0; i < BEATS_PER_PIXEL; i = i + 1)
                beat_buf[i] <= {BEAT_WIDTH_BITS{1'b0}};
        end else begin
            valid_out <= 1'b0;
            if (!sending) begin
                // [INVARIANT:READY_IN_GATING]
                ready_in <= 1'b1;
                if (valid_in && ready_in) begin
                    beat_buf[in_beat_count] <= data_in;
                    if (in_beat_count == BEATS_PER_PIXEL - 1) begin
                        in_beat_count <= {COUNT_W{1'b0}};
                        sending       <= 1'b1;
                        // [INVARIANT:READY_IN_GATING]
                        ready_in      <= 1'b0;
                        for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                            in_byte  = $signed(beat_buf[0][ch*8 +: 8]);
                            // relu: max(in_byte, 0) -> 7-bit ROM index
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
