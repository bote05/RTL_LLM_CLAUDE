// node_relu_24 — tiled-streaming (channel_tiled, channel_tile=32)
// op_type: relu
// Bus: data_in/data_out = 256 bits = 32 channels per beat.
// Total channels = 1024 -> BEATS_PER_PIXEL = 32 beats per logical pixel.
// Latency contract: first valid_out fires exactly 32 cycles after first
// valid_in (base pipeline_latency_cycles=1 + BEATS_PER_PIXEL-1 = 32).
//
// Strategy: buffer one full pixel of 32 beats into a register file, then
// emit 32 ReLU'd beats. The store of the LAST input beat overlaps with the
// emission of the FIRST output beat so the per-vector latency check
// (vector_actual = first_valid_out_cycle - first_valid_in_cycle) lands on
// exactly BEATS_PER_PIXEL = 32.
`timescale 1ns / 1ps

module node_relu_24 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    output reg          ready_in,
    input  wire [255:0] data_in,
    output reg          valid_out,
    input  wire         ready_out,
    output reg  [255:0] data_out
);

    localparam integer IC              = 1024;
    localparam integer OC              = 1024;
    localparam integer CHANNEL_TILE    = 32;
    localparam integer BEAT_WIDTH_BITS = 256;
    localparam integer BEATS_PER_PIXEL = 32; // ceil(IC*8 / BEAT_WIDTH_BITS)
    localparam integer COUNT_W         = 6;  // holds 0..32

    reg [BEAT_WIDTH_BITS-1:0] beat_buf [0:BEATS_PER_PIXEL-1];

    // Tiled-streaming contract beat counters: per-pixel input/output beat
    // indices for the channel-tiled bus.
    reg [COUNT_W-1:0] in_beat_count;
    reg [COUNT_W-1:0] out_beat_count;
    reg               sending;

    integer ch;
    integer i;
    reg signed [7:0] tmp_byte;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ready_in       <= 1'b1;                           // [INVARIANT:READY_IN_GATING]
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
                ready_in <= 1'b1;                             // [INVARIANT:READY_IN_GATING]
                if (valid_in && ready_in) begin
                    beat_buf[in_beat_count] <= data_in;
                    if (in_beat_count == BEATS_PER_PIXEL - 1) begin
                        // Last input beat: also emit beat 0 of output stream
                        // this same cycle so first_valid_out lands exactly
                        // BEATS_PER_PIXEL cycles after first_valid_in.
                        in_beat_count <= {COUNT_W{1'b0}};
                        sending       <= 1'b1;
                        ready_in      <= 1'b0;                // [INVARIANT:READY_IN_GATING]
                        for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                            tmp_byte = $signed(beat_buf[0][ch*8 +: 8]);
                            data_out[ch*8 +: 8] <= (tmp_byte > 8'sd0) ? tmp_byte : 8'sd0;
                        end
                        valid_out      <= 1'b1;               // [INVARIANT:VALID_OUT_LATENCY]
                        out_beat_count <= 6'd1;
                    end else begin
                        in_beat_count <= in_beat_count + 6'd1;
                    end
                end
            end else begin
                if (ready_out) begin
                // Stream remaining BEATS_PER_PIXEL-1 ReLU'd beats.
                for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                    tmp_byte = $signed(beat_buf[out_beat_count][ch*8 +: 8]);
                    data_out[ch*8 +: 8] <= (tmp_byte > 8'sd0) ? tmp_byte : 8'sd0;
                end
                valid_out <= 1'b1;                            // [INVARIANT:VALID_OUT_LATENCY]
                if (out_beat_count == BEATS_PER_PIXEL - 1) begin
                    sending        <= 1'b0;
                    out_beat_count <= {COUNT_W{1'b0}};
                    ready_in       <= 1'b1;                   // [INVARIANT:READY_IN_GATING]
                end else begin
                    out_beat_count <= out_beat_count + 6'd1;
                    end
                end else begin
                    valid_out <= 1'b1;
                end
            end
        end
    end
endmodule
