// Module: node_relu_25
// Op: relu  Contract: tiled-streaming  io_mode: channel_tiled
// channel_tile = 32 (32 INT8 channels per 256-bit beat)
// IC = OC = 256  =>  BEATS_PER_PIXEL = 256 / 32 = 8
// Spec hash: relu_256x256_s14x14_i256_o256_iotiled-streaming_tile32
// Latency contract: first valid_out fires BEATS_PER_PIXEL (=8) cycles
// after first valid_in by overlapping last input-beat capture with
// first output-beat emission.
`timescale 1ns / 1ps

module node_relu_25 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    output reg          ready_in,
    input  wire [255:0] data_in,
    output reg          valid_out,
    input  wire         ready_out,
    output reg  [255:0] data_out
);
    localparam integer CHANNEL_TILE    = 32;
    localparam integer BEAT_WIDTH_BITS = 256;
    localparam integer BEATS_PER_PIXEL = 8;
    localparam integer COUNT_W         = 4;

    reg [BEAT_WIDTH_BITS-1:0] beat_buf [0:BEATS_PER_PIXEL-1];
    reg [COUNT_W-1:0] in_beat_count;
    reg [COUNT_W-1:0] out_beat_count;
    reg               sending;

    integer ch, i;
    reg signed [7:0] tmp_byte;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // [INVARIANT:READY_IN_GATING]
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
                ready_in <= 1'b1;
                if (valid_in && ready_in) begin
                    beat_buf[in_beat_count] <= data_in;
                    if (in_beat_count == BEATS_PER_PIXEL - 1) begin
                        in_beat_count <= {COUNT_W{1'b0}};
                        sending       <= 1'b1;
                        // [INVARIANT:READY_IN_GATING]
                        ready_in      <= 1'b0;
                        for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                            tmp_byte = $signed(beat_buf[0][ch*8 +: 8]);
                            data_out[ch*8 +: 8] <= (tmp_byte > 8'sd0) ? tmp_byte : 8'sd0;
                        end
                        // [INVARIANT:VALID_OUT_LATENCY]
                        valid_out      <= 1'b1;
                        out_beat_count <= 4'd1;
                    end else begin
                        in_beat_count <= in_beat_count + 4'd1;
                    end
                end
            end else begin
                if (ready_out) begin
                for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                    tmp_byte = $signed(beat_buf[out_beat_count][ch*8 +: 8]);
                    data_out[ch*8 +: 8] <= (tmp_byte > 8'sd0) ? tmp_byte : 8'sd0;
                end
                // [INVARIANT:VALID_OUT_LATENCY]
                valid_out <= 1'b1;
                if (out_beat_count == BEATS_PER_PIXEL - 1) begin
                    sending        <= 1'b0;
                    out_beat_count <= {COUNT_W{1'b0}};
                    // [INVARIANT:READY_IN_GATING]
                    ready_in       <= 1'b1;
                end else begin
                    out_beat_count <= out_beat_count + 4'd1;
                    end
                end else begin
                    valid_out <= 1'b1;
                end
            end
        end
    end
endmodule
