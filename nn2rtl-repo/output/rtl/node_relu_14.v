// node_relu_14 — tiled-streaming ReLU
// IC=OC=128, bus=256-bit, channel_tile=32 -> BEATS_PER_PIXEL=4
// First valid_out fires BEATS_PER_PIXEL cycles after first valid_in
// (overlapping last-input-beat capture with first-output-beat emission).
`timescale 1ns / 1ps

module node_relu_14 (
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
    localparam integer BEATS_PER_PIXEL = 4;
    localparam integer COUNT_W         = 3;

    reg [BEAT_WIDTH_BITS-1:0] beat_buf [0:BEATS_PER_PIXEL-1];
    reg [COUNT_W-1:0] in_beat_count;
    reg [COUNT_W-1:0] out_beat_count;
    reg               sending;

    integer ch, i;
    reg signed [7:0] tmp_byte;

    // [K1-FDCE] sync-only memory write -- no reset clause (same pattern as
    // node_relu.v): beat_buf is gather DATA, fully rewritten each pixel
    // before the sending phase reads it. Also unblocks LUTRAM inference.
    always @(posedge clk) begin
        if (!sending && valid_in && ready_in) begin
            beat_buf[in_beat_count] <= data_in;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ready_in       <= 1'b1; // [INVARIANT:READY_IN_GATING]
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
                            data_out[ch*8 +: 8] <= (tmp_byte > 8'sd0) ? tmp_byte : 8'sd0;
                        end
                        valid_out      <= 1'b1; // [INVARIANT:VALID_OUT_LATENCY]
                        out_beat_count <= 3'd1;
                    end else begin
                        in_beat_count <= in_beat_count + 3'd1;
                    end
                end
            end else begin
                if (ready_out) begin
                for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                    tmp_byte = $signed(beat_buf[out_beat_count][ch*8 +: 8]);
                    data_out[ch*8 +: 8] <= (tmp_byte > 8'sd0) ? tmp_byte : 8'sd0;
                end
                valid_out <= 1'b1; // [INVARIANT:VALID_OUT_LATENCY]
                if (out_beat_count == BEATS_PER_PIXEL - 1) begin
                    sending        <= 1'b0;
                    out_beat_count <= {COUNT_W{1'b0}};
                    ready_in       <= 1'b1; // [INVARIANT:READY_IN_GATING]
                end else begin
                    out_beat_count <= out_beat_count + 3'd1;
                    end
                end else begin
                    valid_out <= 1'b1;
                end
            end
        end
    end
endmodule
