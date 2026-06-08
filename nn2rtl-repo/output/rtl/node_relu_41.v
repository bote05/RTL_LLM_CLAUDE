// Tiled-streaming ReLU
// module_id   : node_relu_41
// spec_hash   : relu_512x512_s7x7_i256_o256_iotiled-streaming_tile32
// IC=OC=512, spatial 7x7, bus 256b, channel_tile=32, BEATS_PER_PIXEL=16
// Latency contract: first valid_out fires BEATS_PER_PIXEL (16) cycles after first valid_in.
`timescale 1ns / 1ps

module node_relu_41 (
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
    localparam integer BEATS_PER_PIXEL = 16;
    localparam integer COUNT_W         = 5;

    reg [BEAT_WIDTH_BITS-1:0] beat_buf [0:BEATS_PER_PIXEL-1];
    reg [COUNT_W-1:0] in_beat_count;
    reg [COUNT_W-1:0] out_beat_count;
    reg               sending;

    integer ch, i;
    localparam integer RS_MULT  = 2329;
    localparam integer RS_SHIFT = 11;
    localparam integer RS_ROUND = 1024;
    reg signed [7:0] tmp_byte;
    reg signed [31:0] rs_in, rs_out;

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
                ready_in <= 1'b1; // [INVARIANT:READY_IN_GATING]
                if (valid_in && ready_in) begin
                    beat_buf[in_beat_count] <= data_in;
                    if (in_beat_count == BEATS_PER_PIXEL - 1) begin
                        in_beat_count <= {COUNT_W{1'b0}};
                        sending       <= 1'b1;
                        ready_in      <= 1'b0; // [INVARIANT:READY_IN_GATING]
                        // Overlap last input capture with first ReLU emission so
                        // first valid_out fires exactly BEATS_PER_PIXEL cycles
                        // after first valid_in.
                        for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                            tmp_byte = $signed(beat_buf[0][ch*8 +: 8]);
                            begin
                            rs_in  = (tmp_byte > 8'sd0) ? $signed(tmp_byte) : 32'sd0;
                            rs_out = (rs_in * RS_MULT + RS_ROUND) >>> RS_SHIFT;
                            data_out[ch*8 +: 8] <= (rs_out > 32'sd127) ? 8'sd127 : rs_out[7:0];
                        end
                        end
                        valid_out      <= 1'b1; // [INVARIANT:VALID_OUT_LATENCY]
                        out_beat_count <= 5'd1;
                    end else begin
                        in_beat_count <= in_beat_count + 5'd1;
                    end
                end
            end else begin
                if (ready_out) begin
                for (ch = 0; ch < CHANNEL_TILE; ch = ch + 1) begin
                    tmp_byte = $signed(beat_buf[out_beat_count][ch*8 +: 8]);
                    begin
                            rs_in  = (tmp_byte > 8'sd0) ? $signed(tmp_byte) : 32'sd0;
                            rs_out = (rs_in * RS_MULT + RS_ROUND) >>> RS_SHIFT;
                            data_out[ch*8 +: 8] <= (rs_out > 32'sd127) ? 8'sd127 : rs_out[7:0];
                        end
                end
                valid_out <= 1'b1; // [INVARIANT:VALID_OUT_LATENCY]
                if (out_beat_count == BEATS_PER_PIXEL - 1) begin
                    sending        <= 1'b0;
                    out_beat_count <= {COUNT_W{1'b0}};
                    ready_in       <= 1'b1; // [INVARIANT:READY_IN_GATING]
                end else begin
                    out_beat_count <= out_beat_count + 5'd1;
                    end
                end else begin
                    valid_out <= 1'b1;
                end
            end
        end
    end
endmodule
