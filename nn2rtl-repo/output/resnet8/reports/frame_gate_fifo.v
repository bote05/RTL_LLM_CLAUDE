// frame_gate_fifo: a skip_fifo that buffers a full FRAME of `FRAME` beats
// before releasing them as a CONTIGUOUS gap-free burst. Needed in front of the
// "cycle_count(whole-frame)" convs (node_conv2d_2/3/6) whose output trigger is
// time-based and therefore mis-fires on gappy input. Once FRAME beats have been
// pushed, out_ready-driven draining begins and the buffered beats stream out
// one/cycle with no gaps; after the frame drains the gate re-arms for the next.
//
// Interface mirrors skip_fifo (FWFT). in_*: push side (free-running producer).
// out_*: pull side (the conv). out_valid is HELD LOW until a full frame is
// buffered, then asserted contiguously while the conv consumes.
module frame_gate_fifo #(
    parameter integer WIDTH = 128,
    parameter integer DEPTH = 2048,
    parameter integer FRAME = 1024
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             in_valid,
    input  wire [WIDTH-1:0] in_data,
    output wire             in_ready,
    output wire             out_valid,
    output wire [WIDTH-1:0] out_data,
    input  wire             out_ready
);
    function integer clog2; input integer value; integer v; begin
        v = value - 1; for (clog2 = 0; v > 0; clog2 = clog2 + 1) v = v >> 1; end
    endfunction
    localparam integer AW = clog2(DEPTH);
    localparam integer FW = clog2(FRAME + 1);

    reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg [AW:0] wr_ptr, rd_ptr;
    wire [AW-1:0] wr_idx = wr_ptr[AW-1:0];
    wire [AW-1:0] rd_idx = rd_ptr[AW-1:0];
    wire empty = (wr_ptr == rd_ptr);
    wire full  = (wr_ptr[AW] != rd_ptr[AW]) && (wr_ptr[AW-1:0] == rd_ptr[AW-1:0]);

    // gate state: count beats pushed toward the current frame; release when full
    reg [FW-1:0] fill_q;        // beats buffered toward the frame (not yet released)
    reg          releasing_q;   // 1 while draining a complete frame
    reg [FW-1:0] drain_q;       // beats drained in the current release

    wire push = in_valid && ~full;
    wire pop  = out_valid && out_ready;

    assign in_ready  = ~full;
    assign out_valid = releasing_q && ~empty;
    assign out_data  = mem[rd_idx];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 0; rd_ptr <= 0;
            fill_q <= 0; releasing_q <= 1'b0; drain_q <= 0;
        end else begin
            if (push) begin
                mem[wr_idx] <= in_data;
                wr_ptr <= wr_ptr + 1'b1;
            end
            if (pop) rd_ptr <= rd_ptr + 1'b1;

            if (!releasing_q) begin
                // accumulate toward a full frame
                if (push) begin
                    if (fill_q == FRAME - 1) begin
                        releasing_q <= 1'b1;   // frame complete -> release contiguously
                        drain_q     <= 0;
                        fill_q      <= 0;
                    end else begin
                        fill_q <= fill_q + 1'b1;
                    end
                end
            end else begin
                // draining the released frame
                if (pop) begin
                    if (drain_q == FRAME - 1) begin
                        releasing_q <= 1'b0;   // frame drained -> re-arm
                        drain_q     <= 0;
                    end else begin
                        drain_q <= drain_q + 1'b1;
                    end
                end
                // a push during release belongs to the NEXT frame; count it
                if (push) fill_q <= fill_q + 1'b1;
            end
        end
    end
endmodule
