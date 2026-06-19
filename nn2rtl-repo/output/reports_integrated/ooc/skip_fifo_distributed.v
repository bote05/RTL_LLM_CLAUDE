module skip_fifo_distributed #(
    parameter integer WIDTH = 8,
    parameter integer DEPTH = 16
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              in_valid,
    input  wire [WIDTH-1:0]  in_data,
    output wire              in_ready,
    output wire              out_valid,
    output wire [WIDTH-1:0]  out_data,
    input  wire              out_ready
);
    // DEPTH must be a power of 2; ADDR_W = log2(DEPTH).
    function integer clog2;
        input integer value;
        integer v;
        begin
            v = value - 1;
            for (clog2 = 0; v > 0; clog2 = clog2 + 1) v = v >> 1;
        end
    endfunction
    localparam integer ADDR_W = clog2(DEPTH);

    (* ram_style="distributed" *) reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg [ADDR_W:0]  wr_ptr;
    reg [ADDR_W:0]  rd_ptr;
    // [fifo-peak audit] high-water occupancy, printed per-instance at sim end.
    reg  [ADDR_W:0] peak_occ;
    wire [ADDR_W:0] occ_now = wr_ptr - rd_ptr;

    wire [ADDR_W-1:0] wr_idx = wr_ptr[ADDR_W-1:0];
    wire [ADDR_W-1:0] rd_idx = rd_ptr[ADDR_W-1:0];
    wire empty = (wr_ptr == rd_ptr);
    // full when low bits match but top bit differs (one-extra-bit pointer trick).
    wire full  = (wr_ptr[ADDR_W] != rd_ptr[ADDR_W]) &&
                 (wr_ptr[ADDR_W-1:0] == rd_ptr[ADDR_W-1:0]);

    assign in_ready  = ~full;
    assign out_valid = ~empty;
    assign out_data  = mem[rd_idx];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= {(ADDR_W+1){1'b0}};
            rd_ptr <= {(ADDR_W+1){1'b0}};
            peak_occ <= {(ADDR_W+1){1'b0}};
        end else begin
            if (in_valid && ~full)        wr_ptr <= wr_ptr + 1'b1;
            if (out_ready && ~empty)      rd_ptr <= rd_ptr + 1'b1;
            if (occ_now > peak_occ)       peak_occ <= occ_now;
        end
    end

    // Array-memory write split out per knowledge/patterns/protected/08_common_bugs.md.
    always @(posedge clk) begin
        if (in_valid && ~full) mem[wr_idx] <= in_data;
    end

    // [fifo-peak audit] print each instance's high-water occupancy at sim end.
endmodule
