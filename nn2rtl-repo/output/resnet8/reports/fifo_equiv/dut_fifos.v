module skip_fifo #(
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

    (* ram_style = "distributed" *) reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg [ADDR_W:0]  wr_ptr;
    reg [ADDR_W:0]  rd_ptr;

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
        end else begin
            if (in_valid && ~full)        wr_ptr <= wr_ptr + 1'b1;
            if (out_ready && ~empty)      rd_ptr <= rd_ptr + 1'b1;
        end
    end

    // Array-memory write split out per knowledge/patterns/protected/08_common_bugs.md.
    always @(posedge clk) begin
        if (in_valid && ~full) mem[wr_idx] <= in_data;
    end
endmodule

module bram_fifo #(
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
    function integer clog2;
        input integer value;
        integer v;
        begin
            v = value - 1;
            for (clog2 = 0; v > 0; clog2 = clog2 + 1) v = v >> 1;
        end
    endfunction
    localparam integer ADDR_W = clog2(DEPTH);

    (* ram_style = "block" *) reg [WIDTH-1:0] mem [0:DEPTH-1];

    // Ring pointers (one extra bit for full/empty disambiguation).
    reg  [ADDR_W:0] wr_ptr;     // next write slot
    reg  [ADDR_W:0] rd_ptr;     // next slot to READ from BRAM (issued address)
    // Count of beats committed to BRAM but not yet pulled into the output skid.
    reg  [ADDR_W:0] mem_count;
    wire [ADDR_W-1:0] wr_idx = wr_ptr[ADDR_W-1:0];
    wire [ADDR_W-1:0] rd_idx = rd_ptr[ADDR_W-1:0];

    // ---- 2-entry output skid (FWFT) ----
    reg [WIDTH-1:0] out_reg;     // head-of-queue (presented on out_data)
    reg             out_reg_v;
    reg [WIDTH-1:0] skid_reg;    // second prefetched beat
    reg             skid_v;

    // BRAM sync read pipeline: when we issue a read, data arrives next cycle.
    reg             rd_issue_q;  // a read was issued last cycle -> mem_q valid now
    reg [WIDTH-1:0] mem_q;       // BRAM registered-read output

    wire push = in_valid && in_ready;
    wire pop  = out_valid && out_ready;

    // Occupancy for full: beats in BRAM + skid + outreg + the in-flight read.
    wire [ADDR_W+1:0] occupancy = mem_count
                                + (out_reg_v ? 1 : 0)
                                + (skid_v    ? 1 : 0)
                                + (rd_issue_q ? 1 : 0);
    wire full = (occupancy >= DEPTH);

    assign in_ready  = ~full;
    assign out_valid = out_reg_v;
    assign out_data  = out_reg;

    // Issue a BRAM read whenever a beat sits in BRAM and the 2-entry skid will
    // have room for it next cycle (a slot free now, or a pop frees the head now).
    wire skid_has_room = !(out_reg_v && skid_v) || pop;
    wire do_issue = (mem_count != 0) && skid_has_room && !rd_issue_q;

    // ---- BRAM access: write + registered read in ONE clocked block (no reset).
    // This is the canonical Vivado simple-dual-port BRAM template -> infers a
    // RAMB36 cleanly (the split-into-two-blocks form was dissolved to FFs).
    always @(posedge clk) begin
        if (push) mem[wr_idx] <= in_data;
        mem_q <= mem[rd_idx];
    end

    // ---- control / pointers / skid (FFs, async reset; NO memory access here) ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr     <= {(ADDR_W+1){1'b0}};
            rd_ptr     <= {(ADDR_W+1){1'b0}};
            mem_count  <= {(ADDR_W+1){1'b0}};
            out_reg_v  <= 1'b0;
            skid_v     <= 1'b0;
            rd_issue_q <= 1'b0;
            out_reg    <= {WIDTH{1'b0}};
            skid_reg   <= {WIDTH{1'b0}};
        end else begin
            if (push)     wr_ptr <= wr_ptr + 1'b1;
            rd_issue_q <= do_issue;
            if (do_issue) rd_ptr <= rd_ptr + 1'b1;

            // skid / output update: pop -> shift skid -> accept freshly-read mem_q.
            begin : skid_update
                reg or_v;  reg [WIDTH-1:0] or_d;
                reg sk_v;  reg [WIDTH-1:0] sk_d;
                or_v = out_reg_v;  or_d = out_reg;
                sk_v = skid_v;     sk_d = skid_reg;
                if (pop) or_v = 1'b0;                      // head consumed
                if (!or_v && sk_v) begin                   // shift skid -> head
                    or_v = 1'b1; or_d = sk_d; sk_v = 1'b0;
                end
                if (rd_issue_q) begin                      // accept mem_q (valid now)
                    if (!or_v) begin or_v = 1'b1; or_d = mem_q; end
                    else if (!sk_v) begin sk_v = 1'b1; sk_d = mem_q; end
                end
                out_reg_v <= or_v;  out_reg  <= or_d;
                skid_v    <= sk_v;  skid_reg <= sk_d;
            end

            // mem_count: +1 on push, -1 when a beat is pulled from BRAM (do_issue).
            case ({push, do_issue})
                2'b10: mem_count <= mem_count + 1'b1;
                2'b01: mem_count <= mem_count - 1'b1;
                default: mem_count <= mem_count;
            endcase
        end
    end
endmodule
