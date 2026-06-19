// node_conv2d -- 3x3 stride-1 pad-1 conv (IC=3, OC=16, IH=IW=32).
// Fully-parallel: 3-row line buffer + parallel 16-OC MAC, single-cycle requantize.
// pipeline_latency_cycles=72; SCALE_MULT=8098, SCALE_SHIFT=8 (scale_factor=31.6334...).

module node_conv2d (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       valid_in,
    output wire                       ready_in,
    input  wire [23:0]                data_in,
    output wire                       valid_out,
    output wire [127:0]               data_out
);
    localparam integer IC        = 3;
    localparam integer OC        = 16;
    localparam integer IH        = 32;
    localparam integer IW        = 32;
    localparam integer OH        = 32;
    localparam integer OW        = 32;
    localparam integer KH        = 3;
    localparam integer KW        = 3;
    localparam integer K_TOTAL   = IC*KH*KW;       // 27
    localparam integer IN_TOTAL  = IH*IW;          // 1024
    localparam integer OUT_TOTAL = OH*OW;          // 1024
    localparam integer FIRST_OUT_LATENCY = 72;

    localparam integer SCALE_MULT  = 8098;
    localparam integer SCALE_SHIFT = 8;

    // Weight and bias ROMs (loaded via $readmemh)
    (* rom_style = "block" *) reg signed [7:0]  w_rom [0:OC*K_TOTAL-1];
    (* rom_style = "block" *) reg signed [31:0] b_rom [0:OC-1];

    initial begin
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/weights/node_conv2d_weights.hex", w_rom);
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/weights/node_conv2d_bias.hex", b_rom);
    end

    // 3-row x 32-col x 3-channel line buffer (BRAM-style)
    (* ram_style = "block" *) reg signed [7:0] lb [0:2][0:IW-1][0:IC-1];

    // Input ingest position
    reg [11:0] in_idx;
    reg [4:0]  in_col;
    reg [4:0]  in_row;
    reg [1:0]  in_slot;

    // Latency tracking
    reg        first_seen;
    reg [7:0]  cyc;

    // Output emission position
    reg        out_active;
    reg [11:0] out_idx;
    reg [4:0]  out_row;
    reg [4:0]  out_col;
    reg [1:0]  out_slot_mid;

    // Registered outputs
    reg                valid_out_r;
    reg [127:0]        data_out_r;

    assign valid_out = valid_out_r;
    assign data_out  = data_out_r;
    // [INVARIANT:READY_IN_GATING] backpressure when current vector fully ingested
    assign ready_in  = (in_idx < IN_TOTAL[11:0]);

    // Slot indices for the 3 active rows of the line buffer
    wire [1:0] out_slot_top = (out_slot_mid == 2'd0) ? 2'd2 : (out_slot_mid - 2'd1);
    wire [1:0] out_slot_bot = (out_slot_mid == 2'd2) ? 2'd0 : (out_slot_mid + 2'd1);

    // Combinational window: 3x3x3 = 27 bytes with zero-padding
    wire signed [7:0] win [0:KH*KW*IC-1];

    genvar gkr, gkc, gic;
    generate
        for (gkr = 0; gkr < KH; gkr = gkr + 1) begin : G_R
            wire signed [6:0] r_s = $signed({2'b00, out_row}) + (gkr - 1);
            wire              r_ok = (r_s >= 0) && (r_s < IH);
            wire [1:0] row_slot = (gkr == 0) ? out_slot_top :
                                  (gkr == 1) ? out_slot_mid : out_slot_bot;
            for (gkc = 0; gkc < KW; gkc = gkc + 1) begin : G_C
                wire signed [6:0] c_s = $signed({2'b00, out_col}) + (gkc - 1);
                wire              c_ok = (c_s >= 0) && (c_s < IW);
                wire              in_bounds = r_ok && c_ok;
                for (gic = 0; gic < IC; gic = gic + 1) begin : G_IC
                    assign win[(gkr*KW + gkc)*IC + gic] =
                        in_bounds ? lb[row_slot][c_s[4:0]][gic] : 8'sd0;
                end
            end
        end
    endgenerate

    // Per-OC dot product (combinational sum tree)
    wire signed [21:0] acc [0:OC-1];

    genvar goc, gk;
    generate
        for (goc = 0; goc < OC; goc = goc + 1) begin : G_OC
            wire signed [21:0] partial [0:K_TOTAL];
            assign partial[0] = 22'sd0;
            for (gk = 0; gk < K_TOTAL; gk = gk + 1) begin : G_K
                assign partial[gk+1] = partial[gk] +
                    $signed(win[gk]) * $signed(w_rom[goc*K_TOTAL + gk]);
            end
            assign acc[goc] = partial[K_TOTAL];
        end
    endgenerate

    // Bias + scale + sign-aware round + INT8 saturate
    wire signed [33:0] biased    [0:OC-1];
    wire signed [49:0] scaled    [0:OC-1];
    wire signed [49:0] rounded_q [0:OC-1];
    wire signed [7:0]  byte_c    [0:OC-1];

    localparam signed [49:0] SCALE_HALF    = 50'sd128;   // 1 << (SHIFT-1) = 1<<7
    localparam signed [49:0] SCALE_HALF_M1 = 50'sd127;

    genvar goc2;
    generate
        for (goc2 = 0; goc2 < OC; goc2 = goc2 + 1) begin : G_REQ
            assign biased[goc2] = {{12{acc[goc2][21]}}, acc[goc2]} + b_rom[goc2];
            assign scaled[goc2] = biased[goc2] * $signed(SCALE_MULT[15:0]);
            // [INVARIANT:ROUNDING] sign-aware round-half-to-even-of-zero
            assign rounded_q[goc2] =
                (scaled[goc2] + (scaled[goc2][49] ? SCALE_HALF_M1 : SCALE_HALF)) >>> SCALE_SHIFT;
            assign byte_c[goc2] =
                (rounded_q[goc2] >  50'sd127)  ? 8'sd127  :
                (rounded_q[goc2] < -50'sd128)  ? -8'sd128 :
                                                  rounded_q[goc2][7:0];
        end
    endgenerate

    // Main sequential block (async reset)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            first_seen   <= 1'b0;
            cyc          <= 8'd0;
            in_idx       <= 12'd0;
            in_col       <= 5'd0;
            in_row       <= 5'd0;
            in_slot      <= 2'd0;
            out_active   <= 1'b0;
            out_idx      <= 12'd0;
            out_row      <= 5'd0;
            out_col      <= 5'd0;
            out_slot_mid <= 2'd0;
            valid_out_r  <= 1'b0;
            data_out_r   <= 128'd0;
        end else begin
            valid_out_r <= 1'b0;

            // Latency counter: starts at 1 on first ingest, increments each cycle
            if (valid_in && ready_in && !first_seen) begin
                first_seen <= 1'b1;
                cyc        <= 8'd1;
            end else if (first_seen) begin
                cyc <= cyc + 8'd1;
            end

            // Ingest pointer
            if (valid_in && ready_in) begin
                in_idx <= in_idx + 12'd1;
                if (in_col == 5'd31) begin
                    in_col <= 5'd0;
                    if (in_row == 5'd31) in_row <= 5'd0;
                    else                 in_row <= in_row + 5'd1;
                    if (in_slot == 2'd2) in_slot <= 2'd0;
                    else                 in_slot <= in_slot + 2'd1;
                end else begin
                    in_col <= in_col + 5'd1;
                end
            end

            // [INVARIANT:VALID_OUT_LATENCY] first valid_out at posedge T0+72
            if (first_seen && (cyc == FIRST_OUT_LATENCY-1) && !out_active) begin
                out_active <= 1'b1;
            end

            // Output emission: one pixel per cycle once activated
            if ((out_active || (first_seen && (cyc == FIRST_OUT_LATENCY-1))) &&
                (out_idx < OUT_TOTAL[11:0])) begin
                valid_out_r <= 1'b1;
                data_out_r  <= {byte_c[15], byte_c[14], byte_c[13], byte_c[12],
                                byte_c[11], byte_c[10], byte_c[9],  byte_c[8],
                                byte_c[7],  byte_c[6],  byte_c[5],  byte_c[4],
                                byte_c[3],  byte_c[2],  byte_c[1],  byte_c[0]};
                out_idx <= out_idx + 12'd1;
                if (out_col == 5'd31) begin
                    out_col <= 5'd0;
                    if (out_row == 5'd31) out_row <= 5'd0;
                    else                  out_row <= out_row + 5'd1;
                    if (out_slot_mid == 2'd2) out_slot_mid <= 2'd0;
                    else                      out_slot_mid <= out_slot_mid + 2'd1;
                end else begin
                    out_col <= out_col + 5'd1;
                end
            end

            // End-of-vector full reset for next stream
            if (out_active && (out_idx == OUT_TOTAL[11:0])) begin
                first_seen   <= 1'b0;
                cyc          <= 8'd0;
                in_idx       <= 12'd0;
                in_col       <= 5'd0;
                in_row       <= 5'd0;
                in_slot      <= 2'd0;
                out_active   <= 1'b0;
                out_idx      <= 12'd0;
                out_row      <= 5'd0;
                out_col      <= 5'd0;
                out_slot_mid <= 2'd0;
            end
        end
    end

    // Line buffer write (sync-only, no async reset -> BRAM inference)
    always @(posedge clk) begin
        if (valid_in && ready_in && (in_idx < IN_TOTAL[11:0])) begin
            lb[in_slot][in_col][0] <= $signed(data_in[7:0]);
            lb[in_slot][in_col][1] <= $signed(data_in[15:8]);
            lb[in_slot][in_col][2] <= $signed(data_in[23:16]);
        end
    end

endmodule
