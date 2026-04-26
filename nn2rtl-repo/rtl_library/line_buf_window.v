// line_buf_window — KH-row line buffer + KH×KW×IC registered shift-register
// window, with vertical rotation on input-row transitions.
//
// Part of the split spatial-conv architecture (see SPLIT_ARCHITECTURE.md).
// Consumes a handshake-driven INT8 stream and publishes a flat
// KH*KW*IC*8-bit window representing the receptive field for the
// scheduler's CURRENT coordinate.
//
// Fixed-position semantics: line_buf[0] is always the OLDEST row of the
// receptive field, line_buf[KH-1] is always the NEWEST (the row being
// written). On the cycle the scheduler advances past sched_in_col ==
// MAX_IN_COL (which is always the right-padded column, so the window
// load for that cycle is zero-filled and does NOT read line_buf), the
// buffer vertically shifts: line_buf[i] <= line_buf[i+1] for i<KH-1;
// line_buf[KH-1] is cleared. The next cycle (in_col=0 of the new row)
// then reads line_buf[i][0] with post-shift values, so the window sees
// the correct rows.
//
// Multi-frame: asserting `frame_start` (typically wired from the
// top-level's start_pulse) clears line_buf and window to zero so
// back-to-back input frames don't inherit stale state.

module line_buf_window #(
    parameter integer IC = 64,
    parameter integer IW = 112,
    parameter integer IH = 112,
    parameter integer KH = 3,
    parameter integer KW = 3,
    parameter integer PW = 1,
    parameter integer PH = 1
) (
    input  wire                               clk,
    input  wire                               rst_n,

    // One-cycle pulse wired from top-level's start_pulse. Clears line_buf
    // and window so the next frame starts with zero history (top-pad
    // zeros for the first fill_rows of output).
    input  wire                               frame_start,

    // Scheduler observers.
    input  wire [$clog2(IH + PH + 1)-1:0]     sched_in_row,
    input  wire [$clog2(IW + PW + 1)-1:0]     sched_in_col,
    input  wire                               sched_needs_real_input,
    input  wire                               sched_advance,
    input  wire                               sched_output_fires,

    // Upstream pixel stream (handshake with the scheduler, not with us).
    input  wire                               valid_in,
    input  wire [IC*8-1:0]                    data_in,

    // Flat packed window for the datapath. Layout:
    //   window_flat[(kh*KW*IC + kw*IC + ic)*8 +: 8]
    output wire [KH*KW*IC*8-1:0]              window_flat
);

    localparam integer MAX_IN_COL = IW + PW - 1;

    // Fixed-position line buffer and shift-register window.
    reg signed [7:0] line_buf [0:KH-1][0:IW-1][0:IC-1];
    reg signed [7:0] window   [0:KH-1][0:KW-1][0:IC-1];

    // Edge classifiers.
    wire right_padded  = (sched_in_col >= IW);
    wire bottom_padded = (sched_in_row >= IH);

    // REAL-region handshake.
    wire handshake_real = sched_advance && sched_needs_real_input && valid_in;

    // Row-transition event: SAME CYCLE as the scheduler's advance past
    // MAX_IN_COL. At this cycle sched_in_col == MAX_IN_COL (right-padded)
    // which means the window load takes the all-zero right-pad branch
    // and does NOT read line_buf. So firing the vertical shift in this
    // same cycle is safe — no RHS/LHS conflict on line_buf.
    wire row_wrap_this_cycle =
        sched_advance && (sched_in_col == MAX_IN_COL[$clog2(IW + PW + 1)-1:0]);

    integer i, j, c_ch;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < KH; i = i + 1)
                for (j = 0; j < IW; j = j + 1)
                    for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                        line_buf[i][j][c_ch] <= 8'sd0;
            for (i = 0; i < KH; i = i + 1)
                for (j = 0; j < KW; j = j + 1)
                    for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                        window[i][j][c_ch] <= 8'sd0;
        end else if (frame_start) begin
            // Same zeroing as reset — back-to-back frames get a clean slate.
            for (i = 0; i < KH; i = i + 1)
                for (j = 0; j < IW; j = j + 1)
                    for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                        line_buf[i][j][c_ch] <= 8'sd0;
            for (i = 0; i < KH; i = i + 1)
                for (j = 0; j < KW; j = j + 1)
                    for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                        window[i][j][c_ch] <= 8'sd0;
        end else begin
            // --------------------------------------------------------
            // Line-buffer vertical shift on row transition. Fires on the
            // SAME cycle as the advance past MAX_IN_COL, which is a
            // right-padded cycle (sched_in_col >= IW), so the window
            // load below takes the right_padded zero branch and never
            // reads line_buf. RHS/LHS safe.
            // --------------------------------------------------------
            if (row_wrap_this_cycle) begin
                for (i = 0; i < KH - 1; i = i + 1)
                    for (j = 0; j < IW; j = j + 1)
                        for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                            line_buf[i][j][c_ch] <= line_buf[i+1][j][c_ch];
                for (j = 0; j < IW; j = j + 1)
                    for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                        line_buf[KH-1][j][c_ch] <= 8'sd0;
            end

            // --------------------------------------------------------
            // Line-buffer write on REAL handshake.
            // --------------------------------------------------------
            if (handshake_real && !right_padded && !bottom_padded) begin
                for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                    line_buf[KH-1][sched_in_col][c_ch] <=
                        $signed(data_in[c_ch*8 +: 8]);
            end

            // --------------------------------------------------------
            // Window shift + load on every scheduler advance (except
            // frozen during sched_output_fires — window must hold
            // receptive field steady through the MAC pipeline).
            // --------------------------------------------------------
            if (sched_advance && !sched_output_fires) begin
                // Shift columns left by one.
                for (i = 0; i < KH; i = i + 1)
                    for (j = 0; j < KW - 1; j = j + 1)
                        for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                            window[i][j][c_ch] <= window[i][j+1][c_ch];

                // Load new rightmost column.
                //   Upper rows (0..KH-2): from line_buf[i][sched_in_col].
                //     (Fixed-position semantics: line_buf[0..KH-2] hold the
                //     KH-1 most recent completed rows; line_buf[KH-1] is
                //     the row being written this cycle, whose current pixel
                //     we bypass directly from data_in.)
                //   Bottom row (KH-1): data_in bypass on REAL handshake.
                //   Right-pad zeros for all rows; bottom-pad zeros for
                //   just the bottom row.
                for (i = 0; i < KH - 1; i = i + 1) begin
                    for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1) begin
                        if (right_padded) begin
                            window[i][KW-1][c_ch] <= 8'sd0;
                        end else begin
                            window[i][KW-1][c_ch] <= line_buf[i][sched_in_col][c_ch];
                        end
                    end
                end
                for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1) begin
                    if (right_padded || bottom_padded) begin
                        window[KH-1][KW-1][c_ch] <= 8'sd0;
                    end else if (handshake_real) begin
                        window[KH-1][KW-1][c_ch] <= $signed(data_in[c_ch*8 +: 8]);
                    end else begin
                        // REAL region, no handshake: should not occur
                        // because sched_advance in REAL region implies
                        // handshake. Defensive zero.
                        window[KH-1][KW-1][c_ch] <= 8'sd0;
                    end
                end
            end
        end
    end

    // Flatten window. Layout matches conv_datapath's tap_at():
    //   window_flat[(kh*KW*IC + kw*IC + ic)*8 +: 8]
    genvar g_kh, g_kw, g_ic;
    generate
        for (g_kh = 0; g_kh < KH; g_kh = g_kh + 1) begin : gen_win_kh
            for (g_kw = 0; g_kw < KW; g_kw = g_kw + 1) begin : gen_win_kw
                for (g_ic = 0; g_ic < IC; g_ic = g_ic + 1) begin : gen_win_ic
                    assign window_flat[(g_kh*KW*IC + g_kw*IC + g_ic)*8 +: 8] =
                        window[g_kh][g_kw][g_ic];
                end
            end
        end
    endgenerate

endmodule
