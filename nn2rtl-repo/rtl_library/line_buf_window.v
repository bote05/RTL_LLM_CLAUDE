// line_buf_window -- KH-row line buffer (BRAM-inferred via per-slot
// single-port memories with a rotating row pointer) + KH x KW x IC
// registered shift-register window.
//
// Replaces the prior flop-based bulk-vertical-shift implementation:
// the legacy "line_buf[i] <= line_buf[i+1]" copy is now a single
// counter increment on `oldest_slot`. Per-slot BRAMs hold the actual
// pixel data; `row_valid[KH-1:0]` masks reads from slots whose data
// is stale relative to the current frame.
//
// Vivado/Artix-7 BRAM inference targets:
//   line_buf : KH separate BRAM18/BRAM36 instances. Each holds
//              IW+PW words of (IC*8) bits with sync read+write at
//              the same address (standard single-port template).
//   window   : KH x KW x IC * 8-bit shift register in fabric flops
//              (small: 9*64*8 = 4.6 kbit for 3x3 conv2 = trivial).
//
// Multi-frame correctness (this is exactly where the prior bug class
// lived, so it is reasoned about explicitly here):
//
//   At frame_start:
//     oldest_slot <= 0
//     row_valid   <= 0     (all KH bits cleared)
//     window      <= 0     (shift reg cleared)
//     bypass_reg  <= 0     (data_in bypass FF cleared)
//
//   The BRAM cells themselves are NOT cleared on frame_start. They
//   retain whatever data was written during the prior frame. The
//   `row_valid` mask is the sole guarantee that top-pad reads (from
//   slots whose row hasn't been written yet this frame) return zero
//   instead of stale frame-N-1 pixels:
//
//       window_kwm1_wire[i] = row_valid[slot_for_rf[i]]
//                              ? q_array[slot_for_rf[i]]
//                              : 0
//
//   row_valid is updated at row_wrap_this_cycle:
//     - the slot that just finished filling becomes "history" and
//       has its bit set
//     - the slot that becomes the new "currently writing" has its
//       bit cleared (it will be overwritten with this frame's
//       next input row, so its prior contents are stale)
//
//   Right-pad columns: writes are gated by `!right_padded`, so the
//   BRAM cell at addr=MAX_IN_COL is never touched. Vivado initialises
//   BRAMs to zero at FPGA configuration, so reads at MAX_IN_COL
//   return zero -- no explicit right_padded mask is needed in the
//   read path.
//
// Latency: this rewrite preserves the legacy 1-cycle read pipeline
// EXACTLY. The legacy design had async-read of line_buf followed by
// a window[i][KW-1] FF (1 cycle). The new design replaces that with
// a per-slot BRAM whose output register q_array[s] IS the window's
// rightmost column (no extra FF stage). compute_conv2d_latency_cycles
// in scripts/golden_impl.py is unchanged.

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

    // One-cycle pulse wired from top-level start_pulse. Clears
    // the rotating pointer, row_valid bits, window shift register,
    // and the data_in bypass FF. Does NOT clear BRAM cells (they
    // retain prior-frame data; row_valid is the multi-frame
    // safety net).
    input  wire                               frame_start,

    // Scheduler observers.
    input  wire [$clog2(IH + PH + 1)-1:0]     sched_in_row,
    input  wire [$clog2(IW + PW + 1)-1:0]     sched_in_col,
    input  wire                               sched_needs_real_input,
    input  wire                               sched_advance,
    input  wire                               sched_output_fires,

    // Upstream pixel stream.
    input  wire                               valid_in,
    input  wire [IC*8-1:0]                    data_in,

    // Flat packed window for the datapath. Layout matches
    // conv_datapath's tap_at():
    //   window_flat[(kh*KW*IC + kw*IC + ic)*8 +: 8]
    output wire [KH*KW*IC*8-1:0]              window_flat
);

    // ---------------- Derived constants ------------------------------
    localparam integer MAX_IN_COL = IW + PW - 1;
    localparam integer MEM_DEPTH  = IW + PW;
    localparam integer SLOT_W     = (KH > 1) ? $clog2(KH) : 1;
    localparam integer SCHED_COL_W = $clog2(IW + PW + 1);
    localparam integer SCHED_ROW_W = $clog2(IH + PH + 1);

    // KH is a 32-bit `parameter integer`; we perform pointer arithmetic
    // and modulo-KH reductions in (SLOT_W+1)-bit space. The width
    // truncation / expansion in the conditionals below is intentional
    // -- KH always fits in SLOT_W+1 bits by definition of SLOT_W. Wrap
    // the affected expressions in lint_off pragmas to silence Verilator's
    // strict-lint warnings without disabling them globally.
    /* verilator lint_off WIDTH */

    // ---------------- Edge classifiers + handshake -------------------
    wire right_padded  = (sched_in_col >= IW[SCHED_COL_W-1:0]);
    wire bottom_padded = (sched_in_row >= IH[SCHED_ROW_W-1:0]);
    wire handshake_real =
        sched_advance && sched_needs_real_input && valid_in;
    wire row_wrap_this_cycle =
        sched_advance && (sched_in_col == MAX_IN_COL[SCHED_COL_W-1:0]);

    // ---------------- Rotating slot pointer + row_valid --------------
    reg [SLOT_W-1:0] oldest_slot;
    reg [KH-1:0]     row_valid;

    // current_write_slot = (oldest_slot + KH - 1) mod KH.
    // For non-power-of-2 KH (e.g. 3 or 7) we need explicit mod logic.
    wire [SLOT_W:0]   cws_sum    = {1'b0, oldest_slot} + (KH - 1);
    wire [SLOT_W-1:0] current_write_slot =
        (cws_sum >= KH) ? (cws_sum - KH) : cws_sum[SLOT_W-1:0];

    // After row_wrap the new "currently writing" slot is the one
    // that was previously the oldest (= oldest_slot pre-increment).
    // That slot's row_valid must be cleared because its prior
    // contents will be overwritten by the new input row's writes.

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            oldest_slot <= {SLOT_W{1'b0}};
            row_valid   <= {KH{1'b0}};
        end else if (frame_start) begin
            oldest_slot <= {SLOT_W{1'b0}};
            row_valid   <= {KH{1'b0}};
        end else if (row_wrap_this_cycle) begin
            oldest_slot <= (oldest_slot == (KH - 1)) ?
                            {SLOT_W{1'b0}} : (oldest_slot + 1'b1);
            // The slot that just finished filling is only "valid history"
            // if the row that just ended was a REAL input row. Bottom-pad
            // rows (sched_in_row >= IH) emit row_wrap events too -- the
            // scheduler walks all in_row positions including the bottom-
            // pad fringe -- but no writes have actually landed in the
            // current_write_slot during that row. Marking the slot valid
            // would cause subsequent reads to return stale data from KH
            // rows ago (the previous occupant), breaking bottom-pad
            // correctness. Gate on !bottom_padded so bottom-pad row_wraps
            // leave row_valid[current_write_slot] cleared.
            row_valid[current_write_slot] <= !bottom_padded;
            row_valid[oldest_slot]        <= 1'b0;
        end
    end

    // ---------------- Per-slot BRAM storage --------------------------
    // KH separate single-port memories. Vivado infers each as a
    // BRAM18/BRAM36 instance from the standard "if (we) write; q <=
    // mem[addr];" template + the (* ram_style = "block" *) attribute.
    // The address path is unconditional (sched_in_col) so synth
    // sees a clean inference target -- no muxes wrapped around the
    // address.

    wire [IC*8-1:0] q_array [0:KH-1];

    genvar g_slot;
    generate
        for (g_slot = 0; g_slot < KH; g_slot = g_slot + 1) begin : gen_slot
            (* ram_style = "block" *)
            reg [IC*8-1:0] mem [0:MEM_DEPTH-1];
            reg [IC*8-1:0] q_reg;

            wire is_writing = (current_write_slot == g_slot[SLOT_W-1:0]);
            wire write_en   =
                handshake_real && !right_padded && !bottom_padded && is_writing;

            always @(posedge clk) begin
                if (write_en) begin
                    mem[sched_in_col] <= data_in;
                end
                // q_reg must be FROZEN whenever the scheduler is stalled.
                // The scheduler asserts output_fires the cycle AFTER it
                // advances past a firing coordinate, with sched_in_col
                // already pointing at the NEXT coord. If q_reg free-ran
                // off live sched_in_col, the rightmost window column would
                // change mid-MAC -- the MAC's tap_at() reads window_flat
                // at every k_counter step, so a moving q would corrupt
                // every k>=1 contribution. Gating on sched_advance keeps
                // q stable from output_fires through the entire MAC pass
                // (stall_in = mac_busy holds sched_advance low for that
                // window, by design of the split-architecture contract).
                if (sched_advance) begin
                    q_reg <= mem[sched_in_col];
                end
            end

            assign q_array[g_slot] = q_reg;
        end
    endgenerate

    // ---------------- Slot mux for window's rightmost column ----------
    // For each receptive-field row i in 0..KH-2, look up the slot
    // currently holding that row's data and route q_array[slot] to
    // window_kwm1_wire[i], gated by row_valid for top-pad / cross-
    // frame stale data.
    //
    // slot_for_rf[i] = (oldest_slot + i) mod KH.

    wire [IC*8-1:0] window_kwm1_wire [0:KH-2];

    genvar g_rf;
    generate
        for (g_rf = 0; g_rf < KH - 1; g_rf = g_rf + 1) begin : gen_rf_mux
            wire [SLOT_W:0]   sum_g    = {1'b0, oldest_slot} + g_rf[SLOT_W:0];
            wire [SLOT_W-1:0] slot_idx =
                (sum_g >= KH) ? (sum_g - KH) : sum_g[SLOT_W-1:0];

            assign window_kwm1_wire[g_rf] =
                row_valid[slot_idx] ? q_array[slot_idx]
                                    : {(IC*8){1'b0}};
        end
    endgenerate
    /* verilator lint_on WIDTH */

    // ---------------- Window shift register --------------------------
    // window[i][j] for i in 0..KH-1, j in 0..KW-2 are the shift-reg
    // stages. The KW-1 column is reconstructed from window_kwm1_wire
    // (for i < KH-1) or from bypass_reg (for i = KH-1, the row
    // currently being received).
    //
    // The shift gate matches the legacy semantics: shift fires on
    // every scheduler advance EXCEPT when sched_output_fires is high
    // (during MAC pipeline the window must hold the receptive field
    // steady).

    reg [7:0] window [0:KH-1][0:KW-2][0:IC-1];
    reg [7:0] bypass_reg [0:IC-1];

    integer i, j, c_ch;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < KH; i = i + 1)
                for (j = 0; j < KW - 1; j = j + 1)
                    for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                        window[i][j][c_ch] <= 8'sd0;
            for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                bypass_reg[c_ch] <= 8'sd0;
        end else if (frame_start) begin
            for (i = 0; i < KH; i = i + 1)
                for (j = 0; j < KW - 1; j = j + 1)
                    for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                        window[i][j][c_ch] <= 8'sd0;
            for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                bypass_reg[c_ch] <= 8'sd0;
        end else if (sched_advance && !sched_output_fires) begin
            // Shift columns left by one. The new KW-2 column samples
            // the (combinational) KW-1 column wire / bypass_reg.
            for (i = 0; i < KH; i = i + 1) begin
                for (j = 0; j < KW - 2; j = j + 1)
                    for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                        window[i][j][c_ch] <= window[i][j+1][c_ch];
            end

            // Last shift stage -- column KW-2 latches column KW-1.
            // Top KH-1 rows: from BRAM mux. Bottom row: from bypass.
            for (i = 0; i < KH - 1; i = i + 1) begin
                for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                    window[i][KW-2][c_ch] <=
                        $signed(window_kwm1_wire[i][c_ch*8 +: 8]);
            end
            for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1)
                window[KH-1][KW-2][c_ch] <= $signed(bypass_reg[c_ch]);

            // bypass_reg update (data_in for the row currently being
            // received, masked by right_padded/bottom_padded).
            for (c_ch = 0; c_ch < IC; c_ch = c_ch + 1) begin
                if (right_padded || bottom_padded) begin
                    bypass_reg[c_ch] <= 8'sd0;
                end else if (handshake_real) begin
                    bypass_reg[c_ch] <= $signed(data_in[c_ch*8 +: 8]);
                end else begin
                    bypass_reg[c_ch] <= 8'sd0;
                end
            end
        end
    end

    // ---------------- Flatten window output --------------------------
    // Layout matches conv_datapath::tap_at():
    //   window_flat[(kh*KW*IC + kw*IC + ic)*8 +: 8]
    //
    // Columns 0..KW-2 come from the shift-reg flops. Column KW-1
    // is mixed: rows 0..KH-2 come from window_kwm1_wire (combinational
    // BRAM mux); row KH-1 comes from bypass_reg.

    genvar g_kh, g_kw, g_ic;
    generate
        for (g_kh = 0; g_kh < KH; g_kh = g_kh + 1) begin : gen_win_kh
            for (g_kw = 0; g_kw < KW; g_kw = g_kw + 1) begin : gen_win_kw
                for (g_ic = 0; g_ic < IC; g_ic = g_ic + 1) begin : gen_win_ic
                    if (g_kw < KW - 1) begin : gen_shift_col
                        assign window_flat[(g_kh*KW*IC + g_kw*IC + g_ic)*8 +: 8] =
                            window[g_kh][g_kw][g_ic];
                    end else if (g_kh < KH - 1) begin : gen_bram_col
                        assign window_flat[(g_kh*KW*IC + g_kw*IC + g_ic)*8 +: 8] =
                            window_kwm1_wire[g_kh][g_ic*8 +: 8];
                    end else begin : gen_bypass_col
                        assign window_flat[(g_kh*KW*IC + g_kw*IC + g_ic)*8 +: 8] =
                            bypass_reg[g_ic];
                    end
                end
            end
        end
    endgenerate

endmodule
