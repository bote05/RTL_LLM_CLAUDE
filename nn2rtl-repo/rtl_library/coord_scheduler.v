// Handwritten coordinate FSM for spatial conv / maxpool.
//
// Universal: works for any IH/IW/OH/OW/KH/KW/SH/SW/PH/PW. No network-
// specific constants baked in.
//
// --------------------------------------------------------------------------
// Firing semantics (pixel-delivery-safe).
// --------------------------------------------------------------------------
//
// The key subtlety is that a firing coord (a spatial-conv output pixel is
// ready) is ALSO a real-input coord whose pixel must be delivered before
// the MAC can use the correct receptive field. An older design gated
// `advance` on `!at_output_coord`, which blocked the handshake at the
// firing coord and caused pixel loss. This version fixes that:
//
//   * Scheduler advances every cycle that handshake or pad_step fires
//     (no at_output_coord gate). When the advance lands on a firing
//     coord, the cycle of advance also writes the corresponding pixel
//     into line_buf_window via the external handshake (valid_in &&
//     ready_in). Pixel is delivered; window updates one cycle later via
//     the registered shift inside line_buf_window.
//
//   * `output_fires` is a REGISTERED one-cycle pulse, asserted the
//     cycle AFTER an advance that moved past a firing coord. External
//     consumers (datapath) see `output_fires = 1` for exactly one cycle
//     with a stable (post-advance) window.
//
//   * Internally, the scheduler treats `output_fires` as part of its
//     effective stall. The cycle `output_fires` pulses, the scheduler
//     does not advance further — giving the datapath time to latch
//     start_mac and transition to ST_MAC (on the next posedge). Once
//     mac_busy rises, external stall_in keeps the scheduler frozen
//     across the MAC pipeline. When mac_busy drops (datapath returns
//     to ST_IDLE), the scheduler resumes normal advance.
//
//   * No `mac_done` input and no `release_advance` are needed —
//     scheduler advance is purely driven by handshake/pad_step under
//     the `eff_stall = stall_in || output_fires` gate.
//
// Termination: `out_frame_done` fires the cycle after `outputs_emitted
// == OH*OW`. Never gated on `in_row > IH-1+PH`.

module coord_scheduler #(
    parameter integer IH = 32,
    parameter integer IW = 32,
    parameter integer OH = 16,
    parameter integer OW = 16,
    parameter integer KH = 3,
    parameter integer KW = 3,
    parameter integer SH = 1,
    parameter integer SW = 1,
    parameter integer PH = 1,
    parameter integer PW = 1
) (
    input  wire                                     clk,
    input  wire                                     rst_n,

    // One-cycle pulse to begin a new input frame. Resets counters to (0,0)
    // and arms the scheduler.
    input  wire                                     start,

    // External backpressure from the datapath. Drive combinationally from
    // `mac_busy` only. The scheduler internally also stalls for one cycle
    // on the output_fires pulse — callers do not need to include that.
    input  wire                                     stall_in,

    // Real-input handshake. Upstream asserts `valid_in` when it has a
    // pixel ready; scheduler asserts `ready_in` when able to accept.
    input  wire                                     valid_in,
    output wire                                     ready_in,

    // Combinational: high when the current coord is in the REAL region
    // (needs an upstream pixel). Low in padded regions (free-running).
    output wire                                     needs_real_input,

    // Current input coordinate.
    output reg  [$clog2(IH + PH + 1)-1:0]           in_row,
    output reg  [$clog2(IW + PW + 1)-1:0]           in_col,

    // One-cycle pulse emitted the cycle AFTER an advance past a firing
    // coord. Drive this into the datapath's `start_mac` input directly.
    output reg                                      output_fires,

    // Combinational: high on cycles where the scheduler actually advances
    // (either handshake or pad_step this cycle). Line_buf_window consumes
    // this to know when to shift the window and write line_buf.
    output wire                                     advance,

    // Frame boundaries. Each pulses for exactly one cycle.
    output reg                                      in_frame_done,
    output reg                                      out_frame_done,

    // Saturating count of output pixels completed since the last `start`.
    output reg  [$clog2(OH * OW + 1)-1:0]           outputs_emitted
);

    // Derived constants.
    localparam integer MAX_IN_ROW = IH + PH - 1;
    localparam integer MAX_IN_COL = IW + PW - 1;
    localparam integer TOTAL_OUTS = OH * OW;

    reg running;

    // Combinational firing predicate on CURRENT coord (pre-edge values).
    wire signed [$clog2(IH + PH + 1):0] row_num_signed =
        $signed({1'b0, in_row}) + $signed(PH) - $signed(KH - 1);
    wire signed [$clog2(IW + PW + 1):0] col_num_signed =
        $signed({1'b0, in_col}) + $signed(PW) - $signed(KW - 1);

    wire row_in_range =
        (row_num_signed >= 0) && (row_num_signed < $signed(OH * SH));
    wire col_in_range =
        (col_num_signed >= 0) && (col_num_signed < $signed(OW * SW));
    wire row_stride_ok = (SH == 1) || ((row_num_signed % $signed(SH)) == 0);
    wire col_stride_ok = (SW == 1) || ((col_num_signed % $signed(SW)) == 0);

    wire at_output_coord =
        running &&
        row_in_range && row_stride_ok &&
        col_in_range && col_stride_ok &&
        (outputs_emitted < TOTAL_OUTS);

    // Region + handshake.
    assign needs_real_input = running && (in_row < IH) && (in_col < IW);
    assign ready_in         = needs_real_input && !stall_in && !output_fires;

    // Effective stall. Scheduler freezes on any external stall AND on the
    // registered output_fires cycle (gives the datapath time to see
    // start_mac and enter ST_MAC).
    wire eff_stall = stall_in || output_fires;

    // Advance — purely handshake or pad_step, under eff_stall gating.
    // NO at_output_coord gate: the scheduler DOES advance past a firing
    // coord, because the cycle it advances is also the cycle the pixel
    // at that coord gets handshaked and delivered to line_buf_window.
    wire handshake = needs_real_input && valid_in && !eff_stall;
    wire pad_step  = running && !needs_real_input && !eff_stall;
    assign advance = handshake || pad_step;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_row          <= 0;
            in_col          <= 0;
            in_frame_done   <= 1'b0;
            out_frame_done  <= 1'b0;
            output_fires    <= 1'b0;
            outputs_emitted <= 0;
            running         <= 1'b0;
        end else begin
            // Defaults — pulses drop to 0 unless re-set this cycle.
            in_frame_done  <= 1'b0;
            out_frame_done <= 1'b0;
            output_fires   <= 1'b0;

            if (start) begin
                in_row          <= 0;
                in_col          <= 0;
                outputs_emitted <= 0;
                output_fires    <= 1'b0;
                running         <= 1'b1;
            end else if (advance) begin
                // Count the output at the coord we JUST consumed (pre-edge
                // value of at_output_coord). Pulse output_fires the cycle
                // AFTER this advance, so downstream sees it paired with a
                // stable post-advance window.
                if (at_output_coord) begin
                    outputs_emitted <= outputs_emitted + 1;
                    output_fires    <= 1'b1;
                end

                // Advance the coordinate.
                if (in_col == MAX_IN_COL[$clog2(IW + PW + 1)-1:0]) begin
                    in_col <= 0;
                    if (in_row == MAX_IN_ROW[$clog2(IH + PH + 1)-1:0]) begin
                        in_row        <= 0;
                        in_frame_done <= 1'b1;
                    end else begin
                        in_row <= in_row + 1;
                    end
                end else begin
                    in_col <= in_col + 1;
                end

                // Frame termination on count.
                if (outputs_emitted + (at_output_coord ? 1 : 0) == TOTAL_OUTS) begin
                    out_frame_done <= 1'b1;
                    running        <= 1'b0;
                end
            end
        end
    end

endmodule
