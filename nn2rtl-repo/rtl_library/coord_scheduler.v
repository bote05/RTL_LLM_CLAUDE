// Handwritten coordinate FSM for spatial conv / maxpool.
//
// Why this exists: the coordinate math (row counter, column counter with
// IW-1+PW wrap, output-fires gate with stride/padding divisibility, and
// termination by outputs_emitted == OH*OW) is the single most bug-prone
// piece of the nn2rtl pipeline. Every Foundry attempt reinvents it with
// variable correctness. This module replaces that reinvention — for any
// conv2d with KH*KW > 1 or any maxpool, the generated top-level module
// must instantiate this module rather than roll its own coordinate logic.
//
// Universal: works for any IH/IW/OH/OW/KH/KW/SH/SW/PH/PW. No network-
// specific constants baked in.
//
// Termination: `out_frame_done` fires on the cycle AFTER
// `outputs_emitted == OH*OW`. Termination is NEVER gated on
// `in_row > IH-1+PH` (the old drain-row comparison) — that comparison has
// been the source of every spatial-conv drain bug in the pipeline.
//
// Handshake contract (this is what changed vs. the free-running draft):
//
//   The current coordinate can be in one of two regions:
//
//     1. REAL region  — in_row < IH && in_col < IW.  The coord corresponds
//        to an actual input pixel that upstream must deliver. The scheduler
//        exposes `needs_real_input = 1`, asserts `ready_in` combinationally
//        when it can accept the pixel (running && !stall_in), and advances
//        ONLY on a valid_in && ready_in handshake cycle. This keeps the
//        scheduler locked to the upstream data stream.
//
//     2. PADDED region — in_row >= IH || in_col >= IW.  The coord
//        corresponds to a synthesized zero-pad position; no upstream
//        handshake is required. The scheduler advances on any cycle where
//        `!stall_in && running`, free-running through the padding.
//
//   The external FSM, on observing `output_fires`, must drive `stall_in`
//   high combinationally (or on the same cycle) so the scheduler freezes
//   on the firing coordinate for the duration of the MAC computation.
//   On MAC completion the external FSM drops stall_in; the scheduler
//   advances past the firing coord, increments `outputs_emitted`, and
//   resumes streaming.

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
    // and arms the scheduler. On the cycle start is high, the scheduler
    // exposes (0,0); handshake or pad-step on subsequent cycles advances.
    input  wire                                     start,

    // External backpressure. Raised combinationally by the external FSM
    // when `output_fires` is observed (so the scheduler freezes on the
    // firing coordinate) and held high for the duration of the MAC /
    // bias / scale / output stages. Dropped on the last cycle of the
    // output stage so the scheduler advances past the firing coord.
    input  wire                                     stall_in,

    // Real-input handshake. Upstream asserts `valid_in` when it has a
    // real INT8 pixel on its data_in bus; the scheduler asserts
    // `ready_in` when the current coord is in the REAL region and the
    // pipeline is not stalled. Transfer happens on the cycle where both
    // are high.
    input  wire                                     valid_in,
    output wire                                     ready_in,

    // Combinational: high when the current coord is in the REAL region.
    // External FSM uses this to decide whether to gate `ready_in` into
    // its upstream handshake (real coord) or to step through the
    // scheduler without consuming upstream data (padded coord).
    output wire                                     needs_real_input,

    // Current input coordinate. Visible range is 0..IH-1+PH on in_row
    // and 0..IW-1+PW on in_col; values in [IH, IH+PH-1] / [IW, IW+PW-1]
    // correspond to padded positions on the bottom / right edges.
    output reg  [$clog2(IH + PH + 1)-1:0]           in_row,
    output reg  [$clog2(IW + PW + 1)-1:0]           in_col,

    // Pulses combinational on cycles where the current coord completes
    // an output pixel's receptive field. Held high while the scheduler
    // sits on the firing coord (stall_in high); drops on advance.
    output wire                                     output_fires,

    // Pulses high one cycle after the last input coord is absorbed.
    output reg                                      in_frame_done,

    // Pulses high one cycle after `outputs_emitted == OH * OW`.
    output reg                                      out_frame_done,

    // Saturating count of output pixels completed since the last `start`.
    output reg  [$clog2(OH * OW + 1)-1:0]           outputs_emitted
);

    // Derived constants.
    localparam integer MAX_IN_ROW = IH + PH - 1;  // inclusive upper bound on in_row
    localparam integer MAX_IN_COL = IW + PW - 1;  // inclusive upper bound on in_col
    localparam integer TOTAL_OUTS = OH * OW;

    // Running flag: active between `start` and (out_frame_done | in_frame_done).
    reg running;

    // Combinational output-fire predicate — derived from the CURRENT counters.
    // `row_num` / `col_num` may be negative (top/left padding region) or
    // beyond the output grid (right/bottom padding region); we filter those
    // with signed comparisons.
    wire signed [$clog2(IH + PH + 1):0] row_num_signed =
        $signed({1'b0, in_row}) + $signed(PH) - $signed(KH - 1);
    wire signed [$clog2(IW + PW + 1):0] col_num_signed =
        $signed({1'b0, in_col}) + $signed(PW) - $signed(KW - 1);

    wire row_in_range =
        (row_num_signed >= 0) &&
        (row_num_signed < $signed(OH * SH));
    wire col_in_range =
        (col_num_signed >= 0) &&
        (col_num_signed < $signed(OW * SW));

    // Stride divisibility. For SH == 1 this is trivially true.
    wire row_stride_ok = (SH == 1) ||
        ((row_num_signed % $signed(SH)) == 0);
    wire col_stride_ok = (SW == 1) ||
        ((col_num_signed % $signed(SW)) == 0);

    wire at_output_coord =
        running &&
        row_in_range && row_stride_ok &&
        col_in_range && col_stride_ok &&
        (outputs_emitted < TOTAL_OUTS);

    // Region classification + handshake.
    assign needs_real_input = running && (in_row < IH) && (in_col < IW);
    assign ready_in = needs_real_input && !stall_in;

    // Advance gating — the scheduler never advances when stall_in is high.
    // In the REAL region advance requires an explicit upstream handshake;
    // in the PADDED region advance free-runs. This is the core of the
    // handshake contract.
    wire handshake = needs_real_input && valid_in && !stall_in;
    wire pad_step  = running && !needs_real_input && !stall_in;
    wire advance   = handshake || pad_step;

    // `output_fires` is combinational on the current coord. It's held high
    // while the scheduler sits on a firing coord (stall_in high), so the
    // external FSM sees a persistent signal during the MAC pipeline.
    assign output_fires = at_output_coord;

    // Coordinate + termination update.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_row          <= 0;
            in_col          <= 0;
            in_frame_done   <= 1'b0;
            out_frame_done  <= 1'b0;
            outputs_emitted <= 0;
            running         <= 1'b0;
        end else begin
            // Default: pulse signals low each cycle.
            in_frame_done  <= 1'b0;
            out_frame_done <= 1'b0;

            if (start) begin
                in_row          <= 0;
                in_col          <= 0;
                outputs_emitted <= 0;
                running         <= 1'b1;
            end else if (advance) begin
                // Count this coord's output if we're leaving a firing coord.
                // Each firing coord counts exactly once because advance=1 is
                // the only path that both counts and moves past the coord.
                if (at_output_coord) begin
                    outputs_emitted <= outputs_emitted + 1;
                end

                // Advance the coordinate. Wrap at IW-1+PW on in_col; wrap
                // in_row after the last col. The in_col wrap handles the
                // right-edge padding inline — visible in_col range is
                // 0..MAX_IN_COL.
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

                // Terminate on outputs_emitted == TOTAL_OUTS. NOT on in_row
                // exiting MAX_IN_ROW. Output termination is driven by count,
                // not by input-side coordinate comparisons — this is the
                // universal contract, independent of network geometry.
                if (outputs_emitted + (at_output_coord ? 1 : 0) == TOTAL_OUTS) begin
                    out_frame_done <= 1'b1;
                    running        <= 1'b0;
                end
            end
        end
    end

endmodule
