`timescale 1ns/1ps

// skip_fifo_block_dut.v
// --------------------------------------------------------------------------
// Cycle-accurate residual-block timing model for the Task-04 Phase-B
// FIFO-sizing harness, REVISED for task 04c (throttled producer +
// BRAM-bounded FIFO with backpressure).
//
// Two deployment-level behaviours are modelled:
//
//   (a) `engine_busy → spatial_throttle` gates the producer. While the
//       engine is busy the producer stops pushing. The C++ tb drives
//       `throttle` from a periodic engine_busy schedule (k pulses of
//       `engine_worst_case_cycles` each).
//
//   (b) BRAM-side backpressure. The skip FIFO is bounded at
//       `fifo_depth`; when it is at capacity the producer stalls until
//       main pops something. Combined with (a) this lets us size the
//       FIFO by the U250's on-chip memory budget instead of by the
//       per-frame pipeline-fill latency.
//
// Warmup model
// ------------
// We collapse the skip-side pipeline latency into the producer (skip
// emit happens the same cycle as the producer push; the skip latency
// is folded into the main-vs-skip warmup gap). The main side waits
// (main_latency - skip_latency) REAL cycles after the first producer
// push before it can start consuming from the FIFO. After warmup, main
// emits one sample per cycle while the FIFO is non-empty.
//
// `eff_cycle` (effective time = cycles where the producer was actually
// pushing) is reported for audit but does not gate main consumption —
// that would create a false deadlock during a backpressure stall, since
// in real hardware samples already in the main pipeline keep advancing
// even when the producer is stalled.
// --------------------------------------------------------------------------

module skip_fifo_block_dut (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire        throttle,
    input  wire [31:0] main_latency,
    input  wire [31:0] skip_latency,
    input  wire [31:0] fifo_depth,
    input  wire [31:0] num_inputs,
    input  wire [31:0] cycle_budget,

    output reg         done,
    output reg  [31:0] cycles_run,
    output reg  [31:0] eff_cycles_run,
    output reg  [31:0] peak_fifo_occupancy,
    output reg         overflow_detected,
    output reg         deadlock_detected,
    output reg  [31:0] deadlock_cycle,
    output reg  [31:0] outputs_produced,
    output reg  [31:0] throttled_cycles,
    output reg  [31:0] backpressure_cycles
);

    reg        active;
    reg  [31:0] main_lat_l, skip_lat_l, fifo_d_l, n_in_l, budget_l;
    reg  [31:0] cycle_count;
    reg  [31:0] eff_cycle;

    reg  [31:0] inputs_emitted;
    reg  [31:0] main_emitted_count;
    reg  [31:0] skip_emitted_count;
    reg  [31:0] fifo_count;

    reg  [31:0] stall_cycles;
    localparam [31:0] STALL_THRESHOLD = 32'd16384;

    // Warmup state — count real cycles since the first producer push.
    // main_can_emit kicks in once that count reaches (main_lat - skip_lat).
    reg        first_push_seen;
    reg  [31:0] first_push_cycle;
    wire [31:0] warmup_gap = (main_lat_l > skip_lat_l) ?
                              (main_lat_l - skip_lat_l) : 32'd0;
    wire [31:0] cycles_since_first_push =
                cycle_count - first_push_cycle;
    wire        main_warmup_done = first_push_seen &&
                                   (cycles_since_first_push >= warmup_gap);

    // BRAM-side backpressure: producer stalls when FIFO at capacity.
    wire fifo_full       = (fifo_count >= fifo_d_l);
    wire active_step     = active && !throttle && !fifo_full;
    wire producer_active = active_step && (inputs_emitted < n_in_l);
    // Skip emit collapses into the producer push — see header.
    wire skip_can_emit   = producer_active;
    // Main emit is independent of throttle/backpressure once warmed up:
    // samples already in the main pipeline keep draining even when the
    // producer is stalled.
    wire main_can_emit   = active && main_warmup_done &&
                           (main_emitted_count < n_in_l);
    wire fifo_nonempty   = (fifo_count != 32'd0);
    wire add_fires       = main_can_emit & fifo_nonempty;
    wire skip_in_flight  = (skip_emitted_count < n_in_l);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active              <= 1'b0;
            main_lat_l          <= 32'd0;
            skip_lat_l          <= 32'd0;
            fifo_d_l            <= 32'd0;
            n_in_l              <= 32'd0;
            budget_l            <= 32'd0;
            cycle_count         <= 32'd0;
            eff_cycle           <= 32'd0;
            inputs_emitted      <= 32'd0;
            main_emitted_count  <= 32'd0;
            skip_emitted_count  <= 32'd0;
            fifo_count          <= 32'd0;
            stall_cycles        <= 32'd0;
            done                <= 1'b0;
            cycles_run          <= 32'd0;
            eff_cycles_run      <= 32'd0;
            peak_fifo_occupancy <= 32'd0;
            overflow_detected   <= 1'b0;
            deadlock_detected   <= 1'b0;
            deadlock_cycle      <= 32'd0;
            outputs_produced    <= 32'd0;
            throttled_cycles    <= 32'd0;
            backpressure_cycles <= 32'd0;
            first_push_seen     <= 1'b0;
            first_push_cycle    <= 32'd0;
        end else begin
            if (!active && start) begin
                active           <= 1'b1;
                main_lat_l       <= main_latency;
                skip_lat_l       <= skip_latency;
                fifo_d_l         <= fifo_depth;
                n_in_l           <= num_inputs;
                budget_l         <= cycle_budget;
                cycle_count      <= 32'd0;
                eff_cycle        <= 32'd0;
                first_push_seen  <= 1'b0;
                first_push_cycle <= 32'd0;
            end else if (active && !done) begin
                cycle_count <= cycle_count + 32'd1;
                cycles_run  <= cycle_count + 32'd1;

                if (throttle) begin
                    throttled_cycles <= throttled_cycles + 32'd1;
                end else if (fifo_full && (inputs_emitted < n_in_l)) begin
                    backpressure_cycles <= backpressure_cycles + 32'd1;
                end else if (producer_active) begin
                    eff_cycle      <= eff_cycle + 32'd1;
                    eff_cycles_run <= eff_cycle + 32'd1;
                end

                if (producer_active) begin
                    inputs_emitted <= inputs_emitted + 32'd1;
                    if (!first_push_seen) begin
                        first_push_seen  <= 1'b1;
                        first_push_cycle <= cycle_count;
                    end
                end

                // Skip emit collapses into the producer push: a sample
                // pushed this cycle also enters the FIFO this cycle.
                if (skip_can_emit) begin
                    skip_emitted_count <= skip_emitted_count + 32'd1;
                end

                if (main_can_emit) begin
                    if (fifo_nonempty) begin
                        main_emitted_count <= main_emitted_count + 32'd1;
                        outputs_produced   <= outputs_produced + 32'd1;
                        stall_cycles       <= 32'd0;
                    end else if (!skip_in_flight) begin
                        stall_cycles <= stall_cycles + 32'd1;
                    end
                end

                case ({skip_can_emit, add_fires})
                    2'b01: fifo_count <= fifo_count - 32'd1;
                    2'b10: fifo_count <= fifo_count + 32'd1;
                    2'b11: fifo_count <= fifo_count;
                    default: ;
                endcase

                if ((fifo_count + (skip_can_emit ? 32'd1 : 32'd0) -
                                 (add_fires      ? 32'd1 : 32'd0))
                    > peak_fifo_occupancy) begin
                    peak_fifo_occupancy <=
                        fifo_count + (skip_can_emit ? 32'd1 : 32'd0)
                                   - (add_fires      ? 32'd1 : 32'd0);
                end

                // Overflow watchdog — should never trigger because
                // producer_active is gated on !fifo_full.
                if ((fifo_count + (skip_can_emit ? 32'd1 : 32'd0))
                    > fifo_d_l) begin
                    overflow_detected <= 1'b1;
                end

                if (stall_cycles >= STALL_THRESHOLD && !deadlock_detected) begin
                    deadlock_detected <= 1'b1;
                    deadlock_cycle    <= cycle_count;
                end

                if ((main_emitted_count == n_in_l && n_in_l != 32'd0) ||
                    (cycle_count + 32'd1 >= budget_l)) begin
                    done <= 1'b1;
                end
            end
        end
    end

endmodule
