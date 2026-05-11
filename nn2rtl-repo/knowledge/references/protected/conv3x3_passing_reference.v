// Spatial 3x3 conv2d reference -- layer1_0_conv2 of ResNet-50.
// IC=64, OC=64, IH=IW=112, KH=KW=3, stride=1, padding=1, MP=4.
//
// Concrete instantiation of the split-architecture pattern documented in
// `knowledge/patterns/protected/03_conv3x3_pad1.md`. Foundry's job for any 3x3
// spatial conv is structural wiring only: pick the LayerIR's IC/OC/IH/IW/
// MP/scale_factor/$readmemh paths and adapt the localparam block + the
// two `$readmemh` parameter strings on the `conv_datapath` instantiation.
// The three library modules (coord_scheduler, line_buf_window,
// conv_datapath) are bundled into every iverilog / Verilator / Vivado
// invocation via `RTL_LIBRARY_SOURCES` in `mcp/tools.ts`, so no extra
// `include` or copy is needed.
//
// Latency contract for this layer
// -------------------------------
// K_TOTAL = IC*KH*KW = 576. MP = 4. OC_PASSES = ceil(64/4) = 16.
// pass_cycles = MP*K_TOTAL + 6 = 4*576 + 6 = 2310 (CONV_PIPELINE_STAGES = 6
// covers the 3-stage MAC pipeline + ST_BIAS + ST_SCALE + ST_OUTPUT).
// Spatial fill = max(KH-1-PH, 0) * (IW+PW) + max(KW-PW, 1)
//              = 1 * 113 + 2
//              = 115.
// Total = 115 + 16 * 2310 = 37075 cycles, matching
// `compute_conv2d_latency_cycles` in `scripts/golden_impl.py` for this
// LayerIR shape.
//
// Foundry MUST NOT
// ----------------
// - hand-write a line buffer, window, or MAC FSM (those live in `rtl_library/`)
// - declare `weights` / `biases` / `line_buf` / `window` arrays (the
//   library modules own them; the structural preflight knows to skip
//   the readmemh / line-buffer / window-register checks when
//   line_buf_window and conv_datapath are instantiated)
// - add `always @(posedge clk)` blocks except the single one for
//   `start_pulse` shown below
//
// Adapt to a new 3x3 LayerIR by changing:
// - the localparam block (IC, OC, IH, IW, OH, OW, SH, SW, PH, PW, MP)
// - SCALE_MULT, SCALE_SHIFT (run `compute_scale_approx(scale_factor)` in
//   `scripts/golden_impl.py`, mirrored by `computeScaleApprox` in
//   `sdk/orchestrate.ts` -- both pick the same constants, so RTL and
//   golden requantize agree bit-for-bit)
// - the two `$readmemh`-equivalent string parameters on `conv_datapath`

module layer1_0_conv2 (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       valid_in,
    output wire                       ready_in,
    input  wire [511:0]               data_in,
    output wire                       valid_out,
    output wire [511:0]               data_out
);
    // --- Parameters from LayerIR ---
    localparam integer IC        = 64;
    localparam integer OC        = 64;
    localparam integer IH        = 112;
    localparam integer IW        = 112;
    localparam integer OH        = 112;
    localparam integer OW        = 112;
    localparam integer KH        = 3;
    localparam integer KW        = 3;
    localparam integer SH        = 1;
    localparam integer SW        = 1;
    localparam integer PH        = 1;
    localparam integer PW        = 1;
    localparam integer K_TOTAL   = IC * KH * KW;       // 576
    localparam integer MP        = 4;

    // For scale_factor = 0.004047910194092914 the
    // (compute_scale_approx) sweep picks (SCALE_MULT=8489, SCALE_SHIFT=21),
    // matching what `sdk/orchestrate.ts::computeScaleApprox` would pick.
    localparam integer SCALE_MULT  = 8489;
    localparam integer SCALE_SHIFT = 21;

    // --- One-cycle start pulse on reset deassertion. Re-arms when
    //     sched_out_frame_done fires AND the last pixel's MAC pipeline
    //     has fully drained (mac_busy goes back to 0). Critically,
    //     start does NOT wait on valid_in: the static testbench waits
    //     for ready_in before asserting valid_in, and ready_in stays
    //     low until the scheduler is running (which requires `start`).
    //     Pulsing start on `!started` breaks that circular wait.
    //
    //     The `pending_rearm` latch + mac_busy gate fix a real bug: when
    //     `sched_out_frame_done` pulses at the cycle the LAST firing
    //     coord advances, conv_datapath has just begun the last pixel's
    //     ST_MAC stage. Without gating, started<-0 fires immediately
    //     and the next cycle pulses start_pulse, which routes to
    //     line_buf_window.frame_start and zeros the window MID-MAC,
    //     corrupting the very last output pixel. Waiting for mac_busy
    //     to drop ensures the last pixel finishes computing before the
    //     line buffer is cleared for the next frame.
    reg started, start_pulse, pending_rearm;
    wire sched_out_frame_done;

    // --- Scheduler <-> datapath wires (declared BEFORE the start_pulse
    //     always block so iverilog/Verilog-2001 elaboration sees mac_busy
    //     and sched_out_frame_done as already-declared identifiers when
    //     the always block references them). ---
    wire                              sched_needs_real_input;
    wire                              sched_ready_in;
    wire                              sched_output_fires;
    wire                              sched_advance;
    wire [$clog2(IH + PH + 1)-1:0]    sched_in_row;
    wire [$clog2(IW + PW + 1)-1:0]    sched_in_col;
    wire [$clog2(OH * OW + 1)-1:0]    sched_outputs_emitted;

    wire [KH*KW*IC*8-1:0]             window_flat;
    wire                              mac_busy;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            started       <= 1'b0;
            start_pulse   <= 1'b0;
            pending_rearm <= 1'b0;
        end else begin
            start_pulse <= 1'b0;
            if (sched_out_frame_done) begin
                pending_rearm <= 1'b1;
            end
            if (!started) begin
                started       <= 1'b1;
                start_pulse   <= 1'b1;
            end else if (pending_rearm && !mac_busy) begin
                // Last pixel's MAC has finished -- safe to clear
                // line_buf_window (via start_pulse -> frame_start) and
                // re-arm the scheduler for the next frame.
                started       <= 1'b0;
                pending_rearm <= 1'b0;
            end
        end
    end

    // stall_in is just mac_busy. No output_fires or mac_done plumbing --
    // the scheduler's registered `output_fires` pulse + its internal
    // `eff_stall = stall_in || output_fires` handle the firing-coord
    // freeze on its own.
    wire stall_in = mac_busy;

    // --- Coord scheduler ---
    coord_scheduler #(
        .IH(IH), .IW(IW), .OH(OH), .OW(OW),
        .KH(KH), .KW(KW), .SH(SH), .SW(SW),
        .PH(PH), .PW(PW)
    ) scheduler (
        .clk(clk), .rst_n(rst_n),
        .start(start_pulse),
        .stall_in(stall_in),
        .valid_in(valid_in),
        .ready_in(sched_ready_in),
        .needs_real_input(sched_needs_real_input),
        .in_row(sched_in_row),
        .in_col(sched_in_col),
        .output_fires(sched_output_fires),
        .advance(sched_advance),
        .in_frame_done(),
        .out_frame_done(sched_out_frame_done),
        .outputs_emitted(sched_outputs_emitted)
    );

    // --- Line buffer + shift-register window ---
    line_buf_window #(
        .IC(IC), .IW(IW), .IH(IH),
        .KH(KH), .KW(KW), .PW(PW), .PH(PH)
    ) lbw (
        .clk(clk), .rst_n(rst_n),
        .frame_start(start_pulse),         // clears line_buf + window between frames
        .sched_in_row(sched_in_row),
        .sched_in_col(sched_in_col),
        .sched_needs_real_input(sched_needs_real_input),
        .sched_advance(sched_advance),
        .sched_output_fires(sched_output_fires),
        .valid_in(valid_in),
        .data_in(data_in),
        .window_flat(window_flat)
    );

    // --- Datapath: MAC / bias / scale / output packing ---
    // The 3-stage MAC pipeline (sync ROM read, registered DSP multiply,
    // indexed accumulate) lives entirely inside conv_datapath.v. Vivado
    // infers a DSP48E1 for the multiplier via the `(* use_dsp = "yes" *)`
    // attribute on the registered `mul_q` inside the library module.
    // TEMPLATE: WEIGHTS_PATH/BIAS_PATH must come from the LayerIR sidecar
    // (sidecar.weights_path / sidecar.bias_path). Do not paste a literal
    // user-machine prefix; the orchestrator injects the correct absolute
    // path at generation time.
    conv_datapath #(
        .IC(IC), .OC(OC), .KH(KH), .KW(KW),
        .K_TOTAL(K_TOTAL), .MP(MP),
        .SCALE_MULT(SCALE_MULT), .SCALE_SHIFT(SCALE_SHIFT),
        .WEIGHTS_PATH("output/weights/<MODULE_ID>_weights.hex"),
        .BIAS_PATH("output/weights/<MODULE_ID>_bias.hex")
    ) dp (
        .clk(clk), .rst_n(rst_n),
        .window_flat(window_flat),
        .start_mac(sched_output_fires),
        .valid_out(valid_out),
        .data_out(data_out),
        .mac_busy(mac_busy)
    );

    // --- Top-level ready_in passes through the scheduler's handshake. ---
    assign ready_in = sched_ready_in;

endmodule
