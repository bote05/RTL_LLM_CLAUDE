// node_conv_196 -- 7x7 stride-2 pad-3 stem conv (IC=3, OC=64, IH=IW=224, OH=OW=112).
// increase-throughput variant: MP raised from 4 to 8 (doubled MAC-lane parallelism in
// conv_datapath), with a 48-cycle (valid_out, data_out) shift register so the public
// pipeline-fill latency stays at 10190 cycles. Per-pixel cycles drop from 16*594=9504
// to 8*1182=9456 (48 saved/pixel x 12544 pixels ~= 602k cycles/frame lower II).
// Functional semantics preserved: conv_datapath is parameterized on MP; the per-OC
// arithmetic is independent of how lanes are scheduled.
// scale_factor=0.0013958489978772297 -> SCALE_MULT=11709, SCALE_SHIFT=23.

module node_conv_196 (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       valid_in,
    output wire                       ready_in,
    input  wire [23:0]                data_in,
    output wire                       valid_out,
    output wire [255:0]               data_out
);
    localparam integer IC        = 3;
    localparam integer OC        = 64;
    localparam integer IH        = 224;
    localparam integer IW        = 224;
    localparam integer OH        = 112;
    localparam integer OW        = 112;
    localparam integer KH        = 7;
    localparam integer KW        = 7;
    localparam integer SH        = 2;
    localparam integer SW        = 2;
    localparam integer PH        = 3;
    localparam integer PW        = 3;
    localparam integer K_TOTAL   = IC * KH * KW;
    // 2026-05-26 throughput fix: switched to conv_datapath_parallel which
    // has MP parallel multipliers (one MAC per lane per cycle, not one MAC
    // for all MP lanes serialized). With MP=8: per-pixel cycles drop from
    // OC_PASSES*(MP*K_TOTAL+6)=8*1182=9456 to OC_PASSES*(K_TOTAL+6)=8*153=1224.
    // 7.7× faster per output pixel. Total cycles/frame: 118.6M -> 15.3M.
    //
    // Weights are now read via WEIGHTS_PATH_WIDE (MP packed per line);
    // scripts/repack_weights_wide.py produces the .hex file.
    localparam integer MP          = 8;    // NOTE: this stem wrapper is SPECIAL (fixed 48-cyc output shift-reg, not the backpressured streamer); MP-increase DEADLOCKS it (verified 2026-05-30). Do NOT change MP here without reworking the wrapper.
    // 2026-05-26 truncation-bug fix (was previously emitting 512-bit
    // data_out, but nn2rtl_top.v wires it through a 256-bit wire which
    // silently truncates channels 32..63). Reverted to original Foundry
    // tiled-streaming contract: data_out is 256 bits, output is split into
    // BEATS_PER_PIXEL=2 beats. Beat 0 forwarded combinationally on the same
    // cycle as dp.valid_out (channels 0..31). Beat 1 registered and emitted
    // on the next cycle (channels 32..63). Downstream node_relu expects
    // exactly this 2-beat-per-pixel pattern. Throughput effect: relu now
    // groups 2 conv_196 beats as 1 logical pixel correctly, NOT halving the
    // chain rate from a phantom beat-pair miscount.

    localparam integer SCALE_MULT  = 11709;
    localparam integer SCALE_SHIFT = 23;

    reg started, start_pulse, pending_rearm;
    wire sched_out_frame_done;

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
                started       <= 1'b0;
                pending_rearm <= 1'b0;
            end
        end
    end

    wire stall_in = mac_busy;

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

    line_buf_window #(
        .IC(IC), .IW(IW), .IH(IH),
        .KH(KH), .KW(KW), .PW(PW), .PH(PH)
    ) lbw (
        .clk(clk), .rst_n(rst_n),
        .frame_start(start_pulse),
        .sched_in_row(sched_in_row),
        .sched_in_col(sched_in_col),
        .sched_needs_real_input(sched_needs_real_input),
        .sched_advance(sched_advance),
        .sched_output_fires(sched_output_fires),
        .valid_in(valid_in),
        .data_in(data_in),
        .window_flat(window_flat)
    );

    wire           dp_valid_out;
    wire [511:0]   dp_data_out;

    conv_datapath_mp_k #(.DSP_INPUT_PIPE(1),
        .IC(IC), .OC(OC), .KH(KH), .KW(KW),
        .K_TOTAL(K_TOTAL), .MP(MP),
        .MP_K(7),
        .SCALE_MULT(SCALE_MULT), .SCALE_SHIFT(SCALE_SHIFT),.SCALE_PATH("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_196_scale.mem"),
        .WEIGHTS_PATH("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_196_weights_mp_k_7.hex"),
        .BIAS_PATH("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_196_bias.hex")
    ) dp (
        .clk(clk), .rst_n(rst_n),
        .window_flat(window_flat),
        .start_mac(sched_output_fires),
        .valid_out(dp_valid_out),
        .data_out(dp_data_out),
        .mac_busy(mac_busy)
    );

    // 2-beat splitter: dp_data_out is 512 bits (64 channels × 8b). Downstream
    // consumes 256-bit beats with BEATS_PER_PIXEL=2 → emit channels 0..31 on
    // the same cycle as dp.valid_out (beat 0), then channels 32..63 on the
    // next cycle (beat 1). valid_out stays high for both cycles, never drops
    // on the same cycle as live data.
    reg [255:0] held_beat1;
    reg         emit_beat1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            held_beat1 <= 256'd0;
            emit_beat1 <= 1'b0;
        end else if (dp_valid_out) begin
            held_beat1 <= dp_data_out[511:256];
            emit_beat1 <= 1'b1;
        end else begin
            emit_beat1 <= 1'b0;
        end
    end
    assign valid_out = dp_valid_out | emit_beat1;
    assign data_out  = emit_beat1 ? held_beat1 : dp_data_out[255:0];
    assign ready_in  = sched_ready_in;

endmodule
