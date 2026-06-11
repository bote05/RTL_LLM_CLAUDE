// node_conv_246: conv2d 256->256, 28x28 -> 14x14, 3x3 stride 2 padding 1.
// tiled-streaming contract (256-bit bus), channel_tile=32, MP=4.
// Split-module architecture (coord_scheduler + line_buf_window + conv_datapath).
module node_conv_246 (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              valid_in,
    output wire              ready_in,
    input  wire [255:0]      data_in,
    output wire              valid_out,
    output wire [255:0]      data_out
);
    localparam integer IC          = 256;
    localparam integer OC          = 256;
    localparam integer IH          = 28;
    localparam integer IW          = 28;
    localparam integer OH          = 14;
    localparam integer OW          = 14;
    localparam integer KH          = 3;
    localparam integer KW          = 3;
    localparam integer SH          = 2;
    localparam integer SW          = 2;
    localparam integer PH          = 1;
    localparam integer PW          = 1;
    localparam integer K_TOTAL     = IC * KH * KW;
    localparam integer MP          = 4;
    localparam integer CHANNEL_TILE = 32;
    localparam integer IN_BEATS    = IC / CHANNEL_TILE;
    localparam integer OUT_BEATS   = OC / CHANNEL_TILE;
    localparam integer TILE_BITS   = CHANNEL_TILE * 8;
    localparam integer SCALE_MULT  = 19599;
    localparam integer SCALE_SHIFT = 23;

    reg [IC*8-1:0] in_pixel_reg;
    reg [3:0]      in_beat_cnt;
    reg            pix_valid;
    wire sched_ready_in_w;
    wire pix_accept = pix_valid && sched_ready_in_w;
    // [INVARIANT:READY_IN_GATING]
    assign ready_in = !pix_valid || pix_accept;

    // Sync-only memory write (no reset clause) so Vivado infers RAM cleanly
    // for the indexed slice write into in_pixel_reg and the preflight rule
    // activation_memory_in_async_reset_block stays satisfied. Control regs
    // (in_beat_cnt, pix_valid) keep their async reset in the block below.
    always @(posedge clk) begin
        if (valid_in && ready_in) begin
            in_pixel_reg[in_beat_cnt*TILE_BITS +: TILE_BITS] <= data_in;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin in_beat_cnt <= 4'd0; pix_valid <= 1'b0; end
        else begin
            if (pix_accept) pix_valid <= 1'b0;
            if (valid_in && ready_in) begin
                if (in_beat_cnt == IN_BEATS - 1) begin in_beat_cnt <= 4'd0; pix_valid <= 1'b1; end
                else in_beat_cnt <= in_beat_cnt + 4'd1;
            end
        end
    end

    reg started, start_pulse, pending_rearm;
    wire sched_out_frame_done; wire mac_busy;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin started<=1'b0; start_pulse<=1'b0; pending_rearm<=1'b0; end
        else begin
            start_pulse <= 1'b0;
            if (sched_out_frame_done) pending_rearm <= 1'b1;
            if (!started) begin started<=1'b1; start_pulse<=1'b1; end
            else if (pending_rearm && !mac_busy) begin started<=1'b0; pending_rearm<=1'b0; end
        end
    end

    wire sched_needs_real_input, sched_output_fires, sched_advance;
    wire [$clog2(IH+PH+1)-1:0] sched_in_row;
    wire [$clog2(IW+PW+1)-1:0] sched_in_col;
    wire [$clog2(OH*OW+1)-1:0] sched_outputs_emitted;
    wire [KH*KW*IC*8-1:0] window_flat;
    wire stall_in = mac_busy;

    coord_scheduler #(.IH(IH),.IW(IW),.OH(OH),.OW(OW),.KH(KH),.KW(KW),.SH(SH),.SW(SW),.PH(PH),.PW(PW)) scheduler (.clk(clk),.rst_n(rst_n),.start(start_pulse),.stall_in(stall_in),.valid_in(pix_valid),.ready_in(sched_ready_in_w),.needs_real_input(sched_needs_real_input),.in_row(sched_in_row),.in_col(sched_in_col),.output_fires(sched_output_fires),.advance(sched_advance),.in_frame_done(),.out_frame_done(sched_out_frame_done),.outputs_emitted(sched_outputs_emitted));

    line_buf_window #(.IC(IC),.IW(IW),.IH(IH),.KH(KH),.KW(KW),.PW(PW),.PH(PH)) lbw (.clk(clk),.rst_n(rst_n),.frame_start(start_pulse),.sched_in_row(sched_in_row),.sched_in_col(sched_in_col),.sched_needs_real_input(sched_needs_real_input),.sched_advance(sched_advance),.sched_output_fires(sched_output_fires),.valid_in(pix_valid),.data_in(in_pixel_reg),.window_flat(window_flat));

    wire [OC*8-1:0] dp_data_out; wire dp_valid_out;
    conv_datapath_mp_k #(.DSP_INPUT_PIPE(1),.IC(IC),.OC(OC),.KH(KH),.KW(KW),.K_TOTAL(K_TOTAL),.MP(MP),
        .MP_K(9),
        .SCALE_MULT(SCALE_MULT),.SCALE_SHIFT(SCALE_SHIFT),.WEIGHTS_PATH("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_246_weights_mp_k_9.hex"),.BIAS_PATH("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/node_conv_246_bias.hex")) dp (.clk(clk),.rst_n(rst_n),.window_flat(window_flat),.start_mac(sched_output_fires),.valid_out(dp_valid_out),.data_out(dp_data_out),.mac_busy(mac_busy));

    reg [OC*8-1:0] out_buf; reg [3:0] out_beat_cnt; reg out_streaming;
    // [INVARIANT:VALID_OUT_LATENCY]
    assign valid_out = out_streaming || dp_valid_out;
    assign data_out = out_streaming ? out_buf[out_beat_cnt*TILE_BITS +: TILE_BITS] : dp_data_out[0 +: TILE_BITS];
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin out_buf<={(OC*8){1'b0}}; out_beat_cnt<=4'd0; out_streaming<=1'b0; end
        else begin
            if (dp_valid_out && !out_streaming) begin out_buf <= dp_data_out; out_beat_cnt <= 4'd1; out_streaming <= (OUT_BEATS>1); end
            else if (out_streaming) begin
                if (out_beat_cnt == OUT_BEATS-1) begin out_streaming<=1'b0; out_beat_cnt<=4'd0; end
                else out_beat_cnt <= out_beat_cnt + 4'd1;
            end
        end
    end
endmodule
