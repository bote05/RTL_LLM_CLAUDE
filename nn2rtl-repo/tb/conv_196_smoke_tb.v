// Minimal smoke TB: drive 1 frame into node_conv_196 (with conv_datapath_mp_k),
// count outputs. Fail-fast if no outputs after a small cycle budget.
`timescale 1ns / 1ps

module conv_196_smoke_tb;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0;

    reg          valid_in = 0;
    wire         ready_in;
    reg  [23:0]  data_in = 24'h010203;
    wire         valid_out;
    wire [255:0] data_out;

    node_conv_196 dut (
        .clk(clk), .rst_n(rst_n),
        .valid_in(valid_in),
        .ready_in(ready_in),
        .data_in(data_in),
        .valid_out(valid_out),
        .data_out(data_out)
    );

    integer cycle_count = 0;
    integer input_count = 0;
    integer output_count = 0;
    integer first_output_cycle = -1;
    integer last_output_cycle = 0;

    // Drive 224x224 = 50176 ACTUAL handshakes (valid_in && ready_in), then drop valid_in.
    // Count only completed handshakes, like the e2e TB does. The previous smoke
    // version counted ready_in cycles which over-counted under fast-mac timing.
    always @(posedge clk) begin
        cycle_count <= cycle_count + 1;
        if (cycle_count == 5) rst_n <= 1;
        // Always hold valid_in high until we've delivered 50176 handshakes
        if (rst_n && input_count < 50176) valid_in <= 1'b1;
        else                              valid_in <= 1'b0;
        // Increment ONLY on completed handshake (matches scheduler advance)
        if (valid_in && ready_in && input_count < 50176) begin
            input_count <= input_count + 1;
            if (input_count == 49999) begin
                $display("[smoke] all 50176 inputs accepted at cycle %0d", cycle_count);
            end
        end

        if (valid_out) begin
            if (first_output_cycle == -1) begin
                first_output_cycle = cycle_count;
                $display("[smoke] FIRST output at cycle %0d", cycle_count);
            end
            output_count <= output_count + 1;
            last_output_cycle = cycle_count;
        end

        // Status every 100k cycles, with internal probe
        if (cycle_count > 0 && cycle_count % 100000 == 0) begin
            $display("[smoke] cyc=%0d in=%0d/50176 out=%0d/25088 dp_state=%0d k=%0d oc=%0d mac_busy=%b sched_out_done=%b started=%b pend=%b ready_in=%b",
                     cycle_count, input_count, output_count,
                     dut.dp.state, dut.dp.k_group, dut.dp.oc_group, dut.mac_busy,
                     dut.sched_out_frame_done, dut.started, dut.pending_rearm,
                     ready_in);
        end

        // Hard stop
        if (cycle_count > 5000000) begin
            $display("[smoke] TIMEOUT at cyc=%0d in=%0d out=%0d first_out=%0d last_out=%0d",
                     cycle_count, input_count, output_count, first_output_cycle, last_output_cycle);
            $finish;
        end

        // Done condition: 25088 output beats (112*112*2 BEATS_PER_PIXEL)
        if (output_count >= 25088) begin
            $display("[smoke] DONE: %0d outputs at cyc=%0d first=%0d", output_count, cycle_count, first_output_cycle);
            $finish;
        end
    end
endmodule
