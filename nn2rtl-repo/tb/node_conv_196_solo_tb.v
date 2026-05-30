// Solo testbench for node_conv_196 — drives the full module wrapper
// (coord_scheduler + line_buf_window + conv_datapath_parallel) with
// continuous input and measures the actual cycles-per-output rate.
//
// Goal: distinguish wrapper-level throttle vs chain-level throttle.
// If output rate matches parallel datapath's isolation rate (~1224
// cycles/pixel), the wrapper isn't the bottleneck. If it matches the
// original (~9456), coord_scheduler or line_buf is throttling.
//
// Build: iverilog -g2012 -DRESNET_FIRST_CONV ...

`timescale 1ns / 1ps

module node_conv_196_solo_tb;

    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0;

    reg          valid_in = 0;
    wire         ready_in;
    reg  [23:0]  data_in = 24'h010203;
    wire         valid_out;
    wire [511:0] data_out;

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
    integer first_input_cycle = -1;
    integer first_output_cycle = -1;
    integer last_output_cycle = 0;
    integer scheduler_done_cycle = 0;

    // Drive valid_in continuously after reset.
    always @(posedge clk) begin
        cycle_count <= cycle_count + 1;

        if (cycle_count == 5) begin
            rst_n <= 1;
        end

        if (cycle_count > 10) begin
            valid_in <= 1;
            // Vary data_in slightly so window has interesting values.
            data_in <= {data_in[15:0], data_in[23:16]} + 24'h010101;
        end

        // Track handshakes.
        if (valid_in && ready_in) begin
            if (first_input_cycle < 0) first_input_cycle <= cycle_count;
            input_count <= input_count + 1;
        end

        if (valid_out) begin
            if (first_output_cycle < 0) first_output_cycle <= cycle_count;
            last_output_cycle <= cycle_count;
            output_count <= output_count + 1;
        end
    end

    // Periodic status print.
    initial begin
        // Wait for sim to actually start (cycle_count > 0)
        @(posedge clk);
        @(posedge clk);

        // Status print every 200K cycles, up to 4M cycles total.
        // 4M cycles is plenty to see steady-state rate.
        forever begin
            repeat (200000) @(posedge clk);
            $display("[tb] cycle=%0d input=%0d output=%0d ready_in=%0d valid_out_recent=%0d",
                     cycle_count, input_count, output_count, ready_in,
                     valid_out);
            if (cycle_count >= 4000000 || output_count >= 200) begin
                $display("[tb] === FINAL ===");
                $display("[tb] cycle_count        = %0d", cycle_count);
                $display("[tb] input_count        = %0d", input_count);
                $display("[tb] output_count       = %0d", output_count);
                $display("[tb] first_input_cycle  = %0d", first_input_cycle);
                $display("[tb] first_output_cycle = %0d", first_output_cycle);
                if (output_count > 1) begin
                    $display("[tb] steady-state cycles per output = %0d",
                             (last_output_cycle - first_output_cycle) / (output_count - 1));
                end
                if (input_count > 1 && first_input_cycle >= 0) begin
                    $display("[tb] steady-state cycles per input  = %0d",
                             (cycle_count - first_input_cycle) / input_count);
                end
                $finish;
            end
        end
    end

endmodule
