// Equivalence TB: old serial add (node_add_25_old) vs new parallel add
// (node_add_25). Drives identical beats into both, honoring each one's
// ready_in protocol, and asserts the produced output beats are byte-identical
// in order. Exhaustive-ish over random INT8 pairs for all 16 channels.
`timescale 1ns/1ps

module add_equiv_tb;
    localparam OC = 16;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    // shared stimulus
    reg  [255:0] data_in;
    reg          ser_valid_in, par_valid_in;
    wire         ser_ready_in, par_ready_in;
    wire         ser_valid_out, par_valid_out;
    wire [127:0] ser_data_out, par_data_out;

    node_add_25_old u_ser (
        .clk(clk), .rst_n(rst_n), .valid_in(ser_valid_in),
        .ready_in(ser_ready_in), .data_in(data_in),
        .valid_out(ser_valid_out), .data_out(ser_data_out));

    node_add_25 u_par (
        .clk(clk), .rst_n(rst_n), .valid_in(par_valid_in),
        .ready_in(par_ready_in), .data_in(data_in),
        .valid_out(par_valid_out), .data_out(par_data_out));

    integer n, mism, ser_count, par_count, k;
    reg [127:0] ser_q [0:4095];
    reg [127:0] par_q [0:4095];
    reg [255:0] stim [0:1023];

    // capture serial outputs
    always @(posedge clk) if (rst_n && ser_valid_out) begin
        ser_q[ser_count] = ser_data_out; ser_count = ser_count + 1;
    end
    // capture parallel outputs
    always @(posedge clk) if (rst_n && par_valid_out) begin
        par_q[par_count] = par_data_out; par_count = par_count + 1;
    end

    integer seed, idx;
    initial begin
        seed = 12345;
        mism = 0; ser_count = 0; par_count = 0;
        ser_valid_in = 0; par_valid_in = 0; data_in = 0;
        // build 1024 random stimulus beats
        for (n = 0; n < 1024; n = n + 1) begin
            for (k = 0; k < 8; k = k + 1)
                stim[n][k*32 +: 32] = $random(seed);
        end
        // include some edge cases in first beats
        stim[0] = 256'h0;
        stim[1] = {128'h7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f, 128'h7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f};
        stim[2] = {128'h80808080808080808080808080808080, 128'h80808080808080808080808080808080};
        stim[3] = {128'h7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f, 128'h80808080808080808080808080808080};

        repeat (4) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // --- drive the SERIAL add: respect ready_in (one beat at a time) ---
        idx = 0;
        while (idx < 1024) begin
            @(negedge clk);
            if (ser_ready_in) begin
                data_in = stim[idx];
                ser_valid_in = 1;
                @(negedge clk);
                ser_valid_in = 0;
                idx = idx + 1;
                // wait for the serial FSM to return to ready
                while (!ser_ready_in) @(negedge clk);
            end else begin
                @(negedge clk);
            end
        end
        repeat (50) @(posedge clk);

        // --- drive the PARALLEL add: free-running, one beat/cycle ---
        for (idx = 0; idx < 1024; idx = idx + 1) begin
            @(negedge clk);
            data_in = stim[idx];
            par_valid_in = 1;
        end
        @(negedge clk);
        par_valid_in = 0;
        repeat (20) @(posedge clk);

        // --- compare ---
        $display("[equiv] ser_count=%0d par_count=%0d", ser_count, par_count);
        if (ser_count != 1024 || par_count != 1024) begin
            $display("[equiv] FAIL beat count mismatch (expected 1024 each)");
            $finish;
        end
        for (n = 0; n < 1024; n = n + 1) begin
            if (ser_q[n] !== par_q[n]) begin
                if (mism < 10)
                    $display("[equiv] MISMATCH beat %0d ser=%032x par=%032x", n, ser_q[n], par_q[n]);
                mism = mism + 1;
            end
        end
        if (mism == 0) $display("[equiv][summary] result=PASS mismatches=0 beats=1024");
        else           $display("[equiv][summary] result=FAIL mismatches=%0d beats=1024", mism);
        $finish;
    end
endmodule
