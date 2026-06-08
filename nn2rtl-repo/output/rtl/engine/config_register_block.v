`timescale 1ns/1ps

// config_register_block.v
// --------------------------------------------------------------------------
// Wave 2 task 10 sub-block. Port list is locked by
// docs/agent_tasks/00_engine_skeleton_spec_PORTS.md `## SUBBLOCK:
// config_register_block`. Spec: docs/agent_tasks/10_engine_config_register_block.md.
//
// AXI4-Lite slave that holds the engine's per-layer configuration.
// AXI4-Lite byte offsets are the SOURCE OF TRUTH set by task 10's register
// map and are what scripts/build_scheduler.py writes via its AXI4-Lite
// master FSM. Field-bit positions within each 32-bit register exactly match
// what the scheduler packs into s_axil_wdata.
//
//   0x00 INPUT_CHANNELS       wdata[11:0]  -> cfg_ic
//   0x04 OUTPUT_CHANNELS      wdata[11:0]  -> cfg_oc
//   0x08 KERNEL_H_W           wdata[6:4]   -> cfg_kh,  wdata[2:0]  -> cfg_kw
//   0x0C STRIDE_H_W           wdata[5:3]   -> cfg_stride_h,  wdata[2:0] -> cfg_stride_w
//   0x10 PADDING_H_W          wdata[5:3]   -> cfg_pad_h,     wdata[2:0] -> cfg_pad_w
//   0x14 INPUT_H_W            wdata[23:16] -> cfg_ih,        wdata[7:0] -> cfg_iw
//   0x18 OUTPUT_H_W           wdata[23:16] -> cfg_oh,        wdata[7:0] -> cfg_ow
//   0x1C WEIGHT_BASE_WORD     wdata[21:0]  -> cfg_weight_uram_base
//   0x20 BIAS_BASE_WORD       wdata[21:0]  -> cfg_bias_uram_base
//   0x24 SCALE_MULT           wdata[31:0]  -> cfg_scale_mult (32-bit; widened from 16b in 13a audit Fix 2)
//   0x28 SCALE_SHIFT_AND_ZP   wdata[5:0]   -> cfg_scale_shift  (wdata[13:6] = zero_point, stored
//                                            but no port -- PORTS spec does not export it)
//   0x2C CONTROL              wdata[0]=1   -> engine_start_pulse (if !engine_busy_in)
//                             reads:  bit[1]=engine_busy_in (BUSY mirror)
//   0x30 STATUS               read-only:  bit[0]=engine_done_in, bit[1]=engine_busy_in
//   0x34 ACT_IN_BASE          wdata[15:0]  -> cfg_act_in_bram_base   (extension over task 10;
//                                            scheduler writes 0 until task 03 is regenerated
//                                            to cover the BRAM bases)
//   0x38 ACT_OUT_BASE         wdata[15:0]  -> cfg_act_out_bram_base
//
// Read-back returns the full 32-bit value last written (so a scoreboard
// can write-then-read each register and expect bit-exact match). Each
// config register lives as a 32-bit reg; the narrower cfg_* outputs are
// combinational slices of the storage.
//
// engine_start_pulse rises for exactly one cycle either when the master
// writes 1 to CONTROL.bit[0] or when engine_start_ext is asserted, and
// only if engine_busy_in is low at the trigger cycle. AXI write and
// engine_start_ext arriving the same cycle produce a single pulse, not
// two (the pulse register is a single-cycle latch).
//
// All registers reset to 0 on !rst_n; engine_start_pulse stays low except
// for its one-cycle assertion.
//
// Universal-bugs rule (knowledge/patterns/protected/08_common_bugs.md
// §"Array memory write in async-reset always block") does NOT fire: every
// register is a flat scalar `reg [W-1:0] name`, never an indexed array.
// --------------------------------------------------------------------------

module config_register_block (
    input  wire         clk,
    input  wire         rst_n,
    // AXI4-Lite slave -- write channels
    input  wire         s_axil_awvalid,
    output wire         s_axil_awready,
    input  wire [7:0]   s_axil_awaddr,
    input  wire         s_axil_wvalid,
    output wire         s_axil_wready,
    input  wire [31:0]  s_axil_wdata,
    input  wire [3:0]   s_axil_wstrb,
    output wire         s_axil_bvalid,
    input  wire         s_axil_bready,
    output wire [1:0]   s_axil_bresp,
    // AXI4-Lite slave -- read channels
    input  wire         s_axil_arvalid,
    output wire         s_axil_arready,
    input  wire [7:0]   s_axil_araddr,
    output wire         s_axil_rvalid,
    input  wire         s_axil_rready,
    output wire [31:0]  s_axil_rdata,
    output wire [1:0]   s_axil_rresp,
    // Engine handshake
    input  wire         engine_start_ext,
    input  wire         engine_busy_in,
    input  wire         engine_done_in,
    output wire         engine_busy_ext,
    output wire         engine_done_ext,
    output wire         engine_start_pulse,
    // Decoded configuration outputs (consumed by address_generator / requant_pipeline)
    output wire [11:0]  cfg_ic,
    output wire [11:0]  cfg_oc,
    output wire [2:0]   cfg_kh,
    output wire [2:0]   cfg_kw,
    output wire [7:0]   cfg_ih,
    output wire [7:0]   cfg_iw,
    output wire [7:0]   cfg_oh,
    output wire [7:0]   cfg_ow,
    output wire [2:0]   cfg_stride_h,
    output wire [2:0]   cfg_stride_w,
    output wire [2:0]   cfg_pad_h,
    output wire [2:0]   cfg_pad_w,
    output wire [31:0]  cfg_scale_mult,
    output wire [5:0]   cfg_scale_shift,
    output wire [21:0]  cfg_weight_uram_base,
    output wire [21:0]  cfg_bias_uram_base,
    output wire [15:0]  cfg_act_in_bram_base,
    output wire [15:0]  cfg_act_out_bram_base
);

    // ----------------------------------------------------------------------
    // Register storage. One 32-bit reg per writable AXI address so read-back
    // returns exactly what the master wrote.
    // ----------------------------------------------------------------------
    reg [31:0] reg_input_channels;
    reg [31:0] reg_output_channels;
    reg [31:0] reg_kernel_h_w;
    reg [31:0] reg_stride_h_w;
    reg [31:0] reg_padding_h_w;
    reg [31:0] reg_input_h_w;
    reg [31:0] reg_output_h_w;
    reg [31:0] reg_weight_base;
    reg [31:0] reg_bias_base;
    reg [31:0] reg_scale_mult;
    reg [31:0] reg_scale_shift_zp;
    reg [31:0] reg_act_in_base;
    reg [31:0] reg_act_out_base;

    // ----------------------------------------------------------------------
    // AXI4-Lite handshake registers.
    // Write: accept awvalid+wvalid together; issue bvalid one cycle later.
    // Read: accept arvalid; issue rvalid one cycle later.
    // ----------------------------------------------------------------------
    reg        bvalid_r;
    reg        rvalid_r;
    reg [31:0] rdata_r;

    wire awready_w = ~bvalid_r;
    wire wready_w  = ~bvalid_r;
    wire arready_w = ~rvalid_r;

    wire write_handshake = s_axil_awvalid & s_axil_wvalid & awready_w & wready_w;
    wire read_handshake  = s_axil_arvalid & arready_w;

    // engine_start trigger sources
    wire axi_start_write = write_handshake
                         & (s_axil_awaddr == 8'h2C)
                         & s_axil_wdata[0];
    wire start_trigger   = (axi_start_write | engine_start_ext) & ~engine_busy_in;

    reg engine_start_pulse_r;

    // ----------------------------------------------------------------------
    // Wire-out assignments
    // ----------------------------------------------------------------------
    assign s_axil_awready = awready_w;
    assign s_axil_wready  = wready_w;
    assign s_axil_bvalid  = bvalid_r;
    assign s_axil_bresp   = 2'b00;
    assign s_axil_arready = arready_w;
    assign s_axil_rvalid  = rvalid_r;
    assign s_axil_rdata   = rdata_r;
    assign s_axil_rresp   = 2'b00;

    assign engine_start_pulse = engine_start_pulse_r;
    assign engine_busy_ext    = engine_busy_in;
    assign engine_done_ext    = engine_done_in;

    // Decoded configuration outputs
    assign cfg_ic                = reg_input_channels[11:0];
    assign cfg_oc                = reg_output_channels[11:0];
    assign cfg_kh                = reg_kernel_h_w[6:4];
    assign cfg_kw                = reg_kernel_h_w[2:0];
    assign cfg_stride_h          = reg_stride_h_w[5:3];
    assign cfg_stride_w          = reg_stride_h_w[2:0];
    assign cfg_pad_h             = reg_padding_h_w[5:3];
    assign cfg_pad_w             = reg_padding_h_w[2:0];
    assign cfg_ih                = reg_input_h_w[23:16];
    assign cfg_iw                = reg_input_h_w[7:0];
    assign cfg_oh                = reg_output_h_w[23:16];
    assign cfg_ow                = reg_output_h_w[7:0];
    assign cfg_weight_uram_base  = reg_weight_base[21:0];
    assign cfg_bias_uram_base    = reg_bias_base[21:0];
    assign cfg_scale_mult        = reg_scale_mult;
    assign cfg_scale_shift       = reg_scale_shift_zp[5:0];
    assign cfg_act_in_bram_base  = reg_act_in_base[15:0];
    assign cfg_act_out_bram_base = reg_act_out_base[15:0];

    // ----------------------------------------------------------------------
    // Write path + engine_start_pulse generation.
    // ----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bvalid_r              <= 1'b0;
            engine_start_pulse_r  <= 1'b0;
            reg_input_channels    <= 32'd0;
            reg_output_channels   <= 32'd0;
            reg_kernel_h_w        <= 32'd0;
            reg_stride_h_w        <= 32'd0;
            reg_padding_h_w       <= 32'd0;
            reg_input_h_w         <= 32'd0;
            reg_output_h_w        <= 32'd0;
            reg_weight_base       <= 32'd0;
            reg_bias_base         <= 32'd0;
            reg_scale_mult        <= 32'd0;
            reg_scale_shift_zp    <= 32'd0;
            reg_act_in_base       <= 32'd0;
            reg_act_out_base      <= 32'd0;
        end else begin
            // engine_start_pulse defaults low; raised below if a trigger fires.
            engine_start_pulse_r <= 1'b0;

            // Single-cycle handshake: clear bvalid once master accepts it.
            if (bvalid_r & s_axil_bready) begin
                bvalid_r <= 1'b0;
            end

            // Write decode
            if (write_handshake) begin
                bvalid_r <= 1'b1;
                case (s_axil_awaddr)
                    8'h00:   reg_input_channels  <= s_axil_wdata;
                    8'h04:   reg_output_channels <= s_axil_wdata;
                    8'h08:   reg_kernel_h_w      <= s_axil_wdata;
                    8'h0C:   reg_stride_h_w      <= s_axil_wdata;
                    8'h10:   reg_padding_h_w     <= s_axil_wdata;
                    8'h14:   reg_input_h_w       <= s_axil_wdata;
                    8'h18:   reg_output_h_w      <= s_axil_wdata;
                    8'h1C:   reg_weight_base     <= s_axil_wdata;
                    8'h20:   reg_bias_base       <= s_axil_wdata;
                    8'h24:   reg_scale_mult      <= s_axil_wdata;
                    8'h28:   reg_scale_shift_zp  <= s_axil_wdata;
                    8'h2C:   ; // CONTROL.start handled via start_trigger below
                    // 0x30 STATUS is read-only
                    8'h34:   reg_act_in_base     <= s_axil_wdata;
                    8'h38:   reg_act_out_base    <= s_axil_wdata;
                    default: ; // unrecognised offset: write completes with OKAY but no register changes
                endcase
            end

            // engine_start_pulse latch: any trigger (AXI START write or external
            // pin) that arrives while engine_busy is low produces exactly one
            // cycle of engine_start_pulse on the FOLLOWING clock edge.
            if (start_trigger) begin
                engine_start_pulse_r <= 1'b1;
            end
        end
    end

    // ----------------------------------------------------------------------
    // Read path. Single-cycle latency: rvalid asserted one cycle after the
    // arvalid handshake; rdata is the full 32-bit stored register value.
    // ----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rvalid_r <= 1'b0;
            rdata_r  <= 32'd0;
        end else begin
            if (rvalid_r & s_axil_rready) begin
                rvalid_r <= 1'b0;
            end
            if (read_handshake) begin
                rvalid_r <= 1'b1;
                case (s_axil_araddr)
                    8'h00:   rdata_r <= reg_input_channels;
                    8'h04:   rdata_r <= reg_output_channels;
                    8'h08:   rdata_r <= reg_kernel_h_w;
                    8'h0C:   rdata_r <= reg_stride_h_w;
                    8'h10:   rdata_r <= reg_padding_h_w;
                    8'h14:   rdata_r <= reg_input_h_w;
                    8'h18:   rdata_r <= reg_output_h_w;
                    8'h1C:   rdata_r <= reg_weight_base;
                    8'h20:   rdata_r <= reg_bias_base;
                    8'h24:   rdata_r <= reg_scale_mult;
                    8'h28:   rdata_r <= reg_scale_shift_zp;
                    8'h2C:   rdata_r <= {30'd0, engine_busy_in, 1'b0};
                    8'h30:   rdata_r <= {30'd0, engine_busy_in, engine_done_in};
                    8'h34:   rdata_r <= reg_act_in_base;
                    8'h38:   rdata_r <= reg_act_out_base;
                    default: rdata_r <= 32'd0;
                endcase
            end
        end
    end

    // Tie-off of write-strobe input -- single 32-bit registers accept full-word
    // writes only. Strobes are consumed by the AXI spec but ignored by the
    // decode (any non-zero strobe issues a full-word write). Reference the
    // signal so the linter does not flag it as unused.
    wire _unused_wstrb = |s_axil_wstrb;

endmodule
