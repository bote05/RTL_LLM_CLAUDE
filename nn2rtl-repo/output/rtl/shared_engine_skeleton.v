`timescale 1ns/1ps

// shared_engine_skeleton.v
// --------------------------------------------------------------------------
// Top-level skeleton for the Phase 2 shared compute engine on Alveo U250.
// Authoritative description: docs/agent_tasks/00_engine_skeleton_spec.md
// Port spec:                 docs/agent_tasks/00_engine_skeleton_spec_PORTS.md
// FSM spec:                  docs/agent_tasks/00_engine_skeleton_spec_FSM.md
//
// Architecture commitments (deployment plan §6.1):
//   * 256 MACs, output-channel-parallel.
//   * 3-stage requantisation pipeline (bias-add -> scale-mult -> shift+saturate).
//   * On-chip URAM weight read port. NO AXI4-MM to DDR.
//
// This file is human-written and intentionally empty inside the SUBBLOCK
// stubs. Wave 2 tasks (07-11) replace each empty `module <name> ...
// endmodule` stub at the bottom of this file with a full implementation
// in `output/rtl/engine/<name>.v`. The integration build pulls the real
// sub-block files; the stubs at the bottom of this file exist solely so
// `iverilog -t null output/rtl/shared_engine_skeleton.v` compiles
// cleanly during the task-00 review gate.
//
// Largest layer the engine must support (from output/layer_ir.json):
//   node_conv_298 -- conv2d, [512, 512, 3, 3], IH=IW=7, stride=1, pad=1,
//   K_TOTAL = 512 * 3 * 3 = 4608, num_weights = 2,359,296 (~2.36 MB).
//   node_conv_288 (the structural seed) has the widest channel count:
//   IC=1024, OC=2048. Bus widths below are sized for whichever bound is
//   larger across the heavy module set.
// --------------------------------------------------------------------------

module shared_engine #(
    // ---- MAC array shape ----
    parameter integer MAC_COUNT        = 256,

    // ---- Layer-size bounds across heavy modules ----
    parameter integer MAX_IC           = 2048,
    parameter integer MAX_OC           = 2048,
    parameter integer MAX_KH           = 3,
    parameter integer MAX_KW           = 3,
    parameter integer MAX_IH           = 14,
    parameter integer MAX_IW           = 14,
    parameter integer MAX_OH           = 14,
    parameter integer MAX_OW           = 14,

    // ---- Arithmetic widths (drives sub-block widths; keep in sync with
    //      PORTS.md). These are flat numbers, NOT `localparam` expressions,
    //      so the sub-block ports are easy to mirror in their own modules. ----
    parameter integer ACT_W            = 8,                       // INT8
    parameter integer WGT_W            = 4,                       // INT4 (nibble-packed weights)
    parameter integer BIAS_W           = 32,                      // INT32 accumulator-domain bias
    parameter integer ACC_W            = 32,                      // 16 + clog2(K_TOTAL_MAX=4608) = 16+13 = 29, rounded to 32
    parameter integer SCALE_MULT_W     = 32,                      // Task 13a Bundle A audit fix: real ResNet-50 scale_mult values reach ~30 bits; old 16b silently truncated.
    parameter integer SCALE_SHIFT_W    = 6,                       // max SCALE_SHIFT in heavy set is < 32, fits in 6b

    // ---- External bus widths ----
    parameter integer ACT_BUS_W        = 2048,                    // = MAC_COUNT * ACT_W; one 256-channel BRAM beat
    parameter integer URAM_DATA_W      = 1024,                    // = MAC_COUNT * WGT_W (INT4); one 256-nibble weight beat
    parameter integer ACT_BRAM_ADDR_W  = 16,                      // word-addressed activation BRAM (64KB region per layer)
    parameter integer URAM_ADDR_W      = 22,                      // up to 4M-word URAM region (full 22.4 MB weight space)

    // ---- AXI4-Lite control slave ----
    parameter integer AXIL_ADDR_W      = 8,                       // 256 bytes of config registers
    parameter integer AXIL_DATA_W      = 32,

    // ---- Engine-output backpressure (default OFF = byte-identical legacy) ----
    // When 0 (DEFAULT) the new `out_ready` input is IGNORED: the effective
    // downstream-ready is forced to constant 1'b1, so the produce path (bridge
    // write half + FSM ST_REQUANT/ST_DRAIN advance) behaves EXACTLY as the
    // original design. ResNet's top and every engine-iso harness instantiate
    // shared_engine WITHOUT this parameter and WITHOUT connecting out_ready, so
    // they are byte-identical to today (the undriven out_ready value is gated
    // out, never observed). Only the MobileNetV2 engine top sets this to 1 and
    // drives out_ready from engine_output_fifo.in_ready, enabling the stall.
    parameter integer ENABLE_OUTPUT_BACKPRESSURE = 0,

    // ---- Depthwise per-lane mode (default OFF = byte-identical legacy) ----
    // [DW-ENGINE P1 2026-06-10] When 0 (DEFAULT) the cfg_depthwise register
    // (0x3C) is force-gated to 0, so the address_generator's K walk and the
    // mac_array's activation select are EXACTLY the original dense forms —
    // every ResNet instance (which never sets this parameter and whose
    // scheduler never writes 0x3C) is bit- and cycle-identical. Only the
    // MobileNetV2 engine top sets this to 1 so its scheduler can dispatch the
    // 3 wide depthwise convs (conv_896/902/908: C=960, 3x3, lanes==channels).
    parameter integer ENABLE_DEPTHWISE = 0,

    // ---- K-tap parallelism (default 1 = bit/cycle-identical legacy) ----
    // [KPAR4 2026-06-10] When 1 (DEFAULT) every K_PAR generate-if in this
    // file and in mac_array/address_generator elaborates the ORIGINAL
    // serial logic VERBATIM — ResNet (which never sets K_PAR) is provably
    // unchanged. When 4 (MBV2 engine top + the KPAR4 iso build):
    //   * URAM_DATA_W must be K_PAR*MAC_COUNT*WGT_W (4 tap-major words per
    //     repacked bank line, tap0 lowest);
    //   * weight_rd_addr exports the GROUP address (old word addr >> 2);
    //   * FAST-eligible dense 1x1 layers run 4 taps/cycle; depthwise and
    //     unaligned-base layers fall back to the serial walk through a
    //     2-cycle-piped subword select (byte-exact, legacy-rate).
    parameter integer K_PAR = 1
) (
    // ---- Clock + reset ----
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- AXI4-Lite control slave ----
    input  wire                          s_axil_awvalid,
    output wire                          s_axil_awready,
    input  wire [AXIL_ADDR_W-1:0]        s_axil_awaddr,
    input  wire                          s_axil_wvalid,
    output wire                          s_axil_wready,
    input  wire [AXIL_DATA_W-1:0]        s_axil_wdata,
    input  wire [(AXIL_DATA_W/8)-1:0]    s_axil_wstrb,
    output wire                          s_axil_bvalid,
    input  wire                          s_axil_bready,
    output wire [1:0]                    s_axil_bresp,
    input  wire                          s_axil_arvalid,
    output wire                          s_axil_arready,
    input  wire [AXIL_ADDR_W-1:0]        s_axil_araddr,
    output wire                          s_axil_rvalid,
    input  wire                          s_axil_rready,
    output wire [AXIL_DATA_W-1:0]        s_axil_rdata,
    output wire [1:0]                    s_axil_rresp,

    // ---- Engine handshake (scheduler-driven) ----
    input  wire                          engine_start,
    output wire                          engine_busy,
    output wire                          engine_done,

    // ---- BRAM activation INPUT port (engine reads activations) ----
    output wire [ACT_BRAM_ADDR_W-1:0]    act_in_rd_addr,
    output wire                          act_in_rd_en,
    input  wire [ACT_BUS_W-1:0]          act_in_rd_data,

    // ---- BRAM activation OUTPUT port (engine writes activations) ----
    output wire [ACT_BRAM_ADDR_W-1:0]    act_out_wr_addr,
    output wire                          act_out_wr_en,
    output wire [ACT_BUS_W-1:0]          act_out_wr_data,

    // ---- Engine-output backpressure (NEW) ----
    // Downstream (engine_output_fifo) can accept an act_out beat THIS cycle.
    // IGNORED unless ENABLE_OUTPUT_BACKPRESSURE != 0 (see parameter note); when
    // ignored the engine treats downstream as always-ready (constant 1'b1) so
    // an unconnected/undriven out_ready never affects behavior. Leaving it
    // unconnected on legacy instances (ResNet, iso harnesses) is byte-identical.
    input  wire                          out_ready,

    // ---- URAM weight read port ----
    output wire [URAM_ADDR_W-1:0]        weight_rd_addr,
    output wire                          weight_rd_en,
    input  wire [URAM_DATA_W-1:0]        weight_rd_data,

    // ---- On-chip bias read port (added by task 13a Bundle A / fix 5).
    //      One wide bias word per oc_pass = MAC_COUNT INT32 biases packed
    //      = 8192 bits for MAC_COUNT=256. The bias memory lives in the
    //      top wrapper; this is the engine's read interface to it. ----
    output wire [URAM_ADDR_W-1:0]        bias_rd_addr,
    output wire                          bias_rd_en,
    input  wire [MAC_COUNT*BIAS_W-1:0]   bias_rd_data,
    // PER-OUTPUT-CHANNEL requant scale ROM (Phase 2 INT4-GPTQ). Read at the
    // SAME address/enable as bias (scale_memory_map base_words == bias's), so
    // scale_rd_data arrives aligned with bias_rd_data -> requant_scale_in.
    output wire [URAM_ADDR_W-1:0]        scale_rd_addr,
    output wire                          scale_rd_en,
    input  wire [MAC_COUNT*32-1:0]       scale_rd_data
);

    // ====================================================================
    // Internal wires that cross sub-block boundaries.
    // Wave 2 tasks must drive / sample exactly these names.
    // ====================================================================

    // -- Config wires (config_register_block -> address_generator + others) --
    wire [11:0] cfg_ic;
    wire [11:0] cfg_oc;
    wire [2:0]  cfg_kh;
    wire [2:0]  cfg_kw;
    wire [7:0]  cfg_ih;
    wire [7:0]  cfg_iw;
    wire [7:0]  cfg_oh;
    wire [7:0]  cfg_ow;
    wire [2:0]  cfg_stride_h;
    wire [2:0]  cfg_stride_w;
    wire [2:0]  cfg_pad_h;
    wire [2:0]  cfg_pad_w;
    wire [SCALE_MULT_W-1:0]    cfg_scale_mult;
    wire [SCALE_SHIFT_W-1:0]   cfg_scale_shift;
    wire [URAM_ADDR_W-1:0]     cfg_weight_uram_base;
    wire [URAM_ADDR_W-1:0]     cfg_bias_uram_base;
    wire [ACT_BRAM_ADDR_W-1:0] cfg_act_in_bram_base;
    wire [ACT_BRAM_ADDR_W-1:0] cfg_act_out_bram_base;
    // [DW-ENGINE P1] depthwise flag from config reg 0x3C; dw_mode is the
    // PARAMETER-GATED version consumed by the datapath (hard 0 when the
    // feature is disabled, so legacy instances are bit-identical).
    wire cfg_depthwise;
    wire dw_mode = (ENABLE_DEPTHWISE != 0) ? cfg_depthwise : 1'b0;

    // -- FSM <-> config_register_block handshake --
    wire engine_start_pulse;        // config -> FSM (1-cycle pulse, sourced from engine_start input)
    wire fsm_engine_busy;           // FSM -> config (drives external engine_busy)
    wire fsm_engine_done;           // FSM -> config (drives external engine_done)

    // -- FSM -> address_generator control --
    wire        run_active;
    wire [2:0]  oc_pass_idx;        // 0..MAX_OC/MAC_COUNT-1
    wire [7:0]  pixel_h;            // current output pixel row
    wire [7:0]  pixel_w;            // current output pixel col

    // -- address_generator outputs --
    wire [URAM_ADDR_W-1:0]     ag_weight_rd_addr;
    wire                       ag_weight_rd_en;
    wire [URAM_ADDR_W-1:0]     ag_bias_rd_addr;
    wire                       ag_bias_rd_en;
    wire [ACT_BRAM_ADDR_W-1:0] ag_act_in_rd_addr;
    wire                       ag_act_in_rd_en;
    wire [7:0]                 ag_act_in_ic_byte_idx;  // which byte from act_in_rd_data
    wire [ACT_BRAM_ADDR_W-1:0] ag_act_out_wr_addr;
    wire [15:0]                ag_k_index;
    wire                       ag_mac_done;
    wire                       ag_pixel_done;
    wire [3:0]                 ag_k_tap_mask;   // [KPAR4] per-tap valid of the issued group

    // -- bram_to_stream_bridge wires (act_in side) --
    wire [ACT_W-1:0]           mac_act_byte;
    wire                       mac_act_byte_valid;
    // -- bram_to_stream_bridge wires (act_out side) --
    wire                       bridge_busy;

    // -- mac_array <-> requant_pipeline path --
    wire                              mac_clear;
    wire                              mac_valid_in;
    wire [K_PAR*MAC_COUNT*WGT_W-1:0]  mac_weight_bus;       // [KPAR4] K_PAR taps x (MAC_COUNT*WGT_W); 2048 b at K_PAR=1
    wire [MAC_COUNT*ACC_W-1:0]        mac_acc_out;          // 8192 b
    wire                              mac_busy;

    wire                              requant_valid_in;
    wire [MAC_COUNT*BIAS_W-1:0]       requant_bias_in;      // 8192 b
    wire [MAC_COUNT*ACT_W-1:0]        requant_data_out;     // 2048 b
    wire                              requant_valid_out;

    // ====================================================================
    // External port -> internal wire hookup
    // ====================================================================
    // The address_generator owns the URAM weight/bias address paths and
    // the BRAM read/write address paths. config_register_block owns the
    // AXI4-Lite slave. The FSM owns engine_busy/engine_done.

    // [KPAR4] K_PAR>1 exports the GROUP address (old word addr >> 2): the
    // MBV2 banks are repacked 4-taps-per-line (repack_mbv2_kpar4_banks.py).
    // K_PAR==1 is the verbatim legacy passthrough.
    generate if (K_PAR == 1) begin : g_waddr_legacy
        assign weight_rd_addr  = ag_weight_rd_addr;
    end else begin : g_waddr_kpar
        assign weight_rd_addr  = {2'b00, ag_weight_rd_addr[21:2]};
    end endgenerate
    assign weight_rd_en    = ag_weight_rd_en;
    assign act_in_rd_addr  = ag_act_in_rd_addr;
    assign act_in_rd_en    = ag_act_in_rd_en;
    assign act_out_wr_addr = ag_act_out_wr_addr;

    // engine_busy / engine_done are owned by config_register_block, which
    // synthesises them from FSM-internal status (see fsm_engine_busy/done).
    // Hooked through the config module so the AXI4-Lite status register
    // reads the same signal the external pin sees.

    // ====================================================================
    // FSM outline -- state encoding + transitions only. Per-state datapath
    // control (driving sub-block enables, weight/bias muxes, etc.) is left
    // to Wave 2 to fill in via additional always blocks that read these
    // state encodings.
    // ====================================================================

    localparam ST_IDLE         = 3'd0;
    localparam ST_LOAD_CONFIG  = 3'd1;
    localparam ST_RUN          = 3'd2;
    localparam ST_REQUANT      = 3'd3;
    localparam ST_DRAIN        = 3'd4;
    localparam ST_DONE         = 3'd5;

    reg [2:0] state;
    reg [2:0] next_state;

    // Placeholder next-state logic. Conditions are gated on placeholder
    // signals that Wave 2 will replace with sub-block-emitted handshakes.
    // Each state has exactly one outgoing arc defined here; the FSM doc
    // (00_engine_skeleton_spec_FSM.md) is the authoritative arc list.
    // [IV-HOIST] moved oc_pass_total* up (used in the FSM below) for iverilog use-before-decl.
    wire [3:0] oc_pass_total    = cfg_oc[11:8] + {3'b0, |cfg_oc[7:0]};
    wire [3:0] oc_pass_total_m1 = (oc_pass_total == 4'd0) ? 4'd0 : (oc_pass_total - 4'd1);

    // [ENGINE-OUTPUT BACKPRESSURE] Effective downstream-ready. When the feature
    // is disabled (DEFAULT, ENABLE_OUTPUT_BACKPRESSURE==0) this is a hard 1'b1,
    // so every downstream check below (the bridge write half + the FSM
    // ST_REQUANT advance) collapses to the original always-ready behavior and is
    // byte-identical. The out_ready input is only consulted when the feature is
    // enabled (MobileNetV2 engine top), so an undriven out_ready on legacy
    // instances is irrelevant.
    wire eff_out_ready = (ENABLE_OUTPUT_BACKPRESSURE != 0) ? out_ready : 1'b1;

    // Sticky flag: in ST_REQUANT this oc_pass's requant beat HAS been produced
    // (requant_valid_out pulsed) but the FSM is being held because the engine
    // output cannot drain (eff_out_ready low) — so the produce path must freeze
    // BEFORE starting the next oc_pass's MAC run. Only ever set when backpressure
    // is enabled AND eff_out_ready is low at the requant_valid_out cycle; with
    // the feature disabled it is permanently 0 (inert) and the FSM is unchanged.
    reg req_done_pending;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            req_done_pending <= 1'b0;
        end else if (state != ST_REQUANT) begin
            req_done_pending <= 1'b0;
        end else if (requant_valid_out && !eff_out_ready) begin
            req_done_pending <= 1'b1;   // beat done, stalled on downstream
        end else if (eff_out_ready) begin
            req_done_pending <= 1'b0;   // free to advance -> clear
        end
    end
    always @* begin
        next_state = state;
        case (state)
            ST_IDLE:        if (engine_start_pulse) next_state = ST_LOAD_CONFIG;
            ST_LOAD_CONFIG: next_state = ST_RUN;
            ST_RUN:         if (ag_mac_done)        next_state = ST_REQUANT;
            // Task 13a Bundle A audit fix: previously this hardcoded
            // `oc_pass_idx == (MAX_OC/MAC_COUNT-1) = 7`, but oc_pass_idx_r
            // wraps at the LAYER's actual oc_pass count (oc_pass_total_m1).
            // For any layer with cfg_oc < MAX_OC the FSM would deadlock
            // here. Use the layer-specific count instead.
            // [ENGINE-OUTPUT BACKPRESSURE] On the last oc_pass go to ST_DRAIN
            // (which already waits on !bridge_busy, so a held output write keeps
            // bridge_busy high and ST_DRAIN waits for it). On an intermediate
            // oc_pass advance to ST_RUN ONLY when the engine output can drain
            // (eff_out_ready) — otherwise HOLD in ST_REQUANT so the next pass's
            // MAC run does not start (and never produces a new requant beat that
            // would clobber the still-held output beat). req_done_pending keeps
            // the wait sticky after the 1-cycle requant_valid_out pulse. With
            // backpressure disabled eff_out_ready==1'b1 and req_done_pending==0,
            // so this collapses to the original arc (byte-identical):
            //   if (requant_valid_out) next = last ? ST_DRAIN : ST_RUN;
            ST_REQUANT:     if (requant_valid_out || req_done_pending) begin
                                if (oc_pass_idx == oc_pass_total_m1[2:0])
                                    next_state = ST_DRAIN;
                                else
                                    next_state = eff_out_ready ? ST_RUN : ST_REQUANT;
                            end
            ST_DRAIN:       if (!bridge_busy)       next_state = ag_pixel_done ? ST_DONE : ST_RUN;
            ST_DONE:        if (!engine_start)      next_state = ST_IDLE;
            default:        next_state = ST_IDLE;
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= ST_IDLE;
        else        state <= next_state;
    end

    assign run_active      = (state == ST_RUN);
    assign fsm_engine_busy = (state != ST_IDLE);
    assign fsm_engine_done = (state == ST_DONE);

    // ============================================================
    // Task 13a Bundle A (Fix 4): real datapath wiring replaces the
    // earlier `assign mac_valid_in = 1'b0;` and friends. The engine
    // FSM owns oc_pass_idx / pixel_h / pixel_w as its outer-loop
    // position; the address_generator walks the inner ic/kh/kw
    // counters and drives mac_done / pixel_done. Read pipelines
    // (weight, activation, bias) are aligned with the MAC array's
    // and requant pipeline's valid strobes.
    // ============================================================

    // ---- Outer-loop counters owned by the FSM ----
    reg [2:0] oc_pass_idx_r;
    reg [7:0] pixel_h_r;
    reg [7:0] pixel_w_r;
    assign oc_pass_idx = oc_pass_idx_r;
    assign pixel_h     = pixel_h_r;
    assign pixel_w     = pixel_w_r;

    // Number of oc_passes for this layer = ceil(cfg_oc / MAC_COUNT).
    // For MAC_COUNT=256: cfg_oc=512 -> 2 passes; cfg_oc=2048 -> 8 passes;
    // cfg_oc=384 -> 2 passes (NOT 1, since 384/256=1.5).
    //
    // 13a audit fix: previous formula `cfg_oc[11:8] - 1` gave the wrong
    // answer for non-multiple-of-256 OC. ResNet-50's heavy layers are
    // all powers of two (256, 512, 1024, 2048), so the FSM exited
    // REQUANT one pass too early for MobileNetV2-style channel counts
    // (96, 144, 192, 320, 576, 960). Now matches the ceil in
    // output/rtl/engine/address_generator.v line 199.
    // [IV-HOIST] oc_pass_total / oc_pass_total_m1 moved above the FSM (use-before-decl).
    wire [7:0] pixel_w_m1 = cfg_ow - 8'd1;
    wire [7:0] pixel_h_m1 = cfg_oh - 8'd1;

    // Pulse high for one cycle on each state transition to ST_RUN; the
    // mac_array uses it to zero its 256 accumulators before a new
    // (oc_pass, pixel) dot product begins.
    reg state_run_d;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) state_run_d <= 1'b0;
        else        state_run_d <= (state == ST_RUN);
    end
    wire run_entered = (state == ST_RUN) && !state_run_d;
    assign mac_clear = run_entered;

    // Outer counters tick on the FSM transitions documented in
    // 00_engine_skeleton_spec_FSM.md.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            oc_pass_idx_r <= 3'd0;
            pixel_h_r     <= 8'd0;
            pixel_w_r     <= 8'd0;
        end else begin
            // Reset both counters at the start of a layer.
            if (state == ST_LOAD_CONFIG) begin
                oc_pass_idx_r <= 3'd0;
                pixel_h_r     <= 8'd0;
                pixel_w_r     <= 8'd0;
            end
            // After each oc_pass's requant completes: either advance
            // oc_pass_idx (same pixel) or reset it (DRAIN then next pixel).
            if (state == ST_REQUANT && requant_valid_out) begin
                if (oc_pass_idx_r == oc_pass_total_m1[2:0]) begin
                    oc_pass_idx_r <= 3'd0;
                end else begin
                    oc_pass_idx_r <= oc_pass_idx_r + 3'd1;
                end
            end
            // When DRAIN completes one pixel, advance (pixel_w, pixel_h).
            // ag_pixel_done from the address_generator gates whether the
            // whole frame is done.
            if (state == ST_DRAIN && !bridge_busy && !ag_pixel_done) begin
                if (pixel_w_r == pixel_w_m1) begin
                    pixel_w_r <= 8'd0;
                    pixel_h_r <= (pixel_h_r == pixel_h_m1) ? 8'd0 : (pixel_h_r + 8'd1);
                end else begin
                    pixel_w_r <= pixel_w_r + 8'd1;
                end
            end
        end
    end

    // ---- URAM weight read -> mac_array alignment.
    // DEPLOYMENT URAM read latency = 2 cycles (xpm READ_LATENCY_A=2 in the top
    // wrapper's uram_weight_bank; see nn2rtl_top.v). The address_generator
    // asserts weight_rd_en in cycle N; weight_rd_data lands on cycle N+2. The
    // activation BRAM is 1-cycle (act_in_rd_data at N+1). So the MAC must fire at
    // N+2, with the activation HELD one extra cycle to realign with the (later)
    // weight, and the ic_byte index pipelined to match.
    //
    // [FIX 2026-05-28] Previously this assumed 1-cycle weight latency. That
    // matched the engine-sweep TB's 1-cycle BEHAVIORAL uram_weight_bank (so the
    // sweep was byte-exact = false confidence) but NOT the 2-cycle deployment
    // URAM. In-chain every MAC then multiplied the STALE (previous) weight,
    // producing the ~30% in-chain conv error. De-confounded via a clean
    // cycle-accurate engine-isolation harness (1-cyc byte-exact, 2-cyc reproduced
    // the exact in-chain error). This 2-cycle alignment is the fix.
    localparam integer WEIGHT_RD_LATENCY = 2;
    reg ag_weight_rd_en_d, ag_weight_rd_en_d2;
    reg ag_act_in_rd_en_d, ag_act_in_rd_en_d2;
    reg [7:0] ag_act_in_ic_byte_idx_d, ag_act_in_ic_byte_idx_d2;
    reg [ACT_BUS_W-1:0] act_in_rd_data_d;   // activation word held one extra cycle
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ag_weight_rd_en_d        <= 1'b0;
            ag_weight_rd_en_d2       <= 1'b0;
            ag_act_in_rd_en_d        <= 1'b0;
            ag_act_in_rd_en_d2       <= 1'b0;
            ag_act_in_ic_byte_idx_d  <= 8'd0;
            ag_act_in_ic_byte_idx_d2 <= 8'd0;
        end else begin
            ag_weight_rd_en_d        <= ag_weight_rd_en;
            ag_weight_rd_en_d2       <= ag_weight_rd_en_d;
            ag_act_in_rd_en_d        <= ag_act_in_rd_en;
            ag_act_in_rd_en_d2       <= ag_act_in_rd_en_d;
            ag_act_in_ic_byte_idx_d  <= ag_act_in_ic_byte_idx;
            ag_act_in_ic_byte_idx_d2 <= ag_act_in_ic_byte_idx_d;
        end
    end
    // [K1-FDCE] act_in_rd_data_d (2048b) is a DATAPATH hold register: it only
    // reaches the MAC when the (reset-held) ..._rd_en_d2 gates are high, and
    // it is rewritten every cycle -> reset value dead. No-reset => FDRE.
    always @(posedge clk) begin
        act_in_rd_data_d         <= act_in_rd_data;  // read-N act (valid N+1) -> held at N+2
    end

    // mac_valid_in asserts at N+2, when the 2-cycle weight has landed and the
    // held activation is realigned to the same read N.
    assign mac_valid_in = ag_weight_rd_en_d2 & ag_act_in_rd_en_d2;

    // [KPAR4] K_PAR==1 (DEFAULT): the ORIGINAL single-tap hookup, verbatim.
    // K_PAR==4: weight_rd_data carries 4 tap-major words per (group-
    // addressed) line. Tap0 is selected by the OLD address's [1:0], piped
    // 2 cycles exactly like ..._rd_en_d/_d2 to meet the URAM READ_LATENCY=2
    // data: FAST dense groups are 4-aligned so the subsel is 0 and tap0 ==
    // slice0; SERIAL dispatches (depthwise, FC base 13413%4==1) walk one
    // old word/cycle with mask 4'b0001, so tap0 tracks the subword and
    // taps 1..3 are masked dead inside mac_array.
    wire [3:0]  mac_tap_mask;
    wire [23:0] mac_act_bytes_ext;
    generate if (K_PAR == 1) begin : g_ktap_legacy
        // mac_weight_bus is the full URAM-wide weight read; at N+2 it carries the
        // weights requested at read N (one URAM word = MAC_COUNT INT8 weights = 2048b).
        assign mac_weight_bus = weight_rd_data[MAC_COUNT*WGT_W-1:0];
        assign mac_tap_mask      = 4'b0001;
        assign mac_act_bytes_ext = 24'd0;
        /* verilator lint_off UNUSED */
        wire _unused_kpar_skel = &{1'b0, ag_k_tap_mask};
        /* verilator lint_on UNUSED */
    end else begin : g_ktap_kpar
        // subword + mask pipes (2-cycle, mirroring ag_weight_rd_en_d/_d2).
        reg [1:0] wsub_d1, wsub_d2;
        reg [3:0] ktap_d1, ktap_d2;
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                wsub_d1 <= 2'd0;    wsub_d2 <= 2'd0;
                ktap_d1 <= 4'b0001; ktap_d2 <= 4'b0001;
            end else begin
                wsub_d1 <= ag_weight_rd_addr[1:0]; wsub_d2 <= wsub_d1;
                ktap_d1 <= ag_k_tap_mask;          ktap_d2 <= ktap_d1;
            end
        end
        // tap0 = subword-selected old word (slice 0 for aligned fast groups).
        assign mac_weight_bus[MAC_COUNT*WGT_W-1:0] =
            weight_rd_data[wsub_d2*(MAC_COUNT*WGT_W) +: MAC_COUNT*WGT_W];
        assign mac_weight_bus[1*(MAC_COUNT*WGT_W) +: MAC_COUNT*WGT_W] =
            weight_rd_data[1*(MAC_COUNT*WGT_W) +: MAC_COUNT*WGT_W];
        assign mac_weight_bus[2*(MAC_COUNT*WGT_W) +: MAC_COUNT*WGT_W] =
            weight_rd_data[2*(MAC_COUNT*WGT_W) +: MAC_COUNT*WGT_W];
        assign mac_weight_bus[3*(MAC_COUNT*WGT_W) +: MAC_COUNT*WGT_W] =
            weight_rd_data[3*(MAC_COUNT*WGT_W) +: MAC_COUNT*WGT_W];
        assign mac_tap_mask = ktap_d2;
        // dense taps 1..3 act bytes: consecutive ic bytes of the HELD act
        // word. Fast groups are 4-aligned (idx%4==0 and 256%4==0) so idx+3
        // never crosses the word; the +j adds use 8-bit WRAP intermediates
        // so a serial-mode idx=255 stays an in-range (masked-dead) select.
        wire [7:0] kidx1 = ag_act_in_ic_byte_idx_d2 + 8'd1;
        wire [7:0] kidx2 = ag_act_in_ic_byte_idx_d2 + 8'd2;
        wire [7:0] kidx3 = ag_act_in_ic_byte_idx_d2 + 8'd3;
        assign mac_act_bytes_ext[7:0]   = act_in_rd_data_d[kidx1*ACT_W +: ACT_W];
        assign mac_act_bytes_ext[15:8]  = act_in_rd_data_d[kidx2*ACT_W +: ACT_W];
        assign mac_act_bytes_ext[23:16] = act_in_rd_data_d[kidx3*ACT_W +: ACT_W];
    end endgenerate

    // Select the input-channel byte from the HELD activation word using the
    // twice-pipelined ic_byte index (both aligned to read N at cycle N+2).
    wire [ACT_W-1:0] mac_act_byte_sel =
        act_in_rd_data_d[ag_act_in_ic_byte_idx_d2*ACT_W +: ACT_W];
    assign mac_act_byte       = mac_act_byte_sel;
    assign mac_act_byte_valid = ag_act_in_rd_en_d2;

    // ---- Bias read -> requant_pipeline alignment.
    //
    // The address_generator pulses bias_rd_en once at the start of each
    // ST_RUN entry (one cycle after run_active rises). The bias memory in
    // the wrapper returns the wide bias word (256 INT32 = 8192 bits) one
    // cycle later and HOLDS it (synchronous BRAM read with rd_en gating);
    // bias_rd_data therefore stays stable through the entire MAC
    // accumulation up to and including the requant pipeline's bias-add
    // stage.
    //
    // requant_valid_in MUST NOT be driven by ag_bias_rd_en_d (that fires
    // at the START of the OC pass, before mac_array has accumulated
    // anything). Per the FSM spec (00_engine_skeleton_spec_FSM.md
    // §"ST_REQUANT"), the requant pipeline captures acc_out into stage-1
    // on ST_REQUANT entry. Trigger at ag_mac_done delayed by the
    // mac_array's accumulator latency:
    //   ag emits weight_rd_en at cycle K -> ag_weight_rd_en_d2 at K+2 (2-cyc
    //   weight latency) -> mac_array.mul_q1 latched at K+3 -> acc updated at K+4.
    // So acc is final at K+4; requant_valid_in fires at K+4. [FIX 2026-05-28:
    // was d3 for the old 1-cycle-weight assumption; +1 stage now that mac_valid_in
    // is one cycle later.]
    // [FIX 2026-05-31] +1 MORE stage (d4->d5): at the K+4 posedge the acc's FINAL
    // (last-term) product is added (acc<=acc+mul_q1) at the SAME edge requant would
    // capture acc, so a d4 capture races and reads the PRE-update acc => the final
    // (ic=255,kh=2,kw=2) product is DROPPED => systematically-LOW conv output (the
    // exact in-chain signature: conv_246 66% wrong, mostly negative). Capturing at
    // d5 (K+5) reads acc AFTER the K+4 last-accumulate. Root-caused via a real-memory
    // engine isolation harness (WLAT=2 wrong/low, WLAT=1 scrambled => 2-cyc alignment
    // is correct, bug is the acc-capture drain). See project_phase2_e2e_localization.
    reg ag_mac_done_d1, ag_mac_done_d2, ag_mac_done_d3, ag_mac_done_d4, ag_mac_done_d5;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ag_mac_done_d1   <= 1'b0;
            ag_mac_done_d2   <= 1'b0;
            ag_mac_done_d3   <= 1'b0;
            ag_mac_done_d4   <= 1'b0;
            ag_mac_done_d5   <= 1'b0;
        end else begin
            ag_mac_done_d1   <= ag_mac_done;
            ag_mac_done_d2   <= ag_mac_done_d1;
            ag_mac_done_d3   <= ag_mac_done_d2;
            ag_mac_done_d4   <= ag_mac_done_d3;
            ag_mac_done_d5   <= ag_mac_done_d4;
        end
    end
    assign requant_bias_in  = bias_rd_data;
    // Per-OC scale: read at bias's address/enable (base_words identical), so the
    // scale word for the current oc_pass arrives aligned with bias_rd_data.
    assign scale_rd_addr    = ag_bias_rd_addr;
    assign scale_rd_en      = ag_bias_rd_en;
    wire [MAC_COUNT*32-1:0] requant_scale_in = scale_rd_data;
    assign requant_valid_in = ag_mac_done_d5;

    // [DBG-BSC] dump per-OC bias + scale the requant actually uses (first requant).
    reg dbg_bsc_done = 0;
    always @(posedge clk) if (requant_valid_in && !dbg_bsc_done) begin
        $display("[BSC] bias0_3=%0d %0d %0d %0d  scale0_3=%h %h %h %h",
            $signed(bias_rd_data[0*32 +: 32]), $signed(bias_rd_data[1*32 +: 32]),
            $signed(bias_rd_data[2*32 +: 32]), $signed(bias_rd_data[3*32 +: 32]),
            scale_rd_data[0*32 +: 32], scale_rd_data[1*32 +: 32],
            scale_rd_data[2*32 +: 32], scale_rd_data[3*32 +: 32]);
        dbg_bsc_done <= 1'b1;
    end

    // [DBG-PULSE] count mac_valid_in pulses per output pixel (reset on mac_clear);
    // dump at requant capture + lane0 acc. conv_246 expects K_TOTAL=IC*KH*KW=2304.
    // Pulses<K_TOTAL => a term's product never accumulated. Revert after RCA.
    integer dbg_pulses = 0; integer dbg_pix = 0; integer dbg_t = 0;
    always @(posedge clk) begin
        if (mac_clear) dbg_pulses <= 0;
        else if (mac_valid_in) dbg_pulses <= dbg_pulses + 1;
        if (requant_valid_in && dbg_pix < 6) begin
            $display("[PULSE] pix=%0d pulses=%0d acc_lane0=%0d", dbg_pix, dbg_pulses, $signed(mac_acc_out[31:0]));
            if (dbg_pix == 0) begin
                $display("[ACC8] pix0 lanes0..7: %0d %0d %0d %0d %0d %0d %0d %0d",
                    $signed(mac_acc_out[ 0*32 +: 32]), $signed(mac_acc_out[ 1*32 +: 32]),
                    $signed(mac_acc_out[ 2*32 +: 32]), $signed(mac_acc_out[ 3*32 +: 32]),
                    $signed(mac_acc_out[ 4*32 +: 32]), $signed(mac_acc_out[ 5*32 +: 32]),
                    $signed(mac_acc_out[ 6*32 +: 32]), $signed(mac_acc_out[ 7*32 +: 32]));
            end
            dbg_pix <= dbg_pix + 1;
        end
        // per-term act + weights (oc0..oc2) for pixel0's first 24 terms
        if (mac_clear) dbg_t <= 0;
        else if (mac_valid_in) begin
            if (dbg_pix == 0 && dbg_t < 32)
                $display("[TERM] t=%0d act=%0d wbus=%08h", dbg_t,
                    $signed(mac_act_byte), mac_weight_bus[31:0]);
            dbg_t <= dbg_t + 1;
        end
    end

    // ---- Expose the bias read interface to the top wrapper
    //      (address generator -> external pins) ----
    assign bias_rd_addr = ag_bias_rd_addr;
    assign bias_rd_en   = ag_bias_rd_en;

    // ====================================================================
    // SUBBLOCK: config_register_block
    //   see docs/agent_tasks/10_engine_config_register_block.md
    //   port spec: 00_engine_skeleton_spec_PORTS.md `## SUBBLOCK: config_register_block`
    //
    // AXI4-Lite slave. Decodes config writes and exposes them as the
    // cfg_* wires consumed by address_generator. Owns external engine_busy
    // and engine_done pins; synthesises them from fsm_engine_busy/done.
    // ====================================================================
    config_register_block u_config_register_block (
        .clk                   (clk),
        .rst_n                 (rst_n),
        .s_axil_awvalid        (s_axil_awvalid),
        .s_axil_awready        (s_axil_awready),
        .s_axil_awaddr         (s_axil_awaddr),
        .s_axil_wvalid         (s_axil_wvalid),
        .s_axil_wready         (s_axil_wready),
        .s_axil_wdata          (s_axil_wdata),
        .s_axil_wstrb          (s_axil_wstrb),
        .s_axil_bvalid         (s_axil_bvalid),
        .s_axil_bready         (s_axil_bready),
        .s_axil_bresp          (s_axil_bresp),
        .s_axil_arvalid        (s_axil_arvalid),
        .s_axil_arready        (s_axil_arready),
        .s_axil_araddr         (s_axil_araddr),
        .s_axil_rvalid         (s_axil_rvalid),
        .s_axil_rready         (s_axil_rready),
        .s_axil_rdata          (s_axil_rdata),
        .s_axil_rresp          (s_axil_rresp),
        .engine_start_ext      (engine_start),
        .engine_busy_in        (fsm_engine_busy),
        .engine_done_in        (fsm_engine_done),
        .engine_busy_ext       (engine_busy),
        .engine_done_ext       (engine_done),
        .engine_start_pulse    (engine_start_pulse),
        .cfg_ic                (cfg_ic),
        .cfg_oc                (cfg_oc),
        .cfg_kh                (cfg_kh),
        .cfg_kw                (cfg_kw),
        .cfg_ih                (cfg_ih),
        .cfg_iw                (cfg_iw),
        .cfg_oh                (cfg_oh),
        .cfg_ow                (cfg_ow),
        .cfg_stride_h          (cfg_stride_h),
        .cfg_stride_w          (cfg_stride_w),
        .cfg_pad_h             (cfg_pad_h),
        .cfg_pad_w             (cfg_pad_w),
        .cfg_scale_mult        (cfg_scale_mult),
        .cfg_scale_shift       (cfg_scale_shift),
        .cfg_weight_uram_base  (cfg_weight_uram_base),
        .cfg_bias_uram_base    (cfg_bias_uram_base),
        .cfg_act_in_bram_base  (cfg_act_in_bram_base),
        .cfg_act_out_bram_base (cfg_act_out_bram_base),
        .cfg_depthwise         (cfg_depthwise)
    );

    // ====================================================================
    // SUBBLOCK: address_generator
    //   see docs/agent_tasks/09_engine_address_generator.md
    //   port spec: 00_engine_skeleton_spec_PORTS.md `## SUBBLOCK: address_generator`
    //
    // Walks the K_TOTAL dimension during ST_RUN. Emits URAM weight/bias
    // addresses and BRAM activation read/write addresses. Signals
    // ag_mac_done when the current OC pass has consumed all K_TOTAL
    // weights, and ag_pixel_done when all OH*OW output pixels have been
    // emitted.
    // ====================================================================
    address_generator #(.K_PAR(K_PAR)) u_address_generator (
        .clk                   (clk),
        .rst_n                 (rst_n),
        .run_active            (run_active),
        .cfg_ic                (cfg_ic),
        .cfg_oc                (cfg_oc),
        .cfg_kh                (cfg_kh),
        .cfg_kw                (cfg_kw),
        .cfg_ih                (cfg_ih),
        .cfg_iw                (cfg_iw),
        .cfg_oh                (cfg_oh),
        .cfg_ow                (cfg_ow),
        .cfg_stride_h          (cfg_stride_h),
        .cfg_stride_w          (cfg_stride_w),
        .cfg_pad_h             (cfg_pad_h),
        .cfg_pad_w             (cfg_pad_w),
        .cfg_weight_uram_base  (cfg_weight_uram_base),
        .cfg_bias_uram_base    (cfg_bias_uram_base),
        .cfg_act_in_bram_base  (cfg_act_in_bram_base),
        .cfg_act_out_bram_base (cfg_act_out_bram_base),
        .cfg_depthwise         (dw_mode),
        .oc_pass_idx           (oc_pass_idx),
        .pixel_h               (pixel_h),
        .pixel_w               (pixel_w),
        .weight_rd_addr        (ag_weight_rd_addr),
        .weight_rd_en          (ag_weight_rd_en),
        .bias_rd_addr          (ag_bias_rd_addr),
        .bias_rd_en            (ag_bias_rd_en),
        .act_in_rd_addr        (ag_act_in_rd_addr),
        .act_in_rd_en          (ag_act_in_rd_en),
        .act_in_ic_byte_idx    (ag_act_in_ic_byte_idx),
        .act_out_wr_addr       (ag_act_out_wr_addr),
        .k_index               (ag_k_index),
        .mac_done              (ag_mac_done),
        .pixel_done            (ag_pixel_done),
        .k_tap_mask            (ag_k_tap_mask)
    );

    // ====================================================================
    // SUBBLOCK: bram_to_stream_bridge
    //   see docs/agent_tasks/11_engine_bram_to_stream_bridge.md
    //   port spec: 00_engine_skeleton_spec_PORTS.md `## SUBBLOCK: bram_to_stream_bridge`
    //
    // Two halves:
    //   Read side: takes ACT_BUS_W-wide BRAM beats, presents one IC byte
    //              per cycle to mac_array (output-channel-parallel broadcast).
    //   Write side: accumulates requant_pipeline's MAC_COUNT INT8 outputs
    //              per OC pass and writes ACT_BUS_W-wide beats to the
    //              activation output BRAM.
    // ====================================================================
    // 13a audit fix: previously the bridge's `mac_act_byte` /
    // `mac_act_byte_valid` outputs were connected to the same wires
    // driven by the skeleton's continuous assigns at lines 358-359
    // (`assign mac_act_byte = mac_act_byte_sel;` etc) — a multi-driver
    // conflict that Vivado synth flags and Verilator resolves to X.
    // The skeleton's assigns use `ag_act_in_ic_byte_idx_d` (1-cycle
    // delayed to align with the BRAM read latency) which is the CORRECT
    // alignment; the bridge's `mac_act_byte <= act_in_rd_data[
    // act_in_ic_byte_idx*8 +: 8]` uses an un-delayed byte index and is
    // off-by-one against the BRAM word. Dangle the bridge's read-half
    // outputs so the skeleton is the sole driver.
    bram_to_stream_bridge u_bram_to_stream_bridge (
        .clk                   (clk),
        .rst_n                 (rst_n),
        .act_in_rd_data        (act_in_rd_data),
        .act_in_rd_data_valid  (ag_act_in_rd_en),
        .act_in_ic_byte_idx    (ag_act_in_ic_byte_idx),
        .mac_act_byte          (),                   // dangled — see fix note above
        .mac_act_byte_valid    (),                   // dangled — see fix note above
        .requant_data          (requant_data_out),
        .requant_valid         (requant_valid_out),
        // [ENGINE-OUTPUT BACKPRESSURE] eff_out_ready is constant 1'b1 unless the
        // feature is enabled; with it 1'b1 the bridge write half is byte-identical.
        .out_ready             (eff_out_ready),
        .act_out_wr_data       (act_out_wr_data),
        .act_out_wr_en         (act_out_wr_en),
        .bridge_busy           (bridge_busy)
    );

    // ====================================================================
    // SUBBLOCK: mac_array
    //   see docs/agent_tasks/07_engine_mac_array.md
    //   port spec: 00_engine_skeleton_spec_PORTS.md `## SUBBLOCK: mac_array`
    //
    // 256 parallel signed-INT8 multiply-accumulate lanes,
    // output-channel-parallel. All lanes share the broadcast act_byte
    // and select their lane-specific weight from weight_bus. Accumulators
    // cleared on mac_clear; the FSM drives mac_clear at the start of each
    // OC pass for the current output pixel.
    // ====================================================================
    // [INT3-MIXED] forward WGT_W so the lane weight slice width matches the
    // engine's bit-width (4=INT4 default, 3=INT3). mac_weight_bus is already
    // MAC_COUNT*WGT_W wide, so this stays consistent at either width.
    mac_array #(.WGT_W(WGT_W), .K_PAR(K_PAR)) u_mac_array (
        .clk           (clk),
        .rst_n         (rst_n),
        .mac_clear     (mac_clear),
        .mac_valid_in  (mac_valid_in & mac_act_byte_valid),
        .act_byte      (mac_act_byte),
        .weight_bus    (mac_weight_bus),
        // [KPAR4] taps 1..3 act bytes + per-tap mask (legacy-inert ties).
        .act_bytes_ext (mac_act_bytes_ext),
        .tap_mask      (mac_tap_mask),
        // [DW-ENGINE P1] per-lane act source for depthwise mode: the HELD
        // activation word (same N+2 alignment as the dense byte select above).
        // dw_mode is a hard 0 unless ENABLE_DEPTHWISE — legacy bit-identical.
        .dw_mode       (dw_mode),
        .act_word      (act_in_rd_data_d),
        .acc_out       (mac_acc_out),
        .mac_busy      (mac_busy)
    );

    // ====================================================================
    // SUBBLOCK: requant_pipeline
    //   see docs/agent_tasks/08_engine_requant_pipeline.md
    //   port spec: 00_engine_skeleton_spec_PORTS.md `## SUBBLOCK: requant_pipeline`
    //
    // 3-stage pipeline operating on all 256 lanes in parallel:
    //   stage 1: biased[lane] = acc[lane] + bias[lane]
    //   stage 2: scaled[lane] = biased[lane] * SCALE_MULT_CONST
    //   stage 3: out[lane]    = saturate_int8((scaled[lane] + sign_aware_round) >>> SCALE_SHIFT)
    // Sign-aware rounding bias MUST match the canonical pattern in
    // knowledge/patterns/protected/01_context.md.
    // ====================================================================
    requant_pipeline u_requant_pipeline (
        .clk          (clk),
        .rst_n        (rst_n),
        .valid_in     (requant_valid_in),
        .acc_in       (mac_acc_out),
        .bias_in      (requant_bias_in),
        .scale_in     (requant_scale_in),
        .valid_out    (requant_valid_out),
        .data_out     (requant_data_out)
    );

endmodule


// ==========================================================================
// Empty SUBBLOCK stubs.
//
// These stubs exist solely so the skeleton compiles standalone under
// `iverilog -t null output/rtl/shared_engine_skeleton.v` during the task-00
// review gate. Wave 2 (tasks 07-11) implements each sub-block in its own
// file under `output/rtl/engine/<name>.v`. The integration build pulls
// those real files; the stubs below are NOT pulled into integration.
//
// Task 04c hookup: the wrapper (output/rtl/nn2rtl_top.v) emits
// `\`define NN2RTL_ENGINE_SUBBLOCKS_PROVIDED` at the top of its file so
// the integration parse
//   `iverilog -t null nn2rtl_top.v ... shared_engine_skeleton.v engine/*.v`
// suppresses these stubs and uses the real implementations under
// output/rtl/engine/. The standalone task-00 parse leaves the macro
// undefined, so the stubs are visible and the skeleton parses cleanly.
//
// PORTS.md in `docs/agent_tasks/00_engine_skeleton_spec_PORTS.md` is the
// authoritative interface contract. If you find yourself editing the
// stub port lists below, also update PORTS.md or the
// `scripts/check_subblock_ports.py` review gate will reject the sub-block
// implementation later.
// ==========================================================================

`ifndef NN2RTL_ENGINE_SUBBLOCKS_PROVIDED

module mac_array #(
    parameter integer WGT_W = 4,
    parameter integer K_PAR = 1
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         mac_clear,
    input  wire         mac_valid_in,
    input  wire [7:0]   act_byte,
    input  wire [K_PAR*256*WGT_W-1:0] weight_bus,
    input  wire [23:0]  act_bytes_ext, // [KPAR4]
    input  wire [3:0]   tap_mask,      // [KPAR4]
    input  wire         dw_mode,      // [DW-ENGINE P1]
    input  wire [2047:0] act_word,    // [DW-ENGINE P1]
    output wire [8191:0] acc_out,
    output wire         mac_busy
);
    assign acc_out  = {8192{1'b0}};
    assign mac_busy = 1'b0;
endmodule


module requant_pipeline (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    input  wire [8191:0] acc_in,
    input  wire [8191:0] bias_in,
    // 13a audit fix: scale_mult widened 16 -> 32 bits to match the real
    // output/rtl/engine/requant_pipeline.v port (also a stub-vs-real
    // width mismatch was caught by an external reviewer). The skeleton's
    // own cfg_scale_mult internal wire is already 32 bits via
    // SCALE_MULT_W=32; only this stub fallback (used during the standalone
    // skeleton parse, when NN2RTL_ENGINE_SUBBLOCKS_PROVIDED is undefined)
    // had the legacy 16-bit width.
    input  wire [31:0]  scale_mult,
    input  wire [5:0]   scale_shift,
    output wire         valid_out,
    output wire [2047:0] data_out
);
    assign valid_out = 1'b0;
    assign data_out  = {2048{1'b0}};
endmodule


module address_generator #(
    parameter integer K_PAR = 1
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         run_active,
    input  wire [11:0]  cfg_ic,
    input  wire [11:0]  cfg_oc,
    input  wire [2:0]   cfg_kh,
    input  wire [2:0]   cfg_kw,
    input  wire [7:0]   cfg_ih,
    input  wire [7:0]   cfg_iw,
    input  wire [7:0]   cfg_oh,
    input  wire [7:0]   cfg_ow,
    input  wire [2:0]   cfg_stride_h,
    input  wire [2:0]   cfg_stride_w,
    input  wire [2:0]   cfg_pad_h,
    input  wire [2:0]   cfg_pad_w,
    input  wire [21:0]  cfg_weight_uram_base,
    input  wire [21:0]  cfg_bias_uram_base,
    input  wire [15:0]  cfg_act_in_bram_base,
    input  wire [15:0]  cfg_act_out_bram_base,
    input  wire         cfg_depthwise,   // [DW-ENGINE P1]
    input  wire [2:0]   oc_pass_idx,
    input  wire [7:0]   pixel_h,
    input  wire [7:0]   pixel_w,
    output wire [21:0]  weight_rd_addr,
    output wire         weight_rd_en,
    output wire [21:0]  bias_rd_addr,
    output wire         bias_rd_en,
    output wire [15:0]  act_in_rd_addr,
    output wire         act_in_rd_en,
    output wire [7:0]   act_in_ic_byte_idx,
    output wire [15:0]  act_out_wr_addr,
    output wire [15:0]  k_index,
    output wire         mac_done,
    output wire         pixel_done,
    output wire [3:0]   k_tap_mask     // [KPAR4]
);
    assign weight_rd_addr     = 22'd0;
    assign weight_rd_en       = 1'b0;
    assign bias_rd_addr       = 22'd0;
    assign bias_rd_en         = 1'b0;
    assign act_in_rd_addr     = 16'd0;
    assign act_in_rd_en       = 1'b0;
    assign act_in_ic_byte_idx = 8'd0;
    assign act_out_wr_addr    = 16'd0;
    assign k_index            = 16'd0;
    assign mac_done           = 1'b1;
    assign pixel_done         = 1'b1;
    assign k_tap_mask         = 4'b0001;   // [KPAR4]
endmodule


module config_register_block (
    input  wire         clk,
    input  wire         rst_n,
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
    input  wire         s_axil_arvalid,
    output wire         s_axil_arready,
    input  wire [7:0]   s_axil_araddr,
    output wire         s_axil_rvalid,
    input  wire         s_axil_rready,
    output wire [31:0]  s_axil_rdata,
    output wire [1:0]   s_axil_rresp,
    input  wire         engine_start_ext,
    input  wire         engine_busy_in,
    input  wire         engine_done_in,
    output wire         engine_busy_ext,
    output wire         engine_done_ext,
    output wire         engine_start_pulse,
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
    output wire [15:0]  cfg_act_out_bram_base,
    output wire         cfg_depthwise    // [DW-ENGINE P1]
);
    assign s_axil_awready        = 1'b0;
    assign s_axil_wready         = 1'b0;
    assign s_axil_bvalid         = 1'b0;
    assign s_axil_bresp          = 2'b00;
    assign s_axil_arready        = 1'b0;
    assign s_axil_rvalid         = 1'b0;
    assign s_axil_rdata          = 32'd0;
    assign s_axil_rresp          = 2'b00;
    assign engine_busy_ext       = engine_busy_in;
    assign engine_done_ext       = engine_done_in;
    assign engine_start_pulse    = engine_start_ext;
    assign cfg_ic                = 12'd0;
    assign cfg_oc                = 12'd0;
    assign cfg_kh                = 3'd0;
    assign cfg_kw                = 3'd0;
    assign cfg_ih                = 8'd0;
    assign cfg_iw                = 8'd0;
    assign cfg_oh                = 8'd0;
    assign cfg_ow                = 8'd0;
    assign cfg_stride_h          = 3'd0;
    assign cfg_stride_w          = 3'd0;
    assign cfg_pad_h             = 3'd0;
    assign cfg_pad_w             = 3'd0;
    assign cfg_scale_mult        = 32'd0;
    assign cfg_scale_shift       = 6'd0;
    assign cfg_weight_uram_base  = 22'd0;
    assign cfg_bias_uram_base    = 22'd0;
    assign cfg_act_in_bram_base  = 16'd0;
    assign cfg_act_out_bram_base = 16'd0;
    assign cfg_depthwise         = 1'b0;   // [DW-ENGINE P1]
endmodule


module bram_to_stream_bridge (
    input  wire         clk,
    input  wire         rst_n,
    input  wire [2047:0] act_in_rd_data,
    input  wire         act_in_rd_data_valid,
    input  wire [7:0]   act_in_ic_byte_idx,
    output wire [7:0]   mac_act_byte,
    output wire         mac_act_byte_valid,
    input  wire [2047:0] requant_data,
    input  wire         requant_valid,
    output wire [2047:0] act_out_wr_data,
    output wire         act_out_wr_en,
    output wire         bridge_busy
);
    assign mac_act_byte       = 8'd0;
    assign mac_act_byte_valid = 1'b0;
    assign act_out_wr_data    = {2048{1'b0}};
    assign act_out_wr_en      = 1'b0;
    assign bridge_busy        = 1'b0;
endmodule

`endif // NN2RTL_ENGINE_SUBBLOCKS_PROVIDED
