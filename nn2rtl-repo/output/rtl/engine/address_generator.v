`timescale 1ns/1ps

// address_generator.v
// --------------------------------------------------------------------------
// Wave 2 task 09 sub-block. Port list locked by
//   docs/agent_tasks/00_engine_skeleton_spec_PORTS.md `## SUBBLOCK: address_generator`.
// Spec:    docs/agent_tasks/09_engine_address_generator.md
// Seed:    output/rtl/node_conv_288.v (ar_pass_target / k_counter loop).
//
// Role
// ----
// During engine ST_RUN the address generator walks the K_TOTAL = IC*KH*KW
// dimension for a single (oc_pass, output_pixel) tuple. Every cycle it emits:
//   * weight_rd_addr / weight_rd_en   — one URAM engine-word (256 INT8) per cycle
//   * act_in_rd_addr / act_in_rd_en   — one BRAM beat; rd_en drops on padded
//                                       receptive-field positions so the
//                                       bridge substitutes 0 for the lane
//   * act_in_ic_byte_idx              — which signed-INT8 byte of the beat
//                                       the bridge will broadcast on
//                                       mac_act_byte
//   * k_index                         — running 0..K_TOTAL-1 counter
//   * act_out_wr_addr                 — pre-computed destination beat in
//                                       the output BRAM (used by the bridge
//                                       in ST_DRAIN)
// Once per OC pass (one cycle after run_active rises) it pulses bias_rd_en
// with bias_rd_addr = cfg_bias_uram_base + oc_pass_idx, because one wide
// bias word packs all 256 lane biases for the current OC pass
// (see task 09 spec §"Address granularity").
//
// FSM scope contract
// -------------------
// (oc_pass_idx, pixel_h, pixel_w) are external inputs driven by the engine
// FSM. The address generator does NOT own those counters; it only walks
// the inner three (kh, kw, ic) loops. mac_done pulses when k_index has
// reached K_TOTAL-1 (one output pixel finished for the current OC pass).
// pixel_done LATCHES high when mac_done fires for the final (pixel_h,
// pixel_w, oc_pass_idx) of the layer, and stays high until the next
// layer's first run_active rising edge (oc_pass=0, pixel=0,0). The FSM
// samples pixel_done in ST_DRAIN, after REQUANT.
//
// Convolution walk (matches task 09 spec pseudocode, bit-exact with the
// Python golden in address_generator_tb.cpp):
//   for kh in 0..KH-1:
//     for kw in 0..KW-1:
//       for ic in 0..IC-1:
//         in_r = pixel_h * SH + kh - PH
//         in_c = pixel_w * SW + kw - PW
//         if in_r in [0,IH) and in_c in [0,IW):
//             act_addr   = act_in_base + (in_r*IW + in_c)
//             act_in_en  = 1
//         else:
//             act_in_en  = 0       // bridge substitutes 0
//         weight_addr    = weight_base + oc_pass*IC*KH*KW
//                                      + ic*KH*KW + kh*KW + kw
//
// Counter advance order is ic-innermost, then kw, then kh, matching the
// pseudocode. k_index increments by 1 every active cycle and wraps from
// K_TOTAL-1 back to 0 simultaneously with the mac_done pulse.
//
// No arrays / no `reg [..] mem [..:..]` — all state is scalar regs, so
// the universal "Array memory write in async-reset always block" rule
// (knowledge/patterns/protected/08_common_bugs.md §"Array memory write")
// is N/A here.
// --------------------------------------------------------------------------

module address_generator (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        run_active,

    // Per-layer configuration (driven by config_register_block; stable from
    // ST_LOAD_CONFIG through engine_done).
    input  wire [11:0] cfg_ic,
    input  wire [11:0] cfg_oc,
    input  wire [2:0]  cfg_kh,
    input  wire [2:0]  cfg_kw,
    input  wire [7:0]  cfg_ih,
    input  wire [7:0]  cfg_iw,
    input  wire [7:0]  cfg_oh,
    input  wire [7:0]  cfg_ow,
    input  wire [2:0]  cfg_stride_h,
    input  wire [2:0]  cfg_stride_w,
    input  wire [2:0]  cfg_pad_h,
    input  wire [2:0]  cfg_pad_w,
    input  wire [21:0] cfg_weight_uram_base,
    input  wire [21:0] cfg_bias_uram_base,
    input  wire [15:0] cfg_act_in_bram_base,
    input  wire [15:0] cfg_act_out_bram_base,

    // Outer-loop position driven by the engine FSM.
    input  wire [2:0]  oc_pass_idx,
    input  wire [7:0]  pixel_h,
    input  wire [7:0]  pixel_w,

    // URAM weight read port.
    output reg  [21:0] weight_rd_addr,
    output reg         weight_rd_en,

    // URAM (wide-word) bias read port; one read per OC pass.
    output reg  [21:0] bias_rd_addr,
    output reg         bias_rd_en,

    // BRAM activation read port.
    output reg  [15:0] act_in_rd_addr,
    output reg         act_in_rd_en,
    output reg  [7:0]  act_in_ic_byte_idx,

    // BRAM activation write port (drained by the bridge in ST_DRAIN).
    output reg  [15:0] act_out_wr_addr,

    // K-loop index and completion strobes.
    output reg  [15:0] k_index,
    output reg         mac_done,
    output reg         pixel_done
);

    // ----------------------------------------------------------------------
    // Internal counters. Widths picked to cover the engine's MAX_IC/MAX_OC
    // (2048) and MAX_KH/KW (3) commitments from the deployment plan §6.1.
    // ----------------------------------------------------------------------
    reg [11:0] ic_cnt;       // 0..IC-1                  (innermost)
    reg [2:0]  kw_cnt;       // 0..KW-1
    reg [2:0]  kh_cnt;       // 0..KH-1                  (outermost of the three)
    reg [15:0] k_cnt;        // 0..K_TOTAL-1; mirrors k_index

    // Rising-edge detector for run_active.
    reg        run_active_d;

    // Latched layer-completion flag (pixel_done semantics — see header).
    reg        pixel_done_latch;

    // ----------------------------------------------------------------------
    // Wide arithmetic helpers. K_TOTAL for the heavy engine layers fits in
    // 15 bits (IC*KH*KW <= 2048*3*3 = 18432); the per-pass weight offset
    // (oc_pass * IC*KH*KW) fits in 18 bits with oc_pass <= 7. Sums with
    // cfg_weight_uram_base (22b) still fit in 22 bits because the
    // weight_memory_map.json layout reserves enough room.
    //
    // The Verilator -Wall pass flags any extra unused MSB so every
    // intermediate is declared at the exact width needed.
    // ----------------------------------------------------------------------
    // KH*KW <= 7*7 = 49, needs 6 bits. K_TOTAL = IC*KH*KW <= 2048*49 ≈ 100k,
    // needs 17 bits. Per-pass weight offset = oc_pass * K_TOTAL <= 7 * 100k
    // ≈ 700k, needs 20 bits.
    wire [5:0]  kh_kw_prod   = cfg_kh * cfg_kw;                     // <= 49
    wire [16:0] k_total      = cfg_ic * {11'b0, kh_kw_prod};        // <= 100352
    wire [15:0] k_total_m1   = k_total[15:0] - 16'd1;
    wire [19:0] pass_offset  = {3'b0, k_total} * oc_pass_idx;       // <= 7 * 100352
    wire [16:0] ic_weight    = ic_cnt * {6'b0, kh_kw_prod};         // ic * (KH*KW) <= 18432; 17b is plenty
    wire [8:0]  kh_offset    = kh_cnt * cfg_kw;                     // <= 6*7 = 42 → fits in 6 bits, give 9b for headroom
    wire [21:0] weight_offset_22 =
        {2'b0, pass_offset} + {5'b0, ic_weight} + {13'b0, kh_offset} + {19'b0, kw_cnt};
    wire [21:0] weight_addr_next = cfg_weight_uram_base + weight_offset_22;
    // 13a audit fix: use the portable `UNUSED` pragma (works on Verilator
    // 4.x and 5.x); the previous `UNUSEDSIGNAL` form was a Verilator-5.x-only
    // refinement and tripped older toolchains during the third-party audit.
    /* verilator lint_off UNUSED */
    wire        _unused_k_total_msb = k_total[16];
    /* verilator lint_on UNUSED */

    // ----------------------------------------------------------------------
    // Receptive-field bounds. Use 12-bit signed to cover the worst case
    //   pixel_h(255) * stride(7) + kh(6) - pad(7)   = ~1792
    // (well over the worst real layer of 224, but the extra slack costs us
    // nothing and keeps the comparator simple).
    // ----------------------------------------------------------------------
    wire signed [11:0] base_r   = $signed({4'b0, pixel_h}) * $signed({9'b0, cfg_stride_h});
    wire signed [11:0] base_c   = $signed({4'b0, pixel_w}) * $signed({9'b0, cfg_stride_w});
    wire signed [11:0] in_r     = base_r + $signed({9'b0, kh_cnt}) - $signed({9'b0, cfg_pad_h});
    wire signed [11:0] in_c     = base_c + $signed({9'b0, kw_cnt}) - $signed({9'b0, cfg_pad_w});
    wire        in_r_in_bounds  = (in_r >= 12'sd0) && (in_r < $signed({4'b0, cfg_ih}));
    wire        in_c_in_bounds  = (in_c >= 12'sd0) && (in_c < $signed({4'b0, cfg_iw}));
    wire        in_bounds       = in_r_in_bounds & in_c_in_bounds;

    // Activation BRAM address = act_in_base + (in_r * IW + in_c) * IC_CHUNKS
    //                         + ic_chunk_idx.
    //
    // For IC <= 256 (MAC_COUNT) the wide BRAM word (ACT_BUS_W=2048 bits =
    // 256 bytes) holds ALL channels of one (in_r, in_c) pixel, so
    // ic_chunks = 1 and the stride collapses to the legacy formula
    // base + (in_r*IW + in_c).
    //
    // For IC > 256 the upstream spatial layer (or this engine's own
    // act_out write path, which already strides by oc_passes) wrote
    // ceil(IC/256) BRAM words per output pixel — channels [0..255] at
    // chunk 0, [256..511] at chunk 1, etc. Match that stride here so
    // ic_cnt=256 lands at chunk 1 instead of re-reading chunk 0's
    // channels 0..255.
    //
    // ic_cnt[11:8] is exactly floor(ic_cnt / 256) for ic_cnt in [0, 4096),
    // which covers MAX_IC=2048 with one bit of headroom.
    wire [3:0]  ic_chunks_total = cfg_ic[11:8] + {3'b0, |cfg_ic[7:0]};
    wire [3:0]  ic_chunk_idx    = ic_cnt[11:8];
    wire [15:0] pixel_word_idx  = in_r[7:0] * cfg_iw + {8'b0, in_c[7:0]};
    wire [19:0] act_in_offset20 = pixel_word_idx * {12'b0, ic_chunks_total}
                                 + {16'b0, ic_chunk_idx};
    wire [15:0] act_in_addr_n   = cfg_act_in_bram_base + act_in_offset20[15:0];

    // Output BRAM address — one beat per (output_pixel, oc_pass). Pre-computed
    // so the bridge can latch it on entering ST_DRAIN.
    wire [15:0] pixel_index     = pixel_h * cfg_ow + {8'b0, pixel_w};
    wire [3:0]  oc_passes_total = cfg_oc[11:8] + {3'b0, |cfg_oc[7:0]};  // ceil(OC/256), valid for OC<=2048
    wire [15:0] act_out_addr_n  =
        cfg_act_out_bram_base + pixel_index * {12'b0, oc_passes_total} + {13'b0, oc_pass_idx};

    // Layer-completion gate. last_oc_pass is true when (oc_pass_idx+1)*256
    // covers the full OC; equivalently oc_pass_idx == oc_passes_total - 1.
    wire        last_oc_pass    = ({1'b0, oc_pass_idx} == (oc_passes_total - 4'd1));
    wire        last_pixel      = (pixel_h == (cfg_oh - 8'd1)) &&
                                  (pixel_w == (cfg_ow - 8'd1));
    wire        k_at_last       = (k_cnt == k_total_m1);

    // ----------------------------------------------------------------------
    // Sequential body. Single always block keeps the reset state and the
    // per-cycle walk in one place.
    // ----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ic_cnt              <= 12'd0;
            kw_cnt              <= 3'd0;
            kh_cnt              <= 3'd0;
            k_cnt               <= 16'd0;
            run_active_d        <= 1'b0;
            pixel_done_latch    <= 1'b0;

            weight_rd_addr      <= 22'd0;
            weight_rd_en        <= 1'b0;
            bias_rd_addr        <= 22'd0;
            bias_rd_en          <= 1'b0;
            act_in_rd_addr      <= 16'd0;
            act_in_rd_en        <= 1'b0;
            act_in_ic_byte_idx  <= 8'd0;
            act_out_wr_addr     <= 16'd0;
            k_index             <= 16'd0;
            mac_done            <= 1'b0;
            pixel_done          <= 1'b0;
        end else begin
            // ---- one-cycle defaults ----
            mac_done    <= 1'b0;
            bias_rd_en  <= 1'b0;

            run_active_d <= run_active;

            // ---- rising-edge actions: reset inner counters, issue bias read ----
            if (run_active && !run_active_d) begin
                ic_cnt           <= 12'd0;
                kw_cnt           <= 3'd0;
                kh_cnt           <= 3'd0;
                k_cnt            <= 16'd0;
                bias_rd_addr     <= cfg_bias_uram_base + {19'd0, oc_pass_idx};
                bias_rd_en       <= 1'b1;
                // First k=0 of the new layer: clear the layer-done latch.
                if (oc_pass_idx == 3'd0 && pixel_h == 8'd0 && pixel_w == 8'd0) begin
                    pixel_done_latch <= 1'b0;
                end
            end

            // ---- per-cycle walk ----
            if (run_active) begin
                // Emit URAM weight read every active cycle.
                weight_rd_addr      <= weight_addr_next;
                // 2026-05-24 fix: gating must allow the LAST legitimate
                // weight read while still suppressing the stray read that
                // would otherwise leak into the cycle after k_at_last.
                //
                // The stale `~k_at_last` gating dropped the read at cycle
                // T(k_at_last)+1, but T(k_at_last)+1 is exactly where the
                // address generator's address line carries the FINAL legit
                // weight (the BASE+K_TOTAL-1 address was computed during
                // T(k_at_last) inside the same `if (run_active)` branch
                // and registered into weight_rd_addr). Suppressing it
                // dropped the very last MAC of every output pixel, which
                // shows up as 2 off-by-one mismatches on node_conv_246
                // (pixel[1,4] ch124 and ch238) where the accumulator
                // landed exactly on the requant rounding boundary.
                //
                // Gating on `~mac_done` (the REGISTERED output that fires
                // exactly ONE cycle after k_at_last) preserves the last
                // legit read AND suppresses the stray cleanly:
                //   T(k_at_last):    mac_done=0 -> weight_rd_en<=1
                //                    (last legit read fires next cycle)
                //   T(k_at_last)+1:  mac_done=1 -> weight_rd_en<=0
                //                    (stray suppressed)
                //   T(k_at_last)+2:  state == ST_REQUANT, run_active=0;
                //                    else branch enforces weight_rd_en=0.
                weight_rd_en        <= ~mac_done;

                // Activation read — gated by receptive-field bounds AND
                // by ~mac_done for the same reason as weight_rd_en above.
                act_in_rd_addr      <= act_in_addr_n;
                act_in_rd_en        <= in_bounds & ~mac_done;
                act_in_ic_byte_idx  <= ic_cnt[7:0];

                // Output BRAM write address (constant within an OC pass).
                act_out_wr_addr     <= act_out_addr_n;

                // k_index mirrors the running k_cnt counter.
                k_index             <= k_cnt;

                // ---- advance (ic, kw, kh) — ic innermost ----
                //
                // 2026-05-24 fix (conv_290 cluster): gate the advance block
                // on `~mac_done`. The cycle AFTER k_at_last fires keeps
                // `run_active=1` because the FSM transitions ST_RUN ->
                // ST_REQUANT one cycle later. Without this gate the walk's
                // else branch runs in that cycle and bumps ic_cnt from 0
                // (just reset by k_at_last) to 1. The leftover ic_cnt=1
                // persists through ST_REQUANT/ST_DRAIN and into the next
                // OC pass's ST_RUN rising edge, where the walk overrides
                // the rising-edge reset (later non-blocking assigns win) —
                // making every pass after the first one skip ic=0. The
                // fix preserves the LAST legitimate MAC (which fires from
                // weight_rd_addr / act_in_rd_addr LATCHED in cycle T(k_at_last))
                // because the address/enable updates above remain ungated.
                if (!mac_done) begin
                    if (k_at_last) begin
                        // End of inner loop for this (oc_pass, pixel). Pulse
                        // mac_done; reset counters in case the FSM keeps
                        // run_active high (it does not in the locked FSM, but
                        // resetting here makes the module robust to either).
                        mac_done <= 1'b1;
                        ic_cnt   <= 12'd0;
                        kw_cnt   <= 3'd0;
                        kh_cnt   <= 3'd0;
                        k_cnt    <= 16'd0;
                        if (last_pixel && last_oc_pass) begin
                            pixel_done_latch <= 1'b1;
                        end
                    end else if (ic_cnt == (cfg_ic - 12'd1)) begin
                        ic_cnt <= 12'd0;
                        if (kw_cnt == (cfg_kw - 3'd1)) begin
                            kw_cnt <= 3'd0;
                            kh_cnt <= kh_cnt + 3'd1;
                        end else begin
                            kw_cnt <= kw_cnt + 3'd1;
                        end
                        k_cnt <= k_cnt + 16'd1;
                    end else begin
                        ic_cnt <= ic_cnt + 12'd1;
                        k_cnt  <= k_cnt + 16'd1;
                    end
                end
            end else begin
                // run_active deasserted — drop the per-cycle read enables.
                // weight_rd_addr, act_in_rd_addr, act_in_ic_byte_idx, k_index
                // and act_out_wr_addr keep their last value; the engine FSM
                // is in ST_REQUANT / ST_DRAIN and does not need them.
                weight_rd_en       <= 1'b0;
                act_in_rd_en       <= 1'b0;
            end

            // pixel_done mirrors the latch so the FSM sees it stable from
            // the cycle mac_done fires for the final pixel through to the
            // next layer dispatch.
            pixel_done <= pixel_done_latch;
        end
    end

endmodule
