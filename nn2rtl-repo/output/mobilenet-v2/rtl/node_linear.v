// node_linear — gemm (fully-connected) classifier. [BEATSPLIT 2026-06-03] CHANNEL-TILED INPUT.
// data_in = 2048b (256 features) x N_TILES=5 beats -> fills in_buf[0:1279] (feature k = beat k/256,
// lane k%256). bias, scale (MULT=4071,SHIFT=20), round-half, clamp, and the 8000b single-beat
// OUTPUT are UNCHANGED -> byte-identical logits.
//
// [SYNTH-FIT 2026-06-06] The K=1280 dot product was UNROLLED COMBINATIONALLY in one cycle
// (`for k=0..1279: acc += in_buf[k]*weights[m*K+k]`) -> 1280 parallel mults + 1280-input adder
// tree + 1280 parallel weight/in_buf reads. That cone is the MobileNet-specific synth-RAM blowup
// (ResNet's classifier streams K serially). REWRITTEN to a SERIAL MAC: one product/cycle into a
// persistent acc_reg over K cycles, then a finalize cycle (bias/scale/round/clamp -> out_buf[m]).
// Integer sum is associative -> the accumulated sum is BIT-IDENTICAL to the unrolled sum; the
// bias/scale/round/clamp math is copied verbatim -> logits are byte-identical. Verilator ignores
// the cycle count; the only change is internal latency (M*(K+1) ~= 1.28M cyc, once/frame).
// Verified byte-exact vs the prior version via verify_node_linear/tb_equiv.sv (EQUIV_RESULT PASS).
//
// [SYNTH-FIT BANKED 2026-06-07] The single 1.28M-deep weight ROM (`reg [7:0] weights[0:M*K-1]`)
// would NOT infer block RAM even with a registered address + registered read + rom_style=block:
// Vivado mapped it to ~366K LUT (the `node_linear|weights|2097152x8|LUT` decision) because a single
// 1,280,000-deep array is too DEEP for clean single-array auto-inference. FIX: split the flat ROM
// into N_BANKS=5 explicit banks of BANK_DEPTH=2^18=262144 entries each (bank = w_addr[20:18],
// in-bank = w_addr[17:0]). Each bank is a separate (* ram_style="block" *) reg array filled by
// $readmemh of a CONTIGUOUS line-slice of node_linear_weights.hex -> bank b holds flat indices
// [b*262144, (b+1)*262144) -> byte-IDENTICAL to weights[idx]=bank{idx>>18}[idx&0x3FFFF]. Each
// 262144x8 bank infers ~58 RAMB36. The read OUTPUT register w_q is now a COMBINATIONAL mux of the
// 5 per-bank REGISTERED reads selected by the REGISTERED bank index -> w_q is still exactly ONE
// cycle behind w_addr (same pipeline depth, NO added latency). Values/order unchanged -> the
// accumulated dot product, bias/scale/round/clamp, and 8000b output are byte-identical.
module node_linear #(
    parameter ENABLE_BACKPRESSURE = 0
)(
    input  wire                clk,
    input  wire                rst_n,
    input  wire                valid_in,
    output reg                 ready_in,
    input  wire [2047:0]       data_in,
    input  wire                out_ready_in,
    output wire                valid_out,
    output wire [7999:0]       data_out
);

    // ---- datapath output regs + 1-deep output skid (output unchanged: single 8000b beat) ----
    reg                 dp_valid_out;
    reg  [7999:0]       dp_data_out;
    reg                 out_full;
    reg  [7999:0]       out_data;
    wire skid_block = (ENABLE_BACKPRESSURE != 0) && out_full && !out_ready_in;

    generate
    if (ENABLE_BACKPRESSURE == 0) begin : g_out_legacy
        assign valid_out = dp_valid_out;
        assign data_out  = dp_data_out;
    end else begin : g_out_bp
        assign valid_out = out_full;
        assign data_out  = out_data;
    end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_full <= 1'b0;
            out_data <= 8000'd0;
        end else begin
            if (out_full && out_ready_in)
                out_full <= 1'b0;
            if (dp_valid_out) begin
                out_data <= dp_data_out;
                out_full <= 1'b1;
            end
        end
    end

    localparam integer K             = 1280;
    localparam integer M             = 1000;
    localparam integer KLOG2         = 11;
    localparam integer PROD_W        = 16;
    localparam integer ACC_W         = PROD_W + KLOG2;
    localparam integer BIAS_W        = 32;
    localparam integer BIASED_W      = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1;
    localparam integer SCALE_MULT    = 4071;
    localparam integer SCALE_SHIFT   = 20;
    localparam integer SCALE_MAG_W   = 15;
    localparam integer SCALE_CONST_W = SCALE_MAG_W + 1;
    localparam integer SCALED_W      = BIASED_W + SCALE_CONST_W;

    localparam integer N_TILES       = 5;    // [BEATSPLIT] 5 beats x 256 features = 1280
    localparam integer TILE_CH       = 256;

    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST = 16'sd4071;
    localparam signed [SCALED_W-1:0]      SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0]      SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - $signed({{(SCALED_W-1){1'b0}}, 1'b1});

    // [SYNTH-FIT BANKED 2026-06-07] weight ROM split into N_BANKS power-of-2-deep banks.
    localparam integer BANK_AW    = 18;            // 2^18 = 262144 entries per bank
    localparam integer BANK_DEPTH = (1 << BANK_AW);
    localparam integer N_BANKS    = 5;             // ceil(M*K / BANK_DEPTH) = ceil(1280000/262144)
    localparam integer SELW       = 3;             // bits to index 5 banks (w_addr[20:18])

    (* rom_style = "block", ram_style = "block" *) reg signed [7:0] bank0 [0:BANK_DEPTH-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [7:0] bank1 [0:BANK_DEPTH-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [7:0] bank2 [0:BANK_DEPTH-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [7:0] bank3 [0:BANK_DEPTH-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [7:0] bank4 [0:BANK_DEPTH-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases  [0:M-1];

    initial begin
        // bank b holds the CONTIGUOUS flat slice [b*BANK_DEPTH, (b+1)*BANK_DEPTH) of the row-major
        // weight hex -> byte-identical to the prior single weights[] array. Last bank is partially
        // filled (covers indices 1048576..1279999); the unused tail is never addressed.
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_linear_weights_bank0.hex", bank0);
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_linear_weights_bank1.hex", bank1);
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_linear_weights_bank2.hex", bank2);
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_linear_weights_bank3.hex", bank3);
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_linear_weights_bank4.hex", bank4);
        $readmemh("C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_linear_bias.hex", biases);
    end

    localparam [2:0] ST_IDLE      = 3'd0;
    localparam [2:0] ST_MAC       = 3'd1;   // serial multiply-accumulate over k = 0..K-1 (registered read)
    localparam [2:0] ST_MAC_DRAIN = 3'd2;   // 1-cycle drain: accumulate the last registered tap (k=K-1)
    localparam [2:0] ST_FIN       = 3'd3;   // finalize: bias/scale/round/clamp -> out_buf[m]
    localparam [2:0] ST_EMIT      = 3'd4;

    reg [2:0]        state;
    reg [15:0]       m_counter;
    reg [KLOG2-1:0]  k_counter;          // 0..K-1
    reg              emit_now;
    reg [2:0]        load_tile;          // [BEATSPLIT] 0..N_TILES-1
    // [SYNTH-FIT A1 2026-06-06 / regread 2026-06-07] SINGLE FLAT ROM READ-ADDRESS REGISTER.
    // The weight ROM read index is a PLAIN REGISTER (w_addr), NOT a combinational sum
    // (weight_base + k_counter). w_addr free-runs 0,1,2,...,M*K-1 across the frame -- the row-major
    // read order is exactly m*K+k -- and is incremented by +1 each MAC tap. A registered address +
    // a registered read OUTPUT (w_q) is the canonical synchronous-ROM pattern; banked here so each
    // 262144-deep bank infers RAMB36. w_addr spans 0..1,279,999 (< 2^21), a clean depth = M*K.
    // Byte-identical: the (m,k) -> index map is unchanged (still m*K+k).
    reg [20:0]       w_addr;             // 0 .. (M-1)*K+(K-1) = 1,279,999 < 2^21 (flat ROM read addr)

    // [SYNTH-FIT 2026-06-06] 2D wide-word reshape: one 2048b word per input beat (5 deep).
    // Feature k lives at in_buf2d[k>>8][(k&255)*8 +: 8] -- byte-identical mapping to the prior
    // in_buf[k] (beat k/256, lane k%256). ONE write port (whole word) kills the 256-write dissolve.
    (* ram_style = "block" *) reg [TILE_CH*8-1:0] in_buf2d [0:N_TILES-1];
    reg signed [7:0] out_buf [0:M-1];

    integer m, lane;
    reg signed [ACC_W-1:0]    acc_reg;   // persistent serial accumulator
    // [SYNTH-FIT 2026-06-07] registered-read pipeline: the weight ROM read OUTPUT must be REGISTERED
    // for Vivado to infer block RAM (a pure-add address alone is necessary but NOT sufficient). Each
    // ST_MAC cycle registers the per-bank ROM reads (b*_q), the registered bank-select (bank_sel_q),
    // the aligned input byte (x_q) and a valid/first flag; w_q is a COMBINATIONAL mux of the
    // registered bank reads -> still exactly ONE cycle behind w_addr (NO added latency). The
    // accumulate runs one cycle behind off those registers. Same products, same integer sum
    // (associative) => bit-identical acc_reg; only +1 cycle/row of internal latency (ST_MAC_DRAIN).
    reg signed [7:0]          b0_q, b1_q, b2_q, b3_q, b4_q; // registered per-bank weight reads (=> BRAM)
    reg [SELW-1:0]            bank_sel_q;                   // registered bank index (w_addr[20:18])
    reg signed [7:0]          x_q;       // registered aligned input byte
    reg                       mac_v;     // a tap was registered last cycle -> accumulate it now
    reg                       first_q;   // the registered tap was k_counter==0 (clears prior m's sum)
    reg signed [BIASED_W-1:0] biased_tmp;
    reg signed [SCALED_W-1:0] scaled_tmp;
    reg signed [SCALED_W-1:0] v_tmp;
    reg signed [7:0]          clamped_tmp;

    // w_q: combinational mux of the 5 REGISTERED bank reads by the REGISTERED bank index. This is
    // identical in timing/value to the prior `w_q <= weights[w_addr]` (one-cycle-behind read).
    reg signed [7:0]          w_q;
    always @(*) begin
        case (bank_sel_q)
            3'd0:    w_q = b0_q;
            3'd1:    w_q = b1_q;
            3'd2:    w_q = b2_q;
            3'd3:    w_q = b3_q;
            default: w_q = b4_q;   // bank 4 (last)
        endcase
    end

    always @(posedge clk) begin
        // [BEATSPLIT] fill in_buf over N_TILES beats of 256 features (beat load_tile -> features
        // load_tile*256 .. +255). Identical bytes land in identical in_buf slots vs the flat latch.
        if (state == ST_IDLE && valid_in && ready_in) begin
            in_buf2d[load_tile] <= data_in;   // one wide-word write, ONE write port
        end

        // [SYNTH-FIT BANKED 2026-06-07] STAGE 1 (registered read): every ST_MAC cycle, register each
        // bank's read at the SAME in-bank address (w_addr[17:0]) -- a PLAIN REGISTER ADDRESS into a
        // registered OUTPUT per bank, the canonical synchronous-ROM that infers block RAM -- plus the
        // registered bank index (w_addr[20:18]), the aligned input byte (x_q), and the valid/first
        // flags. Nothing combinational sits between a bank's ROM read and its b*_q register; w_q then
        // picks the addressed bank via bank_sel_q (same cycle alignment as the prior single read).
        b0_q       <= bank0[w_addr[BANK_AW-1:0]];
        b1_q       <= bank1[w_addr[BANK_AW-1:0]];
        b2_q       <= bank2[w_addr[BANK_AW-1:0]];
        b3_q       <= bank3[w_addr[BANK_AW-1:0]];
        b4_q       <= bank4[w_addr[BANK_AW-1:0]];
        bank_sel_q <= w_addr[20:BANK_AW];
        x_q        <= $signed(in_buf2d[k_counter[KLOG2-1:8]][(k_counter[7:0])*8 +: 8]);
        mac_v      <= (state == ST_MAC);
        first_q    <= (k_counter == 0);

        // STAGE 2 (accumulate, one cycle behind): consume the registered tap. On the first tap
        // (first_q) acc_reg starts from the product (clears the prior m's sum); subsequent taps add.
        // The tap registered when k_counter==K-1 is accumulated in ST_MAC_DRAIN; after that acc_reg
        // holds the COMPLETE dot product (read in ST_FIN). Same products, same integer sum
        // (associative) => bit-identical to the prior serial/unrolled acc.
        if (mac_v) begin
            if (first_q)
                acc_reg <= $signed(x_q) * $signed(w_q);
            else
                acc_reg <= acc_reg + $signed(x_q) * $signed(w_q);
        end

        // [SYNTH-FIT] finalize for m_counter: acc_reg = complete dot product. Math copied verbatim.
        if (state == ST_FIN) begin
            biased_tmp = acc_reg + $signed(biases[m_counter]);
            scaled_tmp = biased_tmp * SCALE_MULT_CONST;
            // [INVARIANT:ROUNDING] unconditional +2^(SHIFT-1)
            v_tmp = (scaled_tmp + SCALE_ROUND_HALF) >>> SCALE_SHIFT;
            clamped_tmp = (v_tmp > 127)  ?  8'sd127 :
                          (v_tmp < -128) ? -8'sd128 : v_tmp[7:0];
            out_buf[m_counter] <= clamped_tmp;
        end

        if (emit_now) begin
            for (m = 0; m < M; m = m + 1) begin
                dp_data_out[m*8 +: 8] <= out_buf[m];
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= ST_IDLE;
            m_counter    <= 16'd0;
            k_counter    <= {KLOG2{1'b0}};
            w_addr       <= 21'd0;
            load_tile    <= 3'd0;
            ready_in     <= 1'b1; // [INVARIANT:READY_IN_GATING]
            dp_valid_out <= 1'b0;
            emit_now     <= 1'b0;
        end else begin
            dp_valid_out <= 1'b0;
            emit_now     <= 1'b0;
            case (state)
                ST_IDLE: begin
                    ready_in <= !skid_block;
                    if (valid_in && ready_in && !skid_block) begin
                        // [BEATSPLIT] accept N_TILES input beats, then start the serial MAC.
                        if (load_tile == N_TILES - 1) begin
                            load_tile <= 3'd0;
                            ready_in  <= 1'b0; // [INVARIANT:READY_IN_GATING]
                            m_counter <= 16'd0;
                            k_counter <= {KLOG2{1'b0}};
                            w_addr    <= 21'd0;   // flat ROM read addr starts at index 0 (m=0,k=0)
                            state     <= ST_MAC;
                        end else begin
                            load_tile <= load_tile + 3'd1;
                        end
                    end
                end
                ST_MAC: begin
                    // walk taps 0..K-1 (registered each cycle, accumulated one cycle behind); reset
                    // k_counter on the last tap so the next m starts at 0. The tap registered this
                    // cycle (k=K-1) is accumulated in ST_MAC_DRAIN before ST_FIN reads acc_reg.
                    // w_addr free-runs +1 per tap: after row m's K taps it sits at (m+1)*K, which is
                    // exactly row m+1's first index (held through DRAIN/FIN). Row-major order m*K+k.
                    w_addr <= w_addr + 21'd1;
                    if (k_counter == K - 1) begin
                        k_counter <= {KLOG2{1'b0}};
                        state     <= ST_MAC_DRAIN;
                    end else begin
                        k_counter <= k_counter + 1'b1;
                    end
                end
                ST_MAC_DRAIN: begin
                    // 1-cycle drain: this cycle the accumulate consumes the last registered tap
                    // (k=K-1). After it lands, acc_reg holds the COMPLETE dot product for ST_FIN.
                    state <= ST_FIN;
                end
                ST_FIN: begin
                    // out_buf[m_counter] written this cycle (datapath). Advance to next m or emit.
                    if (m_counter == M - 1) begin
                        emit_now  <= 1'b1;
                        m_counter <= 16'd0;
                        w_addr    <= 21'd0;    // frame done: reset flat ROM addr for the next frame
                        state     <= ST_EMIT;
                    end else begin
                        m_counter <= m_counter + 16'd1;
                        // w_addr already == (m_counter+1)*K (free-ran in ST_MAC); next row resumes here.
                        state     <= ST_MAC;   // k_counter already 0 (reset in ST_MAC last tap)
                    end
                end
                ST_EMIT: begin
                    dp_valid_out <= 1'b1; // [INVARIANT:VALID_OUT_LATENCY]
                    ready_in     <= !skid_block; // [INVARIANT:READY_IN_GATING]
                    state        <= ST_IDLE;
                end
                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
