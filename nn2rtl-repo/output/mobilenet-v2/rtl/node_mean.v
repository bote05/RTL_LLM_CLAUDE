// node_mean - INT8 global average pool. [BRAM-FIX 2026-06-05] SYNTH-RAM fix.
// PRIOR PROBLEM: acc/scaled/rounded were flat reg arrays [0:1279] touched by single-cycle
// FULL-WIDTH parallel for-loops (ST_ACCUM 256-wide RMW, ST_ROUND/ST_PACK all-1280). Vivado
// could not infer block RAM (>2 write ports / "RAM has too many ports") so it DISSOLVED
// acc into 20480 registers + giant mux cones -> synth OOM'd 96GB (see mbv2_synth.log
// [Synth 8-4767]/[Synth 8-13159]). FIX: reshape the three arrays into 2D PACKED WIDE-WORD
// memories addressed ONE word/cycle, and serialize ST_ROUND/ST_PACK to the 16-lane pattern
// ST_SCALE already used (scale_idx). Each array is now 1 read + 1 write port -> BRAM-friendly,
// no mux-cone explosion. MATH IS BYTE-IDENTICAL (same accumulate / MULT=7619 / SHIFT=18 /
// round-half / clamp; same per-channel output byte order); only the I/O-stable schedule grows
// by ~158 cycles (ST_ROUND 1->80, ST_PACK 1->80). GAP runs once/frame so latency is negligible.
// Decoupled multi-beat output emitter (emit_busy/emit_tile) unchanged.
module node_mean #(
    parameter ENABLE_BACKPRESSURE = 0
)(
    input  wire                clk,
    input  wire                rst_n,
    input  wire                valid_in,
    output reg                 ready_in,
    input  wire [2047:0]       data_in,
    input  wire                out_ready_in,
    output wire                valid_out,
    output wire [2047:0]       data_out
);
    localparam integer C             = 1280;
    localparam integer HW            = 49;
    localparam integer N_TILES       = 5;     // 5 * 256ch = 1280 ; TILE_W = 2048b
    localparam integer TILE_CH       = 256;
    localparam integer ACC_W         = 16;
    localparam integer SCALE_MULT    = 7619;
    localparam integer SCALE_SHIFT   = 18;
    localparam integer SCALE_CONST_W = 14;
    localparam integer SCALED_W      = ACC_W + SCALE_CONST_W;
    localparam integer ROUNDED_W     = 16;
    localparam integer SCALE_LANES   = 16;
    localparam integer SCALE_STEPS   = C / SCALE_LANES; // 80

    localparam signed [SCALE_CONST_W-1:0] SCALE_MULT_CONST = SCALE_MULT;
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF =
        {{(SCALED_W-1){1'b0}}, 1'b1} <<< (SCALE_SHIFT - 1);
    localparam signed [SCALED_W-1:0] SCALE_ROUND_HALF_M1 =
        SCALE_ROUND_HALF - {{(SCALED_W-1){1'b0}}, 1'b1};

    localparam [1:0] ST_ACCUM = 2'd0;
    localparam [1:0] ST_SCALE = 2'd1;
    localparam [1:0] ST_ROUND = 2'd2;
    localparam [1:0] ST_PACK  = 2'd3;

    reg [1:0] state;
    reg [6:0] cell_count;  // 0 .. HW-1
    reg [2:0] in_tile;     // 0 .. N_TILES-1 (input tile within a spatial cell)
    reg [6:0] scale_idx;   // 0 .. SCALE_STEPS-1
    reg [6:0] round_idx;   // 0 .. SCALE_STEPS-1  [BRAM-FIX] serialized ROUND
    reg [6:0] pack_idx;    // 0 .. SCALE_STEPS-1  [BRAM-FIX] serialized PACK

    // [BRAM-FIX] 2D packed wide-word memories: one addressed word per cycle (1R/1W) =>
    // block-RAM-inferable, no 256/1280-way parallel-port mux cone.
    //   acc_mem   : N_TILES   x (TILE_CH*ACC_W)     = 5  x 4096b
    //   scaled_mem: SCALE_STEPS x (SCALE_LANES*SCALED_W)  = 80 x 480b
    //   rounded_mem:SCALE_STEPS x (SCALE_LANES*ROUNDED_W) = 80 x 256b
    reg        [TILE_CH*ACC_W-1:0]            acc_mem     [0:N_TILES-1];
    (* ram_style = "block" *) reg [SCALE_LANES*SCALED_W-1:0]  scaled_mem  [0:SCALE_STEPS-1];
    (* ram_style = "block" *) reg [SCALE_LANES*ROUNDED_W-1:0] rounded_mem [0:SCALE_STEPS-1];
    reg signed [SCALED_W-1:0]  v_tmp;

    // ---- DECOUPLED output emitter (independent of the main FSM) ----
    reg [C*8-1:0] emit_data;   // local 10240b result store (NOT a module bus)
    reg [2:0]     emit_tile;   // 0 .. N_TILES-1
    reg           emit_busy;
    wire emit_accept = (ENABLE_BACKPRESSURE == 0) ? 1'b1 : out_ready_in;
    assign valid_out = emit_busy;
    assign data_out  = emit_data[emit_tile*2048 +: 2048];

    integer lane;   // datapath-block loop var (ST_ACCUM/ST_SCALE/ST_ROUND)
    integer plane;  // FSM-block loop var (ST_PACK) — separate to avoid shared-loop-var race

    // ---- datapath: tiled accumulate + time-mux scale + serialized round (math UNCHANGED) ----
    always @(posedge clk) begin
        // ST_ACCUM: read-modify-write ONE 256-channel tile word at addr in_tile (single word port).
        if (state == ST_ACCUM && valid_in && ready_in && !emit_busy) begin
            for (lane = 0; lane < TILE_CH; lane = lane + 1) begin
                if (cell_count == 7'd0)
                    acc_mem[in_tile][lane*ACC_W +: ACC_W] <= $signed(data_in[lane*8 +: 8]);
                else
                    acc_mem[in_tile][lane*ACC_W +: ACC_W] <=
                        $signed(acc_mem[in_tile][lane*ACC_W +: ACC_W]) + $signed(data_in[lane*8 +: 8]);
            end
        end

        // ST_SCALE: 16 lanes/cycle (scale_idx 0..79). The 16 channels scale_idx*16..+15 live in
        // acc tile (scale_idx>>4) at sub-offset (scale_idx&15)*16. Read that sub-word, write scaled word.
        if (state == ST_SCALE) begin
            for (lane = 0; lane < SCALE_LANES; lane = lane + 1)
                scaled_mem[scale_idx][lane*SCALED_W +: SCALED_W] <=
                    $signed(acc_mem[scale_idx >> 4][((scale_idx & 7'd15)*SCALE_LANES + lane)*ACC_W +: ACC_W])
                    * $signed(SCALE_MULT_CONST);
        end

        // ST_ROUND: serialized 16 lanes/cycle (round_idx 0..79). Same round-half math as before.
        if (state == ST_ROUND) begin
            for (lane = 0; lane < SCALE_LANES; lane = lane + 1) begin
                v_tmp = ($signed(scaled_mem[round_idx][lane*SCALED_W +: SCALED_W]) +
                         (scaled_mem[round_idx][lane*SCALED_W + (SCALED_W-1)] ? SCALE_ROUND_HALF_M1
                                                                             : SCALE_ROUND_HALF)
                        ) >>> SCALE_SHIFT; // [INVARIANT:ROUNDING]
                rounded_mem[round_idx][lane*ROUNDED_W +: ROUNDED_W] <= v_tmp[ROUNDED_W-1:0];
            end
        end
    end

    // ---- FSM + decoupled emitter (single driver for emit_busy/emit_tile/state/counters) ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= ST_ACCUM;
            cell_count <= 7'd0;
            in_tile    <= 3'd0;
            scale_idx  <= 7'd0;
            round_idx  <= 7'd0;
            pack_idx   <= 7'd0;
            ready_in   <= 1'b1;
            emit_busy  <= 1'b0;
            emit_tile  <= 3'd0;
            emit_data  <= {(C*8){1'b0}};
        end else begin
            // --- decoupled emitter: drains 5 beats independent of `state` ---
            if (emit_busy && emit_accept) begin
                if (emit_tile == N_TILES - 1) begin
                    emit_busy <= 1'b0;
                    emit_tile <= 3'd0;
                end else begin
                    emit_tile <= emit_tile + 3'd1;
                end
            end

            // --- main accumulate/scale/round/pack FSM ---
            case (state)
                ST_ACCUM: begin
                    ready_in <= !emit_busy;
                    if (valid_in && ready_in && !emit_busy) begin
                        if (in_tile == N_TILES - 1) begin
                            in_tile <= 3'd0;
                            if (cell_count == HW - 1) begin
                                cell_count <= 7'd0;
                                scale_idx  <= 7'd0;
                                ready_in   <= 1'b0; // [INVARIANT:READY_IN_GATING]
                                state      <= ST_SCALE;
                            end else begin
                                cell_count <= cell_count + 7'd1;
                            end
                        end else begin
                            in_tile <= in_tile + 3'd1;
                        end
                    end
                end

                ST_SCALE: begin
                    if (scale_idx == SCALE_STEPS - 1) begin
                        scale_idx <= 7'd0;
                        round_idx <= 7'd0;
                        state     <= ST_ROUND;
                    end else begin
                        scale_idx <= scale_idx + 7'd1;
                    end
                end

                ST_ROUND: begin
                    if (round_idx == SCALE_STEPS - 1) begin
                        round_idx <= 7'd0;
                        pack_idx  <= 7'd0;
                        state     <= ST_PACK;
                    end else begin
                        round_idx <= round_idx + 7'd1;
                    end
                end

                ST_PACK: begin
                    // serialized clamp+pack: 16 channels/cycle (pack_idx 0..79) -> emit_data.
                    for (plane = 0; plane < SCALE_LANES; plane = plane + 1)
                        emit_data[(pack_idx*SCALE_LANES + plane)*8 +: 8] <=
                            ($signed(rounded_mem[pack_idx][plane*ROUNDED_W +: ROUNDED_W]) >  16'sd127)  ?  8'sd127 :
                            ($signed(rounded_mem[pack_idx][plane*ROUNDED_W +: ROUNDED_W]) < -16'sd128)  ? -8'sd128 :
                                                                       rounded_mem[pack_idx][plane*ROUNDED_W +: 8];
                    if (pack_idx == SCALE_STEPS - 1) begin
                        pack_idx  <= 7'd0;
                        emit_busy <= 1'b1;
                        emit_tile <= 3'd0;
                        state     <= ST_ACCUM;
                    end else begin
                        pack_idx <= pack_idx + 7'd1;
                    end
                end

                default: state <= ST_ACCUM;
            endcase
        end
    end

endmodule
