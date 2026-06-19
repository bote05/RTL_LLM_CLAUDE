// node_mean - INT8 global average pool. [BEATSPLIT 2026-06-03 v2] CHANNEL-TILED I/O.
// data_in/data_out tiled to 2048b (256ch) x N_TILES=5 = 1280ch (was a single 10240b beat).
// Accumulate/time-mux-scale(MULT=7619,SHIFT=18)/round-half/clamp math is BYTE-IDENTICAL to the
// flat version; only the I/O beat structure changed (channel c = beat c/256, lane c%256).
// OUTPUT uses a DECOUPLED multi-beat emitter (emit_busy/emit_tile), separate from the main FSM,
// so the FSM returns to ST_ACCUM after ST_PACK while the emitter drains the 5 beats independently
// (mirrors the decoupling the original 1-deep skid gave). [v1's in-FSM ST_EMIT coupled the FSM to
// the spatial_run-gated handshake and DEADLOCKED; this decoupled version fixes that.]
module node_mean_ref #(
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

    reg signed [ACC_W-1:0]     acc     [0:C-1];
    reg signed [SCALED_W-1:0]  scaled  [0:C-1];
    reg signed [ROUNDED_W-1:0] rounded [0:C-1];
    reg signed [SCALED_W-1:0]  v_tmp;

    // ---- DECOUPLED output emitter (independent of the main FSM) ----
    reg [C*8-1:0] emit_data;   // local 10240b result store (NOT a module bus)
    reg [2:0]     emit_tile;   // 0 .. N_TILES-1
    reg           emit_busy;
    wire emit_accept = (ENABLE_BACKPRESSURE == 0) ? 1'b1 : out_ready_in;
    assign valid_out = emit_busy;
    assign data_out  = emit_data[emit_tile*2048 +: 2048];

    integer i, lane, base;

    // ---- datapath: tiled accumulate + time-mux scale + round (math UNCHANGED) ----
    always @(posedge clk) begin
        if (state == ST_ACCUM && valid_in && ready_in && !emit_busy) begin
            for (lane = 0; lane < TILE_CH; lane = lane + 1) begin
                if (cell_count == 7'd0)
                    acc[in_tile*TILE_CH + lane] <= $signed(data_in[lane*8 +: 8]);
                else
                    acc[in_tile*TILE_CH + lane] <=
                        acc[in_tile*TILE_CH + lane] + $signed(data_in[lane*8 +: 8]);
            end
        end

        if (state == ST_SCALE) begin
            base = scale_idx * SCALE_LANES;
            for (lane = 0; lane < SCALE_LANES; lane = lane + 1)
                scaled[base + lane] <= $signed(acc[base + lane]) * $signed(SCALE_MULT_CONST);
        end

        if (state == ST_ROUND) begin
            for (i = 0; i < C; i = i + 1) begin
                v_tmp = (scaled[i] +
                         (scaled[i][SCALED_W-1] ? SCALE_ROUND_HALF_M1
                                                : SCALE_ROUND_HALF)
                        ) >>> SCALE_SHIFT; // [INVARIANT:ROUNDING]
                rounded[i] <= v_tmp[ROUNDED_W-1:0];
            end
        end
    end

    // ---- FSM + decoupled emitter (single driver for emit_busy/emit_tile/state) ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= ST_ACCUM;
            cell_count <= 7'd0;
            in_tile    <= 3'd0;
            scale_idx  <= 7'd0;
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
                    // freeze new-frame accept while the emitter is still draining the prior result
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
                        state     <= ST_ROUND;
                    end else begin
                        scale_idx <= scale_idx + 7'd1;
                    end
                end

                ST_ROUND: state <= ST_PACK;

                ST_PACK: begin
                    // latch the clamped result into the emitter store + start the decoupled drain
                    for (i = 0; i < C; i = i + 1) begin
                        emit_data[i*8 +: 8] <= (rounded[i] > 16'sd127)  ?  8'sd127 :
                                               (rounded[i] < -16'sd128) ? -8'sd128 :
                                                                           rounded[i][7:0];
                    end
                    emit_busy <= 1'b1;
                    emit_tile <= 3'd0;
                    state     <= ST_ACCUM;
                end

                default: state <= ST_ACCUM;
            endcase
        end
    end

endmodule
