// node_linear — gemm (fully-connected) classifier. [BEATSPLIT 2026-06-03] CHANNEL-TILED INPUT.
// data_in = 2048b (256 features) x N_TILES=5 beats -> fills in_buf[0:1279] (feature k = beat k/256,
// lane k%256). Serial MAC, bias, scale (MULT=4071,SHIFT=20), round-half, clamp, and the 8000b
// single-beat OUTPUT are UNCHANGED -> byte-identical logits; only the input beat structure changed.
module node_linear_ref #(
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

    (* rom_style = "block", ram_style = "block" *) reg signed [7:0]  weights [0:M*K-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases  [0:M-1];

    initial begin
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_linear_weights.hex", weights);
        $readmemh("D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/mobilenet-v2/weights/node_linear_bias.hex", biases);
    end

    localparam [1:0] ST_IDLE    = 2'd0;
    localparam [1:0] ST_COMPUTE = 2'd1;
    localparam [1:0] ST_WAIT    = 2'd2;
    localparam [1:0] ST_EMIT    = 2'd3;

    reg [1:0]  state;
    reg [15:0] cycle_count;
    reg [15:0] m_counter;
    reg        emit_now;
    reg [2:0]  load_tile;   // [BEATSPLIT] 0..N_TILES-1

    reg signed [7:0] in_buf  [0:K-1];
    reg signed [7:0] out_buf [0:M-1];

    integer i, k, m, lane;
    reg signed [ACC_W-1:0]    acc_tmp;
    reg signed [BIASED_W-1:0] biased_tmp;
    reg signed [SCALED_W-1:0] scaled_tmp;
    reg signed [SCALED_W-1:0] v_tmp;
    reg signed [7:0]          clamped_tmp;

    always @(posedge clk) begin
        // [BEATSPLIT] fill in_buf over N_TILES beats of 256 features (beat load_tile -> features
        // load_tile*256 .. +255). Identical bytes land in identical in_buf slots vs the flat latch.
        if (state == ST_IDLE && valid_in && ready_in) begin
            for (lane = 0; lane < TILE_CH; lane = lane + 1) begin
                in_buf[load_tile*TILE_CH + lane] <= $signed(data_in[lane*8 +: 8]);
            end
        end

        if (state == ST_COMPUTE) begin
            acc_tmp = {ACC_W{1'b0}};
            for (k = 0; k < K; k = k + 1) begin
                acc_tmp = acc_tmp + $signed(in_buf[k]) * $signed(weights[m_counter * K + k]);
            end
            biased_tmp = acc_tmp + $signed(biases[m_counter]);
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
            state       <= ST_IDLE;
            cycle_count <= 16'd0;
            m_counter   <= 16'd0;
            load_tile   <= 3'd0;
            ready_in    <= 1'b1; // [INVARIANT:READY_IN_GATING]
            dp_valid_out <= 1'b0;
            emit_now    <= 1'b0;
        end else begin
            dp_valid_out <= 1'b0;
            emit_now  <= 1'b0;
            case (state)
                ST_IDLE: begin
                    ready_in <= !skid_block;
                    if (valid_in && ready_in && !skid_block) begin
                        // [BEATSPLIT] accept N_TILES input beats, then start the MAC.
                        if (load_tile == N_TILES - 1) begin
                            load_tile   <= 3'd0;
                            ready_in    <= 1'b0; // [INVARIANT:READY_IN_GATING]
                            cycle_count <= 16'd1;
                            m_counter   <= 16'd0;
                            state       <= ST_COMPUTE;
                        end else begin
                            load_tile <= load_tile + 3'd1;
                        end
                    end
                end
                ST_COMPUTE: begin
                    cycle_count <= cycle_count + 16'd1;
                    if (m_counter == M - 1) begin
                        state     <= ST_WAIT;
                        m_counter <= 16'd0;
                    end else begin
                        m_counter <= m_counter + 16'd1;
                    end
                end
                ST_WAIT: begin
                    cycle_count <= cycle_count + 16'd1;
                    if (cycle_count == 16'd1321) begin
                        emit_now <= 1'b1;
                        state    <= ST_EMIT;
                    end
                end
                ST_EMIT: begin
                    dp_valid_out <= 1'b1; // [INVARIANT:VALID_OUT_LATENCY]
                    ready_in  <= !skid_block; // [INVARIANT:READY_IN_GATING]
                    state     <= ST_IDLE;
                end
                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
