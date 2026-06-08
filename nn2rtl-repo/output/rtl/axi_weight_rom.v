`timescale 1ns / 1ps
// ---------------------------------------------------------------------------
// axi_weight_rom -- SIM-ONLY behavioral DRAM weight model.
//
// The DRAM-backed convs (conv_284/288/292/298) stream their weights from
// external DRAM over a simplified AXI4 read channel. On real hardware that
// DRAM holds the weights; in the integrated Verilator sim there is no DRAM, so
// the weight AXI was tied off and those convs could never load weights (they
// hang in their AR_ISSUE/AR_WAIT_DATA FSM). This model serves the weights from
// the same per-byte .hex used elsewhere, as 64-bit read beats, so the sim can
// run the full network end-to-end.
//
// Protocol (matches node_conv_28x weight FSM):
//   * arvalid/arready single-cycle AR handshake; araddr is a BYTE address,
//     arlen is "beats-1" (the convs use 255 -> 256-beat bursts).
//   * then (arlen+1) R beats: rvalid/rready, rdata=64b, rlast on final beat.
//   * rdata is the little-endian pack of 8 consecutive bytes
//     (rdata[k*8 +: 8] = mem[araddr + beat*8 + k]), matching the conv's
//     cache_word[byte*8 +: 8] read where byte = weight_addr[2:0].
//
// NOT SYNTHESIZABLE (huge reg array + $readmemh). Sim/verification only.
// ---------------------------------------------------------------------------
module axi_weight_rom #(
    parameter integer WEIGHT_BYTES = 2359296,
    parameter         WEIGHTS_PATH = "weights.hex"
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        arvalid,
    output reg         arready,
    input  wire [31:0] araddr,
    input  wire [7:0]  arlen,
    output reg         rvalid,
    input  wire        rready,
    output reg  [63:0] rdata,
    output reg         rlast
);
    reg [7:0] wmem [0:WEIGHT_BYTES-1];
    initial $readmemh(WEIGHTS_PATH, wmem);

    localparam S_IDLE = 1'b0, S_DATA = 1'b1;
    reg        state;
    reg [31:0] base;     // byte address of the CURRENT beat
    reg [8:0]  beat;     // 0 .. arlen
    reg [8:0]  len_l;    // latched arlen (9 bits to hold 255 + margin)
    integer k;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= S_IDLE;
            arready <= 1'b1;
            rvalid  <= 1'b0;
            rlast   <= 1'b0;
            base    <= 32'd0;
            beat    <= 9'd0;
            len_l   <= 9'd0;
            rdata   <= 64'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    rvalid  <= 1'b0;
                    rlast   <= 1'b0;
                    arready <= 1'b1;
                    if (arvalid && arready) begin
                        arready <= 1'b0;
                        base    <= araddr;
                        len_l   <= {1'b0, arlen};
                        beat    <= 9'd0;
                        for (k = 0; k < 8; k = k + 1)
                            rdata[k*8 +: 8] <= wmem[araddr + k];
                        rvalid  <= 1'b1;
                        rlast   <= (arlen == 8'd0);
                        state   <= S_DATA;
                    end
                end
                S_DATA: begin
                    if (rvalid && rready) begin
                        if (beat == len_l) begin
                            rvalid  <= 1'b0;
                            rlast   <= 1'b0;
                            arready <= 1'b1;
                            state   <= S_IDLE;
                        end else begin
                            beat <= beat + 9'd1;
                            base <= base + 32'd8;
                            for (k = 0; k < 8; k = k + 1)
                                rdata[k*8 +: 8] <= wmem[base + 32'd8 + k];
                            rlast <= ((beat + 9'd1) == len_l);
                        end
                    end
                end
            endcase
        end
    end
endmodule
