`timescale 1ns/1ps
// output_serializer.v  [FMAX/OOC-FIX 2026-06-08]
// ---------------------------------------------------------------------------
// Serializes node_linear's W_IN-bit parallel logit word into NBEATS beats of
// BEATW bits on an AXI4-Stream-style valid/ready/last handshake -- the SAME
// streaming output contract ResNet uses (output/rtl/nn2rtl_top.v m_axis = 256b
// stream), instead of MobileNetV2's 8000-bit single-beat parallel output.
//
// WHY: the 8000-bit m_axis_tdata bus is (a) 8000 top-level OUTPUT PINS, which
// exceeds the U250's ~676 user I/O -> forces out-of-context implementation; and
// (b) ~8000 parallel output nets feeding one pin bundle = a congestion source.
// Narrowing the pin bus to 256b lets the whole design place IN-CONTEXT (no OOC)
// exactly like ResNet, and deletes the 8000-net bundle.
//
// BYTE-EXACT: this is a pure in-order RE-SLICE. The concatenation of the NBEATS
// emitted beats (low beat first) reproduces data_in[W_IN-1:0] bit-for-bit; the
// final beat's unused high lanes (NBEATS*BEATW - W_IN bits) are zero pad that no
// consumer reads. node_linear is UNCHANGED -> its verified MAC/requant is intact.
//
// Handshake: ready_out is high only when idle (accepts ONE word, then streams).
// node_linear emits one word per frame (~1.28M cycles apart) so it never stalls.
// Each beat is held until ready_in; last_out marks beat NBEATS-1.
// ---------------------------------------------------------------------------
module output_serializer #(
    parameter integer W_IN  = 8000,
    parameter integer BEATW = 256
) (
    input  wire             clk,
    input  wire             rst_n,
    // upstream (node_linear): one W_IN-bit word
    input  wire             valid_in,
    input  wire [W_IN-1:0]  data_in,
    output wire             ready_out,
    // downstream (m_axis): BEATW-bit stream
    output reg              valid_out,
    output reg  [BEATW-1:0] data_out,
    output reg              last_out,
    input  wire             ready_in
);
    localparam integer NBEATS = (W_IN + BEATW - 1) / BEATW;        // ceil(8000/256)=32
    localparam integer BCW    = (NBEATS <= 1) ? 1 : $clog2(NBEATS); // 5
    localparam integer BUF_W  = NBEATS * BEATW;                     // 8192 (>= W_IN)

    reg [BUF_W-1:0] buf_data;   // zero-extended word being streamed
    reg             busy;
    reg [BCW-1:0]   beat;

    assign ready_out = !busy;

    /* verilator lint_off WIDTH */
    // [K1-MBV2] buf_data/data_out are stream DATA (sync-only, no reset):
    // buf_data is fully written on the accept edge and read strictly after
    // (beats 1..NBEATS-1); data_out is sampled by the consumer only under
    // valid_out (reset-kept). Guards replicate the original branch
    // conditions exactly (busy/valid_in/valid_out/ready_in/beat control all
    // keep their async reset). Write sites are mutually exclusive on busy.
    always @(posedge clk) begin
        if (!busy) begin
            if (valid_in) begin
                buf_data <= {{(BUF_W-W_IN){1'b0}}, data_in};
                data_out <= data_in[0 +: BEATW];
            end
        end else begin
            if (valid_out && ready_in && (beat != NBEATS-1)) begin
                data_out <= buf_data[(beat + 1'b1)*BEATW +: BEATW];
            end
        end
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy      <= 1'b0;
            beat      <= {BCW{1'b0}};
            valid_out <= 1'b0;
            last_out  <= 1'b0;
        end else begin
            if (!busy) begin
                if (valid_in) begin
                    // latch the word (zero-extended) and emit beat 0 (read directly from
                    // data_in this cycle since buf_data only settles next cycle).
                    busy      <= 1'b1;
                    beat      <= {BCW{1'b0}};
                    valid_out <= 1'b1;
                    last_out  <= (NBEATS == 1);
                end
            end else begin
                if (valid_out && ready_in) begin       // current beat accepted
                    if (beat == NBEATS-1) begin
                        busy      <= 1'b0;
                        valid_out <= 1'b0;
                        last_out  <= 1'b0;
                    end else begin
                        beat     <= beat + 1'b1;
                        last_out <= ((beat + 1'b1) == NBEATS-1);
                    end
                end
            end
        end
    end
    /* verilator lint_on WIDTH */
endmodule
