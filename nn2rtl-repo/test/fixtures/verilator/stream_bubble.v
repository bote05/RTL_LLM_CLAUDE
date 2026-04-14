module stream_bubble (
  input  wire       clk,
  input  wire       rst_n,
  input  wire       valid_in,
  input  wire [7:0] data_in,
  output wire       ready_in,
  output reg        valid_out,
  output reg  [7:0] data_out
);
  localparam IDLE = 1'd0;
  localparam OUTPUTTING = 1'd1;

  reg state;
  reg [7:0] latched_data;

  assign ready_in = (state == IDLE);

  always @(posedge clk) begin
    if (!rst_n) begin
      state <= IDLE;
      latched_data <= 8'd0;
      valid_out <= 1'b0;
      data_out <= 8'd0;
    end else begin
      valid_out <= 1'b0;

      case (state)
        IDLE: begin
          if (valid_in) begin
            latched_data <= data_in;
            state <= OUTPUTTING;
          end
        end
        OUTPUTTING: begin
          valid_out <= 1'b1;
          data_out <= latched_data;
          state <= IDLE;
        end
        default: begin
          state <= IDLE;
        end
      endcase
    end
  end
endmodule
