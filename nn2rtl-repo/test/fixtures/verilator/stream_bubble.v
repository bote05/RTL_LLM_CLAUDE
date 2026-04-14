module stream_bubble (
  input  wire       clk,
  input  wire       rst_n,
  input  wire       valid_in,
  input  wire [7:0] data_in,
  output wire       ready_in,
  output reg        valid_out,
  output reg  [7:0] data_out
);
  localparam IDLE = 2'd0;
  localparam WAITING = 2'd1;
  localparam OUTPUTTING = 2'd2;

  reg [1:0] state;
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
            state <= WAITING;
          end
        end
        WAITING: begin
          state <= OUTPUTTING;
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
