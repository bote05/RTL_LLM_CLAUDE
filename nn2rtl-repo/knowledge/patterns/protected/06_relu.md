# 06 — ReLU

## When to use

`op_type == "relu"`.

## Semantics

Per channel:

```
out_i = (in_i > 0) ? in_i : 8'sd0
```

No scale factor. No saturation (negative clamps to 0, positive is already
INT8). Combinational except for the output register.

## Latency contract

Typically `pipeline_latency_cycles == 1`. Use the LayerIR value.

## Required FSM

A bare register on the output with a ready/valid pipeline stage:

```verilog
always @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    valid_out <= 1'b0;
    ready_in  <= 1'b1;
    data_out  <= 0;
  end else begin
    valid_out <= valid_in;
    if (valid_in) begin
      for (i = 0; i < OC; i = i + 1) begin
        // [INVARIANT:ROUNDING]  -- no rounding needed but left marker-free
        data_out[i*8 +: 8] <= ($signed(data_in[i*8 +: 8]) > 0)
                               ? data_in[i*8 +: 8]
                               : 8'sd0;
      end
    end
  end
end
```

No weights, no biases, no window, no line buffer. The output-counter
preflight rule does not apply.

## Known failure modes

- `sign_extension_error` — comparison performed in unsigned context, so
  all negative channels pass through as large positives. Always use
  `$signed(...)`.
- `saturation_missing` — somebody added redundant min/max clamps and
  introduced a bug. The output of ReLU on INT8 is by construction in
  `[0, 127]`; no extra saturation is needed.
