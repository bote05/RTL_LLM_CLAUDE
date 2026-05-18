# 12 — Depthwise Convolution

## When to use

`op_type == "conv2d"` AND `groups == input_channels == output_channels`. The
import path emits `contract_id == "depthwise-conv"` for this case. LayerIR
geometry: standard 2D conv with `weight_shape = [C, 1, KH, KW]` (one filter
per channel; the second dim is 1 because each channel sees only itself).

This contract powers MobileNet-style separable conv blocks. The standard
1×1 expand and 1×1 project convs in those blocks stay on `flat-bus`; only
the 3×3 depthwise core uses this contract.

## How depthwise differs from standard conv

Standard 3×3 conv at the same layer shape:

```
out[oc, h, w] = Σ_{ic, kh, kw} in[ic, h+kh, w+kw] · weight[oc, ic, kh, kw]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              IC * KH * KW MACs per output, ALL channels reduced
```

Depthwise 3×3:

```
out[c,  h, w] = Σ_{kh, kw} in[c, h+kh, w+kw] · weight[c, kh, kw]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              KH * KW MACs per output, NO cross-channel reduction
```

The structural change: **no adder tree across channels**. Each output
channel is a fully independent 2D conv on its own input channel. For
`KH=KW=3`, every output is a 9-tap dot product, end of story.

## Semantics

For each spatial position `(h, w)` and channel `c`:

```
acc[c]    = Σ_{kh, kw} in[c, h_in + kh, w_in + kw] · weight[c, kh, kw]
biased    = acc[c] + bias[c]                           # INT32 add
scaled    = round_half_up( biased · SCALE_MULT >> SCALE_SHIFT )
out[c]    = clamp(scaled, -128, 127)
```

`(h_in, w_in)` accounts for stride and padding the usual way (see
`03_conv3x3_pad1.md` for the coord_scheduler details — depthwise uses the
SAME line-buffer + window machinery). The composite SCALE is the same
shape as standard conv: `input_scale * weight_scale / output_scale`.

## Latency contract

Reuse the spatial-conv latency formula. The only thing that changes is
the MAC budget per output:

```
fill_rows = max(KH - 1 - PH, 0)
fill_cols = max(KW - PW, 1)
mac_cycles_per_output = ceil((KH * KW) / mac_parallelism)
                       = ceil(9 / MP)             # for 3x3
pipeline_stages = 3                                # BIAS + SCALE + OUTPUT
latency = fill_rows * (IW + PW) + fill_cols + mac_cycles_per_output * (OH * OW) + pipeline_stages
```

Read `pipeline_latency_cycles` from the LayerIR — Python is the source of
truth. The compute_conv2d_latency_cycles helper handles this when the
groups field is set correctly.

## Architecture hints (no reference module provided)

Foundry must design this contract from the pattern alone. Hints:

- Re-use the conv library modules (`coord_scheduler.v`, `line_buf_window.v`)
  as the spatial backbone. They handle padding, stride, and window
  shifting identically for depthwise. The contract change is only in the
  datapath stage, not in the spatial machinery.
- The datapath REPLACES `conv_datapath.v`'s cross-channel adder tree with
  a per-channel 9-tap (or KH·KW) dot product. Each output channel reads
  its own 9 weights from a `[C, KH, KW]` weight memory; there is no
  IC-axis reduction.
- Weight memory: with C up to ~576 (MobileNetV2 worst case) and a 3×3
  kernel, total INT8 weight bytes = C·9 = 5184 — trivially one BRAM18.
  Address as `weight[c * KH*KW + kh*KW + kw]`.
- MP (mac_parallelism): with C up to several hundred and KH·KW = 9, the
  MP=4 lane scheme can either parallelize across the 9 taps per channel,
  or across MP channels concurrently. The latter typically maps better
  because there is no inter-channel data dependency.
- No adder tree to register-balance. Pipeline depth is much smaller than
  the standard-conv contract for the same layer shape, so Fmax tends to
  be HIGHER than the corresponding standard conv at the same channel
  count.

## Required public interface

Same canonical interface as every other op: `clk, rst_n, valid_in,
ready_in, data_in, valid_out, data_out`. Width fields come from the
LayerIR. The bus carries one packed pixel ([N, C, h, w]) per beat —
identical to flat-bus.

## Activation-memory discipline (MANDATORY — synthesis preflight rejects violations)

Depthwise layers stream large activation footprints (e.g. 96 ch × 112×112
= 1.2M bytes; 144 ch × 56×56 = 451k bytes). The way you declare those
memories is the difference between Vivado synthesizing in 3 minutes and
Vivado rejecting the design at preflight (or worse, hanging for 30
minutes before erroring). Two structural-preflight gates fire often on
this contract:

### `activation_memory_in_async_reset_block` (HARD FAIL)

You MUST NOT declare an activation-buffer write inside an
`always @(posedge clk or negedge rst_n)` block. Vivado refuses to infer
BRAM/LUTRAM from a memory whose array entries are reset by an async
edge — the only legal pattern is synchronous reset, OR no reset on the
array at all.

**WRONG (preflight rejects):**

```verilog
reg signed [7:0] act_buf [0:DEPTH-1];
always @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    for (i = 0; i < DEPTH; i = i + 1) act_buf[i] <= 8'sd0;   // ← FAIL
  end else if (write_en) begin
    act_buf[write_addr] <= data_in;
  end
end
```

**CORRECT (BRAM-inferable, no reset on the array):**

```verilog
(* ram_style = "block" *) reg signed [7:0] act_buf [0:DEPTH-1];

// Memory writes — NO reset clause, NO async-reset always
always @(posedge clk) begin
  if (write_en) act_buf[write_addr] <= data_in;
end

// Control signals (write_en, addresses, valid flags) DO get reset —
// in a SEPARATE always block. The memory contents are "don't care"
// after reset; correctness comes from the control flow ignoring stale
// entries until they're overwritten in normal operation.
always @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    write_en   <= 1'b0;
    write_addr <= '0;
  end else begin
    write_en   <= /* ... */;
    write_addr <= /* ... */;
  end
end
```

### `activation_memory_exceeds_vivado_variable_bit_limit` (HARD FAIL)

Vivado's parser has a per-variable bit cap around **900,000 bits**
(roughly 1 Mb). A single packed-flat declaration of a large activation
buffer trips this immediately.

**WRONG (preflight rejects when total > ~900k bits):**

```verilog
reg [144*56*56*8-1:0] act_buf;   // 3.6M bits → FAIL
```

**CORRECT — declare as unpacked memory (BRAM-friendly, no bit cap):**

```verilog
reg signed [7:0] act_buf [0:144*56*56-1];   // 451k unpacked entries — OK
```

The unpacked form is what Vivado expects for BRAM inference and has no
single-variable cap (each cell is its own scalar). Always prefer
`reg [WORD-1:0] mem [0:DEPTH-1]` over `reg [DEPTH*WORD-1:0] mem`.

### When a single unpacked array still exceeds the BRAM budget

For really large activation footprints (e.g. 96 ch × 112×112 = 1.2M
bytes ≈ 9.6 Mb), even unpacked memory may exceed comfortable per-BRAM
sizing or trip secondary preflight gates. In that case **bank by
channel group**:

```verilog
localparam BANKS = 4;
localparam BANK_DEPTH = (C * IH * IW) / BANKS;
(* ram_style = "block" *) reg signed [7:0] act_buf_b0 [0:BANK_DEPTH-1];
(* ram_style = "block" *) reg signed [7:0] act_buf_b1 [0:BANK_DEPTH-1];
(* ram_style = "block" *) reg signed [7:0] act_buf_b2 [0:BANK_DEPTH-1];
(* ram_style = "block" *) reg signed [7:0] act_buf_b3 [0:BANK_DEPTH-1];

wire [1:0] bank_sel = write_addr[ADDR_BITS-1:ADDR_BITS-2];
wire [LOCAL_ADDR_BITS-1:0] local_addr = write_addr[LOCAL_ADDR_BITS-1:0];
// Per-bank synchronous writes, no reset, gated by bank_sel
```

This both keeps each variable safely under the 1 Mb cap AND maps cleanly
to four parallel BRAM36s.

### Companion preflight gate: `large_scalarized_activation_memory`

If you declare the activation buffer as a SCALAR ARRAY of large
single-channel cells (instead of one BRAM-inferable unpacked memory),
the preflight rejects it for a different reason — thousands of
independently-addressable scalar registers can't be packed into BRAM
and explode the LUT-as-memory count. Same fix: unpacked memory with a
packed-wide word.

## Known failure modes (anticipated — there is no proven reference yet)

- `cross_channel_reduction_present` — accidentally summing across the IC
  axis. The depthwise output for channel `c` depends ONLY on input
  channel `c`. Any `Σ_ic` is a bug; the loop has only the spatial axes.
- `weight_layout_wrong` — addressing weight as `[oc, ic, kh, kw]` (the
  standard conv layout) when the hex file is packed `[c, kh, kw]`. The
  ONNX frontend writes one filter per channel; reading from `ic` dim is
  out of bounds.
- `bias_shape_mismatch` — bias is per-output-channel as usual, `[C]`.
  Same as standard conv.
- `signedness_dropped` — INT8 multiplies must be `$signed(...)` on both
  operands. Mixed signed/unsigned is rejected by `use-dsp` patterns and
  will block DSP inference.
- `activation_memory_in_async_reset_block` — frequent on this contract.
  Declaring activation-buffer writes inside `always @(posedge clk or
  negedge rst_n)` makes Vivado refuse BRAM inference. See the
  "Activation-memory discipline" section above. Symptom: Verilator
  passes (often bit-exact), then synthesis preflight hard-fails the
  attempt before Vivado runs.
- `activation_memory_exceeds_vivado_variable_bit_limit` — declaring the
  activation buffer as one giant packed `reg` (e.g.
  `reg [144*56*56*8-1:0] act_buf`) trips Vivado's per-variable bit cap.
  Use the unpacked-memory form `reg [7:0] act_buf [0:N-1]` and bank
  across channel groups when N exceeds the per-BRAM budget.
- `pipeline_overlap_breaks_timing_contract` — overlapping the next
  channel's MAC with the previous channel's requantize is fine, but the
  FIRST `valid_out` must still land exactly on
  `pipeline_latency_cycles`. The deterministic assayer rejects timing
  drift.
