# Task 13 — Engine one-layer verification report

End-to-end Verilog testbench that drives the shared compute engine on ONE
heavy ResNet-50 layer, compares the result to the per-layer .goldout, and
emits a PASS/FAIL verdict.

## What was built

| Artifact | Path | Purpose |
| --- | --- | --- |
| Testbench | `tb/engine_one_layer_tb.v` | Self-contained iverilog TB; instantiates `shared_engine`, 8x `uram_weight_bank`, `bias_mem`, `act_unified_mem`. Pre-loads weight/bias `.mem` files, hand-loads activation BRAM from `.goldin` vector 0, runs the 13-step AXI4-Lite config program, pulses `engine_start`, polls `engine_done`, dumps `act_unified_mem[CFG_ACT_OUT_BASE..+196)` to `output/engine_tb_observed.hex`. |
| Comparator | `scripts/compare_engine_output.py` | Reads the observed wide-BRAM hex dump and `output/goldens/node_conv_246.goldout` vector 0, prints PASS/FAIL plus max_error / mismatch_count plus per-byte first-mismatch detail (pixel coords, channel, expected, got). Exit code 0 on PASS. |
| Driver | `scripts/run_engine_one_layer_tb.sh` | Git-Bash compatible. Sets `YOSYSHQ_ROOT` + `PATH` so iverilog can spawn its preprocessor/codegen children. Compiles with `iverilog -g2012 -gno-strict-declaration -DNN2RTL_ENGINE_SUBBLOCKS_PROVIDED`, runs `vvp` with `+TIMEOUT_CYCLES=10000000`, calls the comparator. |

## Layer picked

First dispatch in `output/rtl/nn2rtl_scheduler_schedule.json` (`dispatches[0]`):

| Field | Value |
| --- | --- |
| module_id | `node_conv_246` |
| op_type | conv2d |
| input shape | [1, 256, 28, 28] |
| output shape | [1, 256, 14, 14] |
| kernel / stride / padding | 3x3 / 2x2 / 1x1 |
| K_TOTAL = IC*KH*KW | 256 * 9 = 2304 |
| oc_passes (OC/256) | 1 |
| output pixels (OH*OW) | 196 |
| weight_base_word | 11155 |
| bias_base_word | 31 |
| scale_mult | 1284434803 |
| scale_shift | 39 |
| zero_point | 0 |
| act_in_base | 8192 |
| act_out_base | 4096 |
| .goldin path | `output/goldens/node_conv_246.goldin` |
| .goldout path | `output/goldens/node_conv_246.goldout` |

The .goldin / .goldout files are the NN2V binary vector format (see
`scripts/golden_impl.py`); the TB consumes vector index 0 of each
(8 test vectors total per file, only vector 0 used).

## Hard rules followed

- No file under `output/rtl/`, `knowledge/`, `output/weights/*.mem`, or
  `output/goldens/*` was modified.
- The TB does NOT regenerate goldens — it reads the existing files on disk.
- iverilog only (no Verilator). Use of `-gno-strict-declaration` is the
  only relaxation needed; the warning is harmless and only acknowledges
  a forward `wire oc_pass_total_m1` reference inside
  `shared_engine_skeleton.v` (line 234 → 279). Verilator and Vivado
  accept this naturally.

## How to run

```sh
bash scripts/run_engine_one_layer_tb.sh
```

The script exits 0 on PASS (max_error=0, mismatch_count=0) and non-zero
on FAIL. The full vvp log is captured in
`build_engine_one_layer_tb/sim.log`.

## Result (latest run)

- Cycles taken (engine_start → engine_done): **453 157**
- Verdict: **FAIL** — 2-byte off-by-one
- max_error: **1**
- mismatch_count: **2** (out of 50 176 total output bytes; bit-exactness rate 99.9960%)
- First mismatch:
  - `byte[4732]` = pixel `[1, 4]` (linear index 18 of 196), channel 124:
    expected `0x01` (+1), got `0x00` (0)
  - `byte[4846]` = pixel `[1, 4]` channel 238: expected `0x00` (0), got
    `0xFF` (−1)

Both errors are off-by-one in INT8 and live in the same output pixel,
strongly suggesting a sign-aware-rounding edge case on the requantisation
boundary (the `(scaled + sign_bias) >>> SCALE_SHIFT` step in
`output/rtl/engine/requant_pipeline.v`) for two specific channels. The
other 50 174 bytes match exactly, including the rest of pixel `[1, 4]`,
so the dispatch program / AXI config / weight URAM / bias BRAM / address
generator / accumulator chain are byte-exact against the Python golden.

Per the task hard rule ("Do NOT modify the engine RTL to make it pass.
The point is to detect bugs, not paper over them"), the TB + comparator
+ run script + this report are left in place reporting the FAIL.

Artifact left on disk (re-run-stable):

- `output/engine_tb_observed.hex` (196 lines, 512 hex chars/line; one
  2048-bit BRAM word per line; MSB-first hex, so byte 0 = channel 0 sits
  in the last two hex characters of each line)
- `build_engine_one_layer_tb/sim.log` (full vvp transcript, including
  heartbeats every 50 000 cycles)
