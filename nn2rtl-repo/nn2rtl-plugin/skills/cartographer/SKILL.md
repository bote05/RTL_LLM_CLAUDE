---
name: cartographer
description: torch.fx extraction reference for nn2rtl, including batchnorm folding, weight hex emission, timing metadata, and README-aligned PipelineIR formatting.
---
# Cartographer Skill

Use this skill when extracting a quantized ResNet-50 residual block stack into `PipelineIR`.

## Extraction Rules

- Trace the quantized checkpoint with `torch.fx`
- Fold batch normalization into the preceding convolution parameters
- Emit only runtime ops: `conv2d`, `relu`, and `add`
- Write weight and bias tensors to `output/weights/*.hex`
- Never inline raw weights in JSON

## Required `LayerIR` Fields

- `weights_path`
- `bias_path`
- `weight_shape`
- `num_weights`
- `scale_factor`
- `zero_point`
- `pipeline_latency_cycles`
- `clock_period_ns`
- `input_width_bits`
- `output_width_bits`
- `clock_signal` — must equal the literal `"clk"`
- `reset_signal` — must equal the literal `"rst_n"`
- `valid_in_signal` — must equal the literal `"valid_in"`
- `valid_out_signal` — must equal the literal `"valid_out"`
- `ready_in_signal` — must equal the literal `"ready_in"`
- `data_in_signal` — must equal the literal `"data_in"`
- `data_out_signal` — must equal the literal `"data_out"`
- `golden_inputs_path` — absolute POSIX path to the binary `.goldin` file under `output/goldens/`
- `golden_outputs_path` — absolute POSIX path to the binary `.goldout` file under `output/goldens/`
- `lhs_scale_factor`, `rhs_scale_factor` — only populated for `op_type: "add"` modules

## Hex File Format

- `$readmemh` compatible
- one value per line
- uppercase hex
- emitted under `output/weights/`
