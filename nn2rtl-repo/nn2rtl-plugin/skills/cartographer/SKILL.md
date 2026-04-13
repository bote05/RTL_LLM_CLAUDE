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
- `valid_in_signal`
- `valid_out_signal`
- `clock_signal`
- `reset_signal`
- `golden_inputs`
- `golden_outputs`

## Hex File Format

- `$readmemh` compatible
- one value per line
- uppercase hex
- emitted under `output/weights/`
