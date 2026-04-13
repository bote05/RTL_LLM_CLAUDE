---
name: cartographer
description: torch.fx extraction reference for nn2rtl, including INT8 symmetric per-tensor quantization assumptions, golden vector generation, and PipelineIR formatting.
---
# Cartographer Skill

Use this skill when extracting a quantized ResNet-50 residual block stack into `PipelineIR`.

## Extraction Approach

- Load the quantized checkpoint into a ResNet-50 compatible module.
- Trace the forward path with `torch.fx`.
- Identify the residual block operations that will be lowered to RTL.
- Emit one `LayerIR` per relevant op:
  - `conv2d`
  - `batchnorm`
  - `relu`
  - `add`

## Quantization Assumptions

- Quantization mode is `int8_symmetric_per_tensor`.
- Store one scalar `scale_factor` per emitted layer.
- Preserve signed INT8 ranges when serializing weights and golden vectors.

## Golden Vector Procedure

1. Use a deterministic input tensor.
2. Capture the activation tensor before and after each target operation.
3. Flatten or serialize activations consistently so downstream Verilator comparisons are stable.
4. Write the resulting `PipelineIR` to `output/layer_ir.json`.

## Exact PipelineIR Format

```json
{
  "model_name": "resnet50",
  "quantization": "int8_symmetric_per_tensor",
  "generated_at": "2026-04-14T00:00:00.000Z",
  "layers": [
    {
      "module_id": "layer1_block0_conv1",
      "op_type": "conv2d",
      "input_shape": [1, 64, 56, 56],
      "output_shape": [1, 64, 56, 56],
      "weight_int8": [[1, -2, 3]],
      "scale_factor": 0.03125,
      "golden_inputs": [[0, 0, 0]],
      "golden_outputs": [[0, 0, 0]]
    }
  ]
}
```

## LayerIR Schema Reminder

```json
{
  "type": "object",
  "required": [
    "module_id",
    "op_type",
    "input_shape",
    "output_shape",
    "weight_int8",
    "scale_factor",
    "golden_inputs",
    "golden_outputs"
  ],
  "properties": {
    "module_id": { "type": "string" },
    "op_type": {
      "type": "string",
      "enum": ["conv2d", "batchnorm", "relu", "add"]
    },
    "input_shape": {
      "type": "array",
      "items": { "type": "integer" }
    },
    "output_shape": {
      "type": "array",
      "items": { "type": "integer" }
    },
    "weight_int8": {
      "type": "array",
      "items": {
        "type": "array",
        "items": { "type": "integer" }
      }
    },
    "scale_factor": { "type": "number" },
    "golden_inputs": {
      "type": "array",
      "items": {
        "type": "array",
        "items": { "type": "number" }
      }
    },
    "golden_outputs": {
      "type": "array",
      "items": {
        "type": "array",
        "items": { "type": "number" }
      }
    }
  }
}
```
