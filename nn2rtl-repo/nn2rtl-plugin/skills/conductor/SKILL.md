---
name: conductor
description: Full pipeline state machine reference for nn2rtl, including transition rules, retry limits, JSON schemas, and agent invocation patterns.
---
# Conductor Skill

Use this skill when coordinating the full NN-to-RTL pipeline.

## State Table

| Current state | Trigger | Next state | Action |
| --- | --- | --- | --- |
| `pending` | scheduler tick | `generating` | invoke Foundry |
| `generating` | generation completed | `verifying` | invoke Assayer |
| `verifying` | verification passed | `pass` | store final result |
| `verifying` | verification failed and retries remain | `fail_retry` | prepare Surgeon retry |
| `verifying` | verification failed and retries exhausted | `fail_abort` | stop retrying this module |
| `fail_retry` | scheduler tick | `generating` | invoke Surgeon |

## Retry Policy

- Maximum Surgeon retries per module: `3`
- A module in `fail_abort` is terminal and must not block the rest of the pipeline.
- A module in `pass` is terminal and must not be revisited.

## Agent Invocation Patterns

- Cartographer
  - Runs once at the beginning when `output/layer_ir.json` is missing.
  - Input: checkpoint path and quantization config as JSON in the prompt string.
  - Output: `PipelineIR`
- Foundry
  - Runs for a single `LayerIR`
  - Input: one `LayerIR` JSON object in the prompt string
  - Output: `VerilogModule`
- Assayer
  - Runs after every Foundry or Surgeon generation
  - Input: candidate module payload plus golden vectors in the prompt string
  - Output: `VerifResult`
- Surgeon
  - Runs only from `fail_retry`
  - Input: broken `VerilogModule`, `VerifResult`, and original `LayerIR`
  - Output: repaired `VerilogModule`

## PipelineState JSON Schema

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "run_id",
    "started_at",
    "modules",
    "attempts",
    "results",
    "max_retries"
  ],
  "properties": {
    "run_id": { "type": "string" },
    "started_at": { "type": "string" },
    "modules": {
      "type": "object",
      "additionalProperties": {
        "type": "string",
        "enum": [
          "pending",
          "generating",
          "verifying",
          "pass",
          "fail_retry",
          "fail_abort"
        ]
      }
    },
    "attempts": {
      "type": "object",
      "additionalProperties": { "type": "integer", "minimum": 0 }
    },
    "results": {
      "type": "object",
      "additionalProperties": { "$ref": "#/$defs/VerifResult" }
    },
    "max_retries": { "type": "integer", "minimum": 0 }
  },
  "$defs": {
    "VerifResult": {
      "type": "object",
      "additionalProperties": false,
      "required": ["module_id", "status"],
      "properties": {
        "module_id": { "type": "string" },
        "status": { "type": "string", "enum": ["pass", "fail", "syntax_error"] },
        "mismatch_layer": { "type": "string" },
        "expected": {
          "type": "array",
          "items": { "type": "number" }
        },
        "got": {
          "type": "array",
          "items": { "type": "number" }
        },
        "max_error": { "type": "number" },
        "mean_error": { "type": "number" },
        "fix_hint": { "type": "string" },
        "iverilog_stderr": { "type": "string" },
        "verilator_stderr": { "type": "string" }
      }
    }
  }
}
```

## LayerIR JSON Schema

```json
{
  "type": "object",
  "additionalProperties": false,
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

## VerilogModule JSON Schema

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "module_id",
    "spec_hash",
    "verilog_source",
    "generated_by",
    "attempt"
  ],
  "properties": {
    "module_id": { "type": "string" },
    "spec_hash": { "type": "string" },
    "verilog_source": { "type": "string" },
    "generated_by": {
      "type": "string",
      "enum": ["Foundry", "Surgeon"]
    },
    "attempt": { "type": "integer", "minimum": 0 }
  }
}
```

## VerifResult JSON Schema

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["module_id", "status"],
  "properties": {
    "module_id": { "type": "string" },
    "status": { "type": "string", "enum": ["pass", "fail", "syntax_error"] },
    "mismatch_layer": { "type": "string" },
    "expected": {
      "type": "array",
      "items": { "type": "number" }
    },
    "got": {
      "type": "array",
      "items": { "type": "number" }
    },
    "max_error": { "type": "number" },
    "mean_error": { "type": "number" },
    "fix_hint": { "type": "string" },
    "iverilog_stderr": { "type": "string" },
    "verilator_stderr": { "type": "string" }
  }
}
```
