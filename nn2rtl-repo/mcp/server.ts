import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  type CallToolResult,
} from "@modelcontextprotocol/sdk/types.js";

import {
  read_weights,
  run_iverilog,
  run_verilator,
  run_yosys,
  write_verilog,
} from "./tools.js";
import type { FailureClass, VerilogModule } from "./types.js";

const server = new Server(
  {
    name: "nn2rtl-tools",
    version: "0.1.0",
  },
  {
    capabilities: {
      tools: {},
    },
  },
);

const FAILURE_CLASSES = [
  "integer_overflow",
  "sign_extension_error",
  "bit_shift_wrong",
  "rounding_mode_wrong",
  "saturation_missing",
  "loop_bounds_incorrect",
  "array_indexing_error",
  "port_width_mismatch",
  "residual_addition_overflow",
  "missing_pipeline_register",
  "pipeline_latency_wrong",
  "reset_logic_broken",
  "enable_signal_ignored",
  "scale_factor_misapplied",
  "bias_term_missing",
  "batch_norm_not_folded",
] as const satisfies readonly FailureClass[];

function toToolResult(payload: Record<string, unknown>): CallToolResult {
  return {
    content: [{ type: "text", text: JSON.stringify(payload, null, 2) }],
    structuredContent: payload,
  };
}

const layerIrSchema = {
  type: "object",
  additionalProperties: false,
  required: [
    "module_id",
    "op_type",
    "input_shape",
    "output_shape",
    "weights_path",
    "bias_path",
    "weight_shape",
    "num_weights",
    "scale_factor",
    "zero_point",
    "pipeline_latency_cycles",
    "clock_period_ns",
    "input_width_bits",
    "output_width_bits",
    "valid_in_signal",
    "valid_out_signal",
    "clock_signal",
    "reset_signal",
    "golden_inputs",
    "golden_outputs",
  ],
  properties: {
    module_id: { type: "string" },
    op_type: { type: "string", enum: ["conv2d", "relu", "add"] },
    input_shape: { type: "array", items: { type: "number" } },
    output_shape: { type: "array", items: { type: "number" } },
    weights_path: { type: "string" },
    bias_path: {
      anyOf: [
        { type: "string" },
        { type: "null" },
      ],
    },
    weight_shape: { type: "array", items: { type: "number" } },
    num_weights: { type: "number" },
    scale_factor: { type: "number" },
    zero_point: { type: "number" },
    pipeline_latency_cycles: { type: "number" },
    clock_period_ns: { type: "number" },
    input_width_bits: { type: "number" },
    output_width_bits: { type: "number" },
    valid_in_signal: { type: "string" },
    valid_out_signal: { type: "string" },
    clock_signal: { type: "string" },
    reset_signal: { type: "string" },
    golden_inputs: {
      type: "array",
      items: {
        type: "array",
        items: { type: "number" },
      },
    },
    golden_outputs: {
      type: "array",
      items: {
        type: "array",
        items: { type: "number" },
      },
    },
  },
} as const;

const verilogModuleSchema = {
  type: "object",
  additionalProperties: false,
  required: [
    "module_id",
    "spec_hash",
    "verilog_source",
    "generated_by",
    "attempt",
  ],
  properties: {
    module_id: { type: "string" },
    spec_hash: { type: "string" },
    verilog_source: { type: "string" },
    generated_by: { type: "string", enum: ["Foundry", "Surgeon"] },
    attempt: { type: "number" },
  },
} as const;

const verifResultSchema = {
  type: "object",
  additionalProperties: false,
  required: ["module_id", "status"],
  properties: {
    module_id: { type: "string" },
    status: { type: "string", enum: ["pass", "fail", "syntax_error"] },
    timing_pass: { type: "boolean" },
    timing_actual_cycles: { type: "number" },
    timing_expected_cycles: { type: "number" },
    mismatch_layer: { type: "string" },
    expected: { type: "array", items: { type: "number" } },
    got: { type: "array", items: { type: "number" } },
    max_error: { type: "number" },
    mean_error: { type: "number" },
    failure_class: {
      anyOf: [
        { type: "string", enum: [...FAILURE_CLASSES] },
        { type: "null" },
      ],
    },
    fix_hint: { type: "string" },
    iverilog_stderr: { type: "string" },
    verilator_stderr: { type: "string" },
  },
} as const;

const pipelineIrSchema = {
  type: "object",
  additionalProperties: false,
  required: ["model_name", "quantization", "generated_at", "layers"],
  properties: {
    model_name: { type: "string" },
    quantization: { type: "string", const: "int8_symmetric_per_tensor" },
    generated_at: { type: "string" },
    layers: {
      type: "array",
      items: layerIrSchema,
    },
  },
} as const;

const toolDefinitions = [
  {
    name: "run_iverilog",
    description: "Run iverilog syntax checking for a candidate Verilog module.",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      required: ["verilog_source", "module_name"],
      properties: {
        verilog_source: { type: "string" },
        module_name: { type: "string" },
      },
    },
    outputSchema: {
      type: "object",
      additionalProperties: false,
      required: ["success", "stderr"],
      properties: {
        success: { type: "boolean" },
        stderr: { type: "string" },
      },
    },
  },
  {
    name: "run_verilator",
    description: "Run Verilator lint and simulation against golden vectors for a candidate module.",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      required: ["verilog_source", "module_name", "sidecar_path"],
      properties: {
        verilog_source: { type: "string" },
        module_name: { type: "string" },
        sidecar_path: { type: "string" },
      },
    },
    outputSchema: verifResultSchema,
  },
  {
    name: "run_yosys",
    description: "Run Yosys synthesis reporting for a candidate Verilog module.",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      required: ["verilog_source", "module_name"],
      properties: {
        verilog_source: { type: "string" },
        module_name: { type: "string" },
      },
    },
    outputSchema: {
      type: "object",
      additionalProperties: false,
      required: ["success", "lut_count", "fmax_mhz", "report"],
      properties: {
        success: { type: "boolean" },
        lut_count: { type: "number" },
        fmax_mhz: { type: "number" },
        report: { type: "string" },
      },
    },
  },
  {
    name: "read_weights",
    description: "Read a quantized checkpoint and return PipelineIR.",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      required: ["checkpoint_path", "quantization_config"],
      properties: {
        checkpoint_path: { type: "string" },
        quantization_config: {
          type: "object",
          additionalProperties: true,
        },
      },
    },
    outputSchema: pipelineIrSchema,
  },
  {
    name: "write_verilog",
    description: "Persist a generated Verilog module and its metadata.",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      required: ["module", "output_dir"],
      properties: {
        module: verilogModuleSchema,
        output_dir: { type: "string" },
      },
    },
    outputSchema: {
      type: "object",
      additionalProperties: false,
      required: ["path"],
      properties: {
        path: { type: "string" },
      },
    },
  },
] as const;

async function handleRunIverilog(args: Record<string, unknown>): Promise<CallToolResult> {
  // TODO: Replace these manual casts with shared runtime validation so malformed MCP arguments fail with field-level error messages.
  const result = await run_iverilog(
    String(args.verilog_source),
    String(args.module_name),
  );
  return toToolResult(result);
}

async function handleRunVerilator(args: Record<string, unknown>): Promise<CallToolResult> {
  // TODO: Replace these manual casts with shared runtime validation so malformed MCP arguments fail with field-level error messages.
  const result = await run_verilator(
    String(args.verilog_source),
    String(args.module_name),
    String(args.sidecar_path),
  );
  return toToolResult(result as unknown as Record<string, unknown>);
}

async function handleRunYosys(args: Record<string, unknown>): Promise<CallToolResult> {
  // TODO: Replace these manual casts with shared runtime validation so malformed MCP arguments fail with field-level error messages.
  const result = await run_yosys(
    String(args.verilog_source),
    String(args.module_name),
  );
  return toToolResult(result);
}

async function handleReadWeights(args: Record<string, unknown>): Promise<CallToolResult> {
  // TODO: Replace these manual casts with shared runtime validation so malformed MCP arguments fail with field-level error messages.
  const result = await read_weights(
    String(args.checkpoint_path),
    (args.quantization_config ?? {}) as object,
  );
  return toToolResult(result as unknown as Record<string, unknown>);
}

async function handleWriteVerilog(args: Record<string, unknown>): Promise<CallToolResult> {
  // TODO: Replace these manual casts with shared runtime validation so malformed MCP arguments fail with field-level error messages.
  const writtenPath = await write_verilog(
    args.module as VerilogModule,
    String(args.output_dir),
  );
  return toToolResult({ path: writtenPath });
}

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [...toolDefinitions],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const name = request.params.name;
  const args = (request.params.arguments ?? {}) as Record<string, unknown>;

  switch (name) {
    case "run_iverilog":
      return handleRunIverilog(args);
    case "run_verilator":
      return handleRunVerilator(args);
    case "run_yosys":
      return handleRunYosys(args);
    case "read_weights":
      return handleReadWeights(args);
    case "write_verilog":
      return handleWriteVerilog(args);
    default:
      return {
        content: [{ type: "text", text: `Unknown tool '${name}'.` }],
        isError: true,
      };
  }
});

async function main(): Promise<void> {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
