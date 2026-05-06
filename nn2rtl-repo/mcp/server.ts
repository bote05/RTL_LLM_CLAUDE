import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  type CallToolResult,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";

import {
  get_failure_corpus,
  get_rtl_patterns,
  read_weights,
  run_iverilog,
  run_verilator,
  run_vivado,
  write_verilog,
} from "./tools.js";
import {
  getRtlPatternsInput,
  getRtlPatternsOutput,
  getFailureCorpusInput,
  getFailureCorpusOutput,
  pipelineIrSchema,
  readWeightsInput,
  runIverilogInput,
  runIverilogOutput,
  runVerilatorInput,
  runVivadoInput,
  runVivadoOutput,
  verifResultSchema,
  writeVerilogInput,
  writeVerilogOutput,
} from "./schemas.js";

export type ToolImplementations = {
  get_failure_corpus: typeof get_failure_corpus;
  get_rtl_patterns: typeof get_rtl_patterns;
  read_weights: typeof read_weights;
  run_iverilog: typeof run_iverilog;
  run_verilator: typeof run_verilator;
  run_vivado: typeof run_vivado;
  write_verilog: typeof write_verilog;
};

const DEFAULT_TOOL_IMPLEMENTATIONS: ToolImplementations = {
  get_failure_corpus,
  get_rtl_patterns,
  read_weights,
  run_iverilog,
  run_verilator,
  run_vivado,
  write_verilog,
};

function toToolResult(payload: Record<string, unknown>): CallToolResult {
  return {
    content: [{ type: "text", text: JSON.stringify(payload, null, 2) }],
    structuredContent: payload,
  };
}

function toJsonSchema(schema: z.ZodType): Record<string, unknown> {
  return z.toJSONSchema(schema) as Record<string, unknown>;
}

export const toolDefinitions = [
  {
    name: "run_iverilog",
    description: "Run iverilog syntax checking for a candidate Verilog module.",
    inputSchema: toJsonSchema(runIverilogInput),
    outputSchema: toJsonSchema(runIverilogOutput),
  },
  {
    name: "run_verilator",
    description: "Run Verilator lint and simulation against golden vectors for a candidate module.",
    inputSchema: toJsonSchema(runVerilatorInput),
    outputSchema: toJsonSchema(verifResultSchema),
  },
  {
    name: "run_vivado",
    description: "Run Vivado synth-only reporting for a candidate Verilog module.",
    inputSchema: toJsonSchema(runVivadoInput),
    outputSchema: toJsonSchema(runVivadoOutput),
  },
  {
    name: "read_weights",
    description: "Read a quantized checkpoint and return PipelineIR.",
    inputSchema: toJsonSchema(readWeightsInput),
    outputSchema: toJsonSchema(pipelineIrSchema),
  },
  {
    name: "write_verilog",
    description: "Persist a generated Verilog module and its metadata.",
    inputSchema: toJsonSchema(writeVerilogInput),
    outputSchema: toJsonSchema(writeVerilogOutput),
  },
  {
    name: "get_rtl_patterns",
    description:
      "Look up architectural-pattern markdown and an optional proven reference " +
      "Verilog for an op_type (+ kernel dims for conv2d), optionally filtered by contract_id. Call this before " +
      "emitting any Verilog (Foundry) or when diagnosing a synth / sim failure " +
      "(Surgeon). Returns { pattern_markdown, reference_verilog, license_notice }.",
    inputSchema: toJsonSchema(getRtlPatternsInput),
    outputSchema: toJsonSchema(getRtlPatternsOutput),
  },
  {
    name: "get_failure_corpus",
    description:
      "Retrieve visible scored failed RTL attempts from output/failure_corpus/visible. " +
      "Returns summaries plus rtl_path/failure_path; optionally includes Verilog source. Archived failures are intentionally hidden.",
    inputSchema: toJsonSchema(getFailureCorpusInput),
    outputSchema: toJsonSchema(getFailureCorpusOutput),
  },
] as const;

async function handleRunIverilog(
  args: Record<string, unknown>,
  toolImpls: ToolImplementations,
): Promise<CallToolResult> {
  const input = runIverilogInput.parse(args);
  const result = await toolImpls.run_iverilog(input.verilog_source, input.module_name);
  return toToolResult(result);
}

async function handleRunVerilator(
  args: Record<string, unknown>,
  toolImpls: ToolImplementations,
): Promise<CallToolResult> {
  const input = runVerilatorInput.parse(args);
  const result = await toolImpls.run_verilator(
    input.verilog_source,
    input.module_name,
    input.sidecar_path,
  );
  return toToolResult(result as unknown as Record<string, unknown>);
}

async function handleRunVivado(
  args: Record<string, unknown>,
  toolImpls: ToolImplementations,
): Promise<CallToolResult> {
  const input = runVivadoInput.parse(args);
  const result = await toolImpls.run_vivado(
    input.verilog_source,
    input.module_name,
    input.clock_period_ns,
    input.part,
    input.threads,
  );
  return toToolResult(result);
}

async function handleReadWeights(
  args: Record<string, unknown>,
  toolImpls: ToolImplementations,
): Promise<CallToolResult> {
  const input = readWeightsInput.parse(args);
  const result = await toolImpls.read_weights(input.checkpoint_path, input.quantization_config);
  return toToolResult(result as unknown as Record<string, unknown>);
}

async function handleWriteVerilog(
  args: Record<string, unknown>,
  toolImpls: ToolImplementations,
): Promise<CallToolResult> {
  const input = writeVerilogInput.parse(args);
  const writtenPath = await toolImpls.write_verilog(input.module, input.output_dir);
  return toToolResult({ path: writtenPath });
}

async function handleGetRtlPatterns(
  args: Record<string, unknown>,
  toolImpls: ToolImplementations,
): Promise<CallToolResult> {
  const input = getRtlPatternsInput.parse(args);
  const result = await toolImpls.get_rtl_patterns(
    input.op_type,
    input.kernel_h,
    input.kernel_w,
    input.contract_id,
  );
  return toToolResult(result as unknown as Record<string, unknown>);
}

async function handleGetFailureCorpus(
  args: Record<string, unknown>,
  toolImpls: ToolImplementations,
): Promise<CallToolResult> {
  const input = getFailureCorpusInput.parse(args);
  const result = await toolImpls.get_failure_corpus(input);
  return toToolResult(result as unknown as Record<string, unknown>);
}

export async function handleToolCall(
  name: string,
  args: Record<string, unknown>,
  toolImpls: ToolImplementations = DEFAULT_TOOL_IMPLEMENTATIONS,
): Promise<CallToolResult> {
  switch (name) {
    case "run_iverilog":
      return handleRunIverilog(args, toolImpls);
    case "run_verilator":
      return handleRunVerilator(args, toolImpls);
    case "run_vivado":
      return handleRunVivado(args, toolImpls);
    case "read_weights":
      return handleReadWeights(args, toolImpls);
    case "write_verilog":
      return handleWriteVerilog(args, toolImpls);
    case "get_rtl_patterns":
      return handleGetRtlPatterns(args, toolImpls);
    case "get_failure_corpus":
      return handleGetFailureCorpus(args, toolImpls);
    default:
      return {
        content: [{ type: "text", text: `Unknown tool '${name}'.` }],
        isError: true,
      };
  }
}

export function createServer(
  toolImpls: ToolImplementations = DEFAULT_TOOL_IMPLEMENTATIONS,
): Server {
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

  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [...toolDefinitions],
  }));

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const name = request.params.name;
    const args = (request.params.arguments ?? {}) as Record<string, unknown>;
    return handleToolCall(name, args, toolImpls);
  });

  return server;
}

export async function startServer(
  toolImpls: ToolImplementations = DEFAULT_TOOL_IMPLEMENTATIONS,
): Promise<Server> {
  const server = createServer(toolImpls);
  const transport = new StdioServerTransport();
  await server.connect(transport);
  return server;
}
