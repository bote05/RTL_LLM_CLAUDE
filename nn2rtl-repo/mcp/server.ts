import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  type CallToolResult,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";

import {
  read_weights,
  run_iverilog,
  run_verilator,
  run_yosys,
  write_verilog,
} from "./tools.js";
import {
  pipelineIrSchema,
  readWeightsInput,
  runIverilogInput,
  runIverilogOutput,
  runVerilatorInput,
  runYosysInput,
  runYosysOutput,
  verifResultSchema,
  writeVerilogInput,
  writeVerilogOutput,
} from "./schemas.js";

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

function toToolResult(payload: Record<string, unknown>): CallToolResult {
  return {
    content: [{ type: "text", text: JSON.stringify(payload, null, 2) }],
    structuredContent: payload,
  };
}

function toJsonSchema(schema: z.ZodType): Record<string, unknown> {
  return z.toJSONSchema(schema) as Record<string, unknown>;
}

const toolDefinitions = [
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
    name: "run_yosys",
    description: "Run Yosys synthesis reporting for a candidate Verilog module.",
    inputSchema: toJsonSchema(runYosysInput),
    outputSchema: toJsonSchema(runYosysOutput),
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
] as const;

async function handleRunIverilog(args: Record<string, unknown>): Promise<CallToolResult> {
  const input = runIverilogInput.parse(args);
  const result = await run_iverilog(input.verilog_source, input.module_name);
  return toToolResult(result);
}

async function handleRunVerilator(args: Record<string, unknown>): Promise<CallToolResult> {
  const input = runVerilatorInput.parse(args);
  const result = await run_verilator(
    input.verilog_source,
    input.module_name,
    input.sidecar_path,
  );
  return toToolResult(result as unknown as Record<string, unknown>);
}

async function handleRunYosys(args: Record<string, unknown>): Promise<CallToolResult> {
  const input = runYosysInput.parse(args);
  const result = await run_yosys(input.verilog_source, input.module_name);
  return toToolResult(result);
}

async function handleReadWeights(args: Record<string, unknown>): Promise<CallToolResult> {
  const input = readWeightsInput.parse(args);
  const result = await read_weights(input.checkpoint_path, input.quantization_config);
  return toToolResult(result as unknown as Record<string, unknown>);
}

async function handleWriteVerilog(args: Record<string, unknown>): Promise<CallToolResult> {
  const input = writeVerilogInput.parse(args);
  const writtenPath = await write_verilog(input.module, input.output_dir);
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
