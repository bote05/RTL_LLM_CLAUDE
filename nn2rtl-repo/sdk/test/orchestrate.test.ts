import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { afterEach, describe, expect, it } from "vitest";
import { z } from "zod";

import {
  buildDelegationPrompt,
  createOrchestratorRuntime,
  findLayer,
  handlePipelineError,
  loadPluginAgentDefinition,
  parseCliArgs,
  parseFrontmatter,
  preflightVerilogModule,
  readJsonFile,
  requireStructuredOutput,
  toStringList,
  writeJsonFile,
} from "../orchestrate.js";
import { layerIrSchema } from "../schemas.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../..");
const tempDirs: string[] = [];

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
});

describe("orchestrate helpers", () => {
  it("parses YAML-style frontmatter blocks", () => {
    const parsed = parseFrontmatter("---\nmodel: sonnet\ntools: Agent\n---\nHello");
    expect(parsed.frontmatter).toEqual({ model: "sonnet", tools: "Agent" });
    expect(parsed.body).toBe("Hello");
  });

  it("rejects markdown without frontmatter", () => {
    expect(() => parseFrontmatter("Hello")).toThrow("Expected agent markdown");
  });

  it("normalizes frontmatter list values from CSV strings and YAML arrays", () => {
    expect(toStringList("a, b, , c")).toEqual(["a", "b", "c"]);
    expect(toStringList(["a", "b", "c"])).toEqual(["a", "b", "c"]);
    expect(toStringList(undefined)).toBeUndefined();
    expect(toStringList(null)).toBeUndefined();
  });

  it("parses YAML lists in frontmatter", () => {
    const parsed = parseFrontmatter("---\nmodel: sonnet\ntools:\n  - Bash\n  - Read\n---\nBody");
    expect(parsed.frontmatter).toEqual({ model: "sonnet", tools: ["Bash", "Read"] });
    expect(parsed.body).toBe("Body");
  });

  it("builds delegation prompts with embedded JSON", () => {
    const prompt = buildDelegationPrompt("foundry", { module_id: "m1" });
    expect(prompt).toContain("Invoke the `foundry` subagent immediately.");
    expect(prompt).toContain('"module_id": "m1"');
  });

  it("prefers structured_output and falls back to result JSON", () => {
    const helperSchema = z.object({
      module_id: z.string(),
      status: z.literal("pass"),
    });

    expect(
      requireStructuredOutput(
        {
          type: "result",
          subtype: "success",
          total_cost_usd: 0,
          modelUsage: {},
          result: "{}",
          structured_output: { module_id: "m1", status: "pass" },
        },
        "assayer",
        helperSchema,
      ),
    ).toEqual({ module_id: "m1", status: "pass" });

    expect(
      requireStructuredOutput(
        {
          type: "result",
          subtype: "success",
          total_cost_usd: 0,
          modelUsage: {},
          result: JSON.stringify({ module_id: "m2" }),
        },
        "foundry",
        layerIrSchema.pick({ module_id: true }),
      ),
    ).toEqual({ module_id: "m2" });
  });

  it("rejects invalid structured outputs", () => {
    expect(() =>
      requireStructuredOutput(
        {
          type: "result",
          subtype: "success",
          total_cost_usd: 0,
          modelUsage: {},
          result: JSON.stringify({ nope: true }),
        },
        "cartographer",
        layerIrSchema.pick({ module_id: true }),
      ),
    ).toThrow("returned invalid output");
  });

  it("reads and writes validated JSON artifacts", async () => {
    const tempDir = await mkdtemp(path.join(os.tmpdir(), "nn2rtl-sdk-orchestrate-"));
    tempDirs.push(tempDir);
    const filePath = path.join(tempDir, "artifact.json");
    const layer = {
      module_id: "m1",
      op_type: "conv2d",
      input_shape: [1],
      output_shape: [1],
      weights_path: "/tmp/w.hex",
      bias_path: "/tmp/b.hex",
      weight_shape: [1],
      num_weights: 1,
      scale_factor: 0.5,
      zero_point: 0,
      pipeline_latency_cycles: 1,
      clock_period_ns: 20,
      input_width_bits: 8,
      output_width_bits: 8,
      clock_signal: "clk",
      reset_signal: "rst_n",
      valid_in_signal: "valid_in",
      valid_out_signal: "valid_out",
      ready_in_signal: "ready_in",
      data_in_signal: "data_in",
      data_out_signal: "data_out",
      golden_inputs_path: "/tmp/unit_module.goldin",
      golden_outputs_path: "/tmp/unit_module.goldout",
    };

    await writeJsonFile(filePath, layer);
    await expect(readJsonFile(filePath, layerIrSchema)).resolves.toEqual(layer);
  });

  it("finds layers by module id", async () => {
    const pipelineIr = await readJsonFile(
      path.join(repoRoot, "test", "fixtures", "pipeline_ir.json"),
    );
    expect(findLayer(pipelineIr as never, "unit_module").module_id).toBe("unit_module");
    expect(() => findLayer(pipelineIr as never, "missing")).toThrow("was not found");
  });

  it("parses CLI arguments including resume", () => {
    expect(parseCliArgs(["checkpoint.pth", "--resume"])).toEqual({
      checkpointPath: "checkpoint.pth",
      resume: true,
    });
    expect(() => parseCliArgs([])).toThrow("Usage:");
  });

  it("loads plugin agent definitions with merged MCP tools and skills", async () => {
    const foundry = await loadPluginAgentDefinition("foundry");
    expect(foundry.tools).toContain("mcp__nn2rtl-tools__write_verilog");
    expect(foundry.prompt).toContain("Supplemental skill reference");
    expect(foundry.model).toBe("sonnet");
  });

  it("creates overridable runtime defaults", () => {
    const fixed = new Date("2026-04-14T00:00:00Z");
    const runtime = createOrchestratorRuntime({
      now: () => fixed,
    });
    expect(runtime.now()).toBe(fixed);
  });

  it("flags canonical port-direction mismatches during deterministic preflight", () => {
    const issues = preflightVerilogModule(
      {
        module_id: "m1",
        spec_hash: "hash",
        generated_by: "Foundry",
        attempt: 1,
        verilog_source: [
          "module m1(",
          "  input wire clk,",
          "  input wire rst_n,",
          "  input wire valid_in,",
          "  input wire ready_in,",
          "  input wire [511:0] data_in,",
          "  output reg valid_out,",
          "  output reg [511:0] data_out",
          ");",
          "endmodule",
        ].join("\n"),
      },
      {
        module_id: "m1",
        op_type: "conv2d",
        input_shape: [1, 64, 1, 1],
        output_shape: [1, 64, 1, 1],
        weights_path: "/tmp/w.hex",
        bias_path: "/tmp/b.hex",
        weight_shape: [64, 64, 1, 1],
        num_weights: 4096,
        scale_factor: 0.5,
        zero_point: 0,
        pipeline_latency_cycles: 67,
        clock_period_ns: 20,
        input_width_bits: 512,
        output_width_bits: 512,
        clock_signal: "clk",
        reset_signal: "rst_n",
        valid_in_signal: "valid_in",
        valid_out_signal: "valid_out",
        ready_in_signal: "ready_in",
        data_in_signal: "data_in",
        data_out_signal: "data_out",
        golden_inputs_path: "/tmp/in.goldin",
        golden_outputs_path: "/tmp/out.goldout",
      },
    );

    expect(issues).toContain(
      "Top-level port 'ready_in' must be declared as output, found input in 'input wire ready_in'.",
    );
  });

  it("fails fast when LayerIR bus widths do not match the channel contract", async () => {
    const runtime = createOrchestratorRuntime();

    await expect(
      runtime.assayerFn(
        {
          module_id: "m1",
          spec_hash: "hash",
          verilog_source: "module m1; endmodule",
          generated_by: "Foundry",
          attempt: 1,
        },
        {
          module_id: "m1",
          op_type: "conv2d",
          input_shape: [1, 64, 1, 1],
          output_shape: [1, 64, 1, 1],
          weights_path: "/tmp/w.hex",
          bias_path: "/tmp/b.hex",
          weight_shape: [64, 64, 1, 1],
          num_weights: 4096,
          scale_factor: 0.5,
          zero_point: 0,
          pipeline_latency_cycles: 67,
          clock_period_ns: 20,
          input_width_bits: 8,
          output_width_bits: 512,
          clock_signal: "clk",
          reset_signal: "rst_n",
          valid_in_signal: "valid_in",
          valid_out_signal: "valid_out",
          ready_in_signal: "ready_in",
          data_in_signal: "data_in",
          data_out_signal: "data_out",
          golden_inputs_path: "/tmp/in.goldin",
          golden_outputs_path: "/tmp/out.goldout",
        },
      ),
    ).rejects.toThrow("input_width_bits");
  });

  it("returns a deterministic preflight failure before invoking simulators", async () => {
    const runtime = createOrchestratorRuntime();
    const result = await runtime.assayerFn(
      {
        module_id: "m1",
        spec_hash: "hash",
        generated_by: "Foundry",
        attempt: 1,
        verilog_source: [
          "module m1(",
          "  input wire clk,",
          "  input wire rst_n,",
          "  input wire valid_in,",
          "  input wire ready_in,",
          "  input wire [511:0] data_in,",
          "  output reg valid_out,",
          "  output reg [511:0] data_out",
          ");",
          "endmodule",
        ].join("\n"),
      },
      {
        module_id: "m1",
        op_type: "conv2d",
        input_shape: [1, 64, 1, 1],
        output_shape: [1, 64, 1, 1],
        weights_path: "/tmp/w.hex",
        bias_path: "/tmp/b.hex",
        weight_shape: [64, 64, 1, 1],
        num_weights: 4096,
        scale_factor: 0.5,
        zero_point: 0,
        pipeline_latency_cycles: 67,
        clock_period_ns: 20,
        input_width_bits: 512,
        output_width_bits: 512,
        clock_signal: "clk",
        reset_signal: "rst_n",
        valid_in_signal: "valid_in",
        valid_out_signal: "valid_out",
        ready_in_signal: "ready_in",
        data_in_signal: "data_in",
        data_out_signal: "data_out",
        golden_inputs_path: "/tmp/in.goldin",
        golden_outputs_path: "/tmp/out.goldout",
      },
    );

    expect(result.status).toBe("fail");
    expect(result.failure_class).toBe("port_width_mismatch");
    expect(result.fix_hint).toContain("Deterministic preflight rejected the RTL");
    expect(result.fix_hint).toContain("ready_in");
  });

  it("records fatal pipeline errors to the run log", async () => {
    await handlePipelineError(new Error("boom"), {
      now: () => new Date("2026-04-14T00:00:00Z"),
    });

    const runLogPath = path.join(repoRoot, "output", "reports", "run_log.jsonl");
    const log = await readFile(runLogPath, "utf8");
    expect(log).toContain('"event":"pipeline_error"');
    expect(log).toContain('"error":"boom"');
  });
});
