import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { afterEach, describe, expect, it } from "vitest";
import { z } from "zod";

import {
  buildDelegationPrompt,
  checkBusWidthCapability,
  createOrchestratorRuntime,
  findLayer,
  handlePipelineError,
  loadPluginAgentDefinition,
  parseCliArgs,
  parseFrontmatter,
  preflightVerilogModule,
  readJsonFile,
  requireStructuredOutput,
  structuralPreflightViolations,
  toStringList,
  writeJsonFile,
} from "../orchestrate.js";
import { layerIrBaseSchema, layerIrSchema } from "../schemas.js";

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
    expect(prompt).toContain("You are the `foundry` agent.");
    expect(prompt).toContain('"module_id": "m1"');
  });

  it("adds a compact generation brief for Foundry", () => {
    const prompt = buildDelegationPrompt("foundry", {
      layer_ir: {
        module_id: "m1",
        op_type: "conv2d",
        input_shape: [1, 3, 8, 8],
        output_shape: [1, 4, 8, 8],
        weights_path: "/tmp/w.hex",
        bias_path: "/tmp/b.hex",
        weight_shape: [4, 3, 3, 3],
        num_weights: 108,
        scale_factor: 0.5,
        zero_point: 0,
        pipeline_latency_cycles: 40,
        clock_period_ns: 20,
        input_width_bits: 24,
        output_width_bits: 32,
        clock_signal: "clk",
        reset_signal: "rst_n",
        valid_in_signal: "valid_in",
        valid_out_signal: "valid_out",
        ready_in_signal: "ready_in",
        data_in_signal: "data_in",
        data_out_signal: "data_out",
        golden_inputs_path: "/tmp/in.goldin",
        golden_outputs_path: "/tmp/out.goldout",
        stride: [1, 1],
        padding: [1, 1],
      },
      expected_spec_hash: "conv2d_3x4x3x3_s8x8_st1x1_p1x1_i24_o32",
    });

    expect(prompt).toContain("Compact generation brief:");
    expect(prompt).toContain("return spec_hash=conv2d_3x4x3x3_s8x8_st1x1_p1x1_i24_o32 exactly");
    expect(prompt).toContain("spatial conv rule");
  });

  it("adds a compact repair brief for Surgeon", () => {
    const prompt = buildDelegationPrompt("surgeon", {
      layer_ir: {
        module_id: "m1",
        op_type: "add",
        input_shape: [1, 4, 8, 8],
        output_shape: [1, 4, 8, 8],
        weights_path: "/tmp/w.hex",
        bias_path: null,
        weight_shape: [1],
        num_weights: 0,
        scale_factor: 0.5,
        lhs_scale_factor: 0.25,
        rhs_scale_factor: 0.25,
        zero_point: 0,
        pipeline_latency_cycles: 1,
        clock_period_ns: 20,
        input_width_bits: 64,
        output_width_bits: 32,
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
      verif_result: {
        module_id: "m1",
        status: "syntax_error",
        iverilog_stderr: "m1.v:42: syntax error",
      },
    });

    expect(prompt).toContain("Compact repair brief:");
    expect(prompt).toContain("compiler-first rule");
    expect(prompt).toContain("invariant scope");
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
        layerIrBaseSchema.pick({ module_id: true }),
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
        layerIrBaseSchema.pick({ module_id: true }),
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
      maxRetries: undefined,
      only: undefined,
      except: [],
    });
    expect(() => parseCliArgs([])).toThrow("Usage:");
  });

  it("loads plugin agent definitions with merged MCP tools and skills", async () => {
    const foundry = await loadPluginAgentDefinition("foundry");
    const cartographer = await loadPluginAgentDefinition("cartographer");
    expect(foundry.tools).toContain("Bash");
    expect(foundry.tools).toContain("mcp__nn2rtl-tools__write_verilog");
    expect(foundry.tools).toContain("mcp__nn2rtl-tools__get_rtl_patterns");
    expect(foundry.prompt).toContain("Knowledge catalog");
    expect(foundry.prompt).toContain("knowledge/patterns/02_conv1x1.md");
    expect(foundry.prompt).toContain("rtl_library/coord_scheduler.v");
    expect(foundry.prompt).not.toContain("Supplemental skill reference");
    expect(cartographer.prompt).toContain("Supplemental skill reference");
    expect(foundry.model).toBe("claude-opus-4-7");
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

  it("ignores commas inside inline // comments when parsing the port list", () => {
    // Regression: Surgeon-emitted RTL sometimes annotates data_in with an
    // inline comment that contains commas (e.g. "// [7:0]=ch0, [15:8]=ch1").
    // Previously splitTopLevelCommaList split on those commas before comments
    // were stripped, fragmenting the port block and making later ports
    // inherit the direction/width of their predecessor. Strip comments first.
    const issues = preflightVerilogModule(
      {
        module_id: "m1",
        spec_hash: "hash",
        generated_by: "Surgeon",
        attempt: 2,
        verilog_source: [
          "module m1(",
          "  input  wire         clk,",
          "  input  wire         rst_n,",
          "  input  wire         valid_in,",
          "  output wire         ready_in,",
          "  input  wire [23:0]  data_in,    // 3 channels x 8b: [23:16]=ch2, [15:8]=ch1, [7:0]=ch0",
          "  output reg          valid_out,",
          "  output reg  [511:0] data_out",
          ");",
          "endmodule",
        ].join("\n"),
      },
      {
        module_id: "m1",
        op_type: "conv2d",
        input_shape: [1, 3, 224, 224],
        output_shape: [1, 64, 112, 112],
        weights_path: "/tmp/w.hex",
        bias_path: "/tmp/b.hex",
        weight_shape: [64, 3, 7, 7],
        num_weights: 9408,
        scale_factor: 0.003,
        zero_point: 0,
        pipeline_latency_cycles: 826,
        clock_period_ns: 20,
        input_width_bits: 24,
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

    expect(issues).toEqual([]);
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

  it("reports structural preflight violations for a spatial conv missing line_buf / window / readmemh / output counter", () => {
    const violations = structuralPreflightViolations(
      {
        module_id: "m1",
        spec_hash: "hash",
        generated_by: "Foundry",
        attempt: 1,
        // Minimal RTL that passes ANSI port parsing but misses every
        // structural rule for a 3x3 conv.
        verilog_source: [
          "module m1(",
          "  input wire clk, input wire rst_n,",
          "  input wire valid_in, output reg ready_in,",
          "  input wire [7:0] data_in,",
          "  output reg valid_out, output reg [7:0] data_out",
          ");",
          "endmodule",
        ].join("\n"),
      },
      {
        module_id: "m1",
        op_type: "conv2d",
        input_shape: [1, 1, 8, 8],
        output_shape: [1, 1, 8, 8],
        weights_path: "/tmp/w.hex",
        bias_path: "/tmp/b.hex",
        weight_shape: [1, 1, 3, 3],
        num_weights: 9,
        scale_factor: 0.5,
        zero_point: 0,
        pipeline_latency_cycles: 100,
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
        golden_inputs_path: "/tmp/in.goldin",
        golden_outputs_path: "/tmp/out.goldout",
      },
    );
    const rules = new Set(violations.map((v) => v.rule));
    expect(rules.has("line_buffer_missing")).toBe(true);
    expect(rules.has("window_not_registered")).toBe(true);
    expect(rules.has("readmemh_missing")).toBe(true);
    expect(rules.has("output_counter_missing")).toBe(true);
  });

  it("flags weights_packed as a structural violation", () => {
    const violations = structuralPreflightViolations(
      {
        module_id: "m1",
        spec_hash: "hash",
        generated_by: "Surgeon",
        attempt: 2,
        verilog_source: [
          "module m1();",
          "  reg [63:0] weights_packed [0:15];",
          "  reg signed [7:0] weights [0:63];",
          "  initial $readmemh(\"w.hex\", weights);",
          "  reg [31:0] outputs_emitted;",
          "  reg [7:0] line_buf [0:31][0:15];",
          "  reg [7:0] window [0:8];",
          "  always @(posedge clk) begin window[0] <= 8'h00; end",
          "endmodule",
        ].join("\n"),
      },
      {
        module_id: "m1",
        op_type: "conv2d",
        input_shape: [1, 1, 8, 8],
        output_shape: [1, 1, 8, 8],
        weights_path: "/tmp/w.hex",
        bias_path: null,
        weight_shape: [1, 1, 3, 3],
        num_weights: 9,
        scale_factor: 0.5,
        zero_point: 0,
        pipeline_latency_cycles: 100,
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
        golden_inputs_path: "/tmp/in.goldin",
        golden_outputs_path: "/tmp/out.goldout",
      },
    );
    const rules = violations.map((v) => v.rule);
    expect(rules).toContain("weights_packed_forbidden");
  });

  it("passes structural preflight for a pointwise (1x1) conv without needing line_buf / window / output counter", () => {
    // Regression: the output_counter_missing rule must NOT fire on pointwise
    // conv2d. Pointwise is 1:1 pixel-in-to-pixel-out; adding a frame-level
    // `outputs_emitted` counter into a pointwise FSM latches it into a
    // terminal state after the first frame and breaks back-to-back frames.
    const violations = structuralPreflightViolations(
      {
        module_id: "m1",
        spec_hash: "hash",
        generated_by: "Foundry",
        attempt: 1,
        verilog_source: [
          "module m1();",
          "  reg signed [7:0] weights [0:63];",
          "  reg signed [31:0] biases [0:0];",
          "  initial begin",
          "    $readmemh(\"w.hex\", weights);",
          "    $readmemh(\"b.hex\", biases);",
          "  end",
          "endmodule",
        ].join("\n"),
      },
      {
        module_id: "m1",
        op_type: "conv2d",
        input_shape: [1, 64, 1, 1],
        output_shape: [1, 1, 1, 1],
        weights_path: "/tmp/w.hex",
        bias_path: "/tmp/b.hex",
        weight_shape: [1, 64, 1, 1],
        num_weights: 64,
        scale_factor: 0.5,
        zero_point: 0,
        pipeline_latency_cycles: 67,
        clock_period_ns: 20,
        input_width_bits: 512,
        output_width_bits: 8,
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
    expect(violations).toEqual([]);
  });

  it("checkBusWidthCapability rejects layers over MAX_SUPPORTED_BUS_BITS", () => {
    const baseLayer = {
      module_id: "m1",
      op_type: "conv2d" as const,
      input_shape: [1, 1024, 1, 1],
      output_shape: [1, 1024, 1, 1],
      weights_path: "/tmp/w.hex",
      bias_path: "/tmp/b.hex",
      weight_shape: [1024, 1024, 1, 1],
      num_weights: 1048576,
      scale_factor: 0.5,
      zero_point: 0,
      pipeline_latency_cycles: 1,
      clock_period_ns: 20,
      input_width_bits: 8192,
      output_width_bits: 8192,
      clock_signal: "clk" as const,
      reset_signal: "rst_n" as const,
      valid_in_signal: "valid_in" as const,
      valid_out_signal: "valid_out" as const,
      ready_in_signal: "ready_in" as const,
      data_in_signal: "data_in" as const,
      data_out_signal: "data_out" as const,
      golden_inputs_path: "/tmp/in.goldin",
      golden_outputs_path: "/tmp/out.goldout",
    };
    const reason = checkBusWidthCapability(baseLayer);
    expect(reason).toContain("requires tiled channel streaming which is not yet implemented");
    expect(reason).toContain("MAX_SUPPORTED_BUS_BITS");

    const smallLayer = { ...baseLayer, input_width_bits: 512, output_width_bits: 512 };
    expect(checkBusWidthCapability(smallLayer)).toBeNull();
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
