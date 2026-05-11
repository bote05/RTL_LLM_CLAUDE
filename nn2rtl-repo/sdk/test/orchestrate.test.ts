import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { afterEach, describe, expect, it } from "vitest";
import { z } from "zod";

import {
  buildDelegationPrompt,
  buildFailureClassifierPrompt,
  buildFoundryRetrospectorInjectionPrompt,
  buildRetrospectorPrompt,
  checkBusWidthCapability,
  contractConformanceViolations,
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
  synthesisPreflightViolations,
  toStringList,
  writeJsonFile,
} from "../orchestrate.js";
import { PIPELINE_CONFIG, parseBooleanEnv, parsePositiveIntEnv } from "../config.js";
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

  it("surfaces retrospector_advice routing in the Surgeon repair brief", () => {
    const prompt = buildDelegationPrompt("surgeon", {
      layer_ir: {
        module_id: "m1",
        op_type: "conv2d",
        input_shape: [1, 64, 4, 4],
        output_shape: [1, 64, 4, 4],
        weights_path: "/tmp/w.hex",
        bias_path: "/tmp/b.hex",
        weight_shape: [64, 64, 1, 1],
        num_weights: 4096,
        scale_factor: 0.5,
        zero_point: 0,
        pipeline_latency_cycles: 40,
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
      verif_result: {
        module_id: "m1",
        status: "fail",
        status_class: "sim_completed_mismatch",
        max_error: 1,
        first_mismatch_index: 100,
      },
      retrospector_advice: {
        analysis: "trailing input pixels desync the active counter",
        suggestion: "stop consuming inputs once active outputs reach OH*OW",
        next_actor: "surgeon",
        base_artifact: "best_known",
        repair_scope: "targeted_fsm_or_datapath_fix",
      },
    });
    expect(prompt).toContain("post-retrospector final attempt");
    expect(prompt).toContain("scope=targeted_fsm_or_datapath_fix");
    expect(prompt).toContain("highest-scoring artifact across all prior attempts");
  });

  it("retrospector prompt documents the next_actor / base_artifact / repair_scope routing fields", () => {
    const prompt = buildRetrospectorPrompt({
      original_spec: {
        module_id: "m1",
        op_type: "conv2d",
        input_shape: [1, 64, 4, 4],
        output_shape: [1, 64, 4, 4],
        weights_path: "/tmp/w.hex",
        bias_path: "/tmp/b.hex",
        weight_shape: [64, 64, 1, 1],
        num_weights: 4096,
        scale_factor: 0.5,
        zero_point: 0,
        pipeline_latency_cycles: 40,
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
      contract: {
        interface: {
          clock: "clk",
          reset: "rst_n",
          valid_in: "valid_in",
          ready_in: "ready_in",
          valid_out: "valid_out",
          data_in_bits: 512,
          data_out_bits: 512,
        },
        timing: { pipeline_latency_cycles: 40, clock_period_ns: 20, fmax_target_mhz: 50 },
        capability_limits: {
          max_supported_bus_bits: 8192,
          target_part: "xczu9eg-ffvb1156-2-e",
          target_part_resources: {
            lut_logic: 274080,
            lut_distributed_ram: 144000,
            ff_total: 548160,
            block_ram_36k: 912,
            block_ram_18k_equiv: 1824,
            uram_288k: 0,
            dsp_slices: 2520,
            dsp_type: "DSP48E2",
          },
        },
      },
      current_contract: { id: "flat-bus", complexity: 0, description: "fixture flat-bus" },
      available_contracts: [],
      doc_used: { pattern_markdown: "fixture", reference_verilog: null, license_notice: null },
      knowledge_docs_used: [],
      foundry_versions: [],
      failure_attempts: [],
      failure_corpus: [],
    });
    expect(prompt).toContain("ROUTING (next_actor / base_artifact / repair_scope)");
    expect(prompt).toContain("`next_actor: \"surgeon\"`");
    expect(prompt).toContain("`base_artifact`");
    expect(prompt).toContain("`repair_scope`");
  });

  it("builds failure-classifier prompts with logs and contract-fit indicators", () => {
    const layer = {
      module_id: "m1",
      op_type: "conv2d",
      input_shape: [1, 64, 8, 8],
      output_shape: [1, 64, 8, 8],
      weights_path: "/tmp/w.hex",
      bias_path: "/tmp/b.hex",
      weight_shape: [64, 64, 3, 3],
      num_weights: 36864,
      scale_factor: 0.5,
      zero_point: 0,
      pipeline_latency_cycles: 100,
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
      stride: [1, 1],
      padding: [1, 1],
    };
    const prompt = buildFailureClassifierPrompt({
      module_spec: layer as never,
      contract: {
        interface: {
          clock: "clk",
          reset: "rst_n",
          valid_in: "valid_in",
          ready_in: "ready_in",
          valid_out: "valid_out",
          data_in_bits: 512,
          data_out_bits: 512,
        },
        timing: { pipeline_latency_cycles: 100, clock_period_ns: 20, fmax_target_mhz: 50 },
        capability_limits: {
          max_supported_bus_bits: 4096,
          target_part: "xczu9eg-ffvb1156-2-e",
          zcu102_capacity: { lut: 274080, ff: 548160, dsp: 2520, bram18: 1824 },
        },
        operation: { op_type: "conv2d" },
      },
      failure_result: {
        module_id: "m1",
        status: "fail",
        failure_class: "synthesis_failed",
        fix_hint: "Vivado failed.",
      },
      logs: {
        synthesis_report: "ERROR: DSP48 exhausted; resource utilization exceeds available DSP.",
      },
    });

    expect(prompt).toContain("code_bug");
    expect(prompt).toContain("architectural_fit");
    expect(prompt).toContain("verification_env");
    expect(prompt).toContain("unknown");
    expect(prompt).toContain("Contract-fit indicators");
    expect(prompt).toContain("DSP48 exhausted");
    expect(prompt).toContain("violated_resource");
  });

  it("builds retrospector and resumed Foundry prompts with advisory-only memory injection", () => {
    const layer = {
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
      pipeline_latency_cycles: 3,
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
    } as const;
    const advice = {
      analysis: "The attempts reuse a stale channel index.",
      suggestion: "Register the channel-local sum with the matching index before saturation.",
    };

    const retroPrompt = buildRetrospectorPrompt({
      original_spec: layer as never,
      contract: {
        interface: {
          clock: "clk",
          reset: "rst_n",
          valid_in: "valid_in",
          ready_in: "ready_in",
          valid_out: "valid_out",
          data_in_bits: 64,
          data_out_bits: 32,
        },
        timing: { pipeline_latency_cycles: 3, clock_period_ns: 20, fmax_target_mhz: 50 },
        capability_limits: {
          max_supported_bus_bits: 4096,
          target_part: "xczu9eg-ffvb1156-2-e",
          zcu102_capacity: { lut: 274080, ff: 548160, dsp: 2520, bram18: 1824 },
        },
        operation: { op_type: "add" },
      },
      doc_used: { pattern_markdown: "add pattern", reference_verilog: null, license_notice: null },
      current_contract: {
        id: "flat-bus",
        complexity: 0,
        description: "Full packed activation bus, current default contract.",
      },
      available_contracts: [
        {
          id: "flat-bus",
          complexity: 0,
          description: "Full packed activation bus, current default contract.",
        },
        {
          id: "tiled-streaming",
          complexity: 1,
          description: "Channel-tiled stream contract.",
        },
      ],
      knowledge_docs_used: [
        {
          id: "auto_add_doc",
          tier: "active",
          kind: "pattern",
          op_type: "add",
          path: "/repo/knowledge/patterns/active/auto_add_doc.md",
          relative_path: "knowledge/patterns/active/auto_add_doc.md",
        },
      ],
      foundry_versions: [
        {
          version_index: 1,
          session_id: "session-1",
          tool_use_summary: {},
          documents_used: [],
          module: {
            module_id: "m1",
            spec_hash: "add_4x4_s8x8_i64_o32",
            generated_by: "Foundry",
            attempt: 1,
            verilog_source: "module m1; endmodule",
          },
        },
      ],
      failure_attempts: [
        {
          attempt_index: 1,
          stage: "foundry_assayer",
          module: { module_id: "m1", spec_hash: "add_4x4_s8x8_i64_o32", generated_by: "Foundry", attempt: 1 },
          result: { module_id: "m1", status: "fail", failure_category: "code_bug" },
          logs: { verilator_stderr: "mismatch" },
        },
      ],
    });
    expect(retroPrompt).toContain("You are the `retrospector`");
    expect(retroPrompt).toContain("advisory JSON only");
    expect(retroPrompt).toContain("doc_fault");
    expect(retroPrompt).toContain("module m1; endmodule");
    expect(retroPrompt).toContain("add pattern");

    const foundryPrompt = buildFoundryRetrospectorInjectionPrompt({
      layer_ir: layer as never,
      expected_spec_hash: "add_4x4_s8x8_i64_o32",
      write_verilog_output_dir: "/tmp/rtl",
      retrospector_advice: advice,
      self_improve_doc_request: {
        enabled: true,
        destination_tier: "probationary",
      },
    });
    expect(foundryPrompt).toContain("existing `foundry` agent conversation");
    expect(foundryPrompt).toContain("preserve the working memory");
    expect(foundryPrompt).toContain("exactly one final RTL attempt");
    expect(foundryPrompt).toContain("draft_doc");
    expect(foundryPrompt).toContain(advice.suggestion);
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

  it("exposes self-improvement mode and parses boolean env toggles", () => {
    expect(typeof PIPELINE_CONFIG.self_improve).toBe("boolean");
    expect(PIPELINE_CONFIG.doc_promotion_success_threshold).toBeGreaterThan(0);
    expect(parseBooleanEnv({}, "NN2RTL_SELF_IMPROVE", false)).toBe(false);
    expect(parseBooleanEnv({ NN2RTL_SELF_IMPROVE: "on" }, "NN2RTL_SELF_IMPROVE", false)).toBe(true);
    expect(parseBooleanEnv({ NN2RTL_SELF_IMPROVE: "0" }, "NN2RTL_SELF_IMPROVE", true)).toBe(false);
    expect(parseBooleanEnv({}, "NN2RTL_SELF_IMPROVE", true)).toBe(true);
    expect(parsePositiveIntEnv({ NN2RTL_DOC_PROMOTION_SUCCESSES: "5" }, "NN2RTL_DOC_PROMOTION_SUCCESSES", 3)).toBe(5);
    expect(parsePositiveIntEnv({ NN2RTL_DOC_PROMOTION_SUCCESSES: "0" }, "NN2RTL_DOC_PROMOTION_SUCCESSES", 3)).toBe(3);
  });

  it("loads plugin agent definitions with merged MCP tools and skills", async () => {
    const foundry = await loadPluginAgentDefinition("foundry");
    const surgeon = await loadPluginAgentDefinition("surgeon");
    const cartographer = await loadPluginAgentDefinition("cartographer");
    expect(foundry.tools).toContain("Bash");
    expect(foundry.tools).toContain("mcp__nn2rtl-tools__write_verilog");
    expect(foundry.tools).toContain("mcp__nn2rtl-tools__get_rtl_patterns");
    expect(foundry.tools).toContain("mcp__nn2rtl-tools__get_failure_corpus");
    expect(foundry.tools).toContain("mcp__nn2rtl-tools__compute_layer_reference");
    expect(surgeon.tools).toContain("mcp__nn2rtl-tools__compute_layer_reference");
    expect(foundry.prompt).toContain("Knowledge catalog");
    expect(foundry.prompt).toContain("knowledge/patterns/protected/02_conv1x1.md");
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
        spec_hash: "conv2d_3x64x7x7_s224x224_i24_o512",
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
          "  // filler to keep this fixture past the truncated-stub guard",
          "  // 01",
          "  // 02",
          "  // 03",
          "  // 04",
          "  // 05",
          "  // 06",
          "  // 07",
          "  // 08",
          "  // 09",
          "  // 10",
          "  // 11",
          "  // 12",
          "  // 13",
          "  // 14",
          "  // 15",
          "  // 16",
          "  // 17",
          "  // 18",
          "  // 19",
          "  // 20",
          "  // 21",
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
        spec_hash: "conv2d_3x64x7x7_s224x224_i24_o512",
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

  it("rejects wrong-contract spec hashes before verification", () => {
    const issues = preflightVerilogModule(
      {
        module_id: "m1",
        spec_hash: "conv2d_64x64x1x1_s1x1_i512_o512_iodram-backed-weights_tile32",
        generated_by: "Foundry",
        attempt: 1,
        verilog_source: [
          "module m1(",
          "  input wire clk,",
          "  input wire rst_n,",
          "  input wire valid_in,",
          "  output wire ready_in,",
          "  input wire [511:0] data_in,",
          "  output wire valid_out,",
          "  output wire [511:0] data_out",
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

    expect(issues.some((issue) => issue.includes("does not match expected spec_hash"))).toBe(true);
    expect(issues.some((issue) => issue.includes("selected contract 'flat-bus'"))).toBe(true);
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
          "  // filler to keep this fixture past the short-stub guard",
          "  // 01",
          "  // 02",
          "  // 03",
          "  // 04",
          "  // 05",
          "  // 06",
          "  // 07",
          "  // 08",
          "  // 09",
          "  // 10",
          "  // 11",
          "  // 12",
          "  // 13",
          "  // 14",
          "  // 15",
          "  // 16",
          "  // 17",
          "  // 18",
          "  // 19",
          "  // 20",
          "  // 21",
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
          "  initial $readmemh(\"/tmp/w.hex\", weights);",
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

  it("flags large scalarized activation memories before Vivado but accepts packed beat memories", () => {
    const layer = {
      module_id: "m1",
      op_type: "conv2d" as const,
      input_shape: [1, 512, 14, 14],
      output_shape: [1, 512, 7, 7],
      weights_path: "/tmp/w.hex",
      bias_path: "/tmp/b.hex",
      weight_shape: [512, 512, 3, 3],
      num_weights: 512 * 512 * 3 * 3,
      scale_factor: 0.5,
      zero_point: 0,
      pipeline_latency_cycles: 100,
      clock_period_ns: 20,
      input_width_bits: 256,
      output_width_bits: 256,
      clock_signal: "clk" as const,
      reset_signal: "rst_n" as const,
      valid_in_signal: "valid_in" as const,
      valid_out_signal: "valid_out" as const,
      ready_in_signal: "ready_in" as const,
      data_in_signal: "data_in" as const,
      data_out_signal: "data_out" as const,
      golden_inputs_path: "/tmp/in.goldin",
      golden_outputs_path: "/tmp/out.goldout",
      channel_tile: 32,
    };

    const bad = synthesisPreflightViolations(
      {
        module_id: "m1",
        spec_hash: "hash",
        generated_by: "Foundry" as const,
        attempt: 1,
        verilog_source: [
          "module m1();",
          "  localparam IH = 14;",
          "  localparam IW = 14;",
          "  localparam IC = 512;",
          "  reg signed [7:0] line_buf [0:IH*IW-1][0:IC-1];",
          "endmodule",
        ].join("\n"),
      },
      layer,
    );
    expect(bad.map((v) => v.rule)).toContain("large_scalarized_activation_memory");

    const good = synthesisPreflightViolations(
      {
        module_id: "m1",
        spec_hash: "hash",
        generated_by: "Foundry" as const,
        attempt: 1,
        verilog_source: [
          "module m1();",
          "  localparam BEAT_BITS = 256;",
          "  reg [BEAT_BITS-1:0] line_buf [0:63][0:15];",
          "endmodule",
        ].join("\n"),
      },
      layer,
    );
    expect(good).toEqual([]);
  });

  it("flags declarations inside always blocks as a structural violation", () => {
    const violations = structuralPreflightViolations(
      {
        module_id: "m1",
        spec_hash: "hash",
        generated_by: "Surgeon",
        attempt: 3,
        verilog_source: [
          "module m1(input wire clk);",
          "  reg signed [7:0] weights [0:63];",
          "  reg signed [31:0] biases [0:0];",
          "  initial begin",
          "    $readmemh(\"/tmp/w.hex\", weights);",
          "    $readmemh(\"/tmp/b.hex\", biases);",
          "  end",
          "  always @(posedge clk) begin : BIAS_LANE",
          "    integer bias_oc;",
          "    bias_oc = 0;",
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
    expect(violations.map((v) => v.rule)).toContain("procedural_declaration_forbidden");
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
          "    $readmemh(\"/tmp/w.hex\", weights);",
          "    $readmemh(\"/tmp/b.hex\", biases);",
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
    expect(reason).toContain("does not fit contract 'flat-bus'");
    expect(reason).toContain("max_bus_width_bits=4096");

    const smallLayer = { ...baseLayer, input_width_bits: 512, output_width_bits: 512 };
    expect(checkBusWidthCapability(smallLayer)).toBeNull();

    const layer2Add = {
      ...baseLayer,
      op_type: "add" as const,
      input_shape: [1, 512, 28, 28],
      output_shape: [1, 512, 28, 28],
      weight_shape: [1],
      num_weights: 0,
      bias_path: null,
      input_width_bits: 8192,
      output_width_bits: 4096,
    };
    expect(checkBusWidthCapability(layer2Add)).toBeNull();

    const layer3Add = {
      ...layer2Add,
      input_shape: [1, 1024, 14, 14],
      output_shape: [1, 1024, 14, 14],
      input_width_bits: 16384,
      output_width_bits: 8192,
    };
    expect(checkBusWidthCapability(layer3Add)).toContain("effective_input_width_bits=8192");
  });

  it("preflight uses selected contract metadata for extra interface ports", () => {
    const layer = {
      module_id: "dram_conv",
      op_type: "conv2d" as const,
      contract_id: "dram-backed-weights" as const,
      input_shape: [1, 64, 1, 1],
      output_shape: [1, 64, 1, 1],
      weights_path: "/tmp/w.hex",
      bias_path: "/tmp/b.hex",
      weight_shape: [64, 64, 1, 1],
      num_weights: 4096,
      scale_factor: 0.5,
      zero_point: 0,
      pipeline_latency_cycles: 1,
      clock_period_ns: 20,
      input_width_bits: 512,
      output_width_bits: 512,
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
    const module = {
      module_id: "dram_conv",
      spec_hash: "fixture",
      generated_by: "Foundry" as const,
      attempt: 1,
      verilog_source: `
module dram_conv(
  input wire clk,
  input wire rst_n,
  input wire valid_in,
  output wire ready_in,
  input wire [511:0] data_in,
  output wire valid_out,
  output wire [511:0] data_out
);
endmodule
`,
    };

    const issues = preflightVerilogModule(module, layer);
    expect(issues.some((issue) => issue.includes("weights_arvalid"))).toBe(true);
  });

  it("rejects dram-backed RTL that ties off external weights or stores the full tensor on chip", () => {
    const layer = {
      module_id: "dram_conv",
      op_type: "conv2d" as const,
      contract_id: "dram-backed-weights" as const,
      input_shape: [1, 64, 1, 1],
      output_shape: [1, 64, 1, 1],
      weights_path: "/tmp/w.hex",
      bias_path: "/tmp/b.hex",
      weight_shape: [64, 64, 1, 1],
      num_weights: 4096,
      scale_factor: 0.5,
      zero_point: 0,
      pipeline_latency_cycles: 1,
      clock_period_ns: 20,
      input_width_bits: 512,
      output_width_bits: 512,
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

    const violations = contractConformanceViolations(
      {
        module_id: "dram_conv",
        spec_hash: "fixture",
        generated_by: "Foundry" as const,
        attempt: 1,
        verilog_source: `
module dram_conv();
  assign weights_arvalid = 1'b0;
  reg [7:0] weights [0:OC*K_TOTAL-1];
  initial begin
    $readmemh("/tmp/w.hex", weights);
  end
endmodule
`,
      },
      layer,
    );

    expect(violations.map((violation) => violation.rule)).toEqual([
      "contract_dram_weights_arvalid_tied_off",
      "contract_dram_full_weight_array",
      "contract_dram_full_weight_readmemh",
    ]);
  });

  it("rejects negative-half fixed-point rounding before simulation", () => {
    const layer = {
      module_id: "rounding_conv",
      op_type: "conv2d" as const,
      contract_id: "dram-backed-weights" as const,
      input_shape: [1, 64, 1, 1],
      output_shape: [1, 64, 1, 1],
      weights_path: "/tmp/w.hex",
      bias_path: "/tmp/b.hex",
      weight_shape: [64, 64, 1, 1],
      num_weights: 4096,
      scale_factor: 0.5,
      zero_point: 0,
      pipeline_latency_cycles: 1,
      clock_period_ns: 20,
      input_width_bits: 512,
      output_width_bits: 512,
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

    const violations = structuralPreflightViolations(
      {
        module_id: "rounding_conv",
        spec_hash: "fixture",
        generated_by: "Foundry" as const,
        attempt: 1,
        verilog_source: `
module rounding_conv();
  localparam signed [63:0] ROUND_BIAS_POS = 64'sd524288;
  localparam signed [63:0] ROUND_BIAS_NEG = -64'sd524288;
  wire signed [63:0] rnd0 = (scaled + (scaled[63] ? -SCALE_ROUND_HALF : SCALE_ROUND_HALF)) >>> SCALE_SHIFT;
endmodule
`,
      },
      layer,
    );

    expect(violations.map((violation) => violation.rule)).toContain(
      "rounding_negative_half_forbidden",
    );
    expect(violations.find((violation) => violation.rule === "rounding_negative_half_forbidden")?.detail).toContain(
      "HALF - 1",
    );
  });

  it("rejects $readmemh with a relative-path string literal", () => {
    const layer = {
      module_id: "relpath_conv",
      op_type: "conv2d" as const,
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
    const violations = structuralPreflightViolations(
      {
        module_id: "relpath_conv",
        spec_hash: "fixture",
        generated_by: "Foundry" as const,
        attempt: 1,
        verilog_source: [
          "module relpath_conv();",
          "  reg signed [7:0] weights [0:63];",
          "  reg signed [31:0] biases [0:0];",
          "  initial begin",
          "    $readmemh(\"output/weights/w.hex\", weights);",
          "    $readmemh(\"b.hex\", biases);",
          "  end",
          "endmodule",
        ].join("\n"),
      },
      layer,
    );
    expect(violations.map((v) => v.rule)).toContain("readmemh_relative_path_forbidden");
    expect(violations.find((v) => v.rule === "readmemh_relative_path_forbidden")?.detail).toContain(
      "absolute path",
    );
  });

  it("rejects multi-vector contract RTL whose ST_DONE locks the FSM", () => {
    const layer = {
      module_id: "lock_conv",
      op_type: "conv2d" as const,
      contract_id: "dram-backed-weights" as const,
      input_shape: [1, 1024, 14, 14],
      output_shape: [1, 2048, 7, 7],
      weights_path: "/tmp/w.hex",
      bias_path: "/tmp/b.hex",
      weight_shape: [2048, 1024, 1, 1],
      num_weights: 2097152,
      scale_factor: 0.015,
      zero_point: 0,
      pipeline_latency_cycles: 2100225,
      clock_period_ns: 20,
      input_width_bits: 256,
      output_width_bits: 256,
      clock_signal: "clk" as const,
      reset_signal: "rst_n" as const,
      valid_in_signal: "valid_in" as const,
      valid_out_signal: "valid_out" as const,
      ready_in_signal: "ready_in" as const,
      data_in_signal: "data_in" as const,
      data_out_signal: "data_out" as const,
      golden_inputs_path: "/tmp/in.goldin",
      golden_outputs_path: "/tmp/out.goldout",
      io_mode: "dram_backed_weights" as const,
      channel_tile: 32,
    };
    const violations = structuralPreflightViolations(
      {
        module_id: "lock_conv",
        spec_hash: "fixture",
        generated_by: "Foundry" as const,
        attempt: 1,
        verilog_source: `
module lock_conv();
  reg signed [7:0] weights [0:63];
  initial $readmemh("/tmp/w.hex", weights);
  // Real AR FSM elsewhere in the module — fake here to keep fixture small.
  output reg weights_arvalid;
  always @* begin end
  // The bad block: ST_DONE holds ready_in low forever and never transitions.
  reg [3:0] state;
  localparam ST_DONE = 4'd9;
  always @(posedge clk) begin
    case (state)
      ST_DONE: begin
        valid_out <= 1'b0;
        ready_in  <= 1'b0;
      end
    endcase
  end
endmodule
`,
      },
      layer,
    );
    expect(violations.map((v) => v.rule)).toContain("dram_backed_weights_terminal_done_lock");
    expect(violations.find((v) => v.rule === "dram_backed_weights_terminal_done_lock")?.detail).toContain(
      "vector N+1",
    );
  });

  it("rejects weight-tiling RTL with a fake scheduler or full active tile", () => {
    const layer = {
      module_id: "wt_conv",
      op_type: "conv2d" as const,
      contract_id: "weight-tiling" as const,
      input_shape: [1, 64, 8, 8],
      output_shape: [1, 64, 8, 8],
      weights_path: "/tmp/w.hex",
      bias_path: "/tmp/b.hex",
      weight_shape: [64, 64, 3, 3],
      num_weights: 36864,
      scale_factor: 0.5,
      zero_point: 0,
      pipeline_latency_cycles: 1,
      clock_period_ns: 20,
      input_width_bits: 256,
      output_width_bits: 256,
      clock_signal: "clk" as const,
      reset_signal: "rst_n" as const,
      valid_in_signal: "valid_in" as const,
      valid_out_signal: "valid_out" as const,
      ready_in_signal: "ready_in" as const,
      data_in_signal: "data_in" as const,
      data_out_signal: "data_out" as const,
      golden_inputs_path: "/tmp/in.goldin",
      golden_outputs_path: "/tmp/out.goldout",
      channel_tile: 32,
    };

    const violations = contractConformanceViolations(
      {
        module_id: "wt_conv",
        spec_hash: "fixture",
        generated_by: "Foundry" as const,
        attempt: 1,
        verilog_source: `
module wt_conv();
  coord_scheduler u_coord_scheduler(.clk(clk), .valid_in(1'b0), .ready_in());
  reg signed [7:0] active_weight_tile [0:OC*K_TOTAL-1];
endmodule
`,
      },
      layer,
    );

    expect(violations.map((violation) => violation.rule)).toEqual([
      "contract_weight_tiling_fake_scheduler",
      "contract_weight_tiling_full_active_tile",
    ]);
  });

  it("rejects tiled-streaming RTL without deterministic tile beat counters", () => {
    const layer = {
      module_id: "tile_conv",
      op_type: "conv2d" as const,
      contract_id: "tiled-streaming" as const,
      input_shape: [1, 64, 8, 8],
      output_shape: [1, 64, 8, 8],
      weights_path: "/tmp/w.hex",
      bias_path: "/tmp/b.hex",
      weight_shape: [64, 64, 1, 1],
      num_weights: 4096,
      scale_factor: 0.5,
      zero_point: 0,
      pipeline_latency_cycles: 1,
      clock_period_ns: 20,
      input_width_bits: 512,
      output_width_bits: 256,
      clock_signal: "clk" as const,
      reset_signal: "rst_n" as const,
      valid_in_signal: "valid_in" as const,
      valid_out_signal: "valid_out" as const,
      ready_in_signal: "ready_in" as const,
      data_in_signal: "data_in" as const,
      data_out_signal: "data_out" as const,
      golden_inputs_path: "/tmp/in.goldin",
      golden_outputs_path: "/tmp/out.goldout",
      channel_tile: 32,
    };

    const violations = contractConformanceViolations(
      {
        module_id: "tile_conv",
        spec_hash: "fixture",
        generated_by: "Foundry" as const,
        attempt: 1,
        verilog_source: "module tile_conv(); endmodule",
      },
      layer,
    );

    expect(violations.map((violation) => violation.rule)).toEqual([
      "contract_tiled_streaming_bus_width",
      "contract_tiled_streaming_beat_counter_missing",
    ]);
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
