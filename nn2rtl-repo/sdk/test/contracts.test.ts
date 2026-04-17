import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  layerIrSchema as mcpLayerIrSchema,
  verificationSidecarSchema as mcpVerificationSidecarSchema,
} from "../../mcp/schemas.js";
import {
  layerIrSchema as sdkLayerIrSchema,
  verificationSidecarSchema as sdkVerificationSidecarSchema,
} from "../schemas.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../..");

function extractInterface(source: string, interfaceName: string): string {
  const match = source.match(
    new RegExp(`export interface ${interfaceName} \\{([\\s\\S]*?)\\n\\}`, "m"),
  );
  if (!match) {
    throw new Error(`Could not find interface '${interfaceName}'.`);
  }
  return match[1].trim();
}

function extractSchemaDefinition(source: string, schemaName: string): string {
  const match = source.match(
    new RegExp(`export const ${schemaName} = z([\\s\\S]*?)\\.strict\\(\\);`, "m"),
  );
  if (!match) {
    throw new Error(`Could not find schema '${schemaName}'.`);
  }
  return `z${match[1]}.strict();`;
}

describe("contract parity", () => {
  it("keeps shared interfaces aligned across sdk and mcp", async () => {
    const sdkTypes = await readFile(path.join(repoRoot, "sdk", "types.ts"), "utf8");
    const mcpTypes = await readFile(path.join(repoRoot, "mcp", "types.ts"), "utf8");

    expect(extractInterface(sdkTypes, "LayerIR")).toBe(extractInterface(mcpTypes, "LayerIR"));
    expect(extractInterface(sdkTypes, "VerificationSidecar")).toBe(
      extractInterface(mcpTypes, "VerificationSidecar"),
    );
  });

  it("keeps sdk and mcp schemas aligned on canonical signals", async () => {
    const fixture = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "pipeline_ir.json"), "utf8"),
    );
    const layer = fixture.layers[0];

    expect(sdkLayerIrSchema.parse(layer)).toEqual(layer);
    expect(mcpLayerIrSchema.parse(layer)).toEqual(layer);
  });

  it("rejects maxpool LayerIR missing kernel_size / pool_stride / pool_padding", () => {
    // Regression guard for finding #4: before the superRefine, a
    // {op_type: "maxpool"} LayerIR with no geometry fields validated clean
    // and reached Foundry with zeroed placeholders in spec_hash.
    const baseLayer = {
      module_id: "m1",
      op_type: "maxpool" as const,
      input_shape: [1, 64, 112, 112],
      output_shape: [1, 64, 56, 56],
      weights_path: "/tmp/w.hex",
      bias_path: null,
      weight_shape: [1],
      num_weights: 0,
      scale_factor: 1.0,
      zero_point: 0,
      pipeline_latency_cycles: 227,
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

    // Missing every geometry field → must fail.
    for (const schema of [sdkLayerIrSchema, mcpLayerIrSchema]) {
      const parsed = schema.safeParse(baseLayer);
      expect(parsed.success).toBe(false);
      if (!parsed.success) {
        const issue = parsed.error.issues.find((i) =>
          i.message.includes("missing required geometry fields"),
        );
        expect(issue).toBeDefined();
      }
    }

    // Present on all three → must succeed.
    const ok = { ...baseLayer, kernel_size: [3, 3], pool_stride: [2, 2], pool_padding: [1, 1] };
    for (const schema of [sdkLayerIrSchema, mcpLayerIrSchema]) {
      expect(schema.safeParse(ok).success).toBe(true);
    }

    // conv2d without geometry stays legal (superRefine only fires for maxpool).
    const conv = { ...baseLayer, op_type: "conv2d" as const, weight_shape: [64, 64, 3, 3], num_weights: 36864 };
    for (const schema of [sdkLayerIrSchema, mcpLayerIrSchema]) {
      expect(schema.safeParse(conv).success).toBe(true);
    }
  });

  it("keeps the verification sidecar schema byte-identical across sdk and mcp", async () => {
    const sdkSchemas = await readFile(path.join(repoRoot, "sdk", "schemas.ts"), "utf8");
    const mcpSchemas = await readFile(path.join(repoRoot, "mcp", "schemas.ts"), "utf8");

    expect(extractSchemaDefinition(sdkSchemas, "verificationSidecarSchema")).toBe(
      extractSchemaDefinition(mcpSchemas, "verificationSidecarSchema"),
    );
  });

  it("matches the static testbench sidecar contract", async () => {
    const tbReadme = await readFile(path.join(repoRoot, "tb", "README.md"), "utf8");
    const tbCpp = await readFile(path.join(repoRoot, "tb", "static_verilator_tb.cpp"), "utf8");
    const sidecar = mcpVerificationSidecarSchema.parse({
      module_name: "unit_module",
      module_id: "unit_module",
      clock_signal: "clk",
      reset_signal: "rst_n",
      valid_in_signal: "valid_in",
      valid_out_signal: "valid_out",
      ready_in_signal: "ready_in",
      data_in_signal: "data_in",
      data_out_signal: "data_out",
      bus_bytes_per_sample: 1,
      input_width_bits: 8,
      output_width_bits: 8,
      pipeline_latency_cycles: 1,
      clock_period_ns: 20,
      golden_inputs_path: "/tmp/in.json",
      golden_outputs_path: "/tmp/out.json",
      results_path: "/tmp/results.json",
      testbench_template_path: "/tmp/tb.cpp",
    });
    expect(sdkVerificationSidecarSchema.parse(sidecar)).toEqual(sidecar);

    expect(sidecar.ready_in_signal).toBe("ready_in");
    expect(tbReadme).toContain('"ready_in_signal": "ready_in"');
    expect(tbReadme).toContain('"data_in_signal": "data_in"');
    expect(tbReadme).toContain('"data_out_signal": "data_out"');
    expect(tbReadme).toContain('"bus_bytes_per_sample": 1');
    expect(tbCpp).toContain("ready_in_signal");
    expect(tbCpp).toContain("data_in_signal");
    expect(tbCpp).toContain("data_out_signal");
    expect(tbCpp).toContain("bus_bytes_per_sample");
  });
});
