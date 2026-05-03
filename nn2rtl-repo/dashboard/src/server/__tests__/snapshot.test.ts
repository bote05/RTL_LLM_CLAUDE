import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import { buildSnapshot } from "../snapshot.js";

let tempRoot: string | null = null;

async function writeJson(filePath: string, value: unknown): Promise<void> {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

async function seedRepo(): Promise<string> {
  tempRoot = await mkdtemp(path.join(os.tmpdir(), "nn2rtl-dashboard-"));
  await writeJson(path.join(tempRoot, "output", "layer_ir.json"), {
    model_name: "fixture-net",
    quantization: "int8_symmetric_per_tensor",
    generated_at: "2026-05-03T00:00:00Z",
    layers: [
      {
        module_id: "m0",
        op_type: "conv2d",
        input_shape: [1, 1, 1, 1],
        output_shape: [1, 4, 1, 1],
        weight_shape: [4, 1, 1, 1],
        pipeline_latency_cycles: 4,
      },
      {
        module_id: "m1",
        op_type: "relu",
        input_shape: [1, 4, 1, 1],
        output_shape: [1, 4, 1, 1],
        weight_shape: [1],
        pipeline_latency_cycles: 1,
      },
    ],
  });
  await mkdir(path.join(tempRoot, "output", "rtl"), { recursive: true });
  await writeFile(path.join(tempRoot, "output", "rtl", "m0.v"), "module m0; endmodule\n", "utf8");
  await writeJson(path.join(tempRoot, "output", "reports", "m0.results.json"), {
    module_id: "m0",
    status: "pass",
    timing_pass: true,
    timing_actual_cycles: 4,
    timing_expected_cycles: 4,
  });
  await writeJson(path.join(tempRoot, "output", "reports", "m0.vivado.json"), {
    success: true,
    timing_met: true,
    lut_count: 12,
    ff_count: 5,
    dsp_count: 2,
    bram18_equiv: 1,
    fmax_mhz: 120,
  });
  await writeJson(path.join(tempRoot, "output", "pipeline_state.json"), {
    run_id: "scoped",
    modules: { m0: "pass" },
    attempts: { m0: 0 },
    total_cost_usd: 2.5,
  });
  await writeJson(path.join(tempRoot, "knowledge", "doc_lifecycle.json"), {
    version: 1,
    docs: {
      improved_m0__use_dsp: {
        status: "improved",
        op_type: "conv2d",
        created_by_module: "m0",
        reference_path: "knowledge/references/improved/m0__use-dsp.v",
        improvement_targets: ["use-dsp"],
        successful_modules: ["m0"],
      },
    },
  });
  await mkdir(path.join(tempRoot, "knowledge", "references", "improved"), { recursive: true });
  await writeFile(path.join(tempRoot, "knowledge", "references", "improved", "m0__use-dsp.v"), "module m0; endmodule\n", "utf8");
  await writeJson(path.join(tempRoot, "output", "reports", "improve_m0__use-dsp.json"), {
    module_id: "m0",
    targets: ["use-dsp"],
    success: true,
    final_action: "kept-as-variant",
    attempts: [
      {
        attempt_index: 1,
        failed_gate: null,
        metrics: { lut: 10, dsp: 8, bram: 0 },
        verdict: { overall: true, targets: [{ target: "use-dsp", satisfied: true, reason: "ok" }] },
      },
    ],
    final_verdict: { overall: true, targets: [{ target: "use-dsp", satisfied: true, reason: "ok" }] },
  });
  await writeFile(path.join(tempRoot, "output", "rtl", "orphan.v"), "module orphan; endmodule\n", "utf8");
  return tempRoot;
}

afterEach(async () => {
  if (tempRoot) {
    await rm(tempRoot, { recursive: true, force: true });
    tempRoot = null;
  }
});

describe("buildSnapshot", () => {
  it("uses LayerIR as total coverage truth even when pipeline state is scoped", async () => {
    const root = await seedRepo();
    const snapshot = await buildSnapshot(root);

    expect(snapshot.kpis.totalLayers).toBe(2);
    expect(snapshot.latestPipeline?.stateCounts).toEqual({ pass: 1 });
    expect(snapshot.modules.map((module) => module.moduleId)).toEqual(["m0", "m1"]);
  });

  it("does not crash when reports are missing and links improvement/doc artifacts", async () => {
    const root = await seedRepo();
    const snapshot = await buildSnapshot(root);
    const m0 = snapshot.modules.find((module) => module.moduleId === "m0");
    const m1 = snapshot.modules.find((module) => module.moduleId === "m1");

    expect(m0?.stage).toBe("improved");
    expect(m0?.improvements[0]).toMatchObject({ targetSlug: "use-dsp", success: true });
    expect(m0?.docs[0]).toMatchObject({ id: "improved_m0__use_dsp", tier: "improved" });
    expect(m1?.stage).toBe("missing");
    expect(snapshot.orphanArtifacts.rtl).toContain("output/rtl/orphan.v");
  });
});
