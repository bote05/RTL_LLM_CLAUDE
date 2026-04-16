import { readdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  createOrchestratorRuntime,
  runPipeline,
} from "../orchestrate.js";
import type { SDKMessage, SDKResultMessage } from "../claude-agent-sdk-compat.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../..");
const outputRoot = path.join(repoRoot, "output");
const reportsDir = path.join(outputRoot, "reports");
const rtlDir = path.join(outputRoot, "rtl");

type MockStep = SDKResultMessage | (() => Promise<SDKResultMessage> | SDKResultMessage);
type YosysReport = { success: boolean; lut_count: number; fmax_mhz: number; area_um2?: number; report: string };
type YosysStep = YosysReport | (() => Promise<YosysReport> | YosysReport);
type VerifLike = Record<string, unknown>;
type AssayerStep = VerifLike | (() => Promise<VerifLike> | VerifLike);

const fixedNow = () => new Date("2026-04-14T00:00:00Z");

function createYosysMock(steps: YosysStep[]): ReturnType<typeof vi.fn> {
  return vi.fn(async () => {
    const next = steps.shift();
    if (!next) {
      throw new Error("No mock result queued for yosysFn.");
    }
    const report = typeof next === "function" ? await next() : next;
    return { area_um2: 0, ...report };
  });
}

// Assayer is now deterministic in production (orchestrator calls run_iverilog
// + run_verilator directly), so tests inject a VerifResult stream here.
function createAssayerMock(steps: AssayerStep[]): ReturnType<typeof vi.fn> {
  return vi.fn(async () => {
    const next = steps.shift();
    if (!next) {
      throw new Error("No mock result queued for assayerFn.");
    }
    return typeof next === "function" ? await next() : next;
  });
}

async function resetOutput(): Promise<void> {
  for (const dir of ["reports", "rtl", "tb", "weights"]) {
    const fullDir = path.join(outputRoot, dir);
    for (const entry of await readdir(fullDir)) {
      if (entry === ".gitkeep") {
        continue;
      }
      await rm(path.join(fullDir, entry), { recursive: true, force: true });
    }
  }

  for (const fileName of ["layer_ir.json", "pipeline_state.json", "golden_vectors.json"]) {
    await rm(path.join(outputRoot, fileName), { force: true });
  }
}

async function writeFixture(relativeFixturePath: string, destinationPath: string): Promise<unknown> {
  const raw = await readFile(path.join(repoRoot, "test", "fixtures", relativeFixturePath), "utf8");
  await writeFile(destinationPath, raw, "utf8");
  return JSON.parse(raw);
}

function successResult(structured_output: unknown): SDKResultMessage {
  return {
    type: "result",
    subtype: "success",
    result: JSON.stringify(structured_output),
    structured_output,
    total_cost_usd: 1,
    modelUsage: { fixture: { input_tokens: 1, output_tokens: 1 } },
  };
}

function createQueryMock(
  steps: Partial<Record<"cartographer" | "foundry" | "surgeon", MockStep[]>>,
): ReturnType<typeof vi.fn> {
  return vi.fn(async function* ({
    prompt,
  }: {
    prompt: string;
  }): AsyncGenerator<SDKMessage, void> {
    const key =
      prompt.includes("Invoke the `cartographer`") ? "cartographer"
        : prompt.includes("Invoke the `foundry`") ? "foundry"
        : prompt.includes("Invoke the `surgeon`") ? "surgeon"
        : null;

    if (!key) {
      throw new Error(`Unhandled prompt: ${prompt}`);
    }

    const queue = steps[key];
    if (!queue || queue.length === 0) {
      throw new Error(`No mock result queued for ${key}.`);
    }

    const next = queue.shift();
    const message = typeof next === "function" ? await next() : next;
    yield message;
  });
}

beforeEach(async () => {
  await resetOutput();
});

afterEach(async () => {
  await resetOutput();
});

describe("runPipeline", () => {
  it("uses an existing layer_ir.json without invoking Cartographer", async () => {
    const pipelineIr = await writeFixture("pipeline_ir.json", path.join(outputRoot, "layer_ir.json"));
    const module = await writeFixture("verilog_module.json", path.join(rtlDir, "unit_module.meta.json"));
    await writeFile(path.join(rtlDir, "unit_module.v"), (module as { verilog_source: string }).verilog_source, "utf8");
    const verifPass = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_pass.json"), "utf8"),
    );

    const queryFn = createQueryMock({
      foundry: [successResult(module)],
    });
    const assayerFn = createAssayerMock([verifPass]);
    const yosysFn = createYosysMock([{ success: true, lut_count: 1, fmax_mhz: 75, report: "fixture" }]);

    await runPipeline("checkpoint.pth", {
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, yosysFn, assayerFn }),
    });

    expect(queryFn.mock.calls.some(([call]) => (call as { prompt: string }).prompt.includes("cartographer"))).toBe(false);
    expect(JSON.parse(await readFile(path.join(reportsDir, "pipeline_summary.json"), "utf8"))).toMatchObject({
      model_name: (pipelineIr as { model_name: string }).model_name,
      is_done: true,
    });
  });

  it("invokes Cartographer when layer_ir.json is missing", async () => {
    const pipelineIr = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "pipeline_ir.json"), "utf8"),
    );
    const module = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verilog_module.json"), "utf8"),
    );
    const verifPass = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_pass.json"), "utf8"),
    );

    const queryFn = createQueryMock({
      cartographer: [successResult(pipelineIr)],
      foundry: [async () => {
        await writeFile(path.join(rtlDir, "unit_module.meta.json"), `${JSON.stringify(module, null, 2)}\n`, "utf8");
        return successResult(module);
      }],
    });
    const assayerFn = createAssayerMock([verifPass]);
    const yosysFn = createYosysMock([{ success: true, lut_count: 2, fmax_mhz: 75, report: "fixture" }]);

    await runPipeline("checkpoint.pth", {
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, yosysFn, assayerFn }),
    });

    expect(queryFn.mock.calls.some(([call]) => (call as { prompt: string }).prompt.includes("cartographer"))).toBe(true);
    expect(JSON.parse(await readFile(path.join(outputRoot, "layer_ir.json"), "utf8"))).toEqual(pipelineIr);
  });

  it("runs the Surgeon repair path after a failed verification", async () => {
    await writeFixture("pipeline_ir.json", path.join(outputRoot, "layer_ir.json"));
    const originalModule = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verilog_module.json"), "utf8"),
    );
    const repairedModule = {
      ...originalModule,
      generated_by: "Surgeon",
      attempt: 2,
    };
    const verifFail = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_fail.json"), "utf8"),
    );
    const verifPass = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_pass.json"), "utf8"),
    );

    const queryFn = createQueryMock({
      foundry: [async () => {
        await writeFile(path.join(rtlDir, "unit_module.meta.json"), `${JSON.stringify(originalModule, null, 2)}\n`, "utf8");
        return successResult(originalModule);
      }],
      surgeon: [async () => {
        await writeFile(path.join(rtlDir, "unit_module.meta.json"), `${JSON.stringify(repairedModule, null, 2)}\n`, "utf8");
        return successResult(repairedModule);
      }],
    });
    const assayerFn = createAssayerMock([verifFail, verifPass]);
    const yosysFn = createYosysMock([{ success: true, lut_count: 3, fmax_mhz: 75, report: "fixture" }]);

    await runPipeline("checkpoint.pth", {
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, yosysFn, assayerFn }),
    });

    const prompts = queryFn.mock.calls.map(([call]) => (call as { prompt: string }).prompt);
    expect(prompts.some((prompt) => prompt.includes("Invoke the `surgeon`"))).toBe(true);

    const state = JSON.parse(await readFile(path.join(outputRoot, "pipeline_state.json"), "utf8"));
    expect(state.modules.unit_module).toBe("pass");
    expect(state.attempts.unit_module).toBe(1);
  });

  it("runs the Surgeon repair path after a failed yosys synthesis report", async () => {
    await writeFixture("pipeline_ir.json", path.join(outputRoot, "layer_ir.json"));
    const originalModule = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verilog_module.json"), "utf8"),
    );
    const repairedModule = {
      ...originalModule,
      generated_by: "Surgeon",
      attempt: 2,
    };
    const verifPass = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_pass.json"), "utf8"),
    );

    const queryFn = createQueryMock({
      foundry: [async () => {
        await writeFile(path.join(rtlDir, "unit_module.meta.json"), `${JSON.stringify(originalModule, null, 2)}\n`, "utf8");
        return successResult(originalModule);
      }],
      surgeon: [async () => {
        await writeFile(path.join(rtlDir, "unit_module.meta.json"), `${JSON.stringify(repairedModule, null, 2)}\n`, "utf8");
        return successResult(repairedModule);
      }],
    });
    const assayerFn = createAssayerMock([verifPass, verifPass]);
    const yosysFn = createYosysMock([
      { success: false, lut_count: 0, fmax_mhz: 0, report: "synth failed" },
      { success: true, lut_count: 3, fmax_mhz: 75, report: "fixture" },
    ]);

    await runPipeline("checkpoint.pth", {
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, yosysFn, assayerFn }),
    });

    const prompts = queryFn.mock.calls.map(([call]) => (call as { prompt: string }).prompt);
    expect(yosysFn).toHaveBeenCalledTimes(2);
    expect(prompts.some((prompt) => prompt.includes("Invoke the `surgeon`"))).toBe(true);

    const state = JSON.parse(await readFile(path.join(outputRoot, "pipeline_state.json"), "utf8"));
    expect(state.modules.unit_module).toBe("pass");
    expect(state.attempts.unit_module).toBe(1);
    expect(await readFile(path.join(reportsDir, "run_log.jsonl"), "utf8")).toContain('"reason":"yosys_synthesis_failed"');
  });

  it("treats fmax_mhz=0 as a pass when Yosys succeeded (Sky130 does not always emit a delay)", async () => {
    await writeFixture("pipeline_ir.json", path.join(outputRoot, "layer_ir.json"));
    const originalModule = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verilog_module.json"), "utf8"),
    );
    const verifPass = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_pass.json"), "utf8"),
    );

    const queryFn = createQueryMock({
      foundry: [async () => {
        await writeFile(path.join(rtlDir, "unit_module.meta.json"), `${JSON.stringify(originalModule, null, 2)}\n`, "utf8");
        return successResult(originalModule);
      }],
      surgeon: [],
    });
    const assayerFn = createAssayerMock([verifPass]);
    // Sky130 `stat -liberty` + `abc -liberty` does not always print a
    // delay line, so `fmax_mhz: 0` with `success: true` is not a failure.
    // We still gate on lut_count and on Yosys success itself.
    const yosysFn = createYosysMock([
      { success: true, lut_count: 4, fmax_mhz: 0, area_um2: 123.4, report: "stat ran but no delay line" },
    ]);

    await runPipeline("checkpoint.pth", {
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, yosysFn, assayerFn }),
    });

    expect(yosysFn).toHaveBeenCalledTimes(1);
    const state = JSON.parse(await readFile(path.join(outputRoot, "pipeline_state.json"), "utf8"));
    expect(state.modules.unit_module).toBe("pass");
    expect(state.results.unit_module.failure_class).toBeUndefined();
  });

  it("routes Fmax-below-target to Surgeon with missing_pipeline_register failure class", async () => {
    await writeFixture("pipeline_ir.json", path.join(outputRoot, "layer_ir.json"));
    const originalModule = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verilog_module.json"), "utf8"),
    );
    const repairedModule = { ...originalModule, generated_by: "Surgeon", attempt: 2 };
    const verifPass = JSON.parse(
      await readFile(path.join(repoRoot, "test", "fixtures", "verif_pass.json"), "utf8"),
    );

    const queryFn = createQueryMock({
      foundry: [async () => {
        await writeFile(path.join(rtlDir, "unit_module.meta.json"), `${JSON.stringify(originalModule, null, 2)}\n`, "utf8");
        return successResult(originalModule);
      }],
      surgeon: [async () => {
        await writeFile(path.join(rtlDir, "unit_module.meta.json"), `${JSON.stringify(repairedModule, null, 2)}\n`, "utf8");
        return successResult(repairedModule);
      }],
    });
    const assayerFn = createAssayerMock([verifPass, verifPass]);
    const yosysFn = createYosysMock([
      { success: true, lut_count: 4, fmax_mhz: 35, report: "ABC: Delay = 28571 ps" },
      { success: true, lut_count: 4, fmax_mhz: 75, report: "ABC: Delay = 13333 ps" },
    ]);

    await runPipeline("checkpoint.pth", {
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn, yosysFn, assayerFn }),
    });

    expect(yosysFn).toHaveBeenCalledTimes(2);
    const state = JSON.parse(await readFile(path.join(outputRoot, "pipeline_state.json"), "utf8"));
    expect(state.modules.unit_module).toBe("pass");
    expect(await readFile(path.join(reportsDir, "run_log.jsonl"), "utf8")).toContain('"reason":"yosys_missing_pipeline_register"');
  });

  it("blocks resume when fail_retry has no prior verification result", async () => {
    await writeFixture("pipeline_ir.json", path.join(outputRoot, "layer_ir.json"));
    await writeFile(
      path.join(outputRoot, "pipeline_state.json"),
      JSON.stringify({
        run_id: "run-1",
        started_at: "2026-04-14T00:00:00Z",
        modules: { unit_module: "fail_retry" },
        attempts: { unit_module: 1 },
        results: {},
        max_retries: 3,
        total_cost_usd: 0,
        model_usage: {},
      }),
      "utf8",
    );

    const queryFn = createQueryMock({});
    await expect(
      runPipeline("checkpoint.pth", {
        resume: true,
        runtime: createOrchestratorRuntime({ now: fixedNow, queryFn }),
      }),
    ).rejects.toThrow("fail_retry");
  });

  it("reloads resume state before completing", async () => {
    await writeFixture("pipeline_ir.json", path.join(outputRoot, "layer_ir.json"));
    await writeFile(
      path.join(outputRoot, "pipeline_state.json"),
      JSON.stringify({
        run_id: "run-1",
        started_at: "2026-04-14T00:00:00Z",
        modules: { unit_module: "pass" },
        attempts: { unit_module: 0 },
        results: { unit_module: { module_id: "unit_module", status: "pass" } },
        max_retries: 3,
        total_cost_usd: 0,
        model_usage: {},
      }),
      "utf8",
    );

    const queryFn = createQueryMock({});
    await runPipeline("checkpoint.pth", {
      resume: true,
      runtime: createOrchestratorRuntime({ now: fixedNow, queryFn }),
    });

    expect(queryFn).not.toHaveBeenCalled();
    expect(await readFile(path.join(reportsDir, "run_log.jsonl"), "utf8")).toContain(
      '"event":"pipeline_resume_loaded"',
    );
  });
});
