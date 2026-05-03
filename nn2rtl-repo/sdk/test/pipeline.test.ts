import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { PipelineStateManager } from "../pipeline.js";
import type { PipelineState, VerifResult } from "../types.js";

const PASS_RESULT: VerifResult = {
  module_id: "m1",
  status: "pass",
};

const FAIL_RESULT: VerifResult = {
  module_id: "m1",
  status: "fail",
  failure_class: "pipeline_latency_wrong",
};

const TB_SETUP_FAIL_RESULT: VerifResult = {
  module_id: "m1",
  status: "fail",
  status_class: "tb_setup_error",
  verilator_stderr: "static_verilator_tb.cpp: build failed",
};

const tempDirs: string[] = [];

async function makeTempPath(fileName: string): Promise<string> {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "nn2rtl-sdk-pipeline-"));
  tempDirs.push(tempDir);
  return path.join(tempDir, fileName);
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
});

describe("PipelineStateManager", () => {
  it("schedules pending modules through Foundry first", () => {
    const manager = new PipelineStateManager(["m1", "m2"]);

    expect(manager.tick()).toEqual({ action: "invoke_foundry", module_id: "m1" });
    expect(manager.getState().modules.m1).toBe("generating");
  });

  it("dispatches 2 Foundry calls then 1 Surgeon call before fail_abort", () => {
    const manager = new PipelineStateManager(["m1"], 3);

    // Call 1: Foundry on a fresh module.
    expect(manager.tick()).toEqual({ action: "invoke_foundry", module_id: "m1" });
    expect(manager.getState().attempts.m1).toBe(1);
    manager.applyVerifResult("m1", FAIL_RESULT);
    expect(manager.getState().modules.m1).toBe("fail_retry");

    // Call 2: a SECOND Foundry call (will be dispatched on the resumed session).
    expect(manager.tick()).toEqual({ action: "invoke_foundry", module_id: "m1" });
    expect(manager.getState().attempts.m1).toBe(2);
    manager.applyVerifResult("m1", FAIL_RESULT);
    expect(manager.getState().modules.m1).toBe("fail_retry");

    // Call 3: Surgeon takes over (sees Foundry's two prior attempts).
    expect(manager.tick()).toEqual({ action: "invoke_surgeon", module_id: "m1" });
    expect(manager.getState().attempts.m1).toBe(3);
    manager.applyVerifResult("m1", FAIL_RESULT);
    expect(manager.getState().modules.m1).toBe("fail_abort");
  });

  it("marks passing verification as pass", () => {
    const manager = new PipelineStateManager(["m1"]);
    manager.applyVerifResult("m1", PASS_RESULT);

    expect(manager.getState().modules.m1).toBe("pass");
    expect(manager.isDone()).toBe(true);
  });

  it("marks a failed verification as fail_abort when retries are exhausted", () => {
    const manager = new PipelineStateManager(["m1"], 0);
    manager.applyVerifResult("m1", FAIL_RESULT);

    expect(manager.getState().modules.m1).toBe("fail_abort");
    expect(manager.isDone()).toBe(true);
  });

  it("does not route pure testbench/setup failures to Surgeon", () => {
    const manager = new PipelineStateManager(["m1"], 3);
    manager.applyVerifResult("m1", TB_SETUP_FAIL_RESULT);

    expect(manager.getState().modules.m1).toBe("fail_abort");
    expect(manager.getState().results.m1).toEqual(TB_SETUP_FAIL_RESULT);
  });

  it("retries classifier-confirmed code bugs", () => {
    const manager = new PipelineStateManager(["m1"], 3);
    manager.applyVerifResult("m1", {
      ...FAIL_RESULT,
      failure_category: "code_bug",
      classifier_reason: "Syntax and simple logic failures are retryable.",
    });

    expect(manager.getState().modules.m1).toBe("fail_retry");
  });

  it("fail-aborts architectural-fit and unknown classifier results", () => {
    const archManager = new PipelineStateManager(["m1"], 3);
    archManager.applyVerifResult("m1", {
      ...FAIL_RESULT,
      failure_category: "architectural_fit",
      violated_resource: "BRAM18",
      classifier_reason: "BRAM18 utilization exceeds device capacity.",
    });
    expect(archManager.getState().modules.m1).toBe("fail_abort");

    const unknownManager = new PipelineStateManager(["m1"], 3);
    unknownManager.applyVerifResult("m1", {
      ...FAIL_RESULT,
      failure_category: "unknown",
      classifier_reason: "Evidence is contradictory.",
    });
    expect(unknownManager.getState().modules.m1).toBe("fail_abort");
  });

  it("loads early classifier fail-aborts even when retries remain", async () => {
    const statePath = await makeTempPath("pipeline_state.json");
    const manager = new PipelineStateManager(["m1"], 3);
    manager.applyVerifResult("m1", {
      ...FAIL_RESULT,
      failure_category: "architectural_fit",
      violated_constraint: "MAX_SUPPORTED_BUS_BITS",
      classifier_reason: "The bus contract exceeds the configured cap.",
    });
    await manager.saveState(statePath);

    const reloaded = new PipelineStateManager(["m1"], 3);
    await expect(reloaded.loadState(statePath)).resolves.toBeUndefined();
    expect(reloaded.getState().modules.m1).toBe("fail_abort");
  });

  it("merges nested model usage and total cost", () => {
    const manager = new PipelineStateManager(["m1"]);

    manager.recordAgentUsage(1.5, {
      sonnet: {
        input_tokens: 10,
        server_tool_use: { read_weights: 1 },
      },
    });
    manager.recordAgentUsage(2.5, {
      sonnet: {
        input_tokens: 5,
        output_tokens: 3,
        server_tool_use: { read_weights: 2 },
      },
    });

    expect(manager.getState().total_cost_usd).toBe(4);
    expect(manager.getState().model_usage.sonnet).toEqual({
      input_tokens: 15,
      output_tokens: 3,
      server_tool_use: { read_weights: 3 },
    });
  });

  it("records retrospector calls by module contract key", () => {
    const manager = new PipelineStateManager(["m1"]);

    expect(manager.retrospectorCallCount("m1:contract")).toBe(0);
    manager.recordRetrospectorCall("m1:contract");
    manager.recordRetrospectorCall("m1:contract");

    expect(manager.retrospectorCallCount("m1:contract")).toBe(2);
    expect(manager.getState().retrospector_calls["m1:contract"]).toBe(2);
  });

  it("returns a stable summary table", () => {
    const manager = new PipelineStateManager(["m1"]);
    manager.applyVerifResult("m1", PASS_RESULT);

    expect(manager.summary()).toContain("module_id");
    expect(manager.summary()).toContain("m1");
    expect(manager.summary()).toContain("pass");
  });

  it("saves and reloads valid state", async () => {
    const manager = new PipelineStateManager(["m1"]);
    manager.applyVerifResult("m1", PASS_RESULT);
    const filePath = await makeTempPath("state.json");

    await manager.saveState(filePath);

    const loaded = new PipelineStateManager(["placeholder"]);
    await loaded.loadState(filePath);

    expect(loaded.getState().modules.m1).toBe("pass");
    expect(loaded.getState().results.m1).toEqual(PASS_RESULT);
  });

  it("rejects corrupted state files with field-level validation errors", async () => {
    const filePath = await makeTempPath("invalid-state.json");
    await writeFile(
      filePath,
      JSON.stringify({
        run_id: "123",
        started_at: "now",
        modules: { m1: "pass" },
        attempts: {},
        results: {},
        max_retries: 3,
        total_cost_usd: 0,
        model_usage: {},
      }),
      "utf8",
    );

    const manager = new PipelineStateManager(["m1"]);
    await expect(manager.loadState(filePath)).rejects.toThrow("Corrupted pipeline state");
  });

  it("recovers transient crash states on load", async () => {
    const filePath = await makeTempPath("resume-state.json");
    const state: PipelineState = {
      run_id: "run-1",
      started_at: "2026-04-14T00:00:00Z",
      modules: {
        foundry_crash: "generating",
        surgeon_crash: "verifying",
      },
      attempts: {
        foundry_crash: 0,
        surgeon_crash: 2,
      },
      results: {
        surgeon_crash: {
          module_id: "surgeon_crash",
          status: "fail",
          failure_class: "pipeline_latency_wrong",
        },
      },
      max_retries: 3,
      total_cost_usd: 0,
      model_usage: {},
      retrospector_calls: {},
    };

    await writeFile(filePath, `${JSON.stringify(state, null, 2)}\n`, "utf8");

    const manager = new PipelineStateManager(["placeholder"]);
    await manager.loadState(filePath);

    expect(manager.getState().modules.foundry_crash).toBe("pending");
    expect(manager.getState().modules.surgeon_crash).toBe("fail_retry");
    expect(manager.getState().attempts.surgeon_crash).toBe(1);
  });

  it("writes newline-terminated JSON when saving state", async () => {
    const manager = new PipelineStateManager(["m1"]);
    const filePath = await makeTempPath("state.json");

    await manager.saveState(filePath);

    const raw = await readFile(filePath, "utf8");
    expect(raw.endsWith("\n")).toBe(true);
  });
});
