import { randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

import type {
  ModelUsageEntry,
  ModuleStatus,
  NextAction,
  PipelineState,
  VerifResult,
} from "./types.js";

function cloneState(state: PipelineState): PipelineState {
  return JSON.parse(JSON.stringify(state)) as PipelineState;
}

function mergeUsageEntry(
  existing: Record<string, unknown>,
  incoming: Record<string, unknown>,
): Record<string, unknown> {
  const merged: Record<string, unknown> = { ...existing };

  for (const [key, value] of Object.entries(incoming)) {
    const current = merged[key];

    if (typeof current === "number" && typeof value === "number") {
      merged[key] = current + value;
      continue;
    }

    if (
      current !== null &&
      value !== null &&
      typeof current === "object" &&
      typeof value === "object" &&
      !Array.isArray(current) &&
      !Array.isArray(value)
    ) {
      merged[key] = mergeUsageEntry(
        current as Record<string, unknown>,
        value as Record<string, unknown>,
      );
      continue;
    }

    merged[key] = value;
  }

  return merged;
}

export class PipelineStateManager {
  private readonly moduleOrder: string[];
  private state: PipelineState;

  constructor(moduleIds: string[], maxRetries = 3) {
    this.moduleOrder = [...new Set(moduleIds)];
    this.state = {
      run_id: randomUUID(),
      started_at: new Date().toISOString(),
      modules: Object.fromEntries(
        this.moduleOrder.map((moduleId) => [moduleId, "pending" as ModuleStatus]),
      ),
      attempts: Object.fromEntries(this.moduleOrder.map((moduleId) => [moduleId, 0])),
      results: {},
      max_retries: maxRetries,
      total_cost_usd: 0,
      model_usage: {},
    };
  }

  tick(): NextAction {
    for (const moduleId of this.moduleOrder) {
      const status = this.state.modules[moduleId];

      if (status === "pending") {
        this.state.modules[moduleId] = "generating";
        return { action: "invoke_foundry", module_id: moduleId };
      }

      if (status === "fail_retry") {
        this.state.modules[moduleId] = "generating";
        this.state.attempts[moduleId] = (this.state.attempts[moduleId] ?? 0) + 1;
        return { action: "invoke_surgeon", module_id: moduleId };
      }
    }

    if (this.isDone()) {
      return { action: "done" };
    }

    throw new Error(
      "PipelineStateManager.tick() found no actionable modules, but the pipeline is not in a terminal state.",
    );
  }

  applyVerifResult(module_id: string, result: VerifResult): void {
    this.assertKnownModule(module_id);
    this.state.results[module_id] = JSON.parse(JSON.stringify(result)) as VerifResult;

    if (result.status === "pass") {
      this.state.modules[module_id] = "pass";
      return;
    }

    const attempts = this.state.attempts[module_id] ?? 0;
    this.state.modules[module_id] =
      attempts < this.state.max_retries ? "fail_retry" : "fail_abort";
  }

  setStatus(module_id: string, status: ModuleStatus): void {
    this.assertKnownModule(module_id);
    this.state.modules[module_id] = status;
  }

  recordAgentUsage(totalCostUsd: number, modelUsage: Record<string, ModelUsageEntry>): void {
    this.state.total_cost_usd += totalCostUsd;

    for (const [modelName, usage] of Object.entries(modelUsage)) {
      const existing = this.state.model_usage[modelName] ?? {};
      this.state.model_usage[modelName] = mergeUsageEntry(
        existing as Record<string, unknown>,
        usage as Record<string, unknown>,
      ) as ModelUsageEntry;
    }
  }

  getState(): PipelineState {
    return cloneState(this.state);
  }

  async saveState(filePath: string): Promise<void> {
    await mkdir(path.dirname(filePath), { recursive: true });
    await writeFile(filePath, `${JSON.stringify(this.state, null, 2)}\n`, "utf8");
  }

  async loadState(filePath: string): Promise<void> {
    const raw = await readFile(filePath, "utf8");
    const parsed = JSON.parse(raw) as PipelineState;

    // TODO: Add schema validation here so corrupted resume files fail with clear, field-level errors before mutating in-memory state.
    this.state = parsed;

    const loadedOrder = Object.keys(parsed.modules);
    if (loadedOrder.length > 0) {
      this.moduleOrder.splice(0, this.moduleOrder.length, ...loadedOrder);
    }
  }

  isDone(): boolean {
    return this.moduleOrder.every((moduleId) => {
      const status = this.state.modules[moduleId];
      return status === "pass" || status === "fail_abort";
    });
  }

  summary(): string {
    const rows = this.moduleOrder.map((moduleId) => ({
      module_id: moduleId,
      status: this.state.modules[moduleId],
      attempts: String(this.state.attempts[moduleId] ?? 0),
    }));

    const headers = {
      module_id: "module_id",
      status: "status",
      attempts: "attempts",
    };

    const widths = {
      module_id: Math.max(headers.module_id.length, ...rows.map((row) => row.module_id.length)),
      status: Math.max(headers.status.length, ...rows.map((row) => row.status.length)),
      attempts: Math.max(headers.attempts.length, ...rows.map((row) => row.attempts.length)),
    };

    const formatRow = (left: string, middle: string, right: string): string =>
      `${left.padEnd(widths.module_id)} | ${middle.padEnd(widths.status)} | ${right.padEnd(widths.attempts)}`;

    const divider = `${"-".repeat(widths.module_id)}-+-${"-".repeat(widths.status)}-+-${"-".repeat(widths.attempts)}`;

    return [
      formatRow(headers.module_id, headers.status, headers.attempts),
      divider,
      ...rows.map((row) => formatRow(row.module_id, row.status, row.attempts)),
    ].join("\n");
  }

  private assertKnownModule(moduleId: string): void {
    if (!this.moduleOrder.includes(moduleId)) {
      throw new Error(`Unknown module_id '${moduleId}' in PipelineStateManager.`);
    }
  }
}
