import { randomUUID } from "node:crypto";
import { mkdir, open, readFile, rename } from "node:fs/promises";
import path from "node:path";

import { pipelineStateSchema } from "./schemas.js";
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
      retrospector_calls: {},
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

    // Pure testbench/setup failures mean the RTL never actually ran, so
    // routing them to Surgeon just burns retries on a module-local repair
    // task that cannot succeed. Keep the failure visible on the module and
    // stop the retry loop instead.
    if (result.status_class === "tb_setup_error") {
      this.state.modules[module_id] = "fail_abort";
      return;
    }

    // Architectural-capability gaps (e.g. bus width exceeding
    // MAX_SUPPORTED_BUS_BITS) are not RTL bugs Surgeon can fix — they are
    // tooling limits. Report as fail_abort so the summary separates them
    // from real RTL failures.
    if (result.failure_class === "architectural_unsupported") {
      this.state.modules[module_id] = "fail_abort";
      return;
    }

    if (
      result.failure_category === "toolchain_infra" ||
      result.failure_category === "architectural_fit" ||
      result.failure_category === "unknown"
    ) {
      this.state.modules[module_id] = "fail_abort";
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

  retrospectorCallCount(contractKey: string): number {
    return this.state.retrospector_calls[contractKey] ?? 0;
  }

  recordRetrospectorCall(contractKey: string): void {
    this.state.retrospector_calls[contractKey] =
      (this.state.retrospector_calls[contractKey] ?? 0) + 1;
  }

  resetModuleForContractRetry(module_id: string): void {
    this.assertKnownModule(module_id);
    this.state.modules[module_id] = "pending";
    this.state.attempts[module_id] = 0;
    delete this.state.results[module_id];
  }

  getState(): PipelineState {
    return cloneState(this.state);
  }

  // Assert that the loaded state's module set matches a reference set. The
  // orchestrator calls this after loadState() to reject a stale
  // pipeline_state.json left over from a previous checkpoint's LayerIR. Kept
  // out of loadState() itself so unit tests can exercise transient-status
  // recovery without building a full LayerIR fixture.
  requireModuleIdsMatch(expectedModuleIds: readonly string[]): void {
    const current = new Set(expectedModuleIds);
    const loaded = new Set(Object.keys(this.state.modules));
    const missing = [...current].filter((id) => !loaded.has(id));
    const extra = [...loaded].filter((id) => !current.has(id));
    if (missing.length > 0 || extra.length > 0) {
      throw new Error(
        `pipeline_state.json module set does not match the current LayerIR. ` +
          `Missing in state: [${missing.join(", ")}]. ` +
          `Unknown in state: [${extra.join(", ")}]. ` +
          `Delete output/pipeline_state.json to start a fresh run.`,
      );
    }
  }

  async saveState(filePath: string): Promise<void> {
    // Atomic write: stage to a sibling tmp file, fsync, then rename. A crash
    // mid-write leaves either the previous valid state or the new valid state
    // on disk — never a truncated JSON blob. Required by loadState()'s
    // crash-point recovery logic.
    await mkdir(path.dirname(filePath), { recursive: true });
    const tmpPath = `${filePath}.tmp-${process.pid}-${Date.now()}`;
    const payload = `${JSON.stringify(this.state, null, 2)}\n`;
    const handle = await open(tmpPath, "w");
    try {
      await handle.writeFile(payload, "utf8");
      await handle.sync();
    } finally {
      await handle.close();
    }
    await rename(tmpPath, filePath);
  }

  async loadState(filePath: string): Promise<void> {
    const raw = await readFile(filePath, "utf8");
    const parsed: unknown = JSON.parse(raw);
    const validated = pipelineStateSchema.safeParse(parsed);

    if (!validated.success) {
      throw new Error(
        `Corrupted pipeline state at '${filePath}':\n${JSON.stringify(validated.error.issues, null, 2)}`,
      );
    }

    // Preserve the max_retries the caller passed into the constructor. The
    // CLI exposes a resume-time --max-retries override; blindly replaying the
    // saved value would silently ignore it.
    const activeMaxRetries = this.state.max_retries;
    this.state = validated.data as PipelineState;
    this.state.max_retries = activeMaxRetries;

    const loadedOrder = Object.keys(this.state.modules);
    if (loadedOrder.length > 0) {
      this.moduleOrder.splice(0, this.moduleOrder.length, ...loadedOrder);
    }

    // Transient statuses ('generating', 'verifying') mean the previous run
    // crashed mid-step. Recover to the nearest resumable status so tick()
    // can make progress, and roll back the attempts counter for Surgeon-path
    // crashes so tick()'s re-increment does not over-bill the retry budget.
    //
    // The four crash points the orchestrator can persist:
    //   generating + no prior result  -> Foundry crashed.    Resume: pending.
    //   generating + prior result     -> Surgeon crashed.    Resume: fail_retry, attempts-1.
    //   verifying  + no prior result  -> Assayer crashed after Foundry.
    //                                    Resume: pending (re-run Foundry; Assayer is not a
    //                                    first-class tick() action today).
    //   verifying  + prior result     -> Assayer crashed after Surgeon.
    //                                    Resume: fail_retry, attempts-1 (re-run Surgeon).
    for (const moduleId of this.moduleOrder) {
      const status = this.state.modules[moduleId];
      if (status !== "generating" && status !== "verifying") {
        continue;
      }

      const hasPriorResult = moduleId in this.state.results;
      if (hasPriorResult) {
        this.state.modules[moduleId] = "fail_retry";
        const attempts = this.state.attempts[moduleId] ?? 0;
        this.state.attempts[moduleId] = Math.max(0, attempts - 1);
      } else {
        this.state.modules[moduleId] = "pending";
      }
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
