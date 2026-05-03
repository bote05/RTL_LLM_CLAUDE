import { appendFile, readFile, writeFile } from "node:fs/promises";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import path from "node:path";
import { dashboardRoot, ensureDashboardDirs, jobsDir, jobsLogPath, repoRoot, toRepoRelative } from "./paths.js";
import type { JobAction, JobPreview, JobRecord, JobState } from "../shared/types.js";

type RunningJob = {
  record: JobRecord;
  child: ChildProcessWithoutNullStreams;
  forceTimer?: NodeJS.Timeout;
};
type CheckName = Extract<JobAction, { type: "check" }>["check"];

const running = new Map<string, RunningJob>();
const expensiveQueue: JobRecord[] = [];

function nowIso(): string {
  return new Date().toISOString();
}

function jobId(): string {
  return `job_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function quoteArg(arg: string): string {
  return /[\s"'$`]/.test(arg) ? JSON.stringify(arg) : arg;
}

function commandLine(command: string, args: string[]): string {
  return [command, ...args.map(quoteArg)].join(" ");
}

function npmCommand(): string {
  return process.platform === "win32" ? "npm.cmd" : "npm";
}

function nodeCommand(): string {
  return process.execPath;
}

function tsxLoaderPath(): string {
  return path.join(dashboardRoot, "node_modules", "tsx", "dist", "loader.mjs");
}

function isWindowsAbsolutePath(value: string): boolean {
  return /^[a-zA-Z]:[\\/]/.test(value);
}

function resolveRepoInputPath(inputPath: string): string {
  const normalized = inputPath.trim();
  if (path.isAbsolute(normalized) || isWindowsAbsolutePath(normalized)) {
    return normalized;
  }
  return path.resolve(repoRoot, normalized);
}

export function buildForceKillCommand(pid: number, platform = process.platform): { command: string; args: string[] } {
  if (platform === "win32") {
    return { command: "taskkill", args: ["/PID", String(pid), "/T", "/F"] };
  }
  return { command: "kill", args: ["-KILL", `-${pid}`] };
}

function actionToCommand(action: JobAction): Omit<JobPreview, "action" | "command"> & {
  commandBin: string;
  args: string[];
} {
  switch (action.type) {
    case "pipeline": {
      const args = [
        "--import",
        tsxLoaderPath(),
        path.join(repoRoot, "sdk", "main.ts"),
        resolveRepoInputPath(action.checkpointPath),
      ];
      if (action.resume) args.push("--resume");
      if (action.maxRetries !== undefined) args.push("--max-retries", String(action.maxRetries));
      if (action.only) args.push("--only", action.only);
      if (action.except && action.except.length > 0) args.push("--except", action.except.join(","));
      return {
        title: action.only
          ? `Run pipeline for ${action.only}`
          : action.except?.length
            ? "Run pipeline with exclusions"
            : action.resume
              ? "Resume pipeline"
              : "Run pipeline",
        commandBin: nodeCommand(),
        args,
        cwd: repoRoot,
        writes: ["output/", "knowledge/"],
        costRisk: "high",
        canonicalRisk: true,
        expensive: true,
        stopWarning: "Stopping interrupts the process tree, but files already written under output/ or knowledge/ remain on disk.",
      };
    }
    case "improve": {
      const targets = [...new Set(action.targets)];
      const args = [
        "--import",
        tsxLoaderPath(),
        path.join(repoRoot, "sdk", "main.ts"),
        "improve",
        action.moduleId,
        `--targets=${targets.join(",")}`,
      ];
      if (action.keepReference !== false) args.push("--keep-reference");
      return {
        title: `Improve ${action.moduleId}`,
        commandBin: nodeCommand(),
        args,
        cwd: repoRoot,
        writes: action.keepReference === false
          ? ["output/improve/", "output/reports/", "output/rtl/", "output/rtl/archive/"]
          : ["output/improve/", "output/reports/", "knowledge/references/improved/", "knowledge/patterns/improved/"],
        costRisk: "high",
        canonicalRisk: action.keepReference === false,
        expensive: true,
        stopWarning: "Stopping interrupts Foundry/Vivado if they are running, but any completed attempt artifacts remain on disk.",
      };
    }
    case "promote-variant":
      return {
        title: `Promote ${action.moduleId} ${action.targetSlug}`,
        commandBin: nodeCommand(),
        args: [
          "--import",
          tsxLoaderPath(),
          path.join(repoRoot, "scripts", "promote_variant.ts"),
          action.moduleId,
          action.targetSlug,
        ],
        cwd: repoRoot,
        writes: ["output/rtl/", "output/rtl/archive/", "output/reports/", "output/reports/archive/"],
        costRisk: "none",
        canonicalRisk: true,
        expensive: false,
        stopWarning: "Stopping cannot undo any archive or canonical files already written by the promotion script.",
      };
    case "check": {
      const checks: Record<CheckName, { title: string; commandBin: string; args: string[] }> = {
        twins: { title: "Run SDK/MCP twin check", commandBin: npmCommand(), args: ["run", "check:twins"] },
        "sdk-typecheck": { title: "Run SDK typecheck", commandBin: npmCommand(), args: ["--prefix", "sdk", "run", "typecheck"] },
        "mcp-typecheck": { title: "Run MCP typecheck", commandBin: npmCommand(), args: ["--prefix", "mcp", "run", "typecheck"] },
        "dashboard-typecheck": { title: "Run dashboard typecheck", commandBin: npmCommand(), args: ["--prefix", "dashboard", "run", "typecheck"] },
        "dashboard-test": { title: "Run dashboard tests", commandBin: npmCommand(), args: ["--prefix", "dashboard", "run", "test"] },
        "improve-test": { title: "Run improve tests", commandBin: npmCommand(), args: ["--prefix", "sdk", "exec", "vitest", "run", "test/improve.test.ts"] },
      };
      const check = checks[action.check];
      return {
        title: check.title,
        commandBin: check.commandBin,
        args: check.args,
        cwd: repoRoot,
        writes: ["node_modules/.cache/", "coverage/"],
        costRisk: "none",
        canonicalRisk: false,
        expensive: false,
        stopWarning: "Stopping a check only stops the check process; no generated RTL or knowledge files are changed.",
      };
    }
  }
}

export function previewJob(action: JobAction): JobPreview {
  const plan = actionToCommand(action);
  return {
    action,
    title: plan.title,
    command: commandLine(plan.commandBin, plan.args),
    cwd: plan.cwd,
    writes: plan.writes,
    costRisk: plan.costRisk,
    canonicalRisk: plan.canonicalRisk,
    expensive: plan.expensive,
    stopWarning: plan.stopWarning,
  };
}

async function appendJob(record: JobRecord): Promise<void> {
  await ensureDashboardDirs();
  await appendFile(jobsLogPath, `${JSON.stringify(record)}\n`, "utf8");
}

async function readJobsRaw(): Promise<JobRecord[]> {
  try {
    const lines = (await readFile(jobsLogPath, "utf8")).split(/\r?\n/).filter(Boolean);
    const byId = new Map<string, JobRecord>();
    for (const line of lines) {
      try {
        const record = JSON.parse(line) as JobRecord;
        byId.set(record.id, record);
      } catch {
        // Ignore partial lines from interrupted writes.
      }
    }
    return [...byId.values()].sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  } catch {
    return [];
  }
}

export async function readJobs(): Promise<JobRecord[]> {
  return readJobsRaw();
}

export async function reconcilePersistedJobsAfterRestart(): Promise<void> {
  const jobs = await readJobsRaw();
  const endedAt = nowIso();
  for (const job of jobs) {
    if (running.has(job.id)) continue;
    if (job.state === "queued" && expensiveQueue.some((queued) => queued.id === job.id)) continue;

    if (job.state === "queued") {
      await appendJob({
        ...job,
        state: "stopped",
        endedAt,
        stopRequestedAt: job.stopRequestedAt ?? endedAt,
        stopReason: job.stopReason ?? "dashboard restarted before queued job launched",
      });
    } else if (job.state === "running" || job.state === "stopping") {
      await appendJob({
        ...job,
        state: "failed",
        endedAt,
        exitCode: job.exitCode ?? null,
        stopReason: job.stopReason ?? "dashboard restarted while job was running; process ownership was lost",
      });
    }
  }
}

function canStartExpensive(): boolean {
  return ![...running.values()].some((job) => job.record.expensive);
}

async function updateJob(record: JobRecord, patch: Partial<JobRecord>): Promise<JobRecord> {
  Object.assign(record, patch);
  await appendJob(record);
  return record;
}

async function startRecord(record: JobRecord): Promise<void> {
  await ensureDashboardDirs();
  const plan = actionToCommand(record.action);
  await writeFile(path.join(repoRoot, record.logPath), `$ ${record.command}\n\n`, "utf8");
  await updateJob(record, { state: "running", startedAt: nowIso() });
  const child = spawn(plan.commandBin, plan.args, {
    cwd: plan.cwd,
    env: process.env,
    detached: process.platform !== "win32",
    shell: process.platform === "win32",
  });
  record.pid = child.pid;
  await appendJob(record);
  const runningJob: RunningJob = { record, child };
  running.set(record.id, runningJob);

  const appendLog = (chunk: Buffer): void => {
    void appendFile(path.join(repoRoot, record.logPath), chunk, "utf8");
  };
  child.stdout.on("data", appendLog);
  child.stderr.on("data", appendLog);
  child.on("error", (error) => {
    void appendFile(path.join(repoRoot, record.logPath), `\n[dashboard] ${error.message}\n`, "utf8");
  });
  child.on("close", (code) => {
    if (runningJob.forceTimer) clearTimeout(runningJob.forceTimer);
    running.delete(record.id);
    const stopped = record.state === "stopping";
    void updateJob(record, {
      state: stopped ? "stopped" : code === 0 ? "succeeded" : "failed",
      endedAt: nowIso(),
      exitCode: code,
    }).then(processQueue);
  });
}

function processQueue(): void {
  if (!canStartExpensive()) return;
  const next = expensiveQueue.shift();
  if (next) void startRecord(next);
}

export async function startJob(action: JobAction, confirmed: boolean): Promise<JobRecord> {
  if (!confirmed) {
    throw new Error("Job start requires confirmed=true after showing the confirmation popup.");
  }
  const preview = previewJob(action);
  const id = jobId();
  const record: JobRecord = {
    ...preview,
    id,
    state: preview.expensive && !canStartExpensive() ? "queued" : "running",
    createdAt: nowIso(),
    logPath: toRepoRelative(path.join(jobsDir, `${id}.log`)),
  };
  await appendJob(record);
  if (record.state === "queued") {
    expensiveQueue.push(record);
  } else {
    await startRecord(record);
  }
  return record;
}

export async function stopJob(id: string): Promise<JobRecord> {
  const queuedIndex = expensiveQueue.findIndex((job) => job.id === id);
  if (queuedIndex >= 0) {
    const [record] = expensiveQueue.splice(queuedIndex, 1);
    return updateJob(record, {
      state: "stopped",
      stopRequestedAt: nowIso(),
      stopReason: "queued job canceled before launch",
      endedAt: nowIso(),
    });
  }
  const job = running.get(id);
  if (!job) {
    const existing = (await readJobs()).find((record) => record.id === id);
    if (!existing) throw new Error(`Unknown job '${id}'.`);
    return existing;
  }
  await updateJob(job.record, {
    state: "stopping",
    stopRequestedAt: nowIso(),
    stopReason: "stop requested from dashboard",
  });
  job.child.kill("SIGINT");
  job.forceTimer = setTimeout(() => {
    if (!job.child.pid) return;
    const killPlan = buildForceKillCommand(job.child.pid);
    const killer = spawn(killPlan.command, killPlan.args, {
      cwd: repoRoot,
      shell: process.platform === "win32",
    });
    killer.on("error", () => {
      try {
        if (process.platform !== "win32") process.kill(-job.child.pid!, "SIGKILL");
      } catch {
        // Process may already be gone.
      }
    });
  }, 5000);
  return job.record;
}
