import { appendFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { describe, expect, it, vi } from "vitest";

const childProcessMock = vi.hoisted(() => {
  function emitter() {
    const handlers = new Map<string, Array<(...args: unknown[]) => void>>();
    return {
      on(event: string, handler: (...args: unknown[]) => void) {
        handlers.set(event, [...(handlers.get(event) ?? []), handler]);
        return this;
      },
      emit(event: string, ...args: unknown[]) {
        for (const handler of handlers.get(event) ?? []) handler(...args);
      },
    };
  }

  const spawned: Array<ReturnType<typeof emitter> & {
    pid: number;
    stdout: ReturnType<typeof emitter>;
    stderr: ReturnType<typeof emitter>;
    kill: ReturnType<typeof vi.fn>;
  }> = [];

  const spawn = vi.fn(() => {
    const child = {
      ...emitter(),
      pid: 4321,
      stdout: emitter(),
      stderr: emitter(),
      kill: vi.fn(),
    };
    spawned.push(child);
    return child;
  });

  return { spawn, spawned };
});

vi.mock("node:child_process", () => ({
  default: { spawn: childProcessMock.spawn },
  spawn: childProcessMock.spawn,
}));

import { jobsLogPath } from "../paths.js";
import {
  buildForceKillCommand,
  previewJob,
  readJobs,
  reconcilePersistedJobsAfterRestart,
  startJob,
  stopJob,
} from "../jobs.js";

describe("job safety model", () => {
  it("transitions a running job through stopping and records the reason", async () => {
    const record = await startJob({ type: "check", check: "dashboard-typecheck" }, true);
    const child = childProcessMock.spawned.at(-1);
    expect(child).toBeDefined();

    const stopped = await stopJob(record.id);

    expect(stopped.state).toBe("stopping");
    expect(stopped.stopReason).toBe("stop requested from dashboard");
    expect(child?.kill).toHaveBeenCalledWith("SIGINT");

    child?.emit("close", null);
    await vi.waitFor(async () => {
      const jobs = await readJobs();
      expect(jobs.find((job) => job.id === record.id)?.state).toBe("stopped");
    });
  });

  it("builds command previews for allowlisted actions", () => {
    const preview = previewJob({
      type: "improve",
      moduleId: "node_conv_248",
      targets: ["use-dsp", "use-bram"],
      keepReference: true,
    });

    expect(preview.command).toContain("improve node_conv_248");
    expect(preview.command).toContain("--keep-reference");
    expect(preview.costRisk).toBe("high");
    expect(preview.canonicalRisk).toBe(false);
  });

  it("builds an improve-sweep preview in --plan mode that costs nothing", () => {
    const preview = previewJob({
      type: "improve-sweep",
      preset: "ppa",
      plan: true,
      maxModules: 5,
    });

    expect(preview.command).toMatch(/scripts[\\/]+improve_sweep\.ts/);
    expect(preview.command).toContain("--preset=ppa");
    expect(preview.command).toContain("--targets=use-dsp,reduce-lut,reduce-latency");
    expect(preview.command).toContain("--plan");
    expect(preview.command).toContain("--max-modules=5");
    expect(preview.costRisk).toBe("none");
    expect(preview.canonicalRisk).toBe(false);
    expect(preview.expensive).toBe(false);
  });

  it("builds an improve-sweep preview in --run mode flagged as expensive", () => {
    const preview = previewJob({
      type: "improve-sweep",
      preset: "use-dsp",
      plan: false,
      keepReference: true,
    });

    expect(preview.command).toContain("--run");
    expect(preview.command).toContain("--keep-reference");
    expect(preview.costRisk).toBe("high");
    expect(preview.expensive).toBe(true);
    expect(preview.canonicalRisk).toBe(false);
  });

  it("builds a resynth-module preview that spawns the wrapper script", () => {
    const preview = previewJob({
      type: "resynth-module",
      moduleId: "node_conv_42",
    });

    expect(preview.command).toMatch(/scripts[\\/]+vivado_resynth_module\.ts/);
    expect(preview.command).toContain("node_conv_42");
    expect(preview.command).toContain("--network=resnet-50");
    expect(preview.costRisk).toBe("none");
    expect(preview.canonicalRisk).toBe(false);
  });

  it("requires confirmation before starting any job", async () => {
    await expect(startJob({ type: "check", check: "twins" }, false)).rejects.toThrow("confirmed=true");
  });

  it("knows how to force-kill process trees on Windows and POSIX", () => {
    expect(buildForceKillCommand(123, "win32")).toEqual({
      command: "taskkill",
      args: ["/PID", "123", "/T", "/F"],
    });
    expect(buildForceKillCommand(123, "linux")).toEqual({
      command: "kill",
      args: ["-KILL", "-123"],
    });
  });

  it("marks persisted queued/running jobs stale after dashboard restart", async () => {
    const createdAt = new Date().toISOString();
    const queuedId = `stale_queued_${Date.now()}`;
    const runningId = `stale_running_${Date.now()}`;
    await mkdir(path.dirname(jobsLogPath), { recursive: true });
    await appendFile(
      jobsLogPath,
      [
        JSON.stringify({
          ...previewJob({ type: "pipeline", checkpointPath: "checkpoint.pth" }),
          id: queuedId,
          state: "queued",
          createdAt,
          logPath: `output/dashboard/jobs/${queuedId}.log`,
        }),
        JSON.stringify({
          ...previewJob({ type: "check", check: "dashboard-typecheck" }),
          id: runningId,
          state: "running",
          createdAt,
          logPath: `output/dashboard/jobs/${runningId}.log`,
        }),
      ].join("\n") + "\n",
      "utf8",
    );

    await reconcilePersistedJobsAfterRestart();

    const jobs = await readJobs();
    expect(jobs.find((job) => job.id === queuedId)).toMatchObject({
      state: "stopped",
      stopReason: "dashboard restarted before queued job launched",
    });
    expect(jobs.find((job) => job.id === runningId)).toMatchObject({
      state: "failed",
      stopReason: "dashboard restarted while job was running; process ownership was lost",
      exitCode: null,
    });
  });
});
