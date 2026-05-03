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

import { buildForceKillCommand, previewJob, readJobs, startJob, stopJob } from "../jobs.js";

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
});
