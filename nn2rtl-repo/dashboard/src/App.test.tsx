import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import type { JobPreview, ProjectSnapshot } from "./shared/types";

const snapshot: ProjectSnapshot = {
  generatedAt: "2026-05-03T00:00:00Z",
  repoRoot: "/repo",
  modelName: "fixture-net",
  quantization: "int8_symmetric_per_tensor",
  kpis: {
    totalLayers: 2,
    rtlGenerated: 1,
    verilatorPass: 1,
    vivadoPass: 1,
    failedOrUnknown: 1,
    improvedVariants: 1,
    docsProtected: 2,
    docsActive: 0,
    docsProbationary: 0,
    docsImproved: 1,
    knownCostUsd: 3.14,
  },
  modules: [
    {
      index: 0,
      moduleId: "m0",
      opType: "conv2d",
      contractId: "flat-bus",
      ioMode: "default",
      inputShape: [1],
      outputShape: [1],
      weightShape: [1],
      numWeights: 1,
      pipelineLatencyCycles: 1,
      stage: "improved",
      hasRtl: true,
      hasMeta: true,
      hasGoldenIn: true,
      hasGoldenOut: true,
      verif: { status: "pass" },
      vivado: { success: true, timingMet: true, lut: 10, dsp: 8, bram: 1 },
      docs: [],
      improvements: [
        {
          moduleId: "m0",
          targetSlug: "use-dsp",
          targets: ["use-dsp"],
          success: true,
          finalAction: "kept-as-variant",
          reportPath: "output/reports/improve_m0__use-dsp.json",
          attempts: [],
        },
      ],
      paths: { rtl: "output/rtl/m0.v" },
    },
    {
      index: 1,
      moduleId: "m1",
      opType: "relu",
      contractId: "none",
      ioMode: "default",
      inputShape: [1],
      outputShape: [1],
      weightShape: [1],
      numWeights: 1,
      pipelineLatencyCycles: 1,
      stage: "missing",
      hasRtl: false,
      hasMeta: false,
      hasGoldenIn: false,
      hasGoldenOut: false,
      docs: [],
      improvements: [],
      paths: {},
    },
  ],
  docs: [],
  improvements: [],
  improveRuns: [],
  jobs: [],
  orphanArtifacts: { rtl: [], reports: [] },
};

const preview: JobPreview = {
  action: { type: "check", check: "twins" },
  title: "Run SDK/MCP twin check",
  command: "npm run check:twins",
  cwd: "/repo",
  writes: ["node_modules/.cache/"],
  costRisk: "none",
  canonicalRisk: false,
  expensive: false,
  stopWarning: "Stopping a check only stops the check process.",
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function mockFetch(): void {
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url === "/api/snapshot") {
      return new Response(JSON.stringify(snapshot), { status: 200 });
    }
    if (url === "/api/jobs/preview") {
      return new Response(JSON.stringify(preview), { status: 200 });
    }
    return new Response(JSON.stringify({ id: "job_1", ...preview, state: "running", createdAt: "now", logPath: "output/dashboard/jobs/job_1.log" }), { status: 201 });
  }));
}

describe("App", () => {
  it("renders the overview skyline by default and switches to heatmap on demand", async () => {
    mockFetch();
    render(<App />);

    await screen.findByText(/fixture-net/);
    // Default view: skyline. Switch to heatmap to assert filter behavior on cells.
    expect(screen.getByLabelText("network coverage skyline")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Heatmap" }));
    expect(screen.getByLabelText("module status heatmap")).toBeInTheDocument();
    fireEvent.change(screen.getByPlaceholderText("module"), { target: { value: "m1" } });
    expect(screen.getByTitle("2. m1 · missing")).toBeInTheDocument();
  });

  it("shows confirmation metadata before launching a command", async () => {
    mockFetch();
    render(<App />);

    await screen.findByText(/fixture-net/);
    fireEvent.click(screen.getByRole("button", { name: /commands/i }));
    await waitFor(() => expect(screen.getByText("Command Center")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "twins" }));

    expect(await screen.findByText("Run SDK/MCP twin check")).toBeInTheDocument();
    expect(screen.getByText("npm run check:twins")).toBeInTheDocument();
    expect(screen.getByText("Canonical RTL risk")).toBeInTheDocument();
  });
});
