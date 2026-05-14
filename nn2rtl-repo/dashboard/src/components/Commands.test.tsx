import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { Commands } from "./Commands";
import type { JobAction, LayerSummary, NetworkInfo } from "../shared/types";

const networks: NetworkInfo[] = [
  {
    id: "resnet-50",
    label: "ResNet-50",
    modelName: "resnet50",
    description: "ImageNet, INT8",
    available: true,
    defaultCheckpointPath: "checkpoints/resnet50_int8.pth",
    outputDir: "output",
  },
];

const layer = (overrides: Partial<LayerSummary>): LayerSummary => ({
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
  stage: "vivado-pass",
  hasRtl: true,
  hasMeta: true,
  hasGoldenIn: true,
  hasGoldenOut: true,
  docs: [],
  improvements: [],
  paths: {},
  ...overrides,
});

afterEach(() => cleanup());

describe("Commands task cards", () => {
  it("renders every task card with a button wired to a JobAction", () => {
    const onPreview = vi.fn((_: JobAction): void => {});
    render(
      <Commands
        modules={[layer({ moduleId: "m0", hasRtl: true }), layer({ index: 1, moduleId: "m1", hasRtl: false, stage: "missing" })]}
        networks={networks}
        networkId="resnet-50"
        onPreview={onPreview}
      />,
    );

    expect(screen.getByRole("heading", { name: /Generate RTL for a new model/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Improve a single module/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Improve sweep/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Re-synthesize/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Maintenance checks/i })).toBeInTheDocument();
  });

  it("emits a pipeline action with the network id and checkpoint", () => {
    const onPreview = vi.fn((_: JobAction): void => {});
    render(
      <Commands
        modules={[layer({ moduleId: "m0" })]}
        networks={networks}
        networkId="resnet-50"
        onPreview={onPreview}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /Run pipeline/i }));

    expect(onPreview).toHaveBeenCalledTimes(1);
    expect(onPreview.mock.calls[0][0]).toMatchObject({
      type: "pipeline",
      networkId: "resnet-50",
      checkpointPath: "checkpoints/resnet50_int8.pth",
    });
  });

  it("emits an improve-sweep --plan action with the selected preset and slider value", () => {
    const onPreview = vi.fn((_: JobAction): void => {});
    render(
      <Commands
        modules={[layer({ moduleId: "m0" }), layer({ index: 1, moduleId: "m1" })]}
        networks={networks}
        networkId="resnet-50"
        onPreview={onPreview}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /Plan sweep/i }));

    expect(onPreview).toHaveBeenCalledTimes(1);
    const action = onPreview.mock.calls[0][0];
    expect(action.type).toBe("improve-sweep");
    if (action.type !== "improve-sweep") throw new Error("not improve-sweep");
    expect(action.plan).toBe(true);
    expect(action.preset).toBe("ppa");
    expect(action.networkId).toBe("resnet-50");
  });

  it("emits a resynth-module action for the selected module", () => {
    const onPreview = vi.fn((_: JobAction): void => {});
    render(
      <Commands
        modules={[layer({ moduleId: "m0", hasRtl: true })]}
        networks={networks}
        networkId="resnet-50"
        onPreview={onPreview}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /Re-synthesize/i }));

    expect(onPreview).toHaveBeenCalledWith({
      type: "resynth-module",
      networkId: "resnet-50",
      moduleId: "m0",
    });
  });
});
