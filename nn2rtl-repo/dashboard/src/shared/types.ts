export type ModuleStage =
  | "missing"
  | "rtl"
  | "verilator-pass"
  | "vivado-pass"
  | "failed"
  | "improved";

export type JobState =
  | "queued"
  | "running"
  | "stopping"
  | "stopped"
  | "succeeded"
  | "failed";

export type ImprovementTarget =
  | "use-dsp"
  | "use-bram"
  | "reduce-lut"
  | "reduce-latency"
  | "increase-throughput";

/** Improve-sweep presets — surfaced by the dashboard "Improve sweep" card.
 *  Each preset maps to a fixed ordered target list; the sweep wrapper runs
 *  every selected module through those targets one at a time. */
export type ImproveSweepPreset =
  | "ppa"
  | "ppa-no-dsp"
  | "use-dsp"
  | "reduce-lut"
  | "reduce-latency"
  | "increase-throughput";

export type ImproveSweepPresetSpec = {
  id: ImproveSweepPreset;
  label: string;
  description: string;
  targets: ImprovementTarget[];
};

export const IMPROVE_SWEEP_PRESETS: readonly ImproveSweepPresetSpec[] = [
  {
    id: "ppa",
    label: "PPA (balanced)",
    description: "Apply use-dsp, reduce-lut and reduce-latency in that order.",
    targets: ["use-dsp", "reduce-lut", "reduce-latency"],
  },
  {
    id: "ppa-no-dsp",
    label: "PPA without DSP",
    description: "Balanced PPA but avoids forcing DSP usage (good for DSP-tight devices).",
    targets: ["reduce-lut", "reduce-latency"],
  },
  {
    id: "use-dsp",
    label: "Maximize DSP usage",
    description: "Push every module towards multipliers in DSP blocks.",
    targets: ["use-dsp"],
  },
  {
    id: "reduce-lut",
    label: "Reduce LUTs",
    description: "Trim look-up-table count across the network.",
    targets: ["reduce-lut"],
  },
  {
    id: "reduce-latency",
    label: "Reduce latency",
    description: "Shrink end-to-end cycle counts.",
    targets: ["reduce-latency"],
  },
  {
    id: "increase-throughput",
    label: "Increase throughput",
    description: "Lower the initiation interval (more samples/second).",
    targets: ["increase-throughput"],
  },
];

export type DashboardKpis = {
  totalLayers: number;
  rtlGenerated: number;
  verilatorPass: number;
  vivadoPass: number;
  failedOrUnknown: number;
  improvedVariants: number;
  docsProtected: number;
  docsActive: number;
  docsProbationary: number;
  docsImproved: number;
  knownCostUsd: number;
};

export type LayerSummary = {
  index: number;
  moduleId: string;
  opType: string;
  contractId: string;
  ioMode: string;
  inputShape: number[];
  outputShape: number[];
  weightShape: number[];
  numWeights: number;
  pipelineLatencyCycles: number;
  stage: ModuleStage;
  hasRtl: boolean;
  hasMeta: boolean;
  hasGoldenIn: boolean;
  hasGoldenOut: boolean;
  pipelineStatus?: string;
  pipelineAttempts?: number;
  verif?: {
    status?: string;
    timingPass?: boolean;
    timingActualCycles?: number;
    timingExpectedCycles?: number;
    maxError?: number;
    meanError?: number;
  };
  vivado?: {
    success?: boolean;
    timingMet?: boolean;
    lut?: number;
    ff?: number;
    dsp?: number;
    bram?: number;
    fmaxMhz?: number;
    setupWnsNs?: number | null;
    holdWnsNs?: number | null;
  };
  docs: DocSummary[];
  improvements: ImprovementReportSummary[];
  paths: {
    rtl?: string;
    meta?: string;
    verif?: string;
    vivado?: string;
  };
};

export type DocTier =
  | "protected"
  | "active"
  | "probationary"
  | "improved"
  | "archive";

export type DocSummary = {
  id: string;
  tier: DocTier;
  status: string;
  opType?: string;
  contractId?: string;
  moduleId?: string;
  patternPath?: string;
  referencePath?: string;
  createdAt?: string;
  createdByAgent?: string;
  improvementTargets?: string[];
  successfulModules?: string[];
  failedModules?: string[];
  usedByModules?: string[];
};

export type ImprovementAttemptSummary = {
  attemptIndex: number;
  failedGate?: string | null;
  metrics?: {
    lut?: number;
    dsp?: number;
    bram?: number;
    latency_cycles?: number;
    ii?: number;
  };
  vivado?: LayerSummary["vivado"];
  verifStatus?: string;
  verdictOverall?: boolean;
  artifactPaths: string[];
};

export type ImprovementReportSummary = {
  moduleId: string;
  targetSlug: string;
  targets: ImprovementTarget[];
  success: boolean;
  finalAction: string;
  reportPath: string;
  improvedReferencePath?: string;
  committedModulePath?: string;
  costUsd?: number;
  attempts: ImprovementAttemptSummary[];
  finalVerdict?: {
    overall: boolean;
    targets: Array<{
      target: string;
      satisfied: boolean;
      reason: string;
      required?: string;
      baseline_value?: number;
      new_value?: number;
    }>;
  };
};

export type ImproveRunSummary = {
  moduleId: string;
  runId: string;
  attemptCount: number;
  artifactCount: number;
  artifactPaths: string[];
};

export type JobPreview = {
  action: JobAction;
  title: string;
  command: string;
  cwd: string;
  writes: string[];
  costRisk: "none" | "low" | "high";
  canonicalRisk: boolean;
  expensive: boolean;
  stopWarning: string;
};

export type JobRecord = JobPreview & {
  id: string;
  state: JobState;
  createdAt: string;
  startedAt?: string;
  endedAt?: string;
  exitCode?: number | null;
  pid?: number;
  logPath: string;
  stopRequestedAt?: string;
  stopReason?: string;
  error?: string;
};

import type { NetworkId } from "./networks.js";

export type JobAction =
  | {
      type: "pipeline";
      networkId?: NetworkId;
      checkpointPath: string;
      resume?: boolean;
      maxRetries?: number;
      only?: string;
      except?: string[];
    }
  | {
      type: "improve";
      networkId?: NetworkId;
      moduleId: string;
      targets: ImprovementTarget[];
      keepReference?: boolean;
    }
  | {
      type: "improve-sweep";
      networkId?: NetworkId;
      preset: ImproveSweepPreset;
      /** If true, only print the plan — no Claude calls, no Vivado, no money spent. */
      plan: boolean;
      /** Cap on how many modules to sweep through (UI defaults to 17 for ResNet-50). */
      maxModules?: number;
      /** When `plan=false`, default true: keep the reference instead of overwriting canonical RTL. */
      keepReference?: boolean;
    }
  | {
      type: "resynth-module";
      networkId?: NetworkId;
      moduleId: string;
    }
  | {
      type: "promote-variant";
      networkId?: NetworkId;
      moduleId: string;
      targetSlug: string;
    }
  | {
      type: "check";
      check: "twins" | "sdk-typecheck" | "mcp-typecheck" | "dashboard-typecheck" | "dashboard-test" | "improve-test";
    };

export type NetworkInfo = {
  id: NetworkId;
  label: string;
  modelName: string;
  description: string;
  available: boolean;
  defaultCheckpointPath: string;
  /** Repo-relative output directory used to read snapshot artifacts. */
  outputDir: string;
};

export type ProjectSnapshot = {
  generatedAt: string;
  repoRoot: string;
  /** The network id this snapshot was built for. */
  networkId: NetworkId;
  /** Registry of all networks the dashboard knows about (so the UI shows the selector). */
  networks: NetworkInfo[];
  modelName: string;
  quantization?: string;
  kpis: DashboardKpis;
  modules: LayerSummary[];
  docs: DocSummary[];
  improvements: ImprovementReportSummary[];
  improveRuns: ImproveRunSummary[];
  jobs: JobRecord[];
  latestPipeline?: {
    runId?: string;
    startedAt?: string;
    isDone?: boolean;
    totalCostUsd?: number;
    stateCounts: Record<string, number>;
  };
  orphanArtifacts: {
    rtl: string[];
    reports: string[];
  };
};

export type FileReadResult = {
  path: string;
  content: string;
  sizeBytes: number;
  truncated: boolean;
};
