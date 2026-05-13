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
  | "reduce-ff"
  | "improve-fmax"
  | "reduce-latency"
  | "increase-throughput";

export type ImproveSweepPreset =
  | "ppa"
  | "use-dsp"
  | "reduce-lut"
  | "reduce-ff"
  | "improve-fmax";

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
    ff?: number;
    dsp?: number;
    bram?: number;
    fmax_mhz?: number;
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

export type JobAction =
  | {
      type: "pipeline";
      checkpointPath: string;
      resume?: boolean;
      maxRetries?: number;
      only?: string;
      except?: string[];
    }
  | {
      type: "improve";
      moduleId: string;
      targets: ImprovementTarget[];
      keepReference?: boolean;
    }
  | {
      type: "improve-sweep";
      preset?: ImproveSweepPreset;
      run?: boolean;
      keepReference?: boolean;
      maxModules?: number;
    }
  | {
      type: "promote-variant";
      moduleId: string;
      targetSlug: string;
    }
  | {
      type: "check";
      check: "twins" | "sdk-typecheck" | "mcp-typecheck" | "dashboard-typecheck" | "dashboard-test" | "improve-test";
    };

export type ProjectSnapshot = {
  generatedAt: string;
  repoRoot: string;
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
