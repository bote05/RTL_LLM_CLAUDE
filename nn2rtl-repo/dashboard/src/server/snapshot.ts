import path from "node:path";
import {
  listFilesRecursive,
  outputDirFor,
  pathExists,
  readJson,
  repoRoot,
} from "./paths.js";
import { readJobs } from "./jobs.js";
import {
  DEFAULT_NETWORK_ID,
  NETWORKS,
  getNetwork,
  type NetworkId,
} from "../shared/networks.js";
import type {
  DashboardKpis,
  DocSummary,
  DocTier,
  ImproveRunSummary,
  ImprovementReportSummary,
  LayerSummary,
  ModuleStage,
  NetworkInfo,
  ProjectSnapshot,
} from "../shared/types.js";

type JsonRecord = Record<string, unknown>;

function asRecord(value: unknown): JsonRecord {
  return typeof value === "object" && value !== null ? value as JsonRecord : {};
}

function asArray<T = unknown>(value: unknown): T[] {
  return Array.isArray(value) ? value as T[] : [];
}

function asNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function toRelative(root: string, filePath: string): string {
  return path.relative(root, filePath).split(path.sep).join("/");
}

function reportPath(outputBase: string, moduleId: string, suffix: string): string {
  return path.join(outputBase, "reports", `${moduleId}${suffix}`);
}

function rtlPath(outputBase: string, moduleId: string, suffix: string): string {
  return path.join(outputBase, "rtl", `${moduleId}${suffix}`);
}

function stageFor(input: {
  hasRtl: boolean;
  verifStatus?: string;
  vivadoSuccess?: boolean;
  vivadoTimingMet?: boolean;
  hasSuccessfulImprovement: boolean;
}): ModuleStage {
  if (input.hasSuccessfulImprovement) return "improved";
  if (input.vivadoSuccess && input.vivadoTimingMet) return "vivado-pass";
  if (input.verifStatus === "pass") return "verilator-pass";
  if (input.verifStatus === "fail" || input.verifStatus === "syntax_error" || input.vivadoSuccess === false) {
    return "failed";
  }
  if (input.hasRtl) return "rtl";
  return "missing";
}

function parseTargetSlug(reportPathRel: string, moduleId: string): string {
  const file = path.basename(reportPathRel, ".json");
  const prefix = `improve_${moduleId}__`;
  return file.startsWith(prefix) ? file.slice(prefix.length) : file.replace(/^improve_/, "");
}

function lifecycleTier(status: string | undefined): DocTier {
  if (status === "archived") return "archive";
  if (
    status === "protected" ||
    status === "active" ||
    status === "probationary" ||
    status === "improved" ||
    status === "archive"
  ) {
    return status;
  }
  return "probationary";
}

async function loadDocs(root: string): Promise<DocSummary[]> {
  const lifecycle = asRecord(await readJson(path.join(root, "knowledge", "doc_lifecycle.json")));
  const docs: DocSummary[] = [];
  for (const [id, raw] of Object.entries(asRecord(lifecycle.docs))) {
    const doc = asRecord(raw);
    const status = asString(doc.status) ?? "probationary";
    docs.push({
      id,
      tier: lifecycleTier(status),
      status,
      opType: asString(doc.op_type),
      contractId: asString(doc.contract_id),
      moduleId: asString(doc.created_by_module),
      patternPath: asString(doc.archived_pattern_path) ?? asString(doc.pattern_path),
      referencePath: asString(doc.archived_reference_path) ?? asString(doc.reference_path),
      createdAt: asString(doc.created_at),
      createdByAgent: asString(doc.created_by_agent),
      improvementTargets: asArray<string>(doc.improvement_targets),
      successfulModules: asArray<string>(doc.successful_modules),
      failedModules: asArray<string>(doc.failed_modules),
      usedByModules: asArray<string>(doc.used_by_modules),
    });
  }

  for (const tier of ["protected", "active", "probationary", "improved", "archive"] as DocTier[]) {
    for (const kind of ["patterns", "references"] as const) {
      const files = await listFilesRecursive(path.join(root, "knowledge", kind, tier));
      for (const filePath of files) {
        if (path.basename(filePath) === ".gitkeep") continue;
        const rel = toRelative(root, filePath);
        const existing = docs.find((doc) => doc.patternPath === rel || doc.referencePath === rel);
        if (existing) continue;
        const id = `${tier}_${kind}_${path.basename(filePath).replace(/[^a-zA-Z0-9_-]+/g, "_")}`;
        docs.push({
          id,
          tier,
          status: tier,
          patternPath: kind === "patterns" ? rel : undefined,
          referencePath: kind === "references" ? rel : undefined,
        });
      }
    }
  }
  return docs.sort((a, b) => a.id.localeCompare(b.id));
}

async function loadImprovements(root: string, outputBase: string): Promise<{
  reports: ImprovementReportSummary[];
  totalCostUsd: number;
}> {
  const files = (await listFilesRecursive(path.join(outputBase, "reports")))
    .filter((file) => path.basename(file).startsWith("improve_") && file.endsWith(".json"));
  const reports: ImprovementReportSummary[] = [];
  // Sum cost from raw `messages[].total_cost_usd` while we have the full
  // file open. The summary type strips `messages` (they can be MB each),
  // so cost has to be totaled here or it's lost.
  let totalCostUsd = 0;
  for (const file of files) {
    const raw = asRecord(await readJson(file));
    const moduleId = asString(raw.module_id) ?? "unknown";
    const rawAttempts = asArray<JsonRecord>(raw.attempts);
    let reportCost = 0;
    const attempts = rawAttempts.map((attempt) => {
      const vivado = asRecord(attempt.vivado_report);
      const metrics = asRecord(attempt.metrics);
      const verdict = asRecord(attempt.verdict);
      const verif = asRecord(attempt.assayer_result);
      for (const msg of asArray<JsonRecord>(attempt.messages)) {
        const cost = asNumber(asRecord(msg).total_cost_usd);
        if (cost) reportCost += cost;
      }
      return {
        attemptIndex: asNumber(attempt.attempt_index) ?? 0,
        failedGate: asString(attempt.failed_gate) ?? null,
        metrics: {
          lut: asNumber(metrics.lut),
          dsp: asNumber(metrics.dsp),
          bram: asNumber(metrics.bram),
          latency_cycles: asNumber(metrics.latency_cycles),
          ii: asNumber(metrics.ii),
        },
        vivado: Object.keys(vivado).length > 0 ? {
          success: vivado.success === true,
          timingMet: vivado.timing_met === true,
          lut: asNumber(vivado.lut_count),
          ff: asNumber(vivado.ff_count),
          dsp: asNumber(vivado.dsp_count),
          bram: asNumber(vivado.bram18_equiv),
          fmaxMhz: asNumber(vivado.fmax_mhz),
          setupWnsNs: asNumber(vivado.setup_wns_ns ?? vivado.wns_ns) ?? null,
          holdWnsNs: asNumber(vivado.hold_wns_ns) ?? null,
        } : undefined,
        verifStatus: asString(verif.status),
        verdictOverall: typeof verdict.overall === "boolean" ? verdict.overall : undefined,
        artifactPaths: [asString(attempt.verilog_path)].filter(Boolean) as string[],
      };
    });
    totalCostUsd += reportCost;
    const finalVerdict = asRecord(raw.final_verdict);
    reports.push({
      moduleId,
      targetSlug: parseTargetSlug(toRelative(root, file), moduleId),
      targets: asArray(raw.targets) as ImprovementReportSummary["targets"],
      success: raw.success === true,
      finalAction: asString(raw.final_action) ?? "unknown",
      reportPath: toRelative(root, file),
      improvedReferencePath: asString(raw.improved_reference_path)?.replace(/\\/g, "/"),
      committedModulePath: asString(raw.committed_module_path)?.replace(/\\/g, "/"),
      costUsd: reportCost > 0 ? reportCost : undefined,
      attempts,
      finalVerdict: typeof finalVerdict.overall === "boolean" ? {
        overall: finalVerdict.overall,
        targets: asArray(finalVerdict.targets) as NonNullable<ImprovementReportSummary["finalVerdict"]>["targets"],
      } : undefined,
    });
  }
  reports.sort((a, b) => a.moduleId.localeCompare(b.moduleId) || a.targetSlug.localeCompare(b.targetSlug));
  return { reports, totalCostUsd };
}

async function loadImproveRuns(root: string, outputBase: string): Promise<ImproveRunSummary[]> {
  const improveRoot = path.join(outputBase, "improve");
  const files = await listFilesRecursive(improveRoot);
  const byRun = new Map<string, ImproveRunSummary>();
  // Improve artifacts live at `<outputBase>/improve/<moduleId>/<runId>/...`.
  // We strip the outputBase prefix so the moduleId/runId positions are stable
  // regardless of whether outputBase is "output" (resnet-50) or
  // "output/<network>" (future networks).
  const improveRel = toRelative(root, improveRoot);
  const improvePrefixLen = improveRel === "" ? 0 : improveRel.split("/").length;
  for (const file of files) {
    const rel = toRelative(root, file);
    const parts = rel.split("/");
    if (parts.length < improvePrefixLen + 2) continue;
    const moduleId = parts[improvePrefixLen];
    const runId = parts[improvePrefixLen + 1];
    const key = `${moduleId}/${runId}`;
    const current = byRun.get(key) ?? {
      moduleId,
      runId,
      attemptCount: 0,
      artifactCount: 0,
      artifactPaths: [],
    };
    current.artifactCount += 1;
    current.artifactPaths.push(rel);
    const match = path.basename(file).match(/^attempt_(\d+)/);
    if (match) current.attemptCount = Math.max(current.attemptCount, Number(match[1]));
    byRun.set(key, current);
  }
  return [...byRun.values()].sort((a, b) => b.runId.localeCompare(a.runId));
}

function docsForModule(moduleId: string, opType: string, docs: DocSummary[]): DocSummary[] {
  return docs.filter((doc) => {
    if (doc.moduleId === moduleId) return true;
    if (doc.successfulModules?.includes(moduleId) || doc.usedByModules?.includes(moduleId)) return true;
    if (doc.patternPath?.includes(moduleId) || doc.referencePath?.includes(moduleId)) return true;
    if (doc.tier === "protected" && doc.patternPath && doc.patternPath.includes(opType)) return true;
    return false;
  });
}

export type BuildSnapshotOptions = {
  /**
   * Network whose artifacts to load. Defaults to the registry default
   * (ResNet-50), which keeps the original `output/` flat layout working.
   */
  networkId?: NetworkId;
  /**
   * Override for the resolved output directory. Used by snapshot tests that
   * seed a temp repo and want to point at a specific subtree.
   */
  outputBase?: string;
};

export async function buildSnapshot(
  root: string = repoRoot,
  options: BuildSnapshotOptions = {},
): Promise<ProjectSnapshot> {
  const networkId = options.networkId ?? DEFAULT_NETWORK_ID;
  // When the caller passes a custom root (tests), resolve the output base
  // relative to it. When using the default repo root, defer to the network
  // registry so future networks can sit under `output/<network>/`.
  const outputBase =
    options.outputBase ??
    (root === repoRoot ? outputDirFor(networkId) : path.join(root, "output"));
  const layerIr = asRecord(await readJson(path.join(outputBase, "layer_ir.json")));
  const pipelineState = asRecord(await readJson(path.join(outputBase, "pipeline_state.json")));
  const pipelineSummary = asRecord(await readJson(path.join(outputBase, "reports", "pipeline_summary.json")));
  const layers = asArray<JsonRecord>(layerIr.layers);
  const docs = await loadDocs(root);
  const { reports: improvements, totalCostUsd: improveCostUsd } = await loadImprovements(root, outputBase);
  const improveRuns = await loadImproveRuns(root, outputBase);
  const jobs = root === repoRoot ? await readJobs() : [];
  const moduleIds = new Set(layers.map((layer) => asString(layer.module_id)).filter(Boolean) as string[]);

  const modules: LayerSummary[] = [];
  for (let index = 0; index < layers.length; index += 1) {
    const layer = layers[index];
    const moduleId = asString(layer.module_id) ?? `layer_${index}`;
    const verifPath = reportPath(outputBase, moduleId, ".results.json");
    const vivadoPath = reportPath(outputBase, moduleId, ".vivado.json");
    const rtlFile = rtlPath(outputBase, moduleId, ".v");
    const metaFile = rtlPath(outputBase, moduleId, ".meta.json");
    const verif = asRecord(await readJson(verifPath));
    const vivado = asRecord(await readJson(vivadoPath));
    const moduleImprovements = improvements.filter((report) => report.moduleId === moduleId);
    const hasRtl = await pathExists(rtlFile);
    const vivadoSuccess = typeof vivado.success === "boolean" ? vivado.success : undefined;
    const vivadoTimingMet = typeof vivado.timing_met === "boolean" ? vivado.timing_met : undefined;
    const opType = asString(layer.op_type) ?? "unknown";
    modules.push({
      index,
      moduleId,
      opType,
      contractId: asString(layer.contract_id) ?? "none",
      ioMode: asString(layer.io_mode) ?? "default",
      inputShape: asArray<number>(layer.input_shape),
      outputShape: asArray<number>(layer.output_shape),
      weightShape: asArray<number>(layer.weight_shape),
      numWeights: asNumber(layer.num_weights) ??
        asArray<number>(layer.weight_shape).reduce((acc, dim) => acc * (dim || 1), 1),
      pipelineLatencyCycles: asNumber(layer.pipeline_latency_cycles) ?? 0,
      stage: stageFor({
        hasRtl,
        verifStatus: asString(verif.status),
        vivadoSuccess,
        vivadoTimingMet,
        hasSuccessfulImprovement: moduleImprovements.some((report) => report.success),
      }),
      hasRtl,
      hasMeta: await pathExists(metaFile),
      hasGoldenIn: await pathExists(path.join(outputBase, "goldens", `${moduleId}.goldin`)),
      hasGoldenOut: await pathExists(path.join(outputBase, "goldens", `${moduleId}.goldout`)),
      pipelineStatus: asString(asRecord(pipelineState.modules)[moduleId]),
      pipelineAttempts: asNumber(asRecord(pipelineState.attempts)[moduleId]),
      verif: Object.keys(verif).length > 0 ? {
        status: asString(verif.status),
        timingPass: verif.timing_pass === true,
        timingActualCycles: asNumber(verif.timing_actual_cycles),
        timingExpectedCycles: asNumber(verif.timing_expected_cycles),
        maxError: asNumber(verif.max_error),
        meanError: asNumber(verif.mean_error),
      } : undefined,
      vivado: Object.keys(vivado).length > 0 ? {
        success: vivadoSuccess,
        timingMet: vivadoTimingMet,
        lut: asNumber(vivado.lut_count),
        ff: asNumber(vivado.ff_count),
        dsp: asNumber(vivado.dsp_count),
        bram: asNumber(vivado.bram18_equiv),
        fmaxMhz: asNumber(vivado.fmax_mhz),
        setupWnsNs: asNumber(vivado.setup_wns_ns ?? vivado.wns_ns) ?? null,
        holdWnsNs: asNumber(vivado.hold_wns_ns) ?? null,
      } : undefined,
      docs: docsForModule(moduleId, opType, docs),
      improvements: moduleImprovements,
      paths: {
        rtl: hasRtl ? toRelative(root, rtlFile) : undefined,
        meta: await pathExists(metaFile) ? toRelative(root, metaFile) : undefined,
        verif: await pathExists(verifPath) ? toRelative(root, verifPath) : undefined,
        vivado: await pathExists(vivadoPath) ? toRelative(root, vivadoPath) : undefined,
      },
    });
  }

  const tierCount = (tier: DocTier) => docs.filter((doc) => doc.tier === tier).length;
  const kpis: DashboardKpis = {
    totalLayers: modules.length,
    rtlGenerated: modules.filter((module) => module.hasRtl).length,
    verilatorPass: modules.filter((module) => module.verif?.status === "pass").length,
    vivadoPass: modules.filter((module) => module.vivado?.success && module.vivado.timingMet).length,
    failedOrUnknown: modules.filter((module) => module.stage === "failed" || module.stage === "missing").length,
    improvedVariants: improvements.filter((report) => report.success).length,
    docsProtected: tierCount("protected"),
    docsActive: tierCount("active"),
    docsProbationary: tierCount("probationary"),
    docsImproved: tierCount("improved"),
    knownCostUsd:
      (asNumber(pipelineSummary.total_cost_usd) ?? asNumber(pipelineState.total_cost_usd) ?? 0) +
      improveCostUsd,
  };
  const stateCounts: Record<string, number> = {};
  for (const value of Object.values(asRecord(pipelineState.modules))) {
    if (typeof value !== "string") continue;
    stateCounts[value] = (stateCounts[value] ?? 0) + 1;
  }
  const rtlArtifacts = (await listFilesRecursive(path.join(outputBase, "rtl")))
    .filter((file) => file.endsWith(".v"))
    .map((file) => toRelative(root, file));
  const reportArtifacts = (await listFilesRecursive(path.join(outputBase, "reports")))
    .filter((file) => /\.(results|vivado)\.json$/.test(file))
    .map((file) => toRelative(root, file));
  const networks: NetworkInfo[] = NETWORKS.map((network) => ({
    id: network.id,
    label: network.label,
    modelName: network.modelName,
    description: network.description,
    available: network.available,
    defaultCheckpointPath: network.defaultCheckpointPath,
    outputDir: network.outputDir,
  }));
  return {
    generatedAt: new Date().toISOString(),
    repoRoot: root,
    networkId,
    networks,
    modelName: asString(layerIr.model_name) ?? getNetwork(networkId).modelName,
    quantization: asString(layerIr.quantization),
    kpis,
    modules,
    docs,
    improvements,
    improveRuns,
    jobs,
    latestPipeline: {
      runId: asString(pipelineState.run_id) ?? asString(pipelineSummary.run_id),
      startedAt: asString(pipelineState.started_at),
      isDone: typeof pipelineSummary.is_done === "boolean" ? pipelineSummary.is_done : undefined,
      totalCostUsd: asNumber(pipelineState.total_cost_usd) ?? asNumber(pipelineSummary.total_cost_usd),
      stateCounts,
    },
    orphanArtifacts: {
      rtl: rtlArtifacts.filter((file) => !moduleIds.has(path.basename(file, ".v"))),
      reports: reportArtifacts.filter((file) => !moduleIds.has(path.basename(file).replace(/\.(results|vivado)\.json$/, ""))),
    },
  };
}
