import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

import { contractFitFailure } from "./contracts.js";
import { CONTRACT_PLANS, applyContractPlan } from "./orchestrate.js";
import { signatureBundle } from "./signatures.js";
import type { LayerIR, PipelineIR } from "./types.js";

const execFileAsync = promisify(execFile);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const sdkRoot = path.resolve(
  __dirname,
  path.basename(__dirname) === "dist" ? ".." : ".",
);
const repoRoot = path.resolve(sdkRoot, "..");
const registryPath = path.join(repoRoot, "networks.json");

type NetworkConfig = {
  id: string;
  label: string;
  modelName: string;
  outputDir: string;
  defaultCheckpointPath: string;
  frontend: string;
  available: boolean;
  description: string;
};

type NetworkRegistry = {
  version: number;
  defaultNetworkId: string;
  networks: NetworkConfig[];
};

type ImportCliArgs = {
  id: string;
  label?: string;
  modelName?: string;
  modelPath?: string;
  outputDir?: string;
  frontend?: string;
  description?: string;
  prepare: boolean;
  samples?: number;
};

type LayerReadiness = {
  module_id: string;
  op_type: LayerIR["op_type"];
  status: "runnable" | "blocked" | "preflight_risk";
  contract_id?: string;
  contract_needed?: string;
  block_reason?: string;
  preflight_risks: string[];
  estimated_cycles: number;
  signature_hash?: string;
  exact_reference_key?: string | null;
};

const SCALAR_MEMORY_CELL_THRESHOLD = 16_384;
const PER_VARIABLE_BIT_LIMIT = 900_000;
const MULTIDIM_FF_TOLERATED_BITS = 300_000;

function usage(): string {
  return [
    "Usage:",
    "  npx tsx sdk/main.ts import_network --id <network-id> --checkpoint <path> [--label <name>] [--model-name <name>]",
    "",
    "Options:",
    "  --id <id>             Stable network id, e.g. mobilenet-v2",
    "  --checkpoint <path>   .pth or .onnx model path used by scripts/generate_golden.py",
    "  --output-dir <dir>    Artifact root; defaults to output/<id>",
    "  --frontend <name>     checkpoint or onnx; inferred from model extension when omitted",
    "  --no-prepare          Only update networks.json; do not generate LayerIR/goldens",
  ].join("\n");
}

function parseArgs(argv: string[]): ImportCliArgs {
  const out: ImportCliArgs = { id: "", prepare: true };
  const take = (index: number, flag: string): string => {
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) throw new Error(`${flag} requires a value.\n${usage()}`);
    return value;
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const [flag, inline] = arg.split("=", 2);
    const value = inline ?? null;
    switch (flag) {
      case "--id":
      case "--network":
        out.id = value ?? take(i++, flag);
        break;
      case "--label":
        out.label = value ?? take(i++, flag);
        break;
      case "--model-name":
      case "--name":
        out.modelName = value ?? take(i++, flag);
        break;
      case "--checkpoint":
      case "--model":
      case "--model-path":
        out.modelPath = value ?? take(i++, flag);
        break;
      case "--output-dir":
        out.outputDir = value ?? take(i++, flag);
        break;
      case "--frontend":
        out.frontend = value ?? take(i++, flag);
        break;
      case "--description":
        out.description = value ?? take(i++, flag);
        break;
      case "--samples":
        out.samples = Number(value ?? take(i++, flag));
        break;
      case "--prepare":
        out.prepare = true;
        break;
      case "--no-prepare":
        out.prepare = false;
        break;
      default:
        if (!arg.startsWith("--") && !out.id) {
          out.id = arg;
        } else if (!arg.startsWith("--") && !out.modelPath) {
          out.modelPath = arg;
        } else {
          throw new Error(`Unknown import_network argument '${arg}'.\n${usage()}`);
        }
    }
  }
  if (!/^[a-z0-9][a-z0-9._-]*$/i.test(out.id)) {
    throw new Error(`Network id '${out.id}' must be a stable path-safe id.\n${usage()}`);
  }
  if (out.prepare && !out.modelPath) {
    throw new Error("--checkpoint/--model is required unless --no-prepare is used.");
  }
  return out;
}

async function readRegistry(): Promise<NetworkRegistry> {
  return JSON.parse(await readFile(registryPath, "utf8")) as NetworkRegistry;
}

async function writeRegistry(registry: NetworkRegistry): Promise<void> {
  await writeFile(registryPath, `${JSON.stringify(registry, null, 2)}\n`, "utf8");
}

function inferFrontend(modelPath: string | undefined): string {
  if (!modelPath) return "checkpoint";
  return path.extname(modelPath).toLowerCase() === ".onnx" ? "onnx" : "checkpoint";
}

function humanLabel(id: string): string {
  return id
    .split(/[-_]+/g)
    .filter(Boolean)
    .map((part) => `${part[0]?.toUpperCase() ?? ""}${part.slice(1)}`)
    .join(" ");
}

async function upsertNetwork(args: ImportCliArgs): Promise<NetworkConfig> {
  const registry = await readRegistry();
  const outputDir = args.outputDir ?? `output/${args.id}`;
  const network: NetworkConfig = {
    id: args.id,
    label: args.label ?? humanLabel(args.id),
    modelName: args.modelName ?? args.id,
    outputDir,
    defaultCheckpointPath: args.modelPath ?? "",
    frontend: args.frontend ?? inferFrontend(args.modelPath),
    available: true,
    description: args.description ?? `${args.label ?? humanLabel(args.id)} imported network`,
  };
  const existing = registry.networks.findIndex((entry) => entry.id === args.id);
  if (existing >= 0) {
    registry.networks[existing] = { ...registry.networks[existing], ...network };
  } else {
    registry.networks.push(network);
  }
  await writeRegistry(registry);
  return network;
}

function repoResolve(inputPath: string): string {
  return path.isAbsolute(inputPath) ? inputPath : path.resolve(repoRoot, inputPath);
}

async function prepareNetwork(args: ImportCliArgs, network: NetworkConfig): Promise<void> {
  if (!args.modelPath) return;
  const script = path.join(repoRoot, "scripts", "generate_golden.py");
  const py = process.env.PYTHON ?? "python3";
  const commandArgs = [
    script,
    repoResolve(args.modelPath),
    `--network=${network.id}`,
    `--output-dir=${network.outputDir}`,
    `--name=${network.modelName}`,
  ];
  if (args.samples !== undefined) {
    commandArgs.push(`--samples=${args.samples}`);
  }
  await execFileAsync(py, commandArgs, {
    cwd: repoRoot,
    env: {
      ...process.env,
      NN2RTL_NETWORK_ID: network.id,
      NN2RTL_OUTPUT_DIR: repoResolve(network.outputDir),
    },
    maxBuffer: 16 * 1024 * 1024,
  });

  // Write the checkpoint fingerprint sidecar so `ensureLayerIr` accepts the
  // generated layer_ir.json instead of treating it as stale. Without this,
  // the pipeline refuses to use the imported IR and the next runCli call
  // throws "Stale layer_ir.json found (not tied to checkpoint ...)" — even
  // though import_network just produced that IR from the named checkpoint.
  const outputRootAbs = repoResolve(network.outputDir);
  const layerIrPath = path.join(outputRootAbs, "layer_ir.json");
  const fingerprintPath = `${layerIrPath}.checkpoint`;
  const checkpointAbs = repoResolve(args.modelPath);
  if (existsSync(layerIrPath)) {
    await writeFile(fingerprintPath, checkpointAbs, "utf8");
  }
}

function channels(shape: number[] | undefined): number {
  return Array.isArray(shape) && typeof shape[1] === "number" ? shape[1] : 1;
}

function spatialArea(shape: number[] | undefined): number {
  return Array.isArray(shape) && typeof shape[2] === "number" && typeof shape[3] === "number"
    ? shape[2] * shape[3]
    : 1;
}

function activationBits(layer: LayerIR): number {
  return spatialArea(layer.input_shape) * channels(layer.input_shape) * 8;
}

function importPreflightRisks(layer: LayerIR): string[] {
  const risks = new Set<string>();
  const actCells = spatialArea(layer.input_shape) * channels(layer.input_shape);
  const actBits = activationBits(layer);
  const wideWordBits = spatialArea(layer.input_shape) * Math.max(layer.input_width_bits, layer.output_width_bits);

  if (actCells >= SCALAR_MEMORY_CELL_THRESHOLD) {
    risks.add("large_scalarized_activation_memory");
  }
  if (actBits > PER_VARIABLE_BIT_LIMIT || wideWordBits > PER_VARIABLE_BIT_LIMIT) {
    risks.add("activation_memory_exceeds_vivado_variable_bit_limit");
  }
  if (layer.op_type === "conv2d" && layer.channel_tile && wideWordBits > MULTIDIM_FF_TOLERATED_BITS) {
    risks.add("multidim_wideword_activation_memory");
  }
  return [...risks];
}

function isDepthwiseConv(layer: LayerIR): boolean {
  if (layer.op_type !== "conv2d") return false;
  const groups = typeof layer.groups === "number" ? layer.groups : 1;
  if (groups <= 1) return false;
  const inputChannels = channels(layer.input_shape);
  const outputChannels = channels(layer.output_shape);
  return (
    inputChannels !== null &&
    outputChannels !== null &&
    groups === inputChannels &&
    groups === outputChannels
  );
}

function selectRuntimeLayer(layer: LayerIR): { layer: LayerIR; contract_id: string; block_reason?: string } {
  // Depthwise convs must route to depthwise-conv regardless of bus-width
  // fitness, because the standard flat-bus contract assumes cross-channel
  // reduction (which depthwise specifically does NOT do). Letting the
  // generic walker pick flat-bus would generate RTL with the wrong adder
  // tree and miss the per-channel filter layout.
  if (isDepthwiseConv(layer)) {
    return {
      layer: { ...layer, contract_id: "depthwise-conv" } as LayerIR,
      contract_id: "depthwise-conv",
    };
  }
  for (const plan of CONTRACT_PLANS) {
    const candidate = applyContractPlan(layer, plan);
    const failure = contractFitFailure(candidate);
    if (!failure) return { layer: candidate, contract_id: plan.id };
  }
  return {
    layer,
    contract_id: layer.contract_id ?? "flat-bus",
    block_reason: "no_registered_contract_fits_layer",
  };
}

function readinessForLayer(layer: LayerIR, modelQuantization: string): LayerReadiness {
  const groups = typeof layer.groups === "number" ? layer.groups : 1;
  const inputChannels = channels(layer.input_shape);
  const outputChannels = channels(layer.output_shape);
  if (layer.op_type === "conv2d" && groups > 1) {
    const isDepthwise =
      groups === inputChannels &&
      groups === outputChannels &&
      inputChannels !== null;
    if (isDepthwise) {
      // Depthwise layers now route to the depthwise-conv contract. The
      // runtime LayerIR is tagged with contract_id=depthwise-conv so
      // signature lookup, sidecar writing, and Foundry dispatch use the
      // new contract; signatures are derived post-contract-plan so the
      // depthwise fingerprint differs cleanly from any same-shape
      // standard conv.
      const runtimeLayer: LayerIR = { ...layer, contract_id: "depthwise-conv" };
      const signatures = signatureBundle({
        baseLayer: layer,
        runtimeLayer,
        baseContractId: layer.contract_id ?? "flat-bus",
        runtimeContractId: "depthwise-conv",
        modelQuantization,
      });
      const risks = importPreflightRisks(runtimeLayer);
      return {
        module_id: layer.module_id,
        op_type: layer.op_type,
        status: risks.length > 0 ? "preflight_risk" : "runnable",
        contract_id: "depthwise-conv",
        preflight_risks: risks,
        estimated_cycles: layer.pipeline_latency_cycles,
        signature_hash: signatures.signature_hash,
        exact_reference_key: signatures.exact_reference_key,
      };
    }
    // groups>1 but not strict depthwise (i.e. group-conv variants) — still
    // unsupported in v1.
    return {
      module_id: layer.module_id,
      op_type: layer.op_type,
      status: "blocked",
      contract_needed: "grouped-conv",
      block_reason: `groups=${groups} (in=${inputChannels}, out=${outputChannels}) is not strict depthwise and no grouped-conv contract is registered yet.`,
      preflight_risks: [],
      estimated_cycles: layer.pipeline_latency_cycles,
    };
  }

  const selected = selectRuntimeLayer(layer);
  const signatures = signatureBundle({
    baseLayer: layer,
    runtimeLayer: selected.layer,
    baseContractId: layer.contract_id ?? "flat-bus",
    runtimeContractId: selected.contract_id,
    modelQuantization,
  });
  const risks = importPreflightRisks(selected.layer);
  return {
    module_id: layer.module_id,
    op_type: layer.op_type,
    status: selected.block_reason ? "blocked" : risks.length > 0 ? "preflight_risk" : "runnable",
    contract_id: selected.contract_id,
    block_reason: selected.block_reason,
    preflight_risks: risks,
    estimated_cycles: layer.pipeline_latency_cycles,
    signature_hash: signatures.signature_hash,
    exact_reference_key: signatures.exact_reference_key,
  };
}

async function writeReadinessReport(network: NetworkConfig): Promise<Record<string, unknown>> {
  const outputRoot = repoResolve(network.outputDir);
  const layerIrPath = path.join(outputRoot, "layer_ir.json");
  if (!existsSync(layerIrPath)) {
    const report = {
      network_id: network.id,
      model_name: network.modelName,
      status: "registered_only",
      layer_ir_path: layerIrPath,
      message: "No layer_ir.json exists yet; run import_network with --checkpoint or run scripts/generate_golden.py.",
    };
    await mkdir(path.join(outputRoot, "reports"), { recursive: true });
    await writeFile(path.join(outputRoot, "reports", "import_report.json"), `${JSON.stringify(report, null, 2)}\n`, "utf8");
    return report;
  }

  const pipeline = JSON.parse(await readFile(layerIrPath, "utf8")) as PipelineIR;
  const modelQuantization = pipeline.quantization ?? "mixed_or_unknown";
  const layers = pipeline.layers.map((layer) => readinessForLayer(layer, modelQuantization));
  const annotatedLayers = pipeline.layers.map((layer) => {
    const selected = selectRuntimeLayer(layer);
    const signatures = signatureBundle({
      baseLayer: layer,
      runtimeLayer: selected.layer,
      baseContractId: layer.contract_id ?? "flat-bus",
      runtimeContractId: selected.contract_id,
      modelQuantization,
    });
    // Persist the runtime contract_id (and any other contract-applied
    // fields surfaced by selectRuntimeLayer) so the LayerIR on disk
    // matches the report. Without this, `import_report.json` could say a
    // depthwise layer is runnable under depthwise-conv while the saved
    // layer_ir.json leaves contract_id unset and downstream dispatch
    // falls back to flat-bus.
    const { layer: runtimeLayer } = selected;
    return {
      ...layer,
      ...runtimeLayer,
      contract_id: selected.contract_id as LayerIR["contract_id"],
      quantization_family: signatures.runtime_layer_signature.quantization_family,
      ...signatures,
    };
  });
  const annotatedPipeline: PipelineIR = {
    ...pipeline,
    layers: annotatedLayers,
  };
  await writeFile(layerIrPath, `${JSON.stringify(annotatedPipeline, null, 2)}\n`, "utf8");
  const legacyGoldenPath = path.join(outputRoot, "golden_vectors.json");
  if (existsSync(legacyGoldenPath)) {
    await writeFile(legacyGoldenPath, `${JSON.stringify(annotatedPipeline, null, 2)}\n`, "utf8");
  }
  const runnable = layers.filter((layer) => layer.status === "runnable");
  const blocked = layers.filter((layer) => layer.status === "blocked");
  const preflightRisk = layers.filter((layer) => layer.status === "preflight_risk");
  const totalCycles = layers.reduce((sum, layer) => sum + layer.estimated_cycles, 0);
  const blockedCycles = blocked.reduce((sum, layer) => sum + layer.estimated_cycles, 0);
  const runnableCycles = runnable.reduce((sum, layer) => sum + layer.estimated_cycles, 0);
  const blockedRatio = totalCycles > 0 ? blockedCycles / totalCycles : 0;
  const report = {
    network_id: network.id,
    model_name: pipeline.model_name ?? network.modelName,
    output_root: outputRoot,
    layer_ir_path: layerIrPath,
    generated_at: new Date().toISOString(),
    counts: {
      total_layers: layers.length,
      runnable: runnable.length,
      blocked: blocked.length,
      preflight_risk: preflightRisk.length,
    },
    cycles_per_frame: {
      runnable: runnableCycles,
      blocked: blockedCycles,
      total: totalCycles,
      blocked_ratio: blockedRatio,
    },
    e2e_comparison_reliable: blockedRatio <= 0.2,
    reliability_rule: "e2e comparison is unreliable when blocked_cycles / total_cycles > 0.20",
    preflight_scope:
      "advisory only: import preflight uses LayerIR shape/contract heuristics; deterministic RTL synthesis preflight gates still run before Vivado.",
    layers,
  };
  await mkdir(path.join(outputRoot, "reports"), { recursive: true });
  await writeFile(path.join(outputRoot, "reports", "import_report.json"), `${JSON.stringify(report, null, 2)}\n`, "utf8");
  return report;
}

export async function runImportNetworkCli(argv = process.argv.slice(2)): Promise<void> {
  const args = parseArgs(argv);
  const network = await upsertNetwork(args);
  await mkdir(repoResolve(network.outputDir), { recursive: true });
  if (args.prepare) {
    await prepareNetwork(args, network);
  }
  const report = await writeReadinessReport(network);
  console.log(JSON.stringify(report, null, 2));
}
