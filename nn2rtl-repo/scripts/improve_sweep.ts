// Run the Improve pipeline across every module in the current network's
// pipeline_state, one preset at a time.
//
// This script is a THIN wrapper around `runImprove` from `sdk/improve.ts`. It
// exists so the dashboard's "Improve sweep" card has a single, stable CLI to
// spawn — and so a user running the sweep from a terminal gets the same
// behavior as the button.
//
// Usage:
//   tsx scripts/improve_sweep.ts \
//       --preset=ppa \
//       --targets=use-dsp,reduce-lut,reduce-latency \
//       --network=resnet-50 \
//       (--plan | --run) \
//       [--max-modules=N] \
//       [--keep-reference]
//
// `--plan` only prints the plan and exits — no Claude calls, no Vivado, no
// money spent. `--run` actually executes the sweep.

import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { runImprove } from "../sdk/improve.ts";
import type { ImprovementTarget } from "../sdk/improve.ts";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");

type NetworkRegistry = { defaultNetworkId?: string; networks?: Array<{ id?: string; outputDir?: string }> };

async function networkOutputDir(networkId: string): Promise<string> {
  const registry = JSON.parse(await readFile(path.join(repoRoot, "networks.json"), "utf8")) as NetworkRegistry;
  const network = (registry.networks ?? []).find((entry) => entry.id === networkId);
  if (!network?.outputDir) {
    throw new Error(`Unknown network '${networkId}'. Known: ${(registry.networks ?? []).map((n) => n.id).join(", ")}`);
  }
  return network.outputDir;
}

type Args = {
  preset: string;
  targets: ImprovementTarget[];
  networkId: string;
  plan: boolean;
  run: boolean;
  maxModules: number | undefined;
  keepReference: boolean;
};

async function parseArgs(argv: string[]): Promise<Args> {
  const registry = JSON.parse(await readFile(path.join(repoRoot, "networks.json"), "utf8")) as NetworkRegistry;
  let preset = "ppa";
  let targets: ImprovementTarget[] = [];
  let networkId = registry.defaultNetworkId ?? "resnet-50";
  let plan = false;
  let run = false;
  let maxModules: number | undefined;
  let keepReference = false;
  for (const arg of argv) {
    if (arg.startsWith("--preset=")) preset = arg.slice("--preset=".length);
    else if (arg.startsWith("--targets=")) {
      targets = arg.slice("--targets=".length).split(",").map((s) => s.trim()).filter(Boolean) as ImprovementTarget[];
    } else if (arg.startsWith("--network=")) networkId = arg.slice("--network=".length);
    else if (arg === "--plan") plan = true;
    else if (arg === "--run") run = true;
    else if (arg.startsWith("--max-modules=")) {
      const raw = arg.slice("--max-modules=".length);
      const parsed = Number(raw);
      if (!Number.isInteger(parsed) || parsed <= 0) {
        throw new Error(`--max-modules must be a positive integer, got '${raw}'.`);
      }
      maxModules = parsed;
    } else if (arg === "--keep-reference") keepReference = true;
    else throw new Error(`Unknown flag '${arg}'.`);
  }
  if (plan === run) {
    throw new Error("Exactly one of --plan or --run must be passed.");
  }
  if (targets.length === 0) {
    throw new Error("--targets must be a non-empty comma-separated list of improvement targets.");
  }
  return { preset, targets, networkId, plan, run, maxModules, keepReference };
}

type PipelineState = {
  modules?: Record<string, string>;
};

async function readPipelineState(outputDir: string): Promise<PipelineState> {
  try {
    const raw = await readFile(path.join(outputDir, "pipeline_state.json"), "utf8");
    return JSON.parse(raw) as PipelineState;
  } catch {
    return {};
  }
}

async function main(): Promise<void> {
  const args = await parseArgs(process.argv.slice(2));
  const outputDirRel = await networkOutputDir(args.networkId);
  process.env.NN2RTL_NETWORK_ID = args.networkId;
  process.env.NN2RTL_OUTPUT_DIR = path.resolve(repoRoot, outputDirRel);
  const outputDir = path.resolve(repoRoot, outputDirRel);

  const state = await readPipelineState(outputDir);
  // Sweep only modules that already passed the normal pipeline — improving a
  // module whose canonical RTL hasn't been verified is meaningless.
  const moduleIds = Object.entries(state.modules ?? {})
    .filter(([, status]) => status === "pass")
    .map(([id]) => id);
  const limited = args.maxModules !== undefined ? moduleIds.slice(0, args.maxModules) : moduleIds;

  console.log(`[sweep] preset=${args.preset} targets=${args.targets.join(",")} network=${args.networkId}`);
  console.log(`[sweep] passing modules in pipeline_state: ${moduleIds.length}; will sweep ${limited.length}`);
  if (limited.length === 0) {
    console.log(`[sweep] nothing to do — no modules with status='pass' under ${outputDir}/pipeline_state.json`);
    return;
  }
  for (const [idx, moduleId] of limited.entries()) {
    console.log(`[sweep] [${idx + 1}/${limited.length}] ${moduleId}`);
  }
  if (args.plan) {
    console.log(`[sweep] PLAN mode — exiting without running. Re-invoke with --run to execute.`);
    return;
  }

  let succeeded = 0;
  let failed = 0;
  for (const [idx, moduleId] of limited.entries()) {
    console.log(`\n[sweep] (${idx + 1}/${limited.length}) improve ${moduleId} [${args.targets.join(",")}]`);
    try {
      const result = await runImprove(moduleId, {
        targets: args.targets,
        keepReference: args.keepReference,
      });
      console.log(`[sweep] (${idx + 1}/${limited.length}) ${moduleId}: ${result.final_action} (success=${result.success})`);
      if (result.success) succeeded += 1;
      else failed += 1;
    } catch (error) {
      failed += 1;
      console.error(`[sweep] (${idx + 1}/${limited.length}) ${moduleId} threw:`, error instanceof Error ? error.message : String(error));
    }
  }
  console.log(`\n[sweep] done — succeeded=${succeeded} failed=${failed} total=${limited.length}`);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
