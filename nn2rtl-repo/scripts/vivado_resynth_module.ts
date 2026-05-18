// Re-run Vivado synthesis for a single module that already has RTL on disk.
//
// Reads `<outputDir>/rtl/<moduleId>.v`, extracts the module name, calls
// `run_vivado` with the same clock and part the pipeline uses, and writes
// `<outputDir>/reports/<moduleId>.vivado.json`. No LLM calls, no money spent.
//
// Usage:
//   npx tsx scripts/vivado_resynth_module.ts <moduleId> [--network=resnet-50]
//
// Background: invoked by the dashboard "Resynth (Vivado only) for a module"
// card; the wrapper exists so the dashboard has a stable, single-purpose
// CLI to spawn for that button.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { run_vivado } from "../mcp/tools.ts";

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

async function parseArgs(argv: string[]): Promise<{ moduleId: string; networkId: string }> {
  const registry = JSON.parse(await readFile(path.join(repoRoot, "networks.json"), "utf8")) as NetworkRegistry;
  let networkId = registry.defaultNetworkId ?? "resnet-50";
  const positional: string[] = [];
  for (const arg of argv) {
    if (arg.startsWith("--network=")) {
      networkId = arg.slice("--network=".length);
    } else if (arg.startsWith("--")) {
      throw new Error(`Unknown flag '${arg}'.`);
    } else {
      positional.push(arg);
    }
  }
  if (positional.length !== 1) {
    throw new Error("Usage: tsx scripts/vivado_resynth_module.ts <moduleId> [--network=resnet-50]");
  }
  return { moduleId: positional[0], networkId };
}

async function main(): Promise<void> {
  const { moduleId, networkId } = await parseArgs(process.argv.slice(2));
  const outputDirRel = await networkOutputDir(networkId);
  process.env.NN2RTL_NETWORK_ID = networkId;
  process.env.NN2RTL_OUTPUT_DIR = path.resolve(repoRoot, outputDirRel);
  const outputDir = path.resolve(repoRoot, outputDirRel);
  const rtlPath = path.join(outputDir, "rtl", `${moduleId}.v`);
  const reportPath = path.join(outputDir, "reports", `${moduleId}.vivado.json`);

  let source: string;
  try {
    source = await readFile(rtlPath, "utf8");
  } catch {
    throw new Error(`RTL file not found: ${rtlPath}. Generate RTL for this module first.`);
  }
  const moduleMatch = source.match(/^\s*module\s+([A-Za-z_][A-Za-z0-9_]*)/m);
  if (!moduleMatch) throw new Error("Could not extract module name from RTL.");

  // Read the LayerIR-pinned clock period for this module. Without it
  // `run_vivado` defaults clock_period_ns to 0, which makes the fmax
  // calculation collapse to 0 even when timing closes cleanly.
  const layerIrPath = path.join(outputDir, "layer_ir.json");
  let clockPeriodNs = 0;
  try {
    const ir = JSON.parse(await readFile(layerIrPath, "utf8")) as {
      layers?: Array<{ module_id?: string; clock_period_ns?: number }>;
    };
    const layer = ir.layers?.find((l) => l.module_id === moduleId);
    if (typeof layer?.clock_period_ns === "number" && layer.clock_period_ns > 0) {
      clockPeriodNs = layer.clock_period_ns;
    }
  } catch {
    // ignore — falls through to 0 and the report will be missing fmax,
    // matching the pre-fix behaviour rather than failing the run.
  }

  console.log(`[resynth] module=${moduleId} network=${networkId}`);
  console.log(`[resynth] rtl=${rtlPath}`);
  console.log(`[resynth] report=${reportPath}`);
  console.log(`[resynth] clock_period_ns=${clockPeriodNs}`);
  const t0 = Date.now();
  const report = await run_vivado(source, moduleMatch[1], clockPeriodNs);
  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`[resynth] done in ${elapsed}s — success=${report.success} timing_met=${report.timing_met ?? "n/a"} fmax=${report.fmax_mhz?.toFixed(2) ?? "n/a"}`);

  await mkdir(path.dirname(reportPath), { recursive: true });
  await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  console.log(`[resynth] wrote ${reportPath}`);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
