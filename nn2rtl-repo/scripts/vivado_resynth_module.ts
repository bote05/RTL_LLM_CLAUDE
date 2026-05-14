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

// Mirrors `dashboard/src/shared/networks.ts` — keep these in sync if you
// add a network there. This script intentionally does not import the
// dashboard module to keep its dependency surface tiny (no React etc.).
const NETWORK_OUTPUT_DIRS: Record<string, string> = {
  "resnet-50": "output",
};

function parseArgs(argv: string[]): { moduleId: string; networkId: string } {
  let networkId = "resnet-50";
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
  const { moduleId, networkId } = parseArgs(process.argv.slice(2));
  const outputDirRel = NETWORK_OUTPUT_DIRS[networkId];
  if (!outputDirRel) {
    throw new Error(`Unknown network '${networkId}'. Known: ${Object.keys(NETWORK_OUTPUT_DIRS).join(", ")}`);
  }
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

  console.log(`[resynth] module=${moduleId} network=${networkId}`);
  console.log(`[resynth] rtl=${rtlPath}`);
  console.log(`[resynth] report=${reportPath}`);
  const t0 = Date.now();
  const report = await run_vivado(source, moduleMatch[1]);
  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`[resynth] done in ${elapsed}s — success=${report.success} timing_met=${report.timing_met ?? "n/a"}`);

  await mkdir(path.dirname(reportPath), { recursive: true });
  await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  console.log(`[resynth] wrote ${reportPath}`);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
