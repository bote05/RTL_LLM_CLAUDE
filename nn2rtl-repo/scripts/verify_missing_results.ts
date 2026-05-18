// Re-run the deterministic assayer (iverilog + Verilator + golden compare)
// on modules whose `output/reports/<id>.results.json` is missing. No LLM.
// Writes the JSON back to output/reports/ so the throughput stats can use it.

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync, readFileSync } from "node:fs";

import { runAssayerDeterministic } from "../sdk/orchestrate.ts";
import { layerIrSchema, pipelineIrSchema, verilogModuleSchema } from "../sdk/schemas.ts";
import type { LayerIR, VerilogModule } from "../sdk/types.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
// tsx transpiles to CJS by default, which rejects top-level `await`. Use
// synchronous read for the small registry file so the script stays portable
// across the bundler quirks.
const registry = JSON.parse(readFileSync(path.join(repoRoot, "networks.json"), "utf8")) as {
  defaultNetworkId: string;
  networks: Array<{ id: string; outputDir: string }>;
};
const rawArgs = process.argv.slice(2);
const networkArg = rawArgs.find((arg) => arg.startsWith("--network="))?.split("=", 2)[1];
const networkId = networkArg ?? process.env.NN2RTL_NETWORK_ID ?? registry.defaultNetworkId;
const network = registry.networks.find((entry) => entry.id === networkId);
if (!network) {
  throw new Error(`Unknown network '${networkId}'. Known: ${registry.networks.map((entry) => entry.id).join(", ")}`);
}
const outputRoot = path.resolve(repoRoot, process.env.NN2RTL_OUTPUT_DIR ?? network.outputDir);
process.env.NN2RTL_NETWORK_ID = networkId;
process.env.NN2RTL_OUTPUT_DIR = outputRoot;
const targetArgs = rawArgs.filter((arg) => !arg.startsWith("--network="));

async function loadLayer(moduleId: string): Promise<LayerIR | null> {
  const irPath = path.join(outputRoot, "layer_ir.json");
  const raw = JSON.parse(await readFile(irPath, "utf8"));
  const pipe = pipelineIrSchema.parse(raw);
  const layer = pipe.layers.find((l) => l.module_id === moduleId);
  return layer ? layerIrSchema.parse(layer) : null;
}

async function loadModule(moduleId: string): Promise<VerilogModule | null> {
  const metaPath = path.join(outputRoot, "rtl", `${moduleId}.meta.json`);
  const vPath = path.join(outputRoot, "rtl", `${moduleId}.v`);
  if (existsSync(metaPath)) {
    const m = JSON.parse(await readFile(metaPath, "utf8"));
    return verilogModuleSchema.parse(m);
  }
  if (!existsSync(vPath)) return null;
  const src = await readFile(vPath, "utf8");
  return verilogModuleSchema.parse({
    module_id: moduleId,
    spec_hash: "unknown",
    verilog_source: src,
    generated_by: "Foundry",
    attempt: 1,
  });
}

async function main(): Promise<void> {
  const targets = targetArgs;
  if (targets.length === 0) {
    console.error("usage: tsx scripts/verify_missing_results.ts <module_id> [<module_id> ...]");
    process.exit(1);
  }
  let okCount = 0;
  let failCount = 0;
  for (const mid of targets) {
    process.stdout.write(`[verify] >>> ${mid} ... `);
    const layer = await loadLayer(mid);
    if (!layer) {
      console.log(`SKIP (not in layer_ir.json)`);
      continue;
    }
    const mod = await loadModule(mid);
    if (!mod) {
      console.log(`SKIP (no RTL on disk)`);
      continue;
    }
    try {
      const t0 = Date.now();
      const result = await runAssayerDeterministic(mod, layer);
      const dt = ((Date.now() - t0) / 1000).toFixed(1);
      const outPath = path.join(outputRoot, "reports", `${mid}.results.json`);
      await writeFile(outPath, JSON.stringify(result, null, 2), "utf8");
      const pv = (result as Record<string, unknown>).per_vector;
      const nVec = Array.isArray(pv) ? pv.length : 0;
      console.log(
        `${result.status} (${dt}s, ${nVec} vec, exact=${result.exact_match_count ?? "?"}/${result.sample_count ?? "?"})`,
      );
      if (result.status === "pass") okCount += 1;
      else failCount += 1;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      console.log(`ERROR: ${msg.slice(0, 200)}`);
      failCount += 1;
    }
  }
  console.log(`\n[verify] done. passed=${okCount} failed=${failCount}`);
  if (failCount > 0) process.exit(2);
}

main().catch((e) => {
  console.error("[verify] FATAL:", e instanceof Error ? e.stack ?? e.message : String(e));
  process.exit(1);
});
