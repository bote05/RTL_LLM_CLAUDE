// Re-run the deterministic assayer (iverilog + Verilator + golden compare)
// on modules whose `output/reports/<id>.results.json` is missing. No LLM.
// Writes the JSON back to output/reports/ so the throughput stats can use it.

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";

import { runAssayerDeterministic } from "../sdk/orchestrate.ts";
import { layerIrSchema, pipelineIrSchema, verilogModuleSchema } from "../sdk/schemas.ts";
import type { LayerIR, VerilogModule } from "../sdk/types.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

async function loadLayer(moduleId: string): Promise<LayerIR | null> {
  const irPath = path.join(repoRoot, "output", "layer_ir.json");
  const raw = JSON.parse(await readFile(irPath, "utf8"));
  const pipe = pipelineIrSchema.parse(raw);
  const layer = pipe.layers.find((l) => l.module_id === moduleId);
  return layer ? layerIrSchema.parse(layer) : null;
}

async function loadModule(moduleId: string): Promise<VerilogModule | null> {
  const metaPath = path.join(repoRoot, "output", "rtl", `${moduleId}.meta.json`);
  const vPath = path.join(repoRoot, "output", "rtl", `${moduleId}.v`);
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
  const targets = process.argv.slice(2);
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
      const outPath = path.join(repoRoot, "output", "reports", `${mid}.results.json`);
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
