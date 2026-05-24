// One-off: salvage a Foundry-produced .v whose orchestrator session hung.
// Runs the deterministic Verilator assayer on the existing RTL and writes
// .results.json. Bypasses the dispatch loop entirely.
//
// Usage:  cd sdk && npx tsx salvage_assayer.ts <module_id>
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  applyContractPlan,
  CONTRACT_PLANS,
  createOrchestratorRuntime,
} from "./orchestrate.js";
import type { LayerIR, VerilogModule } from "./types.js";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");

async function main(): Promise<void> {
  const moduleId = process.argv.slice(2).find((arg) => !arg.startsWith("--"));
  if (!moduleId) {
    console.error("usage: tsx salvage_assayer.ts <module_id>");
    process.exit(1);
  }
  const runtime = createOrchestratorRuntime({});
  const outputRoot = path.resolve(repoRoot, "output");
  const rtlDir = path.join(outputRoot, "rtl");
  const verilog = await readFile(path.join(rtlDir, `${moduleId}.v`), "utf8");
  const meta = JSON.parse(
    await readFile(path.join(rtlDir, `${moduleId}.meta.json`), "utf8"),
  ) as VerilogModule;
  const module: VerilogModule = { ...meta, verilog_source: verilog };
  const pipelineIr = JSON.parse(
    await readFile(path.join(outputRoot, "layer_ir.json"), "utf8"),
  ) as { layers: LayerIR[] };
  const baseLayer = pipelineIr.layers.find((l) => l.module_id === moduleId);
  if (!baseLayer) {
    throw new Error(`Layer ${moduleId} not found in layer_ir.json`);
  }
  const specHash = meta.spec_hash ?? "";
  const plan = CONTRACT_PLANS.find(
    (p) =>
      p.id !== "flat-bus" &&
      (specHash.includes(`_io${p.id}_`) || specHash.includes(`_io${p.id}`)),
  );
  const layer = plan ? applyContractPlan(baseLayer, plan) : baseLayer;
  console.log(
    `[salvage] assayer for ${moduleId} (contract=${layer.contract_id ?? "flat-bus"})`,
  );
  const result = await runtime.assayerFn(module, layer);
  console.log(
    `[salvage] status=${result.status} timing_pass=${result.timing_pass} ` +
      `actual=${result.timing_actual_cycles} expected=${result.timing_expected_cycles} ` +
      `max_error=${result.max_error} mismatch_count=${(result as { mismatch_count?: number }).mismatch_count}`,
  );
  const resultsPath = path.join(outputRoot, "reports", `${moduleId}.results.json`);
  await writeFile(resultsPath, JSON.stringify(result, null, 2), "utf8");
  console.log(`[salvage] wrote ${resultsPath}`);
  if (result.status !== "pass") {
    process.exit(2);
  }
}

await main();
