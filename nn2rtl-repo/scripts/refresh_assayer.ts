// One-off recovery: re-run the deterministic Verilator assayer on the
// canonical RTL for a given module and write `.results.json`. Used to
// recover after a stale/clobbered canonical results file.
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { applyContractPlan, CONTRACT_PLANS, createOrchestratorRuntime } from "../sdk/orchestrate.js";
import type { LayerIR, VerilogModule } from "../sdk/types.js";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");
const outputRoot = path.join(repoRoot, "output");

async function main(): Promise<void> {
  const moduleId = process.argv[2];
  if (!moduleId) {
    console.error("usage: tsx scripts/refresh_assayer.ts <module_id>");
    process.exit(1);
  }
  const runtime = createOrchestratorRuntime({});
  const rtlDir = path.join(outputRoot, "rtl");
  const verilog = await readFile(path.join(rtlDir, `${moduleId}.v`), "utf8");
  const meta = JSON.parse(await readFile(path.join(rtlDir, `${moduleId}.meta.json`), "utf8")) as VerilogModule;
  const module: VerilogModule = { ...meta, verilog_source: verilog };
  const pipelineIr = JSON.parse(
    await readFile(path.join(outputRoot, "layer_ir.json"), "utf8"),
  ) as { layers: LayerIR[] };
  const baseLayer = pipelineIr.layers.find((l) => l.module_id === moduleId);
  if (!baseLayer) throw new Error(`Layer ${moduleId} not found in layer_ir.json`);
  // Re-apply contract plan from spec_hash (mirrors inferContractIdFromSpecHash).
  const specHash = meta.spec_hash ?? "";
  const plan = CONTRACT_PLANS.find(
    (p) => p.id !== "flat-bus" && (specHash.includes(`_io${p.id}_`) || specHash.includes(`_io${p.id}`)),
  );
  const layer = plan ? applyContractPlan(baseLayer, plan) : baseLayer;
  console.log(`Running assayer for ${moduleId} (contract=${layer.contract_id ?? "flat-bus"}) ...`);
  const result = await runtime.assayerFn(module, layer);
  console.log(`status=${result.status} timing_pass=${result.timing_pass} ` +
    `actual=${result.timing_actual_cycles} expected=${result.timing_expected_cycles} ` +
    `max_error=${result.max_error}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
