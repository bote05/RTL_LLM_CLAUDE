// Quick check: run preflightVerilogModule + structuralPreflightViolations
// against the current output/rtl/layer1_0_conv1.v + LayerIR. Confirms the
// restored RTL passes the preflight bar before we try a real run.

import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { structuralPreflightViolations, preflightVerilogModule } from "../sdk/orchestrate.ts";

async function main(): Promise<void> {
  const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const registry = JSON.parse(await readFile(path.join(repoRoot, "networks.json"), "utf8"));
  const networkId = process.argv.find((arg) => arg.startsWith("--network="))?.split("=", 2)[1]
    ?? process.env.NN2RTL_NETWORK_ID
    ?? registry.defaultNetworkId;
  const network = registry.networks.find((entry: any) => entry.id === networkId);
  if (!network) throw new Error(`Unknown network '${networkId}'.`);
  const outputRoot = path.resolve(repoRoot, process.env.NN2RTL_OUTPUT_DIR ?? network.outputDir);
  const moduleId = process.argv.slice(2).find((arg) => !arg.startsWith("--")) ?? "layer1_0_conv1";
  const layerIr = JSON.parse(await readFile(path.join(outputRoot, "layer_ir.json"), "utf8"));
  const layer = layerIr.layers.find((L: any) => L.module_id === moduleId);
  if (!layer) throw new Error(`${moduleId} not in layer_ir.json`);
  const meta = JSON.parse(await readFile(path.join(outputRoot, "rtl", `${moduleId}.meta.json`), "utf8"));

  const portIssues = preflightVerilogModule(meta, layer);
  const structIssues = structuralPreflightViolations(meta, layer);

  console.log("[preflight] port-level issues:", portIssues.length);
  for (const i of portIssues) console.log("  -", i);
  console.log("[preflight] structural violations:", structIssues.length);
  for (const v of structIssues) console.log("  -", v.rule, ":", v.detail);
  if (portIssues.length === 0 && structIssues.length === 0) {
    console.log("[preflight] OK — RTL passes both preflight gates.");
  } else {
    process.exit(2);
  }
}

main().catch((err) => { console.error(err); process.exit(1); });
