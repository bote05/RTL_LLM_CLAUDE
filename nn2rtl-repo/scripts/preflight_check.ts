// Quick check: run preflightVerilogModule + structuralPreflightViolations
// against the current output/rtl/layer1_0_conv1.v + LayerIR. Confirms the
// restored RTL passes the preflight bar before we try a real run.

import { readFile } from "node:fs/promises";
import { structuralPreflightViolations, preflightVerilogModule } from "../sdk/orchestrate.ts";

async function main(): Promise<void> {
  const layerIr = JSON.parse(await readFile("output/layer_ir.json", "utf8"));
  const layer = layerIr.layers.find((L: any) => L.module_id === "layer1_0_conv1");
  if (!layer) throw new Error("layer1_0_conv1 not in layer_ir.json");
  const meta = JSON.parse(await readFile("output/rtl/layer1_0_conv1.meta.json", "utf8"));

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
