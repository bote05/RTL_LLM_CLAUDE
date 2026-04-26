// Quick verilator smoke: compile Foundry's RTL via the actual run_verilator
// path and dump the per-vector debug lines we just added to the static TB.

import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { run_verilator } from "../mcp/tools.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

async function main(): Promise<void> {
  const moduleName = process.argv[2] ?? "layer1_0_conv1";
  const verilogPath = path.join(repoRoot, "output", "rtl", `${moduleName}.v`);
  const sidecarPath = path.join(repoRoot, "output", "tb", `${moduleName}.sidecar.json`);
  const source = await readFile(verilogPath, "utf8");
  console.log(`[smoke] Verilator on ${moduleName} via real run_verilator…`);
  const t0 = Date.now();
  const r = await run_verilator(source, moduleName, sidecarPath);
  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`[smoke] returned in ${elapsed}s`);
  console.log("status         =", r.status);
  console.log("status_class   =", r.status_class);
  console.log("timing_pass    =", r.timing_pass);
  console.log("timing_actual  =", r.timing_actual_cycles);
  console.log("timing_expect  =", r.timing_expected_cycles);
  console.log("verilator_stderr (head):");
  console.log((r.verilator_stderr ?? "").slice(0, 4000));
}

main().catch((err) => {
  console.error("[smoke] FATAL:", err);
  process.exit(1);
});
