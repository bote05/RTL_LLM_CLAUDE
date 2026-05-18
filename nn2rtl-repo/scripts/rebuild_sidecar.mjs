// Rebuild output/tb/<module>.sidecar.json from the LayerIR for one module.
// Mirrors what runAssayerDeterministic writes; useful when you need to
// re-invoke run_verilator without going through the orchestrator.

import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const moduleId = process.argv.slice(2).find((arg) => !arg.startsWith("--"));
if (!moduleId) {
  console.error("usage: node scripts/rebuild_sidecar.mjs <module_id>");
  process.exit(1);
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const registry = JSON.parse(await readFile(path.join(repoRoot, "networks.json"), "utf8"));
const networkId = process.argv.find((arg) => arg.startsWith("--network="))?.split("=", 2)[1]
  ?? process.env.NN2RTL_NETWORK_ID
  ?? registry.defaultNetworkId;
const network = registry.networks.find((entry) => entry.id === networkId);
if (!network) { console.error(`unknown network '${networkId}'`); process.exit(1); }
const outputRootRaw = process.env.NN2RTL_OUTPUT_DIR ?? network.outputDir;
const outputRoot = path.isAbsolute(outputRootRaw) ? outputRootRaw : path.join(repoRoot, outputRootRaw);
const layerIrPath = path.join(outputRoot, "layer_ir.json");
const ir = JSON.parse(await readFile(layerIrPath, "utf8"));
const layer = ir.layers.find((L) => L.module_id === moduleId);
if (!layer) { console.error(`module '${moduleId}' not in LayerIR`); process.exit(2); }

const sidecarPath = path.join(outputRoot, "tb", `${moduleId}.sidecar.json`);
const resultsPath = path.join(outputRoot, "reports", `${moduleId}.results.json`);
const tbPath      = path.join(repoRoot, "tb", "static_verilator_tb.cpp");

const sidecar = {
  module_name: moduleId,
  module_id:   moduleId,
  clock_signal: "clk",
  reset_signal: "rst_n",
  valid_in_signal:  "valid_in",
  valid_out_signal: "valid_out",
  ready_in_signal:  "ready_in",
  data_in_signal:   "data_in",
  data_out_signal:  "data_out",
  bus_bytes_per_sample:    layer.input_width_bits / 8,
  input_width_bits:        layer.input_width_bits,
  output_width_bits:       layer.output_width_bits,
  pipeline_latency_cycles: layer.pipeline_latency_cycles,
  clock_period_ns:         layer.clock_period_ns,
  golden_inputs_path:      layer.golden_inputs_path,
  golden_outputs_path:     layer.golden_outputs_path,
  results_path:            resultsPath,
  testbench_template_path: tbPath,
};

await mkdir(path.dirname(sidecarPath), { recursive: true });
await writeFile(sidecarPath, JSON.stringify(sidecar, null, 2) + "\n", "utf8");
console.log(`wrote ${sidecarPath}`);
