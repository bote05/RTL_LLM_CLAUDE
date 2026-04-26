// Rebuild output/tb/<module>.sidecar.json from the LayerIR for one module.
// Mirrors what runAssayerDeterministic writes; useful when you need to
// re-invoke run_verilator without going through the orchestrator.

import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";

const moduleId = process.argv[2];
if (!moduleId) {
  console.error("usage: node scripts/rebuild_sidecar.mjs <module_id>");
  process.exit(1);
}

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname).replace(/^\//, ""), "..");
const layerIrPath = path.join(repoRoot, "output", "layer_ir.json");
const ir = JSON.parse(await readFile(layerIrPath, "utf8"));
const layer = ir.layers.find((L) => L.module_id === moduleId);
if (!layer) { console.error(`module '${moduleId}' not in LayerIR`); process.exit(2); }

const sidecarPath = path.join(repoRoot, "output", "tb", `${moduleId}.sidecar.json`);
const resultsPath = path.join(repoRoot, "output", "reports", `${moduleId}.results.json`);
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
