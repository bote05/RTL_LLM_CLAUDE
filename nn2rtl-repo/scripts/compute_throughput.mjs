// Recompute per-module + whole-pipeline throughput from ground-truth measurements.
// Per-frame cycles  = (last_valid_out_cycle - first_valid_in_cycle) / num_vectors
// num_vectors      = per_vector.length (number of frames simulated)
// fmax_mhz         = vivado.json fmax_mhz (post-synth, xczu9eg-ffvb1156-2-e)
// fps              = fmax_mhz * 1e6 / cycles_per_frame
//
// Whole-network fps = min(per-layer fps) over the pipeline (steady-state bottleneck).
// Whole-network latency_s = sum(per-layer pipeline_fill_cycles / fmax) + 1/network_fps
// where pipeline_fill_cycles = first_valid_out_cycle - first_valid_in_cycle.

import { readFileSync, writeFileSync, existsSync } from "node:fs";
import path from "node:path";

const root = "c:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo";
const reports = path.join(root, "output", "reports");
const ir = JSON.parse(readFileSync(path.join(root, "output", "layer_ir.json"), "utf8"));

const rows = [];
const skipped = [];

for (const layer of ir.layers) {
  const mid = layer.module_id;
  const rPath = path.join(reports, `${mid}.results.json`);
  const vPath = path.join(reports, `${mid}.vivado.json`);
  if (!existsSync(rPath)) { skipped.push({ mid, why: "no results.json" }); continue; }
  if (!existsSync(vPath)) { skipped.push({ mid, why: "no vivado.json" }); continue; }
  const r = JSON.parse(readFileSync(rPath, "utf8"));
  const v = JSON.parse(readFileSync(vPath, "utf8"));
  const numVec = Array.isArray(r.per_vector) ? r.per_vector.length : 0;
  const fmax = typeof v.fmax_mhz === "number" ? v.fmax_mhz : 0;
  if (numVec === 0 || fmax === 0) {
    rows.push({ mid, op: layer.op_type, cpf: 0, fill: 0, fmax, fps: 0 });
    continue;
  }
  const span = r.last_valid_out_cycle - r.first_valid_in_cycle;
  const fill = r.first_valid_out_cycle - r.first_valid_in_cycle;
  const cpf = span / numVec;
  const fps = cpf > 0 ? (fmax * 1e6) / cpf : 0;
  rows.push({ mid, op: layer.op_type, cpf, fill, fmax, fps, numVec });
}

const header = "module_id,op_type,cycles_per_frame,pipeline_fill_cycles,fmax_mhz,fps,num_vectors";
const csv = [header, ...rows.map(r =>
  [r.mid, r.op, r.cpf.toFixed(2), r.fill ?? 0, r.fmax.toFixed(2), r.fps.toFixed(4), r.numVec ?? 0].join(",")
)].join("\n");
writeFileSync(path.join(reports, "throughput_per_module.csv"), csv + "\n", "utf8");

// Whole-network roll-up
const measurable = rows.filter(r => r.fps > 0);
const bottleneck = measurable.reduce((a, b) => (a.fps < b.fps ? a : b), measurable[0]);
const networkFps = bottleneck.fps;
const fillSecondsSum = measurable.reduce((acc, r) =>
  acc + (r.fill / (r.fmax * 1e6)), 0);
const frameSeconds = 1 / networkFps;
const e2eSeconds = fillSecondsSum + frameSeconds;

const summary = {
  layers_total: ir.layers.length,
  layers_measured: rows.length,
  layers_skipped: skipped.length,
  skipped,
  network_bottleneck_module: bottleneck.mid,
  network_bottleneck_op: bottleneck.op,
  network_bottleneck_fps: bottleneck.fps,
  network_steady_state_fps: networkFps,
  pipeline_fill_seconds_sum: fillSecondsSum,
  one_frame_seconds: frameSeconds,
  end_to_end_latency_seconds_one_image: e2eSeconds,
};
writeFileSync(path.join(reports, "throughput_summary.json"), JSON.stringify(summary, null, 2), "utf8");

console.log("rows written:", rows.length, "skipped:", skipped.length);
console.log("bottleneck:", bottleneck.mid, "fps=", bottleneck.fps.toFixed(4));
console.log("e2e latency for one image (s):", e2eSeconds.toFixed(4));
console.log("steady-state network fps:", networkFps.toFixed(4));
if (skipped.length) console.log("skipped:", skipped);
