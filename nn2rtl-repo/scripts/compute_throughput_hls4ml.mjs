// Compute per-layer + whole-pipeline throughput for the hls4ml comparison
// pipeline, mirroring the shape of `compute_throughput.mjs` for nn2rtl.
//
// Inputs:
//   - comparison/tier_a/hls4ml_out/summary.json    (built/failed status per layer)
//   - <hls_dir>/myproject_prj/solution1/syn/report/myproject_csynth.rpt
//     (HLS C-synthesis interval cycles per frame — II min/max for the
//     outermost `dataflow` region)
//   - comparison/tier_a/compare_three_way.csv      (h4_fmax_mhz per layer,
//     measured by Vivado post-synth — authoritative achievable clock)
//
// Outputs:
//   - comparison/tier_a/hls4ml_throughput_per_module.csv
//   - comparison/tier_a/hls4ml_throughput_summary.json
//
// Method:
//   fps = h4_fmax_mhz * 1e6 / interval_cycles
//   bottleneck_fps = min(per-layer fps) across BUILT layers (skips failed)

import { readFileSync, writeFileSync, existsSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const hls4mlRoot = path.join(root, "comparison", "tier_a", "hls4ml_out");
const summaryPath = path.join(hls4mlRoot, "summary.json");
const compareCsv = path.join(root, "comparison", "tier_a", "compare_three_way.csv");

const summary = JSON.parse(readFileSync(summaryPath, "utf8"));

// Parse compare_three_way.csv → { layer_id: h4_fmax_mhz }
const csvText = readFileSync(compareCsv, "utf8").trim().split(/\r?\n/);
const header = csvText[0].split(",");
const layerCol = header.indexOf("layer");
const fmaxCol = header.indexOf("h4_fmax_mhz");
const h4StatusCol = header.indexOf("h4_status");
if (layerCol < 0 || fmaxCol < 0) throw new Error("compare_three_way.csv missing layer or h4_fmax_mhz column");
const fmaxByLayer = new Map();
const h4StatusByLayer = new Map();
for (const row of csvText.slice(1)) {
  const cols = row.split(",");
  fmaxByLayer.set(cols[layerCol], parseFloat(cols[fmaxCol]) || 0);
  h4StatusByLayer.set(cols[layerCol], cols[h4StatusCol] || "");
}

// Translate a WSL /mnt/c/... path back to a local Windows path if needed.
function localize(p) {
  if (!p) return p;
  const wsl = p.match(/^\/mnt\/([a-zA-Z])\/(.*)$/);
  if (wsl) return `${wsl[1].toLowerCase()}:/${wsl[2]}`;
  return p;
}

// Scrape the outermost dataflow `Interval (min, max)` from a csynth report.
// The table we want sits right under "+ Latency:" / "    * Summary:".
function extractIntervalCycles(rptPath) {
  const txt = readFileSync(rptPath, "utf8");
  const lines = txt.split(/\r?\n/);
  // Find first table row of the form `|<cycles>|<cycles>|<abs>|<abs>|<int>|<int>|<type>|`
  // immediately after the "* Summary:" header inside "+ Latency:".
  let inLatency = false;
  let inSummary = false;
  for (let i = 0; i < lines.length; i++) {
    const L = lines[i];
    if (/^\+ Latency:/.test(L)) inLatency = true;
    else if (inLatency && /\* Summary:/.test(L)) inSummary = true;
    else if (inLatency && inSummary) {
      // Skip until we hit the body row (starts with `    |    ` and contains numbers)
      const m = L.match(/^\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*[\d.\s]+\s*\w*\s*\|\s*[\d.\s]+\s*\w*\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|/);
      if (m) {
        return { latency_min: +m[1], latency_max: +m[2], ii_min: +m[3], ii_max: +m[4] };
      }
      // Stop searching once we leave the Latency section
      if (/^\+ /.test(L) && !/Latency/.test(L)) break;
    }
  }
  return null;
}

function findCsynthRpt(hlsDir) {
  // Prefer myproject_csynth.rpt; fallback to first csynth.rpt under syn/report/
  const reportDir = path.join(hlsDir, "myproject_prj", "solution1", "syn", "report");
  if (!existsSync(reportDir)) return null;
  const preferred = path.join(reportDir, "myproject_csynth.rpt");
  if (existsSync(preferred)) return preferred;
  const fallback = readdirSync(reportDir).find((f) => f.endsWith("_csynth.rpt"));
  return fallback ? path.join(reportDir, fallback) : null;
}

const rows = [];
const skipped = [];

for (const [layerId, entry] of Object.entries(summary)) {
  const built = entry.status === "built";
  const hlsDir = localize(entry.hls_dir);
  const fmax = fmaxByLayer.get(layerId) ?? 0;

  if (!built) {
    skipped.push({ layer: layerId, why: `hls4ml_status=${entry.status}` });
    continue;
  }
  if (!hlsDir || !existsSync(hlsDir)) {
    skipped.push({ layer: layerId, why: "hls_dir missing on disk" });
    continue;
  }
  const rpt = findCsynthRpt(hlsDir);
  if (!rpt) {
    skipped.push({ layer: layerId, why: "no csynth report" });
    continue;
  }
  const iv = extractIntervalCycles(rpt);
  if (!iv) {
    skipped.push({ layer: layerId, why: "could not parse Latency table" });
    continue;
  }
  if (fmax <= 0) {
    skipped.push({ layer: layerId, why: "no h4_fmax_mhz" });
    continue;
  }
  const fps = (fmax * 1e6) / iv.ii_max;
  rows.push({
    layer: layerId,
    h4_status: h4StatusByLayer.get(layerId) ?? "",
    latency_cycles_max: iv.latency_max,
    interval_cycles_max: iv.ii_max,
    fmax_mhz: fmax,
    fps,
  });
}

rows.sort((a, b) => a.layer.localeCompare(b.layer));

const perModuleCsv = [
  "layer,h4_status,latency_cycles_max,interval_cycles_max,fmax_mhz,fps",
  ...rows.map(
    (r) =>
      `${r.layer},${r.h4_status},${r.latency_cycles_max},${r.interval_cycles_max},${r.fmax_mhz.toFixed(2)},${r.fps.toFixed(4)}`,
  ),
].join("\n");
const perModulePath = path.join(root, "comparison", "tier_a", "hls4ml_throughput_per_module.csv");
writeFileSync(perModulePath, `${perModuleCsv}\n`, "utf8");

const bottleneck = rows.reduce(
  (best, r) => (best === null || r.fps < best.fps ? r : best),
  /** @type {null | (typeof rows)[number]} */ (null),
);

const summaryOut = {
  source: "hls4ml csynth + Vivado post-synth fmax",
  layers_total: Object.keys(summary).length,
  layers_measured: rows.length,
  layers_skipped: skipped.length,
  skipped,
  bottleneck_layer: bottleneck?.layer ?? null,
  bottleneck_fps: bottleneck?.fps ?? 0,
  steady_state_fps: bottleneck?.fps ?? 0,
  method: "fps = h4_fmax_mhz * 1e6 / interval_cycles_max (HLS II = post-synth achievable interval)",
};
const summaryOutPath = path.join(root, "comparison", "tier_a", "hls4ml_throughput_summary.json");
writeFileSync(summaryOutPath, `${JSON.stringify(summaryOut, null, 2)}\n`, "utf8");

console.log(`rows written: ${rows.length}  skipped: ${skipped.length}`);
if (bottleneck) {
  console.log(`bottleneck: ${bottleneck.layer}  fps=${bottleneck.fps.toFixed(4)}  (II=${bottleneck.interval_cycles_max} cycles @ ${bottleneck.fmax_mhz.toFixed(2)} MHz)`);
}
if (skipped.length > 0) {
  console.log("skipped:", skipped.slice(0, 10));
}
console.log(`per-module CSV: ${path.relative(root, perModulePath)}`);
console.log(`summary JSON:   ${path.relative(root, summaryOutPath)}`);
