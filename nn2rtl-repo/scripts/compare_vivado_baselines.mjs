// Compare two Vivado baseline runs of the same RTL across two FPGA parts.
//
// Reads two parallel report directories under a network's output root:
//   output/<net>/reports/         <- one part (e.g. ZCU102, the "from" baseline)
//   output/<net>/reports_<suffix>/  <- another part (e.g. U250, the "to" baseline)
//
// For every module present in BOTH directories, emits:
//   - per-module CSV with LUT/FF/DSP/BRAM18-equiv/Fmax for each part, plus deltas
//   - per-module Markdown table (printable)
//   - aggregate summary (sums, means, medians, Fmax shift distribution)
//
// Usage:
//   node scripts/compare_vivado_baselines.mjs \
//       --network=resnet-50 \
//       --from-suffix=        \   # empty = the canonical reports/ dir
//       --to-suffix=u250 \
//       [--csv-out=...] [--md-out=...]
//
// Default output paths:
//   output/<net>/reports_<to>/_compare_<from>_vs_<to>.csv
//   output/<net>/reports_<to>/_compare_<from>_vs_<to>.md

import { readFileSync, writeFileSync, readdirSync, existsSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const args = process.argv.slice(2);

function flag(name, fallback) {
  const eq = args.find((a) => a.startsWith(`--${name}=`));
  if (eq) return eq.slice(name.length + 3);
  const idx = args.indexOf(`--${name}`);
  if (idx >= 0 && args[idx + 1] && !args[idx + 1].startsWith("--")) return args[idx + 1];
  return fallback;
}

const registry = JSON.parse(readFileSync(path.join(repoRoot, "networks.json"), "utf8"));
const networkId = flag("network") ?? process.env.NN2RTL_NETWORK_ID ?? registry.defaultNetworkId;
const network = registry.networks.find((n) => n.id === networkId);
if (!network) throw new Error(`Unknown network '${networkId}'`);
const outputRoot = path.resolve(repoRoot, process.env.NN2RTL_OUTPUT_DIR ?? network.outputDir);

const fromSuffix = flag("from-suffix", "");
const toSuffix = flag("to-suffix", "u250");
const fromDir = path.join(outputRoot, fromSuffix ? `reports_${fromSuffix}` : "reports");
const toDir = path.join(outputRoot, `reports_${toSuffix}`);

if (!existsSync(fromDir)) throw new Error(`'from' reports dir missing: ${fromDir}`);
if (!existsSync(toDir)) throw new Error(`'to' reports dir missing: ${toDir}`);

const fromLabel = fromSuffix || "zcu102";
const toLabel = toSuffix;

const csvOutDefault = path.join(toDir, `_compare_${fromLabel}_vs_${toLabel}.csv`);
const mdOutDefault = path.join(toDir, `_compare_${fromLabel}_vs_${toLabel}.md`);
const csvOut = flag("csv-out", csvOutDefault);
const mdOut = flag("md-out", mdOutDefault);

function loadVivado(dir, moduleId) {
  const p = path.join(dir, `${moduleId}.vivado.json`);
  if (!existsSync(p)) return null;
  try {
    return JSON.parse(readFileSync(p, "utf8"));
  } catch {
    return null;
  }
}

function modulesInDir(dir) {
  const out = new Set();
  for (const f of readdirSync(dir)) {
    if (f.endsWith(".vivado.json") && !f.startsWith("_")) {
      out.add(f.replace(/\.vivado\.json$/, ""));
    }
  }
  return out;
}

const fromMods = modulesInDir(fromDir);
const toMods = modulesInDir(toDir);
const common = [...fromMods].filter((m) => toMods.has(m)).sort();
const onlyFrom = [...fromMods].filter((m) => !toMods.has(m)).sort();
const onlyTo = [...toMods].filter((m) => !fromMods.has(m)).sort();

console.log(`compare: ${fromLabel} -> ${toLabel}`);
console.log(`  common modules: ${common.length}`);
console.log(`  only in ${fromLabel}: ${onlyFrom.length}`);
console.log(`  only in ${toLabel}: ${onlyTo.length}`);

function bram18eq(v) {
  return (v.bram18_count ?? 0) + 2 * (v.bram36_count ?? 0);
}

function pctChange(before, after) {
  if (!before) return after ? Infinity : 0;
  return ((after - before) / before) * 100;
}

const rows = [];
for (const m of common) {
  const a = loadVivado(fromDir, m);
  const b = loadVivado(toDir, m);
  if (!a || !b) continue;
  const aLut = a.lut_count ?? 0;
  const bLut = b.lut_count ?? 0;
  const aFf = a.ff_count ?? 0;
  const bFf = b.ff_count ?? 0;
  const aDsp = a.dsp_count ?? 0;
  const bDsp = b.dsp_count ?? 0;
  const aBram = bram18eq(a);
  const bBram = bram18eq(b);
  const aFmax = typeof a.fmax_mhz === "number" ? a.fmax_mhz : 0;
  const bFmax = typeof b.fmax_mhz === "number" ? b.fmax_mhz : 0;
  rows.push({
    module: m,
    a_success: a.success === true,
    b_success: b.success === true,
    a_lut: aLut,
    b_lut: bLut,
    d_lut_pct: pctChange(aLut, bLut),
    a_ff: aFf,
    b_ff: bFf,
    d_ff_pct: pctChange(aFf, bFf),
    a_dsp: aDsp,
    b_dsp: bDsp,
    d_dsp: bDsp - aDsp,
    a_bram18: aBram,
    b_bram18: bBram,
    d_bram18: bBram - aBram,
    a_fmax: aFmax,
    b_fmax: bFmax,
    d_fmax_pct: aFmax > 0 ? pctChange(aFmax, bFmax) : null,
  });
}

const sortDesc = [...rows].sort((x, y) => y.a_lut - x.a_lut);

const header = [
  "module",
  `${fromLabel}_success`,
  `${toLabel}_success`,
  `${fromLabel}_lut`,
  `${toLabel}_lut`,
  "d_lut_pct",
  `${fromLabel}_ff`,
  `${toLabel}_ff`,
  "d_ff_pct",
  `${fromLabel}_dsp`,
  `${toLabel}_dsp`,
  "d_dsp",
  `${fromLabel}_bram18eq`,
  `${toLabel}_bram18eq`,
  "d_bram18eq",
  `${fromLabel}_fmax_mhz`,
  `${toLabel}_fmax_mhz`,
  "d_fmax_pct",
];

const csvBody = sortDesc.map((r) =>
  [
    r.module,
    r.a_success,
    r.b_success,
    r.a_lut,
    r.b_lut,
    Number.isFinite(r.d_lut_pct) ? r.d_lut_pct.toFixed(2) : "",
    r.a_ff,
    r.b_ff,
    Number.isFinite(r.d_ff_pct) ? r.d_ff_pct.toFixed(2) : "",
    r.a_dsp,
    r.b_dsp,
    r.d_dsp,
    r.a_bram18,
    r.b_bram18,
    r.d_bram18,
    r.a_fmax.toFixed(2),
    r.b_fmax.toFixed(2),
    r.d_fmax_pct == null ? "" : r.d_fmax_pct.toFixed(2),
  ].join(","),
);

writeFileSync(csvOut, `${header.join(",")}\n${csvBody.join("\n")}\n`, "utf8");

function median(arr) {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.floor(arr.length / 2)];
}
function mean(arr) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

const aLutAll = rows.map((r) => r.a_lut);
const bLutAll = rows.map((r) => r.b_lut);
const aFmaxOk = rows.map((r) => r.a_fmax).filter((v) => v > 0);
const bFmaxOk = rows.map((r) => r.b_fmax).filter((v) => v > 0);
const dLutPct = rows
  .map((r) => r.d_lut_pct)
  .filter((v) => Number.isFinite(v));
const dFmaxPct = rows
  .map((r) => r.d_fmax_pct)
  .filter((v) => v != null && Number.isFinite(v));

const mdLines = [];
mdLines.push(`# Vivado baseline comparison: ${fromLabel.toUpperCase()} -> ${toLabel.toUpperCase()}`);
mdLines.push("");
mdLines.push(`Network: \`${networkId}\``);
mdLines.push("");
mdLines.push("## Aggregate");
mdLines.push("");
mdLines.push(`| Metric | ${fromLabel.toUpperCase()} | ${toLabel.toUpperCase()} | Shift |`);
mdLines.push("| --- | ---: | ---: | ---: |");
mdLines.push(
  `| Modules compared | ${rows.length} | ${rows.length} | -- |`,
);
mdLines.push(
  `| Sum LUT | ${aLutAll.reduce((a, b) => a + b, 0).toLocaleString()} | ${bLutAll.reduce((a, b) => a + b, 0).toLocaleString()} | ${pctChange(
    aLutAll.reduce((a, b) => a + b, 0),
    bLutAll.reduce((a, b) => a + b, 0),
  ).toFixed(2)}% |`,
);
mdLines.push(`| Mean LUT/module | ${mean(aLutAll).toFixed(0)} | ${mean(bLutAll).toFixed(0)} | -- |`);
mdLines.push(`| Median LUT/module | ${median(aLutAll)} | ${median(bLutAll)} | -- |`);
mdLines.push(
  `| Median LUT shift (per module) | -- | -- | ${median(dLutPct).toFixed(2)}% |`,
);
mdLines.push(
  `| Modules with Fmax reported | ${aFmaxOk.length} | ${bFmaxOk.length} | -- |`,
);
mdLines.push(`| Mean Fmax MHz | ${mean(aFmaxOk).toFixed(1)} | ${mean(bFmaxOk).toFixed(1)} | -- |`);
mdLines.push(`| Median Fmax MHz | ${median(aFmaxOk).toFixed(1)} | ${median(bFmaxOk).toFixed(1)} | -- |`);
mdLines.push(
  `| Median Fmax shift (per module) | -- | -- | ${dFmaxPct.length ? median(dFmaxPct).toFixed(2) : "n/a"}% |`,
);
mdLines.push("");

const heavy = sortDesc.slice(0, 15);
mdLines.push(`## Top 15 modules by ${fromLabel.toUpperCase()} LUT`);
mdLines.push("");
mdLines.push(
  `| Module | LUT ${fromLabel} | LUT ${toLabel} | ΔLUT% | DSP ${fromLabel}→${toLabel} | Fmax ${fromLabel} | Fmax ${toLabel} | ΔFmax% |`,
);
mdLines.push("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
for (const r of heavy) {
  mdLines.push(
    `| ${r.module} | ${r.a_lut.toLocaleString()} | ${r.b_lut.toLocaleString()} | ${
      Number.isFinite(r.d_lut_pct) ? r.d_lut_pct.toFixed(1) + "%" : ""
    } | ${r.a_dsp}→${r.b_dsp} | ${r.a_fmax.toFixed(0)} | ${r.b_fmax.toFixed(0)} | ${
      r.d_fmax_pct == null ? "" : r.d_fmax_pct.toFixed(1) + "%"
    } |`,
  );
}
mdLines.push("");

const outliers = rows.filter(
  (r) => Number.isFinite(r.d_lut_pct) && Math.abs(r.d_lut_pct) > 10,
);
if (outliers.length) {
  mdLines.push(`## LUT outliers (|ΔLUT%| > 10%)`);
  mdLines.push("");
  mdLines.push(
    `| Module | LUT ${fromLabel} | LUT ${toLabel} | ΔLUT% |`,
  );
  mdLines.push("| --- | ---: | ---: | ---: |");
  for (const r of outliers.sort((a, b) => Math.abs(b.d_lut_pct) - Math.abs(a.d_lut_pct))) {
    mdLines.push(
      `| ${r.module} | ${r.a_lut.toLocaleString()} | ${r.b_lut.toLocaleString()} | ${r.d_lut_pct.toFixed(1)}% |`,
    );
  }
  mdLines.push("");
} else {
  mdLines.push(`## LUT outliers (|ΔLUT%| > 10%)`);
  mdLines.push("");
  mdLines.push("_None. All modules are within ±10% LUT shift across the two chips — the toolchain switch is clean._");
  mdLines.push("");
}

if (onlyFrom.length || onlyTo.length) {
  mdLines.push("## Coverage gaps");
  mdLines.push("");
  if (onlyFrom.length) {
    mdLines.push(`Modules present in ${fromLabel} but missing from ${toLabel}: ${onlyFrom.length}`);
    mdLines.push("");
    mdLines.push("```");
    mdLines.push(onlyFrom.join("\n"));
    mdLines.push("```");
    mdLines.push("");
  }
  if (onlyTo.length) {
    mdLines.push(`Modules present in ${toLabel} but missing from ${fromLabel}: ${onlyTo.length}`);
    mdLines.push("");
    mdLines.push("```");
    mdLines.push(onlyTo.join("\n"));
    mdLines.push("```");
    mdLines.push("");
  }
}

writeFileSync(mdOut, mdLines.join("\n"), "utf8");

console.log(`wrote ${path.relative(repoRoot, csvOut)}`);
console.log(`wrote ${path.relative(repoRoot, mdOut)}`);
console.log("");
console.log(`median ΔLUT%: ${median(dLutPct).toFixed(2)}%   (negative = U250 smaller than ZCU102)`);
console.log(
  `median ΔFmax%: ${dFmaxPct.length ? median(dFmaxPct).toFixed(2) + "%" : "n/a"}   (positive = U250 faster than ZCU102)`,
);
