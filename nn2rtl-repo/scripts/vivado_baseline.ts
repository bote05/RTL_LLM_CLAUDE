// Re-baseline every passing module of a network against a given FPGA part.
//
// Unlike vivado_resynth_failed.ts (which targets only fail_abort modules and
// overwrites the canonical reports/<m>.vivado.json), this script:
//   - iterates over every module marked "pass" in pipeline_state.json
//   - writes results to a parallel directory (output/reports_<suffix>/) so the
//     existing ZCU102 baseline reports are preserved
//   - prints per-module timing and an aggregate summary at the end
//   - never touches pipeline_state.json
//
// Usage:
//   set NN2RTL_VIVADO_BIN=D:/vivado/2025.2/Vivado/bin/vivado.bat
//   npx tsx scripts/vivado_baseline.ts \
//       --network=resnet-50 \
//       --part=xcu250-figd2104-2L-e \
//       --reports-suffix=u250 \
//       [--clock-ns=20] \
//       [--only=node_conv_196,node_conv_298]
//
// Output goes to:
//   output/<network>/reports_u250/<module>.vivado.json
//   output/<network>/reports_u250/_aggregate.json
//
// For the default network (ResNet-50), output/ is the network root (legacy
// layout). For other networks it is output/<network>/.

import { readFile, writeFile, mkdir, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readFileSync } from "node:fs";

import { run_vivado } from "../mcp/tools.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const rawArgs = process.argv.slice(2);

function flag(name: string, fallback?: string): string | undefined {
  const eq = rawArgs.find((a) => a.startsWith(`--${name}=`));
  if (eq) return eq.slice(name.length + 3);
  const idx = rawArgs.indexOf(`--${name}`);
  if (idx >= 0 && rawArgs[idx + 1] && !rawArgs[idx + 1].startsWith("--")) return rawArgs[idx + 1];
  return fallback;
}

const registry = JSON.parse(readFileSync(path.join(repoRoot, "networks.json"), "utf8")) as {
  defaultNetworkId: string;
  networks: Array<{ id: string; outputDir: string }>;
};
const networkId = flag("network") ?? process.env.NN2RTL_NETWORK_ID ?? registry.defaultNetworkId;
const network = registry.networks.find((entry) => entry.id === networkId);
if (!network) throw new Error(`Unknown network '${networkId}'.`);
const outputRoot = path.resolve(repoRoot, process.env.NN2RTL_OUTPUT_DIR ?? network.outputDir);
process.env.NN2RTL_NETWORK_ID = networkId;
process.env.NN2RTL_OUTPUT_DIR = outputRoot;

const part = flag("part") ?? "xcu250-figd2104-2L-e";
const suffix = (flag("reports-suffix") ?? part.split("-")[0].replace(/^xc/, "")).toLowerCase();
const clockNs = Number(flag("clock-ns") ?? "20");
const onlyList = flag("only");
const onlySet = onlyList ? new Set(onlyList.split(",").map((s) => s.trim()).filter(Boolean)) : null;

const reportsDir = path.join(outputRoot, `reports_${suffix}`);

async function ensureDir(p: string) {
  await mkdir(p, { recursive: true }).catch(() => {});
}

async function exists(p: string): Promise<boolean> {
  try {
    await stat(p);
    return true;
  } catch {
    return false;
  }
}

type ModuleResult = {
  module_id: string;
  success: boolean;
  lut: number;
  ff: number;
  dsp: number;
  bram18: number;
  bram36: number;
  fmax_mhz: number;
  wns_ns: number | null;
  elapsed_s: number;
};

async function baselineOne(moduleId: string): Promise<ModuleResult | null> {
  const rtlPath = path.join(outputRoot, "rtl", `${moduleId}.v`);
  if (!(await exists(rtlPath))) {
    console.log(`[baseline] [${moduleId}] SKIP: no RTL at ${rtlPath}`);
    return null;
  }
  const source = await readFile(rtlPath, "utf8");
  const moduleMatch = source.match(/^\s*module\s+([A-Za-z_][A-Za-z0-9_]*)/m);
  if (!moduleMatch) {
    console.log(`[baseline] [${moduleId}] SKIP: no module decl in RTL`);
    return null;
  }
  const moduleName = moduleMatch[1];

  const t0 = Date.now();
  console.log(`[baseline] >>> ${moduleId} (module=${moduleName})`);
  const report = await run_vivado(source, moduleName, clockNs, part);
  const elapsed = (Date.now() - t0) / 1000;

  const result: ModuleResult = {
    module_id: moduleId,
    success: report.success === true,
    lut: report.lut_count ?? 0,
    ff: report.ff_count ?? 0,
    dsp: report.dsp_count ?? 0,
    bram18: report.bram18_count ?? 0,
    bram36: report.bram36_count ?? 0,
    fmax_mhz: typeof report.fmax_mhz === "number" ? report.fmax_mhz : 0,
    wns_ns: typeof report.wns_ns === "number" ? report.wns_ns : null,
    elapsed_s: elapsed,
  };

  console.log(
    `[baseline]     ${elapsed.toFixed(1)}s  success=${result.success}  ` +
      `lut=${result.lut}  ff=${result.ff}  dsp=${result.dsp}  ` +
      `bram18=${result.bram18}  bram36=${result.bram36}  ` +
      `wns=${result.wns_ns}  fmax=${result.fmax_mhz.toFixed(2)}`,
  );

  const outPath = path.join(reportsDir, `${moduleId}.vivado.json`);
  await writeFile(outPath, JSON.stringify(report, null, 2), "utf8");

  if (!report.success) {
    const head = (report.report || "").split(/\r?\n/).slice(0, 40).join("\n");
    console.log(`[baseline]     ---- report head ----`);
    console.log(head);
    console.log(`[baseline]     ---- /report head ----`);
  }

  return result;
}

function median(arr: number[]): number {
  if (arr.length === 0) return 0;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

async function main(): Promise<void> {
  await ensureDir(reportsDir);

  const statePath = path.join(outputRoot, "pipeline_state.json");
  if (!(await exists(statePath))) {
    throw new Error(`pipeline_state.json not found at ${statePath}`);
  }
  const state = JSON.parse(await readFile(statePath, "utf8")) as {
    modules: Record<string, string>;
  };
  let modules = Object.entries(state.modules)
    .filter(([, status]) => status === "pass")
    .map(([id]) => id);
  if (onlySet) modules = modules.filter((m) => onlySet.has(m));

  console.log(`[baseline] NN2RTL_VIVADO_BIN = ${process.env.NN2RTL_VIVADO_BIN ?? "(unset)"}`);
  console.log(`[baseline] network          = ${networkId}`);
  console.log(`[baseline] part             = ${part}`);
  console.log(`[baseline] reports dir      = ${path.relative(repoRoot, reportsDir)}`);
  console.log(`[baseline] modules          = ${modules.length}${onlySet ? ` (filtered by --only)` : ""}`);

  const results: ModuleResult[] = [];
  const failed: string[] = [];
  const skipped: string[] = [];

  for (const moduleId of modules) {
    try {
      const r = await baselineOne(moduleId);
      if (r === null) {
        skipped.push(moduleId);
      } else if (!r.success) {
        failed.push(moduleId);
        results.push(r);
      } else {
        results.push(r);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`[baseline] [${moduleId}] FATAL: ${msg}`);
      failed.push(moduleId);
    }
  }

  const ok = results.filter((r) => r.success);
  const luts = ok.map((r) => r.lut);
  const ffs = ok.map((r) => r.ff);
  const dsps = ok.map((r) => r.dsp);
  const bram18eq = ok.map((r) => r.bram18 + 2 * r.bram36);
  const fmaxs = ok.map((r) => r.fmax_mhz).filter((v) => v > 0);

  const agg = {
    network: networkId,
    part,
    reports_suffix: suffix,
    modules_total: modules.length,
    modules_success: ok.length,
    modules_failed: failed.length,
    modules_skipped: skipped.length,
    sum_lut: luts.reduce((a, b) => a + b, 0),
    sum_ff: ffs.reduce((a, b) => a + b, 0),
    sum_dsp: dsps.reduce((a, b) => a + b, 0),
    sum_bram18_equiv: bram18eq.reduce((a, b) => a + b, 0),
    lut: { mean: mean(luts), median: median(luts), max: Math.max(0, ...luts) },
    ff: { mean: mean(ffs), median: median(ffs), max: Math.max(0, ...ffs) },
    dsp: { mean: mean(dsps), median: median(dsps), max: Math.max(0, ...dsps) },
    bram18_equiv: { mean: mean(bram18eq), median: median(bram18eq), max: Math.max(0, ...bram18eq) },
    fmax: {
      n_with_timing: fmaxs.length,
      mean: mean(fmaxs),
      median: median(fmaxs),
      min: fmaxs.length ? Math.min(...fmaxs) : 0,
      max: fmaxs.length ? Math.max(...fmaxs) : 0,
    },
    failed_modules: failed,
    skipped_modules: skipped,
  };

  const aggPath = path.join(reportsDir, "_aggregate.json");
  await writeFile(aggPath, JSON.stringify(agg, null, 2), "utf8");

  console.log("");
  console.log("[baseline] aggregate:");
  console.log(`  success=${agg.modules_success}  failed=${agg.modules_failed}  skipped=${agg.modules_skipped}`);
  console.log(`  sum  LUT=${agg.sum_lut}  FF=${agg.sum_ff}  DSP=${agg.sum_dsp}  BRAM18-eq=${agg.sum_bram18_equiv}`);
  console.log(`  mean LUT=${agg.lut.mean.toFixed(0)}  FF=${agg.ff.mean.toFixed(0)}  DSP=${agg.dsp.mean.toFixed(2)}  BRAM18-eq=${agg.bram18_equiv.mean.toFixed(1)}`);
  console.log(`  Fmax MHz (${agg.fmax.n_with_timing} modules with timing): mean=${agg.fmax.mean.toFixed(1)}  median=${agg.fmax.median.toFixed(1)}`);
  console.log(`  wrote ${path.relative(repoRoot, aggPath)}`);

  if (failed.length > 0) process.exit(2);
}

main().catch((err) => {
  console.error("[baseline] FATAL:", err instanceof Error ? err.stack ?? err.message : String(err));
  process.exit(1);
});
