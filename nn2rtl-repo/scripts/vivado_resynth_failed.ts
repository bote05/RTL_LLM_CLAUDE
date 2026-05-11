// Re-run Vivado synthesis ONLY on modules whose RTL is verified-correct
// (assayer passed → state went to "pass" → then Vivado infra hit a Windows
// EBUSY/file-lock race and the pipeline flipped them to "fail_abort").
//
// This script:
//   - reads each module's existing .v from output/rtl/
//   - calls run_vivado serially (no concurrency → no temp-dir collisions)
//   - writes output/reports/<module_id>.vivado.json on success
//   - flips pipeline_state.json modules[<id>] from "fail_abort" to "pass"
//
// Does NOT invoke any LLM agent. Does NOT regenerate RTL.
//
// Usage:
//   set NN2RTL_VIVADO_BIN=D:/vivado/2025.2/Vivado/bin/vivado.bat
//   npx tsx scripts/vivado_resynth_failed.ts node_conv_284 node_conv_292 node_conv_298

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { run_vivado } from "../mcp/tools.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

const DEFAULT_MODULES = ["node_conv_284", "node_conv_292", "node_conv_298"];
const DEFAULT_PART = "xczu9eg-ffvb1156-2-e";
const DEFAULT_CLOCK_NS = 20;

async function readJsonIfPresent<T>(p: string): Promise<T | null> {
  try {
    return JSON.parse(await readFile(p, "utf8")) as T;
  } catch {
    return null;
  }
}

async function resynthOne(moduleId: string): Promise<{ ok: boolean; msg: string }> {
  const rtlPath = path.join(repoRoot, "output", "rtl", `${moduleId}.v`);
  let source: string;
  try {
    source = await readFile(rtlPath, "utf8");
  } catch {
    return { ok: false, msg: `[${moduleId}] missing RTL at ${rtlPath}` };
  }
  const moduleMatch = source.match(/^\s*module\s+([A-Za-z_][A-Za-z0-9_]*)/m);
  if (!moduleMatch) return { ok: false, msg: `[${moduleId}] no module decl in RTL` };
  const moduleName = moduleMatch[1];

  console.log(`[resynth] >>> ${moduleId} (module=${moduleName})`);
  const t0 = Date.now();
  const report = await run_vivado(source, moduleName, DEFAULT_CLOCK_NS, DEFAULT_PART);
  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`[resynth]     ${elapsed}s  success=${report.success} lut=${report.lut_count} ff=${report.ff_count} dsp=${report.dsp_count} bram18=${report.bram18_count} bram36=${report.bram36_count} wns=${report.wns_ns} fmax=${report.fmax_mhz.toFixed(2)}`);

  const outPath = path.join(repoRoot, "output", "reports", `${moduleId}.vivado.json`);
  await writeFile(outPath, JSON.stringify(report, null, 2), "utf8");
  console.log(`[resynth]     wrote ${path.relative(repoRoot, outPath)}`);

  if (!report.success) {
    // Dump head of report so we can see why Vivado bailed.
    console.log(`[resynth]     ---- report head ----`);
    console.log(report.report.split(/\r?\n/).slice(0, 60).join("\n"));
    console.log(`[resynth]     ---- /report head ----`);
  }
  return { ok: report.success, msg: report.success ? "ok" : "vivado reported failure" };
}

async function flipState(moduleId: string): Promise<void> {
  const statePath = path.join(repoRoot, "output", "pipeline_state.json");
  const state = await readJsonIfPresent<{ modules: Record<string, string> }>(statePath);
  if (!state || !state.modules) return;
  if (state.modules[moduleId] !== "pass") {
    state.modules[moduleId] = "pass";
    await writeFile(statePath, JSON.stringify(state, null, 2), "utf8");
    console.log(`[resynth]     pipeline_state.modules[${moduleId}] → pass`);
  }
}

async function main(): Promise<void> {
  const modules = process.argv.slice(2).length > 0 ? process.argv.slice(2) : DEFAULT_MODULES;
  console.log("[resynth] NN2RTL_VIVADO_BIN =", process.env.NN2RTL_VIVADO_BIN ?? "(unset)");
  console.log("[resynth] modules           =", modules.join(", "));

  let passed = 0;
  let failed = 0;
  for (const moduleId of modules) {
    try {
      const r = await resynthOne(moduleId);
      if (r.ok) {
        await flipState(moduleId);
        passed += 1;
      } else {
        console.log(`[resynth]     SKIP state flip: ${r.msg}`);
        failed += 1;
      }
    } catch (err) {
      const msg = err instanceof Error ? err.stack ?? err.message : String(err);
      console.error(`[resynth] [${moduleId}] FATAL:\n${msg}`);
      failed += 1;
    }
  }

  console.log(`[resynth] done. passed=${passed} failed=${failed}`);
  if (failed > 0) process.exit(2);
}

main().catch((err) => {
  console.error("[resynth] FATAL:", err instanceof Error ? err.stack ?? err.message : String(err));
  process.exit(1);
});
