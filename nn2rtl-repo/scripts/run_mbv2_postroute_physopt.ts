// MobileNetV2 — POST-ROUTE phys_opt on an already-routed dcp at a TARGET clock.
//
// Opens a ROUTED dcp (default = the banked floor mbv2_route_routed_limits_c7.dcp),
// (re)creates the clock at --clock-ns, runs phys_opt_design with a chosen directive
// (default AggressiveExplore), then re-reports post-route timing. Writes to a NEW
// tagged dcp + rpt -- NEVER over the input. Cheapest Fmax lever: targeted post-route
// optimization may recover slack on the routed netlist without a full re-place/route.
//
// Usage:
//   set NN2RTL_VIVADO_BIN=D:/vivado/2025.2/Vivado/bin/vivado.bat
//   set NN2RTL_RAM_KILL_PCT=95
//   npx tsx scripts/run_mbv2_postroute_physopt.ts [--clock-ns=7] [--threads=16]
//        [--directive=AggressiveExplore] [--tag=_physopt_aggr]
//        [--input=output/mobilenet-v2/reports/synth/checkpoints/mbv2_route_routed_limits_c7.dcp]

import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import os from "node:os";

import {
  parseVivadoReport,
  resolveVivadoCommand,
  toVivadoPath,
  withTempDir,
  VIVADO_TIMEOUT_MS,
  VIVADO_MAX_BUFFER_BYTES,
} from "../mcp/tools.ts";

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

const part = flag("part") ?? "xcu250-figd2104-2L-e";
const clockNs = Number(flag("clock-ns") ?? "7");
const threads = Number(flag("threads") ?? "16");
const directive = flag("directive") ?? "AggressiveExplore";
const ckptDir = path.join(repoRoot, "output", "mobilenet-v2", "reports", "synth", "checkpoints");
const inputRaw = flag("input") ?? path.join(ckptDir, "mbv2_route_routed_limits_c7.dcp");
const inputDcp = path.isAbsolute(inputRaw) ? inputRaw : path.resolve(repoRoot, inputRaw);
const tag = flag("tag") ?? `_physopt`;

const reportsDir = path.join(repoRoot, "output", "mobilenet-v2", "reports", "synth");
const jsonReportPath = path.join(reportsDir, `mbv2_postroute_physopt${tag}.json`);
const logPath = path.join(reportsDir, `mbv2_postroute_physopt${tag}.log`);

function tclQuote(value: string): string {
  return `"${toVivadoPath(value).replace(/(["$[\]])/g, "\\$1")}"`;
}

function buildTcl(i: { inDcp: string; outDcp: string; timing: string; util: string; timingSink: string }): string {
  return [
    `set_param general.maxThreads ${threads}`,
    `puts "NN2RTL_INFO: maxThreads requested=${threads} effective=[get_param general.maxThreads]"`,
    `puts "NN2RTL_INFO: open_checkpoint ${path.basename(i.inDcp)} (ROUTED)"`,
    `open_checkpoint ${tclQuote(i.inDcp)}`,
    `if {[llength [get_ports -quiet clk]] == 0} { error "no clk port in checkpoint" }`,
    `puts "NN2RTL_INFO: (re)create_clock clk period=${clockNs}ns"`,
    `create_clock -name clk -period ${clockNs} [get_ports clk]`,
    `puts "NN2RTL_INFO: pre-physopt timing"`,
    `report_timing_summary -no_header -no_detailed_paths`,
    `puts "NN2RTL_INFO: phys_opt_design -directive ${directive}"`,
    `phys_opt_design -directive ${directive}`,
    `write_checkpoint -force ${tclQuote(i.outDcp)}`,
    `puts "NN2RTL_INFO: post-physopt reports"`,
    `report_utilization -file ${tclQuote(i.util)}`,
    `report_timing_summary -check_timing_verbose -max_paths 30 -file ${tclQuote(i.timing)}`,
    `file copy -force ${tclQuote(i.timing)} ${tclQuote(i.timingSink)}`,
    `puts "NN2RTL_INFO: mbv2 postroute physopt complete"`,
  ].join("\n") + "\n";
}

const execFileP = promisify(execFile);

async function main(): Promise<void> {
  await mkdir(reportsDir, { recursive: true });
  await mkdir(ckptDir, { recursive: true });
  if (!existsSync(inputDcp)) throw new Error(`routed checkpoint not found: ${inputDcp}`);
  const outDcp = path.join(ckptDir, `mbv2_route_routed${tag}.dcp`);
  if (path.resolve(outDcp) === path.resolve(inputDcp)) throw new Error("refusing to overwrite input dcp; choose a different --tag");
  console.log(`[mbv2-physopt] input=${inputDcp}`);
  console.log(`[mbv2-physopt] part=${part} clock_ns=${clockNs} threads=${threads} directive=${directive} tag=${tag}`);

  const timing = path.join(ckptDir, `mbv2_route_postroute_timing${tag}.rpt`);
  const util = path.join(ckptDir, `mbv2_route_postroute_util${tag}.rpt`);

  const report = await withTempDir("nn2rtl-mbv2physopt-", async (tempDir) => {
    const timingSink = path.join(tempDir, "timing.rpt");
    const tclPath = path.join(tempDir, "physopt.tcl");
    await writeFile(tclPath, buildTcl({ inDcp: inputDcp, outDcp, timing, util, timingSink }), "utf8");

    const vivadoBin = resolveVivadoCommand(process.env);
    const vivadoArgs = ["-mode", "batch", "-source", toVivadoPath(tclPath), "-notrace"];
    const isWindowsBatch = process.platform === "win32" && /\.(bat|cmd)$/i.test(vivadoBin);
    const spawnFile = isWindowsBatch ? "cmd.exe" : vivadoBin;
    const spawnArgs = isWindowsBatch ? ["/c", vivadoBin, ...vivadoArgs] : vivadoArgs;
    console.log(`[mbv2-physopt] launching: ${spawnFile} ${spawnArgs.join(" ")}`);

    const timeoutMs = (() => {
      const e = process.env.NN2RTL_VIVADO_TIMEOUT_MS;
      return e && Number.isFinite(Number(e)) && Number(e) > 0 ? Number(e) : VIVADO_TIMEOUT_MS;
    })();

    const t0 = Date.now();
    let stdout = "", stderr = "", exitOk = true, ramKillMsg = "";
    {
      const ramKillPct = (() => {
        const v = Number(process.env.NN2RTL_RAM_KILL_PCT);
        return Number.isFinite(v) && v > 0 && v < 100 ? v : 90;
      })();
      const totalMem = os.totalmem();
      console.log(`[mbv2-physopt] RAM watchdog armed: kill at >= ${ramKillPct}% (total ${(totalMem / 1073741824).toFixed(1)}GB)`);
      const res = await new Promise<{ stdout: string; stderr: string; ok: boolean }>((resolve) => {
        const child = execFile(spawnFile, spawnArgs,
          { cwd: tempDir, env: process.env, timeout: timeoutMs, maxBuffer: VIVADO_MAX_BUFFER_BYTES },
          (err, soCb, seCb) => {
            clearInterval(poll);
            const so = typeof soCb === "string" ? soCb : (soCb?.toString() ?? "");
            const se = typeof seCb === "string" ? seCb : (seCb?.toString() ?? (err as Error | null)?.message ?? "");
            resolve({ stdout: so, stderr: se, ok: !err && !ramKillMsg });
          });
        const poll = setInterval(() => {
          const usedPct = (1 - os.freemem() / totalMem) * 100;
          if (usedPct >= ramKillPct && !ramKillMsg) {
            ramKillMsg = `[mbv2-physopt][WATCHDOG] RAM ${usedPct.toFixed(1)}% >= ${ramKillPct}% — KILLING Vivado`;
            console.error(ramKillMsg);
            clearInterval(poll);
            try { if (child.pid) execFileP("taskkill", ["/PID", String(child.pid), "/T", "/F"]).catch(() => {}); } catch { /* ignore */ }
            try { execFileP("taskkill", ["/IM", "vivado.exe", "/T", "/F"]).catch(() => {}); } catch { /* ignore */ }
          }
        }, 4000);
      });
      stdout = res.stdout; stderr = res.stderr; exitOk = res.ok;
    }
    const elapsed = (Date.now() - t0) / 1000;
    console.log(`[mbv2-physopt] vivado returned in ${elapsed.toFixed(1)}s (ok=${exitOk}${ramKillMsg ? ", RAM-KILLED" : ""})`);

    const timingText = existsSync(timingSink) ? await readFile(timingSink, "utf8") : "";
    const combined = [ramKillMsg, stdout, stderr, "--- timing ---", timingText].filter(Boolean).join("\n");
    await writeFile(logPath, combined, "utf8");
    const parsed = parseVivadoReport(combined, clockNs, part);
    parsed.success = exitOk && parsed.success;
    return { ...parsed, elapsed_s: elapsed };
  });

  await writeFile(jsonReportPath, JSON.stringify(report, null, 2), "utf8");
  const r = report as { success: boolean; setup_wns_ns?: number | null; elapsed_s: number };
  console.log(`[mbv2-physopt] wrote ${path.relative(repoRoot, jsonReportPath)}`);
  console.log(`[mbv2-physopt] success=${r.success} WNS=${r.setup_wns_ns ?? "n/a"} clock_ns=${clockNs} -> Fmax~=${r.setup_wns_ns != null ? (1000 / (clockNs - r.setup_wns_ns)).toFixed(1) : "n/a"}MHz elapsed=${r.elapsed_s.toFixed(0)}s`);
  if (!r.success) { console.error("[mbv2-physopt] FAILED/incomplete — see log"); process.exit(2); }
}

main().catch((err) => {
  console.error("[mbv2-physopt] FATAL:", err instanceof Error ? (err.stack ?? err.message) : String(err));
  process.exit(1);
});
