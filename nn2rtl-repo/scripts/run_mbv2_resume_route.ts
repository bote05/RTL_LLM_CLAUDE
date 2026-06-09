// MobileNetV2 — resume place&route FROM a synth checkpoint at a TARGET clock.
//
// Companion to run_mbv2_synth.ts --synth-only (which persists mbv2_post_synth.dcp).
// This opens that checkpoint and runs opt -> place(Explore) -> phys_opt -> route(Explore)
// -> post-route phys_opt -> reports, WITHOUT redoing the ~2.7h synth pass. It RE-CREATES
// the clock at --clock-ns after open_checkpoint, so the SAME synth checkpoint can be swept
// across clock targets (the Fmax campaign) -- the checkpoint's own (synth-time) clock period
// is overridden here. routed Fmax = 1 / (clock_ns - WNS).
//
// Usage:
//   set NN2RTL_VIVADO_BIN=D:/vivado/2025.2/Vivado/bin/vivado.bat
//   set NN2RTL_VIVADO_TIMEOUT_MS=28800000
//   set NN2RTL_RAM_KILL_PCT=90
//   npx tsx scripts/run_mbv2_resume_route.ts [--clock-ns=8] [--threads=8]
//        [--input=output/mobilenet-v2/reports/synth/checkpoints/mbv2_post_synth.dcp]
//        [--tag=_c8] [--part=xcu250-figd2104-2L-e]
//
// Outputs (in output/mobilenet-v2/reports/synth/checkpoints/):
//   mbv2_route_opt{tag}.dcp / mbv2_route_placed{tag}.dcp / mbv2_route_physopt{tag}.dcp /
//   mbv2_route_routed{tag}.dcp + mbv2_route_postroute_util{tag}.rpt / _timing{tag}.rpt / _power{tag}.rpt

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
const clockNs = Number(flag("clock-ns") ?? "8");
const threads = Number(flag("threads") ?? "8");
// [FMAX 2026-06-07] --fast = CHARACTERIZATION pass: default place/route directives, NO
// phys_opt. ~half the wall-clock of the Explore+phys_opt flow. Use it for the FIRST route
// to surface the critical path + a ballpark Fmax fast (the critical-path LOCATION is the same
// as the high-effort flow; only the squeezed timing differs). The COMMITTED Fmax number is
// always taken from a full (non-fast) run -- honors the "no quality-reducing flags for the
// deliverable" rule; --fast is diagnostic only.
const fast = rawArgs.includes("--fast");
// [FMAX 2026-06-08] Placement directive. Route #1 @8ns proved the critical path is 91% ROUTE
// delay (logic only 1.15ns) -> the design is CONGESTION-limited, not logic-limited. The congestion
// map shows a tight cluster (RAMB/URAM locally 100%) + heavy SLR crossing (SLR1<->SLR0 at 50% SLL)
// driven by the high-Rent (0.66-0.75) retile_gather bridges. SSI_SpreadLogic_high spreads logic
// across the 4 dies to cut local density + SLL demand -> shorter routes -> higher Fmax. This is a
// congestion-MITIGATION (Explore-class effort) directive, NOT a quality/runtime-reducing flag.
const placeDirective = flag("place-directive") ?? "Explore";
const routeDirective = flag("route-directive") ?? "Explore";
const ckptDir = path.join(repoRoot, "output", "mobilenet-v2", "reports", "synth", "checkpoints");
// [FMAX-PBLOCK 2026-06-09] The SLR pblock floorplan (mbv2_fmax_pblock.xdc) is wired into the FULL
// synth flow after opt_design — but a --synth-only checkpoint never saw it, and resume-route would
// silently place WITHOUT the floorplan. Mirror the full flow here: read_xdc after opt (cells exist),
// before place. Placement-only -> byte-exact by construction. --no-pblock opts out; --xdc overrides.
const noPblock = rawArgs.includes("--no-pblock");
const pblockXdcRaw = flag("xdc") ?? path.join(repoRoot, "output", "mobilenet-v2", "reports", "synth", "mbv2_fmax_pblock.xdc");
const pblockXdc = path.isAbsolute(pblockXdcRaw) ? pblockXdcRaw : path.resolve(repoRoot, pblockXdcRaw);
const inputRaw = flag("input") ?? path.join(ckptDir, "mbv2_post_synth.dcp");
const inputDcp = path.isAbsolute(inputRaw) ? inputRaw : path.resolve(repoRoot, inputRaw);
const tag = flag("tag") ?? `_c${String(clockNs).replace(".", "p")}`;

const reportsDir = path.join(repoRoot, "output", "mobilenet-v2", "reports", "synth");
const jsonReportPath = path.join(reportsDir, `mbv2_route${tag}.json`);
const logPath = path.join(reportsDir, `mbv2_route${tag}.log`);

function tclQuote(value: string): string {
  return `"${toVivadoPath(value).replace(/(["$[\]])/g, "\\$1")}"`;
}

function buildTcl(i: {
  synthDcp: string; optDcp: string; placedDcp: string; physoptDcp: string; routedDcp: string;
  postRouteUtil: string; postRouteTiming: string; postRoutePower: string; congestionRpt: string;
  utilSink: string; timingSink: string;
}): string {
  return [
    `set_param general.maxThreads ${threads}`,
    `puts "NN2RTL_INFO: maxThreads requested=${threads} effective=[get_param general.maxThreads]"`,
    `puts "NN2RTL_INFO: open_checkpoint ${path.basename(i.synthDcp)}"`,
    `open_checkpoint ${tclQuote(i.synthDcp)}`,
    // Override the synth-time clock with the Fmax-campaign target. create_clock on the same
    // named clock redefines its period (Vivado applies the new constraint). Guard for a clk port.
    `if {[llength [get_ports -quiet clk]] == 0} { error "no clk port in checkpoint" }`,
    `puts "NN2RTL_INFO: (re)create_clock clk period=${clockNs}ns (target for this route)"`,
    `create_clock -name clk -period ${clockNs} [get_ports clk]`,
    `puts "NN2RTL_INFO: opt_design"`,
    `opt_design`,
    `write_checkpoint -force ${tclQuote(i.optDcp)}`,
    // SLR floorplan pblock: read AFTER opt (cells exist), BEFORE place — same wiring as the full
    // synth flow (a --synth-only checkpoint never saw the XDC). catch + warn so a missing/renamed
    // cell in the XDC degrades to no-floorplan instead of failing the route.
    ...(noPblock
      ? [`puts "NN2RTL_INFO: pblock skipped (--no-pblock)"`]
      : [`if {[file exists ${tclQuote(pblockXdc)}]} { puts "NN2RTL_INFO: read_xdc pblock"; catch { read_xdc ${tclQuote(pblockXdc)} } err; if {$err ne ""} { puts "NN2RTL_WARN: pblock read: $err" } } else { puts "NN2RTL_WARN: pblock XDC missing -> placing without floorplan" }`]),
    // place/route. --fast (characterization): default directives, NO phys_opt (~half wall-clock,
    // same critical-path LOCATION). default (committed Fmax): place Explore -> phys_opt ->
    // route Explore -> post-route phys_opt (HIGH-QUALITY, no Fmax-reducing flags).
    ...(fast
      ? [
          `puts "NN2RTL_INFO: [FAST] place_design (default directive)"`,
          `place_design`,
          `write_checkpoint -force ${tclQuote(i.placedDcp)}`,
          `puts "NN2RTL_INFO: [FAST] route_design (default directive)"`,
          `route_design`,
          `write_checkpoint -force ${tclQuote(i.routedDcp)}`,
        ]
      : [
          `puts "NN2RTL_INFO: place_design (directive=${placeDirective})"`,
          `place_design -directive ${placeDirective}`,
          `write_checkpoint -force ${tclQuote(i.placedDcp)}`,
          // Congestion region map (post-place) -- the DEFINITIVE localization (which modules/tiles
          // are hot, SLR-crossing demand) for floorplanning. Captured even if route later stalls.
          `puts "NN2RTL_INFO: report_design_analysis -congestion (post-place map)"`,
          `catch { report_design_analysis -congestion -file ${tclQuote(i.congestionRpt)} }`,
          `puts "NN2RTL_INFO: phys_opt_design (pre-route timing closure)"`,
          `phys_opt_design`,
          `write_checkpoint -force ${tclQuote(i.physoptDcp)}`,
          `puts "NN2RTL_INFO: route_design (directive=${routeDirective})"`,
          `route_design -directive ${routeDirective}`,
          `write_checkpoint -force ${tclQuote(i.routedDcp)}`,
          `puts "NN2RTL_INFO: post-route phys_opt_design (final timing closure)"`,
          `catch { phys_opt_design }`,
          `write_checkpoint -force ${tclQuote(i.routedDcp)}`,
        ]),
    `puts "NN2RTL_INFO: post-route reports"`,
    `report_utilization -file ${tclQuote(i.postRouteUtil)}`,
    `report_timing_summary -check_timing_verbose -max_paths 30 -file ${tclQuote(i.postRouteTiming)}`,
    `report_power -file ${tclQuote(i.postRoutePower)}`,
    `file copy -force ${tclQuote(i.postRouteUtil)} ${tclQuote(i.utilSink)}`,
    `file copy -force ${tclQuote(i.postRouteTiming)} ${tclQuote(i.timingSink)}`,
    `puts "NN2RTL_INFO: mbv2 resume-route complete"`,
  ].join("\n") + "\n";
}

const execFileP = promisify(execFile);

async function main(): Promise<void> {
  await mkdir(reportsDir, { recursive: true });
  await mkdir(ckptDir, { recursive: true });
  if (!existsSync(inputDcp)) throw new Error(`synth checkpoint not found: ${inputDcp}`);
  console.log(`[mbv2-route] input=${inputDcp}`);
  console.log(`[mbv2-route] part=${part} clock_ns=${clockNs} threads=${threads} tag=${tag}`);

  const optDcp = path.join(ckptDir, `mbv2_route_opt${tag}.dcp`);
  const placedDcp = path.join(ckptDir, `mbv2_route_placed${tag}.dcp`);
  const physoptDcp = path.join(ckptDir, `mbv2_route_physopt${tag}.dcp`);
  const routedDcp = path.join(ckptDir, `mbv2_route_routed${tag}.dcp`);
  const postRouteUtil = path.join(ckptDir, `mbv2_route_postroute_util${tag}.rpt`);
  const postRouteTiming = path.join(ckptDir, `mbv2_route_postroute_timing${tag}.rpt`);
  const postRoutePower = path.join(ckptDir, `mbv2_route_postroute_power${tag}.rpt`);
  const congestionRpt = path.join(ckptDir, `mbv2_route_congestion${tag}.rpt`);

  const report = await withTempDir("nn2rtl-mbv2route-", async (tempDir) => {
    const utilSink = path.join(tempDir, "route_util.rpt");
    const timingSink = path.join(tempDir, "route_timing.rpt");
    const tclPath = path.join(tempDir, "mbv2_route.tcl");
    await writeFile(tclPath, buildTcl({
      synthDcp: inputDcp, optDcp, placedDcp, physoptDcp, routedDcp,
      postRouteUtil, postRouteTiming, postRoutePower, congestionRpt, utilSink, timingSink,
    }), "utf8");

    const vivadoBin = resolveVivadoCommand(process.env);
    const vivadoArgs = ["-mode", "batch", "-source", toVivadoPath(tclPath), "-notrace"];
    const isWindowsBatch = process.platform === "win32" && /\.(bat|cmd)$/i.test(vivadoBin);
    const spawnFile = isWindowsBatch ? "cmd.exe" : vivadoBin;
    const spawnArgs = isWindowsBatch ? ["/c", vivadoBin, ...vivadoArgs] : vivadoArgs;
    console.log(`[mbv2-route] launching: ${spawnFile} ${spawnArgs.join(" ")}`);

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
      console.log(`[mbv2-route] RAM watchdog armed: kill at >= ${ramKillPct}% used (total ${(totalMem / 1073741824).toFixed(1)}GB, poll 4s)`);
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
          const freeGB = os.freemem() / 1073741824;
          const usedPct = (1 - os.freemem() / totalMem) * 100;
          if (usedPct >= ramKillPct && !ramKillMsg) {
            ramKillMsg = `[mbv2-route][WATCHDOG] RAM ${usedPct.toFixed(1)}% used >= ${ramKillPct}% (free ${freeGB.toFixed(1)}GB) — KILLING Vivado tree; resume from placed/physopt dcp`;
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
    console.log(`[mbv2-route] vivado returned in ${elapsed.toFixed(1)}s (ok=${exitOk}${ramKillMsg ? ", RAM-KILLED" : ""})`);

    const utilText = existsSync(utilSink) ? await readFile(utilSink, "utf8") : "";
    const timingText = existsSync(timingSink) ? await readFile(timingSink, "utf8") : "";
    const combined = [ramKillMsg, stdout, stderr, "--- util ---", utilText, "--- timing ---", timingText].filter(Boolean).join("\n");
    await writeFile(logPath, combined, "utf8");
    const parsed = parseVivadoReport(combined, clockNs, part);
    parsed.success = exitOk && parsed.success;
    return { ...parsed, elapsed_s: elapsed };
  });

  await writeFile(jsonReportPath, JSON.stringify(report, null, 2), "utf8");
  const r = report as { success: boolean; lut_count?: number; setup_wns_ns?: number | null; elapsed_s: number };
  console.log(`[mbv2-route] wrote ${path.relative(repoRoot, jsonReportPath)}`);
  console.log(`[mbv2-route] success=${r.success} WNS=${r.setup_wns_ns ?? "n/a"} clock_ns=${clockNs} -> Fmax~=${r.setup_wns_ns != null ? (1000 / (clockNs - r.setup_wns_ns)).toFixed(1) : "n/a"}MHz elapsed=${r.elapsed_s.toFixed(0)}s`);
  if (!r.success) { console.error("[mbv2-route] FAILED/incomplete — see log + checkpoints"); process.exit(2); }
}

main().catch((err) => {
  console.error("[mbv2-route] FATAL:", err instanceof Error ? (err.stack ?? err.message) : String(err));
  process.exit(1);
});
