// Surgical-pblock route experiment for the chan_window ResNet-50 design.
//
// The chan_window route plateaus at 2022 node overlaps (logic/congestion-bound,
// NOT BRAM-bound: global routing util only 32-38%). The overlapping nets are the
// deep-3x3 line-buffer-window (lbw) mux fanout of node_conv_220/240/292 + the
// residual adds (add_13/14/15) + relu beat-buffers, all crammed into clock-region
// windows that are RAMB-100% (incidental) but only LUT 67-82%. A surgical pblock
// that SPREADS that logic into adjacent free-LUT area can relieve the contention
// WITHOUT the global Fmax trade of AltSpreadLogic.
//
// Two modes:
//   --probe-only   open the fully-placed checkpoint and dump (a) which clock
//                  regions the hotspot cells currently occupy and (b) per-clock-
//                  region SLICE/RAMB36 usage, so the pblock geometry is grounded
//                  in real placement data, not guessed. READ-ONLY, fast, low RAM.
//   (default)      open the opt checkpoint, read a pblock XDC, then
//                  place_design Explore -> phys_opt -> route_design Explore ->
//                  reports. Heavy (multi-hour). Same RAM watchdog as the resume
//                  script (kill the Vivado tree at NN2RTL_RAM_KILL_PCT, default 90).
//
// Usage:
//   set NN2RTL_VIVADO_BIN=D:/vivado/2025.2/Vivado/bin/vivado.bat
//   npx tsx scripts/run_pblock_route.ts --probe-only \
//     [--placed=output/reports_integrated/checkpoints/first_light_placed_chanwindow.dcp]
//   npx tsx scripts/run_pblock_route.ts \
//     --pblock-xdc=output/reports_integrated/pblock_hotspot.xdc \
//     [--input=output/reports_integrated/checkpoints/first_light_opt_chanwindow.dcp] \
//     [--clock-ns=40] [--threads=8] [--tag=_pblock]

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
function hasFlag(name: string): boolean {
  return rawArgs.includes(`--${name}`);
}

const probeOnly = hasFlag("probe-only");
const part = flag("part") ?? "xcu250-figd2104-2L-e";
const clockNs = Number(flag("clock-ns") ?? "40");
const threads = Number(flag("threads") ?? "8");
const safeCheckpointDir = path.join(repoRoot, "output", "reports_integrated", "checkpoints");
const reportsDir = path.join(repoRoot, "output", "reports_integrated");

function abs(p: string): string {
  return path.isAbsolute(p) ? p : path.resolve(repoRoot, p);
}

const placedRaw = flag("placed") ?? path.join(safeCheckpointDir, "first_light_placed_chanwindow.dcp");
const placedDcp = abs(placedRaw);
const inputRaw = flag("input") ?? path.join(safeCheckpointDir, "first_light_opt_chanwindow.dcp");
const inputDcp = abs(inputRaw);
const pblockXdcRaw = flag("pblock-xdc");
const pblockXdc = pblockXdcRaw ? abs(pblockXdcRaw) : "";
const tag = flag("tag") ?? "_pblock";

function tclQuote(value: string): string {
  return `"${toVivadoPath(value).replace(/(["$[\]])/g, "\\$1")}"`;
}

const HOTSPOT_PATTERNS = [
  "u_node_conv_220/lbw",
  "u_node_conv_240/lbw",
  "u_node_conv_242/dp",
  "u_node_conv_292/lbw",
  "u_node_add_6",
  "u_node_add_13",
  "u_node_add_14",
  "u_node_add_15",
  "u_node_relu_41",
  "u_node_relu_42",
  "u_node_relu_48",
];

function buildProbeTcl(probeRpt: string): string {
  const lines: string[] = [
    `set_param general.maxThreads ${threads}`,
    `puts "NN2RTL_INFO: opening placed checkpoint for probe"`,
    `open_checkpoint ${tclQuote(placedDcp)}`,
    `set fh [open ${tclQuote(probeRpt)} w]`,
    `puts $fh "=== device clock regions ==="`,
    `puts $fh "  [lsort [get_clock_regions]]"`,
    `puts $fh ""`,
    `puts $fh "=== per-clock-region SLICE / RAMB36 / URAM usage (used/total) ==="`,
    `foreach cr [lsort [get_clock_regions]] {`,
    `  set st [llength [get_sites -quiet -of_objects $cr -filter {SITE_TYPE =~ SLICE*}]]`,
    `  set su [llength [get_sites -quiet -of_objects $cr -filter {SITE_TYPE =~ SLICE* && IS_USED}]]`,
    `  set rt [llength [get_sites -quiet -of_objects $cr -filter {SITE_TYPE =~ RAMB36*}]]`,
    `  set ru [llength [get_sites -quiet -of_objects $cr -filter {SITE_TYPE =~ RAMB36* && IS_USED}]]`,
    `  set ut [llength [get_sites -quiet -of_objects $cr -filter {SITE_TYPE =~ URAM*}]]`,
    `  set uu [llength [get_sites -quiet -of_objects $cr -filter {SITE_TYPE =~ URAM* && IS_USED}]]`,
    `  set sp 0 ; if {$st>0} { set sp [expr {100*$su/$st}] }`,
    `  set rp 0 ; if {$rt>0} { set rp [expr {100*$ru/$rt}] }`,
    `  set up 0 ; if {$ut>0} { set up [expr {100*$uu/$ut}] }`,
    `  puts $fh [format "  %-8s SLICE %5d/%-5d (%3d%%)  RAMB36 %4d/%-4d (%3d%%)  URAM %3d/%-3d (%3d%%)" $cr $su $st $sp $ru $rt $rp $uu $ut $up]`,
    `}`,
    `puts $fh ""`,
    `puts $fh "=== hotspot module placement (clock regions occupied + cell/slice extent) ==="`,
  ];
  for (const pat of HOTSPOT_PATTERNS) {
    lines.push(
      `set cells [get_cells -quiet -hier -filter {NAME =~ "*${pat}*"}]`,
      `if {[llength $cells]==0} {`,
      `  puts $fh "  ${pat}: 0 cells"`,
      `} else {`,
      `  set crs [lsort -unique [get_clock_regions -quiet -of_objects $cells]]`,
      `  set sites [get_sites -quiet -of_objects $cells]`,
      `  puts $fh [format "  %-26s %6d cells, %5d sites, CRs: %s" "${pat}" [llength $cells] [llength $sites] $crs]`,
      `}`,
    );
  }
  lines.push(
    `puts $fh ""`,
    `puts $fh "=== existing pblocks ==="`,
    `foreach pb [get_pblocks -quiet] { puts $fh "  $pb : [get_property GRID_RANGES $pb]" }`,
    `if {[llength [get_pblocks -quiet]]==0} { puts $fh "  (none)" }`,
    `close $fh`,
    `puts "NN2RTL_INFO: probe complete -> ${toVivadoPath(probeRpt)}"`,
  );
  return lines.join("\n") + "\n";
}

function buildPblockRouteTcl(input: {
  optDcp: string;
  pblockXdc: string;
  placedDcp: string;
  physoptDcp: string;
  routedDcp: string;
  postRouteUtil: string;
  postRouteTiming: string;
  routeStatus: string;
}): string {
  return [
    `set_param general.maxThreads ${threads}`,
    `puts "NN2RTL_INFO: requested general.maxThreads=${threads}, effective=[get_param general.maxThreads]"`,
    `puts "NN2RTL_INFO: opening opt checkpoint"`,
    `open_checkpoint ${tclQuote(input.optDcp)}`,
    `puts "NN2RTL_INFO: applying pblock xdc"`,
    `read_xdc ${tclQuote(input.pblockXdc)}`,
    `puts "NN2RTL_INFO: pblocks now defined: [get_pblocks -quiet]"`,
    `puts "NN2RTL_INFO: starting place_design (directive=Explore) with pblock"`,
    `place_design -directive Explore`,
    `write_checkpoint -force ${tclQuote(input.placedDcp)}`,
    `puts "NN2RTL_INFO: starting phys_opt_design (timing closure)"`,
    `phys_opt_design`,
    `write_checkpoint -force ${tclQuote(input.physoptDcp)}`,
    `puts "NN2RTL_INFO: starting route_design (directive=Explore) with pblock"`,
    `route_design -directive Explore`,
    `write_checkpoint -force ${tclQuote(input.routedDcp)}`,
    `puts "NN2RTL_INFO: post-route phys_opt_design (final timing closure)"`,
    `catch { phys_opt_design }`,
    `write_checkpoint -force ${tclQuote(input.routedDcp)}`,
    `puts "NN2RTL_INFO: route status"`,
    `report_route_status -file ${tclQuote(input.routeStatus)}`,
    `puts "NN2RTL_INFO: post-route utilization"`,
    `report_utilization -file ${tclQuote(input.postRouteUtil)}`,
    `puts "NN2RTL_INFO: post-route timing"`,
    `report_timing_summary -max_paths 20 -file ${tclQuote(input.postRouteTiming)}`,
    `puts "NN2RTL_INFO: pblock route complete"`,
  ].join("\n") + "\n";
}

const execFileP = promisify(execFile);

async function main(): Promise<void> {
  await mkdir(reportsDir, { recursive: true });
  await mkdir(safeCheckpointDir, { recursive: true });

  if (probeOnly) {
    if (!existsSync(placedDcp)) throw new Error(`placed checkpoint not found: ${placedDcp}`);
    console.log(`[pblock-probe] placed dcp: ${placedDcp}`);
  } else {
    if (!existsSync(inputDcp)) throw new Error(`opt checkpoint not found: ${inputDcp}`);
    if (!pblockXdc || !existsSync(pblockXdc)) throw new Error(`--pblock-xdc not found: ${pblockXdc}`);
    console.log(`[pblock-route] opt dcp: ${inputDcp}`);
    console.log(`[pblock-route] pblock xdc: ${pblockXdc}`);
    console.log(`[pblock-route] part=${part} clock_ns=${clockNs} threads=${threads} tag=${tag}`);
  }

  const probeRptSafe = path.join(reportsDir, "pblock_probe.rpt");
  const logPath = path.join(reportsDir, probeOnly ? "pblock_probe.log" : "pblock_route.log");
  const jsonReportPath = path.join(reportsDir, probeOnly ? "pblock_probe.json" : "pblock_route.json");

  const placedDcpSafe = path.join(safeCheckpointDir, `first_light_placed${tag}.dcp`);
  const physoptDcpSafe = path.join(safeCheckpointDir, `first_light_physopt${tag}.dcp`);
  const routedDcpSafe = path.join(safeCheckpointDir, `first_light_routed${tag}.dcp`);
  const postRouteUtilSafe = path.join(safeCheckpointDir, `first_light_postroute_util${tag}.rpt`);
  const postRouteTimingSafe = path.join(safeCheckpointDir, `first_light_postroute_timing${tag}.rpt`);
  const routeStatusSafe = path.join(safeCheckpointDir, `first_light_route_status${tag}.rpt`);

  const report = await withTempDir("nn2rtl-pblock-", async (tempDir) => {
    const utilSink = path.join(tempDir, "util.rpt");
    const timingSink = path.join(tempDir, "timing.rpt");
    const tclPath = path.join(tempDir, probeOnly ? "probe.tcl" : "pblock_route.tcl");

    if (probeOnly) {
      await writeFile(tclPath, buildProbeTcl(probeRptSafe), "utf8");
    } else {
      await writeFile(
        tclPath,
        buildPblockRouteTcl({
          optDcp: inputDcp,
          pblockXdc,
          placedDcp: placedDcpSafe,
          physoptDcp: physoptDcpSafe,
          routedDcp: routedDcpSafe,
          postRouteUtil: postRouteUtilSafe,
          postRouteTiming: postRouteTimingSafe,
          routeStatus: routeStatusSafe,
        }),
        "utf8",
      );
    }

    const vivadoBin = resolveVivadoCommand(process.env);
    const vivadoArgs = ["-mode", "batch", "-source", toVivadoPath(tclPath), "-notrace"];
    const isWindowsBatch = process.platform === "win32" && /\.(bat|cmd)$/i.test(vivadoBin);
    const spawnFile = isWindowsBatch ? "cmd.exe" : vivadoBin;
    const spawnArgs = isWindowsBatch ? ["/c", vivadoBin, ...vivadoArgs] : vivadoArgs;

    console.log(`[pblock] launching vivado in ${tempDir} (${probeOnly ? "PROBE" : "ROUTE"})`);
    const t0 = Date.now();
    let ramKillMsg = "";
    const timeoutMs = (() => {
      const envVal = process.env.NN2RTL_VIVADO_TIMEOUT_MS;
      if (envVal && Number.isFinite(Number(envVal)) && Number(envVal) > 0) return Number(envVal);
      // probe is fast; only the full route needs the long timeout
      return probeOnly ? 30 * 60 * 1000 : VIVADO_TIMEOUT_MS;
    })();
    const ramKillPct = (() => {
      const v = Number(process.env.NN2RTL_RAM_KILL_PCT);
      return Number.isFinite(v) && v > 0 && v < 100 ? v : 90;
    })();
    const totalMem = os.totalmem();
    console.log(
      `[pblock] RAM watchdog armed: kill at >= ${ramKillPct}% used ` +
        `(total ${(totalMem / 1073741824).toFixed(1)}GB, poll 4s), timeout ${(timeoutMs / 60000).toFixed(0)}min`,
    );
    const res = await new Promise<{ stdout: string; stderr: string; ok: boolean }>((resolve) => {
      const child = execFile(
        spawnFile,
        spawnArgs,
        { cwd: tempDir, env: process.env, timeout: timeoutMs, maxBuffer: VIVADO_MAX_BUFFER_BYTES },
        (err, soCb, seCb) => {
          clearInterval(poll);
          const so = typeof soCb === "string" ? soCb : (soCb?.toString() ?? "");
          const se = typeof seCb === "string" ? seCb : (seCb?.toString() ?? (err as Error | null)?.message ?? "");
          resolve({ stdout: so, stderr: se, ok: !err && !ramKillMsg });
        },
      );
      const poll = setInterval(() => {
        const freeGB = os.freemem() / 1073741824;
        const usedPct = (1 - os.freemem() / totalMem) * 100;
        if (usedPct >= ramKillPct && !ramKillMsg) {
          ramKillMsg =
            `[pblock][WATCHDOG] RAM ${usedPct.toFixed(1)}% used >= ${ramKillPct}% (free ${freeGB.toFixed(1)}GB) — KILLING Vivado tree (pid ${child.pid})`;
          console.error(ramKillMsg);
          clearInterval(poll);
          try { if (child.pid) execFileP("taskkill", ["/PID", String(child.pid), "/T", "/F"]).catch(() => {}); } catch { /* */ }
          try { execFileP("taskkill", ["/IM", "vivado.exe", "/T", "/F"]).catch(() => {}); } catch { /* */ }
        }
      }, 4000);
    });
    const elapsed = (Date.now() - t0) / 1000;
    console.log(`[pblock] vivado returned in ${elapsed.toFixed(1)}s (ok=${res.ok}${ramKillMsg ? ", RAM-KILLED" : ""})`);

    const utilText = existsSync(utilSink) ? await readFile(utilSink, "utf8") : "";
    const timingText = existsSync(timingSink) ? await readFile(timingSink, "utf8") : "";
    const combined = [res.stdout, res.stderr, utilText, timingText].filter(Boolean).join("\n");
    await writeFile(logPath, combined, "utf8");
    const parsed = parseVivadoReport(combined, clockNs, part);
    parsed.success = res.ok && parsed.success;
    return { ...parsed, elapsed_s: elapsed, ramKilled: Boolean(ramKillMsg) };
  });

  await writeFile(jsonReportPath, JSON.stringify(report, null, 2), "utf8");
  console.log(`[pblock] wrote ${path.relative(repoRoot, jsonReportPath)}; success=${report.success}`);
  if (probeOnly) console.log(`[pblock] probe report -> ${path.relative(repoRoot, probeRptSafe)}`);
}

main().catch((err: unknown) => {
  console.error(err instanceof Error ? err.stack ?? err.message : String(err));
  process.exit(1);
});
