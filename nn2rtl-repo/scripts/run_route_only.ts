// Open a placed checkpoint and finish: route_design + post-route reports + power.
//
// Used after `run_first_light_synth.ts` ran synth+opt+place+route but route_design
// failed (OOM, congestion timeout, etc.) — recovers without redoing the 6-8h
// synth/opt/place sequence. Reads the safe-copy of the placed checkpoint at
// output/reports_integrated/checkpoints/first_light_placed.dcp.
//
// Usage:
//   set NN2RTL_VIVADO_BIN=D:/vivado/2025.2/Vivado/bin/vivado.bat
//   set NN2RTL_VIVADO_TIMEOUT_MS=21600000   # 6 hours
//   npx tsx scripts/run_route_only.ts [--part=xcu250-figd2104-2L-e]
//                                     [--clock-ns=20] [--threads=8]
//                                     [--checkpoint=path/to/first_light_placed.dcp]
//
// Output:
//   output/reports_integrated/route_only_synth.{json,log}
//   output/reports_integrated/checkpoints/first_light_routed.dcp
//   output/reports_integrated/checkpoints/first_light_postroute_util.rpt
//   output/reports_integrated/checkpoints/first_light_postroute_timing.rpt
//   output/reports_integrated/checkpoints/first_light_postroute_power.rpt
//   docs/agent_tasks/13_integration_first_light_REPORT.md

import { readFile, writeFile, mkdir, copyFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

import {
  parseVivadoReport,
  resolveVivadoCommand,
  toVivadoPath,
  withTempDir,
  VIVADO_TIMEOUT_MS,
  VIVADO_MAX_BUFFER_BYTES,
  type VivadoSynthesisReport,
} from "../mcp/tools.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const rawArgs = process.argv.slice(2);

function flag(name: string, fallback?: string): string | undefined {
  const eq = rawArgs.find((a) => a.startsWith(`--${name}=`));
  if (eq) return eq.slice(name.length + 3);
  const idx = rawArgs.indexOf(`--${name}`);
  if (idx >= 0 && rawArgs[idx + 1] && !rawArgs[idx + 1].startsWith("--")) {
    return rawArgs[idx + 1];
  }
  return fallback;
}

const part = flag("part") ?? "xcu250-figd2104-2L-e";
const clockNs = Number(flag("clock-ns") ?? "20");
const threads = Number(flag("threads") ?? "8");
const checkpointInput = flag("checkpoint") ?? path.join(repoRoot, "output", "reports_integrated", "checkpoints", "first_light_placed.dcp");

const reportsDir = path.join(repoRoot, "output", "reports_integrated");
const safeCheckpointDir = path.join(reportsDir, "checkpoints");
const jsonReportPath = path.join(reportsDir, "route_only_synth.json");
const logPath = path.join(reportsDir, "route_only_synth.log");
const mdReportPath = path.join(repoRoot, "docs", "agent_tasks", "13_integration_first_light_REPORT.md");

function tclQuote(value: string): string {
  return `"${toVivadoPath(value).replace(/(["$[\]])/g, "\\$1")}"`;
}

function buildRouteTcl(input: {
  placedCheckpointPath: string;
  routedCheckpointPath: string;
  postRouteUtilPath: string;
  postRouteTimingPath: string;
  postRoutePowerPath: string;
  utilReportPath: string;     // canonical sink for parseVivadoReport
  timingReportPath: string;   // canonical sink for parseVivadoReport
}): string {
  return [
    `set_param general.maxThreads ${threads}`,
    `puts "NN2RTL_INFO: requested general.maxThreads=${threads}, effective=[get_param general.maxThreads]"`,
    `puts "NN2RTL_INFO: opening placed checkpoint"`,
    `open_checkpoint ${tclQuote(input.placedCheckpointPath)}`,
    `puts "NN2RTL_INFO: starting route_design"`,
    `route_design`,
    `puts "NN2RTL_INFO: write routed checkpoint"`,
    `write_checkpoint -force ${tclQuote(input.routedCheckpointPath)}`,
    `puts "NN2RTL_INFO: post-route utilization"`,
    `report_utilization -file ${tclQuote(input.postRouteUtilPath)}`,
    `puts "NN2RTL_INFO: post-route timing summary"`,
    `report_timing_summary -check_timing_verbose -max_paths 20 -file ${tclQuote(input.postRouteTimingPath)}`,
    `puts "NN2RTL_INFO: post-route power (vectorless)"`,
    `report_power -file ${tclQuote(input.postRoutePowerPath)}`,
    // Mirror the post-route reports to the conventional sink filenames so
    // parseVivadoReport picks up the post-route numbers.
    `file copy -force ${tclQuote(input.postRouteUtilPath)} ${tclQuote(input.utilReportPath)}`,
    `file copy -force ${tclQuote(input.postRouteTimingPath)} ${tclQuote(input.timingReportPath)}`,
    `puts "NN2RTL_INFO: route_only flow complete"`,
  ].join("\n") + "\n";
}

const execFileP = promisify(execFile);

async function main(): Promise<void> {
  await mkdir(reportsDir, { recursive: true });
  await mkdir(safeCheckpointDir, { recursive: true });
  await mkdir(path.dirname(mdReportPath), { recursive: true });

  if (!existsSync(checkpointInput)) {
    throw new Error(`placed checkpoint not found: ${checkpointInput}`);
  }
  console.log(`[route-only] using placed checkpoint: ${checkpointInput}`);
  console.log(`[route-only] part=${part} clock_ns=${clockNs} threads=${threads}`);

  const report = await withTempDir("nn2rtl-routeonly-", async (tempDir) => {
    // Persist the routed checkpoint AND reports to a SAFE location (output/reports_integrated/checkpoints/)
    // so Windows Temp cleanup doesn't wipe them.
    const routedDcpSafe = path.join(safeCheckpointDir, "first_light_routed.dcp");
    const postRouteUtilSafe = path.join(safeCheckpointDir, "first_light_postroute_util.rpt");
    const postRouteTimingSafe = path.join(safeCheckpointDir, "first_light_postroute_timing.rpt");
    const postRoutePowerSafe = path.join(safeCheckpointDir, "first_light_postroute_power.rpt");
    const utilReportPath = path.join(tempDir, "route_only_util.rpt");
    const timingReportPath = path.join(tempDir, "route_only_timing.rpt");
    const tclPath = path.join(tempDir, "route_only.tcl");

    await writeFile(
      tclPath,
      buildRouteTcl({
        placedCheckpointPath: checkpointInput,
        routedCheckpointPath: routedDcpSafe,
        postRouteUtilPath: postRouteUtilSafe,
        postRouteTimingPath: postRouteTimingSafe,
        postRoutePowerPath: postRoutePowerSafe,
        utilReportPath,
        timingReportPath,
      }),
      "utf8",
    );

    const vivadoBin = resolveVivadoCommand(process.env);
    const vivadoArgs = ["-mode", "batch", "-source", toVivadoPath(tclPath), "-notrace"];
    const isWindowsBatch = process.platform === "win32" && /\.(bat|cmd)$/i.test(vivadoBin);
    const spawnFile = isWindowsBatch ? "cmd.exe" : vivadoBin;
    const spawnArgs = isWindowsBatch ? ["/c", vivadoBin, ...vivadoArgs] : vivadoArgs;

    console.log(`[route-only] launching vivado: ${spawnFile} ${spawnArgs.join(" ")}`);
    const t0 = Date.now();
    let stdout = "";
    let stderr = "";
    let exitOk = true;
    try {
      const timeoutMs = (() => {
        const envVal = process.env.NN2RTL_VIVADO_TIMEOUT_MS;
        if (envVal && Number.isFinite(Number(envVal)) && Number(envVal) > 0) {
          return Number(envVal);
        }
        return VIVADO_TIMEOUT_MS;
      })();
      const result = await execFileP(spawnFile, spawnArgs, {
        cwd: tempDir,
        env: process.env,
        timeout: timeoutMs,
        maxBuffer: VIVADO_MAX_BUFFER_BYTES,
      });
      stdout = result.stdout;
      stderr = result.stderr;
    } catch (err: unknown) {
      exitOk = false;
      const e = err as { stdout?: string | Buffer; stderr?: string | Buffer; message?: string };
      stdout = typeof e.stdout === "string" ? e.stdout : (e.stdout?.toString() ?? "");
      stderr = typeof e.stderr === "string" ? e.stderr : (e.stderr?.toString() ?? e.message ?? "");
    }
    const elapsed = (Date.now() - t0) / 1000;
    console.log(`[route-only] vivado returned in ${elapsed.toFixed(1)}s (ok=${exitOk})`);

    const utilReport = existsSync(utilReportPath) ? await readFile(utilReportPath, "utf8") : "";
    const timingReport = existsSync(timingReportPath) ? await readFile(timingReportPath, "utf8") : "";
    const combinedReport = [
      stdout,
      stderr,
      "--- route_only_util.rpt ---",
      utilReport,
      "--- route_only_timing.rpt ---",
      timingReport,
    ].filter(Boolean).join("\n");

    await writeFile(logPath, combinedReport, "utf8");
    const parsed = parseVivadoReport(combinedReport, clockNs, part);
    parsed.success = exitOk && parsed.success;
    return { ...parsed, elapsed_s: elapsed };
  });

  await writeFile(jsonReportPath, JSON.stringify(report, null, 2), "utf8");
  console.log(`[route-only] wrote ${path.relative(repoRoot, jsonReportPath)}`);

  const lines: string[] = [
    "# Task 13 — Integration post-route report (route-only resume)",
    "",
    `Generated by \`scripts/run_route_only.ts\` from \`first_light_placed.dcp\`.`,
    "",
    `- part: \`${part}\``,
    `- clock period: ${clockNs} ns`,
    `- elapsed (route_design + reports): ${(report as VivadoSynthesisReport & { elapsed_s: number }).elapsed_s.toFixed(1)} s`,
    `- success: **${report.success}**`,
    "",
    "## Post-route resource utilisation",
    "",
    `- LUT: ${report.lut_count.toLocaleString()}`,
    `- FF : ${report.ff_count.toLocaleString()}`,
    `- DSP: ${report.dsp_count.toLocaleString()}`,
    `- BRAM18: ${report.bram18_count}`,
    `- BRAM36: ${report.bram36_count}`,
    `- BRAM18-eq: ${report.bram18_equiv}`,
    "",
    "## Post-route timing",
    "",
    `- WNS (setup): ${report.setup_wns_ns ?? "n/a"} ns`,
    `- WNS (hold) : ${report.hold_wns_ns ?? "n/a"} ns`,
    `- timing_met: ${report.timing_met}`,
    `- Fmax (estimate): ${report.fmax_mhz.toFixed(2)} MHz`,
    "",
    "## Artifacts on disk",
    "",
    "- `output/reports_integrated/checkpoints/first_light_routed.dcp` — routed checkpoint",
    "- `output/reports_integrated/checkpoints/first_light_postroute_util.rpt`",
    "- `output/reports_integrated/checkpoints/first_light_postroute_timing.rpt`",
    "- `output/reports_integrated/checkpoints/first_light_postroute_power.rpt`",
    "- `output/reports_integrated/route_only_synth.{json,log}`",
    "",
  ].join("\n");
  await writeFile(mdReportPath, lines, "utf8");
  console.log(`[route-only] wrote ${path.relative(repoRoot, mdReportPath)}`);

  if (!report.success) {
    console.log("[route-only] FAILED — see log for details");
    process.exitCode = 1;
  } else {
    console.log("[route-only] success");
  }
}

main().catch((err: unknown) => {
  console.error(err instanceof Error ? err.stack ?? err.message : String(err));
  process.exit(1);
});
