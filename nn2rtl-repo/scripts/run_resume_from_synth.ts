// Resume from a synth checkpoint: opt_design + place_design + route_design +
// post-route reports + power. Saves all intermediate dcps to the safe location.
//
// Companion to run_first_light_synth.ts --synth-only: that script writes a
// first_light_synth*.dcp into output/reports_integrated/checkpoints/, and this
// script picks it up and finishes the implementation flow without redoing the
// 60-90 min synth pass.
//
// Usage:
//   set NN2RTL_VIVADO_BIN=D:/vivado/2025.2/Vivado/bin/vivado.bat
//   set NN2RTL_VIVADO_TIMEOUT_MS=43200000   # 12 hours
//   npx tsx scripts/run_resume_from_synth.ts \
//     [--input=output/reports_integrated/checkpoints/first_light_synth_URAM.dcp] \
//     [--clock-ns=20] [--threads=8] [--part=xcu250-figd2104-2L-e]
//
// Output (in output/reports_integrated/checkpoints/):
//   first_light_opt{_URAM}.dcp
//   first_light_placed{_URAM}.dcp
//   first_light_routed{_URAM}.dcp
//   first_light_postroute_util.rpt / _timing.rpt / _power.rpt

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
const safeCheckpointDir = path.join(repoRoot, "output", "reports_integrated", "checkpoints");
const inputRaw = flag("input") ?? path.join(safeCheckpointDir, "first_light_synth_URAM.dcp");
const inputDcp = path.isAbsolute(inputRaw) ? inputRaw : path.resolve(repoRoot, inputRaw);

const reportsDir = path.join(repoRoot, "output", "reports_integrated");
const jsonReportPath = path.join(reportsDir, "resume_from_synth.json");
const logPath = path.join(reportsDir, "resume_from_synth.log");

function tclQuote(value: string): string {
  return `"${toVivadoPath(value).replace(/(["$[\]])/g, "\\$1")}"`;
}

function buildTcl(input: {
  synthDcp: string;
  optDcp: string;
  placedDcp: string;
  routedDcp: string;
  postRouteUtil: string;
  postRouteTiming: string;
  postRoutePower: string;
  utilSink: string;
  timingSink: string;
}): string {
  return [
    `set_param general.maxThreads ${threads}`,
    `puts "NN2RTL_INFO: requested general.maxThreads=${threads}, effective=[get_param general.maxThreads]"`,
    `puts "NN2RTL_INFO: opening synth checkpoint"`,
    `open_checkpoint ${tclQuote(input.synthDcp)}`,
    `puts "NN2RTL_INFO: starting opt_design"`,
    `opt_design`,
    `write_checkpoint -force ${tclQuote(input.optDcp)}`,
    `puts "NN2RTL_INFO: starting place_design"`,
    `place_design`,
    `write_checkpoint -force ${tclQuote(input.placedDcp)}`,
    `puts "NN2RTL_INFO: starting route_design"`,
    `route_design`,
    `write_checkpoint -force ${tclQuote(input.routedDcp)}`,
    `puts "NN2RTL_INFO: post-route utilization"`,
    `report_utilization -file ${tclQuote(input.postRouteUtil)}`,
    `puts "NN2RTL_INFO: post-route timing"`,
    `report_timing_summary -check_timing_verbose -max_paths 20 -file ${tclQuote(input.postRouteTiming)}`,
    `puts "NN2RTL_INFO: post-route power"`,
    `report_power -file ${tclQuote(input.postRoutePower)}`,
    `file copy -force ${tclQuote(input.postRouteUtil)} ${tclQuote(input.utilSink)}`,
    `file copy -force ${tclQuote(input.postRouteTiming)} ${tclQuote(input.timingSink)}`,
    `puts "NN2RTL_INFO: resume_from_synth complete"`,
  ].join("\n") + "\n";
}

const execFileP = promisify(execFile);

async function main(): Promise<void> {
  await mkdir(reportsDir, { recursive: true });
  await mkdir(safeCheckpointDir, { recursive: true });
  if (!existsSync(inputDcp)) {
    throw new Error(`synth checkpoint not found: ${inputDcp}`);
  }
  console.log(`[resume] input synth dcp: ${inputDcp}`);
  console.log(`[resume] part=${part} clock_ns=${clockNs} threads=${threads}`);

  // Stable destination names — append _URAM if the input was the URAM build.
  const tag = path.basename(inputDcp).includes("_URAM") ? "_URAM" : "";
  const optDcpSafe = path.join(safeCheckpointDir, `first_light_opt${tag}.dcp`);
  const placedDcpSafe = path.join(safeCheckpointDir, `first_light_placed${tag}.dcp`);
  const routedDcpSafe = path.join(safeCheckpointDir, `first_light_routed${tag}.dcp`);
  const postRouteUtilSafe = path.join(safeCheckpointDir, `first_light_postroute_util${tag}.rpt`);
  const postRouteTimingSafe = path.join(safeCheckpointDir, `first_light_postroute_timing${tag}.rpt`);
  const postRoutePowerSafe = path.join(safeCheckpointDir, `first_light_postroute_power${tag}.rpt`);

  const report = await withTempDir("nn2rtl-resume-", async (tempDir) => {
    const utilSink = path.join(tempDir, "resume_util.rpt");
    const timingSink = path.join(tempDir, "resume_timing.rpt");
    const tclPath = path.join(tempDir, "resume.tcl");

    await writeFile(
      tclPath,
      buildTcl({
        synthDcp: inputDcp,
        optDcp: optDcpSafe,
        placedDcp: placedDcpSafe,
        routedDcp: routedDcpSafe,
        postRouteUtil: postRouteUtilSafe,
        postRouteTiming: postRouteTimingSafe,
        postRoutePower: postRoutePowerSafe,
        utilSink,
        timingSink,
      }),
      "utf8",
    );

    const vivadoBin = resolveVivadoCommand(process.env);
    const vivadoArgs = ["-mode", "batch", "-source", toVivadoPath(tclPath), "-notrace"];
    const isWindowsBatch = process.platform === "win32" && /\.(bat|cmd)$/i.test(vivadoBin);
    const spawnFile = isWindowsBatch ? "cmd.exe" : vivadoBin;
    const spawnArgs = isWindowsBatch ? ["/c", vivadoBin, ...vivadoArgs] : vivadoArgs;

    console.log(`[resume] launching vivado in ${tempDir}`);
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
      const res = await execFileP(spawnFile, spawnArgs, {
        cwd: tempDir,
        env: process.env,
        timeout: timeoutMs,
        maxBuffer: VIVADO_MAX_BUFFER_BYTES,
      });
      stdout = res.stdout;
      stderr = res.stderr;
    } catch (err: unknown) {
      exitOk = false;
      const e = err as { stdout?: string | Buffer; stderr?: string | Buffer; message?: string };
      stdout = typeof e.stdout === "string" ? e.stdout : (e.stdout?.toString() ?? "");
      stderr = typeof e.stderr === "string" ? e.stderr : (e.stderr?.toString() ?? e.message ?? "");
    }
    const elapsed = (Date.now() - t0) / 1000;
    console.log(`[resume] vivado returned in ${elapsed.toFixed(1)}s (ok=${exitOk})`);

    const utilText = existsSync(utilSink) ? await readFile(utilSink, "utf8") : "";
    const timingText = existsSync(timingSink) ? await readFile(timingSink, "utf8") : "";
    const combined = [stdout, stderr, utilText, timingText].filter(Boolean).join("\n");
    await writeFile(logPath, combined, "utf8");
    const parsed = parseVivadoReport(combined, clockNs, part);
    parsed.success = exitOk && parsed.success;
    return { ...parsed, elapsed_s: elapsed };
  });

  await writeFile(jsonReportPath, JSON.stringify(report, null, 2), "utf8");
  console.log(`[resume] wrote ${path.relative(repoRoot, jsonReportPath)}`);
  console.log(`[resume] success=${report.success}`);
}

main().catch((err: unknown) => {
  console.error(err instanceof Error ? err.stack ?? err.message : String(err));
  process.exit(1);
});
