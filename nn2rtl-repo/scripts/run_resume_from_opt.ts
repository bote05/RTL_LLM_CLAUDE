// Resume from opt checkpoint: place + route + reports + power. Saves dcps to
// the safe location. Companion to run_resume_from_synth.ts — covers the case
// where place/route died after opt completed.
//
// Usage:
//   set NN2RTL_VIVADO_BIN=D:/vivado/2025.2/Vivado/bin/vivado.bat
//   set NN2RTL_VIVADO_TIMEOUT_MS=43200000
//   npx tsx scripts/run_resume_from_opt.ts \
//     [--input=output/reports_integrated/checkpoints/first_light_opt_URAM.dcp] \
//     [--clock-ns=20] [--threads=8] [--part=xcu250-figd2104-2L-e]

import { readFile, writeFile, mkdir } from "node:fs/promises";
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
const inputRaw = flag("input") ?? path.join(safeCheckpointDir, "first_light_opt_URAM.dcp");
const inputDcp = path.isAbsolute(inputRaw) ? inputRaw : path.resolve(repoRoot, inputRaw);
const reportsDir = path.join(repoRoot, "output", "reports_integrated");
const jsonReportPath = path.join(reportsDir, "resume_from_opt.json");
const logPath = path.join(reportsDir, "resume_from_opt.log");

function tclQuote(value: string): string {
  return `"${toVivadoPath(value).replace(/(["$[\]])/g, "\\$1")}"`;
}

function buildTcl(input: {
  optDcp: string; placedDcp: string; routedDcp: string;
  postRouteUtil: string; postRouteTiming: string; postRoutePower: string;
}): string {
  return [
    `set_param general.maxThreads ${threads}`,
    `puts "NN2RTL_INFO: opening opt checkpoint"`,
    `open_checkpoint ${tclQuote(input.optDcp)}`,
    `puts "NN2RTL_INFO: starting place_design"`,
    `place_design`,
    `write_checkpoint -force ${tclQuote(input.placedDcp)}`,
    `puts "NN2RTL_INFO: starting route_design"`,
    `route_design`,
    `write_checkpoint -force ${tclQuote(input.routedDcp)}`,
    `report_utilization -file ${tclQuote(input.postRouteUtil)}`,
    `report_timing_summary -check_timing_verbose -max_paths 20 -file ${tclQuote(input.postRouteTiming)}`,
    `report_power -file ${tclQuote(input.postRoutePower)}`,
    `puts "NN2RTL_INFO: resume_from_opt complete"`,
  ].join("\n") + "\n";
}

const execFileP = promisify(execFile);

async function main(): Promise<void> {
  await mkdir(reportsDir, { recursive: true });
  await mkdir(safeCheckpointDir, { recursive: true });
  if (!existsSync(inputDcp)) throw new Error(`opt checkpoint not found: ${inputDcp}`);
  console.log(`[resume-opt] input: ${inputDcp}`);
  const tag = path.basename(inputDcp).includes("_URAM") ? "_URAM" : "";
  const placedDcpSafe = path.join(safeCheckpointDir, `first_light_placed${tag}.dcp`);
  const routedDcpSafe = path.join(safeCheckpointDir, `first_light_routed${tag}.dcp`);
  const postRouteUtilSafe = path.join(safeCheckpointDir, `first_light_postroute_util${tag}.rpt`);
  const postRouteTimingSafe = path.join(safeCheckpointDir, `first_light_postroute_timing${tag}.rpt`);
  const postRoutePowerSafe = path.join(safeCheckpointDir, `first_light_postroute_power${tag}.rpt`);

  const report = await withTempDir("nn2rtl-resumeopt-", async (tempDir) => {
    const tclPath = path.join(tempDir, "resume.tcl");
    await writeFile(tclPath, buildTcl({
      optDcp: inputDcp, placedDcp: placedDcpSafe, routedDcp: routedDcpSafe,
      postRouteUtil: postRouteUtilSafe, postRouteTiming: postRouteTimingSafe, postRoutePower: postRoutePowerSafe,
    }), "utf8");

    const vivadoBin = resolveVivadoCommand(process.env);
    const vivadoArgs = ["-mode", "batch", "-source", toVivadoPath(tclPath), "-notrace"];
    const isWindowsBatch = process.platform === "win32" && /\.(bat|cmd)$/i.test(vivadoBin);
    const spawnFile = isWindowsBatch ? "cmd.exe" : vivadoBin;
    const spawnArgs = isWindowsBatch ? ["/c", vivadoBin, ...vivadoArgs] : vivadoArgs;

    const t0 = Date.now();
    let stdout = "", stderr = "", exitOk = true;
    try {
      const timeoutMs = (() => {
        const e = process.env.NN2RTL_VIVADO_TIMEOUT_MS;
        return e && Number.isFinite(Number(e)) && Number(e) > 0 ? Number(e) : VIVADO_TIMEOUT_MS;
      })();
      const res = await execFileP(spawnFile, spawnArgs, { cwd: tempDir, env: process.env, timeout: timeoutMs, maxBuffer: VIVADO_MAX_BUFFER_BYTES });
      stdout = res.stdout; stderr = res.stderr;
    } catch (err: unknown) {
      exitOk = false;
      const e = err as { stdout?: string | Buffer; stderr?: string | Buffer; message?: string };
      stdout = typeof e.stdout === "string" ? e.stdout : (e.stdout?.toString() ?? "");
      stderr = typeof e.stderr === "string" ? e.stderr : (e.stderr?.toString() ?? e.message ?? "");
    }
    const elapsed = (Date.now() - t0) / 1000;
    const utilText = existsSync(postRouteUtilSafe) ? await readFile(postRouteUtilSafe, "utf8") : "";
    const timingText = existsSync(postRouteTimingSafe) ? await readFile(postRouteTimingSafe, "utf8") : "";
    const combined = [stdout, stderr, utilText, timingText].filter(Boolean).join("\n");
    await writeFile(logPath, combined, "utf8");
    const parsed = parseVivadoReport(combined, clockNs, part);
    parsed.success = exitOk && parsed.success;
    return { ...parsed, elapsed_s: elapsed };
  });

  await writeFile(jsonReportPath, JSON.stringify(report, null, 2), "utf8");
  console.log(`[resume-opt] success=${report.success}`);
}

main().catch((err: unknown) => { console.error(err instanceof Error ? err.stack ?? err.message : String(err)); process.exit(1); });
