// Synthesize ONLY the shared engine + its 5 sub-blocks (no top wrapper,
// no scheduler, no per-layer modules). Used to answer the open question
// from the U250 deployment plan §6.1: what is the engine's standalone
// LUT/FF/DSP/BRAM cost on U250?
//
// The shared_engine_skeleton.v file contains the top module + empty stubs
// for the 5 sub-blocks (gated by `ifndef NN2RTL_ENGINE_SUBBLOCKS_PROVIDED`).
// Defining that macro before reading the skeleton suppresses the stubs;
// then reading the real implementations from output/rtl/engine/ gives
// Vivado the complete design.
//
// Usage:
//   set NN2RTL_VIVADO_BIN=D:/vivado/2025.2/Vivado/bin/vivado.bat
//   npx tsx scripts/run_engine_only_synth.ts [--part=xcu250-figd2104-2L-e]
//                                             [--clock-ns=20] [--threads=8]
//
// Output:
//   output/reports_integrated/engine_only_synth.{json,log}
//   docs/agent_tasks/00_engine_only_synth_REPORT.md

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
const topModule = "shared_engine";

const reportsDir = path.join(repoRoot, "output", "reports_integrated");
const jsonReportPath = path.join(reportsDir, "engine_only_synth.json");
const logPath = path.join(reportsDir, "engine_only_synth.log");
const mdReportPath = path.join(repoRoot, "docs", "agent_tasks", "00_engine_only_synth_REPORT.md");

function tclQuote(value: string): string {
  return `"${toVivadoPath(value).replace(/(["$[\]])/g, "\\$1")}"`;
}

const ENGINE_SOURCES = [
  path.join(repoRoot, "output", "rtl", "shared_engine_skeleton.v"),
  path.join(repoRoot, "output", "rtl", "engine", "address_generator.v"),
  path.join(repoRoot, "output", "rtl", "engine", "config_register_block.v"),
  path.join(repoRoot, "output", "rtl", "engine", "mac_array.v"),
  path.join(repoRoot, "output", "rtl", "engine", "requant_pipeline.v"),
  path.join(repoRoot, "output", "rtl", "engine", "bram_to_stream_bridge.v"),
];

function buildEngineTcl(input: {
  verilogPaths: string[];
  utilReportPath: string;
  timingReportPath: string;
  checkpointPath: string;
}): string {
  // The skeleton file is copied with `define NN2RTL_ENGINE_SUBBLOCKS_PROVIDED 1`
  // prepended (see main()), so its empty stubs are excluded and the real
  // implementations in engine/*.v are picked up.
  const sources = input.verilogPaths.map(tclQuote).join(" \\\n                 ");
  return [
    `set_param general.maxThreads ${threads}`,
    `puts "NN2RTL_INFO: requested general.maxThreads=${threads}, effective=[get_param general.maxThreads]"`,
    `read_verilog -sv ${sources}`,
    `synth_design -top ${topModule} -part ${part} -flatten_hierarchy rebuilt`,
    `create_clock -name clk -period ${clockNs} [get_ports clk]`,
    `report_utilization -file ${tclQuote(input.utilReportPath)}`,
    `report_timing_summary -check_timing_verbose -max_paths 20 -file ${tclQuote(input.timingReportPath)}`,
    `write_checkpoint -force ${tclQuote(input.checkpointPath)}`,
  ].join("\n") + "\n";
}

const execFileP = promisify(execFile);

async function main(): Promise<void> {
  await mkdir(reportsDir, { recursive: true });
  await mkdir(path.dirname(mdReportPath), { recursive: true });

  for (const src of ENGINE_SOURCES) {
    if (!existsSync(src)) {
      throw new Error(`engine source missing: ${src}`);
    }
  }
  console.log(`[engine-only] collected ${ENGINE_SOURCES.length} source files`);
  console.log(`[engine-only] top=${topModule} part=${part} clock_ns=${clockNs} threads=${threads}`);

  const report = await withTempDir("nn2rtl-engine-only-", async (tempDir) => {
    const copiedPaths: string[] = [];
    for (const src of ENGINE_SOURCES) {
      const dest = path.join(tempDir, path.basename(src));
      let text = await readFile(src, "utf8");
      // Skeleton has `ifndef NN2RTL_ENGINE_SUBBLOCKS_PROVIDED` guards around
      // its empty sub-block stubs. Prepend the define so synth picks the
      // real engine/*.v implementations instead.
      if (path.basename(src) === "shared_engine_skeleton.v") {
        text = `\`define NN2RTL_ENGINE_SUBBLOCKS_PROVIDED 1\n${text}`;
      }
      await writeFile(dest, text, "utf8");
      copiedPaths.push(dest);
    }

    const utilReportPath = path.join(tempDir, "engine_only_util.rpt");
    const timingReportPath = path.join(tempDir, "engine_only_timing.rpt");
    const checkpointPath = path.join(tempDir, "engine_only.dcp");
    const tclPath = path.join(tempDir, "engine_only.tcl");

    await writeFile(
      tclPath,
      buildEngineTcl({
        verilogPaths: copiedPaths,
        utilReportPath,
        timingReportPath,
        checkpointPath,
      }),
      "utf8",
    );

    const vivadoBin = resolveVivadoCommand(process.env);
    const vivadoArgs = ["-mode", "batch", "-source", toVivadoPath(tclPath), "-notrace"];
    const isWindowsBatch = process.platform === "win32" && /\.(bat|cmd)$/i.test(vivadoBin);
    const spawnFile = isWindowsBatch ? "cmd.exe" : vivadoBin;
    const spawnArgs = isWindowsBatch ? ["/c", vivadoBin, ...vivadoArgs] : vivadoArgs;

    console.log(`[engine-only] launching vivado: ${spawnFile} ${spawnArgs.join(" ")}`);
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
    console.log(`[engine-only] vivado returned in ${elapsed.toFixed(1)}s (ok=${exitOk})`);

    const utilReport = existsSync(utilReportPath) ? await readFile(utilReportPath, "utf8") : "";
    const timingReport = existsSync(timingReportPath) ? await readFile(timingReportPath, "utf8") : "";
    const combinedReport = [
      "--- vivado stdout ---",
      stdout,
      "--- vivado stderr ---",
      stderr,
      "",
      "--- engine_only_util.rpt ---",
      utilReport,
      "--- engine_only_timing.rpt ---",
      timingReport,
    ].join("\n");
    await writeFile(logPath, combinedReport, "utf8");

    const parsed = parseVivadoReport(combinedReport, clockNs);
    return { parsed, elapsed, exitOk };
  });

  await writeFile(jsonReportPath, JSON.stringify(report.parsed, null, 2) + "\n", "utf8");
  console.log(`[engine-only] wrote ${path.relative(repoRoot, jsonReportPath)}`);

  const lines = [
    "# Engine-only Vivado synth report",
    "",
    "Generated by `scripts/run_engine_only_synth.ts`.",
    "",
    `- part: \`${part}\``,
    `- top: \`${topModule}\``,
    `- clock period: ${clockNs} ns`,
    `- elapsed: ${report.elapsed.toFixed(1)} s`,
    `- success: **${report.parsed.success}**`,
    "",
    "## Resource utilisation (synth_design)",
    "",
    `- LUT: ${report.parsed.lut_count.toLocaleString()}`,
    `- FF : ${report.parsed.ff_count.toLocaleString()}`,
    `- DSP: ${report.parsed.dsp_count.toLocaleString()}`,
    `- BRAM18: ${report.parsed.bram18_count}`,
    `- BRAM36: ${report.parsed.bram36_count}`,
    `- BRAM18-eq: ${report.parsed.bram18_equiv}`,
    "",
    "## Timing",
    "",
    `- WNS (setup): ${report.parsed.setup_wns_ns ?? "n/a"} ns`,
    `- WNS (hold) : ${report.parsed.hold_wns_ns ?? "n/a"} ns`,
    `- timing_met: ${report.parsed.timing_met}`,
    `- Fmax (estimate): ${report.parsed.fmax_mhz.toFixed(2)} MHz`,
    "",
    "## Notes",
    "",
    "- This is post-synth (no place/route) — actual fit depends on routing congestion.",
    "- Full log: `output/reports_integrated/engine_only_synth.log`",
    "- Parsed JSON: `output/reports_integrated/engine_only_synth.json`",
    "",
  ].join("\n");
  await writeFile(mdReportPath, lines, "utf8");
  console.log(`[engine-only] wrote ${path.relative(repoRoot, mdReportPath)}`);

  if (!report.parsed.success) {
    console.log("[engine-only] FAILED — see log for details");
    process.exitCode = 1;
  } else {
    console.log("[engine-only] success");
  }
}

main().catch((err: unknown) => {
  console.error(err instanceof Error ? err.stack ?? err.message : String(err));
  process.exit(1);
});
