// Task 13 first-light Vivado synth driver for the integrated top.
//
// Unlike scripts/vivado_baseline.ts (per-module synth, no inter-module
// elaboration), this script synthesises the full integrated design:
//   - top wrapper (output/rtl/nn2rtl_top.v)
//   - scheduler (output/rtl/nn2rtl_scheduler.v)
//   - shared engine skeleton (output/rtl/shared_engine_skeleton.v)
//   - 5 engine sub-blocks (output/rtl/engine/*.v)
//   - every per-layer module (output/rtl/node_*.v)
//   - the 3 conv-library helpers (rtl_library/*.v)
//
// All $readmemh paths inside the source files are rewritten to absolute paths
// so Vivado finds bias.mem and the 8 uram_weights_bank*.mem files from its
// tmp working dir.
//
// Usage:
//   set NN2RTL_VIVADO_BIN=D:/vivado/2025.2/Vivado/bin/vivado.bat
//   npx tsx scripts/run_first_light_synth.ts [--part=xcu250-figd2104-2L-e]
//                                            [--clock-ns=20] [--threads=8]
//
// Output:
//   output/reports_integrated/first_light_synth.json — parsed VivadoSynthesisReport
//   output/reports_integrated/first_light_synth.log  — full stdout+stderr
//   docs/agent_tasks/13_integration_first_light_REPORT.md — human-readable summary

import { readFile, writeFile, mkdir, copyFile, readdir } from "node:fs/promises";
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
const topModule = "nn2rtl_top";

const reportsDir = path.join(repoRoot, "output", "reports_integrated");
const jsonReportPath = path.join(reportsDir, "first_light_synth.json");
const logPath = path.join(reportsDir, "first_light_synth.log");
const mdReportPath = path.join(repoRoot, "docs", "agent_tasks", "13_integration_first_light_REPORT.md");

function tclQuote(value: string): string {
  return `"${toVivadoPath(value).replace(/(["$[\]])/g, "\\$1")}"`;
}

function convertReadmemhAbs(source: string, repoRootAbs: string): string {
  const fix = (p: string): string => {
    const absolute = path.isAbsolute(p) ? p : path.resolve(repoRootAbs, p);
    return toVivadoPath(absolute);
  };
  return source
    // Direct $readmemh("...") literals in initial blocks
    .replace(
      /(\$readmemh\s*\(\s*)"([^"]+)"/g,
      (_match, prefix: string, p: string) => `${prefix}"${fix(p)}"`,
    )
    // MEM_INIT_FILE parameter values — both in module parameter
    // declarations (`parameter MEM_INIT_FILE = "..."` and similar) and in
    // module instance overrides (`.MEM_INIT_FILE("...")`). Skip empty
    // strings (the parameter's default before instantiation).
    .replace(
      /(MEM_INIT_FILE\s*[=(]\s*)"([^"]+)"/g,
      (_match, prefix: string, p: string) =>
        p.length === 0 ? `${prefix}""` : `${prefix}"${fix(p)}"`,
    );
}

async function collectSources(): Promise<string[]> {
  const out: string[] = [
    path.join(repoRoot, "output", "rtl", "nn2rtl_top.v"),
    path.join(repoRoot, "output", "rtl", "nn2rtl_scheduler.v"),
    path.join(repoRoot, "output", "rtl", "shared_engine_skeleton.v"),
    path.join(repoRoot, "output", "rtl", "engine", "address_generator.v"),
    path.join(repoRoot, "output", "rtl", "engine", "config_register_block.v"),
    path.join(repoRoot, "output", "rtl", "engine", "mac_array.v"),
    path.join(repoRoot, "output", "rtl", "engine", "requant_pipeline.v"),
    path.join(repoRoot, "output", "rtl", "engine", "bram_to_stream_bridge.v"),
    path.join(repoRoot, "rtl_library", "conv_datapath.v"),
    path.join(repoRoot, "rtl_library", "coord_scheduler.v"),
    path.join(repoRoot, "rtl_library", "line_buf_window.v"),
  ];

  const rtlDir = path.join(repoRoot, "output", "rtl");
  const entries = await readdir(rtlDir);
  for (const entry of entries) {
    if (entry.startsWith("node_") && entry.endsWith(".v")) {
      out.push(path.join(rtlDir, entry));
    }
  }

  for (const p of out) {
    if (!existsSync(p)) {
      throw new Error(`source missing: ${p}`);
    }
  }
  return out;
}

function buildIntegratedTcl(input: {
  verilogPaths: string[];
  utilReportPath: string;
  timingReportPath: string;
  checkpointPath: string;
}): string {
  return [
    `set_param general.maxThreads ${threads}`,
    `puts "NN2RTL_INFO: requested general.maxThreads=${threads}, effective=[get_param general.maxThreads]"`,
    `read_verilog -sv ${input.verilogPaths.map(tclQuote).join(" \\\n                 ")}`,
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

  const sources = await collectSources();
  console.log(`[first-light] collected ${sources.length} source files`);
  console.log(`[first-light] top=${topModule} part=${part} clock_ns=${clockNs} threads=${threads}`);

  const report = await withTempDir("nn2rtl-firstlight-", async (tempDir) => {
    const copiedPaths: string[] = [];
    for (const src of sources) {
      const dest = path.join(tempDir, path.basename(src));
      const text = await readFile(src, "utf8");
      await writeFile(dest, convertReadmemhAbs(text, repoRoot), "utf8");
      copiedPaths.push(dest);
    }

    const utilReportPath = path.join(tempDir, "first_light_util.rpt");
    const timingReportPath = path.join(tempDir, "first_light_timing.rpt");
    const checkpointPath = path.join(tempDir, "first_light.dcp");
    const tclPath = path.join(tempDir, "first_light.tcl");

    await writeFile(
      tclPath,
      buildIntegratedTcl({
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

    console.log(`[first-light] launching vivado: ${spawnFile} ${spawnArgs.join(" ")}`);
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
    console.log(`[first-light] vivado returned in ${elapsed.toFixed(1)}s (ok=${exitOk})`);

    const utilReport = existsSync(utilReportPath) ? await readFile(utilReportPath, "utf8") : "";
    const timingReport = existsSync(timingReportPath) ? await readFile(timingReportPath, "utf8") : "";
    const combinedReport = [
      stdout,
      stderr,
      "--- first_light_util.rpt ---",
      utilReport,
      "--- first_light_timing.rpt ---",
      timingReport,
    ].filter(Boolean).join("\n");

    await writeFile(logPath, combinedReport, "utf8");
    const parsed = parseVivadoReport(combinedReport, clockNs, part);
    parsed.success = exitOk && parsed.success;
    return { ...parsed, elapsed_s: elapsed };
  });

  await writeFile(jsonReportPath, JSON.stringify(report, null, 2), "utf8");
  console.log(`[first-light] wrote ${path.relative(repoRoot, jsonReportPath)}`);

  const lines: string[] = [
    "# Task 13 — Integration first-light synth report",
    "",
    `Generated by \`scripts/run_first_light_synth.ts\`.`,
    "",
    `- part: \`${part}\``,
    `- top: \`${topModule}\``,
    `- clock period: ${clockNs} ns`,
    `- elapsed: ${(report as VivadoSynthesisReport & { elapsed_s: number }).elapsed_s.toFixed(1)} s`,
    `- success: **${report.success}**`,
    "",
    "## Resource utilisation",
    "",
    `- LUT: ${report.lut_count}`,
    `- FF : ${report.ff_count}`,
    `- DSP: ${report.dsp_count}`,
    `- BRAM18: ${report.bram18_count}`,
    `- BRAM36: ${report.bram36_count}`,
    `- BRAM18-eq: ${report.bram18_equiv}`,
    "",
    "## Timing",
    "",
    `- WNS (setup): ${report.setup_wns_ns ?? "n/a"} ns`,
    `- WNS (hold) : ${report.hold_wns_ns ?? "n/a"} ns`,
    `- timing_met: ${report.timing_met}`,
    `- Fmax (estimate): ${report.fmax_mhz.toFixed(2)} MHz`,
    "",
    "## Notes",
    "",
    "- Phase 3 (bit-exact correctness) and Phase 4a (timing closure) are out of",
    "  scope for first-light. Non-zero WNS / sub-clock-target Fmax is expected.",
    `- Full log: \`${path.relative(repoRoot, logPath)}\``,
    `- Parsed JSON: \`${path.relative(repoRoot, jsonReportPath)}\``,
    "",
  ];

  await writeFile(mdReportPath, lines.join("\n"), "utf8");
  console.log(`[first-light] wrote ${path.relative(repoRoot, mdReportPath)}`);

  if (!report.success) {
    console.error("[first-light] FAILED — see log for details");
    process.exit(2);
  }
}

main().catch((err) => {
  console.error("[first-light] FATAL:", err instanceof Error ? (err.stack ?? err.message) : String(err));
  process.exit(1);
});
