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
import os from "node:os";

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
const synthOnly = rawArgs.includes("--synth-only");
const tagFlag = flag("tag");

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
    path.join(repoRoot, "rtl_library", "conv_datapath_parallel.v"),
    path.join(repoRoot, "rtl_library", "conv_datapath_mp_k.v"),
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
  // Sister report paths for post-route artifacts. We sit them next to the
  // synth-level reports so both pre- and post-route data land in the tmpdir.
  const postRouteUtilPath = input.utilReportPath.replace(/_util\.rpt$/, "_postroute_util.rpt");
  const postRouteTimingPath = input.timingReportPath.replace(/_timing\.rpt$/, "_postroute_timing.rpt");
  const postRoutePowerPath = input.timingReportPath.replace(/_timing\.rpt$/, "_postroute_power.rpt");
  const synthDcpPath = input.checkpointPath.replace(/\.dcp$/, "_synth.dcp");
  const optDcpPath = input.checkpointPath.replace(/\.dcp$/, "_opt.dcp");
  const placedDcpPath = input.checkpointPath.replace(/\.dcp$/, "_placed.dcp");
  return [
    `set_param general.maxThreads ${threads}`,
    `puts "NN2RTL_INFO: requested general.maxThreads=${threads}, effective=[get_param general.maxThreads]"`,
    `read_verilog -sv ${input.verilogPaths.map(tclQuote).join(" \\\n                 ")}`,
    `puts "NN2RTL_INFO: auto_detect_xpm (load XPM library for URAM primitives)"`,
    `auto_detect_xpm`,
    `puts "NN2RTL_INFO: starting synth_design (with -verilog_define NN2RTL_SYNTHESIS for XPM-URAM path)"`,
    `synth_design -top ${topModule} -part ${part} -flatten_hierarchy rebuilt -verilog_define NN2RTL_SYNTHESIS=1`,
    `create_clock -name clk -period ${clockNs} [get_ports clk]`,
    `puts "NN2RTL_INFO: synth-level utilization (saved to ${path.basename(input.utilReportPath)}.synth)"`,
    `report_utilization -file ${tclQuote(input.utilReportPath + '.synth')}`,
    `puts "NN2RTL_INFO: write synth checkpoint"`,
    `write_checkpoint -force ${tclQuote(synthDcpPath)}`,
    ...(synthOnly
      ? [
          // Synth-only spike: stop here and surface the synth-level numbers via
          // the conventional sinks so parseVivadoReport sees them.
          `puts "NN2RTL_INFO: --synth-only: mirroring synth util to conventional sink"`,
          `file copy -force ${tclQuote(input.utilReportPath + '.synth')} ${tclQuote(input.utilReportPath)}`,
          `puts "NN2RTL_INFO: synth-only spike complete (no opt/place/route)"`,
        ]
      : [
          `puts "NN2RTL_INFO: starting opt_design"`,
          `opt_design`,
          `puts "NN2RTL_INFO: write opt checkpoint"`,
          `write_checkpoint -force ${tclQuote(optDcpPath)}`,
          `puts "NN2RTL_INFO: starting place_design"`,
          `place_design`,
          `puts "NN2RTL_INFO: write placed checkpoint"`,
          `write_checkpoint -force ${tclQuote(placedDcpPath)}`,
          // route_design -directive Explore: the 2026-05-25 default route at
          // 20 ns failed at Phase 5.1 with 1.05M overlaps; the 2026-05-26
          // Explore route at 40 ns succeeded with +10 ns slack. With the
          // requant_pipeline fanout fix (per-group g_ctrl replication), the
          // 20 ns target should now be reachable, but keep Explore as the
          // safer default. To use the stock router, set NN2RTL_VIVADO_ROUTE
          // _DIRECTIVE_NONE=1 in the env (not implemented; remove the flag
          // here to revert).
          `puts "NN2RTL_INFO: starting route_design (directive=Explore)"`,
          `route_design -directive Explore`,
          `puts "NN2RTL_INFO: write routed checkpoint"`,
          `write_checkpoint -force ${tclQuote(input.checkpointPath)}`,
          `puts "NN2RTL_INFO: post-route utilization (the meaningful number)"`,
          `report_utilization -file ${tclQuote(postRouteUtilPath)}`,
          `puts "NN2RTL_INFO: post-route timing summary"`,
          `report_timing_summary -check_timing_verbose -max_paths 20 -file ${tclQuote(postRouteTimingPath)}`,
          `puts "NN2RTL_INFO: post-route power (vectorless)"`,
          `report_power -file ${tclQuote(postRoutePowerPath)}`,
          // The TS parser reads utilReportPath + timingReportPath at the
          // conventional names. Make those the POST-ROUTE versions so
          // parseVivadoReport sees the meaningful numbers (the synth-level
          // numbers stay available next to them as `*.rpt.synth`).
          `file copy -force ${tclQuote(postRouteUtilPath)} ${tclQuote(input.utilReportPath)}`,
          `file copy -force ${tclQuote(postRouteTimingPath)} ${tclQuote(input.timingReportPath)}`,
          `puts "NN2RTL_INFO: full integration flow complete"`,
        ]),
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

    // Belt-and-suspenders for XPM MEMORY_INIT_FILE resolution: Vivado looks for
    // the .mem file as a Tcl source file path AND in the working directory. We
    // already rewrite paths to absolute via convertReadmemhAbs; ALSO copy the
    // .mem files into the Vivado tempdir so basename-lookup works (in case some
    // Vivado XPM internal path expects same-dir resolution).
    const weightsDir = path.join(repoRoot, "output", "weights");
    if (existsSync(weightsDir)) {
      const memFiles = (await readdir(weightsDir)).filter((f) => f.endsWith(".mem"));
      for (const mem of memFiles) {
        await copyFile(path.join(weightsDir, mem), path.join(tempDir, mem));
      }
      console.log(`[first-light] copied ${memFiles.length} .mem files into Vivado tempdir`);
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
    let ramKillMsg = "";
    {
      const timeoutMs = (() => {
        const envVal = process.env.NN2RTL_VIVADO_TIMEOUT_MS;
        if (envVal && Number.isFinite(Number(envVal)) && Number(envVal) > 0) {
          return Number(envVal);
        }
        return VIVADO_TIMEOUT_MS;
      })();
      // RAM watchdog: if physical RAM usage crosses the kill threshold (default
      // 90% used; override via NN2RTL_RAM_KILL_PCT), terminate the whole Vivado
      // process tree so the host never thrashes/crashes. The 2026-05 MobileNet
      // synth OOM'd this machine — this is the guard against a repeat.
      const ramKillPct = (() => {
        const v = Number(process.env.NN2RTL_RAM_KILL_PCT);
        return Number.isFinite(v) && v > 0 && v < 100 ? v : 90;
      })();
      const totalMem = os.totalmem();
      console.log(
        `[first-light] RAM watchdog armed: kill Vivado tree at >= ${ramKillPct}% used ` +
          `(total ${(totalMem / 1073741824).toFixed(1)}GB → ~${((totalMem * (100 - ramKillPct)) / 100 / 1073741824).toFixed(1)}GB free floor, poll 4s)`,
      );
      const res = await new Promise<{ stdout: string; stderr: string; ok: boolean }>((resolve) => {
        const child = execFile(
          spawnFile,
          spawnArgs,
          { cwd: tempDir, env: process.env, timeout: timeoutMs, maxBuffer: VIVADO_MAX_BUFFER_BYTES },
          (err, soCb, seCb) => {
            clearInterval(poll);
            const so = typeof soCb === "string" ? soCb : (soCb?.toString() ?? "");
            const se =
              typeof seCb === "string" ? seCb : (seCb?.toString() ?? (err as Error | null)?.message ?? "");
            resolve({ stdout: so, stderr: se, ok: !err && !ramKillMsg });
          },
        );
        const poll = setInterval(() => {
          const freeGB = os.freemem() / 1073741824;
          const usedPct = (1 - os.freemem() / totalMem) * 100;
          if (usedPct >= ramKillPct && !ramKillMsg) {
            ramKillMsg =
              `[first-light][WATCHDOG] RAM ${usedPct.toFixed(1)}% used >= ${ramKillPct}% ` +
              `(free ${freeGB.toFixed(1)}GB) — KILLING Vivado tree (pid ${child.pid}) to protect the host`;
            console.error(ramKillMsg);
            clearInterval(poll);
            // Kill the spawned tree (cmd.exe → vivado.bat → vivado.exe + loader),
            // then sweep any stray vivado.exe by name as belt-and-suspenders.
            try {
              if (child.pid) execFileP("taskkill", ["/PID", String(child.pid), "/T", "/F"]).catch(() => {});
            } catch {
              /* ignore */
            }
            try {
              execFileP("taskkill", ["/IM", "vivado.exe", "/T", "/F"]).catch(() => {});
            } catch {
              /* ignore */
            }
          }
        }, 4000);
      });
      stdout = res.stdout;
      stderr = res.stderr;
      exitOk = res.ok;
    }
    const elapsed = (Date.now() - t0) / 1000;
    console.log(
      `[first-light] vivado returned in ${elapsed.toFixed(1)}s (ok=${exitOk}${ramKillMsg ? ", RAM-KILLED" : ""})`,
    );

    // Persist any intermediate checkpoints from the tmpdir to the safe
    // location so we can resume opt/place/route without redoing synth.
    // The tmpdir is auto-deleted on exit; checkpoints there would be lost.
    const safeCheckpointDir = path.join(reportsDir, "checkpoints");
    await mkdir(safeCheckpointDir, { recursive: true });
    const tag = tagFlag !== undefined ? tagFlag : (synthOnly ? "_URAM" : "");
    const dcpsToPersist = [
      { src: checkpointPath.replace(/\.dcp$/, "_synth.dcp"),  dst: `first_light_synth${tag}.dcp` },
      { src: checkpointPath.replace(/\.dcp$/, "_opt.dcp"),    dst: `first_light_opt${tag}.dcp` },
      { src: checkpointPath.replace(/\.dcp$/, "_placed.dcp"), dst: `first_light_placed${tag}.dcp` },
      { src: checkpointPath,                                   dst: `first_light_routed${tag}.dcp` },
    ];
    for (const { src, dst } of dcpsToPersist) {
      if (existsSync(src)) {
        await copyFile(src, path.join(safeCheckpointDir, dst));
        console.log(`[first-light] persisted ${path.basename(src)} -> ${dst}`);
      }
    }

    const utilReport = existsSync(utilReportPath) ? await readFile(utilReportPath, "utf8") : "";
    const timingReport = existsSync(timingReportPath) ? await readFile(timingReportPath, "utf8") : "";
    const combinedReport = [
      ramKillMsg,
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
