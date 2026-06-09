// MobileNetV2 ENGINE-TOP Vivado synth/impl driver (adapted from
// scripts/run_first_light_synth.ts, which targets the ResNet top).
//
// Synthesises the full INTEGRATED MBV2 engine-active design — the SAME source
// set the Verilator e2e (scripts/run_mbv2_top_engine_value.ts) compiles:
//   - engine top  output/mobilenet-v2/rtl/nn2rtl_top_engine.v   (module nn2rtl_top;
//     skip_fifo / engine_output_fifo / bridges / bias_mem are INLINE in it)
//   - scheduler   output/mobilenet-v2/rtl/nn2rtl_scheduler.v
//   - every per-layer module output/mobilenet-v2/rtl/node_*.v (+ n4_*) — EXCLUDING
//     nn2rtl_top.v (all-spatial duplicate of `module nn2rtl_top`) + *.pre_skipwire.v
//     / *.preimprove / *.bak* snapshots
//   - REAL shared_engine output/rtl/shared_engine_skeleton.v + 5 engine sub-blocks
//     (suppress the skeleton's stub submodules via NN2RTL_ENGINE_SUBBLOCKS_PROVIDED)
//   - rtl_library helpers (conv_datapath{,_parallel,_mp_k}, coord_scheduler,
//     line_buf_window, retile_bridge)
//
// $readmemh paths are rewritten to absolute so Vivado finds the weight .hex/.mem.
//
// PER-STAGE CHECKPOINTS are persisted to output/mobilenet-v2/reports/synth/checkpoints/
// (post_synth / post_opt / post_placed / post_routed) so a failure or timeout NEVER
// forces a full re-run — resume with open_checkpoint.
//
// Usage:
//   set NN2RTL_VIVADO_BIN=D:/vivado/2025.2/Vivado/bin/vivado.bat
//   set NN2RTL_VIVADO_TIMEOUT_MS=43200000   (12h)
//   npx tsx scripts/run_mbv2_synth.ts [--part=xcu250-figd2104-2L-e]
//        [--clock-ns=10] [--threads=8] [--synth-only]

import { readFile, writeFile, mkdir, copyFile, readdir, rm, rename } from "node:fs/promises";
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
  if (idx >= 0 && rawArgs[idx + 1] && !rawArgs[idx + 1].startsWith("--")) return rawArgs[idx + 1];
  return fallback;
}

const part = flag("part") ?? "xcu250-figd2104-2L-e";
const clockNs = Number(flag("clock-ns") ?? "10");
const threads = Number(flag("threads") ?? "8");
const topModule = "nn2rtl_top";
const synthOnly = rawArgs.includes("--synth-only");
// [FMAX 2026-06-07] OUT-OF-CONTEXT synth: the engine-top exposes m_axis_tdata[7999:0] (8000
// parallel logit bits) -> 8000 OBUFs, far over the U250's ~676 I/O pins, so a normal-mode
// place_design dies with "IO Placement failed due to overutilization". This is a streaming
// accelerator CORE meant to attach to on-chip AXI (the 8000-bit output is never pins), so we
// implement it OUT-OF-CONTEXT: synth_design -mode out_of_context inserts NO I/O buffers, the
// ports become OOC boundary nets (no pin placement), and place/route + timing give the true
// internal LOGIC Fmax. I/O buffers are dedicated cells (~0 LUT) so the FIT numbers are unchanged.
const ooc = rawArgs.includes("--ooc");
const flatten = flag("flatten") ?? "rebuilt";  // rebuilt=optimized(RAM-heavy, OOMs here) | none=low-RAM, completes

const mbvRtlDir = path.join(repoRoot, "output", "mobilenet-v2", "rtl");
const weightsDir = path.join(repoRoot, "output", "mobilenet-v2", "weights");
const reportsDir = path.join(repoRoot, "output", "mobilenet-v2", "reports", "synth");
const safeCheckpointDir = path.join(reportsDir, "checkpoints");
const jsonReportPath = path.join(reportsDir, "mbv2_synth.json");
const logPath = path.join(reportsDir, "mbv2_synth.log");

const sharedEngineSkeleton = path.join(repoRoot, "output", "rtl", "shared_engine_skeleton.v");
const engineSubmodules = [
  path.join(repoRoot, "output", "rtl", "engine", "mac_array.v"),
  path.join(repoRoot, "output", "rtl", "engine", "address_generator.v"),
  path.join(repoRoot, "output", "rtl", "engine", "config_register_block.v"),
  path.join(repoRoot, "output", "rtl", "engine", "requant_pipeline.v"),
  path.join(repoRoot, "output", "rtl", "engine", "bram_to_stream_bridge.v"),
];

function isExcludedRtl(entry: string): boolean {
  if (entry === "nn2rtl_top.v") return true;          // all-spatial duplicate of `module nn2rtl_top`
  if (entry.endsWith(".pre_skipwire.v")) return true; // pre-skipwire snapshot (also defines nn2rtl_top)
  if (entry.endsWith(".preimprove")) return true;     // failure-corpus snapshots
  if (entry.includes(".bak")) return true;            // *.bak_* backups
  return false;
}

function tclQuote(value: string): string {
  return `"${toVivadoPath(value).replace(/(["$[\]])/g, "\\$1")}"`;
}

function convertReadmemhAbs(source: string, repoRootAbs: string): string {
  const fix = (p: string): string => toVivadoPath(path.isAbsolute(p) ? p : path.resolve(repoRootAbs, p));
  return source
    .replace(/(\$readmemh\s*\(\s*)"([^"]+)"/g, (_m, prefix: string, p: string) => `${prefix}"${fix(p)}"`)
    .replace(/(MEM_INIT_FILE\s*[=(]\s*)"([^"]+)"/g,
      (_m, prefix: string, p: string) => (p.length === 0 ? `${prefix}""` : `${prefix}"${fix(p)}"`));
}

async function collectSources(): Promise<string[]> {
  const out: string[] = [
    path.join(repoRoot, "rtl_library", "conv_datapath.v"),
    path.join(repoRoot, "rtl_library", "conv_datapath_parallel.v"),
    path.join(repoRoot, "rtl_library", "conv_datapath_mp_k.v"),
    path.join(repoRoot, "rtl_library", "coord_scheduler.v"),
    path.join(repoRoot, "rtl_library", "line_buf_window.v"),
    path.join(repoRoot, "rtl_library", "retile_bridge.v"),
    sharedEngineSkeleton,
    ...engineSubmodules,
  ];
  const entries = await readdir(mbvRtlDir);
  for (const entry of entries) {
    if (entry.endsWith(".v") && !isExcludedRtl(entry)) out.push(path.join(mbvRtlDir, entry));
  }
  const tops = out.filter((p) => path.basename(p) === "nn2rtl_top_engine.v" || path.basename(p) === "nn2rtl_top.v");
  if (tops.length !== 1 || path.basename(tops[0]) !== "nn2rtl_top_engine.v") {
    throw new Error(`expected exactly the engine top, got: ${tops.map((p) => path.basename(p)).join(", ")}`);
  }
  for (const p of out) if (!existsSync(p)) throw new Error(`source missing: ${p}`);
  return out;
}

function buildTcl(input: {
  verilogPaths: string[];
  utilReportPath: string;
  timingReportPath: string;
  checkpointPath: string;
}): string {
  const postRouteUtilPath = input.utilReportPath.replace(/_util\.rpt$/, "_postroute_util.rpt");
  const postRouteTimingPath = input.timingReportPath.replace(/_timing\.rpt$/, "_postroute_timing.rpt");
  const postRoutePowerPath = input.timingReportPath.replace(/_timing\.rpt$/, "_postroute_power.rpt");
  const placedUtilPath = input.utilReportPath.replace(/_util\.rpt$/, "_placed_util.rpt");
  const placedTimingPath = input.timingReportPath.replace(/_timing\.rpt$/, "_placed_timing.rpt");
  const synthDcpPath = input.checkpointPath.replace(/\.dcp$/, "_synth.dcp");
  const optDcpPath = input.checkpointPath.replace(/\.dcp$/, "_opt.dcp");
  const placedDcpPath = input.checkpointPath.replace(/\.dcp$/, "_placed.dcp");
  const physoptDcpPath = input.checkpointPath.replace(/\.dcp$/, "_physopt.dcp");
  // [FMAX 2026-06-09] SLR floorplan: pin each hot deep-conv (854..908)+bridge to ONE SLR so
  // the acc->scheduler->acc loop never crosses an SLL (the c8 critical path was 91% ROUTE,
  // ~10.06ns of it TWO SLR crossings inside conv_866). read_xdc BEFORE place_design. Placement
  // only -> byte-exact by construction. Absolute path to repo's committed pblock XDC.
  const pblockXdc = path.join(reportsDir, "mbv2_fmax_pblock.xdc");
  return [
    `set_param general.maxThreads ${threads}`,
    `puts "NN2RTL_INFO: maxThreads requested=${threads} effective=[get_param general.maxThreads]"`,
    `read_verilog -sv ${input.verilogPaths.map(tclQuote).join(" \\\n                 ")}`,
    `puts "NN2RTL_INFO: auto_detect_xpm"`,
    `auto_detect_xpm`,
    `puts "NN2RTL_INFO: synth_design (defines: NN2RTL_ENGINE_SUBBLOCKS_PROVIDED + NN2RTL_SYNTHESIS)"`,
    `synth_design -top ${topModule} -part ${part}${ooc ? " -mode out_of_context" : ""} -flatten_hierarchy ${flatten} -verilog_define NN2RTL_ENGINE_SUBBLOCKS_PROVIDED=1 -verilog_define NN2RTL_SYNTHESIS=1`,
    `create_clock -name clk -period ${clockNs} [get_ports clk]`,
    `puts "NN2RTL_INFO: synth-level utilization (the FIT answer; saved + mirrored)"`,
    `report_utilization -file ${tclQuote(input.utilReportPath + ".synth")}`,
    `report_utilization -hierarchical -hierarchical_depth 3 -file ${tclQuote(input.utilReportPath + ".synth.hier")}`,
    `puts "NN2RTL_INFO: write SYNTH checkpoint (persisted even if place/route later fail)"`,
    `write_checkpoint -force ${tclQuote(synthDcpPath)}`,
    ...(synthOnly
      ? [
          `file copy -force ${tclQuote(input.utilReportPath + ".synth")} ${tclQuote(input.utilReportPath)}`,
          `puts "NN2RTL_INFO: --synth-only complete (no opt/place/route)"`,
        ]
      : [
          `puts "NN2RTL_INFO: opt_design -directive ExploreWithRemap"`,
          `opt_design -directive ExploreWithRemap`,
          `write_checkpoint -force ${tclQuote(optDcpPath)}`,
          // SLR floorplan pblock: read AFTER opt (cells exist), BEFORE place. -quiet so a
          // stale/renamed instance never aborts the 12h flow; the catch logs but proceeds.
          `if {[file exists ${tclQuote(pblockXdc)}]} { puts "NN2RTL_INFO: read_xdc pblock"; catch { read_xdc ${tclQuote(pblockXdc)} } err; if {$err ne ""} { puts "NN2RTL_WARN: pblock read: $err" } } else { puts "NN2RTL_WARN: pblock XDC missing -> placing without floorplan" }`,
          `puts "NN2RTL_INFO: place_design -directive SSI_SpreadLogic_high"`,
          `place_design -directive SSI_SpreadLogic_high`,
          `write_checkpoint -force ${tclQuote(placedDcpPath)}`,
          `report_utilization -file ${tclQuote(placedUtilPath)}`,
          `report_timing_summary -max_paths 10 -file ${tclQuote(placedTimingPath)}`,
          // Pre-route timing closure: two phys_opt passes (SLL/fanout-aware) on the placed netlist.
          `puts "NN2RTL_INFO: phys_opt_design x2 (pre-route)"`,
          `catch { phys_opt_design }`,
          `catch { phys_opt_design }`,
          `write_checkpoint -force ${tclQuote(physoptDcpPath)}`,
          `puts "NN2RTL_INFO: route_design (directive AggressiveExplore)"`,
          `route_design -directive AggressiveExplore`,
          `write_checkpoint -force ${tclQuote(input.checkpointPath)}`,
          // Post-route timing closure.
          `puts "NN2RTL_INFO: post-route phys_opt_design"`,
          `catch { phys_opt_design }`,
          `write_checkpoint -force ${tclQuote(input.checkpointPath)}`,
          `report_utilization -file ${tclQuote(postRouteUtilPath)}`,
          `report_timing_summary -check_timing_verbose -max_paths 20 -file ${tclQuote(postRouteTimingPath)}`,
          `report_power -file ${tclQuote(postRoutePowerPath)}`,
          `catch { file copy -force ${tclQuote(postRouteUtilPath)} ${tclQuote(input.utilReportPath)} }`,
          `catch { file copy -force ${tclQuote(postRouteTimingPath)} ${tclQuote(input.timingReportPath)} }`,
          // Cheap insurance: re-open the just-written routed dcp in a fresh in-memory project to
          // PROVE it is not the corrupt/partial-write that bit the c8 resume flow. Non-fatal.
          `puts "NN2RTL_INFO: read_checkpoint sanity (routed dcp re-open)"`,
          `catch { read_checkpoint ${tclQuote(input.checkpointPath)} } rcerr; if {$rcerr ne ""} { puts "NN2RTL_WARN: routed dcp re-open FAILED: $rcerr" } else { puts "NN2RTL_INFO: routed dcp re-open OK (not corrupt)" }`,
          `puts "NN2RTL_INFO: full flow complete"`,
        ]),
  ].join("\n") + "\n";
}

const execFileP = promisify(execFile);

async function main(): Promise<void> {
  await mkdir(reportsDir, { recursive: true });
  await mkdir(safeCheckpointDir, { recursive: true });

  // Clear STALE outputs from PRIOR runs so THIS run's fresh checkpoints/reports persist.
  // The poller below is write-once within a run (so it never clobbers a good earlier-stage
  // dcp with a mid-write); but a leftover dst from a previous run would silently block the
  // fresh copy -> stale dcp/hier. Deleting at startup makes every run's outputs current.
  for (const stale of ["mbv2_post_synth.dcp", "mbv2_post_opt.dcp", "mbv2_post_placed.dcp", "mbv2_post_routed.dcp"]) {
    const p = path.join(safeCheckpointDir, stale);
    if (existsSync(p)) { await rm(p, { force: true }); console.log(`[mbv2-synth] cleared stale ${stale}`); }
  }
  for (const stale of ["mbv2_util.rpt.synth", "mbv2_util.rpt.synth.hier", "mbv2_placed_util.rpt",
    "mbv2_placed_timing.rpt", "mbv2_postroute_util.rpt", "mbv2_postroute_timing.rpt"]) {
    const p = path.join(reportsDir, stale);
    if (existsSync(p)) { await rm(p, { force: true }); }
  }

  const sources = await collectSources();
  console.log(`[mbv2-synth] collected ${sources.length} source files`);
  console.log(`[mbv2-synth] top=${topModule} part=${part} clock_ns=${clockNs} threads=${threads} synthOnly=${synthOnly}`);

  const report = await withTempDir("nn2rtl-mbv2synth-", async (tempDir) => {
    const copiedPaths: string[] = [];
    for (const src of sources) {
      const dest = path.join(tempDir, path.basename(src));
      const text = await readFile(src, "utf8");
      await writeFile(dest, convertReadmemhAbs(text, repoRoot), "utf8");
      copiedPaths.push(dest);
    }
    // belt-and-suspenders: copy weight .mem/.hex into tempdir (in case any lookup is basename-relative)
    if (existsSync(weightsDir)) {
      const mems = (await readdir(weightsDir)).filter((f) => f.endsWith(".mem") || f.endsWith(".hex"));
      for (const m of mems) await copyFile(path.join(weightsDir, m), path.join(tempDir, m));
      console.log(`[mbv2-synth] copied ${mems.length} weight .mem/.hex into tempdir`);
    }

    const utilReportPath = path.join(tempDir, "mbv2_util.rpt");
    const timingReportPath = path.join(tempDir, "mbv2_timing.rpt");
    const checkpointPath = path.join(tempDir, "mbv2.dcp");
    const tclPath = path.join(tempDir, "mbv2_synth.tcl");
    await writeFile(tclPath, buildTcl({ verilogPaths: copiedPaths, utilReportPath, timingReportPath, checkpointPath }), "utf8");

    const vivadoBin = resolveVivadoCommand(process.env);
    const vivadoArgs = ["-mode", "batch", "-source", toVivadoPath(tclPath), "-notrace"];
    const isWindowsBatch = process.platform === "win32" && /\.(bat|cmd)$/i.test(vivadoBin);
    const spawnFile = isWindowsBatch ? "cmd.exe" : vivadoBin;
    const spawnArgs = isWindowsBatch ? ["/c", vivadoBin, ...vivadoArgs] : vivadoArgs;
    console.log(`[mbv2-synth] launching: ${spawnFile} ${spawnArgs.join(" ")}`);

    const timeoutMs = (() => {
      const e = process.env.NN2RTL_VIVADO_TIMEOUT_MS;
      return e && Number.isFinite(Number(e)) && Number(e) > 0 ? Number(e) : VIVADO_TIMEOUT_MS;
    })();
    console.log(`[mbv2-synth] timeout=${(timeoutMs / 3600000).toFixed(1)}h`);

    // Persist checkpoints from tempdir to the safe dir AS THEY APPEAR (polled), so a
    // timeout/kill mid-run still leaves us the latest stage to resume from.
    const dcpMap = [
      { src: checkpointPath.replace(/\.dcp$/, "_synth.dcp"), dst: "mbv2_post_synth.dcp" },
      { src: checkpointPath.replace(/\.dcp$/, "_opt.dcp"), dst: "mbv2_post_opt.dcp" },
      { src: checkpointPath.replace(/\.dcp$/, "_placed.dcp"), dst: "mbv2_post_placed.dcp" },
      { src: checkpointPath, dst: "mbv2_post_routed.dcp" },
    ];
    const reportNames = ["mbv2_util.rpt.synth", "mbv2_util.rpt.synth.hier", "mbv2_placed_util.rpt",
      "mbv2_placed_timing.rpt", "mbv2_postroute_util.rpt", "mbv2_postroute_timing.rpt",
      // preserve Vivado's own logs (tempdir is auto-deleted) so synth crashes/errors are diagnosable
      "vivado.log", "vivado.jou"];
    let polling = true;
    // [DCP-CORRUPTION FIX 2026-06-09] The 30s poller can copy a dcp WHILE Vivado's write_checkpoint
    // is still writing it -> a PARTIAL (corrupt) dst. The old `!existsSync(d)` then made the FINAL
    // persist SKIP it (dst exists) -> the partial stuck -> open_checkpoint fails ("checkpoint not
    // open"). Fix: (1) copy to a .tmp then atomic rename so a dst is never a half-written file;
    // (2) the FINAL persist(force=true) ALWAYS re-copies, so the now-COMPLETE src overwrites any
    // partial the poller left. Poller stays best-effort (force=false).
    const persist = async (force = false) => {
      for (const { src, dst } of dcpMap) {
        const d = path.join(safeCheckpointDir, dst);
        if (existsSync(src) && (force || !existsSync(d))) {
          try { const tmp = d + ".tmp"; await copyFile(src, tmp); await rename(tmp, d); console.log(`[mbv2-synth] persisted ${dst}${force ? " (final/forced)" : ""}`); } catch { /* mid-write */ }
        }
      }
      for (const r of reportNames) {
        const s = path.join(tempDir, r), d = path.join(reportsDir, r);
        // vivado.log/.jou grow during the run -> always refresh; reports are write-once.
        const alwaysRefresh = r === "vivado.log" || r === "vivado.jou";
        if (existsSync(s) && (alwaysRefresh || !existsSync(d))) { try { await copyFile(s, d); } catch { /* mid-write */ } }
      }
    };
    const poller = (async () => { while (polling) { await persist(); await new Promise((r) => setTimeout(r, 30000)); } })();

    const t0 = Date.now();
    let stdout = "", stderr = "", exitOk = true;
    let ramKillMsg = "";
    {
      // RAM watchdog (90% default; NN2RTL_RAM_KILL_PCT override): HARD STOP if physical RAM
      // crosses the threshold. The MBV2 -flatten_hierarchy rebuilt synth OOM'd this host before
      // -> this is the guard. The poller persists checkpoints, so a kill is recoverable.
      const ramKillPct = (() => {
        const v = Number(process.env.NN2RTL_RAM_KILL_PCT);
        return Number.isFinite(v) && v > 0 && v < 100 ? v : 90;
      })();
      const totalMem = os.totalmem();
      console.log(
        `[mbv2-synth] RAM watchdog armed: kill at >= ${ramKillPct}% used ` +
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
              `[mbv2-synth][WATCHDOG] RAM ${usedPct.toFixed(1)}% used >= ${ramKillPct}% ` +
              `(free ${freeGB.toFixed(1)}GB) — KILLING Vivado tree (pid ${child.pid}); resume from latest mbv2_post_*.dcp`;
            console.error(ramKillMsg);
            clearInterval(poll);
            try {
              if (child.pid) execFileP("taskkill", ["/PID", String(child.pid), "/T", "/F"]).catch(() => {});
            } catch { /* ignore */ }
            try {
              execFileP("taskkill", ["/IM", "vivado.exe", "/T", "/F"]).catch(() => {});
            } catch { /* ignore */ }
          }
        }, 4000);
      });
      stdout = res.stdout; stderr = res.stderr; exitOk = res.ok;
    }
    polling = false; await poller; await persist(true);   // FINAL persist FORCES re-copy of the complete dcps
    const elapsed = (Date.now() - t0) / 1000;
    console.log(`[mbv2-synth] vivado returned in ${elapsed.toFixed(1)}s (ok=${exitOk}${ramKillMsg ? ", RAM-KILLED" : ""})`);

    const utilReport = existsSync(utilReportPath) ? await readFile(utilReportPath, "utf8")
      : (existsSync(path.join(reportsDir, "mbv2_util.rpt.synth")) ? await readFile(path.join(reportsDir, "mbv2_util.rpt.synth"), "utf8") : "");
    const timingReport = existsSync(timingReportPath) ? await readFile(timingReportPath, "utf8") : "";
    const combined = [ramKillMsg, stdout, stderr, "--- util ---", utilReport, "--- timing ---", timingReport].filter(Boolean).join("\n");
    await writeFile(logPath, combined, "utf8");
    const parsed = parseVivadoReport(combined, clockNs, part);
    parsed.success = exitOk && parsed.success;
    return { ...parsed, elapsed_s: elapsed };
  });

  await writeFile(jsonReportPath, JSON.stringify(report, null, 2), "utf8");
  console.log(`[mbv2-synth] wrote ${path.relative(repoRoot, jsonReportPath)}`);
  const r = report as VivadoSynthesisReport & { elapsed_s: number };
  console.log(`[mbv2-synth] success=${report.success} LUT=${report.lut_count} FF=${report.ff_count} DSP=${report.dsp_count} BRAM36=${report.bram36_count} WNS=${report.setup_wns_ns ?? "n/a"} elapsed=${r.elapsed_s.toFixed(0)}s`);
  if (!report.success) { console.error("[mbv2-synth] FAILED/incomplete — see log + checkpoints"); process.exit(2); }
}

main().catch((err) => {
  console.error("[mbv2-synth] FATAL:", err instanceof Error ? (err.stack ?? err.message) : String(err));
  process.exit(1);
});
