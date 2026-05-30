// Push U250 ResNet-50 integrated design as fast as it goes.
//
// Strategy: open the routed checkpoint (or placed.dcp if --from-placed),
// for each target clock period:
//   1. update the design clock constraint
//   2. unroute, re-route at the new constraint
//   3. report timing + utilization, capture WNS
//   4. save per-iteration checkpoint + reports
//
// Each iteration starts from the same input dcp — we are NOT compounding
// router decisions across iterations. The first iteration that misses
// timing tells us the true Fmax ceiling for this placement.
//
// Usage:
//   set NN2RTL_VIVADO_BIN=D:/vivado/2025.2/Vivado/bin/vivado.bat
//   set NN2RTL_VIVADO_TIMEOUT_MS=21600000      # 6 hours per iteration
//   npx tsx scripts/run_fmax_sweep.ts [--input=routed.dcp]
//                                     [--clock-ns-list=12,10,8,6,5,4]
//                                     [--part=xcu250-figd2104-2L-e]
//                                     [--threads=8]
//                                     [--phys-opt]    # add phys_opt_design after route
//
// Output (under output/reports_integrated/fmax_sweep/):
//   clk{N}ns/routed.dcp
//   clk{N}ns/util.rpt
//   clk{N}ns/timing.rpt
//   clk{N}ns/vivado.log
//   clk{N}ns/result.json
//   summary.json     # one row per iteration: {clock_ns, wns, fmax_mhz, success, elapsed_s, lut, ff, dsp}

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
function hasFlag(name: string): boolean {
  return rawArgs.includes(`--${name}`);
}

const part = flag("part") ?? "xcu250-figd2104-2L-e";
const threads = Number(flag("threads") ?? "8");
const inputDcpRaw = flag("input") ?? path.join(repoRoot, "output", "reports_integrated", "checkpoints", "first_light_routed.dcp");
const inputDcp = path.isAbsolute(inputDcpRaw) ? inputDcpRaw : path.resolve(repoRoot, inputDcpRaw);
const clockNsList = (flag("clock-ns-list") ?? "12,10,8,6,5,4")
  .split(",")
  .map((s) => Number(s.trim()))
  .filter((n) => Number.isFinite(n) && n > 0);
const usePhysOpt = hasFlag("phys-opt");
const stopOnFail = hasFlag("stop-on-fail");
const routeDirective = flag("route-directive") ?? "Default";   // Default | Explore | NoTimingRelaxation | etc.

const sweepDir = path.join(repoRoot, "output", "reports_integrated", "fmax_sweep");
const summaryPath = path.join(sweepDir, "summary.json");

function tclQuote(value: string): string {
  return `"${toVivadoPath(value).replace(/(["$[\]])/g, "\\$1")}"`;
}

function buildSweepTcl(input: {
  inputDcp: string;
  clockNs: number;
  routedDcp: string;
  utilRpt: string;
  timingRpt: string;
  timingDetailRpt: string;
}): string {
  return [
    `set_param general.maxThreads ${threads}`,
    `puts "NN2RTL_INFO: sweep iteration clock_ns=${input.clockNs}"`,
    `puts "NN2RTL_INFO: opening input checkpoint"`,
    `open_checkpoint ${tclQuote(input.inputDcp)}`,
    // Override every primary clock's period by re-issuing create_clock at the
    // same source. PERIOD is read-only on a placed dcp, so set_property fails;
    // create_clock without -add replaces the existing same-name clock.
    `set _clocks [get_clocks]`,
    `puts "NN2RTL_INFO: clocks=[llength $_clocks] names=[get_property NAME $_clocks]"`,
    `foreach _clk $_clocks {`,
    `  set _name [get_property NAME $_clk]`,
    `  set _src  [get_property SOURCE_PINS $_clk]`,
    `  if {$_src eq ""} {`,
    `    puts "NN2RTL_WARN: no SOURCE_PINS for clock $_name, skipping retime"`,
    `    continue`,
    `  }`,
    `  puts "NN2RTL_INFO: retiming clock $_name to ${input.clockNs.toFixed(3)} ns at source $_src"`,
    `  if {[catch { create_clock -period ${input.clockNs.toFixed(3)} -name $_name $_src } _err]} {`,
    `    puts "NN2RTL_WARN: create_clock retime failed for $_name: $_err"`,
    `  }`,
    `}`,
    `puts "NN2RTL_INFO: clock periods updated to ${input.clockNs} ns"`,
    // Sanity-check the new period was applied.
    `foreach _clk [get_clocks] {`,
    `  puts "NN2RTL_INFO: post-retime clock [get_property NAME $_clk] period=[get_property PERIOD $_clk]"`,
    `}`,
    `puts "NN2RTL_INFO: unrouting (no-op if input is placed-only)"`,
    `catch { route_design -unroute }`,
    `puts "NN2RTL_INFO: routing at new constraint (directive=${routeDirective})"`,
    routeDirective === "Default"
      ? `route_design`
      : `route_design -directive ${routeDirective}`,
    ...(usePhysOpt
      ? [
          `puts "NN2RTL_INFO: phys_opt_design"`,
          `catch { phys_opt_design -directive Explore }`,
          `puts "NN2RTL_INFO: incremental re-route after phys_opt"`,
          `route_design`,
        ]
      : []),
    `puts "NN2RTL_INFO: write routed checkpoint"`,
    `write_checkpoint -force ${tclQuote(input.routedDcp)}`,
    `puts "NN2RTL_INFO: post-route utilization"`,
    `report_utilization -file ${tclQuote(input.utilRpt)}`,
    `puts "NN2RTL_INFO: post-route timing summary"`,
    `report_timing_summary -check_timing_verbose -max_paths 50 -file ${tclQuote(input.timingRpt)}`,
    `report_timing -delay_type max -max_paths 20 -nworst 1 -unique_pins -path_type full -file ${tclQuote(input.timingDetailRpt)}`,
    `puts "NN2RTL_INFO: sweep iteration ${input.clockNs} ns complete"`,
  ].join("\n") + "\n";
}

const execFileP = promisify(execFile);

interface IterResult {
  clock_ns: number;
  target_mhz: number;
  wns_ns: number | null;
  fmax_mhz: number | null;
  achieved_target: boolean;
  success: boolean;
  elapsed_s: number;
  lut_count: number;
  ff_count: number;
  dsp_count: number;
  bram36_count: number;
  log_path: string;
  routed_dcp: string;
}

async function runOne(clockNs: number, prevDcp: string): Promise<IterResult> {
  const iterDir = path.join(sweepDir, `clk${clockNs}ns`);
  await mkdir(iterDir, { recursive: true });
  const routedDcp = path.join(iterDir, "routed.dcp");
  const utilRpt = path.join(iterDir, "util.rpt");
  const timingRpt = path.join(iterDir, "timing.rpt");
  const timingDetailRpt = path.join(iterDir, "timing_detail.rpt");
  const persistedLog = path.join(iterDir, "vivado.log");
  const resultJson = path.join(iterDir, "result.json");

  if (existsSync(resultJson) && existsSync(routedDcp)) {
    const prior = JSON.parse(await readFile(resultJson, "utf8")) as IterResult;
    console.log(`[fmax] ${clockNs} ns: cached result wns=${prior.wns_ns} fmax=${prior.fmax_mhz}`);
    return prior;
  }

  console.log(`[fmax] === iteration clock_ns=${clockNs} (target ${(1000 / clockNs).toFixed(1)} MHz) ===`);
  const t0 = Date.now();

  const result = await withTempDir(`nn2rtl-fmax-${clockNs}ns-`, async (tempDir) => {
    const tclPath = path.join(tempDir, "fmax_sweep.tcl");
    await writeFile(
      tclPath,
      buildSweepTcl({
        inputDcp: prevDcp,
        clockNs,
        routedDcp,
        utilRpt,
        timingRpt,
        timingDetailRpt,
      }),
      "utf8",
    );

    const vivadoBin = resolveVivadoCommand(process.env);
    const vivadoArgs = ["-mode", "batch", "-source", toVivadoPath(tclPath), "-notrace"];
    const isWindowsBatch = process.platform === "win32" && /\.(bat|cmd)$/i.test(vivadoBin);
    const spawnFile = isWindowsBatch ? "cmd.exe" : vivadoBin;
    const spawnArgs = isWindowsBatch ? ["/c", vivadoBin, ...vivadoArgs] : vivadoArgs;

    console.log(`[fmax] launching vivado in ${tempDir}`);
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

    const vivadoLog = path.join(tempDir, "vivado.log");
    if (existsSync(vivadoLog)) {
      await writeFile(persistedLog, await readFile(vivadoLog, "utf8"), "utf8");
    } else {
      await writeFile(persistedLog, stdout + "\n" + stderr, "utf8");
    }

    return { exitOk, stdout, stderr };
  });

  const elapsed = (Date.now() - t0) / 1000;
  const utilText = existsSync(utilRpt) ? await readFile(utilRpt, "utf8") : "";
  const timingText = existsSync(timingRpt) ? await readFile(timingRpt, "utf8") : "";
  const combined = [result.stdout, result.stderr, utilText, timingText].filter(Boolean).join("\n");
  const parsed = parseVivadoReport(combined, clockNs, part);

  const wns = parsed.setup_wns_ns;
  const fmax = wns !== null && Number.isFinite(wns) ? 1000 / (clockNs - wns) : null;
  const achieved = wns !== null && wns >= 0;

  const iter: IterResult = {
    clock_ns: clockNs,
    target_mhz: 1000 / clockNs,
    wns_ns: wns,
    fmax_mhz: fmax,
    achieved_target: achieved,
    success: result.exitOk && parsed.success,
    elapsed_s: elapsed,
    lut_count: parsed.lut_count,
    ff_count: parsed.ff_count,
    dsp_count: parsed.dsp_count,
    bram36_count: parsed.bram36_count,
    log_path: persistedLog,
    routed_dcp: routedDcp,
  };
  await writeFile(resultJson, JSON.stringify(iter, null, 2), "utf8");

  console.log(
    `[fmax] ${clockNs} ns done in ${elapsed.toFixed(0)}s — WNS=${wns} ns, ` +
      `Fmax=${fmax === null ? "n/a" : fmax.toFixed(1) + " MHz"}, ` +
      `target ${(1000 / clockNs).toFixed(1)} MHz ${achieved ? "MET" : "MISSED"}`,
  );
  return iter;
}

async function main(): Promise<void> {
  await mkdir(sweepDir, { recursive: true });
  if (!existsSync(inputDcp)) {
    throw new Error(`input checkpoint not found: ${inputDcp}`);
  }
  console.log(`[fmax] input dcp: ${inputDcp}`);
  console.log(`[fmax] part=${part} threads=${threads} phys_opt=${usePhysOpt}`);
  console.log(`[fmax] sweep: ${clockNsList.map((n) => `${n}ns(${(1000 / n).toFixed(0)}MHz)`).join(" -> ")}`);

  const results: IterResult[] = [];
  for (const clockNs of clockNsList) {
    const iter = await runOne(clockNs, inputDcp);
    results.push(iter);
    await writeFile(summaryPath, JSON.stringify({ part, threads, input_dcp: inputDcp, iterations: results }, null, 2), "utf8");
    if (stopOnFail && !iter.achieved_target) {
      console.log(`[fmax] stop-on-fail: ${clockNs} ns missed target — halting sweep`);
      break;
    }
  }

  console.log("\n[fmax] === summary ===");
  console.log("clock_ns | target_MHz | WNS_ns | Fmax_MHz | MET?");
  for (const r of results) {
    console.log(
      `${r.clock_ns.toString().padStart(8)} | ` +
        `${r.target_mhz.toFixed(1).padStart(10)} | ` +
        `${r.wns_ns === null ? "n/a".padStart(6) : r.wns_ns.toFixed(3).padStart(6)} | ` +
        `${r.fmax_mhz === null ? "n/a".padStart(8) : r.fmax_mhz.toFixed(1).padStart(8)} | ` +
        `${r.achieved_target ? "yes" : "NO"}`,
    );
  }

  const bestMet = results.filter((r) => r.achieved_target).sort((a, b) => a.clock_ns - b.clock_ns)[0];
  if (bestMet) {
    console.log(`\n[fmax] tightest constraint met: ${bestMet.clock_ns} ns -> ${bestMet.target_mhz.toFixed(1)} MHz`);
    console.log(`[fmax] true Fmax estimate from that run: ${bestMet.fmax_mhz?.toFixed(1)} MHz`);
  } else {
    console.log(`\n[fmax] NO constraint met — design floor is below ${clockNsList[clockNsList.length - 1]} ns`);
  }
}

main().catch((err: unknown) => {
  console.error(err instanceof Error ? err.stack ?? err.message : String(err));
  process.exit(1);
});
