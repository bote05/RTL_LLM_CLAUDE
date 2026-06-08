// Build + run the Verilator END-TO-END VALUE testbench for the MobileNetV2
// ENGINE-DISPATCHED nn2rtl_top_engine.v
// (output/mobilenet-v2/rtl/nn2rtl_top_engine.v).
//
// This is the engine-active sibling of scripts/run_mbv2_top_value.ts (which
// drives the ALL-SPATIAL nn2rtl_top.v with an INACTIVE engine+scheduler STUB:
// engine_busy=0, spatial_stall=0). Here we compile the REAL engine path:
//   - the engine top output/mobilenet-v2/rtl/nn2rtl_top_engine.v
//   - the REAL shared_engine (output/rtl/shared_engine_skeleton.v, the module
//     at lines 1-679 ONLY) + its 5 REAL submodules
//     output/rtl/engine/{mac_array,address_generator,config_register_block,
//     requant_pipeline,bram_to_stream_bridge}.v
//   - the REAL scheduler output/mobilenet-v2/rtl/nn2rtl_scheduler.v
// so the engine actually dispatches the 34 heavy modules.
//
// Skeleton stub suppression:
//   output/rtl/shared_engine_skeleton.v carries the real shared_engine
//   (lines 1-679) followed by EMPTY stub submodules (lines ~706-893) wrapped
//   in `\`ifndef NN2RTL_ENGINE_SUBBLOCKS_PROVIDED`. nn2rtl_top_engine.v line 13
//   already `\`define`s that macro, but to make the build independent of file
//   ordering we ALSO pass -DNN2RTL_ENGINE_SUBBLOCKS_PROVIDED on the verilator
//   command line, which compiles ONLY the real shared_engine and pulls the
//   real submodules from output/rtl/engine/*.v. No RTL is mutated.
//
// Engine param override (PROVEN byte-exact 34/34):
//   nn2rtl_top_engine.v instantiates shared_engine #(.WGT_W(8),
//   .URAM_DATA_W(2048), ...) at lines 2167-2176; we do NOT touch it.
//
// Duplicate-module avoidance:
//   The engine top and the all-spatial top BOTH define `module nn2rtl_top`.
//   We compile ONLY nn2rtl_top_engine.v and EXCLUDE nn2rtl_top.v and any
//   *.pre_skipwire.v / *.bak_* snapshot. All wrapper-local helper modules
//   (skip_fifo / uram_weight_bank / stream_to_act_bram_bridge / act_unified_mem
//   / engine_output_fifo / bias_mem / engine_output_bridge) are defined INSIDE
//   nn2rtl_top_engine.v itself, so no extra stub is synthesized here.
//
// Weight/bias/scale .mem:
//   nn2rtl_top_engine.v references output/mobilenet-v2/weights/*.mem via
//   $readmemh with paths RELATIVE TO REPO ROOT, so the sim MUST run from the
//   repo root for them to resolve. We launch with cwd = repoRoot.
//
// TB: tb/mbv2_top_value_tb.cpp is reused VERBATIM — same external AXIS
//   interface (24-bit RGB s_axis in, 8000-bit logits m_axis out) and the same
//   node_conv_810.goldin / node_linear.goldout goldens. Only the Verilated top
//   + its source set + the active engine differ.
//
// Usage:
//   npx tsx scripts/run_mbv2_top_engine_value.ts [vector_idx]
//   env:
//     MBV2_VEC=<n>            override vector index (argv wins)
//     MBV2_ALL_VECS=1         run all 8 vectors (slow)
//     MBV2_MAX_CYCLES=<n>     per-frame cycle cap (deadlock fail-fast)
//     NN2RTL_VALUE_RUNONLY=1  reuse the existing exe (skip rebuild)
//     NN2RTL_DUMP_PATH=<p>    raw m_axis capture dump
//
// Output:
//   output/mobilenet-v2/reports/verilator_mbv2_top_engine_value/
//     build.log  run.log  result.json
//   build dir (PRIVATE, distinct from the all-spatial obj_dir_value):
//     output/mobilenet-v2/reports/verilator_mbv2_top_engine_value/obj_dir_engine_value
//
// NOTE: per task, this run BUILDS (verilate + compile) but does NOT run the
// full sim unless invoked with a vector and the build succeeds. The parent
// orchestrator decides when to run the sim.

import { readdir, readFile, writeFile, mkdir, rm } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";
import { spawn } from "node:child_process";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

const mbv2Root  = path.join(repoRoot, "output", "mobilenet-v2");
const rtlDir    = path.join(mbv2Root, "rtl");
const goldenDir = path.join(mbv2Root, "goldens");
const tbCpp     = path.join(repoRoot, "tb", "mbv2_top_value_tb.cpp");
const outDir    = path.join(mbv2Root, "reports", "verilator_mbv2_top_engine_value");
const buildLog  = path.join(outDir, "build.log");
const runLog    = path.join(outDir, "run.log");
const resultJson = path.join(outDir, "result.json");

// REAL engine sources (engine path ACTIVE — NOT stubbed).
const sharedEngineSkeleton = path.join(repoRoot, "output", "rtl", "shared_engine_skeleton.v");
const engineSubmodules = [
  path.join(repoRoot, "output", "rtl", "engine", "mac_array.v"),
  path.join(repoRoot, "output", "rtl", "engine", "address_generator.v"),
  path.join(repoRoot, "output", "rtl", "engine", "config_register_block.v"),
  path.join(repoRoot, "output", "rtl", "engine", "requant_pipeline.v"),
  path.join(repoRoot, "output", "rtl", "engine", "bram_to_stream_bridge.v"),
];

const goldinPath  = path.join(goldenDir, "node_conv_810.goldin");
const goldoutPath = path.join(goldenDir, "node_linear.goldout");

const vectorIdx = process.argv[2] ?? "0";

const verilatorBin = process.platform === "win32"
  ? "C:/Users/User/oss-cad-suite/bin/verilator_bin.exe" : "verilator";
const makeBin = process.platform === "win32"
  ? "C:/Users/User/w64devkit/bin/make.exe" : "make";

function toForwardSlash(p: string): string { return p.replace(/\\/g, "/"); }

// The engine top is the ONLY rtl/*.v file we want named "nn2rtl_top". Every
// other live per-layer module (node_*, n4_*, nn2rtl_scheduler) is included.
// EXCLUDE: nn2rtl_top.v (the all-spatial sibling — duplicate module name),
// any *.pre_skipwire.v snapshot, any *.preimprove, and (defensively) any *.bak*.
function isExcludedRtl(entry: string): boolean {
  if (entry === "nn2rtl_top.v") return true;            // all-spatial duplicate of `module nn2rtl_top`
  if (entry.endsWith(".pre_skipwire.v")) return true;   // pre-skipwire snapshot (also defines nn2rtl_top)
  if (entry.endsWith(".preimprove")) return true;       // failure-corpus snapshots
  if (entry.includes(".bak")) return true;              // *.bak_* backups
  return false;
}

async function collectSources(): Promise<string[]> {
  // 5 rtl_library helpers (shared spatial datapath/scheduler primitives) +
  // the wave-2 tiled<->flat retile bridges (retile_gather / retile_scatter).
  const out: string[] = [
    path.join(repoRoot, "rtl_library", "conv_datapath.v"),
    path.join(repoRoot, "rtl_library", "conv_datapath_parallel.v"),
    path.join(repoRoot, "rtl_library", "conv_datapath_mp_k.v"),
    path.join(repoRoot, "rtl_library", "coord_scheduler.v"),
    path.join(repoRoot, "rtl_library", "line_buf_window.v"),
    path.join(repoRoot, "rtl_library", "retile_bridge.v"),
    // REAL shared_engine (lines 1-679; its stub submodules are suppressed by
    // -DNN2RTL_ENGINE_SUBBLOCKS_PROVIDED) + its 5 REAL submodules.
    sharedEngineSkeleton,
    ...engineSubmodules,
  ];
  // all live .v in the mbv2 rtl dir: the engine top + the 99 per-layer modules
  // + the REAL nn2rtl_scheduler.v. Exclude the all-spatial top + snapshots
  // (see isExcludedRtl) so there is exactly ONE `module nn2rtl_top`.
  const entries = await readdir(rtlDir);
  for (const entry of entries) {
    if (entry.endsWith(".v") && !isExcludedRtl(entry)) {
      out.push(path.join(rtlDir, entry));
    }
  }
  // Sanity: exactly one nn2rtl_top source (the engine top).
  const tops = out.filter((p) => path.basename(p) === "nn2rtl_top_engine.v"
                              || path.basename(p) === "nn2rtl_top.v");
  if (tops.length !== 1 || path.basename(tops[0]) !== "nn2rtl_top_engine.v") {
    throw new Error(`expected exactly the engine top, got: ${tops.map((p) => path.basename(p)).join(", ")}`);
  }
  for (const p of out) if (!existsSync(p)) throw new Error(`source missing: ${p}`);
  return out;
}

function runCmd(cmd: string, args: string[], cwd: string, env: NodeJS.ProcessEnv, label: string): Promise<{ stdout: string; stderr: string; ok: boolean }> {
  console.log(`[${label}] ${cmd} ${args.slice(0, 6).join(" ")}${args.length > 6 ? " ... +" + (args.length - 6) + " args" : ""}`);
  return new Promise((resolve) => {
    const child = spawn(cmd, args, { cwd, env, stdio: ["ignore", "pipe", "pipe"], shell: false });
    let stdout = ""; let stderr = "";
    child.stdout.on("data", (c) => { const s = c.toString(); stdout += s; process.stdout.write(s); });
    child.stderr.on("data", (c) => { const s = c.toString(); stderr += s; process.stderr.write(s); });
    child.on("close", (code) => resolve({ stdout, stderr, ok: code === 0 }));
    child.on("error", (err) => resolve({ stdout, stderr: stderr + "\n" + String(err), ok: false }));
  });
}

async function main(): Promise<void> {
  await mkdir(outDir, { recursive: true });
  if (existsSync(buildLog)) await rm(buildLog);
  if (existsSync(runLog)) await rm(runLog);

  if (!existsSync(goldinPath))  throw new Error(`goldin missing: ${goldinPath}`);
  if (!existsSync(goldoutPath)) throw new Error(`goldout missing: ${goldoutPath}`);
  if (!existsSync(tbCpp))       throw new Error(`testbench missing: ${tbCpp}`);
  if (!existsSync(sharedEngineSkeleton)) throw new Error(`shared_engine missing: ${sharedEngineSkeleton}`);
  for (const m of engineSubmodules) if (!existsSync(m)) throw new Error(`engine submodule missing: ${m}`);
  console.log(`[setup] goldin : ${path.relative(repoRoot, goldinPath)}`);
  console.log(`[setup] goldout: ${path.relative(repoRoot, goldoutPath)}`);

  // PRIVATE build dir — distinct from the all-spatial obj_dir_value so this
  // does not collide with the running clean all-spatial e2e.
  const buildDir = path.join(outDir, "obj_dir_engine_value");
  const env = { ...process.env, PATH: `C:/Users/User/oss-cad-suite/bin;C:/Users/User/w64devkit/bin;${process.env.PATH ?? ""}` };

  if (process.env.NN2RTL_VALUE_RUNONLY !== "1") {
    if (existsSync(buildDir)) {
      console.log(`[setup] clearing previous build dir ${buildDir}`);
      for (let attempt = 0; attempt < 5; attempt++) {
        try { await rm(buildDir, { recursive: true, force: true }); break; }
        catch (e: any) {
          if (attempt === 4) console.log(`[setup] rm failed after retries (${e.code}); proceeding`);
          else { console.log(`[setup] rm attempt ${attempt + 1} failed (${e.code}); retrying...`); await new Promise(r => setTimeout(r, 2000)); }
        }
      }
    }
    await mkdir(buildDir, { recursive: true });

    const sources = await collectSources();
    console.log(`[setup] collected ${sources.length} RTL files (REAL engine path active); tb=${path.relative(repoRoot, tbCpp)}`);
    const verilatorArgs = [
      // [DETERMINISM 2026-06-08, #5 root-caused] default to SINGLE-THREAD. CORRECTED: there is NO
      // false combinational loop -- a full unsuppressed Verilator lint reports 0 UNOPTFLAT / 0
      // MULTIDRIVEN. The --threads 4 path produces WRONG VALUES (e2e mismatch_bytes = 688 on the
      // pre-per-channel netlist, 739 on the current one; identical cycle count, so timing is fine
      // but logits are corrupt), and the result is DETERMINISTIC-PER-NETLIST but structure-dependent
      // (flips when the netlist changes). That signature = a Verilator MULTITHREADED-SCHEDULER data
      // hazard (a cross-partition read/write order the MT engine gets wrong), NOT a design bug:
      // --threads 1 is byte-exact AND Vivado synth is clean = the hardware is correct. Fixing the
      // MT path would be a deep Verilator-internals effort with sim-speed-only payoff, so we PIN
      // --threads 1 as the authoritative gate. Set MBV2_THREADS=4 only for throwaway speed (UNSAFE
      // for verification -- it will report false mismatches).
      "--cc", "--exe", "-O3", "--threads", String(process.env.MBV2_THREADS ?? "1"), "--top-module", "nn2rtl_top", "--Mdir", toForwardSlash(buildDir),
      "-CFLAGS", "-O2 -std=c++17 -DNDEBUG",
      "--x-initial", "0",   // force uninitialized state to 0 (FPGA power-on), hardware-faithful
      "-DNN2RTL_ENGINE_SUBBLOCKS_PROVIDED",  // suppress the skeleton's empty stub submodules; use the REAL engine/*.v
      "-Wno-fatal",         // mbv2 modules carry benign lint; do not abort the build
      "-Wno-WIDTH", "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC", "-Wno-UNUSED", "-Wno-UNOPTFLAT", "-Wno-CASEINCOMPLETE",
      "-Wno-CASEX", "-Wno-COMBDLY", "-Wno-INITIALDLY", "-Wno-IMPLICIT", "-Wno-STMTDLY", "-Wno-MULTIDRIVEN",
      "-Wno-DECLFILENAME", "-Wno-EOFNEWLINE", "-Wno-PINMISSING", "-Wno-WIDTHCONCAT", "--replication-limit", "20000",
      // [DEBUG] NN2RTL_PUBLIC=1 exposes ALL internal nets (engine conv data_out, relus, act-mem)
      // for the per-node localization probe. Slower sim; do NOT use for the final/fit build.
      ...(process.env.NN2RTL_PUBLIC ? ["--public-flat-rw"] : []),
      // [MT-DEBUG 2026-06-08] extra verilator args (e.g. --no-threads-coarsen) for the #5
      // multithread-determinism investigation. Space-separated. No-op when unset.
      ...(process.env.NN2RTL_VERILATOR_EXTRA ? process.env.NN2RTL_VERILATOR_EXTRA.trim().split(/\s+/) : []),
      ...sources.map(toForwardSlash), toForwardSlash(tbCpp),
    ];
    console.log(`[verilate] starting verilator (${sources.length} RTL sources, REAL engine, ~5-15 min)`);
    const verilateRes = await runCmd(verilatorBin, verilatorArgs, repoRoot, env, "verilate");
    await writeFile(buildLog, "--- verilator ---\n" + verilateRes.stdout + "\n" + verilateRes.stderr, "utf8");
    if (!verilateRes.ok) { console.error("[verilate] FAILED — see " + buildLog); process.exit(2); }
    console.log("[make] compiling generated C++ (serialized -j to avoid Windows PCH race; 10-30 min)");
    // Serialize -j (=1) to avoid the Windows precompiled-header race some
    // builds hit when multiple cc invocations touch the same PCH.
    const makeRes = await runCmd(makeBin, ["-j", "1", "-f", "Vnn2rtl_top.mk", "Vnn2rtl_top"], buildDir, env, "make");
    await writeFile(buildLog, await readFile(buildLog, "utf8") + "\n--- make ---\n" + makeRes.stdout + "\n" + makeRes.stderr, "utf8");
    if (!makeRes.ok) { console.error("[make] FAILED — see " + buildLog); process.exit(3); }
  } else {
    console.log("[runonly] skipping build; reusing existing exe");
  }

  const exePath = path.join(buildDir, process.platform === "win32" ? "Vnn2rtl_top.exe" : "Vnn2rtl_top");
  if (!existsSync(exePath)) { console.error(`[run] expected exe not found: ${exePath}`); process.exit(4); }

  // Per task: stop after a successful BUILD unless explicitly told to run.
  // Set NN2RTL_ENGINE_VALUE_RUN=1 to proceed into the full sim.
  if (process.env.NN2RTL_ENGINE_VALUE_RUN !== "1") {
    console.log(`[build-only] exe built: ${exePath}`);
    console.log(`[build-only] set NN2RTL_ENGINE_VALUE_RUN=1 to run the full sim`);
    process.exit(0);
  }

  // CWD = repoRoot so the engine top's RELATIVE $readmemh weight paths
  // (output/mobilenet-v2/weights/*.mem) resolve.
  console.log(`[run] launching ${exePath} vec=${vectorIdx}`);
  const t0 = Date.now();
  const dumpPath = process.env.NN2RTL_DUMP_PATH ?? "";  // argv[4]: raw m_axis capture dump
  const runArgs = dumpPath
    ? [goldinPath, goldoutPath, vectorIdx, dumpPath]
    : [goldinPath, goldoutPath, vectorIdx];
  const runRes = await runCmd(exePath, runArgs, repoRoot, env, "run");
  const elapsedS = (Date.now() - t0) / 1000;
  await writeFile(runLog, runRes.stdout + "\n" + runRes.stderr, "utf8");

  const summaryLine = runRes.stdout.split(/\r?\n/).find((l) => l.includes("[tb][mbv2][summary]"));
  if (!summaryLine) { console.error("[run] no [tb][mbv2][summary] line — sim crashed/timed out"); process.exit(6); }
  const m = (re: RegExp) => { const x = summaryLine.match(re); return x ? x[1] : null; };
  const result = {
    result: m(/result=(\w+)/),
    mismatch_bytes: Number(m(/mismatch_bytes=(-?\d+)/) ?? -1),
    first_mismatch_beat: Number(m(/first_mismatch_beat=(-?\d+)/) ?? -1),
    beats_seen: Number(m(/beats_seen=(\d+)\//) ?? -1),
    beats_expected: Number(m(/beats_seen=\d+\/(\d+)/) ?? -1),
    in_beats_seen: Number(m(/in_beats=(\d+)\//) ?? -1),
    in_beats_expected: Number(m(/in_beats=\d+\/(\d+)/) ?? -1),
    vector_idx: Number(vectorIdx),
    sim_elapsed_s: elapsedS,
    goldin: path.relative(repoRoot, goldinPath),
    goldout: path.relative(repoRoot, goldoutPath),
  };
  await writeFile(resultJson, JSON.stringify(result, null, 2), "utf8");
  console.log(`[result] ${JSON.stringify(result)}`);
  console.log(`[result] wrote ${path.relative(repoRoot, resultJson)}`);
  process.exit(result.result === "PASS" ? 0 : 1);
}

main().catch((err: unknown) => {
  console.error(err instanceof Error ? err.stack ?? err.message : String(err));
  process.exit(1);
});
