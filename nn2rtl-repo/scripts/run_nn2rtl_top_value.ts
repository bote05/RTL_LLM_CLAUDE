// Build + run the Verilator END-TO-END VALUE testbench for nn2rtl_top.v.
//
// Proves the assembled network is byte-exact correct (not just correct beat
// counts). Feeds conv_196 contract .goldin (vec0) into s_axis, captures m_axis,
// compares to relu_48 contract .goldout (vec0). See tb/nn2rtl_top_value_tb.cpp.
//
// Usage:
//   npx tsx scripts/run_nn2rtl_top_value.ts [vector_idx]
//
// Output:
//   output/reports_integrated/verilator_nn2rtl_top_value/
//     build.log  run.log  result.json

import { readdir, readFile, writeFile, mkdir, rm } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";
import { spawn } from "node:child_process";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

const tbCpp     = path.join(repoRoot, "tb", "nn2rtl_top_value_tb.cpp");
const outDir    = path.join(repoRoot, "output", "reports_integrated", "verilator_nn2rtl_top_value");
const buildLog  = path.join(outDir, "build.log");
const runLog    = path.join(outDir, "run.log");
const resultJson = path.join(outDir, "result.json");

const vectorIdx = process.argv[2] ?? "0";

const verilatorBin = process.platform === "win32"
  ? "C:/Users/User/oss-cad-suite/bin/verilator_bin.exe" : "verilator";
const makeBin = process.platform === "win32"
  ? "C:/Users/User/w64devkit/bin/make.exe" : "make";

function toForwardSlash(p: string): string { return p.replace(/\\/g, "/"); }

// Resolve the contract golden paths by globbing the (long) contract dir names.
async function findContractGolden(prefix: string, file: string): Promise<string> {
  const contractsDir = path.join(repoRoot, "output", "goldens", "contracts");
  const entries = await readdir(contractsDir);
  const match = entries.find((e) => e.startsWith(prefix));
  if (!match) throw new Error(`no contract dir starting with '${prefix}' under ${contractsDir}`);
  const p = path.join(contractsDir, match, file);
  if (!existsSync(p)) throw new Error(`golden missing: ${p}`);
  return p;
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
    if (entry.startsWith("node_") && entry.endsWith(".v") && !entry.endsWith(".preimprove")) {
      out.push(path.join(rtlDir, entry));
    }
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

  const goldin  = await findContractGolden("node_conv_196_", "node_conv_196.goldin");
  // NN2RTL_GOLDOUT_PATH overrides the compared output golden (truncated bisect:
  // point at an intermediate layer's bps=32 contract goldout).
  const goldout = process.env.NN2RTL_GOLDOUT_PATH
    ?? await findContractGolden("node_relu_48_",  "node_relu_48.goldout");
  console.log(`[setup] goldin : ${path.relative(repoRoot, goldin)}`);
  console.log(`[setup] goldout: ${path.relative(repoRoot, goldout)}`);

  const buildDir = path.join(outDir, "obj_dir_value");
  const env = { ...process.env, PATH: `C:/Users/User/oss-cad-suite/bin;C:/Users/User/w64devkit/bin;${process.env.PATH ?? ""}` };
  // NN2RTL_VALUE_RUNONLY=1 reuses the existing exe (no rebuild) — for re-running
  // after a sim-harness-only change (e.g. cwd fix for $readmemh weight loading).
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
    const sources = await collectSources();
    console.log(`[setup] collected ${sources.length} RTL files; tb=${path.relative(repoRoot, tbCpp)}`);
    const verilatorArgs = [
      // [MT-DETERMINISM 2026-06-09] default --threads 1. Per MBV2 #5: Verilator's MULTITHREADED
      // scheduler produces WRONG VALUES on these designs (a cross-partition read/write the MT engine
      // orders wrong) -- a SIM-ONLY hazard, NOT an RTL bug (--threads 1 is byte-exact; hardware/Vivado
      // is correct). The old hardcoded --threads 4 was the prime suspect for the ResNet relu_48 "2.7%".
      "--cc", "--exe", "-O3", "--threads", String(process.env.NN2RTL_VALUE_THREADS ?? "1"), "--top-module", "nn2rtl_top", "--Mdir", toForwardSlash(buildDir),
      "-CFLAGS", "-O2 -std=c++17 -DNDEBUG",
      "--x-initial", String(process.env.NN2RTL_VALUE_XINIT ?? "0"),   // [BANK-INIT FIX] default 0 (FPGA power-on, HW-faithful); set NN2RTL_VALUE_XINIT=unique to probe uninitialized-read sensitivity

      "-Wno-WIDTH", "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC", "-Wno-UNUSED", "-Wno-UNOPTFLAT", "-Wno-CASEINCOMPLETE",
      "-Wno-CASEX", "-Wno-COMBDLY", "-Wno-INITIALDLY", "-Wno-IMPLICIT", "-Wno-STMTDLY", "-Wno-MULTIDRIVEN",
      "-Wno-DECLFILENAME", "-Wno-EOFNEWLINE", "-Wno-PINMISSING", "-Wno-WIDTHCONCAT", "--replication-limit", "20000",
      ...sources.map(toForwardSlash), toForwardSlash(tbCpp),
    ];
    console.log(`[verilate] starting verilator (${sources.length} RTL sources, ~5-15 min)`);
    const verilateRes = await runCmd(verilatorBin, verilatorArgs, repoRoot, env, "verilate");
    await writeFile(buildLog, "--- verilator ---\n" + verilateRes.stdout + "\n" + verilateRes.stderr, "utf8");
    if (!verilateRes.ok) { console.error("[verilate] FAILED — see " + buildLog); process.exit(2); }
    console.log("[make] compiling generated C++ (10-30 min)");
    const makeRes = await runCmd(makeBin, ["-j", "16", "-f", "Vnn2rtl_top.mk", "Vnn2rtl_top"], buildDir, env, "make");
    await writeFile(buildLog, await readFile(buildLog, "utf8") + "\n--- make ---\n" + makeRes.stdout + "\n" + makeRes.stderr, "utf8");
    if (!makeRes.ok) { console.error("[make] FAILED — see " + buildLog); process.exit(3); }
  } else {
    console.log("[runonly] skipping build; reusing existing exe");
  }

  const exePath = path.join(buildDir, process.platform === "win32" ? "Vnn2rtl_top.exe" : "Vnn2rtl_top");
  if (!existsSync(exePath)) { console.error(`[run] expected exe not found: ${exePath}`); process.exit(4); }

  console.log(`[run] launching ${exePath} vec=${vectorIdx} (cwd=repoRoot so $readmemh "output/weights/*.mem" resolves)`);
  const t0 = Date.now();
  // CWD MUST be repoRoot: the RTL preloads engine weights/bias via
  // $readmemh("output/weights/*.mem") (relative paths). Running from the build
  // dir leaves those memories zero -> engine outputs zero -> all-zero frame.
  const dumpPath = process.env.NN2RTL_DUMP_PATH ?? "";  // argv[4]: raw m_axis capture dump
  const runArgs = dumpPath ? [goldin, goldout, vectorIdx, dumpPath] : [goldin, goldout, vectorIdx];
  const runRes = await runCmd(exePath, runArgs, repoRoot, env, "run");
  const elapsedS = (Date.now() - t0) / 1000;
  await writeFile(runLog, runRes.stdout + "\n" + runRes.stderr, "utf8");

  const summaryLine = runRes.stdout.split(/\r?\n/).find((l) => l.includes("[tb][value][summary]"));
  if (!summaryLine) { console.error("[run] no [tb][value][summary] line — sim crashed/timed out"); process.exit(6); }
  const m = (re: RegExp) => { const x = summaryLine.match(re); return x ? x[1] : null; };
  const result = {
    result: m(/result=(\w+)/),
    beats_seen: Number(m(/beats=(\d+)\//) ?? -1),
    beats_expected: Number(m(/beats=\d+\/(\d+)/) ?? -1),
    mismatch_bytes: Number(m(/mismatch_bytes=(-?\d+)/) ?? -1),
    first_mismatch_beat: Number(m(/first_mismatch_beat=(-?\d+)/) ?? -1),
    vector_idx: Number(vectorIdx),
    sim_elapsed_s: elapsedS,
    goldin: path.relative(repoRoot, goldin),
    goldout: path.relative(repoRoot, goldout),
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
