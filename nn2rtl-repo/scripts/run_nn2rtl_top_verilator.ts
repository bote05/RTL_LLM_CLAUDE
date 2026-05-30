// Build + run the Verilator cycle-count testbench for nn2rtl_top.v.
//
// Goal: measure end-to-end cycles per inference. Pairs with post-route Fmax
// to give the actual throughput number for the thesis PPA table.
//
// Usage:
//   npx tsx scripts/run_nn2rtl_top_verilator.ts
//
// Output:
//   output/reports_integrated/verilator_nn2rtl_top/
//     build.log           — verilator + g++ build output
//     run.log             — simulation stdout
//     cycles.json         — { e2e_cycles, input_beats, output_beats, fps_at_*_mhz }

import { readdir, readFile, writeFile, mkdir, rm } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";
import { execFile, spawn } from "node:child_process";
import { promisify } from "node:util";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

const tbCpp     = path.join(repoRoot, "tb", "nn2rtl_top_cycle_count_tb.cpp");
const outDir    = path.join(repoRoot, "output", "reports_integrated", "verilator_nn2rtl_top");
const buildLog  = path.join(outDir, "build.log");
const runLog    = path.join(outDir, "run.log");
const cyclesJson = path.join(outDir, "cycles.json");

const verilatorBin = process.platform === "win32"
  ? "C:/Users/User/oss-cad-suite/bin/verilator_bin.exe"
  : "verilator";
const makeBin = process.platform === "win32"
  ? "C:/Users/User/w64devkit/bin/make.exe"
  : "make";
const gxxBin = process.platform === "win32"
  ? "C:/Users/User/w64devkit/bin/g++.exe"
  : "g++";

const execFileP = promisify(execFile);

function toForwardSlash(p: string): string {
  return p.replace(/\\/g, "/");
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
  for (const p of out) {
    if (!existsSync(p)) throw new Error(`source missing: ${p}`);
  }
  return out;
}

async function runCmd(cmd: string, args: string[], cwd: string, env: NodeJS.ProcessEnv, label: string): Promise<{ stdout: string; stderr: string; ok: boolean }> {
  console.log(`[${label}] ${cmd} ${args.slice(0, 6).join(" ")}${args.length > 6 ? " ... +" + (args.length - 6) + " args" : ""}`);
  return new Promise((resolve) => {
    const child = spawn(cmd, args, { cwd, env, stdio: ["ignore", "pipe", "pipe"], shell: false });
    let stdout = ""; let stderr = "";
    child.stdout.on("data", (chunk) => {
      const s = chunk.toString();
      stdout += s;
      process.stdout.write(s);
    });
    child.stderr.on("data", (chunk) => {
      const s = chunk.toString();
      stderr += s;
      process.stderr.write(s);
    });
    child.on("close", (code) => {
      resolve({ stdout, stderr, ok: code === 0 });
    });
    child.on("error", (err) => {
      resolve({ stdout, stderr: stderr + "\n" + String(err), ok: false });
    });
  });
}

async function main(): Promise<void> {
  await mkdir(outDir, { recursive: true });
  if (existsSync(buildLog)) await rm(buildLog);
  if (existsSync(runLog)) await rm(runLog);

  const buildDir = path.join(outDir, "obj_dir");
  // Windows + Verilator: stale .exe sometimes holds a handle to obj_dir even
  // after the process is killed. Retry the cleanup a few times, then fall
  // back to overwriting in place.
  if (existsSync(buildDir)) {
    console.log(`[setup] clearing previous build dir ${buildDir}`);
    for (let attempt = 0; attempt < 5; attempt++) {
      try {
        await rm(buildDir, { recursive: true, force: true });
        break;
      } catch (e: any) {
        if (attempt === 4) {
          console.log(`[setup] rm failed after retries (${e.code}); proceeding with incremental rebuild`);
        } else {
          console.log(`[setup] rm attempt ${attempt + 1} failed (${e.code}); retrying...`);
          await new Promise(r => setTimeout(r, 2000));
        }
      }
    }
  }

  const sources = await collectSources();
  console.log(`[setup] collected ${sources.length} RTL files`);
  console.log(`[setup] tb cpp: ${tbCpp}`);
  console.log(`[setup] build dir: ${buildDir}`);

  // Verilator args.
  // -Wno-* flags suppress non-fatal warnings from the generated wrapper that
  // pollute the log on a 130-file design; they don't affect correctness.
  const traceMode = process.env.NN2RTL_TRACE_FST === "1";
  // VCD instead of FST: oss-cad-suite Windows ships FST without zlib.h.
  // VCD is bigger on disk but no external deps; for 5M cycles this is fine.
  const traceArgs = traceMode ? ["--trace", "--trace-structs"] : [];
  const traceCflags = traceMode ? "-DNN2RTL_TRACE_FST=1" : "";
  const verilatorArgs = [
    "--cc", "--exe", "-O3",
    "--threads", "4",
    "--top-module", "nn2rtl_top",
    "--Mdir", toForwardSlash(buildDir),
    ...traceArgs,
    "-CFLAGS", `-O2 -std=c++17 -DNDEBUG ${traceCflags}`,
    "-Wno-WIDTH", "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC",
    "-Wno-UNUSED", "-Wno-UNOPTFLAT", "-Wno-CASEINCOMPLETE",
    "-Wno-CASEX", "-Wno-COMBDLY", "-Wno-INITIALDLY",
    "-Wno-IMPLICIT", "-Wno-STMTDLY", "-Wno-MULTIDRIVEN",
    "-Wno-DECLFILENAME", "-Wno-EOFNEWLINE",
    "-Wno-PINMISSING", "-Wno-WIDTHCONCAT",
    // conv_288 has OC=2048 -> conv_datapath_mp_k zero-fills a 16384-bit data_out
    // reg ({OC*8{1'b0}}), above Verilator's default 8192 replication limit.
    "--replication-limit", "20000",
    ...sources.map(toForwardSlash),
    toForwardSlash(tbCpp),
  ];

  console.log(`[verilate] starting verilator (${sources.length} RTL sources, ~5-15 min for a design this size)`);
  const env = { ...process.env, PATH: `C:/Users/User/oss-cad-suite/bin;C:/Users/User/w64devkit/bin;${process.env.PATH ?? ""}` };
  const verilateRes = await runCmd(verilatorBin, verilatorArgs, repoRoot, env, "verilate");
  await writeFile(buildLog, "--- verilator ---\n" + verilateRes.stdout + "\n" + verilateRes.stderr, "utf8");
  if (!verilateRes.ok) {
    console.error("[verilate] FAILED — see " + buildLog);
    process.exit(2);
  }

  // After verilator, build the C++ via the generated Makefile.
  console.log("[make] compiling generated C++ (10-30 min for full design)");
  const makeRes = await runCmd(makeBin, ["-j", "16", "-f", "Vnn2rtl_top.mk", "Vnn2rtl_top"], buildDir, env, "make");
  await writeFile(buildLog, await readFile(buildLog, "utf8") + "\n--- make ---\n" + makeRes.stdout + "\n" + makeRes.stderr, "utf8");
  if (!makeRes.ok) {
    console.error("[make] FAILED — see " + buildLog);
    process.exit(3);
  }

  // Run the resulting binary.
  const exePath = path.join(buildDir, process.platform === "win32" ? "Vnn2rtl_top.exe" : "Vnn2rtl_top");
  if (!existsSync(exePath)) {
    console.error(`[run] expected exe not found: ${exePath}`);
    process.exit(4);
  }
  console.log(`[run] launching ${exePath} (sim itself ~3-30 min depending on cycle count + design complexity)`);
  const t0 = Date.now();
  const runRes = await runCmd(exePath, [], buildDir, env, "run");
  const elapsedS = (Date.now() - t0) / 1000;
  await writeFile(runLog, runRes.stdout + "\n" + runRes.stderr, "utf8");
  if (!runRes.ok) {
    console.error(`[run] FAILED in ${elapsedS.toFixed(1)}s — see ${runLog}`);
    process.exit(5);
  }

  // Parse cycle count from stdout.
  const summaryLine = runRes.stdout.split(/\r?\n/).find((l) => l.includes("[tb][summary]"));
  if (!summaryLine) {
    console.error("[run] no [tb][summary] line in output — sim likely timed out");
    process.exit(6);
  }
  const cyclesMatch = summaryLine.match(/e2e_cycles=(\d+)/);
  const inputMatch  = summaryLine.match(/input_beats=(\d+)/);
  const outputMatch = summaryLine.match(/output_beats=(\d+)/);
  const e2eCycles   = cyclesMatch  ? Number(cyclesMatch[1])  : -1;
  const inputBeats  = inputMatch   ? Number(inputMatch[1])   : -1;
  const outputBeats = outputMatch  ? Number(outputMatch[1])  : -1;

  const fps = (mhz: number) => e2eCycles > 0 ? (mhz * 1e6) / e2eCycles : null;
  const result = {
    e2e_cycles: e2eCycles,
    input_beats: inputBeats,
    output_beats: outputBeats,
    sim_elapsed_s: elapsedS,
    fps_at_25_mhz:  fps(25),
    fps_at_33_mhz:  fps(33),
    fps_at_50_mhz:  fps(50),
    fps_at_100_mhz: fps(100),
    fps_at_150_mhz: fps(150),
    fps_at_200_mhz: fps(200),
  };
  await writeFile(cyclesJson, JSON.stringify(result, null, 2), "utf8");

  console.log(`[result] e2e_cycles=${e2eCycles} input_beats=${inputBeats} output_beats=${outputBeats}`);
  console.log(`[result] fps @25MHz=${result.fps_at_25_mhz?.toFixed(2)} @33MHz=${result.fps_at_33_mhz?.toFixed(2)} @50MHz=${result.fps_at_50_mhz?.toFixed(2)} @100MHz=${result.fps_at_100_mhz?.toFixed(2)} @150MHz=${result.fps_at_150_mhz?.toFixed(2)} @200MHz=${result.fps_at_200_mhz?.toFixed(2)}`);
  console.log(`[result] wrote ${path.relative(repoRoot, cyclesJson)}`);
}

main().catch((err: unknown) => {
  console.error(err instanceof Error ? err.stack ?? err.message : String(err));
  process.exit(1);
});
