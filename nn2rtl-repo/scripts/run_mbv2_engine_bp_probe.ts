// Build + run the STEM-SEGMENT backpressure PROBE for the MobileNetV2 engine
// top. Identical RTL source set + verilator flags to
// scripts/run_mbv2_top_engine_value.ts (so the build faithfully matches the
// e2e build), but swaps the TB for tb/mbv2_engine_bp_probe_tb.cpp and adds the
// public-net config tb/mbv2_engine_bp_probe.vlt. Private obj dir so it does not
// collide with the value-run build.
//
// Usage: npx tsx scripts/run_mbv2_engine_bp_probe.ts [vec]
//   env MBV2_MAX_CYCLES, NN2RTL_VALUE_RUNONLY=1 (reuse exe)

import { readdir, writeFile, mkdir, rm, readFile } from "node:fs/promises";
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
const tbCpp     = path.join(repoRoot, "tb", "mbv2_engine_bp_probe_tb.cpp");
const vltCfg    = path.join(repoRoot, "tb", "mbv2_engine_bp_probe.vlt");
const outDir    = path.join(mbv2Root, "reports", "verilator_mbv2_engine_bp_probe");
const buildLog  = path.join(outDir, "build.log");
const runLog    = path.join(outDir, "run.log");

const sharedEngineSkeleton = path.join(repoRoot, "output", "rtl", "shared_engine_skeleton.v");
const engineSubmodules = [
  path.join(repoRoot, "output", "rtl", "engine", "mac_array.v"),
  path.join(repoRoot, "output", "rtl", "engine", "address_generator.v"),
  path.join(repoRoot, "output", "rtl", "engine", "config_register_block.v"),
  path.join(repoRoot, "output", "rtl", "engine", "requant_pipeline.v"),
  path.join(repoRoot, "output", "rtl", "engine", "bram_to_stream_bridge.v"),
];
const goldinPath = path.join(goldenDir, "node_conv_810.goldin");
const vectorIdx = process.argv[2] ?? "0";

const verilatorBin = process.platform === "win32"
  ? "C:/Users/User/oss-cad-suite/bin/verilator_bin.exe" : "verilator";
const makeBin = process.platform === "win32"
  ? "C:/Users/User/w64devkit/bin/make.exe" : "make";
function fs2(p: string): string { return p.replace(/\\/g, "/"); }

function isExcludedRtl(entry: string): boolean {
  if (entry === "nn2rtl_top.v") return true;
  if (entry.endsWith(".pre_skipwire.v")) return true;
  if (entry.endsWith(".preimprove")) return true;
  if (entry.includes(".bak")) return true;
  return false;
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
  const entries = await readdir(rtlDir);
  for (const entry of entries) {
    if (entry.endsWith(".v") && !isExcludedRtl(entry)) out.push(path.join(rtlDir, entry));
  }
  const tops = out.filter((p) => path.basename(p) === "nn2rtl_top_engine.v" || path.basename(p) === "nn2rtl_top.v");
  if (tops.length !== 1 || path.basename(tops[0]) !== "nn2rtl_top_engine.v")
    throw new Error(`expected exactly the engine top, got: ${tops.map((p)=>path.basename(p)).join(", ")}`);
  for (const p of out) if (!existsSync(p)) throw new Error(`source missing: ${p}`);
  return out;
}
function runCmd(cmd: string, args: string[], cwd: string, env: NodeJS.ProcessEnv, label: string): Promise<{ ok: boolean; stdout: string; stderr: string }> {
  console.log(`[${label}] ${cmd} ${args.slice(0,6).join(" ")}${args.length>6?" ... +"+(args.length-6)+" args":""}`);
  return new Promise((resolve) => {
    const child = spawn(cmd, args, { cwd, env, stdio:["ignore","pipe","pipe"], shell:false });
    let stdout="", stderr="";
    child.stdout.on("data",(c)=>{const s=c.toString();stdout+=s;process.stdout.write(s);});
    child.stderr.on("data",(c)=>{const s=c.toString();stderr+=s;process.stderr.write(s);});
    child.on("close",(code)=>resolve({ok:code===0,stdout,stderr}));
    child.on("error",(err)=>resolve({ok:false,stdout,stderr:stderr+"\n"+String(err)}));
  });
}

async function main(): Promise<void> {
  await mkdir(outDir,{recursive:true});
  if (!existsSync(goldinPath)) throw new Error(`goldin missing: ${goldinPath}`);
  if (!existsSync(tbCpp))      throw new Error(`tb missing: ${tbCpp}`);
  if (!existsSync(vltCfg))     throw new Error(`vlt missing: ${vltCfg}`);
  const buildDir = path.join(outDir, "obj_dir_bp_probe");
  const env = { ...process.env, PATH: `C:/Users/User/oss-cad-suite/bin;C:/Users/User/w64devkit/bin;${process.env.PATH ?? ""}` };

  if (process.env.NN2RTL_VALUE_RUNONLY !== "1") {
    if (existsSync(buildDir)) { try { await rm(buildDir,{recursive:true,force:true}); } catch {} }
    await mkdir(buildDir,{recursive:true});
    const sources = await collectSources();
    console.log(`[setup] collected ${sources.length} RTL files; tb=${path.relative(repoRoot,tbCpp)} vlt=${path.relative(repoRoot,vltCfg)}`);
    const verilatorArgs = [
      "--cc","--exe","-O3","--threads","4","--top-module","nn2rtl_top","--Mdir",fs2(buildDir),
      "-CFLAGS","-O2 -std=c++17 -DNDEBUG",
      "--x-initial","0",
      "-DNN2RTL_ENGINE_SUBBLOCKS_PROVIDED",
      "-Wno-fatal",
      "-Wno-WIDTH","-Wno-WIDTHEXPAND","-Wno-WIDTHTRUNC","-Wno-UNUSED","-Wno-UNOPTFLAT","-Wno-CASEINCOMPLETE",
      "-Wno-CASEX","-Wno-COMBDLY","-Wno-INITIALDLY","-Wno-IMPLICIT","-Wno-STMTDLY","-Wno-MULTIDRIVEN",
      "-Wno-DECLFILENAME","-Wno-EOFNEWLINE","-Wno-PINMISSING","-Wno-WIDTHCONCAT","--replication-limit","20000",
      fs2(vltCfg),
      ...sources.map(fs2), fs2(tbCpp),
    ];
    console.log(`[verilate] starting (${sources.length} RTL sources + probe vlt, ~5-15 min)`);
    const vr = await runCmd(verilatorBin, verilatorArgs, repoRoot, env, "verilate");
    await writeFile(buildLog, "--- verilator ---\n"+vr.stdout+"\n"+vr.stderr, "utf8");
    if (!vr.ok) { console.error("[verilate] FAILED — see "+buildLog); process.exit(2); }
    console.log("[make] compiling (serialized)");
    const mr = await runCmd(makeBin, ["-j","1","-f","Vnn2rtl_top.mk","Vnn2rtl_top"], buildDir, env, "make");
    await writeFile(buildLog, (await readFile(buildLog,"utf8"))+"\n--- make ---\n"+mr.stdout+"\n"+mr.stderr, "utf8");
    if (!mr.ok) { console.error("[make] FAILED — see "+buildLog); process.exit(3); }
  } else { console.log("[runonly] reusing existing exe"); }

  const exePath = path.join(buildDir, process.platform==="win32"?"Vnn2rtl_top.exe":"Vnn2rtl_top");
  if (!existsSync(exePath)) { console.error(`[run] exe not found: ${exePath}`); process.exit(4); }
  console.log(`[run] launching ${exePath} vec=${vectorIdx}`);
  const rr = await runCmd(exePath, [goldinPath, vectorIdx], repoRoot, env, "run");
  await writeFile(runLog, rr.stdout+"\n"+rr.stderr, "utf8");
  console.log(`[run] wrote ${path.relative(repoRoot, runLog)}`);
  process.exit(0);
}
main().catch((e)=>{ console.error(e instanceof Error ? e.stack ?? e.message : String(e)); process.exit(1); });
