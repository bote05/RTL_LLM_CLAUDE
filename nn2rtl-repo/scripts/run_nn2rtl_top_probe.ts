// Build + run the CHAIN-PROBE testbench (localizes the e2e all-zero bug).
// Clone of run_nn2rtl_top_value.ts: adds tb/nn2rtl_top_probe.vlt (public vars),
// points at tb/nn2rtl_top_probe_tb.cpp, dumps probe_<id>.bin per checkpoint.
//
// Env NN2RTL_VERILATE_ONLY=1 stops after verilate (to confirm rootp accessors).
// Usage: npx tsx scripts/run_nn2rtl_top_probe.ts [vector_idx]

import { readdir, readFile, writeFile, mkdir, rm } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";
import { spawn } from "node:child_process";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

const tbCpp  = path.join(repoRoot, "tb", "nn2rtl_top_probe_tb.cpp");
const vltCfg = path.join(repoRoot, "tb", "nn2rtl_top_probe.vlt");
const outDir = path.join(repoRoot, "output", "reports_integrated", "verilator_nn2rtl_top_probe");
const buildLog = path.join(outDir, "build.log");
const runLog = path.join(outDir, "run.log");
const vectorIdx = process.argv[2] ?? "0";
const verilateOnly = process.env.NN2RTL_VERILATE_ONLY === "1";

const verilatorBin = process.platform === "win32" ? "C:/Users/User/oss-cad-suite/bin/verilator_bin.exe" : "verilator";
const makeBin = process.platform === "win32" ? "C:/Users/User/w64devkit/bin/make.exe" : "make";
function fs2(p: string){ return p.replace(/\\/g,"/"); }

async function findContractGolden(prefix: string, file: string): Promise<string> {
  const cdir = path.join(repoRoot,"output","goldens","contracts");
  const m = (await readdir(cdir)).find(e=>e.startsWith(prefix));
  if(!m) throw new Error(`no contract dir '${prefix}'`);
  const p = path.join(cdir,m,file);
  if(!existsSync(p)) throw new Error(`golden missing ${p}`);
  return p;
}
async function collectSources(): Promise<string[]> {
  const out=[
    "output/rtl/nn2rtl_top.v","output/rtl/nn2rtl_scheduler.v","output/rtl/shared_engine_skeleton.v",
    "output/rtl/engine/address_generator.v","output/rtl/engine/config_register_block.v","output/rtl/engine/mac_array.v",
    "output/rtl/engine/requant_pipeline.v","output/rtl/engine/bram_to_stream_bridge.v",
    "rtl_library/conv_datapath.v","rtl_library/conv_datapath_parallel.v","rtl_library/conv_datapath_mp_k.v",
    "rtl_library/coord_scheduler.v","rtl_library/line_buf_window.v",
  ].map(p=>path.join(repoRoot,p));
  const rtlDir=path.join(repoRoot,"output","rtl");
  for(const e of await readdir(rtlDir)) if(e.startsWith("node_")&&e.endsWith(".v")&&!e.endsWith(".preimprove")) out.push(path.join(rtlDir,e));
  for(const p of out) if(!existsSync(p)) throw new Error(`source missing ${p}`);
  return out;
}
function runCmd(cmd:string,args:string[],cwd:string,env:NodeJS.ProcessEnv,label:string):Promise<{stdout:string;stderr:string;ok:boolean}>{
  console.log(`[${label}] ${cmd} ${args.slice(0,5).join(" ")}${args.length>5?" ...+"+(args.length-5):""}`);
  return new Promise(res=>{ const c=spawn(cmd,args,{cwd,env,stdio:["ignore","pipe","pipe"],shell:false}); let so="",se="";
    c.stdout.on("data",d=>{const s=d.toString();so+=s;process.stdout.write(s);});
    c.stderr.on("data",d=>{const s=d.toString();se+=s;process.stderr.write(s);});
    c.on("close",code=>res({stdout:so,stderr:se,ok:code===0})); c.on("error",e=>res({stdout:so,stderr:se+String(e),ok:false})); });
}

async function main(){
  await mkdir(outDir,{recursive:true});
  const goldin = await findContractGolden("node_conv_196_","node_conv_196.goldin");
  const buildDir = path.join(outDir,"obj_dir_probe");
  if(existsSync(buildDir) && !verilateOnly){
    for(let a=0;a<5;a++){ try{ await rm(buildDir,{recursive:true,force:true}); break; }catch(e:any){ if(a===4) console.log("[setup] rm failed, proceeding"); else await new Promise(r=>setTimeout(r,2000)); } }
  }
  const sources = await collectSources();
  console.log(`[setup] ${sources.length} RTL files; tb=${path.basename(tbCpp)} vlt=${path.basename(vltCfg)} verilateOnly=${verilateOnly}`);

  const env = { ...process.env, PATH:`C:/Users/User/oss-cad-suite/bin;C:/Users/User/w64devkit/bin;${process.env.PATH??""}` };
  const verilatorArgs = [
    "--cc","--exe","-O3","--threads","4","--top-module","nn2rtl_top","--x-initial","0","--Mdir",fs2(buildDir),
    "-CFLAGS","-O2 -std=c++17 -DNDEBUG",
    "-Wno-WIDTH","-Wno-WIDTHEXPAND","-Wno-WIDTHTRUNC","-Wno-UNUSED","-Wno-UNOPTFLAT","-Wno-CASEINCOMPLETE",
    "-Wno-CASEX","-Wno-COMBDLY","-Wno-INITIALDLY","-Wno-IMPLICIT","-Wno-STMTDLY","-Wno-MULTIDRIVEN",
    "-Wno-DECLFILENAME","-Wno-EOFNEWLINE","-Wno-PINMISSING","-Wno-WIDTHCONCAT","--replication-limit","20000",
    fs2(vltCfg), ...sources.map(fs2), fs2(tbCpp),
  ];
  const vr = await runCmd(verilatorBin,verilatorArgs,repoRoot,env,"verilate");
  await writeFile(buildLog,"--- verilator ---\n"+vr.stdout+"\n"+vr.stderr,"utf8");
  if(!vr.ok){ console.error("[verilate] FAILED — see "+buildLog); process.exit(2); }
  if(verilateOnly){ console.log("[verilate-only] done; headers in "+buildDir); return; }

  const mr = await runCmd(makeBin,["-j","16","-f","Vnn2rtl_top.mk","Vnn2rtl_top"],buildDir,env,"make");
  await writeFile(buildLog, await readFile(buildLog,"utf8")+"\n--- make ---\n"+mr.stdout+"\n"+mr.stderr,"utf8");
  if(!mr.ok){ console.error("[make] FAILED — see "+buildLog); process.exit(3); }

  const exe=path.join(buildDir, process.platform==="win32"?"Vnn2rtl_top.exe":"Vnn2rtl_top");
  if(!existsSync(exe)){ console.error("no exe "+exe); process.exit(4); }
  console.log(`[run] ${exe} vec=${vectorIdx} (cwd=repoRoot for $readmemh weight paths)`);
  // CWD MUST be repoRoot so the RTL's $readmemh("output/weights/*.mem") resolves.
  const rr = await runCmd(exe,[goldin,fs2(outDir),vectorIdx],repoRoot,env,"run");
  await writeFile(runLog,rr.stdout+"\n"+rr.stderr,"utf8");
  console.log(rr.ok?"[run] done":"[run] nonzero exit (frame may be incomplete) — check run.log");
}
main().catch(e=>{ console.error(e instanceof Error?e.stack:String(e)); process.exit(1); });
