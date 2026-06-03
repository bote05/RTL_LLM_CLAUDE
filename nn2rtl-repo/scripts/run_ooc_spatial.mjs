// OOC-measure the spatial conv weights_wide ROM BRAM tile cost at INT3 vs INT4.
// Reuses the proven cmd.exe /c vivado.bat -mode batch launch from run_first_light_synth.ts.
// Light per run (single ROM module, like the engine bank OOC: ~5 LUT / ~120 RAMB), serialized.
import { execFileSync } from "node:child_process";
import { writeFileSync, readFileSync, existsSync, statSync } from "node:fs";

const REPO = "C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo";
const OOC = `${REPO}/output/reports_integrated/ooc`;
const VIVADO = process.env.NN2RTL_VIVADO_BIN || "D:/vivado/2025.2/Vivado/bin/vivado.bat";
const MOD = `${OOC}/ooc_spatial_rom.v`;

// 284/292/298: 16384x432 INT3 (MP16*MPK9*3) / 576 INT4 ; 288: 16384x384 INT3 (MP16*MPK8*3) / 512 INT4. ADDR_W=14.
const CONFIGS = [
  { tag: "sp284_int3", depth: 16384, wide: 432, addr: 14, mem: `${REPO}/output/weights/node_conv_284_weights_mp_k_9.hex` },
  { tag: "sp288_int3", depth: 16384, wide: 384, addr: 14, mem: `${REPO}/output/weights/node_conv_288_weights_mp_k_8.hex` },
  { tag: "sp284_int4", depth: 16384, wide: 576, addr: 14, mem: `${REPO}/backups/allint4_byteexact/node_conv_284_weights_mp_k_9.hex` },
  { tag: "sp288_int4", depth: 16384, wide: 512, addr: 14, mem: `${REPO}/backups/allint4_byteexact/node_conv_288_weights_mp_k_8.hex` },
];

function ramb(rptPath) {
  if (!existsSync(rptPath)) return "NO-REPORT";
  const txt = readFileSync(rptPath, "utf8");
  const grab = (re) => { const m = txt.match(re); return m ? m[1].trim() : "?"; };
  const tile = grab(/Block RAM Tile\s*\|\s*([0-9.]+)/);
  const r36 = grab(/RAMB36\/FIFO\*?\s*\|\s*([0-9.]+)/);
  const r18 = grab(/RAMB18\s*\|\s*([0-9.]+)/);
  const uram = grab(/\|\s*URAM\s*\|\s*([0-9.]+)/);
  return `BlockRAMTile=${tile}  RAMB36=${r36}  RAMB18=${r18}  URAM=${uram}`;
}

const results = [];
for (const c of CONFIGS) {
  const memOk = existsSync(c.mem);
  const memInfo = memOk ? `${statSync(c.mem).size}B` : "MISSING";
  // sanity: first line hex char width should == wide/4
  let lineChars = "?";
  if (memOk) lineChars = (readFileSync(c.mem, "utf8").split(/\r?\n/)[0] || "").trim().length;
  console.log(`\n===== ${c.tag}: ${c.depth}x${c.wide}  mem=${memInfo} lineChars=${lineChars} (expect ${c.wide / 4}) =====`);
  if (!memOk) { results.push(`${c.tag}: MEM MISSING ${c.mem}`); continue; }
  const rpt = `${OOC}/${c.tag}.rpt`;
  const tcl = `${OOC}/ooc_${c.tag}.tcl`;
  writeFileSync(tcl,
    `read_verilog ${MOD}\n` +
    `synth_design -top ooc_spatial_rom -part xcu250-figd2104-2L-e -mode out_of_context ` +
    `-generic DEPTH=${c.depth} -generic WIDE_W=${c.wide} -generic ADDR_W=${c.addr} -generic MEM_INIT=${c.mem}\n` +
    `report_utilization -file ${rpt}\n` +
    `puts EDONE\nexit\n`, "utf8");
  const args = ["/c", VIVADO, "-mode", "batch", "-source", tcl, "-notrace"];
  try {
    const t0 = Date.now();
    execFileSync("cmd.exe", args, { cwd: OOC, timeout: 1200000, maxBuffer: 64 * 1024 * 1024, stdio: ["ignore", "inherit", "inherit"] });
    console.log(`[${c.tag}] vivado done in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  } catch (e) {
    console.log(`[${c.tag}] vivado FAILED: ${e.message?.slice(0, 200)}`);
  }
  const r = ramb(rpt);
  results.push(`${c.tag} (${c.depth}x${c.wide}): ${r}`);
  console.log(`[${c.tag}] ${r}`);
}

console.log("\n========== SPATIAL ROM OOC SUMMARY ==========");
for (const r of results) console.log("  " + r);
console.log("OOC_SPATIAL_DONE");
