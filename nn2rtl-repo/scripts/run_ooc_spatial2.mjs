// Bottom-up Config B spatial-ROM BRAM: OOC-measure the 9 distinct (depth x width) shapes
// among the 35 INT4 spatial convs (the 4 INT3 ones already measured = 840), then weight by count.
// Reuses ooc_spatial_rom.v + the cmd.exe /c vivado.bat launch.
import { execFileSync } from "node:child_process";
import { writeFileSync, readFileSync, existsSync, readdirSync } from "node:fs";

const REPO = "C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo";
const OOC = `${REPO}/output/reports_integrated/ooc`;
const W = `${REPO}/output/weights`;
const VIVADO = process.env.NN2RTL_VIVADO_BIN || "D:/vivado/2025.2/Vivado/bin/vivado.bat";
const MOD = `${OOC}/ooc_spatial_rom.v`;

function hexFor(conv) {
  const f = readdirSync(W).find((x) => new RegExp(`^node_conv_${conv}_weights_mp_k_\\d+\\.hex$`).test(x));
  return f ? `${W}/${f}` : null;
}
// shape: [tag, depth, wide, addr, repConv, count]
const SHAPES = [
  ["d168x224",   168, 224,  8, 196, 1],
  ["d32x512",     32, 512,  6, 198, 1],
  ["d256x576",   256, 576,  9, 200, 3],
  ["d128x512",   128, 512,  8, 202, 6],
  ["d256x512",   256, 512,  9, 218, 1],
  ["d1024x576", 1024, 576, 11, 220, 4],
  ["d512x512",   512, 512, 10, 222, 7],
  ["d1024x512", 1024, 512, 11, 224, 2],
  ["d2048x512", 2048, 512, 12, 248, 10],
];

function ramb(rptPath) {
  if (!existsSync(rptPath)) return null;
  const txt = readFileSync(rptPath, "utf8");
  const m = txt.match(/Block RAM Tile\s*\|\s*([0-9.]+)/);
  return m ? parseFloat(m[1]) : null;
}

const rows = [];
let int4Total = 0;
for (const [tag, depth, wide, addr, conv, count] of SHAPES) {
  const mem = hexFor(conv);
  console.log(`\n===== ${tag}: ${depth}x${wide} (rep conv_${conv}, x${count}) =====`);
  if (!mem) { console.log(`  NO HEX for conv_${conv}`); rows.push([tag, null, count]); continue; }
  const rpt = `${OOC}/sp_${tag}.rpt`;
  const tcl = `${OOC}/ooc_sp_${tag}.tcl`;
  writeFileSync(tcl,
    `read_verilog ${MOD}\n` +
    `synth_design -top ooc_spatial_rom -part xcu250-figd2104-2L-e -mode out_of_context ` +
    `-generic DEPTH=${depth} -generic WIDE_W=${wide} -generic ADDR_W=${addr} -generic MEM_INIT=${mem}\n` +
    `report_utilization -file ${rpt}\nputs EDONE\nexit\n`, "utf8");
  try {
    const t0 = Date.now();
    execFileSync("cmd.exe", ["/c", VIVADO, "-mode", "batch", "-source", tcl, "-notrace"],
      { cwd: OOC, timeout: 1200000, maxBuffer: 64 * 1024 * 1024, stdio: ["ignore", "inherit", "inherit"] });
    console.log(`[${tag}] done ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  } catch (e) { console.log(`[${tag}] FAILED: ${(e.message || "").slice(0, 150)}`); }
  const tiles = ramb(rpt);
  const sub = tiles == null ? null : tiles * count;
  if (sub != null) int4Total += sub;
  rows.push([tag, tiles, count, sub]);
  console.log(`[${tag}] tiles=${tiles} x${count} = ${sub}`);
}

console.log("\n========== SPATIAL INT4 ROM BRAM (bottom-up) ==========");
for (const [tag, tiles, count, sub] of rows) console.log(`  ${tag}: ${tiles} RAMB36 x${count} = ${sub}`);
console.log(`  --- INT4 spatial ROM subtotal = ${int4Total} RAMB36`);
console.log(`  + 4 INT3 spatial ROMs (measured) = 840`);
console.log(`  + engine banks (measured, narrowed) = 960`);
console.log(`  SPATIAL+ENGINE WEIGHT ROM TOTAL = ${int4Total + 840 + 960} RAMB36 (of 2688)`);
console.log("OOC_SPATIAL2_DONE");
