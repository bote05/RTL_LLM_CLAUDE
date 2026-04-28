import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { run_vivado } from "../mcp/tools.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

async function main() {
  const moduleName = "layer1_0_add";
  const verilogPath = path.join(repoRoot, "output", "rtl", `${moduleName}.v`);
  const source = await readFile(verilogPath, "utf8");
  console.log(`[smoke] running run_vivado on ${moduleName}`);
  const t0 = Date.now();
  const r = await run_vivado(source, moduleName, 20, "xc7a100tcsg324-1");
  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`[smoke] returned in ${elapsed}s`);
  for (const k of ['success','stage','lut_count','ff_count','dsp_count','bram18_count','bram36_count','wns_ns','timing_met','fmax_mhz']) {
    console.log(`  ${k}:`, (r as any)[k]);
  }
}
main().catch(e => { console.error(e); process.exit(1); });
