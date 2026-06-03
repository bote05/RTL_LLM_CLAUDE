// OOC Vivado synth of the node_conv_818 compressed variant.
// Synths on the SAME part the baseline report used (xczu9eg-ffvb1156-2-e) for
// an apples-to-apples LUT comparison. Writes the report to improved/ so the
// baseline reports/node_conv_818.vivado.json is NEVER overwritten.
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { run_vivado } from "../mcp/tools.ts";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

async function main() {
  const mod = "node_conv_818";
  const part = process.argv[2] || "xczu9eg-ffvb1156-2-e";
  const vPath = path.join(repoRoot, "output", "mobilenet-v2", "rtl", "improved", `${mod}.compressed.v`);
  const outDir = path.join(repoRoot, "output", "mobilenet-v2", "reports", "improved");
  const reportPath = path.join(outDir, `${mod}.compressed.vivado.json`);
  const src = await readFile(vPath, "utf8");
  // clock_period_ns = 20 (from sidecar; matches baseline 148MHz fmax headroom).
  const clockPeriodNs = 20;
  console.log(`[synth] module=${mod} part=${part} clock_period_ns=${clockPeriodNs}`);
  console.log(`[synth] rtl=${vPath}`);
  const t0 = Date.now();
  const report = await run_vivado(src, mod, clockPeriodNs, part);
  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`[synth] done in ${elapsed}s success=${report.success} lut=${report.lut_count} ff=${report.ff_count} bram18=${report.bram18_count} bram36=${report.bram36_count} dsp=${report.dsp_count} timing_met=${report.timing_met} fmax=${report.fmax_mhz?.toFixed(2)}`);
  await mkdir(outDir, { recursive: true });
  await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  console.log(`[synth] wrote ${reportPath}`);
}

main().catch((e) => {
  console.error(e instanceof Error ? e.message : String(e));
  process.exit(1);
});
