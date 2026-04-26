// Vivado smoke test — runs `run_vivado` against the proven-passing 1×1 conv
// reference to validate the entire MCP-side chain:
//   * Tcl generation
//   * toVivadoPath / readmemh path conversion
//   * batch invocation
//   * report parsing (LUT/FF/DSP/BRAM/WNS/Fmax)
//
// Does NOT touch the LLM, the orchestrator, or the pipeline. If this passes,
// the Vivado integration is wired correctly end-to-end.
//
// Usage:
//   set NN2RTL_VIVADO_BIN=D:/vivado/2025.2/Vivado/bin/vivado.bat
//   npx tsx scripts/vivado_smoke.ts

import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { run_vivado } from "../mcp/tools.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

async function main(): Promise<void> {
  // Use the proven-passing 1×1 reference — has real pipelined logic so we
  // get a non-null WNS / Fmax measurement, not the "no constrained path"
  // case a pure passthrough produces.
  const refPath = path.join(repoRoot, "knowledge", "references", "conv1x1_passing_reference.v");
  const moduleSource = await readFile(refPath, "utf8");
  const moduleMatch = moduleSource.match(/^\s*module\s+([A-Za-z_][A-Za-z0-9_]*)/m);
  if (!moduleMatch) throw new Error("Could not extract module name from reference Verilog.");
  const moduleName = moduleMatch[1];

  console.log("[smoke] NN2RTL_VIVADO_BIN =", process.env.NN2RTL_VIVADO_BIN ?? "(unset, falling back to 'vivado' on PATH)");
  console.log(`[smoke] running run_vivado on '${moduleName}' (${refPath})`);
  console.log("[smoke] target xc7a100tcsg324-1, 20 ns clock (50 MHz)...");
  const t0 = Date.now();
  const report = await run_vivado(
    moduleSource,
    moduleName,
    20,                     // 20 ns = 50 MHz target
    "xc7a100tcsg324-1",
  );
  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

  console.log(`[smoke] run_vivado returned in ${elapsed}s`);
  // Echo the maxThreads + Detected line so we can verify Vivado is actually
  // parallelising on this host.
  const threadLines = report.report
    .split(/\r?\n/)
    .filter((l) => /maxThreads|Detected processor|Detected.*cores|set_param.*general\.maxThreads|Number of threads/i.test(l))
    .slice(0, 8);
  if (threadLines.length > 0) {
    console.log("[smoke] ---- threading lines ----");
    threadLines.forEach((l) => console.log("[smoke]   " + l.trim()));
  }
  console.log("[smoke] success      =", report.success);
  console.log("[smoke] tool         =", report.tool);
  console.log("[smoke] part         =", report.part);
  console.log("[smoke] stage        =", report.stage);
  console.log("[smoke] lut_count    =", report.lut_count);
  console.log("[smoke] ff_count     =", report.ff_count);
  console.log("[smoke] dsp_count    =", report.dsp_count);
  console.log("[smoke] bram18_count =", report.bram18_count);
  console.log("[smoke] bram36_count =", report.bram36_count);
  console.log("[smoke] bram18_equiv =", report.bram18_equiv);
  console.log("[smoke] wns_ns       =", report.wns_ns);
  console.log("[smoke] timing_met   =", report.timing_met);
  console.log("[smoke] fmax_mhz     =", report.fmax_mhz.toFixed(2));
  if (!report.success) {
    console.log("[smoke] ---- report (head) ----");
    console.log(report.report.slice(0, 4000));
    process.exit(2);
  }
  // Always dump the timing-summary slice so we can see why WNS may be null.
  if (report.wns_ns === null) {
    const tStart = report.report.indexOf("post_synth_timing_summary.rpt");
    if (tStart >= 0) {
      // Grab a big slice and grep for the Design Timing Summary block,
      // which is the canonical WNS source. Skip the verbose "checking"
      // sections at the top.
      const fullSlice = report.report.slice(tStart);
      const designSummary = fullSlice.match(/Design Timing Summary[\s\S]{0,3500}/);
      if (designSummary) {
        console.log("[smoke] ---- Design Timing Summary ----");
        console.log(designSummary[0]);
      } else {
        console.log("[smoke] ---- timing-summary head (4 KB) ----");
        console.log(fullSlice.slice(0, 4000));
      }
    }
  }
  // Threading echo: dump the head of the combined report so we see what
  // Vivado actually accepted for general.maxThreads.
  const headLines = report.report.split(/\r?\n/).slice(0, 80);
  const threadEcho = headLines.filter((l) =>
    /maxThreads|Detected processor|Detected.*cores|Number of CPUs|Number of threads/i.test(l),
  );
  if (threadEcho.length > 0) {
    console.log("[smoke] ---- threading lines ----");
    threadEcho.forEach((l) => console.log("[smoke]   " + l.trim()));
  }
  if (report.lut_count === 0 && report.ff_count === 0) {
    console.log("[smoke] WARNING: zero LUT and FF — parser may not have matched the report.");
    // Dump just the utilization-table region so we can see what labels are
    // present and adjust the regexes.
    const utilHead = report.report.indexOf("post_synth_utilization.rpt");
    const utilEnd  = report.report.indexOf("post_synth_ram_utilization.rpt");
    const slice = utilHead >= 0 && utilEnd > utilHead
      ? report.report.slice(utilHead, utilEnd)
      : report.report.slice(0, 10000);
    console.log("[smoke] ---- utilization slice ----");
    console.log(slice);
    process.exit(3);
  }
  console.log("[smoke] OK");
}

main().catch((err) => {
  console.error("[smoke] FATAL:", err instanceof Error ? err.stack ?? err.message : String(err));
  process.exit(1);
});
