// One-shot parser sanity: re-parse the cached ZCU102 Vivado report so we can
// confirm the new setup_wns / hold_wns split lands without a fresh ($) run.
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { parseVivadoReport } from "../mcp/tools.ts";

async function main(): Promise<void> {
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const reportPath = path.resolve(__dirname, "..", "output", "reports", "layer1_0_conv1.vivado.json");
  const blob = JSON.parse(await readFile(reportPath, "utf8"));
  const reparsed = parseVivadoReport(blob.report, 20, blob.part);
  console.log(JSON.stringify(
    {
      part: reparsed.part,
      lut_count: reparsed.lut_count,
      ff_count: reparsed.ff_count,
      dsp_count: reparsed.dsp_count,
      bram18_equiv: reparsed.bram18_equiv,
      wns_ns: reparsed.wns_ns,
      setup_wns_ns: reparsed.setup_wns_ns,
      hold_wns_ns: reparsed.hold_wns_ns,
      timing_met: reparsed.timing_met,
      fmax_mhz: reparsed.fmax_mhz,
    },
    null,
    2,
  ));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
