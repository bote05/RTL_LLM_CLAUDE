// Regenerate the contract goldens (output/goldens/contracts/<key>/) from the
// current LOGICAL goldens, by calling the orchestrator's exact retile
// (materializeContractGoldens). Needed after regenerating goldens at INT4-GPTQ:
// generate_golden only writes the LOGICAL goldens; the contract goldens (used by
// equiv_one + the e2e/probe harnesses) are retiled by the orchestrator and were
// left stale (INT8). This rebuilds them in-place so verification is consistent.
//
//   npx tsx scripts/rebuild_contract_goldens.ts [--only conv_196,node_relu_48]
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { materializeContractGoldens } from "../sdk/orchestrate.ts";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

async function main(): Promise<void> {
  const irPath = path.join(repoRoot, "output", "layer_ir.json");
  const ir = JSON.parse(await readFile(irPath, "utf8")) as { layers: any[] };
  const onlyArg = process.argv.find((a) => a.startsWith("--only="))?.slice("--only=".length)
    ?? (process.argv.includes("--only") ? process.argv[process.argv.indexOf("--only") + 1] : undefined);
  const only = onlyArg ? new Set(onlyArg.split(",")) : null;

  let ok = 0, skip = 0, fail = 0;
  for (const layer of ir.layers) {
    if (only && !only.has(layer.module_id)) { skip += 1; continue; }
    // only ops that have a tiled contract golden (conv/relu/add/maxpool/gap/gemm)
    try {
      const r = await materializeContractGoldens(layer as any);
      ok += 1;
      if (ok <= 4 || only) console.log(`[contract] ${layer.module_id} -> ${path.relative(repoRoot, r.goldenOutputsPath)}`);
    } catch (e) {
      fail += 1;
      console.error(`[contract] FAIL ${layer.module_id}: ${(e as Error).message}`);
    }
  }
  console.log(`[contract] done: ${ok} rebuilt, ${skip} skipped, ${fail} failed`);
  if (fail > 0) process.exit(1);
}

main().catch((e) => { console.error(e); process.exit(1); });
