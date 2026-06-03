// Batch byte-exact verifier for MobileNetV2 modules (post bit-exact fix).
// usage: npx tsx scripts/verify_mbv2_batch.ts <mod1> <mod2> ...
// Appends coord_scheduler dep (convs instantiate it); uses the mobilenet-v2 sidecar/golden.
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { run_verilator } from "../mcp/tools.ts";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

async function main() {
  const mods = process.argv.slice(2);
  let pass = 0, fail = 0;
  for (const mod of mods) {
    const vPath = path.join(repoRoot, "output", "mobilenet-v2", "rtl", `${mod}.v`);
    const sidecar = path.join(repoRoot, "output", "mobilenet-v2", "tb", `${mod}.sidecar.json`);
    let src: string;
    try { src = await readFile(vPath, "utf8"); }
    catch { console.log(`${mod}: NO RTL`); fail++; continue; }
    for (const d of ["coord_scheduler"]) {
      if (src.includes(d + " ") && !new RegExp(`module\\s+${d}\\b`).test(src)) {
        src += "\n" + await readFile(path.join(repoRoot, "rtl_library", `${d}.v`), "utf8");
      }
    }
    try {
      const r = await run_verilator(src, mod, sidecar) as any;
      const ok = r.status === "pass" || r.passed === true;
      if (ok) pass++; else fail++;
      console.log(`${ok ? "PASS" : "FAIL"} ${mod}: status=${r.status} mismatch=${r.mismatch_count} max_err=${r.max_error} (${r.exact_match_count}/${r.sample_count}) timing=${r.timing_pass} ${r.verilator_stderr ? "STDERR:" + String(r.verilator_stderr).slice(0,200) : ""}`);
    } catch (e) {
      fail++;
      console.log(`FAIL ${mod}: EXCEPTION ${(e as Error).message?.slice(0, 200)}`);
    }
  }
  console.log(`\n=== BATCH: ${pass} pass / ${fail} fail of ${mods.length} ===`);
}
main().catch((e) => { console.error(e); process.exit(2); });
