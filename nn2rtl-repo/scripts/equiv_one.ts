// Minimal single-module equivalence runner.
//   npx tsx scripts/equiv_one.ts node_conv_218
//   npx tsx scripts/equiv_one.ts node_conv_284 output/tb/node_conv_284.streaming.sidecar.json
// Reads output/rtl/<mod>.v + a sidecar (default output/tb/<mod>.sidecar.json, or
// an explicit path as argv[3]), runs the static Verilator TB against the golden
// vectors, prints PASS/FAIL. The explicit-sidecar form lets a module that was
// re-architected to a different contract (e.g. dram-backed -> tiled-streaming)
// reuse the SAME golden vectors under the contract its new ports match.
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { run_verilator } from "../mcp/tools.ts";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

async function main() {
  const mod = process.argv[2];
  if (!mod) throw new Error("usage: equiv_one.ts <module_name>");
  const vPath = path.join(repoRoot, "output", "rtl", `${mod}.v`);
  const sidecarArg = process.argv[3];
  const sidecar = sidecarArg
    ? path.resolve(repoRoot, sidecarArg)
    : path.join(repoRoot, "output", "tb", `${mod}.sidecar.json`);
  let src = await readFile(vPath, "utf8");
  // Append any submodule deps not auto-included by the lib copier.
  const deps = ["conv_datapath_mp_k", "conv_datapath_parallel", "coord_scheduler", "line_buf_window"];
  for (const d of deps) {
    // only append if instantiated AND not already in the source
    if (src.includes(d + " ") && !new RegExp(`module\\s+${d}\\b`).test(src)) {
      const dep = await readFile(path.join(repoRoot, "rtl_library", `${d}.v`), "utf8");
      src += "\n" + dep;
    }
  }
  console.log(`[equiv] module=${mod}`);
  const res = await run_verilator(src, mod, sidecar);
  console.log(JSON.stringify(res, null, 2));
  // VerifResult typically has { passed, mismatches, ... }
  const ok = (res as any).passed ?? (res as any).equivalent ?? false;
  process.exit(ok ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(2);
});
