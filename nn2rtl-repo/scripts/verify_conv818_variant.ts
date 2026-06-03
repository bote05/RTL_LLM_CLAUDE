// One-off byte-exact verifier for the node_conv_818 compressed variant.
// Mirrors equiv_one.ts but points at the improved/ variant and the
// mobilenet-v2 sidecar + goldens. Appends coord_scheduler (instantiated dep).
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { run_verilator } from "../mcp/tools.ts";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

async function main() {
  const mod = "node_conv_818";
  const vPath = path.join(repoRoot, "output", "mobilenet-v2", "rtl", "improved", `${mod}.compressed.v`);
  const sidecar = path.join(repoRoot, "output", "mobilenet-v2", "tb", `${mod}.sidecar.json`);
  let src = await readFile(vPath, "utf8");
  const deps = ["coord_scheduler"];
  for (const d of deps) {
    if (src.includes(d + " ") && !new RegExp(`module\\s+${d}\\b`).test(src)) {
      const dep = await readFile(path.join(repoRoot, "rtl_library", `${d}.v`), "utf8");
      src += "\n" + dep;
    }
  }
  console.log(`[verify] module=${mod} variant=${vPath}`);
  const res = await run_verilator(src, mod, sidecar);
  console.log(JSON.stringify(res, null, 2));
  const ok = (res as any).passed ?? (res as any).equivalent ?? false;
  process.exit(ok ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(2);
});
