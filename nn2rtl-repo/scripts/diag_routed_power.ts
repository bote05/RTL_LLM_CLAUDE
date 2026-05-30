// Run report_power against a routed checkpoint. ~10-30 min depending on design
// size. Vectorless (default activity model). Output: post-route power
// breakdown (clocks / signals / logic / BRAM / DSP / IO / static) in watts.

import { writeFile, mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

import { resolveVivadoCommand, toVivadoPath, withTempDir, VIVADO_MAX_BUFFER_BYTES } from "../mcp/tools.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const rawArgs = process.argv.slice(2);

function flag(name: string, fallback?: string): string | undefined {
  const eq = rawArgs.find((a) => a.startsWith(`--${name}=`));
  if (eq) return eq.slice(name.length + 3);
  const idx = rawArgs.indexOf(`--${name}`);
  if (idx >= 0 && rawArgs[idx + 1] && !rawArgs[idx + 1].startsWith("--")) return rawArgs[idx + 1];
  return fallback;
}

const safeDir = path.join(repoRoot, "output", "reports_integrated", "checkpoints");
const inputRaw = flag("input") ?? path.join(safeDir, "first_light_routed_40ns_explore.dcp");
const inputDcp = path.isAbsolute(inputRaw) ? inputRaw : path.resolve(repoRoot, inputRaw);
const outRpt = path.join(safeDir, "first_light_postroute_power_40ns.rpt");

function tclQuote(value: string): string {
  return `"${toVivadoPath(value).replace(/(["$[\]])/g, "\\$1")}"`;
}

const execFileP = promisify(execFile);

async function main(): Promise<void> {
  await mkdir(safeDir, { recursive: true });
  if (!existsSync(inputDcp)) throw new Error(`routed.dcp not found: ${inputDcp}`);

  await withTempDir("nn2rtl-power-", async (tempDir) => {
    const tclPath = path.join(tempDir, "power.tcl");
    const tcl = [
      `puts "NN2RTL_INFO: opening routed checkpoint"`,
      `open_checkpoint ${tclQuote(inputDcp)}`,
      `puts "NN2RTL_INFO: running report_power (vectorless)"`,
      `report_power -file ${tclQuote(outRpt)}`,
      `puts "NN2RTL_INFO: power report complete"`,
    ].join("\n") + "\n";
    await writeFile(tclPath, tcl, "utf8");

    const vivadoBin = resolveVivadoCommand(process.env);
    const vivadoArgs = ["-mode", "batch", "-source", toVivadoPath(tclPath), "-notrace"];
    const isWindowsBatch = process.platform === "win32" && /\.(bat|cmd)$/i.test(vivadoBin);
    const spawnFile = isWindowsBatch ? "cmd.exe" : vivadoBin;
    const spawnArgs = isWindowsBatch ? ["/c", vivadoBin, ...vivadoArgs] : vivadoArgs;

    console.log(`[power] launching vivado on ${path.basename(inputDcp)}`);
    const t0 = Date.now();
    let stdout = "", stderr = "", exitOk = true;
    try {
      const res = await execFileP(spawnFile, spawnArgs, {
        cwd: tempDir, env: process.env,
        timeout: 3600 * 1000, maxBuffer: VIVADO_MAX_BUFFER_BYTES,
      });
      stdout = res.stdout; stderr = res.stderr;
    } catch (err: unknown) {
      exitOk = false;
      const e = err as { stdout?: string | Buffer; stderr?: string | Buffer; message?: string };
      stdout = typeof e.stdout === "string" ? e.stdout : (e.stdout?.toString() ?? "");
      stderr = typeof e.stderr === "string" ? e.stderr : (e.stderr?.toString() ?? e.message ?? "");
    }
    const elapsed = (Date.now() - t0) / 1000;
    console.log(`[power] elapsed=${elapsed.toFixed(1)}s ok=${exitOk}`);
    console.log(`[power] report at ${outRpt}`);
    if (existsSync(outRpt)) {
      const text = await readFile(outRpt, "utf8");
      // Extract key lines for stdout summary
      const lines = text.split(/\r?\n/);
      console.log("\n=== POWER SUMMARY ===");
      for (const l of lines) {
        if (/Total On-Chip Power|Dynamic|Device Static|Effective TJA|Thermal Margin|Confidence Level/i.test(l)) {
          console.log(l);
        }
      }
    }
  });
}

main().catch((err: unknown) => { console.error(err instanceof Error ? err.stack ?? err.message : String(err)); process.exit(1); });
