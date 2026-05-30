// Cheap Fmax pre-check: open routed.dcp, get slack histogram + worst paths
// WITHOUT re-routing. Tells us the true post-route timing headroom in ~5 min.
//
// What it answers: at the original 40 ns route, what's the worst-N path slack?
// If most paths cluster near +5-7 ns headroom -> tighter Fmax is reachable.
// If most cluster at +10+ ns -> design is already near its slow-corner ceiling
// at the chosen routing strategy.
//
// Usage:
//   npx tsx scripts/diag_routed_slack.ts \
//     [--input=output/reports_integrated/checkpoints/first_light_routed_40ns_explore.dcp]

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
const outDir = path.join(repoRoot, "output", "reports_integrated");
const outRpt = path.join(outDir, "diag_routed_slack.rpt");
const outSummary = path.join(outDir, "diag_routed_slack_summary.txt");

function tclQuote(value: string): string {
  return `"${toVivadoPath(value).replace(/(["$[\]])/g, "\\$1")}"`;
}

function buildTcl(input: { dcp: string; rpt: string }): string {
  return [
    `puts "NN2RTL_INFO: opening routed checkpoint"`,
    `open_checkpoint ${tclQuote(input.dcp)}`,
    `puts "NN2RTL_INFO: report_timing_summary at the routed constraint"`,
    `report_timing_summary -file ${tclQuote(input.rpt)} -max_paths 100 -slack_lesser_than 1000`,
    `puts "NN2RTL_INFO: report_timing with nworst=200 (slack histogram source)"`,
    `report_timing -delay_type max -max_paths 200 -nworst 1 -unique_pins -path_type summary -file ${tclQuote(input.rpt + ".paths")}`,
    `puts "NN2RTL_INFO: diagnostic complete"`,
  ].join("\n") + "\n";
}

const execFileP = promisify(execFile);

async function main(): Promise<void> {
  await mkdir(outDir, { recursive: true });
  if (!existsSync(inputDcp)) throw new Error(`routed.dcp not found: ${inputDcp}`);

  await withTempDir("nn2rtl-slack-", async (tempDir) => {
    const tclPath = path.join(tempDir, "diag.tcl");
    await writeFile(tclPath, buildTcl({ dcp: inputDcp, rpt: outRpt }), "utf8");

    const vivadoBin = resolveVivadoCommand(process.env);
    const vivadoArgs = ["-mode", "batch", "-source", toVivadoPath(tclPath), "-notrace"];
    const isWindowsBatch = process.platform === "win32" && /\.(bat|cmd)$/i.test(vivadoBin);
    const spawnFile = isWindowsBatch ? "cmd.exe" : vivadoBin;
    const spawnArgs = isWindowsBatch ? ["/c", vivadoBin, ...vivadoArgs] : vivadoArgs;

    console.log(`[slack] launching vivado on ${inputDcp}`);
    const t0 = Date.now();
    let stdout = "", stderr = "", exitOk = true;
    try {
      const res = await execFileP(spawnFile, spawnArgs, {
        cwd: tempDir, env: process.env,
        timeout: 1800 * 1000, maxBuffer: VIVADO_MAX_BUFFER_BYTES,
      });
      stdout = res.stdout; stderr = res.stderr;
    } catch (err: unknown) {
      exitOk = false;
      const e = err as { stdout?: string | Buffer; stderr?: string | Buffer; message?: string };
      stdout = typeof e.stdout === "string" ? e.stdout : (e.stdout?.toString() ?? "");
      stderr = typeof e.stderr === "string" ? e.stderr : (e.stderr?.toString() ?? e.message ?? "");
    }
    const elapsed = (Date.now() - t0) / 1000;
    console.log(`[slack] elapsed=${elapsed.toFixed(1)}s ok=${exitOk}`);

    // Parse the paths report for slack histogram
    const pathsRpt = outRpt + ".paths";
    if (existsSync(pathsRpt)) {
      const text = await readFile(pathsRpt, "utf8");
      const slackPattern = /Slack \(MET\)\s*:\s*([0-9.\-]+)\s*ns/g;
      const slacks: number[] = [];
      let m: RegExpExecArray | null;
      while ((m = slackPattern.exec(text)) !== null) slacks.push(Number(m[1]));
      slacks.sort((a, b) => a - b);
      const summary: string[] = [];
      summary.push(`Routed checkpoint: ${path.basename(inputDcp)}`);
      summary.push(`Total paths reported: ${slacks.length}`);
      if (slacks.length > 0) {
        summary.push(`Worst slack: ${slacks[0].toFixed(3)} ns`);
        summary.push(`Median slack: ${slacks[Math.floor(slacks.length / 2)].toFixed(3)} ns`);
        summary.push(`Best slack: ${slacks[slacks.length - 1].toFixed(3)} ns`);
        // Histogram: bucketize
        const buckets: Record<string, number> = {};
        for (const s of slacks) {
          const key = s < 0 ? "negative" : s < 5 ? "0-5ns" : s < 10 ? "5-10ns" : s < 15 ? "10-15ns" : s < 20 ? "15-20ns" : s < 25 ? "20-25ns" : "25+ns";
          buckets[key] = (buckets[key] ?? 0) + 1;
        }
        summary.push("Slack histogram:");
        for (const [k, v] of Object.entries(buckets)) summary.push(`  ${k}: ${v} (${((v / slacks.length) * 100).toFixed(1)}%)`);
        // Worst 10 quoted
        summary.push("Worst 10 slacks (ns):");
        for (let i = 0; i < Math.min(10, slacks.length); i++) summary.push(`  ${i + 1}. ${slacks[i].toFixed(3)}`);
        // Cheap Fmax projection
        const worst = slacks[0];
        const constraintNs = 40; // matches the route we did
        summary.push(`At 40ns constraint, worst slack=${worst.toFixed(3)} ns -> slow-corner achievable period ${(40 - worst).toFixed(3)} ns`);
        summary.push(`Slow-corner Fmax estimate: ${(1000 / (40 - worst)).toFixed(2)} MHz`);
      }
      await writeFile(outSummary, summary.join("\n") + "\n", "utf8");
      console.log("\n=== SLACK SUMMARY ===");
      console.log(summary.join("\n"));
      console.log(`\n[slack] full reports in ${outRpt} + ${outRpt}.paths`);
      console.log(`[slack] summary at ${outSummary}`);
    } else {
      console.log("[slack] paths report not generated; vivado.log:");
      const vivadoLog = path.join(tempDir, "vivado.log");
      if (existsSync(vivadoLog)) console.log(await readFile(vivadoLog, "utf8"));
    }
  });
}

main().catch((err: unknown) => { console.error(err instanceof Error ? err.stack ?? err.message : String(err)); process.exit(1); });
