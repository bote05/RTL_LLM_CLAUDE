// Replay a saved improve attempt without spending Foundry tokens:
//  1) load the attempt's module from output/improve/<mid>/<runId>/attempt_<n>.module.json
//  2) run the deterministic assayer against it with the new acceptance logic
//     (increase-throughput tolerates per-vector cycle drop when bit-exact and
//     first-frame timing matches)
//  3) if accepted, run Vivado synth, compute metrics, and evaluate the
//     acceptance gate against the baseline.
//
// Usage:
//   tsx scripts/replay_improve_attempt.ts <module_id> <runId> <attempt_index> \
//                                          --targets=increase-throughput

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { runAssayerDeterministic } from "../sdk/orchestrate.ts";
import {
  layerIrSchema,
  pipelineIrSchema,
  synthesisReportSchema,
  verifResultSchema,
  verilogModuleSchema,
} from "../sdk/schemas.ts";
import * as mcpTools from "../mcp/tools.ts";
import {
  DEFAULT_IMPROVEMENT_CHECKER_CONFIG,
  evaluateImprovementTargets,
  parseImprovementTargets,
  type ImprovementMetrics,
} from "../sdk/improve.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

function getArg(name: string): string | undefined {
  const flag = process.argv.find((a) => a.startsWith(`--${name}=`));
  return flag?.slice(name.length + 3);
}

async function main(): Promise<void> {
  const [moduleId, runId, attemptRaw] = process.argv.slice(2);
  if (!moduleId || !runId || !attemptRaw) {
    console.error(
      "usage: tsx scripts/replay_improve_attempt.ts <module_id> <runId> <attempt_index> --targets=<csv>",
    );
    process.exit(1);
  }
  const attemptIndex = Number.parseInt(attemptRaw, 10);
  const targets = parseImprovementTargets(getArg("targets") ?? "increase-throughput");

  const attemptDir = path.join(repoRoot, "output", "improve", moduleId, runId);
  const modulePath = path.join(attemptDir, `attempt_${attemptIndex}.module.json`);
  const moduleRaw = JSON.parse(await readFile(modulePath, "utf8"));
  const module = verilogModuleSchema.parse(moduleRaw);

  const irPath = path.join(repoRoot, "output", "layer_ir.json");
  const ir = pipelineIrSchema.parse(JSON.parse(await readFile(irPath, "utf8")));
  const layer = layerIrSchema.parse(
    ir.layers.find((l) => l.module_id === moduleId) ??
      (() => { throw new Error(`layer ${moduleId} not in layer_ir.json`); })(),
  );

  // Baseline metrics (canonical reports).
  const baselineVivado = synthesisReportSchema.parse(
    JSON.parse(
      await readFile(path.join(repoRoot, "output", "reports", `${moduleId}.vivado.json`), "utf8"),
    ),
  );
  const baselineVerif = verifResultSchema.parse(
    JSON.parse(
      await readFile(path.join(repoRoot, "output", "reports", `${moduleId}.results.json`), "utf8"),
    ),
  );
  const baselineFrames = (baselineVerif.per_vector ?? []).length;
  const baselineII = baselineFrames > 0 && typeof baselineVerif.last_valid_out_cycle === "number" &&
    typeof baselineVerif.first_valid_in_cycle === "number"
    ? (baselineVerif.last_valid_out_cycle - baselineVerif.first_valid_in_cycle) / baselineFrames
    : undefined;
  const baseline: ImprovementMetrics = {
    lut: baselineVivado.lut_count,
    ff: baselineVivado.ff_count,
    dsp: baselineVivado.dsp_count,
    bram: baselineVivado.bram18_equiv || baselineVivado.bram18_count + baselineVivado.bram36_count * 2,
    fmax_mhz: baselineVivado.fmax_mhz,
    latency_cycles: baselineVerif.timing_actual_cycles && baselineVerif.timing_actual_cycles >= 0
      ? baselineVerif.timing_actual_cycles
      : undefined,
    ii: baselineII,
  };

  console.log("[replay] baseline:", baseline);

  // runAssayerDeterministic + run_vivado both write to canonical report paths
  // (output/reports/<mid>.results.json, .vivado.json) as a side effect. The
  // replay must NOT permanently overwrite those — they describe the canonical
  // baseline, not this hypothetical attempt. Snapshot before, restore after.
  const reportsToProtect = [
    path.join(repoRoot, "output", "reports", `${moduleId}.results.json`),
    path.join(repoRoot, "output", "reports", `${moduleId}.vivado.json`),
    path.join(repoRoot, "output", "rtl", `${moduleId}.v`),
    path.join(repoRoot, "output", "rtl", `${moduleId}.meta.json`),
  ];
  const protectedSnapshots = new Map<string, string>();
  for (const filePath of reportsToProtect) {
    try {
      protectedSnapshots.set(filePath, await readFile(filePath, "utf8"));
    } catch {
      // file may not exist — fine, nothing to restore.
    }
  }

  let verif: Awaited<ReturnType<typeof runAssayerDeterministic>>;
  let vivado: Awaited<ReturnType<typeof mcpTools.run_vivado>>;
  let frames = 0;
  let firstFrame: number | undefined;
  let ii: number | undefined;
  try {
    // Step 1: re-run the deterministic assayer on the saved RTL.
    console.log("[replay] running assayer ...");
    const t0 = Date.now();
    verif = await runAssayerDeterministic(module, layer);
    const dt = ((Date.now() - t0) / 1000).toFixed(1);
    frames = (verif.per_vector ?? []).length;
    firstFrame = verif.per_vector?.[0]?.actual_cycles;
    ii = frames > 0 && typeof verif.last_valid_out_cycle === "number" &&
      typeof verif.first_valid_in_cycle === "number"
      ? (verif.last_valid_out_cycle - verif.first_valid_in_cycle) / frames
      : undefined;
    console.log(
      `[replay] assayer done in ${dt}s: status=${verif.status} mismatch=${verif.mismatch_count} max_err=${verif.max_error} first_frame_cycles=${firstFrame} ii=${ii}`,
    );

    // Step 2: synth.
    console.log("[replay] running vivado synth ...");
    const synthT0 = Date.now();
    vivado = await mcpTools.run_vivado(module.verilog_source, module.module_id);
    const synthDt = ((Date.now() - synthT0) / 1000).toFixed(1);
    console.log(
      `[replay] synth done in ${synthDt}s: lut=${vivado.lut_count} ff=${vivado.ff_count} dsp=${vivado.dsp_count} bram=${vivado.bram18_equiv ?? vivado.bram18_count + vivado.bram36_count * 2} fmax=${vivado.fmax_mhz?.toFixed(2)}`,
    );
  } finally {
    // Always restore canonical reports, even if assayer/synth threw. Without
    // this, a failed replay leaves the canonical reports in an inconsistent
    // half-replay state.
    for (const [filePath, content] of protectedSnapshots.entries()) {
      await writeFile(filePath, content, "utf8");
    }
    console.log(`[replay] restored ${protectedSnapshots.size} canonical file(s)`);
  }

  const nextII = ii ?? Number.POSITIVE_INFINITY;
  const next: ImprovementMetrics = {
    lut: vivado.lut_count,
    ff: vivado.ff_count,
    dsp: vivado.dsp_count,
    bram: vivado.bram18_equiv || vivado.bram18_count + vivado.bram36_count * 2,
    fmax_mhz: vivado.fmax_mhz,
    latency_cycles: verif.timing_actual_cycles ?? undefined,
    ii: nextII,
  };
  console.log("[replay] next:", next);

  // Step 3: acceptance gate.
  const verdict = evaluateImprovementTargets(baseline, next, targets, DEFAULT_IMPROVEMENT_CHECKER_CONFIG);
  console.log("[replay] verdict overall=", verdict.overall);
  for (const t of verdict.targets) {
    console.log(`  - ${t.target} satisfied=${t.satisfied} baseline=${t.baseline_value} new=${t.new_value} reason=${t.reason}`);
  }

  // Persist a small summary for inspection.
  const baselineFps = baseline.fmax_mhz && baseline.ii ? (baseline.fmax_mhz * 1e6) / baseline.ii : null;
  const newFps = next.fmax_mhz && next.ii ? (next.fmax_mhz * 1e6) / next.ii : null;
  const summary = {
    module_id: moduleId,
    run_id: runId,
    attempt_index: attemptIndex,
    targets,
    baseline_metrics: baseline,
    new_metrics: next,
    baseline_fps: baselineFps,
    new_fps: newFps,
    fps_speedup: baselineFps && newFps ? newFps / baselineFps : null,
    verif_status: verif.status,
    verif_bit_exact: verif.mismatch_count === 0 && verif.max_error === 0,
    verif_first_frame_cycles: firstFrame,
    verif_steady_state_cycles: verif.per_vector?.[1]?.actual_cycles ?? null,
    verdict,
  };
  const summaryPath = path.join(repoRoot, "output", "reports", `replay_${moduleId}_${runId}_attempt${attemptIndex}.json`);
  await writeFile(summaryPath, JSON.stringify(summary, null, 2), "utf8");
  console.log("[replay] summary:", summaryPath);
}

main().catch((e) => {
  console.error("[replay] FATAL:", e instanceof Error ? e.stack ?? e.message : String(e));
  process.exit(1);
});
