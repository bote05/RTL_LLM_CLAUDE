// Promote a previously --keep-reference'd improvement variant to canonical.
//
// Reads the variant RTL from `knowledge/references/improved/<id>__<slug>.v`
// and the matching improve report from
// `output/reports/improve_<id>__<slug>.json`, then calls the same
// `commitReplacement` helper that the in-line `replaced` branch of the
// improve flow uses. That archives the prior canonical RTL + reports under
// `output/{rtl,reports}/archive/__<stamp>...` and writes the variant +
// embedded vivado/verif/metrics into the canonical paths.
//
// Usage: tsx scripts/promote_variant.ts <module_id> <target_slug>
//   e.g. tsx scripts/promote_variant.ts node_conv_248 use-dsp
//        tsx scripts/promote_variant.ts node_conv_248 use-dsp-use-bram
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { commitReplacement, defaultImprovePaths, improvementMetricsSchema } from "../sdk/improve.js";
import { createOrchestratorRuntime } from "../sdk/orchestrate.js";
import {
  synthesisReportSchema,
  verifResultSchema,
  verilogModuleSchema,
} from "../sdk/schemas.js";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");

async function main(): Promise<void> {
  const [moduleId, targetSlug] = process.argv.slice(2);
  if (!moduleId || !targetSlug) {
    console.error("usage: tsx scripts/promote_variant.ts <module_id> <target_slug>");
    process.exit(1);
  }

  const paths = defaultImprovePaths(repoRoot);
  const variantPath = path.join(
    paths.repoRoot,
    "knowledge",
    "references",
    "improved",
    `${moduleId}__${targetSlug}.v`,
  );
  const reportPath = path.join(
    paths.reportsDir,
    `improve_${moduleId}__${targetSlug}.json`,
  );

  const variantSource = await readFile(variantPath, "utf8");
  const report = JSON.parse(await readFile(reportPath, "utf8")) as {
    final_action: string;
    success: boolean;
    attempts: Array<{
      attempt_index: number;
      module: { module_id: string; spec_hash: string; verilog_source?: string };
      vivado_report?: unknown;
      assayer_result?: unknown;
      metrics?: unknown;
      verdict?: { overall: boolean };
    }>;
  };

  if (!report.success) {
    throw new Error(
      `Improve report ${reportPath} has success=false; nothing to promote.`,
    );
  }
  // Pick the successful attempt: the last one with verdict.overall=true.
  const winning = [...report.attempts]
    .reverse()
    .find((a) => a.verdict?.overall === true && a.metrics && a.vivado_report && a.assayer_result);
  if (!winning) {
    throw new Error(
      `No attempt in ${reportPath} has verdict.overall=true with full metrics; cannot promote.`,
    );
  }

  // Reconstruct the VerilogModule from the variant file (authoritative
  // source) plus the report's embedded module metadata. The variant on disk
  // is the same RTL the improve run validated, so we trust it over the
  // report's embedded copy.
  const module = verilogModuleSchema.parse({
    module_id: winning.module.module_id,
    spec_hash: winning.module.spec_hash,
    verilog_source: variantSource,
    generated_by: "Foundry",
    attempt: winning.attempt_index,
  });
  const vivadoReport = synthesisReportSchema.parse(winning.vivado_report);
  const verifResult = verifResultSchema.parse(winning.assayer_result);
  const metrics = improvementMetricsSchema.parse(winning.metrics);

  const runtime = createOrchestratorRuntime({});
  const { committedPath, archivedOriginalPath } = await commitReplacement({
    paths,
    moduleId,
    module,
    metrics,
    vivadoReport,
    verifResult,
    runtime,
  });

  // Rewrite a small confirmation summary so the user can sanity-check the
  // promotion without re-reading the report manually.
  console.log(`Promoted ${moduleId} <- ${targetSlug}`);
  console.log(`  canonical .v       : ${committedPath}`);
  console.log(`  archived previous  : ${archivedOriginalPath}`);
  console.log(
    `  metrics            : lut=${metrics.lut} dsp=${metrics.dsp} ` +
      `bram=${metrics.bram} latency_cycles=${metrics.latency_cycles}`,
  );
  console.log(
    `  vivado             : fmax=${vivadoReport.fmax_mhz?.toFixed?.(2) ?? "?"} ` +
      `MHz timing_met=${vivadoReport.timing_met}`,
  );
  console.log(
    `  verilator          : status=${verifResult.status} ` +
      `timing_pass=${verifResult.timing_pass} max_error=${verifResult.max_error}`,
  );

  // Touch the meta.json one more time to make sure the embedded source field
  // is the variant verbatim (commitReplacement writes module to meta.json,
  // which is correct, but defensively re-read+write to confirm parity).
  const metaPath = path.join(paths.rtlDir, `${moduleId}.meta.json`);
  const metaRaw = JSON.parse(await readFile(metaPath, "utf8")) as Record<string, unknown>;
  metaRaw.verilog_source = variantSource;
  await writeFile(metaPath, JSON.stringify(metaRaw, null, 2), "utf8");
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});
