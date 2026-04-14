#!/usr/bin/env node
// Enforce the sdk/ ↔ mcp/ "twin file" invariant.
//
// Why: the MCP server lives in its own TypeScript package (`mcp/`) with a
// separate rootDir from `sdk/`, so it cannot import SDK types or schemas.
// The two packages therefore keep local copies of the shared data contracts
// (LayerIR, VerifResult, VerilogModule, ...). Without a check, those copies
// can drift silently — for example if someone widens a type on one side but
// forgets the other. This script fails the build if they do.
//
// Enforcement:
//   1. `sdk/types.ts` and `mcp/types.ts` must be byte-identical (modulo line
//      endings). Every exported type is shared, so a full equality check is
//      the simplest correct rule.
//   2. `sdk/schemas.ts` and `mcp/schemas.ts` intentionally diverge — only
//      `mcp/` has per-tool input/output schemas; only `sdk/` has pipeline
//      state + synthesis report schemas. But the *shared* Zod exports
//      (failureClass, layerIr, pipelineIr, verilogModule, verifResult) must
//      be byte-identical. We extract each by name and diff individually.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function normalize(text) {
  // Normalize CRLF -> LF so Windows checkouts with autocrlf don't flag spurious diffs.
  return text.replace(/\r\n/g, "\n");
}

function readNormalized(relPath) {
  return normalize(readFileSync(path.join(repoRoot, relPath), "utf8"));
}

const failures = [];

// --- types.ts: full byte equality ---------------------------------------
{
  const sdk = readNormalized("sdk/types.ts");
  const mcp = readNormalized("mcp/types.ts");
  if (sdk !== mcp) {
    failures.push(
      "sdk/types.ts and mcp/types.ts have diverged. These files are twins — " +
      "every exported type is shared. Re-sync by copying one over the other " +
      "and resolve the intent separately.",
    );
  }
}

// --- schemas.ts: shared named exports must match ------------------------
// Match `export const NAME = ...;` at top level, where the value may span
// multiple lines (e.g. `.object({...}).strict()`). We anchor on a line that
// begins with `export const` and terminate at the next top-level `export`
// or end-of-file. The shared names are listed explicitly so a new sdk-only
// or mcp-only export doesn't get flagged.
const SHARED_SCHEMA_EXPORTS = [
  "failureClassSchema",
  "layerIrSchema",
  "pipelineIrSchema",
  "verilogModuleSchema",
  "verifResultSchema",
];

function extractExport(source, name) {
  const lines = source.split("\n");
  const startIdx = lines.findIndex((line) => line.startsWith(`export const ${name} `) || line.startsWith(`export const ${name}=`));
  if (startIdx === -1) {
    return null;
  }
  const collected = [lines[startIdx]];
  for (let i = startIdx + 1; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith("export ") || line.startsWith("// ")) {
      break;
    }
    collected.push(line);
  }
  // Trim trailing blank lines from the block.
  while (collected.length > 0 && collected[collected.length - 1].trim() === "") {
    collected.pop();
  }
  return collected.join("\n");
}

{
  const sdkSource = readNormalized("sdk/schemas.ts");
  const mcpSource = readNormalized("mcp/schemas.ts");

  for (const name of SHARED_SCHEMA_EXPORTS) {
    const sdkBlock = extractExport(sdkSource, name);
    const mcpBlock = extractExport(mcpSource, name);

    if (sdkBlock === null) {
      failures.push(`sdk/schemas.ts is missing the shared export '${name}'.`);
      continue;
    }
    if (mcpBlock === null) {
      failures.push(`mcp/schemas.ts is missing the shared export '${name}'.`);
      continue;
    }
    if (sdkBlock !== mcpBlock) {
      failures.push(
        `Shared schema '${name}' differs between sdk/schemas.ts and mcp/schemas.ts.\n` +
        `--- sdk/schemas.ts ---\n${sdkBlock}\n--- mcp/schemas.ts ---\n${mcpBlock}`,
      );
    }
  }
}

if (failures.length > 0) {
  console.error("check-twins: shared-contract drift detected\n");
  for (const failure of failures) {
    console.error(`* ${failure}\n`);
  }
  process.exit(1);
}

console.log("check-twins: sdk/ and mcp/ shared contracts are in sync.");
