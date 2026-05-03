import { mkdir, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import {
  archivePathFor,
  isGeneratedKnowledgePath,
  readJson,
  repoRoot,
  resolveReadablePath,
  toRepoRelative,
} from "./paths.js";
import { startJob } from "./jobs.js";
import type { JobAction, JobRecord } from "../shared/types.js";

function stamp(): string {
  return new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
}

export async function promoteVariant(input: {
  moduleId: string;
  targetSlug: string;
  confirmed: boolean;
}): Promise<JobRecord> {
  const action: JobAction = {
    type: "promote-variant",
    moduleId: input.moduleId,
    targetSlug: input.targetSlug,
  };
  return startJob(action, input.confirmed);
}

export async function archiveArtifact(input: {
  relativePath: string;
  confirmed: boolean;
}): Promise<{ archivedPath: string }> {
  if (!input.confirmed) {
    throw new Error("Archive action requires confirmed=true.");
  }
  const normalized = input.relativePath.replace(/\\/g, "/");
  if (!isGeneratedKnowledgePath(normalized)) {
    throw new Error("Only generated active/probationary/improved knowledge artifacts can be archived.");
  }
  const source = resolveReadablePath(normalized);
  const archivedRel = archivePathFor(normalized, stamp());
  const destination = path.join(repoRoot, archivedRel);
  await mkdir(path.dirname(destination), { recursive: true });
  await rename(source, destination);

  const lifecyclePath = path.join(repoRoot, "knowledge", "doc_lifecycle.json");
  const lifecycle = await readJson<Record<string, unknown>>(lifecyclePath);
  if (lifecycle && typeof lifecycle === "object") {
    const docs = typeof lifecycle.docs === "object" && lifecycle.docs !== null
      ? lifecycle.docs as Record<string, Record<string, unknown>>
      : {};
    for (const doc of Object.values(docs)) {
      if (doc.pattern_path === normalized) {
        doc.pattern_path = archivedRel;
        doc.status = "archive";
        doc.archived_at = new Date().toISOString();
      }
      if (doc.reference_path === normalized) {
        doc.reference_path = archivedRel;
        doc.status = "archive";
        doc.archived_at = new Date().toISOString();
      }
    }
    await writeFile(lifecyclePath, `${JSON.stringify(lifecycle, null, 2)}\n`, "utf8");
  }

  return { archivedPath: toRepoRelative(destination) };
}
