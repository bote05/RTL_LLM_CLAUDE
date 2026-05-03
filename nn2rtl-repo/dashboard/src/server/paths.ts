import { access, mkdir, readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const dashboardRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
export const repoRoot = path.resolve(dashboardRoot, "..");
export const outputDashboardDir = path.join(repoRoot, "output", "dashboard");
export const jobsDir = path.join(outputDashboardDir, "jobs");
export const jobsLogPath = path.join(outputDashboardDir, "jobs.jsonl");

export async function ensureDashboardDirs(): Promise<void> {
  await mkdir(jobsDir, { recursive: true });
}

export async function pathExists(filePath: string): Promise<boolean> {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

export async function readJson<T = unknown>(filePath: string): Promise<T | undefined> {
  try {
    return JSON.parse(await readFile(filePath, "utf8")) as T;
  } catch (error: unknown) {
    if (
      typeof error === "object" &&
      error !== null &&
      "code" in error &&
      (error as { code?: string }).code === "ENOENT"
    ) {
      return undefined;
    }
    return undefined;
  }
}

export async function listFilesRecursive(root: string): Promise<string[]> {
  if (!(await pathExists(root))) return [];
  const out: string[] = [];
  async function walk(dir: string): Promise<void> {
    const entries = await readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        await walk(full);
      } else if (entry.isFile()) {
        out.push(full);
      }
    }
  }
  await walk(root);
  return out.sort();
}

export function toRepoRelative(filePath: string): string {
  return path.relative(repoRoot, filePath).split(path.sep).join("/");
}

const allowedReadRoots = [
  "output/rtl",
  "output/reports",
  "output/improve",
  "output/dashboard",
  "knowledge/patterns",
  "knowledge/references",
].map((part) => path.resolve(repoRoot, part));

const allowedReadFiles = [
  "output/layer_ir.json",
  "output/pipeline_state.json",
  "output/reports/pipeline_summary.json",
  "output/reports/run_log.jsonl",
  "output/reports/agent_tool_use.jsonl",
  "output/reports/tool_calls.jsonl",
  "knowledge/doc_lifecycle.json",
].map((part) => path.resolve(repoRoot, part));

export function resolveReadablePath(relativePath: string): string {
  const normalized = relativePath.replace(/\\/g, "/").replace(/^\/+/, "");
  const resolved = path.resolve(repoRoot, normalized);
  if (!resolved.startsWith(repoRoot + path.sep) && resolved !== repoRoot) {
    throw new Error("Path escapes the repository.");
  }
  const allowed =
    allowedReadFiles.includes(resolved) ||
    allowedReadRoots.some((root) => resolved === root || resolved.startsWith(root + path.sep));
  if (!allowed) {
    throw new Error("Path is outside dashboard read allowlist.");
  }
  return resolved;
}

export async function readAllowlistedFile(relativePath: string, maxBytes = 1_000_000): Promise<{
  path: string;
  content: string;
  sizeBytes: number;
  truncated: boolean;
}> {
  const resolved = resolveReadablePath(relativePath);
  const info = await stat(resolved);
  const content = await readFile(resolved, "utf8");
  const truncated = Buffer.byteLength(content, "utf8") > maxBytes;
  return {
    path: toRepoRelative(resolved),
    content: truncated ? content.slice(0, maxBytes) : content,
    sizeBytes: info.size,
    truncated,
  };
}

export function isGeneratedKnowledgePath(relativePath: string): boolean {
  const normalized = relativePath.replace(/\\/g, "/");
  return /^knowledge\/(patterns|references)\/(active|probationary|improved)\//.test(normalized);
}

export function archivePathFor(relativePath: string, stamp: string): string {
  const normalized = relativePath.replace(/\\/g, "/");
  const parts = normalized.split("/");
  if (parts.length < 4) {
    throw new Error("Cannot archive path outside knowledge tier.");
  }
  parts[2] = "archive";
  const file = parts.pop() ?? "artifact";
  const ext = path.extname(file);
  const base = file.slice(0, file.length - ext.length);
  parts.push(`${base}__archived_${stamp}${ext}`);
  return parts.join("/");
}
