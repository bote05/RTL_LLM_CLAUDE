import { readFile } from "node:fs/promises";
import http, { type IncomingMessage, type ServerResponse } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { archiveArtifact, promoteVariant } from "./actions.js";
import { previewJob, readJobs, reconcilePersistedJobsAfterRestart, startJob, stopJob } from "./jobs.js";
import {
  dashboardRoot,
  ensureDashboardDirs,
  readAllowlistedFile,
  repoRoot,
  resolveReadablePath,
} from "./paths.js";
import { buildSnapshot } from "./snapshot.js";
import type { JobAction } from "../shared/types.js";

const HOST = "127.0.0.1";
const PORT = Number(process.env.NN2RTL_DASHBOARD_PORT ?? 4177);

function sendJson(res: ServerResponse, value: unknown, status = 200): void {
  const body = `${JSON.stringify(value, null, 2)}\n`;
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
  });
  res.end(body);
}

function sendError(res: ServerResponse, error: unknown, status = 500): void {
  sendJson(res, { error: error instanceof Error ? error.message : String(error) }, status);
}

async function readBody(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  if (chunks.length === 0) return {};
  return JSON.parse(Buffer.concat(chunks).toString("utf8")) as unknown;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function parseAction(value: unknown): JobAction {
  if (!isRecord(value) || typeof value.type !== "string") {
    throw new Error("Missing job action type.");
  }
  return value as JobAction;
}

function contentType(filePath: string): string {
  if (filePath.endsWith(".html")) return "text/html; charset=utf-8";
  if (filePath.endsWith(".js")) return "text/javascript; charset=utf-8";
  if (filePath.endsWith(".css")) return "text/css; charset=utf-8";
  if (filePath.endsWith(".svg")) return "image/svg+xml";
  return "application/octet-stream";
}

async function serveStatic(req: IncomingMessage, res: ServerResponse, url: URL): Promise<void> {
  const distRoot = path.join(dashboardRoot, "dist");
  const requested = url.pathname === "/" ? "/index.html" : url.pathname;
  const resolved = path.resolve(distRoot, `.${requested}`);
  const filePath = resolved.startsWith(distRoot + path.sep) ? resolved : path.join(distRoot, "index.html");
  try {
    const content = await readFile(filePath);
    res.writeHead(200, { "content-type": contentType(filePath) });
    res.end(content);
  } catch {
    try {
      const index = await readFile(path.join(distRoot, "index.html"));
      res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
      res.end(index);
    } catch {
      sendJson(res, {
        message: "nn2rtl dashboard API is running. Start Vite with `npm --prefix dashboard run dev` for the UI.",
      });
    }
  }
  void req;
}

async function handleApi(req: IncomingMessage, res: ServerResponse, url: URL): Promise<void> {
  if (req.method === "GET" && url.pathname === "/api/snapshot") {
    sendJson(res, await buildSnapshot());
    return;
  }

  if (req.method === "GET" && url.pathname === "/api/files") {
    const rel = url.searchParams.get("path");
    if (!rel) throw new Error("Missing path query parameter.");
    sendJson(res, await readAllowlistedFile(rel));
    return;
  }

  if (req.method === "GET" && url.pathname === "/api/jobs") {
    sendJson(res, await readJobs());
    return;
  }

  const logMatch = url.pathname.match(/^\/api\/jobs\/([^/]+)\/log$/);
  if (req.method === "GET" && logMatch) {
    const jobs = await readJobs();
    const job = jobs.find((candidate) => candidate.id === logMatch[1]);
    if (!job) {
      sendError(res, new Error("Job not found."), 404);
      return;
    }
    sendJson(res, await readAllowlistedFile(job.logPath, 2_000_000));
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/jobs/preview") {
    const body = await readBody(req);
    const action = parseAction(isRecord(body) ? body.action ?? body : body);
    sendJson(res, previewJob(action));
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/jobs") {
    const body = await readBody(req);
    if (!isRecord(body)) throw new Error("Invalid job body.");
    sendJson(res, await startJob(parseAction(body.action), body.confirmed === true), 201);
    return;
  }

  const stopMatch = url.pathname.match(/^\/api\/jobs\/([^/]+)\/stop$/);
  if (req.method === "POST" && stopMatch) {
    sendJson(res, await stopJob(stopMatch[1]));
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/actions/promote-variant") {
    const body = await readBody(req);
    if (!isRecord(body)) throw new Error("Invalid promote body.");
    sendJson(res, await promoteVariant({
      moduleId: String(body.moduleId ?? ""),
      targetSlug: String(body.targetSlug ?? ""),
      confirmed: body.confirmed === true,
    }), 201);
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/actions/archive-artifact") {
    const body = await readBody(req);
    if (!isRecord(body)) throw new Error("Invalid archive body.");
    sendJson(res, await archiveArtifact({
      relativePath: String(body.relativePath ?? ""),
      confirmed: body.confirmed === true,
    }));
    return;
  }

  sendError(res, new Error("Route not found."), 404);
}

export function createDashboardServer(): http.Server {
  return http.createServer((req, res) => {
    void (async () => {
      const url = new URL(req.url ?? "/", `http://${req.headers.host ?? `${HOST}:${PORT}`}`);
      if (url.pathname.startsWith("/api/")) {
        await handleApi(req, res, url);
      } else {
        await serveStatic(req, res, url);
      }
    })().catch((error) => {
      sendError(res, error, error instanceof Error && /not found/i.test(error.message) ? 404 : 500);
    });
  });
}

const currentFile = fileURLToPath(import.meta.url);

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(currentFile)) {
  await ensureDashboardDirs();
  await reconcilePersistedJobsAfterRestart();
  // Validate the main read allowlist path once at boot so path bugs are loud.
  resolveReadablePath("output/layer_ir.json");
  createDashboardServer().listen(PORT, HOST, () => {
    console.log(`nn2rtl dashboard API listening on http://${HOST}:${PORT}`);
    console.log(`repo: ${repoRoot}`);
  });
}
