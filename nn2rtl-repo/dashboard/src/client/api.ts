import type { NetworkId } from "../shared/networks";
import type {
  FileReadResult,
  JobAction,
  JobPreview,
  JobRecord,
  NetworkInfo,
  ProjectSnapshot,
} from "../shared/types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: {
      "content-type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(body.error ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

export function getSnapshot(networkId?: NetworkId): Promise<ProjectSnapshot> {
  const query = networkId ? `?network=${encodeURIComponent(networkId)}` : "";
  return request<ProjectSnapshot>(`/api/snapshot${query}`);
}

export function getNetworks(): Promise<{ defaultNetworkId: NetworkId; networks: NetworkInfo[] }> {
  return request<{ defaultNetworkId: NetworkId; networks: NetworkInfo[] }>("/api/networks");
}

export function readFile(path: string): Promise<FileReadResult> {
  return request<FileReadResult>(`/api/files?path=${encodeURIComponent(path)}`);
}

export function getJobs(): Promise<JobRecord[]> {
  return request<JobRecord[]>("/api/jobs");
}

export function getJobLog(id: string): Promise<FileReadResult> {
  return request<FileReadResult>(`/api/jobs/${encodeURIComponent(id)}/log`);
}

export function previewJob(action: JobAction): Promise<JobPreview> {
  return request<JobPreview>("/api/jobs/preview", {
    method: "POST",
    body: JSON.stringify({ action }),
  });
}

export function startJob(action: JobAction): Promise<JobRecord> {
  return request<JobRecord>("/api/jobs", {
    method: "POST",
    body: JSON.stringify({ action, confirmed: true }),
  });
}

export function stopJob(id: string): Promise<JobRecord> {
  return request<JobRecord>(`/api/jobs/${encodeURIComponent(id)}/stop`, {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export function promoteVariant(moduleId: string, targetSlug: string): Promise<JobRecord> {
  return request<JobRecord>("/api/actions/promote-variant", {
    method: "POST",
    body: JSON.stringify({ moduleId, targetSlug, confirmed: true }),
  });
}

export function archiveArtifact(relativePath: string): Promise<{ archivedPath: string }> {
  return request<{ archivedPath: string }>("/api/actions/archive-artifact", {
    method: "POST",
    body: JSON.stringify({ relativePath, confirmed: true }),
  });
}
