// Top-level dashboard layout. Renders the top bar (network selector + status),
// tab nav, search/filter row, and the active panel. Everything heavy lives in
// `src/components/*.tsx`.

import { Boxes, Code2, FileText, GitCompare, RefreshCw, Search, StopCircle, Terminal } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import {
  archiveArtifact,
  getJobLog,
  getSnapshot,
  previewJob,
  readFile,
  startJob,
  stopJob,
} from "./client/api";
import { Commands } from "./components/Commands";
import { ArchiveModal, ConfirmModal, FileModal, ModuleDrawer } from "./components/modals";
import { Improvements, Jobs, Knowledge, ModulesTable, Overview } from "./components/panels";
import { NetworkSelector } from "./components/NetworkSelector";
import { DEFAULT_NETWORK_ID, type NetworkId } from "./shared/networks";
import type {
  DocSummary,
  FileReadResult,
  ImprovementReportSummary,
  JobAction,
  JobPreview,
  LayerSummary,
  ProjectSnapshot,
} from "./shared/types";

type Tab = "overview" | "modules" | "knowledge" | "improvements" | "commands" | "jobs";

function useSnapshot(networkId: NetworkId): {
  snapshot: ProjectSnapshot | null;
  loading: boolean;
  error: string | null;
  reload: () => Promise<void>;
} {
  const [snapshot, setSnapshot] = useState<ProjectSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  async function reload(): Promise<void> {
    setLoading(true);
    try {
      setSnapshot(await getSnapshot(networkId));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => {
    void reload();
    const timer = setInterval(() => void reload(), 10000);
    return () => clearInterval(timer);
    // Reloading on networkId change is the whole point of this hook.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [networkId]);
  return { snapshot, loading, error, reload };
}

export function App(): JSX.Element {
  const [networkId, setNetworkId] = useState<NetworkId>(DEFAULT_NETWORK_ID);
  const { snapshot, loading, error, reload } = useSnapshot(networkId);
  const [tab, setTab] = useState<Tab>("overview");
  const [query, setQuery] = useState("");
  const [opFilter, setOpFilter] = useState("all");
  const [stageFilter, setStageFilter] = useState("all");
  const [selected, setSelected] = useState<LayerSummary | null>(null);
  const [file, setFile] = useState<FileReadResult | null>(null);
  const [confirmPreview, setConfirmPreview] = useState<JobPreview | null>(null);
  const [archiveChoice, setArchiveChoice] = useState<{ doc: DocSummary; path: string } | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(null), 5000);
    return () => clearTimeout(timer);
  }, [notice]);

  const filteredModules = useMemo(() => {
    const text = query.toLowerCase();
    return (snapshot?.modules ?? []).filter((module) => {
      if (text && !module.moduleId.toLowerCase().includes(text)) return false;
      if (opFilter !== "all" && module.opType !== opFilter) return false;
      if (stageFilter !== "all" && module.stage !== stageFilter) return false;
      return true;
    });
  }, [snapshot, query, opFilter, stageFilter]);

  const opTypes = useMemo(
    () => [...new Set((snapshot?.modules ?? []).map((module) => module.opType))].sort(),
    [snapshot],
  );

  async function openFile(path: string): Promise<void> {
    setFile(await readFile(path));
  }

  async function preview(action: JobAction): Promise<void> {
    setConfirmPreview(await previewJob(action));
  }

  async function confirmJob(): Promise<void> {
    if (!confirmPreview) return;
    const record = await startJob(confirmPreview.action);
    setNotice(`Started ${record.title} · job ${record.id}`);
    setConfirmPreview(null);
    await reload();
    setTab("jobs");
  }

  async function promote(report: ImprovementReportSummary): Promise<void> {
    const previewRecord = await previewJob({ type: "promote-variant", moduleId: report.moduleId, targetSlug: report.targetSlug });
    setConfirmPreview(previewRecord);
  }

  async function archiveSelected(): Promise<void> {
    if (!archiveChoice) return;
    const result = await archiveArtifact(archiveChoice.path);
    setNotice(`Archived to ${result.archivedPath}`);
    setArchiveChoice(null);
    await reload();
  }

  async function stop(id: string): Promise<void> {
    await stopJob(id);
    setNotice("Stop requested");
    await reload();
  }

  async function openJobLog(id: string): Promise<void> {
    setFile(await getJobLog(id));
  }

  if (loading && !snapshot) {
    return <main className="shell"><div className="loading">Loading dashboard...</div></main>;
  }
  if (error && !snapshot) {
    return <main className="shell"><div className="error">{error}</div></main>;
  }
  if (!snapshot) {
    return <main className="shell"><div className="error">No snapshot</div></main>;
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="topbar-left">
          <div>
            <h1>nn2rtl Dashboard</h1>
            <p>{snapshot.modelName} · {snapshot.kpis.totalLayers} layers · <code>{snapshot.repoRoot}</code></p>
          </div>
          <NetworkSelector networks={snapshot.networks} value={snapshot.networkId} onChange={setNetworkId} />
        </div>
        <div className="top-actions">
          {(() => {
            const liveCount = snapshot.jobs.filter((job) => job.state === "running" || job.state === "stopping").length;
            const queuedCount = snapshot.jobs.filter((job) => job.state === "queued").length;
            if (liveCount === 0 && queuedCount === 0) return null;
            return (
              <button className="live-pill" onClick={() => setTab("jobs")} title="Open Jobs tab">
                <i className="dot live" />
                {liveCount} running{queuedCount ? ` · ${queuedCount} queued` : ""}
              </button>
            );
          })()}
          {notice && <span className="notice">{notice}</span>}
          <button className="secondary-button" onClick={() => void reload()}><RefreshCw size={16} /> refresh</button>
        </div>
      </header>
      <nav className="tabs">
        {([
          ["overview", Boxes],
          ["modules", Code2],
          ["knowledge", FileText],
          ["improvements", GitCompare],
          ["commands", Terminal],
          ["jobs", StopCircle],
        ] as const).map(([name, Icon]) => (
          <button key={name} className={tab === name ? "active" : ""} onClick={() => setTab(name)}>
            <Icon size={16} /> {name === "commands" ? "tasks" : name}
          </button>
        ))}
      </nav>
      <section className="filters">
        <div className="searchbox">
          <Search size={16} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="module" />
        </div>
        <select value={opFilter} onChange={(event) => setOpFilter(event.target.value)} aria-label="Filter by op">
          <option value="all">all ops</option>
          {opTypes.map((op) => <option key={op} value={op}>{op}</option>)}
        </select>
        <select value={stageFilter} onChange={(event) => setStageFilter(event.target.value)} aria-label="Filter by stage">
          <option value="all">all stages</option>
          {["missing", "rtl", "verilator-pass", "vivado-pass", "failed", "improved"].map((stage) => (
            <option key={stage} value={stage}>{stage}</option>
          ))}
        </select>
      </section>
      {error && <div className="error inline">{error}</div>}
      {tab === "overview" && <Overview snapshot={snapshot} filteredModules={filteredModules} setSelected={setSelected} />}
      {tab === "modules" && <ModulesTable modules={filteredModules} onSelect={setSelected} />}
      {tab === "knowledge" && <Knowledge docs={snapshot.docs} onOpenFile={(p) => void openFile(p)} onArchive={(doc, path) => setArchiveChoice({ doc, path })} />}
      {tab === "improvements" && <Improvements reports={snapshot.improvements} onOpenFile={(p) => void openFile(p)} onPromote={(report) => void promote(report)} />}
      {tab === "commands" && <Commands modules={snapshot.modules} networks={snapshot.networks} networkId={snapshot.networkId} onPreview={(action) => void preview(action)} />}
      {tab === "jobs" && <Jobs jobs={snapshot.jobs} onStop={(id) => void stop(id)} onOpenLog={(id) => void openJobLog(id)} />}
      <ModuleDrawer module={selected} onClose={() => setSelected(null)} onOpenFile={(p) => void openFile(p)} onPromote={(report) => void promote(report)} />
      <FileModal file={file} onClose={() => setFile(null)} />
      <ConfirmModal preview={confirmPreview} onCancel={() => setConfirmPreview(null)} onConfirm={() => void confirmJob()} />
      <ArchiveModal
        doc={archiveChoice?.doc ?? null}
        pathToArchive={archiveChoice?.path ?? null}
        onCancel={() => setArchiveChoice(null)}
        onConfirm={() => void archiveSelected()}
      />
    </main>
  );
}
