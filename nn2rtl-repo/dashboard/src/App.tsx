import {
  Archive,
  Boxes,
  CheckCircle2,
  CircleStop,
  Code2,
  Eye,
  FileText,
  Gauge,
  GitCompare,
  Hammer,
  Play,
  RefreshCw,
  Rocket,
  Search,
  ShieldCheck,
  StopCircle,
  Terminal,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  archiveArtifact,
  getJobLog,
  getSnapshot,
  previewJob,
  promoteVariant,
  readFile,
  startJob,
  stopJob,
} from "./client/api";
import type {
  DocSummary,
  FileReadResult,
  ImprovementReportSummary,
  ImprovementTarget,
  JobAction,
  JobPreview,
  JobRecord,
  LayerSummary,
  ProjectSnapshot,
} from "./shared/types";

type Tab = "overview" | "modules" | "knowledge" | "improvements" | "commands" | "jobs";

const targets: ImprovementTarget[] = [
  "use-dsp",
  "use-bram",
  "reduce-lut",
  "reduce-latency",
  "increase-throughput",
];

function classNames(...values: Array<string | false | undefined>): string {
  return values.filter(Boolean).join(" ");
}

function fmtNumber(value: number | undefined, digits = 0): string {
  if (value === undefined || !Number.isFinite(value)) return "n/a";
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function fmtCost(value: number): string {
  return `$${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
}

function fmtBytes(bytes: number | undefined): string {
  if (bytes === undefined || !Number.isFinite(bytes) || bytes <= 0) return "n/a";
  const units = ["B", "KiB", "MiB", "GiB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const digits = unit === 0 || value >= 100 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(digits)} ${units[unit]}`;
}

function useSnapshot(): {
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
      setSnapshot(await getSnapshot());
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
  }, []);
  return { snapshot, loading, error, reload };
}

function KpiCard({ label, value, icon: Icon }: {
  label: string;
  value: string | number;
  icon: typeof Boxes;
}): JSX.Element {
  return (
    <div className="kpi">
      <Icon size={18} />
      <div>
        <strong>{value}</strong>
        <span>{label}</span>
      </div>
    </div>
  );
}

function Heatmap({ modules, onSelect }: {
  modules: LayerSummary[];
  onSelect: (module: LayerSummary) => void;
}): JSX.Element {
  return (
    <div className="heatmap" aria-label="module status heatmap">
      {modules.map((module) => (
        <button
          key={module.moduleId}
          className={`heat-cell ${module.stage}`}
          title={`${module.index + 1}. ${module.moduleId} · ${module.stage}`}
          onClick={() => onSelect(module)}
        >
          <span>{module.index + 1}</span>
        </button>
      ))}
    </div>
  );
}

// Bin layers into ResNet-style stage bands by the *running-max* channel count
// up to that point. Per-layer max would oscillate inside bottlenecks (64→256
// reduce/expand) and produce dozens of tiny same-band runs; running-max gives
// 5 monotonically-growing regions (stem → stage1 → stage2 → stage3 → stage4)
// matching how people actually think about CNN depth.
function annotateBands(
  modules: LayerSummary[],
): Array<{ module: LayerSummary; band: number }> {
  let runningMax = 0;
  return modules.map((module) => {
    const inC = module.inputShape.length > 1 ? module.inputShape[1] : 0;
    const outC = module.outputShape.length > 1 ? module.outputShape[1] : 0;
    runningMax = Math.max(runningMax, inC, outC);
    let band = 1;
    if (runningMax > 64) band = 2;
    if (runningMax > 256) band = 3;
    if (runningMax > 512) band = 4;
    if (runningMax > 1024) band = 5;
    return { module, band };
  });
}

const STAGE_BAND_LABELS = ["stem", "stage 1", "stage 2", "stage 3", "stage 4"];

function Skyline({ modules, onSelect }: {
  modules: LayerSummary[];
  onSelect: (module: LayerSummary) => void;
}): JSX.Element {
  const annotated = useMemo(() => annotateBands(modules), [modules]);

  // With running-max bands, regions are guaranteed monotonic and contiguous —
  // exactly five (or fewer) regions covering the whole strip, no oscillation.
  const regions = useMemo(() => {
    const groups: Array<{ band: number; start: number; end: number }> = [];
    for (let i = 0; i < annotated.length; i += 1) {
      const item = annotated[i];
      const last = groups[groups.length - 1];
      if (last && last.band === item.band) {
        last.end = i;
      } else {
        groups.push({ band: item.band, start: i, end: i });
      }
    }
    return groups;
  }, [annotated]);

  // Range of log-weights *only over weight-bearing layers* — gives convs a
  // wide spread (smallest conv -> tallest stem), and pushes weightless ops
  // (relu/add) down to the baseline so the silhouette actually shows
  // architecture rather than collapsing every bar into the upper register.
  const range = useMemo(() => {
    const positives = annotated
      .map((item) => item.module.numWeights)
      .filter((value) => value > 1);
    if (positives.length === 0) return { min: 0, max: 1 };
    return {
      min: Math.log10(Math.min(...positives)),
      max: Math.log10(Math.max(...positives)),
    };
  }, [annotated]);

  if (annotated.length === 0) {
    return <div className="skyline-empty muted">No layers in current filter.</div>;
  }

  const total = annotated.length;

  function heightPct(numWeights: number): number {
    if (numWeights <= 1) return 6; // weightless ops: thin stub
    if (range.max <= range.min) return 80;
    const log = Math.log10(numWeights);
    const norm = (log - range.min) / (range.max - range.min);
    return 12 + norm * 80; // 12% baseline → 92% peak
  }

  return (
    <div className="skyline" aria-label="network coverage skyline">
      <div className="skyline-canvas">
        {regions.map((region, idx) => {
          const startPct = (region.start / total) * 100;
          const endPct = ((region.end + 1) / total) * 100;
          const hue = (region.band - 1) * 65 + 165;
          return (
            <div
              key={`band-${idx}`}
              className="skyline-band"
              style={{
                left: `${startPct}%`,
                width: `${endPct - startPct}%`,
                background: `linear-gradient(to bottom, hsl(${hue}, 55%, 50%, ${idx % 2 === 0 ? 0.05 : 0.1}), hsl(${hue}, 55%, 50%, ${idx % 2 === 0 ? 0.02 : 0.05}))`,
              }}
            >
              <span className="skyline-band-label">
                {STAGE_BAND_LABELS[region.band - 1]}
              </span>
            </div>
          );
        })}
        {annotated.map((item, idx) => {
          const leftPct = (idx / total) * 100;
          const widthPct = 100 / total;
          const h = heightPct(item.module.numWeights);
          const isLive =
            item.module.pipelineStatus &&
            item.module.pipelineStatus !== "pass" &&
            item.module.pipelineStatus !== "fail_abort";
          const tooltip = `${item.module.index + 1}. ${item.module.moduleId}\n${item.module.opType} · ${item.module.outputShape.join("x")}\nweights: ${item.module.numWeights.toLocaleString()} bytes\nstage: ${item.module.stage}${item.module.pipelineStatus ? ` · ${item.module.pipelineStatus}` : ""}`;
          return (
            <button
              key={item.module.moduleId}
              className={`skyline-bar ${item.module.stage}${isLive ? " live" : ""}`}
              style={{
                left: `${leftPct}%`,
                width: `calc(${widthPct}% - 1px)`,
                height: `${h}%`,
              }}
              onClick={() => onSelect(item.module)}
              title={tooltip}
            >
              <span className={`op-tick op-${item.module.opType}`} />
            </button>
          );
        })}
        <div className="skyline-baseline" />
        {[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
          .filter((n) => n <= total)
          .map((n) => (
            <span
              key={`tick-${n}`}
              className="skyline-tick"
              style={{ left: `${((n - 1) / total) * 100}%` }}
            >{n}</span>
          ))}
      </div>
      <div className="skyline-legend muted">
        <span>op:</span>
        <span><i className="op-dot op-conv2d" /> conv</span>
        <span><i className="op-dot op-add" /> add</span>
        <span><i className="op-dot op-relu" /> relu</span>
        <span><i className="op-dot op-maxpool" /> pool</span>
        <span className="skyline-axis">bar height = log₁₀(weight bytes), spread over the per-conv min/max</span>
      </div>
    </div>
  );
}

type TreemapBox = {
  module: LayerSummary;
  x: number;
  y: number;
  w: number;
  h: number;
};

// Standard squarified treemap (Bruls, Huizing, van Wijk 2000): keeps cell
// aspect ratios as close to 1 as possible by greedily filling rows along the
// shortest side of the remaining rectangle. Much more readable than the
// previous "sort + chunk into N rows" heuristic.
function squarifiedTreemap(
  modules: LayerSummary[],
  width: number,
  height: number,
): TreemapBox[] {
  const sorted = [...modules].sort((a, b) => b.numWeights - a.numWeights);
  const totalWeight = sorted.reduce((sum, module) => sum + Math.max(1, module.numWeights), 0);
  if (totalWeight === 0) return [];
  const totalArea = width * height;
  const items = sorted.map((module) => ({ module, area: (Math.max(1, module.numWeights) / totalWeight) * totalArea }));

  const out: TreemapBox[] = [];
  type Rect = { x: number; y: number; w: number; h: number };

  function worst(row: { area: number }[], side: number): number {
    if (row.length === 0) return Infinity;
    const total = row.reduce((sum, item) => sum + item.area, 0);
    if (total === 0) return Infinity;
    const max = Math.max(...row.map((item) => item.area));
    const min = Math.min(...row.map((item) => item.area));
    return Math.max(
      (side * side * max) / (total * total),
      (total * total) / (side * side * min),
    );
  }

  function layoutRow(
    row: { module: LayerSummary; area: number }[],
    rect: Rect,
  ): Rect {
    const total = row.reduce((sum, item) => sum + item.area, 0);
    if (total === 0) return rect;
    if (rect.w >= rect.h) {
      const rowW = total / rect.h;
      let y = rect.y;
      for (const item of row) {
        const itemH = item.area / rowW;
        out.push({ module: item.module, x: rect.x, y, w: rowW, h: itemH });
        y += itemH;
      }
      return { x: rect.x + rowW, y: rect.y, w: rect.w - rowW, h: rect.h };
    }
    const rowH = total / rect.w;
    let x = rect.x;
    for (const item of row) {
      const itemW = item.area / rowH;
      out.push({ module: item.module, x, y: rect.y, w: itemW, h: rowH });
      x += itemW;
    }
    return { x: rect.x, y: rect.y + rowH, w: rect.w, h: rect.h - rowH };
  }

  function squarify(
    remaining: { module: LayerSummary; area: number }[],
    row: { module: LayerSummary; area: number }[],
    rect: Rect,
  ): void {
    if (remaining.length === 0) {
      if (row.length > 0) layoutRow(row, rect);
      return;
    }
    const next = remaining[0];
    const candidate = [...row, next];
    const side = Math.min(rect.w, rect.h);
    if (row.length === 0 || worst(candidate, side) <= worst(row, side)) {
      squarify(remaining.slice(1), candidate, rect);
    } else {
      const newRect = layoutRow(row, rect);
      squarify(remaining, [], newRect);
    }
  }

  squarify(items, [], { x: 0, y: 0, w: width, h: height });
  return out;
}

function Treemap({ modules, onSelect }: {
  modules: LayerSummary[];
  onSelect: (module: LayerSummary) => void;
}): JSX.Element {
  // Use a fixed virtual canvas — actual sizing happens via CSS percentages,
  // so absolute coordinate units don't matter as long as the ratio matches
  // the CSS canvas (16:9 reads well at desktop widths).
  const VIRTUAL_W = 1600;
  const VIRTUAL_H = 900;
  const boxes = useMemo(
    () => squarifiedTreemap(modules, VIRTUAL_W, VIRTUAL_H),
    [modules],
  );

  if (boxes.length === 0) {
    return <div className="skyline-empty muted">No layers in current filter.</div>;
  }

  return (
    <div className="treemap">
      {boxes.map((box) => {
        const leftPct = (box.x / VIRTUAL_W) * 100;
        const topPct = (box.y / VIRTUAL_H) * 100;
        const widthPct = (box.w / VIRTUAL_W) * 100;
        const heightPct = (box.h / VIRTUAL_H) * 100;
        // Hide labels for cells too small to fit text comfortably — keeps the
        // visualization legible at the long tail without losing the cell.
        const showLabel = box.w >= 60 && box.h >= 28;
        const showDetail = box.w >= 80 && box.h >= 44;
        return (
          <button
            key={box.module.moduleId}
            className={`treemap-cell ${box.module.stage}`}
            style={{
              left: `${leftPct}%`,
              top: `${topPct}%`,
              width: `${widthPct}%`,
              height: `${heightPct}%`,
            }}
            onClick={() => onSelect(box.module)}
            title={`${box.module.index + 1}. ${box.module.moduleId}\n${box.module.opType} · ${box.module.outputShape.join("x")}\nweights: ${box.module.numWeights.toLocaleString()} bytes\nstage: ${box.module.stage}`}
          >
            {showLabel && (
              <span className="treemap-id">{box.module.moduleId.replace(/^node_conv_/, "c")}</span>
            )}
            {showDetail && (
              <span className="treemap-detail">
                {box.module.numWeights >= 1024 * 1024
                  ? `${(box.module.numWeights / 1024 / 1024).toFixed(1)} MiB`
                  : box.module.numWeights >= 1024
                    ? `${(box.module.numWeights / 1024).toFixed(0)} KiB`
                    : `${box.module.numWeights} B`}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}

function ModuleDrawer({ module, onClose, onOpenFile, onPromote }: {
  module: LayerSummary | null;
  onClose: () => void;
  onOpenFile: (path: string) => void;
  onPromote: (report: ImprovementReportSummary) => void;
}): JSX.Element | null {
  if (!module) return null;
  return (
    <aside className="drawer">
      <div className="drawer-head">
        <div>
          <h2>{module.moduleId}</h2>
          <p>{module.opType} · {module.contractId} · {module.stage}</p>
        </div>
        <button className="icon-button" onClick={onClose} title="Close">
          <X size={18} />
        </button>
      </div>
      <div className="metric-grid">
        <span>Input</span><strong>{module.inputShape.join("x") || "n/a"}</strong>
        <span>Output</span><strong>{module.outputShape.join("x") || "n/a"}</strong>
        <span>Weights</span><strong>{module.weightShape.join("x") || "n/a"} <em>({fmtBytes(module.numWeights)})</em></strong>
        <span>Latency</span><strong>{fmtNumber(module.pipelineLatencyCycles)} cyc</strong>
        <span>Contract</span><strong>{module.contractId} <em>· {module.ioMode}</em></strong>
        <span>Pipeline</span><strong>{module.pipelineStatus ?? "—"}{module.pipelineAttempts !== undefined ? ` · ${module.pipelineAttempts} attempts` : ""}</strong>
        <span>Verilator</span><strong>{module.verif?.status ?? "missing"}{module.verif?.timingPass !== undefined ? ` · timing ${module.verif.timingPass ? "ok" : "fail"}` : ""}</strong>
        <span>Vivado</span><strong>{module.vivado?.success ? "pass" : module.vivado ? "fail" : "missing"}{module.vivado?.timingMet !== undefined ? ` · timing_met=${module.vivado.timingMet}` : ""}</strong>
        <span>LUT / FF</span><strong>{fmtNumber(module.vivado?.lut)} / {fmtNumber(module.vivado?.ff)}</strong>
        <span>DSP / BRAM</span><strong>{fmtNumber(module.vivado?.dsp)} / {fmtNumber(module.vivado?.bram)}</strong>
        <span>Fmax / WNS</span><strong>{fmtNumber(module.vivado?.fmaxMhz, 2)} MHz · {module.vivado?.setupWnsNs !== undefined && module.vivado?.setupWnsNs !== null ? `${module.vivado.setupWnsNs.toFixed(2)} ns` : "n/a"}</strong>
      </div>
      <h3>Artifacts</h3>
      <div className="button-row">
        {Object.entries(module.paths).map(([label, file]) => file ? (
          <button key={label} onClick={() => onOpenFile(file)} className="secondary-button">
            <Eye size={15} /> {label}
          </button>
        ) : null)}
      </div>
      <h3>Improvements</h3>
      <div className="stack">
        {module.improvements.length === 0 && <p className="muted">No variants</p>}
        {module.improvements.map((report) => (
          <div className="row-card" key={report.reportPath}>
            <div>
              <strong>{report.targets.join(", ")}</strong>
              <span>{report.success ? "verified" : "failed"} · {report.finalAction}</span>
            </div>
            <div className="button-row">
              <button className="secondary-button" onClick={() => onOpenFile(report.reportPath)}>
                <FileText size={15} /> report
              </button>
              {report.success && report.finalAction === "kept-as-variant" && (
                <button className="danger-button" onClick={() => onPromote(report)}>
                  <Rocket size={15} /> promote
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
      <h3>Knowledge</h3>
      <div className="stack">
        {module.docs.length === 0 && <p className="muted">No linked docs</p>}
        {module.docs.map((doc) => (
          <div className="row-card" key={doc.id}>
            <div>
              <strong>{doc.id}</strong>
              <span>{doc.tier} · {doc.opType ?? "any op"}</span>
            </div>
            <div className="button-row">
              {doc.patternPath && <button className="secondary-button" onClick={() => onOpenFile(doc.patternPath!)}>pattern</button>}
              {doc.referencePath && <button className="secondary-button" onClick={() => onOpenFile(doc.referencePath!)}>reference</button>}
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}

function FileModal({ file, onClose }: {
  file: FileReadResult | null;
  onClose: () => void;
}): JSX.Element | null {
  if (!file) return null;
  return (
    <div className="modal-backdrop">
      <div className="modal wide">
        <div className="modal-head">
          <h2>{file.path}</h2>
          <button className="icon-button" onClick={onClose} title="Close">
            <X size={18} />
          </button>
        </div>
        <pre className="file-viewer">{file.content}</pre>
        <div className="modal-foot">
          <span>{fmtNumber(file.sizeBytes)} bytes{file.truncated ? " · truncated" : ""}</span>
        </div>
      </div>
    </div>
  );
}

function ConfirmModal({ preview, onCancel, onConfirm }: {
  preview: JobPreview | null;
  onCancel: () => void;
  onConfirm: () => void;
}): JSX.Element | null {
  if (!preview) return null;
  return (
    <div className="modal-backdrop">
      <div className="modal">
        <div className="modal-head">
          <h2>{preview.title}</h2>
          <button className="icon-button" onClick={onCancel} title="Cancel">
            <X size={18} />
          </button>
        </div>
        <div className="confirm-grid">
          <span>Command</span><code>{preview.command}</code>
          <span>Working dir</span><code>{preview.cwd}</code>
          <span>Writes</span><strong>{preview.writes.join(", ")}</strong>
          <span>Cost risk</span><strong>{preview.costRisk}</strong>
          <span>Canonical RTL risk</span><strong>{preview.canonicalRisk ? "yes" : "no"}</strong>
          <span>Stop</span><strong>{preview.stopWarning}</strong>
        </div>
        <div className="modal-actions">
          <button className="secondary-button" onClick={onCancel}>Cancel</button>
          <button className={preview.canonicalRisk ? "danger-button" : "primary-button"} onClick={onConfirm}>
            <Play size={16} /> Start
          </button>
        </div>
      </div>
    </div>
  );
}

function ArchiveModal({ doc, pathToArchive, onCancel, onConfirm }: {
  doc: DocSummary | null;
  pathToArchive: string | null;
  onCancel: () => void;
  onConfirm: () => void;
}): JSX.Element | null {
  if (!doc || !pathToArchive) return null;
  return (
    <div className="modal-backdrop">
      <div className="modal">
        <div className="modal-head">
          <h2>Archive Artifact</h2>
          <button className="icon-button" onClick={onCancel} title="Cancel"><X size={18} /></button>
        </div>
        <div className="confirm-grid">
          <span>Doc</span><strong>{doc.id}</strong>
          <span>Tier</span><strong>{doc.tier}</strong>
          <span>Path</span><code>{pathToArchive}</code>
          <span>Writes</span><strong>knowledge/*/archive and knowledge/doc_lifecycle.json</strong>
        </div>
        <div className="modal-actions">
          <button className="secondary-button" onClick={onCancel}>Cancel</button>
          <button className="danger-button" onClick={onConfirm}><Archive size={16} /> Archive</button>
        </div>
      </div>
    </div>
  );
}

type CoverageView = "skyline" | "heatmap" | "treemap";

function Overview({ snapshot, filteredModules, setSelected }: {
  snapshot: ProjectSnapshot;
  filteredModules: LayerSummary[];
  setSelected: (module: LayerSummary) => void;
}): JSX.Element {
  const [view, setView] = useState<CoverageView>("skyline");
  const liveModules = snapshot.modules.filter((module) =>
    module.pipelineStatus && module.pipelineStatus !== "pass" && module.pipelineStatus !== "fail_abort"
  );
  return (
    <section className="panel">
      <div className="kpi-grid">
        <KpiCard icon={Boxes} label="layers" value={snapshot.kpis.totalLayers} />
        <KpiCard icon={Code2} label="RTL" value={snapshot.kpis.rtlGenerated} />
        <KpiCard icon={CheckCircle2} label="Verilator pass" value={snapshot.kpis.verilatorPass} />
        <KpiCard icon={ShieldCheck} label="Vivado pass" value={snapshot.kpis.vivadoPass} />
        <KpiCard icon={GitCompare} label="improved variants" value={snapshot.kpis.improvedVariants} />
        <KpiCard icon={Gauge} label="known cost" value={fmtCost(snapshot.kpis.knownCostUsd)} />
      </div>
      {liveModules.length > 0 && (
        <div className="live-strip">
          <i className="dot live" />
          <strong>{liveModules.length} module{liveModules.length === 1 ? "" : "s"} mid-flight:</strong>
          {liveModules.slice(0, 6).map((module) => (
            <button key={module.moduleId} className="live-chip" onClick={() => setSelected(module)}>
              {module.moduleId} · <em>{module.pipelineStatus}</em>
            </button>
          ))}
          {liveModules.length > 6 && <span className="muted">+{liveModules.length - 6} more</span>}
        </div>
      )}
      <div className="section-head">
        <h2>Network Coverage</h2>
        <div className="view-toggle">
          {([
            ["skyline", "Skyline"],
            ["heatmap", "Heatmap"],
            ["treemap", "Treemap"],
          ] as const).map(([key, label]) => (
            <button
              key={key}
              className={view === key ? "active" : ""}
              onClick={() => setView(key)}
            >{label}</button>
          ))}
          <span className="muted">{filteredModules.length} visible</span>
        </div>
      </div>
      {view === "skyline" && <Skyline modules={filteredModules} onSelect={setSelected} />}
      {view === "heatmap" && <Heatmap modules={filteredModules} onSelect={setSelected} />}
      {view === "treemap" && <Treemap modules={filteredModules} onSelect={setSelected} />}
      <div className="legend">
        {["missing", "rtl", "verilator-pass", "vivado-pass", "failed", "improved"].map((stage) => (
          <span key={stage}><i className={`dot ${stage}`} /> {stage}</span>
        ))}
      </div>
      {snapshot.latestPipeline?.runId && (
        <div className="latest-run muted">
          Latest run <code>{snapshot.latestPipeline.runId.slice(0, 8)}</code>
          {snapshot.latestPipeline.startedAt ? ` · started ${snapshot.latestPipeline.startedAt}` : ""}
          {snapshot.latestPipeline.totalCostUsd !== undefined ? ` · ${fmtCost(snapshot.latestPipeline.totalCostUsd)}` : ""}
          {Object.keys(snapshot.latestPipeline.stateCounts).length > 0
            ? ` · ${Object.entries(snapshot.latestPipeline.stateCounts).map(([key, count]) => `${count} ${key}`).join(", ")}`
            : ""}
        </div>
      )}
    </section>
  );
}

function ModulesTable({ modules, onSelect }: {
  modules: LayerSummary[];
  onSelect: (module: LayerSummary) => void;
}): JSX.Element {
  return (
    <section className="panel">
      <div className="section-head">
        <h2>Modules</h2>
        <span>{modules.length}</span>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th><th>Module</th><th>Op</th><th>Stage</th><th>Pipeline</th><th>Output</th><th>Weights</th><th>LUT / DSP / BRAM</th><th>Fmax</th><th>Improve</th>
            </tr>
          </thead>
          <tbody>
            {modules.map((module) => (
              <tr key={module.moduleId} onClick={() => onSelect(module)}>
                <td>{module.index + 1}</td>
                <td><strong>{module.moduleId}</strong></td>
                <td>{module.opType}</td>
                <td><span className={`pill ${module.stage}`}>{module.stage}</span></td>
                <td>{module.pipelineStatus ?? "—"}</td>
                <td>{module.outputShape.join("x")}</td>
                <td>{fmtBytes(module.numWeights)}</td>
                <td>{fmtNumber(module.vivado?.lut)} / {fmtNumber(module.vivado?.dsp)} / {fmtNumber(module.vivado?.bram)}</td>
                <td>{module.vivado?.fmaxMhz ? `${module.vivado.fmaxMhz.toFixed(0)} MHz` : "—"}</td>
                <td>{module.improvements.filter((report) => report.success).length}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Knowledge({ docs, onOpenFile, onArchive }: {
  docs: DocSummary[];
  onOpenFile: (path: string) => void;
  onArchive: (doc: DocSummary, pathToArchive: string) => void;
}): JSX.Element {
  const tiers: Array<{ tier: DocSummary["tier"]; label: string }> = [
    { tier: "protected", label: "Protected (canonical, hand-curated)" },
    { tier: "active", label: "Active (promoted from probationary)" },
    { tier: "improved", label: "Improved (from --keep-reference runs)" },
    { tier: "probationary", label: "Probationary (auto-generated, unpromoted)" },
    { tier: "archive", label: "Archive" },
  ];
  return (
    <section className="panel">
      <div className="section-head">
        <h2>References & Knowledge</h2>
        <span>{docs.length} total</span>
      </div>
      <div className="stack">
        {tiers.map(({ tier, label }) => {
          const tierDocs = docs.filter((doc) => doc.tier === tier);
          if (tierDocs.length === 0) return null;
          return (
            <div key={tier} className="doc-tier">
              <div className="doc-tier-head">
                <span className={`pill ${tier === "improved" ? "improved" : tier === "protected" ? "vivado-pass" : tier === "active" ? "verilator-pass" : tier === "probationary" ? "rtl" : "missing"}`}>{tier}</span>
                <span className="muted">{label}</span>
                <span className="muted">{tierDocs.length}</span>
              </div>
              <div className="doc-grid">
                {tierDocs.map((doc) => (
                  <div className="doc-row" key={doc.id}>
                    <div>
                      <strong>{doc.id}</strong>
                      <span>{doc.opType ?? "unclassified"} · {doc.moduleId ?? "global"}{doc.improvementTargets?.length ? ` · ${doc.improvementTargets.join(", ")}` : ""}</span>
                    </div>
                    <div className="button-row">
                      {doc.patternPath && <button className="secondary-button" onClick={() => onOpenFile(doc.patternPath!)}>pattern</button>}
                      {doc.referencePath && <button className="secondary-button" onClick={() => onOpenFile(doc.referencePath!)}>reference</button>}
                      {doc.tier !== "protected" && doc.tier !== "archive" && (doc.patternPath || doc.referencePath) && (
                        <button className="danger-button" onClick={() => onArchive(doc, doc.patternPath ?? doc.referencePath!)}>
                          <Archive size={15} /> archive
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function Improvements({ reports, onOpenFile, onPromote }: {
  reports: ImprovementReportSummary[];
  onOpenFile: (path: string) => void;
  onPromote: (report: ImprovementReportSummary) => void;
}): JSX.Element {
  return (
    <section className="panel">
      <div className="section-head">
        <h2>Improvements & Versions</h2>
        <span>{reports.length} runs</span>
      </div>
      <div className="improve-grid">
        {reports.map((report) => (
          <div className="improve-row" key={report.reportPath}>
            <div className="improve-head">
              <div>
                <strong>{report.moduleId}</strong>
                <span>{report.targets.join(", ")} · {report.finalAction} · {report.attempts.length} attempts{report.costUsd ? ` · ${fmtCost(report.costUsd)}` : ""}</span>
              </div>
              <span className={`pill ${report.success ? "vivado-pass" : "failed"}`}>{report.success ? "verified" : "failed"}</span>
            </div>
            <div className="metric-strip">
              {report.attempts.map((attempt) => {
                const cls = attempt.verdictOverall ? "good" : attempt.failedGate ? "bad" : "";
                const fmax = attempt.vivado?.fmaxMhz;
                const wns = attempt.vivado?.setupWnsNs;
                return (
                  <span key={attempt.attemptIndex} className={cls}>
                    <strong>#{attempt.attemptIndex}</strong> {attempt.failedGate ?? "pass"} · LUT {fmtNumber(attempt.metrics?.lut)} · DSP {fmtNumber(attempt.metrics?.dsp)} · BRAM {fmtNumber(attempt.metrics?.bram)}
                    {fmax !== undefined ? ` · ${fmax.toFixed(0)} MHz` : ""}
                    {wns !== undefined && wns !== null ? ` · WNS ${wns.toFixed(2)} ns` : ""}
                  </span>
                );
              })}
            </div>
            <div className="button-row">
              <button className="secondary-button" onClick={() => onOpenFile(report.reportPath)}><FileText size={15} /> report</button>
              {report.improvedReferencePath && (
                <button className="secondary-button" onClick={() => onOpenFile(report.improvedReferencePath!)}><Code2 size={15} /> variant.v</button>
              )}
              {report.success && report.finalAction === "kept-as-variant" && (
                <button className="danger-button" onClick={() => onPromote(report)}><Rocket size={15} /> promote</button>
              )}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function Commands({ modules, onPreview }: {
  modules: LayerSummary[];
  onPreview: (action: JobAction) => void;
}): JSX.Element {
  const [checkpointPath, setCheckpointPath] = useState("checkpoints/resnet50_int8.pth");
  const [moduleId, setModuleId] = useState(modules[0]?.moduleId ?? "");
  const [targetSet, setTargetSet] = useState<ImprovementTarget[]>(["use-dsp"]);
  const [targetSlug, setTargetSlug] = useState("use-dsp");
  const [except, setExcept] = useState("");
  useEffect(() => {
    if (!moduleId && modules[0]) setModuleId(modules[0].moduleId);
  }, [modules, moduleId]);

  function toggleTarget(target: ImprovementTarget): void {
    setTargetSet((prev) => prev.includes(target) ? prev.filter((item) => item !== target) : [...prev, target]);
  }

  return (
    <section className="panel command-panel">
      <div className="section-head">
        <h2>Command Center</h2>
        <span>allowlisted</span>
      </div>
      <div className="command-grid">
        <div className="command-block">
          <h3><Hammer size={17} /> Pipeline</h3>
          <input value={checkpointPath} onChange={(e) => setCheckpointPath(e.target.value)} />
          <div className="button-row">
            <button className="primary-button" onClick={() => onPreview({ type: "pipeline", checkpointPath })}><Play size={16} /> run</button>
            <button className="secondary-button" onClick={() => onPreview({ type: "pipeline", checkpointPath, resume: true })}><RefreshCw size={16} /> resume</button>
          </div>
          <select value={moduleId} onChange={(e) => setModuleId(e.target.value)}>
            {modules.map((module) => <option key={module.moduleId} value={module.moduleId}>{module.moduleId}</option>)}
          </select>
          <div className="button-row">
            <button className="secondary-button" onClick={() => onPreview({ type: "pipeline", checkpointPath, only: moduleId })}>only selected</button>
            <input value={except} onChange={(e) => setExcept(e.target.value)} placeholder="module_a,module_b" />
            <button className="secondary-button" onClick={() => onPreview({ type: "pipeline", checkpointPath, except: except.split(",").map((s) => s.trim()).filter(Boolean) })}>except</button>
          </div>
        </div>
        <div className="command-block">
          <h3><GitCompare size={17} /> Improve</h3>
          <select value={moduleId} onChange={(e) => setModuleId(e.target.value)}>
            {modules.map((module) => <option key={module.moduleId} value={module.moduleId}>{module.moduleId}</option>)}
          </select>
          <div className="target-grid">
            {targets.map((target) => (
              <label key={target} className={targetSet.includes(target) ? "target active" : "target"}>
                <input type="checkbox" checked={targetSet.includes(target)} onChange={() => toggleTarget(target)} />
                {target}
              </label>
            ))}
          </div>
          <button
            className="primary-button"
            onClick={() => onPreview({ type: "improve", moduleId, targets: targetSet, keepReference: true })}
            disabled={targetSet.length === 0}
          >
            <Rocket size={16} /> improve as variant
          </button>
        </div>
        <div className="command-block">
          <h3><ShieldCheck size={17} /> Checks</h3>
          <div className="button-row wrap">
            {(["twins", "sdk-typecheck", "mcp-typecheck", "dashboard-typecheck", "dashboard-test", "improve-test"] as const).map((check) => (
              <button key={check} className="secondary-button" onClick={() => onPreview({ type: "check", check })}>{check}</button>
            ))}
          </div>
        </div>
        <div className="command-block">
          <h3><Rocket size={17} /> Promote Variant</h3>
          <select value={moduleId} onChange={(e) => setModuleId(e.target.value)}>
            {modules.map((module) => <option key={module.moduleId} value={module.moduleId}>{module.moduleId}</option>)}
          </select>
          <input value={targetSlug} onChange={(e) => setTargetSlug(e.target.value)} />
          <button className="danger-button" onClick={() => onPreview({ type: "promote-variant", moduleId, targetSlug })}>
            promote
          </button>
        </div>
      </div>
    </section>
  );
}

function Jobs({ jobs, onStop, onOpenLog }: {
  jobs: JobRecord[];
  onStop: (id: string) => void;
  onOpenLog: (id: string) => void;
}): JSX.Element {
  return (
    <section className="panel">
      <div className="section-head">
        <h2>Jobs</h2>
        <span>{jobs.length}</span>
      </div>
      <div className="job-list">
        {jobs.map((job) => (
          <div className="job-row" key={job.id}>
            <div>
              <strong>{job.title}</strong>
              <span>{job.state} · {job.createdAt}</span>
              <code>{job.command}</code>
            </div>
            <div className="button-row">
              <button className="secondary-button" onClick={() => onOpenLog(job.id)}><Terminal size={15} /> log</button>
              {(job.state === "running" || job.state === "queued" || job.state === "stopping") && (
                <button className="danger-button" onClick={() => onStop(job.id)}><CircleStop size={15} /> stop</button>
              )}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

export function App(): JSX.Element {
  const { snapshot, loading, error, reload } = useSnapshot();
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

  const opTypes = useMemo(() => [...new Set((snapshot?.modules ?? []).map((module) => module.opType))].sort(), [snapshot]);

  async function openFile(path: string): Promise<void> {
    setFile(await readFile(path));
  }

  async function preview(action: JobAction): Promise<void> {
    setConfirmPreview(await previewJob(action));
  }

  async function confirmJob(): Promise<void> {
    if (!confirmPreview) return;
    const record = await startJob(confirmPreview.action);
    setNotice(`Started ${record.title}`);
    setConfirmPreview(null);
    await reload();
    setTab("jobs");
  }

  async function promote(report: ImprovementReportSummary): Promise<void> {
    const preview = await previewJob({ type: "promote-variant", moduleId: report.moduleId, targetSlug: report.targetSlug });
    setConfirmPreview(preview);
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
        <div>
          <h1>nn2rtl Control Center</h1>
          <p>{snapshot.modelName} · {snapshot.kpis.totalLayers} layers · {snapshot.repoRoot}</p>
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
            <Icon size={16} /> {name}
          </button>
        ))}
      </nav>
      <section className="filters">
        <div className="searchbox">
          <Search size={16} />
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="module" />
        </div>
        <select value={opFilter} onChange={(e) => setOpFilter(e.target.value)}>
          <option value="all">all ops</option>
          {opTypes.map((op) => <option key={op} value={op}>{op}</option>)}
        </select>
        <select value={stageFilter} onChange={(e) => setStageFilter(e.target.value)}>
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
      {tab === "commands" && <Commands modules={snapshot.modules} onPreview={(action) => void preview(action)} />}
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
