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
  Rocket,
  ShieldCheck,
  Terminal,
} from "lucide-react";
import { useState } from "react";

import type {
  DocSummary,
  ImprovementReportSummary,
  JobRecord,
  LayerSummary,
  ProjectSnapshot,
} from "../shared/types";
import { Heatmap, Skyline, Treemap } from "./coverage";
import { fmtBytes, fmtCost, fmtNumber } from "./formatters";
import { LabeledTerm } from "./Tooltip";

function KpiCard({ label, value, icon: Icon, glossaryKey }: {
  label: string;
  value: string | number;
  icon: typeof Boxes;
  glossaryKey?: string;
}): JSX.Element {
  return (
    <div className="kpi">
      <Icon size={18} />
      <div>
        <strong>{value}</strong>
        <span>
          {glossaryKey ? <LabeledTerm term={label} glossaryKey={glossaryKey as keyof typeof import("./Tooltip").GLOSSARY} /> : label}
        </span>
      </div>
    </div>
  );
}

type CoverageView = "skyline" | "heatmap" | "treemap";

export function Overview({ snapshot, filteredModules, setSelected }: {
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
        <KpiCard icon={Code2} label="RTL files" value={snapshot.kpis.rtlGenerated} />
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
        <h2>Network coverage</h2>
        <div className="view-toggle">
          {([
            ["skyline", "Skyline"],
            ["heatmap", "Heatmap"],
            ["treemap", "Treemap"],
          ] as const).map(([key, label]) => (
            <button key={key} className={view === key ? "active" : ""} onClick={() => setView(key)}>{label}</button>
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

export function ModulesTable({ modules, onSelect }: {
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
              <th>#</th>
              <th>Module</th>
              <th>Op</th>
              <th>Stage</th>
              <th>Pipeline</th>
              <th>Output</th>
              <th>Weights</th>
              <th><LabeledTerm term="LUT" /> / <LabeledTerm term="DSP" /> / <LabeledTerm term="BRAM" /></th>
              <th><LabeledTerm term="Fmax" /></th>
              <th>Improvements</th>
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

export function Knowledge({ docs, onOpenFile, onArchive }: {
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
        <h2>References &amp; knowledge</h2>
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

export function Improvements({ reports, onOpenFile, onPromote }: {
  reports: ImprovementReportSummary[];
  onOpenFile: (path: string) => void;
  onPromote: (report: ImprovementReportSummary) => void;
}): JSX.Element {
  return (
    <section className="panel">
      <div className="section-head">
        <h2>Improvements &amp; versions</h2>
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

export function Jobs({ jobs, onStop, onOpenLog }: {
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
        {jobs.length === 0 && <p className="muted">No jobs yet. Run something from the Tasks tab.</p>}
        {jobs.map((job) => (
          <div className="job-row" key={job.id}>
            <div>
              <strong>{job.title}</strong>
              <span>{job.state} · {job.createdAt}</span>
              <code>{job.command}</code>
              {job.error && <span className="job-error">{job.error}</span>}
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

export { Eye };
