import { Archive, Eye, FileText, Play, Rocket, X } from "lucide-react";

import type {
  DocSummary,
  FileReadResult,
  ImprovementReportSummary,
  JobPreview,
  LayerSummary,
} from "../shared/types";
import { fmtBytes, fmtNumber } from "./formatters";
import { LabeledTerm } from "./Tooltip";

export function ModuleDrawer({ module, onClose, onOpenFile, onPromote }: {
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
        <span><LabeledTerm term="Latency" glossaryKey="latency_cycles" /></span><strong>{fmtNumber(module.pipelineLatencyCycles)} cyc</strong>
        <span>Contract</span><strong>{module.contractId} <em>· {module.ioMode}</em></strong>
        <span>Pipeline</span><strong>{module.pipelineStatus ?? "—"}{module.pipelineAttempts !== undefined ? ` · ${module.pipelineAttempts} attempts` : ""}</strong>
        <span>Verilator</span><strong>{module.verif?.status ?? "missing"}{module.verif?.timingPass !== undefined ? ` · timing ${module.verif.timingPass ? "ok" : "fail"}` : ""}</strong>
        <span>Vivado</span><strong>{module.vivado?.success ? "pass" : module.vivado ? "fail" : "missing"}{module.vivado?.timingMet !== undefined ? ` · timing_met=${module.vivado.timingMet}` : ""}</strong>
        <span><LabeledTerm term="LUT" /> / <LabeledTerm term="FF" /></span><strong>{fmtNumber(module.vivado?.lut)} / {fmtNumber(module.vivado?.ff)}</strong>
        <span><LabeledTerm term="DSP" /> / <LabeledTerm term="BRAM" /></span><strong>{fmtNumber(module.vivado?.dsp)} / {fmtNumber(module.vivado?.bram)}</strong>
        <span><LabeledTerm term="Fmax" /> / <LabeledTerm term="WNS" /></span><strong>{fmtNumber(module.vivado?.fmaxMhz, 2)} MHz · {module.vivado?.setupWnsNs !== undefined && module.vivado?.setupWnsNs !== null ? `${module.vivado.setupWnsNs.toFixed(2)} ns` : "n/a"}</strong>
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

export function FileModal({ file, onClose }: {
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

export function ConfirmModal({ preview, onCancel, onConfirm }: {
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

export function ArchiveModal({ doc, pathToArchive, onCancel, onConfirm }: {
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
