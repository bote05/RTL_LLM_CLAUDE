// Redesigned Commands panel — small set of explicit task cards, each one
// wired to a real backend route. Replaces the previous freeform "Command
// Center" which the owner found unreadable.
//
// Every button on this page POSTs `/api/jobs/preview` → confirmation modal →
// `/api/jobs` (start). The action types match `JobAction` in shared/types.ts;
// `dashboard/src/server/jobs.ts` is the single source of truth for the
// actual CLI commands they invoke.

import { GitCompare, Hammer, Play, RefreshCw, Rocket, ShieldCheck, Wand2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import type { NetworkInfo } from "../shared/types";
import type { NetworkId } from "../shared/networks";
import {
  IMPROVE_SWEEP_PRESETS,
  type ImproveSweepPreset,
  type ImprovementTarget,
  type JobAction,
  type LayerSummary,
} from "../shared/types";
import { HelpTooltip, LabeledTerm } from "./Tooltip";

const TARGET_OPTIONS: { id: ImprovementTarget; label: string; hint: string }[] = [
  { id: "use-dsp", label: "Use DSP", hint: "Push multipliers into hard DSP blocks instead of LUTs. Lower LUT, higher DSP count." },
  { id: "use-bram", label: "Use BRAM", hint: "Store weights/activations in on-chip block RAM instead of distributed LUTRAM." },
  { id: "reduce-lut", label: "Reduce LUT", hint: "Trim look-up-table count without making other metrics worse." },
  { id: "reduce-latency", label: "Reduce latency", hint: "Lower the number of cycles from first input to last output." },
  { id: "increase-throughput", label: "Increase throughput", hint: "Lower the initiation interval (II) so the module accepts a new sample sooner." },
];

const CHECK_OPTIONS = [
  { id: "twins" as const, label: "Twin types", hint: "Verifies the SDK/MCP/dashboard type definitions are still in sync." },
  { id: "dashboard-typecheck" as const, label: "Dashboard typecheck", hint: "`tsc --noEmit` on the dashboard." },
  { id: "dashboard-test" as const, label: "Dashboard tests", hint: "Vitest run on this dashboard." },
  { id: "sdk-typecheck" as const, label: "SDK typecheck", hint: "`tsc --noEmit` on the pipeline SDK." },
  { id: "mcp-typecheck" as const, label: "MCP typecheck", hint: "`tsc --noEmit` on the MCP tools." },
  { id: "improve-test" as const, label: "Improve tests", hint: "Vitest for the improve pipeline." },
];

export function Commands({ modules, networks, networkId, onPreview }: {
  modules: LayerSummary[];
  networks: NetworkInfo[];
  networkId: NetworkId;
  onPreview: (action: JobAction) => void;
}): JSX.Element {
  const network = networks.find((entry) => entry.id === networkId);
  const [checkpointPath, setCheckpointPath] = useState(network?.defaultCheckpointPath ?? "");
  const [resume, setResume] = useState(false);
  const [moduleId, setModuleId] = useState<string>(modules[0]?.moduleId ?? "");
  const [targetSet, setTargetSet] = useState<ImprovementTarget[]>(["use-dsp"]);
  const [sweepPreset, setSweepPreset] = useState<ImproveSweepPreset>("ppa");
  const [sweepMode, setSweepMode] = useState<"plan" | "run">("plan");
  const [sweepMaxModules, setSweepMaxModules] = useState<number>(modules.length || 17);
  const [resynthModuleId, setResynthModuleId] = useState<string>(modules.find((module) => module.hasRtl)?.moduleId ?? modules[0]?.moduleId ?? "");

  useEffect(() => {
    if (!moduleId && modules[0]) setModuleId(modules[0].moduleId);
  }, [modules, moduleId]);

  useEffect(() => {
    setCheckpointPath(network?.defaultCheckpointPath ?? "");
  }, [network?.defaultCheckpointPath, networkId]);

  useEffect(() => {
    setSweepMaxModules(modules.length || 17);
  }, [modules.length]);

  function toggleTarget(target: ImprovementTarget): void {
    setTargetSet((prev) => prev.includes(target) ? prev.filter((item) => item !== target) : [...prev, target]);
  }

  const selectedPreset = useMemo(
    () => IMPROVE_SWEEP_PRESETS.find((preset) => preset.id === sweepPreset),
    [sweepPreset],
  );

  return (
    <section className="panel">
      <div className="section-head">
        <div>
          <h2>Tasks</h2>
          <p className="muted">Pick a card, fill in the inputs, click Run. Each card maps to one real CLI command — the dashboard never lies about what will be spawned.</p>
        </div>
      </div>
      <div className="task-grid">
        {/* CARD 1 — Generate RTL for a new model (normal pipeline) */}
        <article className="task-card task-pipeline">
          <header>
            <div className="task-title">
              <Hammer size={18} />
              <h3>Generate RTL for a new model</h3>
            </div>
            <p>Runs the full pipeline: for every layer, ask Claude for Verilog, verify with <LabeledTerm term="Verilator" />, then synthesize with <LabeledTerm term="Vivado" />.</p>
          </header>
          <label>
            Checkpoint path
            <input
              value={checkpointPath}
              onChange={(event) => setCheckpointPath(event.target.value)}
              placeholder="checkpoints/resnet50_int8.pth"
            />
          </label>
          <label className="task-inline">
            <input type="checkbox" checked={resume} onChange={(event) => setResume(event.target.checked)} />
            Resume an in-progress run (skip modules already passing)
          </label>
          <footer>
            <button
              className="primary-button"
              disabled={!checkpointPath.trim()}
              onClick={() => onPreview({ type: "pipeline", networkId, checkpointPath, resume })}
            >
              <Play size={16} /> Run pipeline
            </button>
            <span className="task-cost muted">cost risk: high · expect $$$</span>
          </footer>
        </article>

        {/* CARD 2 — Improve a single module */}
        <article className="task-card task-improve">
          <header>
            <div className="task-title">
              <Wand2 size={18} />
              <h3>Improve a single module</h3>
            </div>
            <p>Refactor one already-passing module for a quality target. Always keeps the original as a variant — your canonical RTL is untouched.</p>
          </header>
          <label>
            Module
            <select value={moduleId} onChange={(event) => setModuleId(event.target.value)}>
              {modules.length === 0 && <option value="">(no modules)</option>}
              {modules.map((module) => (
                <option key={module.moduleId} value={module.moduleId}>
                  {module.index + 1}. {module.moduleId} · {module.opType}
                </option>
              ))}
            </select>
          </label>
          <fieldset className="target-grid">
            <legend className="muted">Targets <HelpTooltip term="targets" hint="What metric you want better. Multiple targets run as an ordered sequence with hard gates against regression — none of the metrics you didn't ask for are allowed to get worse." /></legend>
            {TARGET_OPTIONS.map((option) => (
              <label key={option.id} className={targetSet.includes(option.id) ? "target active" : "target"}>
                <input type="checkbox" checked={targetSet.includes(option.id)} onChange={() => toggleTarget(option.id)} />
                <span>{option.label}</span>
                <HelpTooltip term={option.label} hint={option.hint} />
              </label>
            ))}
          </fieldset>
          <footer>
            <button
              className="primary-button"
              disabled={!moduleId || targetSet.length === 0}
              onClick={() => onPreview({ type: "improve", networkId, moduleId, targets: targetSet, keepReference: true })}
            >
              <Rocket size={16} /> Improve module
            </button>
            <span className="task-cost muted">cost risk: high · expect $-$$ per target</span>
          </footer>
        </article>

        {/* CARD 3 — Improve sweep */}
        <article className="task-card task-improve">
          <header>
            <div className="task-title">
              <GitCompare size={18} />
              <h3>Improve sweep</h3>
            </div>
            <p>Run an improve preset across every module in the network. Start with <strong>Plan</strong> (free, prints what it would do); then re-run with <strong>Run</strong>.</p>
          </header>
          <label>
            Preset
            <select value={sweepPreset} onChange={(event) => setSweepPreset(event.target.value as ImproveSweepPreset)}>
              {IMPROVE_SWEEP_PRESETS.map((preset) => (
                <option key={preset.id} value={preset.id}>{preset.label}</option>
              ))}
            </select>
          </label>
          {selectedPreset && <p className="muted task-preset-desc">{selectedPreset.description} <em>Targets: {selectedPreset.targets.join(" → ")}</em></p>}
          <div className="task-row">
            <label className="task-inline">
              <input type="radio" name="sweep-mode" checked={sweepMode === "plan"} onChange={() => setSweepMode("plan")} />
              Plan (dry run, free)
            </label>
            <label className="task-inline">
              <input type="radio" name="sweep-mode" checked={sweepMode === "run"} onChange={() => setSweepMode("run")} />
              Run (real $)
            </label>
          </div>
          <label>
            Max modules: <strong>{sweepMaxModules}</strong>
            <input
              type="range"
              min={1}
              max={Math.max(1, modules.length || 17)}
              value={sweepMaxModules}
              onChange={(event) => setSweepMaxModules(Number(event.target.value))}
            />
          </label>
          <footer>
            <button
              className={sweepMode === "plan" ? "secondary-button" : "primary-button"}
              onClick={() => onPreview({
                type: "improve-sweep",
                networkId,
                preset: sweepPreset,
                plan: sweepMode === "plan",
                maxModules: sweepMaxModules,
                keepReference: true,
              })}
            >
              {sweepMode === "plan" ? <RefreshCw size={16} /> : <Rocket size={16} />}
              {sweepMode === "plan" ? "Plan sweep" : "Run sweep"}
            </button>
            <span className="task-cost muted">{sweepMode === "plan" ? "cost: $0 — no Claude calls" : "cost risk: high · sweeps every module"}</span>
          </footer>
        </article>

        {/* CARD 4 — Resynth (Vivado only) */}
        <article className="task-card task-resynth">
          <header>
            <div className="task-title">
              <RefreshCw size={18} />
              <h3>Re-synthesize (Vivado only)</h3>
            </div>
            <p>Re-run synthesis for one module that already has RTL on disk. No Claude calls — just Vivado. Useful after upgrading the tool or changing the clock target.</p>
          </header>
          <label>
            Module
            <select value={resynthModuleId} onChange={(event) => setResynthModuleId(event.target.value)}>
              {modules.length === 0 && <option value="">(no modules)</option>}
              {modules.map((module) => (
                <option key={module.moduleId} value={module.moduleId} disabled={!module.hasRtl}>
                  {module.index + 1}. {module.moduleId}{module.hasRtl ? "" : " (no RTL yet)"}
                </option>
              ))}
            </select>
          </label>
          <footer>
            <button
              className="primary-button"
              disabled={!resynthModuleId}
              onClick={() => onPreview({ type: "resynth-module", networkId, moduleId: resynthModuleId })}
            >
              <Play size={16} /> Re-synthesize
            </button>
            <span className="task-cost muted">cost: $0 — Vivado only</span>
          </footer>
        </article>

        {/* CARD 5 — Maintenance checks */}
        <article className="task-card task-check">
          <header>
            <div className="task-title">
              <ShieldCheck size={18} />
              <h3>Maintenance checks</h3>
            </div>
            <p>Free, fast sanity checks. Run these before opening a PR.</p>
          </header>
          <div className="button-row wrap">
            {CHECK_OPTIONS.map((option) => (
              <button key={option.id} className="secondary-button" onClick={() => onPreview({ type: "check", check: option.id })} title={option.hint}>
                {option.label}
              </button>
            ))}
          </div>
        </article>
      </div>
    </section>
  );
}
