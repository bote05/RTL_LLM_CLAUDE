// Coverage visualisations (skyline / heatmap / treemap). Pulled out of
// App.tsx so the entry file stays focused on layout + state.

import { useMemo } from "react";
import type { LayerSummary } from "../shared/types";

export function Heatmap({ modules, onSelect }: {
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
function annotateBands(modules: LayerSummary[]): Array<{ module: LayerSummary; band: number }> {
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

export function Skyline({ modules, onSelect }: {
  modules: LayerSummary[];
  onSelect: (module: LayerSummary) => void;
}): JSX.Element {
  const annotated = useMemo(() => annotateBands(modules), [modules]);
  const regions = useMemo(() => {
    const groups: Array<{ band: number; start: number; end: number }> = [];
    for (let i = 0; i < annotated.length; i += 1) {
      const item = annotated[i];
      const last = groups[groups.length - 1];
      if (last && last.band === item.band) last.end = i;
      else groups.push({ band: item.band, start: i, end: i });
    }
    return groups;
  }, [annotated]);

  const range = useMemo(() => {
    const positives = annotated.map((item) => item.module.numWeights).filter((value) => value > 1);
    if (positives.length === 0) return { min: 0, max: 1 };
    return { min: Math.log10(Math.min(...positives)), max: Math.log10(Math.max(...positives)) };
  }, [annotated]);

  if (annotated.length === 0) {
    return <div className="skyline-empty muted">No layers in current filter.</div>;
  }
  const total = annotated.length;
  function heightPct(numWeights: number): number {
    if (numWeights <= 1) return 6;
    if (range.max <= range.min) return 80;
    const log = Math.log10(numWeights);
    const norm = (log - range.min) / (range.max - range.min);
    return 12 + norm * 80;
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
              <span className="skyline-band-label">{STAGE_BAND_LABELS[region.band - 1]}</span>
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
              style={{ left: `${leftPct}%`, width: `calc(${widthPct}% - 1px)`, height: `${h}%` }}
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
            <span key={`tick-${n}`} className="skyline-tick" style={{ left: `${((n - 1) / total) * 100}%` }}>
              {n}
            </span>
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

type TreemapBox = { module: LayerSummary; x: number; y: number; w: number; h: number };

function squarifiedTreemap(modules: LayerSummary[], width: number, height: number): TreemapBox[] {
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
    return Math.max((side * side * max) / (total * total), (total * total) / (side * side * min));
  }
  function layoutRow(row: { module: LayerSummary; area: number }[], rect: Rect): Rect {
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
  function squarify(remaining: { module: LayerSummary; area: number }[], row: { module: LayerSummary; area: number }[], rect: Rect): void {
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

export function Treemap({ modules, onSelect }: {
  modules: LayerSummary[];
  onSelect: (module: LayerSummary) => void;
}): JSX.Element {
  const VIRTUAL_W = 1600;
  const VIRTUAL_H = 900;
  const boxes = useMemo(() => squarifiedTreemap(modules, VIRTUAL_W, VIRTUAL_H), [modules]);
  if (boxes.length === 0) return <div className="skyline-empty muted">No layers in current filter.</div>;
  return (
    <div className="treemap">
      {boxes.map((box) => {
        const leftPct = (box.x / VIRTUAL_W) * 100;
        const topPct = (box.y / VIRTUAL_H) * 100;
        const widthPct = (box.w / VIRTUAL_W) * 100;
        const heightPct = (box.h / VIRTUAL_H) * 100;
        const showLabel = box.w >= 60 && box.h >= 28;
        const showDetail = box.w >= 80 && box.h >= 44;
        return (
          <button
            key={box.module.moduleId}
            className={`treemap-cell ${box.module.stage}`}
            style={{ left: `${leftPct}%`, top: `${topPct}%`, width: `${widthPct}%`, height: `${heightPct}%` }}
            onClick={() => onSelect(box.module)}
            title={`${box.module.index + 1}. ${box.module.moduleId}\n${box.module.opType} · ${box.module.outputShape.join("x")}\nweights: ${box.module.numWeights.toLocaleString()} bytes\nstage: ${box.module.stage}`}
          >
            {showLabel && <span className="treemap-id">{box.module.moduleId.replace(/^node_conv_/, "c")}</span>}
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
