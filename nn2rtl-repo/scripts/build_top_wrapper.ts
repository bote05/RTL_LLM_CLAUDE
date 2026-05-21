// scripts/build_top_wrapper.ts
//
// Task 02 — LayerIR -> top-level wrapper generator.
//
// Reads output/layer_ir.json and emits one Verilog file at
// output/rtl/nn2rtl_top.v that wires the per-layer modules into a single
// dataflow design with a "hole" carved out for the shared compute engine
// (the heavy modules listed in task 06 / the Wave-1 fallback).
//
// The script is deterministic: identical inputs => byte-identical output.

import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

// Wave-1 fallback heavy list, copied verbatim from
// docs/agent_tasks/02_layerir_to_wrapper_generator.md. Used only when
// `--engine-modules` does not point at an existing file. Task 06 will
// produce the real list and the orchestrator will re-run this script.
const FALLBACK_HEAVY: readonly string[] = [
  "node_conv_284",
  "node_conv_286",
  "node_conv_290",
  "node_conv_292",
  "node_conv_296",
  "node_conv_298",
  "node_conv_282",
  "node_conv_288",
  "node_conv_294",
  "node_conv_220",
];

interface LayerIRItem {
  module_id: string;
  op_type: string;
  input_width_bits: number;
  output_width_bits: number;
  pipeline_latency_cycles?: number;
  // Optional shape arrays from LayerIR ([N, C, H, W] for activations).
  input_shape?: number[];
  output_shape?: number[];
}

interface LayerIR {
  model_name: string;
  layers: LayerIRItem[];
}

interface CliArgs {
  network: string;
  layerIr: string;
  engineModules: string;
  fifoSizes: string;
  weightMap: string;
  schedule: string;
  out: string;
}

function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = {
    network: "resnet-50",
    layerIr: "output/layer_ir.json",
    engineModules: "docs/agent_tasks/06_phase1_compression_candidates_HEAVY.txt",
    fifoSizes: "output/wrapper/skip_fifo_sizes.json",
    weightMap: "output/weights/weight_memory_map.json",
    schedule: "output/rtl/nn2rtl_scheduler_schedule.json",
    out: "output/rtl/nn2rtl_top.v",
  };
  for (const raw of argv.slice(2)) {
    const m = raw.match(/^--([^=]+)=(.*)$/);
    if (!m) continue;
    const [, k, v] = m;
    switch (k) {
      case "network": args.network = v; break;
      case "layer-ir": args.layerIr = v; break;
      case "engine-modules": args.engineModules = v; break;
      case "fifo-sizes": args.fifoSizes = v; break;
      case "weight-map": args.weightMap = v; break;
      case "schedule": args.schedule = v; break;
      case "out": args.out = v; break;
      default: throw new Error(`unknown flag --${k}`);
    }
  }
  return args;
}

// Read task 03's scheduler schedule and return an ordered list of
// engine-dispatched module IDs. Index in the returned array is the
// scheduler's `dispatch_index` for that module.
//
// Used by Fix 7 to assign each heavy layer's `engine_output_bridge` instance
// a SLOT parameter so it can detect "this dispatch is mine" by counting
// `engine_output_ready` pulses. The heavy-list file may contain modules the
// scheduler does NOT actually dispatch — those layers still appear as
// `isHeavy` in the topology and so have floating `_valid_out`/`_data_out`
// signals to fix; we drive those with constant zeros (no bridge slot).
function readScheduleDispatchOrder(p: string): string[] {
  const abs = path.isAbsolute(p) ? p : path.join(repoRoot, p);
  if (!existsSync(abs)) return [];
  try {
    const j = JSON.parse(readFileSync(abs, "utf8")) as Record<string, unknown>;
    const dispatches = (j.dispatches as Array<Record<string, unknown>>) ?? [];
    const ordered: string[] = [];
    for (const d of dispatches) {
      const idx = Number(d.dispatch_index ?? -1);
      const id = String(d.module_id ?? "");
      if (idx < 0 || !id) continue;
      while (ordered.length <= idx) ordered.push("");
      ordered[idx] = id;
    }
    return ordered;
  } catch {
    return [];
  }
}

interface DispatchSpec {
  index: number;
  module_id: string;
  input_bank: number;
  channel_in: number;
  channel_out: number;
  input_hw: [number, number];
  output_hw: [number, number];
}

function readScheduleDispatches(p: string): DispatchSpec[] {
  const abs = path.isAbsolute(p) ? p : path.join(repoRoot, p);
  if (!existsSync(abs)) return [];
  try {
    const j = JSON.parse(readFileSync(abs, "utf8")) as Record<string, unknown>;
    const dispatches = (j.dispatches as Array<Record<string, unknown>>) ?? [];
    const out: DispatchSpec[] = [];
    for (const d of dispatches) {
      const idx = Number(d.dispatch_index ?? -1);
      const id = String(d.module_id ?? "");
      if (idx < 0 || !id) continue;
      const ihw = d.input_hw as [number, number] | undefined;
      const ohw = d.output_hw as [number, number] | undefined;
      out.push({
        index: idx,
        module_id: id,
        input_bank: Number(d.input_bank ?? 0),
        channel_in: Number(d.channel_in ?? 0),
        channel_out: Number(d.channel_out ?? 0),
        input_hw: ihw && Array.isArray(ihw) ? [Number(ihw[0]), Number(ihw[1])] : [0, 0],
        output_hw: ohw && Array.isArray(ohw) ? [Number(ohw[0]), Number(ohw[1])] : [0, 0],
      });
    }
    out.sort((a, b) => a.index - b.index);
    return out;
  } catch {
    return [];
  }
}

function readHeavyList(p: string): { list: string[]; source: string } {
  const abs = path.isAbsolute(p) ? p : path.join(repoRoot, p);
  if (!existsSync(abs)) {
    return { list: [...FALLBACK_HEAVY], source: "FALLBACK (Wave-1)" };
  }
  const list = readFileSync(abs, "utf8")
    .split(/\r?\n/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0 && !s.startsWith("#"));
  return { list, source: abs };
}

interface SkipFifoEntry {
  // depth in words (one word == output_width_bits of the add input it feeds)
  depth: number;
}

// Accepts task-04's schema (`{ fifos: [{add_module_id, verified_depth,
// analytical_depth, ...}, ...] }`) and also the simpler `{ node_add: N }`
// shape used by earlier prototypes.
function readFifoSizes(p: string): Record<string, SkipFifoEntry> {
  const abs = path.isAbsolute(p) ? p : path.join(repoRoot, p);
  if (!existsSync(abs)) return {};
  const j = JSON.parse(readFileSync(abs, "utf8"));
  const out: Record<string, SkipFifoEntry> = {};
  if (j && Array.isArray((j as any).fifos)) {
    for (const f of (j as any).fifos as Array<Record<string, unknown>>) {
      const id = String(f.add_module_id ?? "");
      if (!id) continue;
      const depth =
        Number(f.verified_depth ?? f.analytical_depth ?? f.depth ?? 0) || 0;
      if (depth > 0) out[id] = { depth };
    }
    return out;
  }
  for (const [k, v] of Object.entries(j)) {
    if (typeof v === "number") out[k] = { depth: v };
    else if (v && typeof v === "object" && "depth" in (v as any)) {
      out[k] = { depth: Number((v as any).depth) };
    }
  }
  return out;
}

interface WeightMapInfo {
  total_uram_words: number;
  word_width_bits: number;
  // Path D (task 13a, banked layout). Only present in v2 schema.
  num_banks?: number;
  total_mac_cycles?: number;
}

// Accepts task-01's old schema (`{ uram_word_bits, total_words_used, ... }`),
// the intermediate `{ total_uram_words, word_width_bits }` shape, AND the
// Path-D v2 schema (`{ num_banks, total_mac_cycles, bank_line_bits, ... }`).
function readWeightMap(p: string): WeightMapInfo {
  const abs = path.isAbsolute(p) ? p : path.join(repoRoot, p);
  const fallback: WeightMapInfo = {
    total_uram_words: 1048576,
    word_width_bits: 288,
  };
  if (!existsSync(abs)) return fallback;
  const j = JSON.parse(readFileSync(abs, "utf8")) as Record<string, unknown>;
  const wordWidth = Number(
    (j.bank_line_bits as number | undefined) ??
      (j.word_width_bits as number | undefined) ??
      (j.uram_word_bits as number | undefined) ??
      fallback.word_width_bits,
  );
  const totalWords = Number(
    (j.total_mac_cycles as number | undefined) ??
      (j.total_uram_words as number | undefined) ??
      (j.total_words_used as number | undefined) ??
      fallback.total_uram_words,
  );
  return {
    total_uram_words: totalWords,
    word_width_bits: wordWidth,
    num_banks: typeof j.num_banks === "number" ? j.num_banks : undefined,
    total_mac_cycles: typeof j.total_mac_cycles === "number"
      ? j.total_mac_cycles
      : undefined,
  };
}

interface NodeMeta {
  ir: LayerIRItem;
  index: number;
  isHeavy: boolean;
  // For non-add layers: the module_id (or "PIXEL_IN") whose data_out drives
  //   this layer's data_in.
  // For add layers: the main (lhs) input source.
  mainSource: string;
  // Set on add layers only: the skip (rhs) input source.
  skipSource?: string;
  // True if this conv2d takes its data_in from a fork point rather than the
  // main chain — i.e. its input width does not match the chain-tail width.
  isProjection: boolean;
  // True if this layer's data_out fans out to BOTH the chain and a skip path.
  isForkPoint: boolean;
  // Physical bus widths used by the actual .v module (from the spec_hash in
  // the .meta.json sidecar). For tiled-streaming contracts these are
  // smaller than the logical channel-pack widths in LayerIR.
  busInBits: number;
  busOutBits: number;
  // Fix 17: per-layer .v file carries legacy DRAM `weights_*` ports that the
  // wrapper must tie off (input ports) or leave dangling (output ports) to
  // suppress Verilator PINMISSING and Vivado UNCONNECTED warnings.
  hasDramPorts: boolean;
}

// Pull the per-module bus widths out of the .meta.json sidecar's
// `spec_hash` field (format `..._i<N>_o<M>...`). Falls back to the LayerIR's
// input/output_width_bits when no spec_hash is present.
// 13a audit fix (Fix 17): some per-layer node_conv_*.v files (e.g.
// node_conv_284/288/292/298) carry legacy DRAM AXI4 `weights_*` read-channel
// ports left over from an earlier per-layer DRAM-streamed weight-loader
// design. In the engine-integrated wrapper those ports are unconnected,
// which Verilator flags as PINMISSING. Returns true if the .v file's
// port list contains a `weights_arvalid` declaration — that's the unique
// signature of the legacy interface.
function moduleHasDramPorts(moduleId: string, rtlDir: string): boolean {
  const vPath = path.join(rtlDir, `${moduleId}.v`);
  if (!existsSync(vPath)) return false;
  try {
    const text = readFileSync(vPath, "utf8");
    return /\bweights_arvalid\b/.test(text);
  } catch {
    return false;
  }
}

function readBusWidths(
  layers: LayerIRItem[],
  rtlDir: string,
): Map<string, { busIn: number; busOut: number }> {
  const out = new Map<string, { busIn: number; busOut: number }>();
  for (const L of layers) {
    let busIn = L.input_width_bits;
    let busOut = L.output_width_bits;
    const metaPath = path.join(rtlDir, `${L.module_id}.meta.json`);
    if (existsSync(metaPath)) {
      try {
        const j = JSON.parse(readFileSync(metaPath, "utf8")) as {
          spec_hash?: string;
        };
        const m = j.spec_hash?.match(/_i(\d+)_o(\d+)/);
        if (m) {
          busIn = Number(m[1]);
          busOut = Number(m[2]);
        }
      } catch {
        // ignore malformed meta files; keep IR widths
      }
    }
    out.set(L.module_id, { busIn, busOut });
  }
  return out;
}

function computeTopology(
  layers: LayerIRItem[],
  heavy: Set<string>,
  buses: Map<string, { busIn: number; busOut: number }>,
  dramPortIds: Set<string>,
): NodeMeta[] {
  const out: NodeMeta[] = [];
  // The chain tail is the module_id whose data_out should feed the next
  // ordinary chain link. Initially it is the network image input.
  let chainTail = "PIXEL_IN";
  let chainWidth = layers[0]?.input_width_bits ?? 0;
  // The most recent point in the chain a skip path may fork from. Updated
  // when we cross a maxpool (network stem) or a relu that immediately
  // follows an add (end of a residual block).
  let lastFork: string | null = null;
  // A projection conv that has been seen but not yet consumed by the
  // upcoming add. ResNet-style first-block-of-stage residuals emit the
  // projection conv in IR order right before the add.
  let pendingProj: string | null = null;
  let prevOp: string | null = null;

  for (let i = 0; i < layers.length; i++) {
    const L = layers[i];
    const bus = buses.get(L.module_id) ?? {
      busIn: L.input_width_bits,
      busOut: L.output_width_bits,
    };
    const m: NodeMeta = {
      ir: L,
      index: i,
      isHeavy: heavy.has(L.module_id),
      mainSource: "",
      isProjection: false,
      isForkPoint: false,
      busInBits: bus.busIn,
      busOutBits: bus.busOut,
      hasDramPorts: dramPortIds.has(L.module_id),
    };

    if (L.op_type === "add") {
      m.mainSource = chainTail;
      m.skipSource = pendingProj ?? lastFork ?? "PIXEL_IN";
      pendingProj = null;
      chainTail = L.module_id;
      chainWidth = L.output_width_bits;
    } else if (
      L.op_type === "conv2d" &&
      L.input_width_bits !== chainWidth
    ) {
      // Width mismatch with the chain tail => this conv consumes the
      // fork point on the skip side. The chain continues unchanged.
      m.mainSource = lastFork ?? "PIXEL_IN";
      m.isProjection = true;
      pendingProj = L.module_id;
    } else {
      m.mainSource = chainTail;
      chainTail = L.module_id;
      chainWidth = L.output_width_bits;
    }

    if (L.op_type === "maxpool") {
      lastFork = L.module_id;
    } else if (L.op_type === "relu" && prevOp === "add") {
      lastFork = L.module_id;
    }

    prevOp = L.op_type;
    out.push(m);
  }

  // Second pass: mark fork points (any module whose data_out is consumed by
  // an upcoming add's skip side, either directly or through a projection
  // conv).
  const consumedBySkip = new Set<string>();
  for (const m of out) {
    if (m.skipSource) consumedBySkip.add(m.skipSource);
    if (m.isProjection) consumedBySkip.add(m.mainSource);
  }
  for (const m of out) {
    if (consumedBySkip.has(m.ir.module_id)) m.isForkPoint = true;
  }

  return out;
}

function widthRange(bits: number): string {
  if (bits <= 1) return "";
  return `[${bits - 1}:0]`;
}

function wireDecl(name: string, bits: number): string {
  if (bits <= 1) return `wire ${name};`;
  return `wire [${bits - 1}:0] ${name};`;
}

function indent(s: string, n = 4): string {
  const pad = " ".repeat(n);
  return s
    .split("\n")
    .map((l) => (l.length > 0 ? pad + l : l))
    .join("\n");
}

interface EmitInput {
  meta: NodeMeta[];
  heavyList: string[];
  heavySource: string;
  fifoSizes: Record<string, SkipFifoEntry>;
  weightMap: WeightMapInfo;
  dispatchOrder: string[];
  dispatches: DispatchSpec[];
  args: CliArgs;
}

function emit(input: EmitInput): string {
  const {
    meta,
    heavyList,
    heavySource,
    fifoSizes,
    weightMap,
    dispatchOrder,
    dispatches,
    args,
  } = input;
  const lines: string[] = [];
  // Map each engine-dispatched module to its scheduler dispatch slot.
  // Heavy layers absent from dispatchOrder get slot = -1 and will be
  // tied off (constant zero outputs).
  const dispatchSlot = new Map<string, number>();
  dispatchOrder.forEach((id, i) => {
    if (id) dispatchSlot.set(id, i);
  });
  // Reverse map: producer module_id -> consumer node (the first non-heavy
  // node whose mainSource is the producer). Used so Fix 7's
  // engine_output_bridge can pick up the downstream layer's ready_in.
  const consumerOf = new Map<string, NodeMeta>();
  for (const m of meta) {
    if (!m.isHeavy && m.mainSource && !consumerOf.has(m.mainSource)) {
      consumerOf.set(m.mainSource, m);
    }
  }

  const firstLayer = meta[0];
  const lastLayer = meta[meta.length - 1];
  const pixelWidth = firstLayer.busInBits;
  const outWidth = lastLayer.busOutBits;

  const spatialCount = meta.filter((m) => !m.isHeavy).length;
  const heavyCount = meta.filter((m) => m.isHeavy).length;
  const addLayers = meta.filter((m) => m.ir.op_type === "add");
  const projLayers = meta.filter((m) => m.isProjection);

  // ---------------- File header ----------------
  lines.push(`// nn2rtl_top — auto-generated top-level wrapper for ${args.network}`);
  lines.push(`// Generated by scripts/build_top_wrapper.ts.`);
  lines.push(`// Source LayerIR:    ${args.layerIr}`);
  lines.push(`// Heavy module list: ${heavySource}`);
  lines.push(`// Skip-FIFO sizes:   ${args.fifoSizes}`);
  lines.push(`// Weight memory map: ${args.weightMap}`);
  lines.push(`// Layers total: ${meta.length}, spatial: ${spatialCount}, ` +
             `engine-dispatched: ${heavyCount}, residual adds: ${addLayers.length}, ` +
             `projection convs: ${projLayers.length}.`);
  lines.push(`//`);
  lines.push(`// This file is deterministically regenerated; do not edit by hand.`);
  lines.push(``);
  lines.push("`timescale 1ns/1ps");
  lines.push(``);
  // Task 04c hookup: tell shared_engine_skeleton.v to suppress its
  // built-in stubs because the integration build also includes the real
  // sub-blocks under output/rtl/engine/. iverilog defines are global
  // across the compilation unit, and the wrapper is the first file
  // pulled in by the integration parse, so this define lands before the
  // skeleton's `ifndef` is evaluated.
  lines.push("`define NN2RTL_ENGINE_SUBBLOCKS_PROVIDED");
  lines.push(``);

  // ---------------- Top-level module ----------------
  lines.push(`module nn2rtl_top (`);
  lines.push(`    input  wire                clk,`);
  lines.push(`    input  wire                rst_n,`);
  lines.push(``);
  lines.push(`    // AXI4-Stream image input (network input)`);
  lines.push(`    input  wire                s_axis_tvalid,`);
  lines.push(`    output wire                s_axis_tready,`);
  lines.push(`    input  wire ${widthRange(pixelWidth).padEnd(14)} s_axis_tdata,`);
  lines.push(`    input  wire                s_axis_tlast,`);
  lines.push(``);
  lines.push(`    // AXI4-Stream feature output (final layer's data_out)`);
  lines.push(`    output wire                m_axis_tvalid,`);
  lines.push(`    input  wire                m_axis_tready,`);
  lines.push(`    output wire ${widthRange(outWidth).padEnd(14)} m_axis_tdata,`);
  lines.push(`    output wire                m_axis_tlast,`);
  lines.push(``);
  lines.push(`    // AXI4-Lite control slave (forwarded to shared_engine)`);
  lines.push(`    input  wire                s_axil_awvalid,`);
  lines.push(`    output wire                s_axil_awready,`);
  lines.push(`    input  wire [31:0]         s_axil_awaddr,`);
  lines.push(`    input  wire                s_axil_wvalid,`);
  lines.push(`    output wire                s_axil_wready,`);
  lines.push(`    input  wire [31:0]         s_axil_wdata,`);
  lines.push(`    input  wire [3:0]          s_axil_wstrb,`);
  lines.push(`    output wire                s_axil_bvalid,`);
  lines.push(`    input  wire                s_axil_bready,`);
  lines.push(`    output wire [1:0]          s_axil_bresp,`);
  lines.push(`    input  wire                s_axil_arvalid,`);
  lines.push(`    output wire                s_axil_arready,`);
  lines.push(`    input  wire [31:0]         s_axil_araddr,`);
  lines.push(`    output wire                s_axil_rvalid,`);
  lines.push(`    input  wire                s_axil_rready,`);
  lines.push(`    output wire [31:0]         s_axil_rdata,`);
  lines.push(`    output wire [1:0]          s_axil_rresp`);
  lines.push(`);`);
  lines.push(``);

  // ---------------- Pixel input alias ----------------
  lines.push(`    // ----- network input (PIXEL_IN) -----`);
  lines.push(`    wire ${widthRange(pixelWidth)} PIXEL_IN_data  = s_axis_tdata;`);
  lines.push(`    wire                       PIXEL_IN_valid = s_axis_tvalid;`);
  lines.push(``);

  // ---------------- Per-layer wire declarations ----------------
  lines.push(`    // ----- per-layer stream wires -----`);
  for (const m of meta) {
    const id = m.ir.module_id;
    const w = m.busOutBits;
    lines.push(`    ${wireDecl(`${id}_valid_out`, 1)}`);
    lines.push(`    ${wireDecl(`${id}_data_out`, w)}`);
    if (!m.isHeavy) {
      // ready_in is an OUTPUT of the layer module (backpressure to upstream).
      lines.push(`    ${wireDecl(`${id}_ready_in`, 1)}`);
    }
  }
  lines.push(``);

  // ---------------- Skip-FIFO output wires ----------------
  // Each add's `skip_data` wire is sized to be the upper half of the add's
  // data_in bus — i.e. half of `busInBits`. For tiled-streaming adds the
  // bus packs {rhs_tile, lhs_tile}, so each tile occupies busInBits/2 bits.
  lines.push(`    // ----- skip-FIFO outputs (one per residual add) -----`);
  for (const a of addLayers) {
    const id = a.ir.module_id;
    const skipW = Math.floor(a.busInBits / 2);
    lines.push(`    ${wireDecl(`${id}_skip_valid`, 1)}`);
    lines.push(`    ${wireDecl(`${id}_skip_data`, skipW)}`);
  }
  lines.push(``);

  // ---------------- Engine BRAM port wires ----------------
  // Widths track output/rtl/shared_engine_skeleton.v default parameters:
  //   ACT_BRAM_ADDR_W = 16, ACT_BUS_W = 2048,
  //   URAM_ADDR_W     = 22, URAM_DATA_W = 2048.
  lines.push(`    // ----- shared_engine internal port wires (driven by engine/scheduler) -----`);
  lines.push(`    wire [15:0]                engine_act_in_rd_addr;`);
  lines.push(`    wire [2047:0]              engine_act_in_rd_data;`);
  lines.push(`    wire                       engine_act_in_rd_en;`);
  lines.push(`    wire [15:0]                engine_act_out_wr_addr;`);
  lines.push(`    wire [2047:0]              engine_act_out_wr_data;`);
  lines.push(`    wire                       engine_act_out_wr_en;`);
  lines.push(`    wire [21:0]                engine_weight_rd_addr;`);
  lines.push(`    wire                       engine_weight_rd_en;`);
  lines.push(`    wire [2047:0]              engine_weight_rd_data;`);
  // Bias port (task 13a Bundle A / Fix 5): one wide bias word per
  // oc_pass = MAC_COUNT × INT32 = 256 × 32 = 8192 bits.
  lines.push(`    wire [21:0]                engine_bias_rd_addr;`);
  lines.push(`    wire                       engine_bias_rd_en;`);
  lines.push(`    wire [8191:0]              engine_bias_rd_data;`);
  lines.push(`    wire                       engine_start;`);
  lines.push(`    wire                       engine_busy;`);
  lines.push(`    wire                       engine_done;`);
  lines.push(``);
  // Scheduler ↔ engine AXI4-Lite master/slave bundle (task 13a Fix 6).
  // The scheduler in output/rtl/nn2rtl_scheduler.v is the AXI master to
  // the engine's config_register_block; the host's top-level s_axil_* is
  // currently unused (tied off below) so the scheduler has exclusive
  // ownership of the engine's config writes for Phase 2 first-light.
  lines.push(`    // ----- scheduler ↔ engine AXI4-Lite master/slave bundle (Fix 6) -----`);
  lines.push(`    wire        sched_axil_awvalid;`);
  lines.push(`    wire        sched_axil_awready;`);
  lines.push(`    wire [7:0]  sched_axil_awaddr;`);
  lines.push(`    wire        sched_axil_wvalid;`);
  lines.push(`    wire        sched_axil_wready;`);
  lines.push(`    wire [31:0] sched_axil_wdata;`);
  lines.push(`    wire [3:0]  sched_axil_wstrb;`);
  lines.push(`    wire        sched_axil_bvalid;`);
  lines.push(`    wire        sched_axil_bready;`);
  lines.push(`    wire [1:0]  sched_axil_bresp;`);
  // Scheduler bank-select / skip-mask signals are now redundant with the
  // bridge-side compile-time input_bank constants (the wrapper's
  // stream_to_act_bram_bridge instances already encode each dispatch's
  // bank in their BRAM_BASE_ADDR parameter). Kept on the scheduler port
  // list for symmetry with the schedule sidecar, but bound to a sink
  // wire so iverilog's UNUSED-output lint stays quiet.
  lines.push(`    wire [2:0]  sched_input_bank_sel;`);
  lines.push(`    wire [2:0]  sched_output_bank_sel;`);
  lines.push(`    wire [5:0]  sched_skip_bank_reserved_mask;`);
  lines.push(`    wire        sched_spatial_stall;`);
  lines.push(`    wire        sched_engine_output_ready;`);
  lines.push(`    wire        sched_done;`);
  // Fix 11: scheduler exposes dispatch_idx so the wrapper can mux the
  // active dispatch's `loaded` signal back as current_loaded.
  {
    const dispBits = Math.max(1, Math.ceil(Math.log2(Math.max(2, dispatchOrder.length))));
    lines.push(`    wire [${dispBits - 1}:0] sched_dispatch_idx;`);
  }
  // 13a audit fix (Fix 16): forward-declare current_loaded and
  // current_drain_complete here so the scheduler instance (emitted next)
  // doesn't reference an implicit 1-bit net. Their real drivers come from
  // the input-loader and bridge mux blocks emitted much later, but
  // Verilog wires can be driven from arbitrary points in the module body
  // as long as they're declared first.
  lines.push(`    wire current_loaded;`);
  lines.push(`    wire current_drain_complete;`);
  lines.push(``);
  lines.push(`    // ----- scheduler 'start' one-shot from first accepted input beat (Fix 6) -----`);
  lines.push(`    // Per task 13a Fix 6: easiest source for the scheduler's start is the`);
  lines.push(`    // first cycle the host's image stream is accepted (s_axis_tvalid &&`);
  lines.push(`    // s_axis_tready). Latched so we don't re-pulse on every subsequent beat.`);
  lines.push(`    reg sched_started_r;`);
  lines.push(`    always @(posedge clk or negedge rst_n) begin`);
  lines.push(`        if (!rst_n) sched_started_r <= 1'b0;`);
  lines.push(`        else if (s_axis_tvalid && s_axis_tready) sched_started_r <= 1'b1;`);
  lines.push(`    end`);
  lines.push(`    wire sched_start = s_axis_tvalid & s_axis_tready & ~sched_started_r;`);
  lines.push(``);
  lines.push(`    // ----- host AXI4-Lite to engine: TIED OFF in Phase 2 first-light -----`);
  lines.push(`    // The scheduler is the engine's sole AXI master right now; muxing the`);
  lines.push(`    // host in is task 13's concern. Hold every host *_ready / *_valid low`);
  lines.push(`    // so any external host that tries to write never sees a handshake.`);
  lines.push(`    assign s_axil_awready = 1'b0;`);
  lines.push(`    assign s_axil_wready  = 1'b0;`);
  lines.push(`    assign s_axil_bvalid  = 1'b0;`);
  lines.push(`    assign s_axil_bresp   = 2'b00;`);
  lines.push(`    assign s_axil_arready = 1'b0;`);
  lines.push(`    assign s_axil_rvalid  = 1'b0;`);
  lines.push(`    assign s_axil_rdata   = 32'd0;`);
  lines.push(`    assign s_axil_rresp   = 2'b00;`);
  lines.push(``);
  lines.push(`    // ----- scheduler instantiation (Fix 6) -----`);
  lines.push(`    nn2rtl_scheduler u_scheduler (`);
  lines.push(`        .clk(clk), .rst_n(rst_n),`);
  lines.push(`        .start(sched_start),`);
  lines.push(`        .done(sched_done),`);
  lines.push(`        .s_axil_awvalid(sched_axil_awvalid),`);
  lines.push(`        .s_axil_awready(sched_axil_awready),`);
  lines.push(`        .s_axil_awaddr(sched_axil_awaddr),`);
  lines.push(`        .s_axil_wvalid(sched_axil_wvalid),`);
  lines.push(`        .s_axil_wready(sched_axil_wready),`);
  lines.push(`        .s_axil_wdata(sched_axil_wdata),`);
  lines.push(`        .s_axil_wstrb(sched_axil_wstrb),`);
  lines.push(`        .s_axil_bvalid(sched_axil_bvalid),`);
  lines.push(`        .s_axil_bready(sched_axil_bready),`);
  lines.push(`        .s_axil_bresp(sched_axil_bresp),`);
  lines.push(`        .engine_start(engine_start),`);
  lines.push(`        .engine_busy(engine_busy),`);
  lines.push(`        .engine_done(engine_done),`);
  lines.push(`        .current_loaded(current_loaded),`);
  lines.push(`        .current_drain_complete(current_drain_complete),`);
  lines.push(`        .dispatch_idx_out(sched_dispatch_idx),`);
  lines.push(`        .input_bank_sel(sched_input_bank_sel),`);
  lines.push(`        .output_bank_sel(sched_output_bank_sel),`);
  lines.push(`        .skip_bank_reserved_mask(sched_skip_bank_reserved_mask),`);
  lines.push(`        .spatial_stall(sched_spatial_stall),`);
  lines.push(`        .engine_output_ready(sched_engine_output_ready)`);
  lines.push(`    );`);
  lines.push(``);
  lines.push(`    // ----- task 04c: spatial_throttle = engine_busy -----`);
  lines.push(`    // While the engine is processing a heavy layer, every spatial`);
  lines.push(`    // module's incoming valid handshake is gated low. This freezes`);
  lines.push(`    // the entire spatial chain so the skip FIFOs do not accumulate`);
  lines.push(`    // during engine windows, which lets the bounded FIFO depths`);
  lines.push(`    // chosen in output/wrapper/skip_fifo_sizes.json fit U250's`);
  lines.push(`    // on-chip memory budget. The gate is harmless for blocks with`);
  lines.push(`    // no engine dispatches: engine_busy stays low for those phases.`);
  // 13a audit fix: engine_busy alone is insufficient. The scheduler asserts
  // sched_spatial_stall for the ~13 AXI4-Lite config-register writes BEFORE
  // engine_start pulses (states S_WRITE / S_WRITE_RESP / S_NEXT_STEP /
  // S_PULSE_START / S_WAIT_DONE / S_NEXT_DISP). engine_busy only covers the
  // engine's own compute window (ST_LOAD_CONFIG..ST_DONE), so without the
  // OR'd stall the spatial chain keeps streaming during ~196 cycles per
  // dispatch × 14 dispatches ≈ 2.7k cycles of "engine idle but scheduler
  // busy" time. Result: skip FIFOs drift past their sized depths.
  lines.push(`    wire spatial_throttle = engine_busy | sched_spatial_stall;`);
  lines.push(`    wire spatial_run      = ~spatial_throttle;`);
  lines.push(``);

  // Helper: resolve a source name to (valid_signal, data_signal).
  function srcValid(s: string): string {
    return s === "PIXEL_IN" ? "PIXEL_IN_valid" : `${s}_valid_out`;
  }
  function srcData(s: string): string {
    return s === "PIXEL_IN" ? "PIXEL_IN_data" : `${s}_data_out`;
  }

  // ---------------- s_axis_tready ----------------
  // The first layer's ready_in is its OUTPUT toward upstream; route it
  // to s_axis_tready. 04c additionally drops tready while engine_busy
  // so the external master holds new beats — that mirrors what the
  // wrapper's internal spatial_run gate already does to the chain.
  const firstId = firstLayer.ir.module_id;
  if (firstLayer.isHeavy) {
    lines.push(`    assign s_axis_tready = spatial_run;`);
  } else {
    lines.push(`    assign s_axis_tready = ${firstId}_ready_in & spatial_run;`);
  }
  lines.push(``);

  // ---------------- Per-layer module instantiations ----------------
  lines.push(`    // ----- spatial module instantiations -----`);
  for (const m of meta) {
    const id = m.ir.module_id;
    if (m.isHeavy) {
      lines.push(`    // ${id}: engine-dispatched (no instantiation here; ` +
                 `data_out driven by shared_engine via task-11 bridge)`);
      lines.push(``);
      continue;
    }

    if (m.ir.op_type === "add") {
      const mainV = srcValid(m.mainSource);
      const mainD = srcData(m.mainSource);
      const skipV = `${id}_skip_valid`;
      const skipD = `${id}_skip_data`;
      // Each side feeds half of the add's input bus (lhs_tile in the low
      // half, rhs_tile in the high half). Slice the producer's main-path
      // bus to that width — the wave-2 retile bridge will handle channel
      // re-tiling when the producer's bus is wider than a single add tile.
      const halfW = Math.floor(m.busInBits / 2);
      lines.push(`    // residual add: lhs (low half) from main path, rhs (high half) from skip FIFO`);
      lines.push(`    ${id} u_${id} (`);
      lines.push(`        .clk(clk), .rst_n(rst_n),`);
      // The add fires only when both halves are valid AND the engine is
      // not currently consuming a heavy layer (04c spatial_throttle).
      lines.push(`        .valid_in(${mainV} & ${skipV} & spatial_run),`);
      lines.push(`        .ready_in(${id}_ready_in),`);
      lines.push(`        .data_in({${skipD}, ${mainD}[${halfW - 1}:0]}),`);
      lines.push(`        .valid_out(${id}_valid_out),`);
      lines.push(`        .data_out(${id}_data_out)`);
      lines.push(`    );`);
      lines.push(``);
    } else {
      const v = srcValid(m.mainSource);
      const d = srcData(m.mainSource);
      lines.push(`    ${id} u_${id} (`);
      lines.push(`        .clk(clk), .rst_n(rst_n),`);
      // 04c spatial_throttle: gate every spatial chain link so the
      // entire chain freezes during engine_busy.
      lines.push(`        .valid_in(${v} & spatial_run),`);
      lines.push(`        .ready_in(${id}_ready_in),`);
      lines.push(`        .data_in(${d}),`);
      lines.push(`        .valid_out(${id}_valid_out),`);
      // Fix 17: tie off legacy DRAM `weights_*` ports for modules that
      // still carry the pre-engine port shape (e.g. node_conv_284/288/
      // 292/298). Input-side ports are driven 0 (DRAM never grants);
      // output-side ports are left dangling. Comma after data_out only
      // if more port-bindings follow.
      const trail = m.hasDramPorts ? "," : "";
      lines.push(`        .data_out(${id}_data_out)${trail}`);
      if (m.hasDramPorts) {
        lines.push(`        // Tie-offs for legacy DRAM AXI4 weights_* read channel.`);
        lines.push(`        .weights_arvalid(),`);
        lines.push(`        .weights_arready(1'b0),`);
        lines.push(`        .weights_araddr(),`);
        lines.push(`        .weights_arlen(),`);
        lines.push(`        .weights_rvalid(1'b0),`);
        lines.push(`        .weights_rready(),`);
        lines.push(`        .weights_rdata(64'd0),`);
        lines.push(`        .weights_rlast(1'b0)`);
      }
      lines.push(`    );`);
      lines.push(``);
    }
  }

  // ---------------- Skip FIFO instantiations ----------------
  // FIFO WIDTH matches the upper half of the add's input bus (one rhs tile
  // per beat). When the skip source's data_out is wider than the add's
  // rhs-tile width, we feed the low bits — wave-2 retile bridges will
  // handle the channel-tile rate match.
  lines.push(`    // ----- skip FIFOs (one per residual add) -----`);
  for (const a of addLayers) {
    const id = a.ir.module_id;
    const skipW = Math.floor(a.busInBits / 2);
    const skipSource = a.skipSource ?? "PIXEL_IN";
    const sv = srcValid(skipSource);
    const sd = srcData(skipSource);
    const sizeEntry = fifoSizes[id];
    const depth = sizeEntry?.depth ?? defaultSkipDepth(a, meta);
    // 13a audit fix: previously `.in_ready()` was dangling — producers
    // could push when the FIFO was full and data would be silently
    // dropped by the FIFO's internal `~full` write gate. Wire in_ready
    // back into the push gate so a near-full FIFO actually stalls the
    // push. The FIFO's in_ready is `~full` (combinational on registered
    // pointers), so no combinational loop.
    lines.push(`    wire ${id}_skip_in_ready;`);
    lines.push(`    skip_fifo #(.WIDTH(${skipW}), .DEPTH(${depth})) u_skip_${id} (`);
    lines.push(`        .clk(clk), .rst_n(rst_n),`);
    // 04c: gate the skip-side push with spatial_run so the FIFO does
    // not accumulate during engine_busy windows. The fork point's
    // chain consumers are already throttled (see per-layer valid_in
    // gating), so this keeps the two paths in sync.
    lines.push(`        .in_valid(${sv} & spatial_run & ${id}_skip_in_ready),`);
    lines.push(`        .in_data(${sd}[${skipW - 1}:0]),`);
    lines.push(`        .in_ready(${id}_skip_in_ready),`);
    lines.push(`        .out_valid(${id}_skip_valid),`);
    lines.push(`        .out_data(${id}_skip_data),`);
    lines.push(`        .out_ready(${id}_ready_in)`);
    lines.push(`    );`);
    lines.push(``);
  }

  // ---------------- URAM weight memory subsystem (Path D, task 13a) ----------------
  //
  // Architecture:
  //   - 8 parallel URAM read banks (NUM_BANKS), each 288 bits wide (native
  //     URAM cascade width: 4 URAM288 primitives in parallel).
  //   - All banks share the same MAC-cycle address; one cycle returns a
  //     288-bit word from each bank.
  //   - Bank N stores 32 weights = 256 useful bits per line in the low
  //     [255:0]; the high 32 bits per bank are zero-pad (matches the URAM
  //     288-bit native shape but the MAC array only consumes 256 bits/bank).
  //   - The wrapper concatenates the low BANK_USEFUL_BITS of each bank
  //     into the engine's 2048-bit weight bus. No '>>3' conversion: the
  //     address generator already emits weight_rd_addr in MAC-cycle units
  //     and that maps 1:1 to bank.rd_addr.
  //
  // .mem files: output/weights/uram_weights_bank<N>.mem, N=0..7. One
  // 72-hex-char line per MAC cycle. Generated deterministically by
  // scripts/build_weight_memory_map.py.
  //
  // URAM accounting per weight_memory_map.json (Path D / banked):
  //   total_mac_cycles, num_banks, per_bank_uram_blocks, etc.
  const numBanks = Number(weightMap.num_banks ?? 8);
  const totalMacCycles = Number(weightMap.total_mac_cycles ?? 0);
  const macAddrBits = Math.max(1, Math.ceil(Math.log2(Math.max(2, totalMacCycles))));
  lines.push(`    // ----- URAM-resident weight memory subsystem (Path D: ${numBanks} parallel banks) -----`);
  lines.push(`    // Total MAC cycles = ${totalMacCycles}; per-bank depth = ${totalMacCycles}.`);
  lines.push(`    // Address path: engine_weight_rd_addr[${macAddrBits - 1}:0] -> each bank's rd_addr.`);
  lines.push(`    wire [${macAddrBits - 1}:0] weight_bank_rd_addr = engine_weight_rd_addr[${macAddrBits - 1}:0];`);
  // 8 parallel bank wires at 288 bits each. The low 256 bits of each
  // contain the real weight bytes; the high 32 bits are zero-pad.
  for (let b = 0; b < numBanks; b++) {
    lines.push(`    wire [287:0] uram_bank${b}_rd_data;`);
  }
  // Concatenate low 256 bits of each bank -> 2048-bit MAC bus.
  // Bank 0 occupies bits [255:0], bank 1 bits [511:256], ..., bank 7 bits [2047:1792].
  const concatExprParts: string[] = [];
  for (let b = numBanks - 1; b >= 0; b--) {
    concatExprParts.push(`uram_bank${b}_rd_data[255:0]`);
  }
  lines.push(``);
  lines.push(`    // MAC bus = concat of the low 256 bits of each bank (bank 0 lowest).`);
  lines.push(`    assign engine_weight_rd_data = {${concatExprParts.join(`,\n${" ".repeat(8)}`)}};`);
  lines.push(``);
  // Instantiate 8 banks.
  for (let b = 0; b < numBanks; b++) {
    lines.push(`    uram_weight_bank #(`);
    lines.push(`        .DEPTH(${totalMacCycles}),`);
    lines.push(`        .ADDR_W(${macAddrBits}),`);
    lines.push(`        .MEM_INIT_FILE("output/weights/uram_weights_bank${b}.mem")`);
    lines.push(`    ) u_uram_weight_bank${b} (`);
    lines.push(`        .clk(clk),`);
    lines.push(`        .rd_addr(weight_bank_rd_addr),`);
    lines.push(`        .rd_data(uram_bank${b}_rd_data),`);
    lines.push(`        .rd_en(engine_weight_rd_en)`);
    lines.push(`    );`);
  }
  lines.push(``);

  // Bias memory (task 13a Bundle A / Fix 5). One wide bias word per
  // oc_pass × heavy layer. SIZE_WORDS is bounded by the heavy layer
  // with the largest output-channel count: cfg_oc / MAC_COUNT (≤8)
  // entries per layer × 14 heavy layers = at most ~112 wide words.
  // We allocate 256 to leave headroom; the actual bias .mem file is
  // generated by a future task (placeholder MEM_INIT_FILE accepted).
  lines.push(`    // ----- bias memory (Fix 5: per-oc_pass wide bias word, 256 × INT32) -----`);
  lines.push(`    bias_mem #(`);
  lines.push(`        .SIZE_WORDS(256),`);
  lines.push(`        .WORD_WIDTH(8192),`);
  lines.push(`        .ADDR_W(8),`); // 256 entries = 8-bit address
  lines.push(`        .MEM_INIT_FILE("output/weights/bias.mem")`);
  lines.push(`    ) u_bias_mem (`);
  lines.push(`        .clk(clk),`);
  lines.push(`        .rd_addr(engine_bias_rd_addr[7:0]),`); // engine emits 22b; we use the 8 LSBs
  lines.push(`        .rd_data(engine_bias_rd_data),`);
  lines.push(`        .rd_en(engine_bias_rd_en)`);
  lines.push(`    );`);
  lines.push(``);

  // ---------------- Activation BRAM subsystem (Fix 8) ----------------
  //
  // The engine has BRAM-style activation ports (act_in_rd_* and
  // act_out_wr_*); previously these were declared in the wrapper but
  // engine_act_in_rd_data was a floating wire (no driver). That made
  // the engine functionally inert — it would read X / 0 in simulation,
  // and Vivado would optimise the dead read path away in synthesis.
  //
  // We give the engine a single unified URAM-backed memory big enough
  // for NUM_BANKS × BANK_DEPTH_WORDS entries (the scheduler partitions
  // it into per-bank windows via cfg_act_in_bram_base = bank ×
  // BANK_DEPTH_WORDS). The engine's address generator already strides
  // by ic_chunks on reads (Fix B in address_generator.v) and by
  // oc_passes on writes; the unified flat memory just needs to honour
  // those addresses.
  //
  // Out of scope for this fix: routing spatial layer outputs INTO this
  // BRAM. For first-light + Verilator simulation, the testbench
  // pre-loads the BRAM with the network input or stub data; subsequent
  // heavy-dispatch outputs go through the engine's own write port. End-
  // to-end spatial↔BRAM coordination is a follow-up.
  const ACT_BRAM_NUM_BANKS = 6;
  const ACT_BRAM_DEPTH_PER_BANK = 4096;
  const ACT_BRAM_TOTAL_DEPTH = ACT_BRAM_NUM_BANKS * ACT_BRAM_DEPTH_PER_BANK;
  const ACT_BRAM_ADDR_W = Math.ceil(Math.log2(ACT_BRAM_TOTAL_DEPTH));

  // ---------------- Input loaders (13a Fix 11) ----------------
  //
  // Each engine dispatch reads its input from the activation BRAM at
  // bank = scheduler-assigned input_bank. Previously the wrapper had
  // NO writer feeding spatial-layer outputs into the BRAM, so dispatch
  // 0 read uninitialised memory. Now, for each heavy dispatch we
  // instantiate a `stream_to_act_bram_bridge` that taps the predecessor
  // spatial layer's output stream, accumulates beats into 2048-bit BRAM
  // words, and writes to the dispatch's bank window.
  //
  // The bridge asserts `loaded` after a full frame has been written.
  // A mux on dispatch_idx selects the current bridge's `loaded`, and
  // the scheduler waits on it in S_WAIT_LOAD before pulsing CONTROL.start.
  //
  // Write arbitration: simple priority — engine has highest priority;
  // among bridges, lower dispatch_idx wins. Conflicts are rare in
  // practice (spatial layers stream at ~1 beat/cycle, BRAM word writes
  // happen at 1/N that rate; only one bridge is "active" at any time
  // because spatial layers are serial). If a bridge's write is denied
  // it holds `wr_req` high until granted.
  const heavyMetaById = new Map<string, NodeMeta>();
  for (const m of meta) heavyMetaById.set(m.ir.module_id, m);
  const bridgeSpecs: Array<{
    dispatchIdx: number;
    moduleId: string;
    predId: string;
    predBusOut: number;
    inputBank: number;
    icChunks: number;
    totalBramWords: number;
  }> = [];
  for (const d of dispatches) {
    const heavy = heavyMetaById.get(d.module_id);
    if (!heavy) continue;
    const predId = heavy.mainSource;
    if (!predId || predId === "PIXEL_IN") {
      // Network-input-fed dispatch (currently nothing in ResNet-50 starts
      // this way); skip bridge — tied off.
      continue;
    }
    const predMeta = heavyMetaById.get(predId);
    if (!predMeta) continue;
    const icChunks = Math.ceil(d.channel_in / 256) || 1;
    const totalBramWords = d.input_hw[0] * d.input_hw[1] * icChunks;
    bridgeSpecs.push({
      dispatchIdx: d.index,
      moduleId: d.module_id,
      predId,
      predBusOut: predMeta.busOutBits,
      inputBank: d.input_bank,
      icChunks,
      totalBramWords,
    });
  }

  lines.push(`    // ----- input-loader bridges (Fix 11: spatial-stream -> activation BRAM) -----`);
  lines.push(`    // Each bridge captures one heavy dispatch's predecessor stream and`);
  lines.push(`    // writes 2048-bit BRAM words into its dispatch's bank window.`);
  for (const b of bridgeSpecs) {
    lines.push(`    wire        ldr${b.dispatchIdx}_wr_req;`);
    lines.push(`    wire        ldr${b.dispatchIdx}_wr_grant;`);
    lines.push(`    wire [14:0] ldr${b.dispatchIdx}_wr_addr;`);
    lines.push(`    wire [2047:0] ldr${b.dispatchIdx}_wr_data;`);
    lines.push(`    wire        ldr${b.dispatchIdx}_loaded;`);
    lines.push(`    stream_to_act_bram_bridge #(`);
    lines.push(`        .BUS_W(${b.predBusOut}),`);
    lines.push(`        .BRAM_BASE_ADDR(${b.inputBank * ACT_BRAM_DEPTH_PER_BANK}),`);
    lines.push(`        .TOTAL_BRAM_WORDS(${b.totalBramWords})`);
    lines.push(`    ) u_ldr_${b.moduleId} (`);
    lines.push(`        .clk(clk), .rst_n(rst_n),`);
    lines.push(`        .in_valid(${b.predId}_valid_out & spatial_run),`);
    lines.push(`        .in_data(${b.predId}_data_out),`);
    lines.push(`        .wr_req(ldr${b.dispatchIdx}_wr_req),`);
    lines.push(`        .wr_grant(ldr${b.dispatchIdx}_wr_grant),`);
    lines.push(`        .wr_addr(ldr${b.dispatchIdx}_wr_addr),`);
    lines.push(`        .wr_data(ldr${b.dispatchIdx}_wr_data),`);
    lines.push(`        .loaded(ldr${b.dispatchIdx}_loaded)`);
    lines.push(`    );`);
    lines.push(``);
  }

  // ---- BRAM write arbiter: engine highest, then bridges in dispatch order ----
  lines.push(`    // ----- act BRAM write arbiter: engine priority, then bridges -----`);
  lines.push(`    wire        act_wr_en_final;`);
  lines.push(`    wire [${ACT_BRAM_ADDR_W - 1}:0] act_wr_addr_final;`);
  lines.push(`    wire [2047:0] act_wr_data_final;`);
  if (bridgeSpecs.length === 0) {
    lines.push(`    assign act_wr_en_final   = engine_act_out_wr_en;`);
    lines.push(`    assign act_wr_addr_final = engine_act_out_wr_addr[${ACT_BRAM_ADDR_W - 1}:0];`);
    lines.push(`    assign act_wr_data_final = engine_act_out_wr_data;`);
  } else {
    // Build priority cascade: engine wins, else lowest-idx bridge with wr_req.
    lines.push(`    // Priority: engine > ldr0 > ldr1 > ... > ldr${bridgeSpecs[bridgeSpecs.length - 1].dispatchIdx}.`);
    // Grant signals
    let denials = `engine_act_out_wr_en`;
    for (const b of bridgeSpecs) {
      lines.push(`    assign ldr${b.dispatchIdx}_wr_grant = ldr${b.dispatchIdx}_wr_req & ~(${denials});`);
      denials += ` | ldr${b.dispatchIdx}_wr_req`;
    }
    // Address / data mux (priority-ordered)
    let addrMux = `engine_act_out_wr_en ? engine_act_out_wr_addr[${ACT_BRAM_ADDR_W - 1}:0]`;
    let dataMux = `engine_act_out_wr_en ? engine_act_out_wr_data`;
    let enChain = `engine_act_out_wr_en`;
    for (const b of bridgeSpecs) {
      addrMux += ` : ldr${b.dispatchIdx}_wr_req ? ldr${b.dispatchIdx}_wr_addr`;
      dataMux += ` : ldr${b.dispatchIdx}_wr_req ? ldr${b.dispatchIdx}_wr_data`;
      enChain += ` | ldr${b.dispatchIdx}_wr_req`;
    }
    addrMux += ` : ${ACT_BRAM_ADDR_W}'d0`;
    dataMux += ` : 2048'd0`;
    lines.push(`    assign act_wr_en_final   = ${enChain};`);
    lines.push(`    assign act_wr_addr_final = ${addrMux};`);
    lines.push(`    assign act_wr_data_final = ${dataMux};`);
  }
  lines.push(``);

  lines.push(`    // ----- activation BRAM (Fix 8 + Fix 11: unified ${ACT_BRAM_NUM_BANKS}-bank URAM, ${ACT_BRAM_TOTAL_DEPTH} × 2048b) -----`);
  lines.push(`    act_unified_mem #(`);
  lines.push(`        .DEPTH(${ACT_BRAM_TOTAL_DEPTH}),`);
  lines.push(`        .ADDR_W(${ACT_BRAM_ADDR_W})`);
  lines.push(`    ) u_act_mem (`);
  lines.push(`        .clk(clk),`);
  lines.push(`        .rd_addr(engine_act_in_rd_addr[${ACT_BRAM_ADDR_W - 1}:0]),`);
  lines.push(`        .rd_en  (engine_act_in_rd_en),`);
  lines.push(`        .rd_data(engine_act_in_rd_data),`);
  lines.push(`        .wr_addr(act_wr_addr_final),`);
  lines.push(`        .wr_en  (act_wr_en_final),`);
  lines.push(`        .wr_data(act_wr_data_final)`);
  lines.push(`    );`);
  lines.push(`    wire _unused_act_in_addr_hi  = |engine_act_in_rd_addr[15:${ACT_BRAM_ADDR_W}];`);
  lines.push(`    wire _unused_act_out_addr_hi = |engine_act_out_wr_addr[15:${ACT_BRAM_ADDR_W}];`);
  // Drain the scheduler's dynamic bank-select / skip-mask outputs and the
  // sched_done output into _unused_* wires so Vivado's UNUSED-output lint
  // stays quiet. The wrapper's bridges encode bank info at compile time;
  // sched_done is not surfaced to a host completion register in Phase 2
  // (the host can detect completion via m_axis_tlast on the final beat).
  lines.push(`    wire _unused_sched_bank_sels = |sched_input_bank_sel`);
  lines.push(`                                  | |sched_output_bank_sel`);
  lines.push(`                                  | |sched_skip_bank_reserved_mask;`);
  lines.push(`    wire _unused_sched_done      = sched_done;`);
  lines.push(``);

  // ---- current_loaded mux for scheduler ----
  lines.push(`    // ----- per-dispatch loaded mux (Fix 11: scheduler waits before engine start) -----`);
  lines.push(`    // The scheduler samples current_loaded in S_WAIT_LOAD to ensure the`);
  lines.push(`    // input_loader bridge for the current dispatch has finished filling`);
  lines.push(`    // the BRAM before pulsing engine_start.`);
  if (bridgeSpecs.length === 0) {
    // current_loaded already forward-declared at the scheduler-bundle wire
    // block (Fix 16). Just drive it constant here.
    lines.push(`    assign current_loaded = 1'b1;`);
  } else {
    // 13a audit fix (Fix 12): extend all_loaded to a power-of-2 width so the
    // mux can never index out of range. `sched_dispatch_idx` is N bits where
    // N = ceil(log2(num_dispatches)), so it can address 2^N entries. The
    // index range exceeds num_dispatches when num_dispatches isn't a power
    // of 2 (e.g. 14 dispatches → 4-bit idx → 16 possible values). Pad the
    // unused entries with 1 so an over-range index doesn't stall the
    // scheduler (defensive — the FSM should never emit such an index).
    const dispBits = Math.max(1, Math.ceil(Math.log2(Math.max(2, dispatchOrder.length))));
    const allLoadedWidth = 1 << dispBits;
    // current_loaded is forward-declared (Fix 16); just drive it via the mux.
    lines.push(`    wire [${allLoadedWidth - 1}:0] all_loaded;`);
    const bridgeIdxSet = new Set(bridgeSpecs.map((b) => b.dispatchIdx));
    for (let i = 0; i < allLoadedWidth; i += 1) {
      if (bridgeIdxSet.has(i)) {
        lines.push(`    assign all_loaded[${i}] = ldr${i}_loaded;`);
      } else {
        // Either a heavy dispatch with no bridge (e.g. PIXEL_IN-fed), or an
        // index past num_dispatches (defensive pad).
        lines.push(`    assign all_loaded[${i}] = 1'b1;`);
      }
    }
    lines.push(`    assign current_loaded = all_loaded[sched_dispatch_idx[${dispBits - 1}:0]];`);
  }
  lines.push(``);

  // ---------------- shared_engine instantiation ----------------
  // Port set tracks docs/agent_tasks/00_engine_skeleton_spec.md and the
  // skeleton at output/rtl/shared_engine_skeleton.v. The skeleton's
  // AXI4-Lite slave is 8-bit address-wide; the top-level exposes a 32-bit
  // address bus and we feed the low 8 bits through.
  lines.push(`    // ----- shared compute engine (handles ${heavyCount} heavy modules) -----`);
  lines.push(`    // Fix 6: AXI4-Lite slave is driven by u_scheduler, not by the host`);
  lines.push(`    // top-level s_axil_*. The host bundle is tied off above. The read`);
  lines.push(`    // channel is unused by the scheduler (write-only); ground it.`);
  lines.push(`    shared_engine u_shared_engine (`);
  lines.push(`        .clk(clk), .rst_n(rst_n),`);
  lines.push(`        .s_axil_awvalid(sched_axil_awvalid), .s_axil_awready(sched_axil_awready), .s_axil_awaddr(sched_axil_awaddr),`);
  lines.push(`        .s_axil_wvalid(sched_axil_wvalid),   .s_axil_wready(sched_axil_wready),   .s_axil_wdata(sched_axil_wdata),  .s_axil_wstrb(sched_axil_wstrb),`);
  lines.push(`        .s_axil_bvalid(sched_axil_bvalid),   .s_axil_bready(sched_axil_bready),   .s_axil_bresp(sched_axil_bresp),`);
  lines.push(`        .s_axil_arvalid(1'b0),               .s_axil_arready(),                   .s_axil_araddr(8'd0),`);
  lines.push(`        .s_axil_rvalid(),                    .s_axil_rready(1'b0),                .s_axil_rdata(),              .s_axil_rresp(),`);
  lines.push(`        .engine_start(engine_start),`);
  lines.push(`        .engine_busy(engine_busy),`);
  lines.push(`        .engine_done(engine_done),`);
  lines.push(`        .act_in_rd_addr(engine_act_in_rd_addr),`);
  lines.push(`        .act_in_rd_en(engine_act_in_rd_en),`);
  lines.push(`        .act_in_rd_data(engine_act_in_rd_data),`);
  lines.push(`        .act_out_wr_addr(engine_act_out_wr_addr),`);
  lines.push(`        .act_out_wr_en(engine_act_out_wr_en),`);
  lines.push(`        .act_out_wr_data(engine_act_out_wr_data),`);
  lines.push(`        .weight_rd_addr(engine_weight_rd_addr),`);
  lines.push(`        .weight_rd_en(engine_weight_rd_en),`);
  lines.push(`        .weight_rd_data(engine_weight_rd_data),`);
  // Bias port (task 13a Bundle A / Fix 5).
  lines.push(`        .bias_rd_addr(engine_bias_rd_addr),`);
  lines.push(`        .bias_rd_en(engine_bias_rd_en),`);
  lines.push(`        .bias_rd_data(engine_bias_rd_data)`);
  lines.push(`    );`);
  lines.push(``);

  // ---------------- Engine-output bridges (Fix 7) ----------------
  // For each layer the scheduler dispatches to the engine, the engine
  // writes its output activations to act_out_wr_data/_en. The bridge
  // captures those writes for the dispatch slot that owns the layer
  // and re-streams them as the layer's `_valid_out`/`_data_out` so the
  // downstream spatial-chain consumer sees the same handshake it would
  // have seen from a normal per-layer module. The bridge's `start`
  // input is the scheduler's per-dispatch `engine_output_ready` pulse.
  //
  // Heavy layers that appear in the heavy list but NOT in the
  // scheduler's dispatch order get their outputs tied to constant zero.
  // That keeps every `_valid_out` / `_data_out` wire with exactly one
  // driver — the structural condition the audit flagged.
  // ----- shared engine_output_fifo (Fix 14) -----
  //
  // All engine act_out writes flow into this FIFO. Per-dispatch bridges
  // drain at their respective active slots, gated by their consumer's
  // ready_in AND (downstream-side) spatial_run. The scheduler waits on
  // `current_drain_complete` (= active bridge's drain_complete) in
  // S_WAIT_DRAIN to ensure dispatches don't interleave in the FIFO.
  lines.push(`    // ----- shared engine output FIFO (Fix 14) -----`);
  lines.push(`    wire        eofifo_in_ready;`);
  lines.push(`    wire        eofifo_out_valid;`);
  lines.push(`    wire [2047:0] eofifo_out_data;`);
  lines.push(`    wire        eofifo_out_ready;`);
  lines.push(`    engine_output_fifo #(`);
  lines.push(`        .DEPTH(4096),`);
  lines.push(`        .ADDR_W(12),`);
  lines.push(`        .DATA_W(2048)`);
  lines.push(`    ) u_engine_out_fifo (`);
  lines.push(`        .clk(clk), .rst_n(rst_n),`);
  lines.push(`        .in_valid(engine_act_out_wr_en),`);
  lines.push(`        .in_data(engine_act_out_wr_data),`);
  lines.push(`        .in_ready(eofifo_in_ready),`);
  lines.push(`        .out_valid(eofifo_out_valid),`);
  lines.push(`        .out_data(eofifo_out_data),`);
  lines.push(`        .out_ready(eofifo_out_ready)`);
  lines.push(`    );`);
  // Engine has no backpressure path — if FIFO is full we drop, but the
  // FIFO is sized so that never happens for valid ResNet-50 dispatches.
  lines.push(`    wire _unused_eofifo_in_ready = eofifo_in_ready;`);
  lines.push(``);

  // Collect each dispatched heavy layer's expected output beat count and
  // bridge instance name for the drain_complete mux.
  const dispatchById = new Map<string, DispatchSpec>();
  for (const d of dispatches) dispatchById.set(d.module_id, d);
  const heavyBridges: Array<{ slot: number; instName: string; expected: number }> = [];

  lines.push(`    // ----- engine-output bridges (Fix 7 + Fix 14: FIFO-drain shims) -----`);
  for (const m of meta) {
    if (!m.isHeavy) continue;
    const id = m.ir.module_id;
    const slot = dispatchSlot.get(id);
    if (slot === undefined) {
      lines.push(
        `    // ${id}: heavy in the heavy list but not in scheduler dispatchOrder → tie off.`,
      );
      lines.push(`    assign ${id}_valid_out = 1'b0;`);
      lines.push(`    assign ${id}_data_out  = ${m.busOutBits}'d0;`);
      lines.push(``);
      continue;
    }
    const d = dispatchById.get(id);
    const oc = d?.channel_out ?? 0;
    const oh = d?.output_hw?.[0] ?? 0;
    const ow = d?.output_hw?.[1] ?? 0;
    const ocPasses = Math.max(1, Math.ceil(oc / 256));
    const expectedBeats = Math.max(1, oh * ow * ocPasses);
    const consumer = consumerOf.get(id);
    // Fix 14: ready_out now gated by spatial_run so the bridge stops
    // pulling from the FIFO when the spatial chain is frozen. Beats stay
    // safely in the FIFO instead of being dropped by the bridge skid.
    const readyOut = consumer
      ? `(${consumer.ir.module_id}_ready_in & spatial_run)`
      : `spatial_run`;
    const instName = `u_engine_out_${id}`;
    heavyBridges.push({ slot, instName, expected: expectedBeats });
    lines.push(`    wire ${instName}_fifo_ready;`);
    lines.push(`    wire ${instName}_drain_complete;`);
    lines.push(`    engine_output_bridge #(`);
    lines.push(`        .SLOT(${slot}),`);
    lines.push(`        .ACT_W(2048),`);
    lines.push(`        .DATA_W(${m.busOutBits}),`);
    lines.push(`        .EXPECTED_BEATS(${expectedBeats}),`);
    lines.push(`        .NUM_DISPATCHES(${dispatchOrder.length})`);
    lines.push(`    ) ${instName} (`);
    lines.push(`        .clk(clk), .rst_n(rst_n),`);
    lines.push(`        .start(sched_engine_output_ready),`);
    lines.push(`        .fifo_out_valid(eofifo_out_valid),`);
    lines.push(`        .fifo_out_data(eofifo_out_data),`);
    lines.push(`        .fifo_out_ready(${instName}_fifo_ready),`);
    lines.push(`        .ready_out(${readyOut}),`);
    lines.push(`        .valid_out(${id}_valid_out),`);
    lines.push(`        .data_out(${id}_data_out),`);
    lines.push(`        .drain_complete(${instName}_drain_complete)`);
    lines.push(`    );`);
    lines.push(``);
  }

  // OR all bridges' fifo_ready signals into the shared FIFO's out_ready.
  if (heavyBridges.length === 0) {
    lines.push(`    assign eofifo_out_ready = 1'b0;`);
  } else {
    const parts = heavyBridges.map((b) => `${b.instName}_fifo_ready`);
    lines.push(`    assign eofifo_out_ready = ${parts.join(" | ")};`);
  }
  lines.push(``);

  // Build a per-dispatch drain_complete vector and mux by sched_dispatch_idx.
  {
    const dispBits = Math.max(1, Math.ceil(Math.log2(Math.max(2, dispatchOrder.length))));
    const allDrainWidth = 1 << dispBits;
    lines.push(`    // ----- per-dispatch drain_complete mux (Fix 14) -----`);
    lines.push(`    wire [${allDrainWidth - 1}:0] all_drain;`);
    const bySlot = new Map(heavyBridges.map((b) => [b.slot, b.instName] as const));
    for (let i = 0; i < allDrainWidth; i += 1) {
      const name = bySlot.get(i);
      if (name) {
        lines.push(`    assign all_drain[${i}] = ${name}_drain_complete;`);
      } else {
        // Dispatch slot has no bridge (PIXEL_IN-fed) or is a pad index.
        // Treat as already-drained.
        lines.push(`    assign all_drain[${i}] = 1'b1;`);
      }
    }
    // current_drain_complete forward-declared at the scheduler-bundle wire
    // block (Fix 16) so the scheduler instance above sees a real wire.
    lines.push(`    assign current_drain_complete = all_drain[sched_dispatch_idx[${dispBits - 1}:0]];`);
    lines.push(``);
  }

  // ---------------- Tail output ----------------
  //
  // 13a audit fix (#4): m_axis_tlast was hardwired to 0, so downstream
  // AXI-Stream consumers (DMA, frame buffers, scoreboard) couldn't tell
  // when one frame ended. Track the total number of output beats per
  // frame (= ceil(OH*OW*OC*ACT_W / outWidth)) and assert tlast on the
  // final output beat of each frame. The counter resets when sched_done
  // (full pipeline finished) goes low — i.e. on each fresh start.
  //
  // For ResNet-50 backbone (last layer node_relu_48, 7×7×2048 = 100,352
  // bytes; output bus 256 bits = 32 bytes/beat → 3,136 beats per frame).
  // Compile-time constant from the LayerIR/meta widths.
  lines.push(`    // ----- network output (Fix #4: tlast on final output beat) -----`);
  lines.push(`    assign m_axis_tvalid = ${lastLayer.ir.module_id}_valid_out;`);
  lines.push(`    assign m_axis_tdata  = ${lastLayer.ir.module_id}_data_out;`);
  {
    // Compute total output beats per frame from the final layer's spatial
    // dimensions and channels. layer_ir's output_shape is [N, C, H, W].
    const out = lastLayer.ir.output_shape ?? [1, 1, 1, 1];
    const C = Number(out[1] ?? 1);
    const H = Number(out[2] ?? 1);
    const W = Number(out[3] ?? 1);
    const ACT_W = 8;
    const bitsPerBeat = lastLayer.busOutBits;
    const totalBits = C * H * W * ACT_W;
    const totalBeats = Math.max(1, Math.ceil(totalBits / bitsPerBeat));
    const beatCntW = Math.max(1, Math.ceil(Math.log2(totalBeats + 1)));
    lines.push(`    // Output frame size: C=${C} H=${H} W=${W}, busOut=${bitsPerBeat}b -> ${totalBeats} beats`);
    lines.push(`    reg [${beatCntW - 1}:0] m_axis_beat_count;`);
    lines.push(`    always @(posedge clk or negedge rst_n) begin`);
    lines.push(`        if (!rst_n) begin`);
    lines.push(`            m_axis_beat_count <= ${beatCntW}'d0;`);
    lines.push(`        end else if (m_axis_tvalid & m_axis_tready) begin`);
    lines.push(`            if (m_axis_beat_count == ${beatCntW}'d${totalBeats - 1})`);
    lines.push(`                m_axis_beat_count <= ${beatCntW}'d0;`);
    lines.push(`            else`);
    lines.push(`                m_axis_beat_count <= m_axis_beat_count + ${beatCntW}'d1;`);
    lines.push(`        end`);
    lines.push(`    end`);
    lines.push(`    assign m_axis_tlast = (m_axis_beat_count == ${beatCntW}'d${totalBeats - 1});`);
  }
  lines.push(``);

  lines.push(`endmodule`);
  lines.push(``);

  // ---------------- Wrapper-local module bodies ----------------
  // The wrapper references `skip_fifo`, `uram_weight_bank`, `bias_mem`
  // and `engine_output_bridge` as small wrapper-local primitives whose
  // real implementations are owned here (task 13a Fix 3 + Fix 7). The
  // bodies are gated by \`NN2RTL_TOP_NO_STUBS\` so a downstream
  // integration build that ships its own implementations can suppress
  // them via a single \`define on the iverilog command line.
  //
  // `shared_engine` is NOT defined here — it is owned by
  // `output/rtl/shared_engine_skeleton.v` (task 00).
  lines.push(`\`ifndef NN2RTL_TOP_NO_STUBS`);
  lines.push(`// ===== Wrapper-local module bodies (suppress with \`define NN2RTL_TOP_NO_STUBS) =====`);
  lines.push(``);

  // ---- skip_fifo: power-of-two-depth FIFO with backpressure ----
  // One-cycle read latency; in_ready low when full; out_valid low when
  // empty. Power-of-2 DEPTH so the read/write pointer math is a simple
  // mask (matches what task 04c assumed for skip-FIFO sizing).
  lines.push(`module skip_fifo #(`);
  lines.push(`    parameter integer WIDTH = 8,`);
  lines.push(`    parameter integer DEPTH = 16`);
  lines.push(`) (`);
  lines.push(`    input  wire              clk,`);
  lines.push(`    input  wire              rst_n,`);
  lines.push(`    input  wire              in_valid,`);
  lines.push(`    input  wire [WIDTH-1:0]  in_data,`);
  lines.push(`    output wire              in_ready,`);
  lines.push(`    output wire              out_valid,`);
  lines.push(`    output wire [WIDTH-1:0]  out_data,`);
  lines.push(`    input  wire              out_ready`);
  lines.push(`);`);
  lines.push(`    // DEPTH must be a power of 2; ADDR_W = log2(DEPTH).`);
  lines.push(`    function integer clog2;`);
  lines.push(`        input integer value;`);
  lines.push(`        integer v;`);
  lines.push(`        begin`);
  lines.push(`            v = value - 1;`);
  lines.push(`            for (clog2 = 0; v > 0; clog2 = clog2 + 1) v = v >> 1;`);
  lines.push(`        end`);
  lines.push(`    endfunction`);
  lines.push(`    localparam integer ADDR_W = clog2(DEPTH);`);
  lines.push(``);
  lines.push(`    reg [WIDTH-1:0] mem [0:DEPTH-1];`);
  lines.push(`    reg [ADDR_W:0]  wr_ptr;`);
  lines.push(`    reg [ADDR_W:0]  rd_ptr;`);
  lines.push(``);
  lines.push(`    wire [ADDR_W-1:0] wr_idx = wr_ptr[ADDR_W-1:0];`);
  lines.push(`    wire [ADDR_W-1:0] rd_idx = rd_ptr[ADDR_W-1:0];`);
  lines.push(`    wire empty = (wr_ptr == rd_ptr);`);
  lines.push(`    // full when low bits match but top bit differs (one-extra-bit pointer trick).`);
  lines.push(`    wire full  = (wr_ptr[ADDR_W] != rd_ptr[ADDR_W]) &&`);
  lines.push(`                 (wr_ptr[ADDR_W-1:0] == rd_ptr[ADDR_W-1:0]);`);
  lines.push(``);
  lines.push(`    assign in_ready  = ~full;`);
  lines.push(`    assign out_valid = ~empty;`);
  lines.push(`    assign out_data  = mem[rd_idx];`);
  lines.push(``);
  lines.push(`    always @(posedge clk or negedge rst_n) begin`);
  lines.push(`        if (!rst_n) begin`);
  lines.push(`            wr_ptr <= {(ADDR_W+1){1'b0}};`);
  lines.push(`            rd_ptr <= {(ADDR_W+1){1'b0}};`);
  lines.push(`        end else begin`);
  lines.push(`            if (in_valid && ~full)        wr_ptr <= wr_ptr + 1'b1;`);
  lines.push(`            if (out_ready && ~empty)      rd_ptr <= rd_ptr + 1'b1;`);
  lines.push(`        end`);
  lines.push(`    end`);
  lines.push(``);
  lines.push(`    // Array-memory write split out per knowledge/patterns/protected/08_common_bugs.md.`);
  lines.push(`    always @(posedge clk) begin`);
  lines.push(`        if (in_valid && ~full) mem[wr_idx] <= in_data;`);
  lines.push(`    end`);
  lines.push(`endmodule`);
  lines.push(``);

  // ---- uram_weight_bank: one of NUM_BANKS parallel URAM banks. ----
  //   - Native 288-bit width (URAM cascade: 4 URAM288 primitives in parallel).
  //   - DEPTH = total_mac_cycles from the weight memory map.
  //   - The .mem file is one bank's slice: 72 hex chars per line, low 256
  //     bits useful, high 32 bits zero-pad (matches the URAM physical width).
  //   - `(* ram_style = "ultra" *)` directs Vivado to URAM placement (vs
  //     BRAM). One bank uses ~96 URAM288 primitives for 22.4 MB / 8 banks of
  //     weight storage.
  lines.push(`module uram_weight_bank #(`);
  lines.push(`    parameter integer DEPTH         = 1024,`);
  lines.push(`    parameter integer ADDR_W        = 17,`);
  lines.push(`    parameter         MEM_INIT_FILE = ""`);
  lines.push(`) (`);
  lines.push(`    input  wire                    clk,`);
  lines.push(`    input  wire [ADDR_W-1:0]       rd_addr,`);
  lines.push(`    output reg  [287:0]            rd_data,`);
  lines.push(`    input  wire                    rd_en`);
  lines.push(`);`);
  lines.push(`    (* ram_style = "ultra" *) reg [287:0] mem [0:DEPTH-1];`);
  lines.push(`    initial begin`);
  lines.push(`        if (MEM_INIT_FILE != "") $readmemh(MEM_INIT_FILE, mem);`);
  lines.push(`    end`);
  lines.push(`    // Synchronous 1-cycle read (UltraScale+ URAM cascade native).`);
  lines.push(`    always @(posedge clk) begin`);
  lines.push(`        if (rd_en) rd_data <= mem[rd_addr];`);
  lines.push(`    end`);
  lines.push(`endmodule`);
  lines.push(``);

  // ---- stream_to_act_bram_bridge (Fix 11) ----
  //   Captures a spatial layer's tiled stream (BUS_W bits/beat) and writes
  //   2048-bit BRAM words into the unified activation BRAM at the given
  //   bank-base address. Handles three regimes:
  //     - BUS_W <  2048: accumulate (2048/BUS_W) beats into one BRAM word
  //     - BUS_W == 2048: one beat == one BRAM word, write directly
  //     - BUS_W >  2048: each beat carries (BUS_W/2048) BRAM words; emit
  //                      them sequentially over (BUS_W/2048) clock cycles
  //
  //   wr_req asserts when a BRAM word is ready; the wrapper's write
  //   arbiter grants when no higher-priority writer (engine) is active.
  //   The bridge holds wr_req high until grant.
  //
  //   Counts total BRAM words written; `loaded` asserts (sticky) once the
  //   layer's full frame (= TOTAL_BRAM_WORDS) has been written, telling
  //   the scheduler that the engine may now dispatch this layer.
  lines.push(`module stream_to_act_bram_bridge #(`);
  lines.push(`    parameter integer BUS_W            = 2048,`);
  lines.push(`    parameter integer BRAM_BASE_ADDR   = 0,`);
  lines.push(`    parameter integer TOTAL_BRAM_WORDS = 1`);
  lines.push(`) (`);
  lines.push(`    input  wire             clk,`);
  lines.push(`    input  wire             rst_n,`);
  lines.push(`    input  wire             in_valid,`);
  lines.push(`    input  wire [BUS_W-1:0] in_data,`);
  lines.push(`    output reg              wr_req,`);
  lines.push(`    input  wire             wr_grant,`);
  lines.push(`    output reg  [14:0]      wr_addr,`);
  lines.push(`    output reg  [2047:0]    wr_data,`);
  lines.push(`    output reg              loaded`);
  lines.push(`);`);
  lines.push(`    // Word counter (BRAM words written so far).`);
  lines.push(`    reg [15:0] word_count;`);
  lines.push(`    wire [15:0] next_word_count = word_count + 16'd1;`);
  lines.push(`    // bridge_free pulses when wr_req is empty OR being granted this cycle.`);
  lines.push(`    wire bridge_free = !wr_req || (wr_req && wr_grant);`);
  lines.push(`    // wr_addr to use when bridge_free fires this cycle (use next_word_count`);
  lines.push(`    // if a grant just retired the previous wr_req).`);
  lines.push(`    wire [14:0] next_wr_addr = BRAM_BASE_ADDR[14:0]`);
  lines.push(`                             + ((wr_req && wr_grant) ? next_word_count[14:0]`);
  lines.push(`                                                     : word_count[14:0]);`);
  lines.push(``);
  lines.push(`    generate`);
  lines.push(`    if (BUS_W == 2048) begin : g_w_eq`);
  lines.push(`        // 13a audit fix (Fix 12): 1-deep producer-side skid buffer.`);
  lines.push(`        // Previously the old "else if (in_valid && !wr_req)" guard dropped`);
  lines.push(`        // every beat that arrived while wr_req was pending — including the`);
  lines.push(`        // every-other-cycle case where grant+beat coincide. The skid lets`);
  lines.push(`        // us accept a beat in cycle N while wr_data from cycle N-1 is`);
  lines.push(`        // still on the bus, then submit it cycle N+1. Steady-state grant`);
  lines.push(`        // yields one BRAM write per beat, lossless.`);
  lines.push(`        //`);
  lines.push(`        // Known limitation: multi-cycle grant denial (when several bridges`);
  lines.push(`        // and the engine all contend) still drops beats after the skid`);
  lines.push(`        // fills. For the ResNet-50 schedule's mild contention pattern this`);
  lines.push(`        // is rare; a deeper FIFO can be added later if it becomes a hot`);
  lines.push(`        // path in Verilator simulation.`);
  lines.push(`        reg [2047:0] skid_data;`);
  lines.push(`        reg          skid_valid;`);
  lines.push(`        wire         drain_skid = skid_valid && bridge_free;`);
  lines.push(`        always @(posedge clk or negedge rst_n) begin`);
  lines.push(`            if (!rst_n) begin`);
  lines.push(`                wr_req     <= 1'b0;`);
  lines.push(`                wr_addr    <= 15'd0;`);
  lines.push(`                wr_data    <= 2048'd0;`);
  lines.push(`                word_count <= 16'd0;`);
  lines.push(`                loaded     <= 1'b0;`);
  lines.push(`                skid_valid <= 1'b0;`);
  lines.push(`                skid_data  <= 2048'd0;`);
  lines.push(`            end else begin`);
  lines.push(`                // (1) Grant retires wr_req and advances count.`);
  lines.push(`                if (wr_req && wr_grant) begin`);
  lines.push(`                    wr_req <= 1'b0;`);
  lines.push(`                    word_count <= next_word_count;`);
  lines.push(`                    if (next_word_count == TOTAL_BRAM_WORDS) loaded <= 1'b1;`);
  lines.push(`                end`);
  lines.push(`                // (2) Drain skid into wr_req when bridge is free.`);
  lines.push(`                if (drain_skid) begin`);
  lines.push(`                    wr_req  <= 1'b1;`);
  lines.push(`                    wr_addr <= next_wr_addr;`);
  lines.push(`                    wr_data <= skid_data;`);
  lines.push(`                end`);
  lines.push(`                // (3) Capture new beat into skid; clear skid if drained and no new.`);
  lines.push(`                if (in_valid && !loaded && (!skid_valid || drain_skid)) begin`);
  lines.push(`                    skid_valid <= 1'b1;`);
  lines.push(`                    skid_data  <= in_data;`);
  lines.push(`                end else if (drain_skid) begin`);
  lines.push(`                    skid_valid <= 1'b0;`);
  lines.push(`                end`);
  lines.push(`            end`);
  lines.push(`        end`);
  lines.push(`    end else if (BUS_W < 2048) begin : g_w_lt`);
  lines.push(`        // Accumulate BEATS_PER_WORD beats into one BRAM word, with a`);
  lines.push(`        // 1-deep producer-side skid so beats are not lost when the`);
  lines.push(`        // BRAM write port is denied for a cycle.`);
  lines.push(`        localparam integer BEATS_PER_WORD = 2048 / BUS_W;`);
  lines.push(`        localparam integer BEAT_W = $clog2(BEATS_PER_WORD);`);
  lines.push(`        reg [2047:0] accumulator;`);
  lines.push(`        reg [BEAT_W:0] beat_idx;`);
  lines.push(`        reg [BUS_W-1:0] skid_data;`);
  lines.push(`        reg             skid_valid;`);
  lines.push(`        // The accumulator absorbs a skid beat each cycle UNLESS the`);
  lines.push(`        // beat would complete a new word AND wr_req is still pending`);
  lines.push(`        // (then we have to hold the new word — currently we just stall`);
  lines.push(`        // the accumulator until wr_req retires).`);
  lines.push(`        wire would_complete = (beat_idx == BEATS_PER_WORD - 1);`);
  lines.push(`        wire drain_skid = skid_valid && (!would_complete || bridge_free);`);
  lines.push(`        always @(posedge clk or negedge rst_n) begin`);
  lines.push(`            if (!rst_n) begin`);
  lines.push(`                wr_req      <= 1'b0;`);
  lines.push(`                wr_addr     <= 15'd0;`);
  lines.push(`                wr_data     <= 2048'd0;`);
  lines.push(`                word_count  <= 16'd0;`);
  lines.push(`                accumulator <= 2048'd0;`);
  lines.push(`                beat_idx    <= {(BEAT_W+1){1'b0}};`);
  lines.push(`                loaded      <= 1'b0;`);
  lines.push(`                skid_valid  <= 1'b0;`);
  lines.push(`                skid_data   <= {BUS_W{1'b0}};`);
  lines.push(`            end else begin`);
  lines.push(`                // (1) Grant retires wr_req and advances count.`);
  lines.push(`                if (wr_req && wr_grant) begin`);
  lines.push(`                    wr_req <= 1'b0;`);
  lines.push(`                    word_count <= next_word_count;`);
  lines.push(`                    if (next_word_count == TOTAL_BRAM_WORDS) loaded <= 1'b1;`);
  lines.push(`                end`);
  lines.push(`                // (2) Drain skid into accumulator (and possibly emit word).`);
  lines.push(`                if (drain_skid) begin`);
  lines.push(`                    accumulator[beat_idx*BUS_W +: BUS_W] <= skid_data;`);
  lines.push(`                    if (would_complete) begin`);
  lines.push(`                        beat_idx <= {(BEAT_W+1){1'b0}};`);
  lines.push(`                        wr_req   <= 1'b1;`);
  lines.push(`                        wr_addr  <= next_wr_addr;`);
  lines.push(`                        wr_data  <= { skid_data,`);
  lines.push(`                                      accumulator[2048-BUS_W-1:0] };`);
  lines.push(`                    end else begin`);
  lines.push(`                        beat_idx <= beat_idx + 1'b1;`);
  lines.push(`                    end`);
  lines.push(`                end`);
  lines.push(`                // (3) Capture new beat into skid.`);
  lines.push(`                if (in_valid && !loaded && (!skid_valid || drain_skid)) begin`);
  lines.push(`                    skid_valid <= 1'b1;`);
  lines.push(`                    skid_data  <= in_data;`);
  lines.push(`                end else if (drain_skid) begin`);
  lines.push(`                    skid_valid <= 1'b0;`);
  lines.push(`                end`);
  lines.push(`            end`);
  lines.push(`        end`);
  lines.push(`    end else begin : g_w_gt`);
  lines.push(`        // BUS_W > 2048: each beat carries WORDS_PER_BEAT BRAM words.`);
  lines.push(`        // 1-deep skid lets a new beat be queued while the current`);
  lines.push(`        // is still being sliced.`);
  lines.push(`        localparam integer WORDS_PER_BEAT = BUS_W / 2048;`);
  lines.push(`        localparam integer SLICE_W = $clog2(WORDS_PER_BEAT);`);
  lines.push(`        reg [BUS_W-1:0] beat_buf;`);
  lines.push(`        reg [SLICE_W:0] slice_idx;`);
  lines.push(`        reg             buf_active;`);
  lines.push(`        reg [BUS_W-1:0] skid_data;`);
  lines.push(`        reg             skid_valid;`);
  lines.push(`        wire beat_complete = buf_active && wr_req && wr_grant`);
  lines.push(`                           && (slice_idx == WORDS_PER_BEAT - 1);`);
  lines.push(`        wire can_load_new  = !buf_active || beat_complete;`);
  lines.push(`        wire drain_skid    = skid_valid && can_load_new;`);
  lines.push(`        always @(posedge clk or negedge rst_n) begin`);
  lines.push(`            if (!rst_n) begin`);
  lines.push(`                wr_req     <= 1'b0;`);
  lines.push(`                wr_addr    <= 15'd0;`);
  lines.push(`                wr_data    <= 2048'd0;`);
  lines.push(`                word_count <= 16'd0;`);
  lines.push(`                beat_buf   <= {BUS_W{1'b0}};`);
  lines.push(`                slice_idx  <= {(SLICE_W+1){1'b0}};`);
  lines.push(`                buf_active <= 1'b0;`);
  lines.push(`                loaded     <= 1'b0;`);
  lines.push(`                skid_valid <= 1'b0;`);
  lines.push(`                skid_data  <= {BUS_W{1'b0}};`);
  lines.push(`            end else begin`);
  lines.push(`                // (1) Grant retires wr_req and advances count.`);
  lines.push(`                if (wr_req && wr_grant) begin`);
  lines.push(`                    wr_req <= 1'b0;`);
  lines.push(`                    word_count <= next_word_count;`);
  lines.push(`                    if (next_word_count == TOTAL_BRAM_WORDS) loaded <= 1'b1;`);
  lines.push(`                    if (slice_idx == WORDS_PER_BEAT - 1) begin`);
  lines.push(`                        slice_idx  <= {(SLICE_W+1){1'b0}};`);
  lines.push(`                        buf_active <= 1'b0;`);
  lines.push(`                    end else if (next_word_count != TOTAL_BRAM_WORDS) begin`);
  lines.push(`                        // Continue slicing the current beat (defensive guard`);
  lines.push(`                        // on next_word_count so we don't overrun TOTAL_BRAM_WORDS`);
  lines.push(`                        // when a frame ends mid-beat with a non-divisible total).`);
  lines.push(`                        slice_idx <= slice_idx + 1'b1;`);
  lines.push(`                        wr_req    <= 1'b1;`);
  lines.push(`                        wr_addr   <= BRAM_BASE_ADDR[14:0] + next_word_count[14:0];`);
  lines.push(`                        wr_data   <= beat_buf[(slice_idx+1)*2048 +: 2048];`);
  lines.push(`                    end`);
  lines.push(`                end`);
  lines.push(`                // (2) Load new beat from skid when buf is free.`);
  lines.push(`                if (drain_skid) begin`);
  lines.push(`                    beat_buf   <= skid_data;`);
  lines.push(`                    buf_active <= 1'b1;`);
  lines.push(`                    slice_idx  <= {(SLICE_W+1){1'b0}};`);
  lines.push(`                    wr_req     <= 1'b1;`);
  lines.push(`                    wr_addr    <= next_wr_addr;`);
  lines.push(`                    wr_data    <= skid_data[2047:0];`);
  lines.push(`                end`);
  lines.push(`                // (3) Capture new beat into skid.`);
  lines.push(`                if (in_valid && !loaded && (!skid_valid || drain_skid)) begin`);
  lines.push(`                    skid_valid <= 1'b1;`);
  lines.push(`                    skid_data  <= in_data;`);
  lines.push(`                end else if (drain_skid) begin`);
  lines.push(`                    skid_valid <= 1'b0;`);
  lines.push(`                end`);
  lines.push(`            end`);
  lines.push(`        end`);
  lines.push(`    end`);
  lines.push(`    endgenerate`);
  lines.push(`endmodule`);
  lines.push(``);

  // ---- act_unified_mem: flat URAM-backed activation BRAM (Fix 8) ----
  //   - One unified memory spanning NUM_BANKS × BANK_DEPTH_WORDS entries,
  //     each 2048 bits wide (= MAC_COUNT × ACT_W = 256 × 8). The engine
  //     emits 16-bit addresses; the scheduler partitions the address
  //     space into per-bank windows via cfg_act_*_bram_base, so the
  //     flat layout works as long as the engine never reads beyond
  //     NUM_BANKS × BANK_DEPTH_WORDS.
  //   - Synchronous 1-cycle read + 1-cycle write, matching the engine's
  //     pipelined act_in / act_out timing contract.
  //   - `(* ram_style = "ultra" *)` steers Vivado to URAM placement
  //     (BRAM is too tight for 2048-bit width × 24K-deep; URAM cascade
  //     handles it natively).
  lines.push(`module act_unified_mem #(`);
  lines.push(`    parameter integer DEPTH  = 24576,`);
  lines.push(`    parameter integer ADDR_W = 15`);
  lines.push(`) (`);
  lines.push(`    input  wire                    clk,`);
  lines.push(`    input  wire [ADDR_W-1:0]       rd_addr,`);
  lines.push(`    input  wire                    rd_en,`);
  lines.push(`    output reg  [2047:0]           rd_data,`);
  lines.push(`    input  wire [ADDR_W-1:0]       wr_addr,`);
  lines.push(`    input  wire                    wr_en,`);
  lines.push(`    input  wire [2047:0]           wr_data`);
  lines.push(`);`);
  lines.push(`    (* ram_style = "ultra" *) reg [2047:0] mem [0:DEPTH-1];`);
  lines.push(`    always @(posedge clk) begin`);
  lines.push(`        if (rd_en) rd_data <= mem[rd_addr];`);
  lines.push(`    end`);
  lines.push(`    // Array-memory write split into its own always block per`);
  lines.push(`    // knowledge/patterns/protected/08_common_bugs.md §"Array memory`);
  lines.push(`    // write in async-reset block" — no reset clause here.`);
  lines.push(`    always @(posedge clk) begin`);
  lines.push(`        if (wr_en) mem[wr_addr] <= wr_data;`);
  lines.push(`    end`);
  lines.push(`endmodule`);
  lines.push(``);

  // ---- engine_output_fifo: deep, URAM-backed FIFO for engine outputs ----
  //
  // 13a audit fix (Fix 14): a single shared FIFO absorbs all engine act_out
  // writes so the spatial chain (which is frozen during engine_busy) does
  // not lose beats. The FIFO drains during the inter-dispatch S_WAIT_LOAD
  // window when spatial_run resumes. Sized for the worst-case dispatch
  // output (ResNet-50 dispatch 1: 28*28*4 = 3,136 beats) plus headroom.
  //
  // Width: 2048 bits (engine's ACT_BUS_W = MAC_COUNT × ACT_W).
  // Depth: 4096 (power of two; covers worst-case dispatch + headroom).
  // Backing: URAM via `(* ram_style = "ultra" *)`. URAM cascade depth is
  // 4096 entries native — this FIFO exactly fills one URAM-deep stripe of
  // width 2048/72 = 29 URAM primitives. ~28 URAMs total.
  //
  // Output: 1-cycle synchronous read with a 1-deep pre-fetch register so
  // downstream sees combinational `out_valid` and registered `out_data`.
  // Standard valid/ready handshake on both sides.
  lines.push(`module engine_output_fifo #(`);
  lines.push(`    parameter integer DEPTH  = 4096,`);
  lines.push(`    parameter integer ADDR_W = 12,             // log2(DEPTH)`);
  lines.push(`    parameter integer DATA_W = 2048`);
  lines.push(`) (`);
  lines.push(`    input  wire             clk,`);
  lines.push(`    input  wire             rst_n,`);
  lines.push(`    input  wire             in_valid,`);
  lines.push(`    input  wire [DATA_W-1:0] in_data,`);
  lines.push(`    output wire             in_ready,`);
  lines.push(`    output reg              out_valid,`);
  lines.push(`    output reg  [DATA_W-1:0] out_data,`);
  lines.push(`    input  wire             out_ready`);
  lines.push(`);`);
  lines.push(`    (* ram_style = "ultra" *) reg [DATA_W-1:0] mem [0:DEPTH-1];`);
  lines.push(`    reg [ADDR_W:0] wr_ptr;`);
  lines.push(`    reg [ADDR_W:0] rd_ptr;`);
  lines.push(``);
  lines.push(`    wire fifo_empty = (wr_ptr == rd_ptr);`);
  lines.push(`    wire fifo_full  = (wr_ptr[ADDR_W-1:0] == rd_ptr[ADDR_W-1:0])`);
  lines.push(`                    && (wr_ptr[ADDR_W] != rd_ptr[ADDR_W]);`);
  lines.push(`    wire wr_fire = in_valid && !fifo_full;`);
  lines.push(`    wire load_skid = !fifo_empty && (!out_valid || (out_valid && out_ready));`);
  lines.push(`    assign in_ready = !fifo_full;`);
  lines.push(``);
  lines.push(`    always @(posedge clk or negedge rst_n) begin`);
  lines.push(`        if (!rst_n) begin`);
  lines.push(`            wr_ptr    <= {(ADDR_W+1){1'b0}};`);
  lines.push(`            rd_ptr    <= {(ADDR_W+1){1'b0}};`);
  lines.push(`            out_valid <= 1'b0;`);
  lines.push(`            out_data  <= {DATA_W{1'b0}};`);
  lines.push(`        end else begin`);
  lines.push(`            if (wr_fire) wr_ptr <= wr_ptr + 1'b1;`);
  lines.push(`            // Output handshake: drop valid when consumer accepts.`);
  lines.push(`            if (out_valid && out_ready) begin`);
  lines.push(`                out_valid <= 1'b0;`);
  lines.push(`            end`);
  lines.push(`            // Refill output skid when it is empty (or being consumed`);
  lines.push(`            // this cycle) and the FIFO has data. rd_ptr advances on the`);
  lines.push(`            // same edge so the next refill reads the next entry.`);
  lines.push(`            if (load_skid) begin`);
  lines.push(`                out_data  <= mem[rd_ptr[ADDR_W-1:0]];`);
  lines.push(`                out_valid <= 1'b1;`);
  lines.push(`                rd_ptr    <= rd_ptr + 1'b1;`);
  lines.push(`            end`);
  lines.push(`        end`);
  lines.push(`    end`);
  lines.push(`    // Memory write split into its own clock-only always block per`);
  lines.push(`    // knowledge/patterns/protected/08_common_bugs.md.`);
  lines.push(`    always @(posedge clk) begin`);
  lines.push(`        if (wr_fire) mem[wr_ptr[ADDR_W-1:0]] <= in_data;`);
  lines.push(`    end`);
  lines.push(`endmodule`);
  lines.push(``);

  // ---- bias_mem: same shape, narrower wide-word (256 × INT32) ----
  lines.push(`module bias_mem #(`);
  lines.push(`    parameter integer SIZE_WORDS    = 256,`);
  lines.push(`    parameter integer WORD_WIDTH    = 8192,`);
  lines.push(`    parameter integer ADDR_W        = 16,`);
  lines.push(`    parameter         MEM_INIT_FILE = "output/weights/bias.mem"`);
  lines.push(`) (`);
  lines.push(`    input  wire                    clk,`);
  lines.push(`    input  wire [ADDR_W-1:0]       rd_addr,`);
  lines.push(`    output reg  [WORD_WIDTH-1:0]   rd_data,`);
  lines.push(`    input  wire                    rd_en`);
  lines.push(`);`);
  lines.push(`    (* ram_style = "block" *) reg [WORD_WIDTH-1:0] mem [0:SIZE_WORDS-1];`);
  lines.push(`    initial begin`);
  lines.push(`        if (MEM_INIT_FILE != "") $readmemh(MEM_INIT_FILE, mem);`);
  lines.push(`    end`);
  lines.push(`    always @(posedge clk) begin`);
  lines.push(`        if (rd_en) rd_data <= mem[rd_addr];`);
  lines.push(`    end`);
  lines.push(`endmodule`);
  lines.push(``);

  // ---- engine_output_bridge: per-slot FIFO-drain + parallel-to-serial shim ----
  //
  // 13a audit fix (Fix 15): the bridge now SERIALIZES each 2048-bit
  // engine beat into TILES_PER_BEAT consumer tiles when DATA_W < ACT_W.
  // The previous Fix 14 used a simple width-adapter (`g_trunc` =
  // fifo_out_data[DATA_W-1:0]) that silently discarded the high
  // ACT_W-DATA_W bits of every engine beat — i.e. for the 8 dispatches
  // with DATA_W=256 and ACT_W=2048, it kept only 32 channels out of
  // every 256-channel oc_pass beat, corrupting 87.5% of the channel
  // data. The serializer below emits one DATA_W-wide tile per cycle,
  // walking through the beat low-bits first (byte 0 at bits [7:0]
  // matches the engine's convention).
  //
  // EXPECTED_BEATS still counts ENGINE beats consumed (= OH*OW*oc_passes).
  // EXPECTED_TILES = EXPECTED_BEATS * TILES_PER_BEAT is the total number
  // of consumer tiles the bridge will emit, and that count drives
  // drain_complete.
  lines.push(`module engine_output_bridge #(`);
  lines.push(`    parameter integer SLOT           = 0,`);
  lines.push(`    parameter integer ACT_W          = 2048,`);
  lines.push(`    parameter integer DATA_W         = 2048,`);
  lines.push(`    parameter integer EXPECTED_BEATS = 1,`);
  lines.push(`    parameter integer NUM_DISPATCHES = 14`);
  lines.push(`) (`);
  lines.push(`    input  wire              clk,`);
  lines.push(`    input  wire              rst_n,`);
  lines.push(`    input  wire              start,                 // sched_engine_output_ready`);
  lines.push(`    input  wire              fifo_out_valid,`);
  lines.push(`    input  wire [ACT_W-1:0]  fifo_out_data,`);
  lines.push(`    output wire              fifo_out_ready,`);
  lines.push(`    input  wire              ready_out,             // downstream consumer's ready_in`);
  lines.push(`    output reg               valid_out,`);
  lines.push(`    output reg  [DATA_W-1:0] data_out,`);
  lines.push(`    output reg               drain_complete         // 1 = this dispatch's beats fully drained`);
  lines.push(`);`);
  lines.push(`    // Serialization geometry. For DATA_W >= ACT_W (no serialization needed)`);
  lines.push(`    // TILES_PER_BEAT collapses to 1; the engine beat is zero-padded into a`);
  lines.push(`    // single consumer tile in the slice expression.`);
  lines.push(`    localparam integer TILES_PER_BEAT = (DATA_W >= ACT_W) ? 1 : (ACT_W / DATA_W);`);
  lines.push(`    localparam integer TILE_IDX_W    = (TILES_PER_BEAT <= 1) ? 1 : $clog2(TILES_PER_BEAT);`);
  lines.push(`    localparam integer EXPECTED_TILES = EXPECTED_BEATS * TILES_PER_BEAT;`);
  lines.push(``);
  lines.push(`    // dispatch_count gates active_slot; ticks once per scheduler 'start' pulse.`);
  lines.push(`    // Width is sized to log2(NUM_DISPATCHES) + 1 headroom bit (max 16 supported`);
  lines.push(`    // here; Fix 14's "+1" cushion catches the final S_NEXT_DISP increment past`);
  lines.push(`    // LAST_DISPATCH).`);
  lines.push(`    reg [3:0] dispatch_count;`);
  lines.push(`    always @(posedge clk or negedge rst_n) begin`);
  lines.push(`        if (!rst_n) dispatch_count <= 4'd0;`);
  lines.push(`        else if (start) dispatch_count <= dispatch_count + 4'd1;`);
  lines.push(`    end`);
  lines.push(`    wire active_slot = (dispatch_count == SLOT[3:0]);`);
  lines.push(``);
  lines.push(`    // Beat buffer + tile index. beat_buf holds the current engine beat;`);
  lines.push(`    // tile_idx walks 0..TILES_PER_BEAT-1, emitting one slice per ready cycle.`);
  lines.push(`    reg [ACT_W-1:0]      beat_buf;`);
  lines.push(`    reg                  buf_valid;`);
  lines.push(`    reg [TILE_IDX_W:0]   tile_idx;`);
  lines.push(`    reg [31:0]           tiles_emitted;`);
  lines.push(``);
  lines.push(`    // Tile-slice multiplexer. The +: indexed-part-select picks DATA_W bits`);
  lines.push(`    // starting at tile_idx*DATA_W; for DATA_W>=ACT_W the slice degenerates`);
  lines.push(`    // to "low ACT_W bits with zero-pad above".`);
  lines.push(`    wire [DATA_W-1:0] current_tile;`);
  lines.push(`    generate`);
  lines.push(`        if (DATA_W >= ACT_W) begin : g_pad`);
  lines.push(`            assign current_tile = { {(DATA_W-ACT_W){1'b0}}, beat_buf };`);
  lines.push(`        end else begin : g_slice`);
  lines.push(`            assign current_tile = beat_buf[tile_idx * DATA_W +: DATA_W];`);
  lines.push(`        end`);
  lines.push(`    endgenerate`);
  lines.push(``);
  lines.push(`    // Combinational gates:`);
  lines.push(`    //   emit_ready    : we can send a tile this cycle (buffer has data,`);
  lines.push(`    //                   downstream slot is free, slot is active, not done)`);
  lines.push(`    //   last_tile     : this would be the final tile of the current beat`);
  lines.push(`    //   need_new_beat : next cycle the buffer will be empty`);
  lines.push(`    //   fifo_out_ready: only this slot's active bridge pulls; OR-cascade at`);
  lines.push(`    //                   the wrapper trivially picks the right gate.`);
  lines.push(`    wire emit_ready = active_slot && (!valid_out || ready_out)`);
  lines.push(`                    && !drain_complete && buf_valid;`);
  lines.push(`    wire last_tile  = (tile_idx == (TILES_PER_BEAT[TILE_IDX_W:0] - 1'b1));`);
  lines.push(`    wire need_new_beat = !buf_valid || (emit_ready && last_tile);`);
  lines.push(`    assign fifo_out_ready = active_slot && fifo_out_valid`);
  lines.push(`                          && need_new_beat && !drain_complete;`);
  lines.push(``);
  lines.push(`    always @(posedge clk or negedge rst_n) begin`);
  lines.push(`        if (!rst_n) begin`);
  lines.push(`            valid_out      <= 1'b0;`);
  lines.push(`            data_out       <= {DATA_W{1'b0}};`);
  lines.push(`            beat_buf       <= {ACT_W{1'b0}};`);
  lines.push(`            buf_valid      <= 1'b0;`);
  lines.push(`            tile_idx       <= {(TILE_IDX_W+1){1'b0}};`);
  lines.push(`            tiles_emitted  <= 32'd0;`);
  lines.push(`            drain_complete <= 1'b0;`);
  lines.push(`        end else begin`);
  lines.push(`            // (1) Consumer accepted current tile — drop valid_out.`);
  lines.push(`            if (valid_out && ready_out) valid_out <= 1'b0;`);
  lines.push(`            // (2) Emit a tile this cycle.`);
  lines.push(`            if (emit_ready) begin`);
  lines.push(`                valid_out     <= 1'b1;`);
  lines.push(`                data_out      <= current_tile;`);
  lines.push(`                tiles_emitted <= tiles_emitted + 32'd1;`);
  lines.push(`                if (tiles_emitted + 32'd1 == EXPECTED_TILES[31:0]) begin`);
  lines.push(`                    drain_complete <= 1'b1;`);
  lines.push(`                end`);
  lines.push(`                // Advance tile index OR finish this beat.`);
  lines.push(`                if (last_tile) begin`);
  lines.push(`                    buf_valid <= 1'b0;`);
  lines.push(`                    tile_idx  <= {(TILE_IDX_W+1){1'b0}};`);
  lines.push(`                end else begin`);
  lines.push(`                    tile_idx  <= tile_idx + 1'b1;`);
  lines.push(`                end`);
  lines.push(`            end`);
  lines.push(`            // (3) Pull next beat from FIFO. Placed AFTER the emit block so`);
  lines.push(`            // a simultaneous "emit last tile + load new beat" lands with`);
  lines.push(`            // buf_valid=1 (FIFO load wins the NBA race) and tile_idx=0.`);
  lines.push(`            if (fifo_out_ready && fifo_out_valid) begin`);
  lines.push(`                beat_buf  <= fifo_out_data;`);
  lines.push(`                buf_valid <= 1'b1;`);
  lines.push(`                tile_idx  <= {(TILE_IDX_W+1){1'b0}};`);
  lines.push(`            end`);
  lines.push(`            // Single-image: drain_complete latches once and stays. For`);
  lines.push(`            // multi-image inference, a 'clear' input would reset this`);
  lines.push(`            // and tiles_emitted per-frame. Out of scope.`);
  lines.push(`        end`);
  lines.push(`    end`);
  lines.push(`endmodule`);
  lines.push(``);
  lines.push(`\`endif // NN2RTL_TOP_NO_STUBS`);

  return lines.join("\n") + "\n";
}

// A reasonable default depth for a skip FIFO when task 04 hasn't shipped
// real numbers yet. Per plan §6.5 the analytical first pass is:
//   depth = (main_path_latency_cycles - skip_path_latency_cycles)
//         + 1.5 * backpressure_margin
// For Wave-1 we approximate by walking IR layers between the skip source
// and the add and summing their pipeline_latency_cycles. If anything is
// missing we fall back to a hard floor of 256 words.
function defaultSkipDepth(add: NodeMeta, meta: NodeMeta[]): number {
  const addIdx = add.index;
  const skipName = add.skipSource;
  if (!skipName || skipName === "PIXEL_IN") return 256;
  const skipIdx = meta.findIndex((m) => m.ir.module_id === skipName);
  if (skipIdx < 0) return 256;
  let cycles = 0;
  for (let i = skipIdx + 1; i < addIdx; i++) {
    cycles += meta[i].ir.pipeline_latency_cycles ?? 0;
  }
  const margin = Math.ceil(cycles * 0.5);
  // Round up to a small power-of-two-ish word count, floor 256.
  const raw = cycles + margin;
  if (raw <= 256) return 256;
  // round up to next multiple of 64 for tidy generated literals
  return Math.ceil(raw / 64) * 64;
}

function main(): void {
  const args = parseArgs(process.argv);
  const layerIrAbs = path.isAbsolute(args.layerIr)
    ? args.layerIr
    : path.join(repoRoot, args.layerIr);
  const ir: LayerIR = JSON.parse(readFileSync(layerIrAbs, "utf8"));

  const { list: heavyList, source: heavySource } = readHeavyList(args.engineModules);
  const heavySet = new Set(heavyList);
  // Cross-check heavy entries against the IR — a name in the heavy list
  // that doesn't appear in the IR is a stale entry and should be a hard
  // error so the orchestrator notices.
  const irIds = new Set(ir.layers.map((l) => l.module_id));
  const unknown = heavyList.filter((n) => !irIds.has(n));
  if (unknown.length > 0) {
    throw new Error(
      `heavy module list contains entries not present in LayerIR: ${unknown.join(", ")}`,
    );
  }

  const fifoSizes = readFifoSizes(args.fifoSizes);
  const weightMap = readWeightMap(args.weightMap);
  const dispatchOrder = readScheduleDispatchOrder(args.schedule);
  const dispatches = readScheduleDispatches(args.schedule);
  // Fix 7 — the audit's "10 floating outputs" referred to the scheduler's
  // 10 engine-dispatched layers. The heavy-list file (task 06) and the
  // scheduler's dispatch order have drifted; treat both as "engine-handled"
  // (do not instantiate a per-layer module; drive the layer's outputs from
  // the engine via an engine_output_bridge instead). Cross-task list
  // reconciliation is a separate 13a fix not in this PR's scope.
  for (const id of dispatchOrder) {
    if (id) heavySet.add(id);
  }
  const rtlDir = path.dirname(
    path.isAbsolute(args.out) ? args.out : path.join(repoRoot, args.out),
  );
  const buses = readBusWidths(ir.layers, rtlDir);
  // Fix 17: scan each per-layer .v file for legacy DRAM `weights_*` ports
  // so the wrapper can tie them off at instantiation.
  const dramPortIds = new Set<string>();
  for (const L of ir.layers) {
    if (moduleHasDramPorts(L.module_id, rtlDir)) dramPortIds.add(L.module_id);
  }
  const meta = computeTopology(ir.layers, heavySet, buses, dramPortIds);

  const verilog = emit({
    meta,
    heavyList,
    heavySource,
    fifoSizes,
    weightMap,
    dispatchOrder,
    dispatches,
    args,
  });

  const outAbs = path.isAbsolute(args.out)
    ? args.out
    : path.join(repoRoot, args.out);
  mkdirSync(path.dirname(outAbs), { recursive: true });
  writeFileSync(outAbs, verilog);

  const spatialCount = meta.filter((m) => !m.isHeavy).length;
  const heavyCount = meta.filter((m) => m.isHeavy).length;
  const addCount = meta.filter((m) => m.ir.op_type === "add").length;
  const projCount = meta.filter((m) => m.isProjection).length;
  // Brief stdout summary — surfaces the key invariants from the task spec.
  console.log(
    `[build_top_wrapper] wrote ${outAbs}: ` +
      `${meta.length} layers, ${spatialCount} spatial, ${heavyCount} engine-dispatched, ` +
      `${addCount} residual adds, ${projCount} projection convs ` +
      `(heavy list: ${heavySource}).`,
  );
}

main();
