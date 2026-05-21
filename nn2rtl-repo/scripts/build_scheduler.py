#!/usr/bin/env python3
"""Build the deterministic scheduler FSM Verilog + JSON sidecar.

Reads `output/layer_ir.json`, a heavy-module list (the modules that the
shared engine runs sequentially), and the URAM weight memory map. Emits:

  - output/rtl/nn2rtl_scheduler.v
  - output/rtl/nn2rtl_scheduler_schedule.json

The Verilog module is the AXI4-Lite master that configures the engine's
config_register_block (task 10) before each dispatch, then waits for the
engine to complete. It also drives activation BRAM bank-selection signals
and a `spatial_stall` backpressure signal.

This script is pure deterministic Python — no LLM. Running twice on the
same inputs produces byte-identical output.

Register map source-of-truth: docs/agent_tasks/10_engine_config_register_block.md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Register map (must match docs/agent_tasks/10_engine_config_register_block.md)
# ---------------------------------------------------------------------------
REG_INPUT_CHANNELS     = 0x00
REG_OUTPUT_CHANNELS    = 0x04
REG_KERNEL_H_W         = 0x08
REG_STRIDE_H_W         = 0x0C
REG_PADDING_H_W        = 0x10
REG_INPUT_H_W          = 0x14
REG_OUTPUT_H_W         = 0x18
REG_WEIGHT_BASE_WORD   = 0x1C
REG_BIAS_BASE_WORD     = 0x20
REG_SCALE_MULT         = 0x24
REG_SCALE_SHIFT_AND_ZP = 0x28
REG_CONTROL            = 0x2C
# 0x30 STATUS is read-only; the scheduler polls engine_done via the
# top-level handshake signal, not via this register.
REG_ACT_IN_BASE        = 0x34  # added by task 10 implementer; per-bank BRAM base for engine activation reads
REG_ACT_OUT_BASE       = 0x38  # per-bank BRAM base for engine activation writes

# Write order: 13 config registers, written sequentially via AXI4-Lite.
#
# 13a audit fix (Fix 11): the original 14th step was `CONTROL_START` writing
# 1 to wdata[0], which would trigger the engine via config_register_block's
# `start_trigger = (axi_start_write | engine_start_ext) & ~engine_busy_in`.
# But the engine must NOT start until the input loader has finished filling
# the activation BRAM (S_WAIT_LOAD), and the AXI write fires several cycles
# BEFORE that wait — so the engine would run on a half-filled BRAM. Drop
# the CONTROL.start AXI write; the engine_start_ext pin (pulsed by the
# scheduler in S_PULSE_START, after S_WAIT_LOAD) is now the sole trigger.
WRITE_SEQUENCE = [
    ("INPUT_CHANNELS",     REG_INPUT_CHANNELS),
    ("OUTPUT_CHANNELS",    REG_OUTPUT_CHANNELS),
    ("KERNEL_H_W",         REG_KERNEL_H_W),
    ("STRIDE_H_W",         REG_STRIDE_H_W),
    ("PADDING_H_W",        REG_PADDING_H_W),
    ("INPUT_H_W",          REG_INPUT_H_W),
    ("OUTPUT_H_W",         REG_OUTPUT_H_W),
    ("WEIGHT_BASE_WORD",   REG_WEIGHT_BASE_WORD),
    ("BIAS_BASE_WORD",     REG_BIAS_BASE_WORD),
    ("SCALE_MULT",         REG_SCALE_MULT),
    ("SCALE_SHIFT_AND_ZP", REG_SCALE_SHIFT_AND_ZP),
    ("ACT_IN_BASE",        REG_ACT_IN_BASE),
    ("ACT_OUT_BASE",       REG_ACT_OUT_BASE),
]
NUM_WRITE_STEPS = len(WRITE_SEQUENCE)  # 13

NUM_BANKS = 6  # fixed BRAM bank pool
# Per-bank BRAM word budget. Each activation tensor allocated to a bank
# starts at this multiple of the bank index, giving non-overlapping
# regions inside one flat unified BRAM (the wrapper instantiates one
# unified URAM-backed memory of NUM_BANKS × BANK_DEPTH_WORDS entries).
#
# Sizing rationale: the largest heavy-layer activation is node_conv_250's
# output (28×28×1024 → pixel_index × oc_passes + oc_pass_idx peaks at
# 3135), so we need at least 3136 words/bank. 4096 is the next power of
# 2 and aligns exactly with one URAM288 cascade-depth lane (4096 entries),
# so depth-bumps cost zero URAM blocks compared to 2048.
BANK_DEPTH_WORDS = 4096

# Fallback heavy list (Wave 1 — used until task 06 produces the real list).
# Source: docs/agent_tasks/02_layerir_to_wrapper_generator.md §Heavy module list.
FALLBACK_HEAVY_MODULES = [
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
]


def detect_repo_root(script_path: Path) -> Path:
    override = os.environ.get("NN2RTL_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return script_path.resolve().parent.parent


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--network", default="resnet-50")
    p.add_argument("--layer-ir", default="output/layer_ir.json")
    p.add_argument(
        "--engine-modules",
        default="docs/agent_tasks/06_phase1_compression_candidates_HEAVY.txt",
    )
    p.add_argument("--weight-map", default="output/weights/weight_memory_map.json")
    p.add_argument("--bias-map", default="output/weights/bias_memory_map.json")
    p.add_argument("--out-verilog", default="output/rtl/nn2rtl_scheduler.v")
    p.add_argument(
        "--out-schedule", default="output/rtl/nn2rtl_scheduler_schedule.json"
    )
    return p.parse_args(argv)


def load_heavy_modules(path: Path):
    if not path.exists():
        return list(FALLBACK_HEAVY_MODULES), True
    mods = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            mods.append(line)
    if not mods:
        return list(FALLBACK_HEAVY_MODULES), True
    return mods, False


def load_weight_map(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def load_bias_map(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def annotate_residual_blocks(layers):
    """For each `add` layer, identify (main_idx, skip_src_idx, has_downsample).

    The LayerIR is a linear list; ResNet-50's residual blocks are detectable
    structurally:
      - Each block ends in an `add`.
      - If the two layers immediately before the `add` are both `conv2d`
        (no relu between), the second-to-last conv is the main path's final
        conv and the last conv is the downsample on the skip path.
      - Otherwise the skip activation comes from the most-recent
        "block-boundary output" — the relu produced by the previous block's
        terminating add+relu, or (for the first residual block) the stem's
        maxpool output.
    """
    add_info = {}  # add_idx -> (main_idx, skip_idx, has_downsample)
    last_boundary_idx = None
    for i, L in enumerate(layers):
        op = L["op_type"]
        if op == "maxpool":
            last_boundary_idx = i
        elif op == "relu" and i > 0 and layers[i - 1]["op_type"] == "add":
            last_boundary_idx = i
        elif op == "add":
            # Count consecutive conv2ds immediately preceding the add.
            convs_immediate = []
            j = i - 1
            while j >= 0 and layers[j]["op_type"] == "conv2d":
                convs_immediate.append(j)
                j -= 1
            if len(convs_immediate) >= 2:
                # Downsample: last conv (highest idx) before add is the
                # downsample on the skip path; the conv before that is the
                # main path's final conv.
                main_idx = convs_immediate[1]
                skip_idx = convs_immediate[0]
                has_downsample = True
            else:
                # No downsample: main = last conv before the add; skip =
                # previous boundary.
                j = i - 1
                while j >= 0 and layers[j]["op_type"] != "conv2d":
                    j -= 1
                main_idx = j
                skip_idx = last_boundary_idx
                has_downsample = False
            add_info[i] = (main_idx, skip_idx, has_downsample)
    return add_info


def scale_factor_to_mult_shift(scale_factor):
    """Encode a float scale as (mult[31:0], shift[5:0]) such that
    `mult >> shift` approximates the scale. Aim for ~30 bits of precision."""
    if scale_factor is None or scale_factor <= 0.0:
        return (0, 0)
    target_bits = 30
    shift = int(round(target_bits - math.log2(scale_factor)))
    shift = max(0, min(62, shift))
    mult = int(round(scale_factor * (1 << shift)))
    if mult >= (1 << 31):
        excess = mult.bit_length() - 31
        shift = max(0, shift - excess)
        mult = int(round(scale_factor * (1 << shift)))
        mult = min(mult, (1 << 31) - 1)
    return (mult & 0xFFFFFFFF, shift & 0x3F)


def output_bytes(L):
    n = 1
    for d in L["output_shape"]:
        n *= d
    return n


def build_producer_consumer_graph(layers, add_info):
    """For each layer index, what producer layer-indices feed its input(s)?

    Network input is producer index -1 (synthetic source).

    Rules:
      - relu, maxpool, normal conv2d: input = previous layer's output (linear chain).
      - downsample conv2d: input = the activation that started the current
        residual block (i.e., the layer that wrote the block-input tensor).
        Detected as a conv2d whose immediate next layer is `add` and whose
        immediate previous layer is also a conv2d (no relu between).
      - add: input = (main_source_idx, skip_source_idx) from add_info.
    """
    producers = {}
    chain_pred = -1            # synthetic network input producer
    last_block_boundary = -1   # producer index of the most recent residual-block-start activation
    for i, L in enumerate(layers):
        op = L["op_type"]
        if op == "maxpool":
            producers[i] = [chain_pred]
            chain_pred = i
            last_block_boundary = i
        elif op == "relu":
            producers[i] = [chain_pred]
            chain_pred = i
            if i > 0 and layers[i - 1]["op_type"] == "add":
                last_block_boundary = i  # post-add relu starts a new block
        elif op == "conv2d":
            is_downsample = (
                i + 1 < len(layers)
                and layers[i + 1]["op_type"] == "add"
                and i > 0
                and layers[i - 1]["op_type"] == "conv2d"
            )
            if is_downsample:
                producers[i] = [last_block_boundary]
                # downsample is on the skip branch — does not advance chain_pred
            else:
                producers[i] = [chain_pred]
                chain_pred = i
        elif op == "add":
            main_idx, skip_idx, _has_ds = add_info[i]
            inputs = []
            if main_idx is not None and main_idx >= 0:
                inputs.append(main_idx)
            if skip_idx is not None and skip_idx >= 0:
                inputs.append(skip_idx)
            producers[i] = inputs
            chain_pred = i
        else:
            producers[i] = [chain_pred]
            chain_pred = i
    return producers


def build_consumers(producers):
    consumers = {}
    for c_idx, prods in producers.items():
        for p_idx in prods:
            if p_idx is None:
                continue
            consumers.setdefault(p_idx, []).append(c_idx)
    return consumers


def build_dispatches(layers, heavy_set, add_info, weight_map, bias_map=None):
    """Walk layers in topological order with explicit bank-liveness tracking.

    Bank allocation strategy (longest-lifetime-first, applied lazily):
      - At each layer, free any bank whose stored activation has no future
        consumer (consumer_idx > current_idx).
      - Allocate the layer's output bank from whatever is free (excluding
        the input bank to enforce no-same-bank-read-write).
      - Initial network input (producer index -1) is placed in bank 0.

    For each heavy (engine-dispatched) layer the resulting (input_bank,
    output_bank) is captured along with the mask of *other* banks that are
    still holding live activations during that dispatch — those are the
    "skip" (or block-input-retained) banks the engine must not touch.
    """
    weight_layer_lookup = {}
    if weight_map is not None:
        # Path D banked schema uses base_mac_cycle; legacy schema used base_word.
        for w in weight_map.get("layers", []):
            base = w.get("base_mac_cycle", w.get("base_word"))
            if base is None:
                raise RuntimeError(
                    "weight_memory_map layer %s has neither base_mac_cycle nor base_word"
                    % w.get("module_id")
                )
            weight_layer_lookup[w["module_id"]] = int(base)

    bias_layer_lookup = {}
    if bias_map is not None:
        for b in bias_map.get("layers", []):
            bias_layer_lookup[b["module_id"]] = b["base_word"]

    producers = build_producer_consumer_graph(layers, add_info)
    consumers = build_consumers(producers)

    activation_bank = {-1: 0}                  # producer_idx -> bank id
    bank_holds = {b: None for b in range(NUM_BANKS)}
    bank_holds[0] = -1                         # network input
    bank_max_bytes = [0] * NUM_BANKS
    bank_owners = [[] for _ in range(NUM_BANKS)]
    bank_owners[0].append("network_input")

    dispatches = []
    for i, L in enumerate(layers):
        op = L["op_type"]
        prod_indices = producers.get(i, [])
        input_banks = []
        for p in prod_indices:
            if p is None:
                continue
            b = activation_bank.get(p)
            if b is None or bank_holds[b] != p:
                raise RuntimeError(
                    "layer %d (%s) expects producer %d in a bank, but it's not live"
                    % (i, L["module_id"], p)
                )
            input_banks.append(b)

        # Free banks whose tenant has no remaining consumer after this layer.
        for b in range(NUM_BANKS):
            p = bank_holds[b]
            if p is None:
                continue
            remaining = [c for c in consumers.get(p, []) if c > i]
            if not remaining:
                bank_holds[b] = None

        # Allocate output bank: not equal to any input bank, prefer lower id.
        forbidden = set(input_banks)
        out_bank = None
        for b in range(NUM_BANKS):
            if b in forbidden:
                continue
            if bank_holds[b] is None:
                out_bank = b
                break
        if out_bank is None:
            raise RuntimeError(
                "bank pool exhausted at layer %d (%s); increase NUM_BANKS"
                % (i, L["module_id"])
            )
        bank_holds[out_bank] = i
        activation_bank[i] = out_bank
        bank_owners[out_bank].append("%s_out" % L["module_id"])
        bank_max_bytes[out_bank] = max(bank_max_bytes[out_bank], output_bytes(L))

        if L["module_id"] in heavy_set and op == "conv2d":
            input_bank = input_banks[0] if input_banks else 0
            output_bank = out_bank
            # Live but neither input nor output: the "reserved" banks for
            # skip / block-input activations during this dispatch.
            skip_bank_mask = 0
            reserved_owners = []
            for b in range(NUM_BANKS):
                p = bank_holds[b]
                if p is None or b == input_bank or b == output_bank:
                    continue
                skip_bank_mask |= (1 << b)
                reserved_owners.append({
                    "bank": b,
                    "producer_module_id": (
                        layers[p]["module_id"] if p >= 0 else "network_input"
                    ),
                    "producer_layer_index": p,
                })

            in_shape = L["input_shape"]
            out_shape = L["output_shape"]
            weight_shape = L["weight_shape"]
            sh, sw = L.get("stride", [1, 1])
            ph, pw = L.get("padding", [0, 0])
            scale_mult, scale_shift = scale_factor_to_mult_shift(L["scale_factor"])
            wbase = weight_layer_lookup.get(L["module_id"])
            weight_base_known = wbase is not None
            if wbase is None:
                wbase = 0xDEADBE
            bbase = bias_layer_lookup.get(L["module_id"])
            bias_base_known = bbase is not None
            if bbase is None:
                bbase = 0

            # Identify which add(s) this dispatch is feeding the skip path of
            # (for documentation in the sidecar).
            feeds_skip_of = []
            for add_idx, (m_idx, s_idx, has_ds) in add_info.items():
                if s_idx == i:
                    feeds_skip_of.append(layers[add_idx]["module_id"])

            dispatches.append({
                "dispatch_index": len(dispatches),
                "module_id": L["module_id"],
                "layer_index": i,
                "input_bank": input_bank,
                "output_bank": output_bank,
                "skip_bank_mask": skip_bank_mask,
                "reserved_banks": reserved_owners,
                "feeds_skip_of": feeds_skip_of,
                "weight_base_word": int(wbase),
                "weight_base_known": weight_base_known,
                "bias_base_word": int(bbase),
                "bias_base_known": bias_base_known,
                "channel_in": int(in_shape[1]),
                "channel_out": int(out_shape[1]),
                "kernel": [int(weight_shape[2]), int(weight_shape[3])],
                "stride": [int(sh), int(sw)],
                "padding": [int(ph), int(pw)],
                "input_hw": [int(in_shape[2]), int(in_shape[3])],
                "output_hw": [int(out_shape[2]), int(out_shape[3])],
                "scale_mult": int(scale_mult),
                "scale_shift": int(scale_shift),
                "zero_point": int(L["zero_point"]) & 0xFF,
            })

    banks_summary = [
        {
            "bank_id": b,
            "max_bytes_used": int(bank_max_bytes[b]),
            "module_owners": bank_owners[b],
        }
        for b in range(NUM_BANKS)
    ]
    return dispatches, banks_summary


# ---------------------------------------------------------------------------
# Verilog emission
# ---------------------------------------------------------------------------
def emit_verilog(dispatches, num_banks=NUM_BANKS):
    N = len(dispatches)
    if N == 0:
        return _emit_verilog_no_dispatches(num_banks)

    DISPATCH_BITS = max(1, (N - 1).bit_length())
    STEP_BITS = max(1, (NUM_WRITE_STEPS - 1).bit_length())
    BANK_BITS = max(1, (num_banks - 1).bit_length())

    def fields(name):
        return [d[name] for d in dispatches]

    def kfields(key, idx):
        return [d[key][idx] for d in dispatches]

    rom_specs = [
        ("channel_in",       fields("channel_in"),   16),
        ("channel_out",      fields("channel_out"),  16),
        ("kernel_h",         kfields("kernel", 0),    4),
        ("kernel_w",         kfields("kernel", 1),    4),
        ("stride_h",         kfields("stride", 0),    3),
        ("stride_w",         kfields("stride", 1),    3),
        ("padding_h",        kfields("padding", 0),   3),
        ("padding_w",        kfields("padding", 1),   3),
        ("input_h",          kfields("input_hw", 0),  9),
        ("input_w",          kfields("input_hw", 1),  9),
        ("output_h",         kfields("output_hw", 0), 9),
        ("output_w",         kfields("output_hw", 1), 9),
        ("weight_base_word", fields("weight_base_word"), 20),
        ("bias_base_word",   fields("bias_base_word"),  16),
        ("scale_mult",       fields("scale_mult"),      32),
        ("scale_shift",      fields("scale_shift"),      6),
        ("zero_point",       fields("zero_point"),       8),
        ("input_bank",       fields("input_bank"),  BANK_BITS),
        ("output_bank",      fields("output_bank"), BANK_BITS),
        ("skip_mask",        fields("skip_bank_mask"), num_banks),
        # Per-dispatch BRAM-word base addresses for engine activation
        # reads/writes. Derived deterministically from the bank index by
        # multiplying by BANK_DEPTH_WORDS (each bank gets a contiguous
        # non-overlapping region). 16-bit wide to match cfg_act_in_bram_base
        # / cfg_act_out_bram_base in the engine's config_register_block.
        ("act_in_base_word",  [d["input_bank"]  * BANK_DEPTH_WORDS for d in dispatches], 16),
        ("act_out_base_word", [d["output_bank"] * BANK_DEPTH_WORDS for d in dispatches], 16),
    ]

    out = []
    out.append("// Auto-generated by scripts/build_scheduler.py - DO NOT EDIT.")
    out.append("// Deterministic scheduler FSM for nn2rtl shared-engine dispatch.")
    out.append("// Number of engine dispatches: %d" % N)
    out.append("// Register map source-of-truth: docs/agent_tasks/10_engine_config_register_block.md")
    out.append("")
    out.append("module nn2rtl_scheduler (")
    out.append("    input  wire        clk,")
    out.append("    input  wire        rst_n,")
    out.append("    input  wire        start,")
    out.append("    output reg         done,")
    out.append("    // AXI4-Lite master to engine config_register_block")
    out.append("    output reg         s_axil_awvalid,")
    out.append("    input  wire        s_axil_awready,")
    out.append("    output reg  [7:0]  s_axil_awaddr,")
    out.append("    output reg         s_axil_wvalid,")
    out.append("    input  wire        s_axil_wready,")
    out.append("    output reg  [31:0] s_axil_wdata,")
    out.append("    output reg  [3:0]  s_axil_wstrb,")
    out.append("    input  wire        s_axil_bvalid,")
    out.append("    output reg         s_axil_bready,")
    out.append("    input  wire [1:0]  s_axil_bresp,")
    out.append("    // Engine handshake")
    out.append("    output reg         engine_start,")
    out.append("    input  wire        engine_busy,")
    out.append("    input  wire        engine_done,")
    out.append("    // Per-dispatch input-loader handshake (Fix 11): the wrapper")
    out.append("    // muxes the active dispatch's loaded signal here. The")
    out.append("    // scheduler waits on it in S_WAIT_LOAD before pulsing")
    out.append("    // CONTROL.start, so the engine never reads from a")
    out.append("    // half-filled activation BRAM.")
    out.append("    input  wire        current_loaded,")
    out.append("    // Per-dispatch output-FIFO drain handshake (Fix 14): the")
    out.append("    // engine writes its outputs into a shared FIFO; each")
    out.append("    // dispatch's bridge drains the expected number of beats.")
    out.append("    // The scheduler waits in S_WAIT_DRAIN until this asserts,")
    out.append("    // so dispatch N+1 cannot start while dispatch N's beats")
    out.append("    // still sit in the FIFO (which would interleave them).")
    out.append("    input  wire        current_drain_complete,")
    out.append("    // Dispatch index exposed so the wrapper's loaded-mux can")
    out.append("    // select the correct per-dispatch loaded signal.")
    out.append("    output wire [%d:0] dispatch_idx_out," % (DISPATCH_BITS - 1))
    out.append("    // Activation BRAM bank selection")
    out.append("    output reg  [%d:0] input_bank_sel," % (BANK_BITS - 1))
    out.append("    output reg  [%d:0] output_bank_sel," % (BANK_BITS - 1))
    out.append("    output reg  [%d:0] skip_bank_reserved_mask," % (num_banks - 1))
    out.append("    // Dataflow chain handshake")
    out.append("    output reg         spatial_stall,")
    out.append("    output reg         engine_output_ready")
    out.append(");")
    out.append("")
    out.append("    // ------------------------------------------------------------")
    out.append("    // FSM state encoding")
    out.append("    // ------------------------------------------------------------")
    # NOTE (task 13a fix 1): single S_WRITE state asserting both awvalid and
    # wvalid in the same cycle. The engine's config_register_block (task 10)
    # requires `s_axil_awvalid & s_axil_wvalid & awready & wready` all true
    # in one cycle to fire the write handshake. The previous staggered
    # S_WRITE_ADDR -> S_WRITE_DATA pattern would deadlock against that slave.
    # `aw_acked` / `w_acked` regs latch which channel has already been
    # accepted, so a slave that returns awready and wready on different
    # cycles is also supported.
    out.append("    localparam [3:0] S_IDLE        = 4'd0;")
    out.append("    localparam [3:0] S_WRITE       = 4'd1;")
    out.append("    localparam [3:0] S_WRITE_RESP  = 4'd3;")
    out.append("    localparam [3:0] S_NEXT_STEP   = 4'd4;")
    out.append("    localparam [3:0] S_WAIT_LOAD   = 4'd9;   // Fix 11: wait for input loader before starting engine")
    out.append("    localparam [3:0] S_PULSE_START = 4'd5;")
    out.append("    localparam [3:0] S_WAIT_DONE   = 4'd6;")
    out.append("    localparam [3:0] S_WAIT_DRAIN  = 4'd10;  // Fix 14: wait for engine_output_fifo to drain")
    out.append("    localparam [3:0] S_NEXT_DISP   = 4'd7;")
    out.append("    localparam [3:0] S_DONE        = 4'd8;")
    out.append("")
    out.append("    reg aw_acked_r, w_acked_r;")
    out.append("")
    out.append("    reg [3:0]           state, next_state;")
    out.append("    reg [%d:0] dispatch_idx;" % (DISPATCH_BITS - 1))
    out.append("    assign dispatch_idx_out = dispatch_idx;")
    out.append("    reg [%d:0] write_step;" % (STEP_BITS - 1))
    out.append("    localparam [%d:0] LAST_DISPATCH = %d'd%d;"
               % (DISPATCH_BITS - 1, DISPATCH_BITS, N - 1))
    out.append("    localparam [%d:0] LAST_STEP     = %d'd%d;"
               % (STEP_BITS - 1, STEP_BITS, NUM_WRITE_STEPS - 1))
    out.append("")

    # ROM blocks
    for name, arr, width in rom_specs:
        out.append("    reg [%d:0] %s_rom;" % (width - 1, name))
        out.append("    always @(*) begin")
        out.append("        case (dispatch_idx)")
        for d_idx, v in enumerate(arr):
            mask = (1 << width) - 1
            out.append("            %d'd%d: %s_rom = %d'd%d;"
                       % (DISPATCH_BITS, d_idx, name, width, v & mask))
        out.append("            default: %s_rom = %d'd0;" % (name, width))
        out.append("        endcase")
        out.append("    end")
        out.append("")

    # Per-step AXI write address/data mux
    out.append("    // ------------------------------------------------------------")
    out.append("    // Per-step AXI write address + data")
    out.append("    // ------------------------------------------------------------")
    out.append("    reg [7:0]  step_addr;")
    out.append("    reg [31:0] step_data;")
    out.append("    always @(*) begin")
    out.append("        step_addr = 8'h00;")
    out.append("        step_data = 32'd0;")
    out.append("        case (write_step)")
    for step_idx, (name, off) in enumerate(WRITE_SEQUENCE):
        if name == "INPUT_CHANNELS":
            data = "{16'd0, channel_in_rom}"
        elif name == "OUTPUT_CHANNELS":
            data = "{16'd0, channel_out_rom}"
        elif name == "KERNEL_H_W":
            data = "{24'd0, kernel_h_rom, kernel_w_rom}"
        elif name == "STRIDE_H_W":
            data = "{26'd0, stride_h_rom, stride_w_rom}"
        elif name == "PADDING_H_W":
            data = "{26'd0, padding_h_rom, padding_w_rom}"
        elif name == "INPUT_H_W":
            data = "{7'd0, input_h_rom, 7'd0, input_w_rom}"
        elif name == "OUTPUT_H_W":
            data = "{7'd0, output_h_rom, 7'd0, output_w_rom}"
        elif name == "WEIGHT_BASE_WORD":
            data = "{12'd0, weight_base_word_rom}"
        elif name == "BIAS_BASE_WORD":
            data = "{16'd0, bias_base_word_rom}"
        elif name == "SCALE_MULT":
            data = "scale_mult_rom"
        elif name == "SCALE_SHIFT_AND_ZP":
            data = "{18'd0, zero_point_rom, scale_shift_rom}"
        elif name == "ACT_IN_BASE":
            data = "{16'd0, act_in_base_word_rom}"
        elif name == "ACT_OUT_BASE":
            data = "{16'd0, act_out_base_word_rom}"
        # Fix 12: CONTROL_START is no longer in WRITE_SEQUENCE (engine_start
        # pin is the sole engine trigger). The branch below is dead but kept
        # so a future hand-edit that re-adds the step has a known mapping.
        elif name == "CONTROL_START":
            data = "32'h0000_0001"
        else:
            data = "32'd0"
        out.append("            %d'd%d: begin step_addr = 8'h%02X; step_data = %s; end"
                   % (STEP_BITS, step_idx, off, data))
    out.append("            default: begin step_addr = 8'h00; step_data = 32'd0; end")
    out.append("        endcase")
    out.append("    end")
    out.append("")

    # Next-state logic
    out.append("    // ------------------------------------------------------------")
    out.append("    // FSM next-state logic")
    out.append("    // ------------------------------------------------------------")
    out.append("    always @(*) begin")
    out.append("        next_state = state;")
    out.append("        case (state)")
    out.append("            S_IDLE:        if (start) next_state = S_WRITE;")
    out.append("            // S_WRITE asserts AW and W in the same cycle. A correct AXI4-Lite")
    out.append("            // slave fires the handshake when (awvalid & awready & wvalid & wready)")
    out.append("            // are all true in one cycle. aw_acked_r / w_acked_r let the slave")
    out.append("            // return the two ready signals in different cycles too.")
    out.append("            S_WRITE:")
    out.append("                if ((aw_acked_r || (s_axil_awvalid && s_axil_awready)) &&")
    out.append("                    (w_acked_r  || (s_axil_wvalid  && s_axil_wready)))")
    out.append("                    next_state = S_WRITE_RESP;")
    out.append("            S_WRITE_RESP:  if (s_axil_bvalid)  next_state = S_NEXT_STEP;")
    out.append("            // Fix 11: after the final config write, wait until the input")
    out.append("            // loader bridge for the current dispatch has finished filling")
    out.append("            // the activation BRAM. current_loaded comes from a wrapper-")
    out.append("            // side mux over per-dispatch `loaded` signals.")
    out.append("            S_NEXT_STEP:   next_state = (write_step == LAST_STEP) ? S_WAIT_LOAD : S_WRITE;")
    out.append("            S_WAIT_LOAD:   if (current_loaded) next_state = S_PULSE_START;")
    out.append("            S_PULSE_START: next_state = S_WAIT_DONE;")
    out.append("            // Fix 14: between dispatches, wait in S_WAIT_DRAIN until the")
    out.append("            // engine_output_fifo has been fully drained by this dispatch's")
    out.append("            // bridge. Otherwise dispatch N+1's writes would interleave with")
    out.append("            // N's outputs still in the FIFO. spatial_stall=0 in this state")
    out.append("            // so the spatial chain runs and drains the bridge → FIFO →")
    out.append("            // downstream relu/add → next input_loader path.")
    # 13a audit fix: previously `if (engine_done && !engine_busy)` — these
    # two flags are mutually exclusive (engine_done is high only in
    # ST_DONE, where engine_busy is also high). Latching engine_done's
    # rising edge would deadlock the scheduler permanently in S_WAIT_DONE.
    # engine_done is itself only high in the engine's ST_DONE state, so
    # sampling it alone is sufficient.
    out.append("            S_WAIT_DONE:   if (engine_done) next_state = S_WAIT_DRAIN;")
    out.append("            S_WAIT_DRAIN:  if (current_drain_complete) next_state = S_NEXT_DISP;")
    out.append("            S_NEXT_DISP:   next_state = (dispatch_idx == LAST_DISPATCH) ? S_DONE : S_WRITE;")
    out.append("            S_DONE:        next_state = S_DONE;")
    out.append("            default:       next_state = S_IDLE;")
    out.append("        endcase")
    out.append("    end")
    out.append("")

    # Sequential state
    out.append("    // ------------------------------------------------------------")
    out.append("    // Sequential state + counters")
    out.append("    // ------------------------------------------------------------")
    out.append("    always @(posedge clk or negedge rst_n) begin")
    out.append("        if (!rst_n) begin")
    out.append("            state        <= S_IDLE;")
    out.append("            dispatch_idx <= {%d{1'b0}};" % DISPATCH_BITS)
    out.append("            write_step   <= {%d{1'b0}};" % STEP_BITS)
    out.append("            aw_acked_r   <= 1'b0;")
    out.append("            w_acked_r    <= 1'b0;")
    out.append("        end else begin")
    out.append("            state <= next_state;")
    out.append("            if (state == S_IDLE && start) begin")
    out.append("                dispatch_idx <= {%d{1'b0}};" % DISPATCH_BITS)
    out.append("                write_step   <= {%d{1'b0}};" % STEP_BITS)
    out.append("            end")
    out.append("            if (state == S_NEXT_STEP) begin")
    out.append("                if (write_step == LAST_STEP) begin")
    out.append("                    write_step <= {%d{1'b0}};" % STEP_BITS)
    out.append("                end else begin")
    out.append("                    write_step <= write_step + 1'b1;")
    out.append("                end")
    out.append("            end")
    out.append("            if (state == S_NEXT_DISP && dispatch_idx != LAST_DISPATCH) begin")
    out.append("                dispatch_idx <= dispatch_idx + 1'b1;")
    out.append("            end")
    out.append("            // Latch per-channel handshake acks while in S_WRITE; clear")
    out.append("            // when leaving S_WRITE so the next register write starts fresh.")
    out.append("            if (state == S_WRITE) begin")
    out.append("                if (s_axil_awvalid && s_axil_awready) aw_acked_r <= 1'b1;")
    out.append("                if (s_axil_wvalid  && s_axil_wready)  w_acked_r  <= 1'b1;")
    out.append("            end else begin")
    out.append("                aw_acked_r <= 1'b0;")
    out.append("                w_acked_r  <= 1'b0;")
    out.append("            end")
    out.append("        end")
    out.append("    end")
    out.append("")

    # Output mux
    out.append("    // ------------------------------------------------------------")
    out.append("    // Output assignments")
    out.append("    // ------------------------------------------------------------")
    out.append("    always @(*) begin")
    out.append("        s_axil_awvalid          = 1'b0;")
    out.append("        s_axil_awaddr           = 8'd0;")
    out.append("        s_axil_wvalid           = 1'b0;")
    out.append("        s_axil_wdata            = 32'd0;")
    out.append("        s_axil_wstrb            = 4'b1111;")
    out.append("        s_axil_bready           = 1'b0;")
    out.append("        engine_start            = 1'b0;")
    out.append("        done                    = 1'b0;")
    out.append("        spatial_stall           = 1'b0;")
    out.append("        engine_output_ready     = 1'b0;")
    out.append("        input_bank_sel          = input_bank_rom;")
    out.append("        output_bank_sel         = output_bank_rom;")
    out.append("        skip_bank_reserved_mask = skip_mask_rom;")
    out.append("")
    out.append("        case (state)")
    out.append("            S_IDLE: begin")
    out.append("                spatial_stall = 1'b0;")
    out.append("            end")
    out.append("            // Single S_WRITE: drive AW and W simultaneously. Hold each")
    out.append("            // valid line high until its respective handshake fires, then")
    out.append("            // (per the FSM next-state logic) wait for the other before")
    out.append("            // transitioning to S_WRITE_RESP.")
    out.append("            S_WRITE: begin")
    out.append("                s_axil_awvalid = ~aw_acked_r;")
    out.append("                s_axil_awaddr  = step_addr;")
    out.append("                s_axil_wvalid  = ~w_acked_r;")
    out.append("                s_axil_wdata   = step_data;")
    out.append("                s_axil_wstrb   = 4'b1111;")
    out.append("                spatial_stall  = 1'b1;")
    out.append("            end")
    out.append("            S_WRITE_RESP: begin")
    out.append("                s_axil_bready = 1'b1;")
    out.append("                spatial_stall = 1'b1;")
    out.append("            end")
    out.append("            S_NEXT_STEP: begin")
    out.append("                spatial_stall = 1'b1;")
    out.append("            end")
    out.append("            S_WAIT_LOAD: begin")
    out.append("                spatial_stall = 1'b0; // let the spatial chain keep filling the loader BRAM")
    out.append("            end")
    out.append("            S_PULSE_START: begin")
    out.append("                // Fix 11: CONTROL.start AXI write is removed; this pin is now")
    out.append("                // the SOLE engine-start trigger. We reach here only after")
    out.append("                // S_WAIT_LOAD has observed current_loaded, so the activation")
    out.append("                // BRAM is guaranteed populated for this dispatch.")
    out.append("                engine_start  = 1'b1;")
    out.append("                spatial_stall = 1'b1;")
    out.append("            end")
    out.append("            S_WAIT_DONE: begin")
    out.append("                spatial_stall = 1'b1;")
    out.append("            end")
    out.append("            S_WAIT_DRAIN: begin")
    out.append("                // Fix 14: chain MUST run so the bridge can drain the")
    out.append("                // engine_output_fifo and feed downstream relu/add layers.")
    out.append("                spatial_stall = 1'b0;")
    out.append("            end")
    out.append("            S_NEXT_DISP: begin")
    out.append("                spatial_stall       = 1'b1;")
    out.append("                engine_output_ready = 1'b1;")
    out.append("            end")
    out.append("            S_DONE: begin")
    # 13a audit fix: previously also asserted `engine_output_ready = 1'b1`.
    # S_DONE is sticky (next_state = S_DONE), so the level would re-pulse
    # every cycle, driving each `engine_output_bridge`'s `dispatch_count`
    # past 15 and wrapping back through SLOT matches → bogus re-activation
    # of stale dispatch outputs. The legitimate `engine_output_ready`
    # pulse for dispatch N already fires in S_NEXT_DISP, one cycle before
    # S_DONE on the final dispatch. So nothing is lost by dropping it
    # here, and the bridge counters land at exactly num_dispatches.
    out.append("                done                = 1'b1;")
    out.append("            end")
    out.append("            default: begin")
    out.append("                spatial_stall = 1'b0;")
    out.append("            end")
    out.append("        endcase")
    out.append("    end")
    out.append("")
    # 13a audit fix: scheduler's s_axil_bresp input is not sampled by the
    # FSM (slave only ever returns OKAY). Drain it into an _unused wire so
    # iverilog -Wall / Vivado's UNUSED-input lint stays quiet.
    out.append("    wire _unused_s_axil_bresp = |s_axil_bresp;")
    out.append("")
    out.append("endmodule")
    out.append("")
    return "\n".join(out)


def _emit_verilog_no_dispatches(num_banks):
    BANK_BITS = max(1, (num_banks - 1).bit_length())
    lines = [
        "// Auto-generated by scripts/build_scheduler.py - DO NOT EDIT.",
        "// Degenerate scheduler (zero engine dispatches).",
        "",
        "module nn2rtl_scheduler (",
        "    input  wire        clk,",
        "    input  wire        rst_n,",
        "    input  wire        start,",
        "    output reg         done,",
        "    output reg         s_axil_awvalid,",
        "    input  wire        s_axil_awready,",
        "    output reg  [7:0]  s_axil_awaddr,",
        "    output reg         s_axil_wvalid,",
        "    input  wire        s_axil_wready,",
        "    output reg  [31:0] s_axil_wdata,",
        "    output reg  [3:0]  s_axil_wstrb,",
        "    input  wire        s_axil_bvalid,",
        "    output reg         s_axil_bready,",
        "    input  wire [1:0]  s_axil_bresp,",
        "    output reg         engine_start,",
        "    input  wire        engine_busy,",
        "    input  wire        engine_done,",
        "    output reg  [%d:0] input_bank_sel," % (BANK_BITS - 1),
        "    output reg  [%d:0] output_bank_sel," % (BANK_BITS - 1),
        "    output reg  [%d:0] skip_bank_reserved_mask," % (num_banks - 1),
        "    output reg         spatial_stall,",
        "    output reg         engine_output_ready",
        ");",
        "    always @(*) begin",
        "        s_axil_awvalid          = 1'b0;",
        "        s_axil_awaddr           = 8'd0;",
        "        s_axil_wvalid           = 1'b0;",
        "        s_axil_wdata            = 32'd0;",
        "        s_axil_wstrb            = 4'b1111;",
        "        s_axil_bready           = 1'b0;",
        "        engine_start            = 1'b0;",
        "        done                    = start;",
        "        spatial_stall           = 1'b0;",
        "        engine_output_ready     = 1'b1;",
        "        input_bank_sel          = {%d{1'b0}};" % BANK_BITS,
        "        output_bank_sel         = {%d{1'b0}};" % BANK_BITS,
        "        skip_bank_reserved_mask = {%d{1'b0}};" % num_banks,
        "    end",
        "    // Tie-off use of inputs to suppress unused-signal warnings.",
        "    wire _unused = clk | rst_n | s_axil_awready | s_axil_wready",
        "                 | s_axil_bvalid | engine_busy | engine_done",
        "                 | (|s_axil_bresp);",
        "endmodule",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    repo_root = detect_repo_root(Path(__file__))

    def resolve(p):
        path = Path(p)
        return path if path.is_absolute() else (repo_root / path)

    layer_ir_path = resolve(args.layer_ir)
    heavy_path = resolve(args.engine_modules)
    weight_map_path = resolve(args.weight_map)
    bias_map_path = resolve(args.bias_map)
    out_verilog_path = resolve(args.out_verilog)
    out_schedule_path = resolve(args.out_schedule)

    with layer_ir_path.open() as f:
        ir = json.load(f)
    layers = ir["layers"]

    heavy_modules, used_fallback = load_heavy_modules(heavy_path)
    weight_map = load_weight_map(weight_map_path)
    weight_map_present = weight_map is not None
    bias_map = load_bias_map(bias_map_path)
    bias_map_present = bias_map is not None

    add_info = annotate_residual_blocks(layers)
    heavy_set = set(heavy_modules)
    dispatches, banks_summary = build_dispatches(
        layers, heavy_set, add_info, weight_map, bias_map
    )

    # Sanity gates: every heavy module in LayerIR has exactly one dispatch.
    heavy_in_ir = [L["module_id"] for L in layers if L["module_id"] in heavy_set]
    dispatched_ids = [d["module_id"] for d in dispatches]
    missing = [m for m in heavy_in_ir if m not in dispatched_ids]
    if missing:
        sys.stderr.write(
            "ERROR: heavy modules in LayerIR without a dispatch: %s\n" % missing
        )
        sys.exit(2)
    dupes = {m for m in dispatched_ids if dispatched_ids.count(m) > 1}
    if dupes:
        sys.stderr.write("ERROR: duplicate dispatches: %s\n" % dupes)
        sys.exit(2)
    # No dispatch should read and write the same bank.
    for d in dispatches:
        if d["input_bank"] == d["output_bank"]:
            sys.stderr.write(
                "ERROR: dispatch %d reads and writes bank %d\n"
                % (d["dispatch_index"], d["input_bank"])
            )
            sys.exit(2)

    verilog_text = emit_verilog(dispatches)
    out_verilog_path.parent.mkdir(parents=True, exist_ok=True)
    out_verilog_path.write_text(verilog_text, encoding="utf-8", newline="\n")

    schedule = {
        "model_name": ir.get("model_name", "unknown"),
        "generator": "scripts/build_scheduler.py",
        "wave1_fallback_heavy_list": used_fallback,
        "weight_memory_map_present": weight_map_present,
        "weight_base_word_placeholder": None if weight_map_present else "0xDEADBE",
        "bias_memory_map_present": bias_map_present,
        "bias_base_word_placeholder": None if bias_map_present else 0,
        "num_dispatches": len(dispatches),
        "num_banks": len(banks_summary),
        "num_write_steps_per_dispatch": NUM_WRITE_STEPS,
        "register_map_source": "docs/agent_tasks/10_engine_config_register_block.md",
        "axi4_lite_byte_offsets": [
            {"step": i, "name": name, "offset": "0x%02X" % off}
            for i, (name, off) in enumerate(WRITE_SEQUENCE)
        ],
        "engine_start_ordering": (
            "engine_start pulses for exactly one cycle in S_PULSE_START, "
            "which is reached only AFTER the bvalid handshake of all %d "
            "config-register writes has completed AND `current_loaded` "
            "is high in S_WAIT_LOAD. CONTROL.start at offset 0x%02X is "
            "NO LONGER written (13a Fix 12); the engine_start pin is the "
            "sole engine trigger."
        ) % (NUM_WRITE_STEPS, REG_CONTROL),
        "residual_blocks": [
            {
                "add_module_id": layers[add_idx]["module_id"],
                "main_source_module_id": (
                    layers[main_idx]["module_id"] if main_idx is not None and main_idx >= 0 else None
                ),
                "skip_source_module_id": (
                    layers[skip_idx]["module_id"] if skip_idx is not None and skip_idx >= 0 else None
                ),
                "has_downsample": has_ds,
            }
            for add_idx, (main_idx, skip_idx, has_ds) in sorted(add_info.items())
        ],
        "banks": banks_summary,
        "dispatches": dispatches,
    }

    out_schedule_path.parent.mkdir(parents=True, exist_ok=True)
    out_schedule_path.write_text(
        json.dumps(schedule, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("scheduler: %d dispatches; %d banks allocated"
          % (len(dispatches), len(banks_summary)))
    print("  fallback heavy list:        %s" % used_fallback)
    print("  weight_memory_map present:  %s" % weight_map_present)
    print("  bias_memory_map present:    %s" % bias_map_present)
    print("  verilog:  %s" % out_verilog_path)
    print("  schedule: %s" % out_schedule_path)


if __name__ == "__main__":
    main()
