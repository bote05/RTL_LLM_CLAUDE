"""ONNX-based pipeline IR frontend for nn2rtl.

Universal replacement for the hard-coded ResNet-50 quantize_impl.py + golden_impl.py FX path.
Works with any ONNX model that contains Conv2d, ReLU, Add, and MaxPool2d operators.

Flow
----
1.  Load the ONNX model and run onnxsim to fold BatchNorm into Conv and remove
    dead nodes.
2.  Run ONNX shape inference so every intermediate tensor has a known shape.
3.  Walk the simplified graph in topological order and build an OnnxLayerSpec
    for each supported op (Conv, Relu, Add, MaxPool).
4.  Calibrate: run the float ONNX model with synthetic INT8-range inputs using
    ONNX Runtime, collecting per-tensor max-abs activation statistics.
5.  Quantize: derive per-tensor symmetric INT8 scale factors from the stats.
    Quantize Conv weights to INT8 and biases to INT32 (accumulator domain).
6.  Simulate: run the quantised network forward with synthetic INT8 inputs,
    capturing the INT8 activations at every supported layer.
7.  Write hex artefacts (weights, biases) and binary golden vector files
    (.goldin / .goldout) using the golden_impl.py helper functions.
8.  Return a complete PipelineIR-compatible dict that generate_golden.py can
    write to output/layer_ir.json.
"""

from __future__ import annotations

import copy
import re
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import onnx.shape_inference
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx.symbolic_helper as torch_onnx_symbolic_helper

try:
    import onnxsim
    _ONNXSIM_AVAILABLE = True
except ImportError:
    _ONNXSIM_AVAILABLE = False

PREFERRED_ONNX_OPSET = 17


def resolve_supported_onnx_opset(requested: int = PREFERRED_ONNX_OPSET) -> int:
    """Return the highest ONNX opset the local PyTorch can actually export.

    We prefer opset 17 because it is stable and covers the operator set used by
    this frontend (Conv, ReLU, Add, MaxPool, BatchNormalization). Some local
    developer environments still carry older PyTorch builds, though, so we
    clamp to the highest stable opset that build supports instead of failing
    during export.
    """
    stable_opsets = list(getattr(torch_onnx_symbolic_helper, "_onnx_stable_opsets", []))
    if not stable_opsets:
        fallback = getattr(torch_onnx_symbolic_helper, "_default_onnx_opset_version", requested)
        return int(fallback)
    supported_max = max(int(opset) for opset in stable_opsets)
    return int(min(requested, supported_max))


DEFAULT_ONNX_OPSET = resolve_supported_onnx_opset()

from scripts.golden_impl import (
    CONV_PIPELINE_STAGES,
    SIGNAL_LITERALS,
    GoldenGenerationError,
    conv_mac_parallelism,
    Int8Add,
    Int8Conv2d,
    Int8ReLU,
    build_deterministic_input_stream,
    channel_bus_bits_from_shape,
    compute_add_latency_cycles,
    compute_conv2d_latency_cycles,
    get_golden_artifact_paths,
    get_weight_artifact_paths,
    pack_paired_tensors_to_bus_words,
    pack_tensor_vectors_to_bus_words,
    quantize_tensor_to_int8_range,
    tensor_to_int32_list,
    utc_now_iso8601,
    validate_pipeline_ir_payload,
    write_golden_vector_file,
    write_signed_int8_hex,
    write_signed_int32_hex,
    write_weight_bank_hex_files,
)


# ---------------------------------------------------------------------------
# INT8 MaxPool2d module
# ---------------------------------------------------------------------------

class Int8MaxPool2d(nn.Module):
    """Max pooling in INT8 domain.

    The max operation is monotone so it commutes with INT8 quantisation —
    no requantisation step is required.  The output is clamped to [-128, 127]
    purely as a defensive guard.
    """

    def __init__(
        self,
        kernel_size: Sequence[int],
        stride: Sequence[int],
        padding: Sequence[int],
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = tuple(int(k) for k in kernel_size)
        self.stride = tuple(int(s) for s in stride)
        self.padding = tuple(int(p) for p in padding)
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.max_pool2d(
            x.to(torch.float32),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
        )
        return torch.clamp(y, -128, 127)


# ---------------------------------------------------------------------------
# Per-layer spec
# ---------------------------------------------------------------------------

@dataclass
class OnnxLayerSpec:
    module_id: str
    op_type: str                      # "conv2d" | "relu" | "add" | "maxpool"
    input_tensor_names: list[str]     # ONNX tensor name(s) consumed
    output_tensor_name: str           # ONNX tensor name produced
    input_shape: list[int]            # [N, C, H, W]
    output_shape: list[int]           # [N, C, H, W]

    # Conv2d -----------------------------------------------------------------
    weight: Optional[np.ndarray] = None   # float32 [OC, IC/G, KH, KW]
    bias: Optional[np.ndarray] = None     # float32 [OC]
    stride: list[int] = field(default_factory=lambda: [1, 1])
    padding: list[int] = field(default_factory=lambda: [0, 0])
    dilation: list[int] = field(default_factory=lambda: [1, 1])
    groups: int = 1

    # MaxPool2d --------------------------------------------------------------
    pool_kernel: list[int] = field(default_factory=lambda: [1, 1])
    pool_stride: list[int] = field(default_factory=lambda: [1, 1])
    pool_padding: list[int] = field(default_factory=lambda: [0, 0])
    pool_ceil_mode: bool = False

    # Add (residual) wiring --------------------------------------------------
    add_lhs_tensor: str = ""
    add_rhs_tensor: str = ""

    # Calibration-derived (filled in by fill_calibration_stats) -------------
    input_scale: float = 1.0
    output_scale: float = 1.0
    weight_scale: float = 1.0
    lhs_scale: float = 1.0
    rhs_scale: float = 1.0


# ---------------------------------------------------------------------------
# ONNX loading / simplification
# ---------------------------------------------------------------------------

def load_onnx(path: Path) -> onnx.ModelProto:
    model = onnx.load(str(path))
    model = onnx.shape_inference.infer_shapes(model)
    return model


def simplify_onnx(model: onnx.ModelProto) -> onnx.ModelProto:
    """Simplify the model with onnxsim (BatchNorm fold, dead node removal)."""
    if not _ONNXSIM_AVAILABLE:
        return model
    try:
        simplified, ok = onnxsim.simplify(model)
        if ok:
            return onnx.shape_inference.infer_shapes(simplified)
    except Exception:
        pass
    return model


# ---------------------------------------------------------------------------
# Graph introspection helpers
# ---------------------------------------------------------------------------

def _tensor_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        t = vi.type.tensor_type
        if t.HasField("shape"):
            dims = [int(d.dim_value) if d.dim_value > 0 else -1 for d in t.shape.dim]
            shapes[vi.name] = dims
    return shapes


def _initializers(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}


def _get_attr(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == onnx.AttributeProto.INT:
                return int(attr.i)
            if attr.type == onnx.AttributeProto.FLOAT:
                return float(attr.f)
            if attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            if attr.type == onnx.AttributeProto.FLOATS:
                return list(attr.floats)
    return default


def _sanitize_id(raw: str, fallback: str) -> str:
    """Convert an ONNX node name to a valid Python/Verilog identifier."""
    if not raw:
        return fallback
    # Remove leading slashes (common in PyTorch exports: "/conv1/Conv")
    name = raw.lstrip("/")
    # Replace path separators and non-identifier chars with underscores
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_").lower()
    if not name or not name[0].isalpha():
        name = f"op_{name}" if name else fallback
    return name


# ---------------------------------------------------------------------------
# Per-op spec extraction
# ---------------------------------------------------------------------------

def _extract_conv(
    node: onnx.NodeProto,
    shapes: dict[str, list[int]],
    inits: dict[str, np.ndarray],
    mid: str,
) -> OnnxLayerSpec:
    inp = node.input[0]
    w_name = node.input[1]
    b_name = node.input[2] if len(node.input) > 2 and node.input[2] else None

    weight = inits[w_name].astype(np.float32)
    bias = inits[b_name].astype(np.float32) if b_name and b_name in inits else np.zeros(weight.shape[0], np.float32)

    strides = _get_attr(node, "strides", [1, 1])
    pads = _get_attr(node, "pads", [0, 0, 0, 0])   # [h_begin, w_begin, h_end, w_end]
    dilations = _get_attr(node, "dilations", [1, 1])
    group = _get_attr(node, "group", 1)

    # ONNX symmetric padding: take begin values only (onnxsim ensures begin==end)
    pad_h, pad_w = (pads[0], pads[1]) if len(pads) >= 2 else (0, 0)

    return OnnxLayerSpec(
        module_id=mid,
        op_type="conv2d",
        input_tensor_names=[inp],
        output_tensor_name=node.output[0],
        input_shape=shapes.get(inp, [-1, -1, -1, -1]),
        output_shape=shapes.get(node.output[0], [-1, -1, -1, -1]),
        weight=weight,
        bias=bias,
        stride=list(strides),
        padding=[int(pad_h), int(pad_w)],
        dilation=list(dilations),
        groups=int(group),
    )


def _extract_relu(
    node: onnx.NodeProto,
    shapes: dict[str, list[int]],
    mid: str,
) -> OnnxLayerSpec:
    inp = node.input[0]
    return OnnxLayerSpec(
        module_id=mid,
        op_type="relu",
        input_tensor_names=[inp],
        output_tensor_name=node.output[0],
        input_shape=shapes.get(inp, [-1, -1, -1, -1]),
        output_shape=shapes.get(node.output[0], [-1, -1, -1, -1]),
    )


def _extract_maxpool(
    node: onnx.NodeProto,
    shapes: dict[str, list[int]],
    mid: str,
) -> OnnxLayerSpec:
    inp = node.input[0]
    kernel_shape = _get_attr(node, "kernel_shape", [1, 1])
    strides = _get_attr(node, "strides", [1, 1])
    pads = _get_attr(node, "pads", [0, 0, 0, 0])
    ceil_mode = bool(_get_attr(node, "ceil_mode", 0))
    pad_h, pad_w = (pads[0], pads[1]) if len(pads) >= 2 else (0, 0)

    return OnnxLayerSpec(
        module_id=mid,
        op_type="maxpool",
        input_tensor_names=[inp],
        output_tensor_name=node.output[0],
        input_shape=shapes.get(inp, [-1, -1, -1, -1]),
        output_shape=shapes.get(node.output[0], [-1, -1, -1, -1]),
        pool_kernel=list(kernel_shape),
        pool_stride=list(strides),
        pool_padding=[int(pad_h), int(pad_w)],
        pool_ceil_mode=ceil_mode,
    )


def _extract_add(
    node: onnx.NodeProto,
    shapes: dict[str, list[int]],
    mid: str,
) -> OnnxLayerSpec:
    lhs, rhs = node.input[0], node.input[1]
    return OnnxLayerSpec(
        module_id=mid,
        op_type="add",
        input_tensor_names=[lhs, rhs],
        output_tensor_name=node.output[0],
        input_shape=shapes.get(lhs, [-1, -1, -1, -1]),
        output_shape=shapes.get(node.output[0], [-1, -1, -1, -1]),
        add_lhs_tensor=lhs,
        add_rhs_tensor=rhs,
    )


# ---------------------------------------------------------------------------
# Full graph spec extraction
# ---------------------------------------------------------------------------

def extract_layer_specs(model: onnx.ModelProto) -> list[OnnxLayerSpec]:
    """Walk the simplified ONNX graph and return one OnnxLayerSpec per supported op."""
    shapes = _tensor_shapes(model)
    inits = _initializers(model)
    op_counts: dict[str, int] = {}
    specs: list[OnnxLayerSpec] = []

    for node in model.graph.node:
        op = node.op_type

        # Treat Clip(min=0) as ReLU (used by MobileNet etc.)
        if op == "Clip":
            min_v = None
            # min/max are inputs in opset 11+, attributes in opset 6
            if len(node.input) >= 2 and node.input[1] in inits:
                min_v = float(inits[node.input[1]].flat[0])
            else:
                min_v = _get_attr(node, "min", None)
            if min_v is not None and float(min_v) == 0.0:
                op = "Relu"
            else:
                continue

        if op == "Conv":
            # Weight must be an initializer; skip constant-folded Convs
            if len(node.input) < 2 or node.input[1] not in inits:
                continue
            if node.input[0] in inits:
                continue
            op_key = "conv2d"
        elif op == "Relu":
            op_key = "relu"
        elif op == "MaxPool":
            op_key = "maxpool"
        elif op == "Add":
            # Skip bias-add patterns: both inputs must come from the network
            if node.input[0] in inits or node.input[1] in inits:
                continue
            op_key = "add"
        else:
            continue

        idx = op_counts.get(op_key, 0)
        op_counts[op_key] = idx + 1
        mid = _sanitize_id(node.name, f"{op_key}_{idx}")
        # Ensure uniqueness among specs so far
        existing_ids = {s.module_id for s in specs}
        base = mid
        counter = 0
        while mid in existing_ids:
            counter += 1
            mid = f"{base}_{counter}"

        if op == "Conv":
            specs.append(_extract_conv(node, shapes, inits, mid))
        elif op == "Relu":
            specs.append(_extract_relu(node, shapes, mid))
        elif op == "MaxPool":
            specs.append(_extract_maxpool(node, shapes, mid))
        elif op == "Add":
            specs.append(_extract_add(node, shapes, mid))

    if not specs:
        raise GoldenGenerationError(
            "No supported ops (Conv, Relu, MaxPool, Add) found in ONNX model after simplification."
        )
    return specs


def validate_graph_outputs_covered(
    specs: list[OnnxLayerSpec],
    graph_output_names: set[str],
    graph_input_names: set[str],
) -> None:
    """Fail fast if any model output isn't produced by an extracted supported spec.

    Without this check, a graph that ends in an unsupported tail (e.g.
    ``Conv -> Flatten -> Gemm`` or ``Conv -> Sigmoid``) silently truncates
    to the supported prefix. validate_graph_completeness() only looks
    backward from each spec's inputs, so it cannot detect that the model's
    real outputs lie *after* the last extracted spec. This function closes
    that gap: every declared graph output must either be produced by a
    spec or be a raw graph input (the degenerate identity case).
    """
    produced = {spec.output_tensor_name for spec in specs} | graph_input_names
    missing = [name for name in graph_output_names if name not in produced]
    if not missing:
        return

    final_supported = specs[-1].output_tensor_name if specs else "(none)"
    raise GoldenGenerationError(
        f"ONNX graph has unsupported ops AFTER the last extracted supported "
        f"layer. Model output tensors {missing} are not produced by any "
        f"Conv/Relu/MaxPool/Add/Clip(min=0) spec — truncating to the "
        f"supported prefix would produce an RTL pipeline that does not "
        f"match the ONNX model. "
        f"Last supported-spec output tensor: '{final_supported}'. "
        f"Either extend the frontend to cover the missing op, or "
        f"pre-process the ONNX graph to strip the unsupported tail "
        f"(e.g. remove the classifier head for a feature-extraction pipeline)."
    )


def validate_graph_completeness(
    specs: list[OnnxLayerSpec],
    graph_input_names: set[str],
) -> None:
    """Fail fast if a spec consumes a tensor no prior spec or graph input produces.

    A gap means an unsupported op sits between two supported ops — running the
    INT8 simulation with a silent fallback would corrupt the goldens.  Raising
    here surfaces the missing op while the error trail is still precise.

    Reports the longest prefix of the spec chain that *is* reachable so the
    user can see exactly where the graph breaks.
    """
    produced: set[str] = set(graph_input_names)
    for spec in specs:
        for inp_name in spec.input_tensor_names:
            if inp_name not in produced:
                reachable_ids = [s.module_id for s in specs if s.output_tensor_name in produced]
                raise GoldenGenerationError(
                    f"ONNX graph has an unsupported op between supported layers. "
                    f"Layer '{spec.module_id}' ({spec.op_type}) consumes tensor "
                    f"'{inp_name}', but no prior supported layer or graph input "
                    f"produces it. "
                    f"Reachable chain so far: {reachable_ids or '(empty)'}. "
                    f"Supported ops: Conv, Relu, MaxPool, Add (+ Clip min=0). "
                    f"Extend the frontend or pre-process the offending subgraph."
                )
        produced.add(spec.output_tensor_name)


# ---------------------------------------------------------------------------
# Calibration via ONNX Runtime
# ---------------------------------------------------------------------------

def _expose_intermediates(model: onnx.ModelProto) -> onnx.ModelProto:
    """Add every intermediate tensor as a graph output for ONNX Runtime capture."""
    m = copy.deepcopy(model)
    existing = {o.name for o in m.graph.output}
    for node in m.graph.node:
        for out in node.output:
            if out and out not in existing:
                m.graph.output.append(
                    onnx.helper.make_tensor_value_info(out, onnx.TensorProto.FLOAT, None)
                )
                existing.add(out)
    return m


def _real_graph_inputs(model: onnx.ModelProto) -> list[onnx.ValueInfoProto]:
    """Return only genuine model inputs — excluding initializer shadows.

    In ONNX opset ≤ 6, initializers also appear in ``graph.input``.  Filter
    them out so a single-real-input model doesn't look like a multi-input one.
    """
    init_names = {i.name for i in model.graph.initializer}
    return [gi for gi in model.graph.input if gi.name not in init_names]


def resolve_concrete_shapes(
    model: onnx.ModelProto,
    feeds: dict[str, np.ndarray],
) -> dict[str, list[int]]:
    """Run one forward pass and return tensor_name → concrete shape for all outputs.

    ``feeds`` must supply one ndarray per real model input (as returned by
    ``_real_graph_inputs``).  Legacy single-input callers can still pass
    a bare ndarray via ``resolve_concrete_shapes_single``.
    """
    exposed = _expose_intermediates(model)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as fh:
        tmp = fh.name
    try:
        onnx.save(exposed, tmp)
        sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
        session_inputs = [i.name for i in sess.get_inputs()]
        missing = [name for name in session_inputs if name not in feeds]
        if missing:
            raise GoldenGenerationError(
                f"resolve_concrete_shapes is missing ndarrays for graph inputs: {missing}. "
                f"Provide one feed per model input."
            )
        out_names = [o.name for o in sess.get_outputs()]
        results = sess.run(None, {n: feeds[n] for n in session_inputs})
        shapes: dict[str, list[int]] = {}
        for name, arr in zip(out_names, results):
            if arr is not None and hasattr(arr, "shape"):
                shapes[name] = list(arr.shape)
        for name, arr in feeds.items():
            shapes[name] = list(arr.shape)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return shapes


def backfill_spec_shapes(
    specs: list[OnnxLayerSpec],
    concrete_shapes: dict[str, list[int]],
) -> None:
    """Replace -1 dimensions in spec shapes with concrete values from a forward pass."""
    def _resolve(shape: list[int], tensor_name: str) -> list[int]:
        if tensor_name in concrete_shapes and any(d <= 0 for d in shape):
            return list(concrete_shapes[tensor_name])
        return shape

    for spec in specs:
        if spec.input_tensor_names:
            spec.input_shape = _resolve(spec.input_shape, spec.input_tensor_names[0])
        spec.output_shape = _resolve(spec.output_shape, spec.output_tensor_name)
        # For add: also resolve lhs/rhs (used for input_width_bits)
        if spec.op_type == "add" and any(d <= 0 for d in spec.input_shape):
            if spec.add_lhs_tensor in concrete_shapes:
                spec.input_shape = list(concrete_shapes[spec.add_lhs_tensor])


def calibrate_onnx(
    model: onnx.ModelProto,
    calibration_feeds: list[dict[str, np.ndarray]],
) -> dict[str, float]:
    """Return tensor_name → max_abs_value for all intermediate tensors.

    Each ``calibration_feeds`` entry must contain one ndarray per real model
    input.  Supports multi-input graphs.
    """
    exposed = _expose_intermediates(model)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as fh:
        tmp = fh.name
    try:
        onnx.save(exposed, tmp)
        sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
        session_inputs = [i.name for i in sess.get_inputs()]
        out_names = [o.name for o in sess.get_outputs()]
        stats: dict[str, float] = {}
        for feed in calibration_feeds:
            missing = [name for name in session_inputs if name not in feed]
            if missing:
                raise GoldenGenerationError(
                    f"calibrate_onnx feed is missing inputs: {missing}."
                )
            results = sess.run(None, {n: feed[n] for n in session_inputs})
            for name, arr in zip(out_names, results):
                if arr is not None and hasattr(arr, "shape") and arr.size > 0:
                    max_abs = float(np.abs(arr).max())
                    stats[name] = max(stats.get(name, 0.0), max_abs)
            # Also track input tensor stats
            for name, arr in feed.items():
                if arr.size > 0:
                    max_abs = float(np.abs(arr).max())
                    stats[name] = max(stats.get(name, 0.0), max_abs)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return stats


def _safe_scale(max_abs: float) -> float:
    return max_abs / 127.0 if max_abs > 0.0 else 1.0


def fill_calibration_stats(
    specs: list[OnnxLayerSpec],
    stats: dict[str, float],
    network_input_name: str,
    network_input_max_abs: float = 128.0,
) -> None:
    """Populate scale factors on each spec from calibration statistics."""
    # The network input is fed as INT8-range floats, so its max-abs ≈ 128
    stats_with_input = dict(stats)
    stats_with_input.setdefault(network_input_name, network_input_max_abs)

    for spec in specs:
        # Input scale
        in_name = spec.input_tensor_names[0] if spec.input_tensor_names else ""
        spec.input_scale = _safe_scale(stats_with_input.get(in_name, network_input_max_abs))
        # Output scale
        spec.output_scale = _safe_scale(stats.get(spec.output_tensor_name, 128.0))

        if spec.op_type == "conv2d" and spec.weight is not None:
            spec.weight_scale = _safe_scale(float(np.abs(spec.weight).max()))

        if spec.op_type == "add":
            spec.lhs_scale = _safe_scale(
                stats_with_input.get(spec.add_lhs_tensor, network_input_max_abs)
            )
            spec.rhs_scale = _safe_scale(
                stats_with_input.get(spec.add_rhs_tensor, network_input_max_abs)
            )


# ---------------------------------------------------------------------------
# INT8 quantisation helpers
# ---------------------------------------------------------------------------

def _quantize_weights_int8(weight: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.round(weight / scale), -128, 127).astype(np.int8)


def _composite_conv_scale(spec: OnnxLayerSpec) -> float:
    """Return the composite INT8-requantisation multiplier for a conv layer.

    ``scale_factor`` in the LayerIR (and therefore the RTL's SCALE_MULT /
    SCALE_SHIFT) must encode the mapping from the integer accumulator domain
    back to INT8 output:

        output_int = clamp(round((acc_int + bias_int) * S), -128, 127)
        S          = input_scale * weight_scale / output_scale

    where bias_int = bias_float / (input_scale * weight_scale) sits in the
    same accumulator domain as acc_int.  The old code used just
    ``weight_scale``; that only works when input_scale ≈ output_scale ≈ 1,
    which is the case for synthetic INT8 calibration but not for real
    activation-range calibration.
    """
    acc_scale = spec.input_scale * spec.weight_scale
    if spec.output_scale == 0.0:
        return float(acc_scale)
    return float(acc_scale / spec.output_scale)


def _quantize_bias_int32(
    bias: np.ndarray,
    input_scale: float,
    weight_scale: float,
) -> np.ndarray:
    """Quantize bias to INT32 accumulator domain.

    bias_int32 = round(bias_float / (input_scale * weight_scale))
    This places the bias in the same integer domain as the MAC accumulator
    so the RTL can add them directly without rescaling.
    """
    acc_scale = input_scale * weight_scale or 1.0
    return np.round(bias / acc_scale).astype(np.int32)


# ---------------------------------------------------------------------------
# Build and run the quantised INT8 forward network
# ---------------------------------------------------------------------------

def _build_int8_module(spec: OnnxLayerSpec, rtl_compat_conv: bool = True) -> nn.Module:
    """Instantiate an Int8 PyTorch module from an OnnxLayerSpec.

    ``rtl_compat_conv`` selects between two Conv2d golden-generation modes:

    * ``True``  — match the current RTL's single-MAC / spatially-summed
      datapath.  Goldens verify against RTL bit-for-bit.
    * ``False`` — faithful 2D convolution.  Goldens match the float ONNX
      model but will fail against single-MAC RTL on any non-1×1 kernel.
    """
    if spec.op_type == "conv2d":
        assert spec.weight is not None
        w_int8 = _quantize_weights_int8(spec.weight, spec.weight_scale)
        w_tensor = torch.tensor(w_int8.astype(np.float32))
        b_int32 = _quantize_bias_int32(spec.bias, spec.input_scale, spec.weight_scale)
        b_tensor = torch.tensor(b_int32.astype(np.float32))
        # Composite requantisation multiplier.  Derivation:
        #   input_float = input_int * input_scale
        #   weight_float = weight_int * weight_scale
        #   acc_float    = Σ(input_int * weight_int) * input_scale * weight_scale
        #   output_float = acc_float + bias_float
        #   output_int   = round(output_float / output_scale)
        #                = round( (acc_int + bias_int) * (input_scale * weight_scale / output_scale) )
        # where bias_int = bias_float / (input_scale * weight_scale), which is
        # exactly what _quantize_bias_int32() produces. The RTL applies this
        # composite multiplier as SCALE_MULT / 2^SCALE_SHIFT on (acc + bias).
        acc_scale = spec.input_scale * spec.weight_scale
        if spec.output_scale == 0.0:
            composite_scale = acc_scale
        else:
            composite_scale = acc_scale / spec.output_scale
        return Int8Conv2d(
            weight=w_tensor,
            bias=b_tensor,
            stride=spec.stride,
            padding=spec.padding,
            dilation=spec.dilation,
            groups=spec.groups,
            scale_factor=float(composite_scale),
            rtl_compat=rtl_compat_conv,
        )
    if spec.op_type == "relu":
        return Int8ReLU()
    if spec.op_type == "maxpool":
        return Int8MaxPool2d(
            kernel_size=spec.pool_kernel,
            stride=spec.pool_stride,
            padding=spec.pool_padding,
            ceil_mode=spec.pool_ceil_mode,
        )
    if spec.op_type == "add":
        return Int8Add(
            lhs_scale_factor=spec.lhs_scale,
            rhs_scale_factor=spec.rhs_scale,
            output_scale_factor=spec.output_scale,
        )
    raise GoldenGenerationError(f"Unsupported op_type '{spec.op_type}' in _build_int8_module.")


def run_int8_network(
    specs: list[OnnxLayerSpec],
    modules: dict[str, nn.Module],
    input_tensor: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run one input through the quantised INT8 network.

    Returns a mapping of ONNX tensor name → INT8 activation tensor for every
    supported layer's input and output.
    """
    tensors: dict[str, torch.Tensor] = {}

    # Seed the tensor map with the network input
    network_input_name = specs[0].input_tensor_names[0]
    tensors[network_input_name] = quantize_tensor_to_int8_range(input_tensor.clone())

    for spec in specs:
        mod = modules[spec.module_id]

        if spec.op_type == "add":
            lhs = _require_tensor(tensors, spec.add_lhs_tensor, spec.module_id, "lhs")
            rhs = _require_tensor(tensors, spec.add_rhs_tensor, spec.module_id, "rhs")
            out = mod(lhs, rhs)
        else:
            inp = _require_tensor(tensors, spec.input_tensor_names[0], spec.module_id, "input")
            out = mod(inp)

        tensors[spec.output_tensor_name] = quantize_tensor_to_int8_range(out)

    return tensors


def _require_tensor(
    tensors: dict[str, torch.Tensor],
    name: str,
    mid: str,
    role: str,
) -> torch.Tensor:
    """Look up a tensor by name, failing fast if it is missing.

    Silent fallback (e.g. "most recently produced") would corrupt golden
    vectors: a downstream Conv consuming a tensor produced by an unsupported
    op (Resize, AveragePool, GroupNorm, …) would be fed the wrong activation
    and its goldens would silently validate against the wrong RTL behaviour.
    Fail loudly so the user can either extend the supported-op set or
    pre-process the ONNX graph.
    """
    if name in tensors:
        return tensors[name]
    raise GoldenGenerationError(
        f"Layer '{mid}' {role} tensor '{name}' is not available in the INT8 simulation. "
        f"The ONNX graph contains an unsupported op that produces '{name}'. "
        f"Supported ops: Conv, Relu, MaxPool, Add (and Clip with min=0). "
        f"Inspect the simplified graph (onnxsim) and either extend the frontend "
        f"or rewrite the offending subgraph."
    )


# ---------------------------------------------------------------------------
# MaxPool latency calculation
# ---------------------------------------------------------------------------

def compute_maxpool_latency_cycles(spec: OnnxLayerSpec) -> int:
    """Estimate pipeline latency for a line-buffer MaxPool implementation.

    A 2D sliding-window MaxPool with a line buffer first outputs after
    (kernel_h - 1) complete input rows plus kernel_w input pixels.
    Input row width includes the effective padded width.
    """
    kh, kw = spec.pool_kernel[0], spec.pool_kernel[1]
    ph = spec.pool_padding[0] if spec.pool_padding else 0
    pw = spec.pool_padding[1] if len(spec.pool_padding) > 1 else ph

    input_w = spec.input_shape[3] if len(spec.input_shape) >= 4 and spec.input_shape[3] > 0 else 0
    effective_w = input_w + 2 * pw
    # rows_before_first_output = kh - 1, then kw more pixels
    latency = (kh - 1) * effective_w + kw
    return max(1, int(latency))


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------

def _write_conv_hex_artifacts(
    spec: OnnxLayerSpec,
    repo_root: Path,
) -> tuple[list[int], list[int]]:
    """Write weight and bias hex files for a conv2d spec. Return (weight_values, bias_values)."""
    assert spec.weight is not None
    w_int8 = _quantize_weights_int8(spec.weight, spec.weight_scale)
    b_int32 = _quantize_bias_int32(spec.bias, spec.input_scale, spec.weight_scale)

    weight_values = [int(v) for v in w_int8.reshape(-1).tolist()]
    bias_values = [int(v) for v in b_int32.reshape(-1).tolist()]

    weights_path, bias_path = get_weight_artifact_paths(repo_root, spec.module_id)
    write_signed_int8_hex(weight_values, weights_path)
    write_signed_int32_hex(bias_values, bias_path)
    return weight_values, bias_values


def _write_empty_weight_file(repo_root: Path, module_id: str) -> None:
    """Write an empty placeholder hex file for ops without weights (relu, maxpool, add)."""
    weights_path, _ = get_weight_artifact_paths(repo_root, module_id)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_text("", encoding="utf8")


# ---------------------------------------------------------------------------
# Golden vector helpers
# ---------------------------------------------------------------------------

def _build_goldin_for_spec(
    spec: OnnxLayerSpec,
    all_tensors_per_sample: list[dict[str, torch.Tensor]],
) -> list[list[int]]:
    """Pack golden input vectors for one spec from the per-sample tensor maps."""
    in_bits = _spec_input_width_bits(spec)

    if spec.op_type == "add":
        lhs_tensors = [t[spec.add_lhs_tensor] for t in all_tensors_per_sample if spec.add_lhs_tensor in t]
        rhs_tensors = [t[spec.add_rhs_tensor] for t in all_tensors_per_sample if spec.add_rhs_tensor in t]
        if not lhs_tensors:
            raise GoldenGenerationError(f"No lhs tensors found for add layer '{spec.module_id}'.")
        return pack_paired_tensors_to_bus_words(
            lhs_tensors, rhs_tensors, in_bits, context=f"{spec.module_id}.goldin"
        )

    in_name = spec.input_tensor_names[0]
    tensors = [t[in_name] for t in all_tensors_per_sample if in_name in t]
    if not tensors:
        raise GoldenGenerationError(
            f"Input tensor '{in_name}' not found in simulation outputs for layer '{spec.module_id}'."
        )
    return pack_tensor_vectors_to_bus_words(tensors, in_bits, context=f"{spec.module_id}.goldin")


def _build_goldout_for_spec(
    spec: OnnxLayerSpec,
    all_tensors_per_sample: list[dict[str, torch.Tensor]],
) -> list[list[int]]:
    out_bits = _spec_output_width_bits(spec)
    out_name = spec.output_tensor_name
    tensors = [t[out_name] for t in all_tensors_per_sample if out_name in t]
    if not tensors:
        raise GoldenGenerationError(
            f"Output tensor '{out_name}' not found in simulation outputs for layer '{spec.module_id}'."
        )
    return pack_tensor_vectors_to_bus_words(tensors, out_bits, context=f"{spec.module_id}.goldout")


def _spec_input_width_bits(spec: OnnxLayerSpec) -> int:
    if spec.op_type == "add":
        # Packed wide bus: [lhs_channels*8 + rhs_channels*8]
        return channel_bus_bits_from_shape(spec.input_shape, context=f"{spec.module_id}.input_shape") * 2
    return channel_bus_bits_from_shape(spec.input_shape, context=f"{spec.module_id}.input_shape")


def _spec_output_width_bits(spec: OnnxLayerSpec) -> int:
    return channel_bus_bits_from_shape(spec.output_shape, context=f"{spec.module_id}.output_shape")


# ---------------------------------------------------------------------------
# Pipeline latency
# ---------------------------------------------------------------------------

def _conv_mac_parallelism(spec: OnnxLayerSpec) -> int:
    """Accumulator-group size for an ONNX conv layer: min(OC, MAX_PARALLEL_MACS)."""
    if spec.weight is None or len(spec.weight.shape) < 1:
        return 1
    return conv_mac_parallelism(int(spec.weight.shape[0]))


def _pipeline_latency(spec: OnnxLayerSpec) -> int:
    if spec.op_type == "conv2d":
        weight_shape = list(spec.weight.shape) if spec.weight is not None else [1, 1, 1, 1]
        return compute_conv2d_latency_cycles(
            weight_shape,
            input_shape=spec.input_shape,
            stride=spec.stride,
            padding=spec.padding,
            mac_parallelism=_conv_mac_parallelism(spec),
        )
    if spec.op_type == "maxpool":
        return compute_maxpool_latency_cycles(spec)
    if spec.op_type == "add":
        return compute_add_latency_cycles(spec.output_shape)
    # relu: 1 cycle
    return 1


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_pipeline_ir_from_onnx(
    onnx_path: Path,
    repo_root: Path,
    model_name: str | None = None,
    generated_at: str | None = None,
    num_calibration_samples: int = 8,
    calibration_seed: int = 0,
    rtl_compat_conv: bool = False,
) -> dict[str, Any]:
    """Build a complete PipelineIR dict from an ONNX model file.

    Parameters
    ----------
    onnx_path:
        Path to the .onnx file (already exported from PyTorch or elsewhere).
    repo_root:
        Repository root used to resolve output/ paths.
    model_name:
        Logical model name recorded in the PipelineIR header (defaults to the
        stem of onnx_path).
    generated_at:
        ISO-8601 UTC timestamp; defaults to now.
    num_calibration_samples:
        Number of synthetic INT8-range inputs used for calibration.
    calibration_seed:
        RNG seed for reproducible calibration inputs.

    Returns
    -------
    dict
        A PipelineIR payload that can be written to output/layer_ir.json and
        validated by validate_pipeline_ir_payload().
    """
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    model_name = model_name or onnx_path.stem
    generated_at = generated_at or utc_now_iso8601()

    # --- Step 1: Load and simplify ------------------------------------------
    model = load_onnx(onnx_path)
    model = simplify_onnx(model)

    # --- Step 2: Extract layer specs ----------------------------------------
    specs = extract_layer_specs(model)

    # --- Step 2b: Resolve real model inputs (excluding initializer shadows) -
    real_inputs = _real_graph_inputs(model)
    if not real_inputs:
        raise GoldenGenerationError(
            "ONNX model exposes no real graph inputs (all graph.input entries are initializers)."
        )
    if len(real_inputs) > 1:
        # The RTL pipeline is streaming-single-input.  Multi-input ONNX graphs
        # need explicit pre-processing (merge inputs via Concat, strip auxiliary
        # inputs, etc.) before they can drive a single data_in port.
        names = [i.name for i in real_inputs]
        raise GoldenGenerationError(
            f"ONNX model has {len(real_inputs)} real inputs ({names}); the nn2rtl "
            f"pipeline is single-input (one streaming data_in bus). Merge or drop "
            f"auxiliary inputs before passing the model to this frontend."
        )
    network_input_name = real_inputs[0].name

    # --- Step 2c: Fail fast if the supported-op chain has a gap ------------
    graph_input_names = {gi.name for gi in real_inputs}
    validate_graph_completeness(specs, graph_input_names)

    # Also require that every model output tensor is produced by the spec set.
    # Without this, a Conv->Flatten->Gemm (or Conv->Sigmoid) graph would
    # silently collapse to just Conv — technically valid by completeness but
    # semantically wrong.
    graph_output_names = {o.name for o in model.graph.output}
    validate_graph_outputs_covered(specs, graph_output_names, graph_input_names)

    # --- Step 3: Determine input shape --------------------------------------
    # Prefer the first spec's input shape; fall back to the graph input's shape.
    first_input_shape = list(specs[0].input_shape)
    if any(d <= 0 for d in first_input_shape):
        ti = real_inputs[0].type.tensor_type
        first_input_shape = [
            int(d.dim_value) if d.dim_value > 0 else 1
            for d in ti.shape.dim
        ]
        if first_input_shape:
            first_input_shape[0] = 1  # force batch=1

    # --- Step 4: Build synthetic calibration inputs -------------------------
    rng = np.random.default_rng(calibration_seed)
    cal_feeds: list[dict[str, np.ndarray]] = [
        {network_input_name: rng.integers(
            -128, 128, size=tuple(first_input_shape), dtype=np.int32
        ).astype(np.float32)}
        for _ in range(num_calibration_samples)
    ]

    # --- Step 4b: Resolve concrete shapes (dynamic ONNX exports have -1 dims)
    concrete_shapes = resolve_concrete_shapes(model, cal_feeds[0])
    backfill_spec_shapes(specs, concrete_shapes)
    first_input_shape = [
        concrete_shapes.get(network_input_name, [1] * len(first_input_shape))[i]
        if i < len(concrete_shapes.get(network_input_name, [])) else d
        for i, d in enumerate(first_input_shape)
    ]

    # --- Step 5: Calibrate via ONNX Runtime ---------------------------------
    cal_stats = calibrate_onnx(model, cal_feeds)
    fill_calibration_stats(
        specs, cal_stats,
        network_input_name=network_input_name,
        network_input_max_abs=128.0,
    )

    # --- Step 5b: Warn loudly if caller opts into the legacy RTL-compat path
    if rtl_compat_conv:
        spatial_convs = [
            s.module_id for s in specs
            if s.op_type == "conv2d"
            and s.weight is not None
            and int(s.weight.shape[2]) * int(s.weight.shape[3]) > 1
        ]
        if spatial_convs:
            warnings.warn(
                "rtl_compat_conv=True is the LEGACY path: the following spatial conv "
                f"layers {spatial_convs} will be approximated as spatially-summed 1x1 "
                "convolutions (weight.sum(dim=(2,3))). Goldens will match single-pixel "
                "MAC RTL but will NOT match the real ONNX model. Prefer the default "
                "(rtl_compat_conv=False) which matches the line-buffer datapath in "
                "nn2rtl-plugin/agents/foundry.md § 'Spatial conv datapath'.",
                RuntimeWarning,
                stacklevel=2,
            )

    # --- Step 6: Build INT8 modules -----------------------------------------
    modules: dict[str, nn.Module] = {
        s.module_id: _build_int8_module(s, rtl_compat_conv=rtl_compat_conv) for s in specs
    }

    # --- Step 7: Run INT8 simulation to capture golden activations ----------
    torch.manual_seed(calibration_seed)
    input_tensors = [
        quantize_tensor_to_int8_range(
            torch.tensor(cal_feeds[i][network_input_name]).reshape(first_input_shape)
        )
        for i in range(num_calibration_samples)
    ]

    all_tensors: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for inp_t in input_tensors:
            t_map = run_int8_network(specs, modules, inp_t)
            all_tensors.append(t_map)

    # --- Step 8: Write hex + golden artifacts and build LayerIR entries -----
    layer_payloads: list[dict[str, Any]] = []

    for spec in specs:
        weights_path, bias_path_obj = get_weight_artifact_paths(repo_root, spec.module_id)
        weight_shape: list[int]
        num_weights: int

        if spec.op_type == "conv2d":
            weight_values, bias_values = _write_conv_hex_artifacts(spec, repo_root)
            weight_shape = list(spec.weight.shape)
            num_weights = int(spec.weight.size)
        else:
            _write_empty_weight_file(repo_root, spec.module_id)
            # bias file: empty placeholder
            bias_path_obj.parent.mkdir(parents=True, exist_ok=True)
            bias_path_obj.write_text("", encoding="utf8")
            weight_shape = [1]
            num_weights = 0

        in_bits = _spec_input_width_bits(spec)
        out_bits = _spec_output_width_bits(spec)
        latency = _pipeline_latency(spec)

        goldin_path, goldout_path = get_golden_artifact_paths(repo_root, spec.module_id)

        goldin_vectors = _build_goldin_for_spec(spec, all_tensors)
        goldout_vectors = _build_goldout_for_spec(spec, all_tensors)

        write_golden_vector_file(goldin_vectors, goldin_path, bus_bits=in_bits)
        write_golden_vector_file(goldout_vectors, goldout_path, bus_bits=out_bits)

        layer_payload: dict[str, Any] = {
            "module_id": spec.module_id,
            "op_type": spec.op_type,
            "input_shape": spec.input_shape,
            "output_shape": spec.output_shape,
            "weights_path": weights_path.resolve().as_posix(),
            "bias_path": bias_path_obj.resolve().as_posix() if spec.op_type == "conv2d" else None,
            "weight_shape": weight_shape,
            "num_weights": num_weights,
            # For conv2d the RTL multiplies (acc + bias_int32) by this factor
            # to requantise back to INT8; it is the composite (input_scale *
            # weight_scale / output_scale) not weight_scale alone.  For
            # activation ops (relu / maxpool / add) the LayerIR scale_factor
            # is the output activation scale used by the quantised-add formula.
            "scale_factor": (
                _composite_conv_scale(spec)
                if spec.op_type == "conv2d"
                else float(spec.output_scale)
            ),
            "zero_point": 0,
            "pipeline_latency_cycles": latency,
            "clock_period_ns": 20,
            "input_width_bits": in_bits,
            "output_width_bits": out_bits,
            **SIGNAL_LITERALS,
            "golden_inputs_path": goldin_path.resolve().as_posix(),
            "golden_outputs_path": goldout_path.resolve().as_posix(),
        }

        if spec.op_type == "add":
            layer_payload["lhs_scale_factor"] = float(spec.lhs_scale)
            layer_payload["rhs_scale_factor"] = float(spec.rhs_scale)

        if spec.op_type == "conv2d":
            layer_payload["stride"] = list(spec.stride)
            layer_payload["padding"] = list(spec.padding)
            mac_parallelism = _conv_mac_parallelism(spec)
            weight_bank_paths = write_weight_bank_hex_files(
                weight_values,
                weight_shape,
                mac_parallelism,
                repo_root,
                spec.module_id,
            )
            layer_payload["mac_parallelism"] = mac_parallelism
            layer_payload["weight_bank_paths"] = [
                bank_path.resolve().as_posix() for bank_path in weight_bank_paths
            ]

        if spec.op_type == "maxpool":
            layer_payload["kernel_size"] = spec.pool_kernel
            layer_payload["pool_stride"] = spec.pool_stride
            layer_payload["pool_padding"] = spec.pool_padding

        layer_payloads.append(layer_payload)

    payload: dict[str, Any] = {
        "model_name": model_name,
        "quantization": "int8_symmetric_per_tensor",
        "generated_at": generated_at,
        "layers": layer_payloads,
    }

    validate_pipeline_ir_payload(payload)
    return payload


# ---------------------------------------------------------------------------
# PyTorch → ONNX export helper
# ---------------------------------------------------------------------------

def export_pytorch_to_onnx(
    model: torch.nn.Module,
    onnx_path: Path,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    opset: int = DEFAULT_ONNX_OPSET,
) -> Path:
    """Export a PyTorch model to ONNX and return the output path."""
    model = model.eval()
    dummy = torch.randn(*input_shape)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    return onnx_path
