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
import os
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
    Int8Gemm,
    Int8GlobalAveragePool,
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
    op_type: str                      # "conv2d" | "relu" | "add" | "maxpool" | "global_avg_pool" | "gemm"
    input_tensor_names: list[str]     # ONNX tensor name(s) consumed
    output_tensor_name: str           # ONNX tensor name produced
    input_shape: list[int]            # [N, C, H, W] for conv-like; [N, C, 1, 1] or [N, K] for classifier head
    output_shape: list[int]           # [N, C, H, W] for conv-like; [N, M] for gemm

    # Conv2d / Gemm shared weight storage --------------------------------------
    # For Gemm the weight is 2D [K, M] (transB-canonicalized: each row = one
    # output channel's weights). Bias has shape [M].
    weight: Optional[np.ndarray] = None   # float32 [OC, IC/G, KH, KW] for conv, [M, K] for gemm
    bias: Optional[np.ndarray] = None     # float32 [OC] or [M]
    stride: list[int] = field(default_factory=lambda: [1, 1])
    padding: list[int] = field(default_factory=lambda: [0, 0])
    dilation: list[int] = field(default_factory=lambda: [1, 1])
    groups: int = 1

    # MaxPool2d --------------------------------------------------------------
    pool_kernel: list[int] = field(default_factory=lambda: [1, 1])
    pool_stride: list[int] = field(default_factory=lambda: [1, 1])
    pool_padding: list[int] = field(default_factory=lambda: [0, 0])
    pool_ceil_mode: bool = False

    # GlobalAveragePool -------------------------------------------------------
    # Captured input spatial dims (H, W) so the quantizer/golden code can
    # divide by H*W. Stored explicitly instead of recomputing from input_shape
    # in case shape inference left -1s.
    gap_spatial: list[int] = field(default_factory=lambda: [1, 1])

    # Gemm --------------------------------------------------------------------
    # Canonicalized so weight is [M, K] (M output features, K input features),
    # matching PyTorch nn.Linear.weight. alpha/beta default to 1.0.
    gemm_in_features: int = 0
    gemm_out_features: int = 0

    # Add (residual) wiring --------------------------------------------------
    add_lhs_tensor: str = ""
    add_rhs_tensor: str = ""

    # ReLU6 / clipped-activation upper bound. None means unbounded ReLU; a
    # finite value means the activation is clamped to [0, clip_max] in float
    # domain (e.g. MobileNetV2 sets 6.0 for every ReLU6 in the network).
    clip_max: Optional[float] = None

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

    if len(pads) >= 4 and (int(pads[0]) != int(pads[2]) or int(pads[1]) != int(pads[3])):
        raise GoldenGenerationError(
            f"Conv node '{node.name or mid}' uses asymmetric padding {pads}; "
            "LayerIR currently represents conv padding as [h, w], so asymmetric pads must be "
            "made explicit by Pad nodes or rejected instead of silently collapsed."
        )

    # ONNX symmetric padding: take begin values only after verifying begin==end.
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
    clip_max: Optional[float] = None,
) -> OnnxLayerSpec:
    inp = node.input[0]
    return OnnxLayerSpec(
        module_id=mid,
        op_type="relu",
        input_tensor_names=[inp],
        output_tensor_name=node.output[0],
        input_shape=shapes.get(inp, [-1, -1, -1, -1]),
        output_shape=shapes.get(node.output[0], [-1, -1, -1, -1]),
        clip_max=clip_max,
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


def _extract_global_avg_pool(
    node: onnx.NodeProto,
    shapes: dict[str, list[int]],
    mid: str,
) -> OnnxLayerSpec:
    inp = node.input[0]
    in_shape = shapes.get(inp, [-1, -1, -1, -1])
    h = int(in_shape[2]) if len(in_shape) >= 4 and in_shape[2] > 0 else 1
    w = int(in_shape[3]) if len(in_shape) >= 4 and in_shape[3] > 0 else 1
    out_shape = shapes.get(node.output[0], [-1, -1, 1, 1])
    return OnnxLayerSpec(
        module_id=mid,
        op_type="global_avg_pool",
        input_tensor_names=[inp],
        output_tensor_name=node.output[0],
        input_shape=in_shape,
        output_shape=out_shape,
        gap_spatial=[h, w],
    )


def _extract_gemm(
    node: onnx.NodeProto,
    shapes: dict[str, list[int]],
    inits: dict[str, np.ndarray],
    mid: str,
) -> OnnxLayerSpec:
    """Canonicalize a Gemm node into a `weight = [M, K]` (PyTorch nn.Linear
    layout) plus optional `bias = [M]`. ONNX Gemm carries `transA` / `transB`
    attributes which control the layout of A (input) and B (weight). For most
    PyTorch-exported networks the input is already row-major (A non-transposed)
    and weight is `[M, K]` already (transB=1). We canonicalize to that layout
    so the downstream quantizer / RTL can assume a fixed shape.
    """
    in_tensor = node.input[0]
    w_name = node.input[1]
    if w_name not in inits:
        raise GoldenGenerationError(
            f"Gemm node '{node.name or mid}' weight input '{w_name}' is not a "
            f"graph initializer; dynamic-weight Gemm is unsupported."
        )
    weight = np.array(inits[w_name], dtype=np.float32)
    trans_b = bool(_get_attr(node, "transB", 0))
    trans_a = bool(_get_attr(node, "transA", 0))
    if trans_a:
        raise GoldenGenerationError(
            f"Gemm node '{node.name or mid}' has transA=1; only row-major inputs "
            f"are supported by this frontend."
        )
    # ONNX with transB=1: B is already [M, K]. Otherwise B is [K, M] and must
    # be transposed to match the [M, K] canonical form.
    if not trans_b:
        weight = weight.T.copy()
    out_features, in_features = int(weight.shape[0]), int(weight.shape[1])

    bias = None
    if len(node.input) >= 3 and node.input[2] and node.input[2] in inits:
        bias = np.array(inits[node.input[2]], dtype=np.float32).reshape(-1)
        if bias.shape[0] != out_features:
            raise GoldenGenerationError(
                f"Gemm node '{node.name or mid}' bias shape {bias.shape} does not "
                f"match output features {out_features}."
            )

    # Gemm sees its activations after any upstream Flatten/Reshape — the
    # input to the dot product is a flat [N, K] vector. The ONNX shape inference
    # of the *pre-Flatten* tensor (e.g. [N, C, H, W]) is misleading here
    # because the alias rewriter folds Flatten out and Gemm consumes the
    # underlying [N, K] view. We canonicalize input_shape to [N, K] so the
    # downstream `input_width_bits` calculation produces K*8 (one beat of K
    # bytes), matching the Gemm doc's "one beat of K bytes" contract.
    raw_input_shape = shapes.get(in_tensor, [-1, in_features])
    batch_dim_raw = (
        raw_input_shape[0]
        if isinstance(raw_input_shape, (list, tuple)) and len(raw_input_shape) > 0
        else -1
    )
    try:
        batch_dim = int(batch_dim_raw)
    except (TypeError, ValueError):
        batch_dim = -1
    canonical_input_shape = [batch_dim if batch_dim > 0 else -1, in_features]

    return OnnxLayerSpec(
        module_id=mid,
        op_type="gemm",
        input_tensor_names=[in_tensor],
        output_tensor_name=node.output[0],
        input_shape=canonical_input_shape,
        output_shape=shapes.get(node.output[0], [-1, out_features]),
        weight=weight,
        bias=bias,
        gemm_in_features=in_features,
        gemm_out_features=out_features,
    )


# ---------------------------------------------------------------------------
# Full graph spec extraction
# ---------------------------------------------------------------------------

def extract_layer_specs(model: onnx.ModelProto) -> list[OnnxLayerSpec]:
    """Walk the simplified ONNX graph and return one OnnxLayerSpec per supported op.

    Reshape/Flatten/Squeeze/Unsqueeze are treated as tensor-rename passthroughs:
    they do not produce specs, but downstream specs that consume their outputs
    are silently rewritten to point at the underlying upstream tensor. This
    lets a Conv -> GlobalAveragePool -> Flatten -> Gemm chain work even though
    Flatten itself is not a compute op.
    """
    shapes = _tensor_shapes(model)
    inits = _initializers(model)
    op_counts: dict[str, int] = {}
    specs: list[OnnxLayerSpec] = []
    # Map from tensor produced by a passthrough op to the underlying source
    # tensor (transitively resolved).
    tensor_alias: dict[str, str] = {}

    def resolve_alias(name: str) -> str:
        seen: set[str] = set()
        while name in tensor_alias and name not in seen:
            seen.add(name)
            name = tensor_alias[name]
        return name

    for node in model.graph.node:
        op = node.op_type
        clip_max_v: Optional[float] = None

        # Tensor-rename passthroughs. None of these change the underlying byte
        # stream; they only rearrange dims. The downstream Gemm consumes the
        # same channel-major bus the upstream GAP produced.
        if op in ("Reshape", "Flatten", "Squeeze", "Unsqueeze", "Identity"):
            if node.input and node.output:
                src = resolve_alias(node.input[0])
                tensor_alias[node.output[0]] = src
            continue

        # The Concat-of-one node sometimes inserted by torch.onnx.export's
        # dynamo path immediately before Reshape is a no-op when it has a
        # single input; treat it as a passthrough so it doesn't break the
        # extracted spec chain. Multi-input Concat is unsupported and falls
        # through to the generic skip below.
        if op == "Concat" and len(node.input) == 1 and node.output:
            tensor_alias[node.output[0]] = resolve_alias(node.input[0])
            continue

        # Treat ReduceMean(axes=[2, 3], keepdims=True) as GlobalAveragePool.
        # PyTorch's newer ONNX export path lowers AdaptiveAvgPool2d(1) and
        # F.adaptive_avg_pool2d to ReduceMean over the spatial dims; older
        # exports used GlobalAveragePool directly. The numerical behavior is
        # identical, so route both into the same spec.
        if op == "ReduceMean":
            keepdims = bool(_get_attr(node, "keepdims", 1))
            axes_attr = _get_attr(node, "axes", None)
            axes_input = (
                inits[node.input[1]].flatten().tolist()
                if len(node.input) >= 2 and node.input[1] in inits
                else None
            )
            axes_list = (
                list(axes_input)
                if axes_input is not None
                else list(axes_attr) if axes_attr is not None else []
            )
            normalized_axes = sorted(int(a) % 4 for a in axes_list)
            if keepdims and normalized_axes == [2, 3]:
                op = "GlobalAveragePool"
            else:
                continue

        # Treat Clip(min=0) as ReLU. If a finite max is present (e.g. ReLU6
        # uses Clip(min=0, max=6)), carry it forward as clip_max so the
        # quantizer can fold it into the output scale and the verifier can
        # apply the same clip in golden computation.
        if op == "Clip":
            min_v: Optional[float] = None
            max_v: Optional[float] = None
            # min/max are inputs in opset 11+, attributes in opset 6
            if len(node.input) >= 2 and node.input[1] in inits:
                min_v = float(inits[node.input[1]].flat[0])
            else:
                min_v = _get_attr(node, "min", None)
            if len(node.input) >= 3 and node.input[2] in inits:
                max_v = float(inits[node.input[2]].flat[0])
            else:
                max_v = _get_attr(node, "max", None)
            if min_v is not None and float(min_v) == 0.0:
                op = "Relu"
                if max_v is not None and np.isfinite(float(max_v)):
                    clip_max_v = float(max_v)
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
        elif op == "GlobalAveragePool":
            op_key = "global_avg_pool"
        elif op == "Gemm":
            op_key = "gemm"
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
            spec = _extract_conv(node, shapes, inits, mid)
        elif op == "Relu":
            spec = _extract_relu(node, shapes, mid, clip_max=clip_max_v)
        elif op == "MaxPool":
            spec = _extract_maxpool(node, shapes, mid)
        elif op == "Add":
            spec = _extract_add(node, shapes, mid)
        elif op == "GlobalAveragePool":
            spec = _extract_global_avg_pool(node, shapes, mid)
        elif op == "Gemm":
            spec = _extract_gemm(node, shapes, inits, mid)
        else:
            continue
        # Walk through tensor passthroughs so each spec consumes the actual
        # producer's output tensor name, not a downstream Reshape/Flatten
        # rename. validate_graph_completeness compares against producers; the
        # rewrite keeps it accurate even when the user inserted shape ops.
        spec.input_tensor_names = [resolve_alias(t) for t in spec.input_tensor_names]
        if hasattr(spec, "add_lhs_tensor") and spec.add_lhs_tensor:
            spec.add_lhs_tensor = resolve_alias(spec.add_lhs_tensor)
        if hasattr(spec, "add_rhs_tensor") and spec.add_rhs_tensor:
            spec.add_rhs_tensor = resolve_alias(spec.add_rhs_tensor)
        specs.append(spec)

    if not specs:
        raise GoldenGenerationError(
            "No supported ops (Conv, Relu, MaxPool, Add, GlobalAveragePool, Gemm) found in ONNX model after simplification."
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
            if spec.op_type == "gemm":
                # Gemm may consume an aliased pre-Flatten tensor such as
                # [N, C, H, W], but the public Gemm contract is a flat [N, K]
                # vector. Preserve that canonical shape while still resolving
                # a dynamic batch dimension from the concrete producer tensor.
                producer_shape = concrete_shapes.get(spec.input_tensor_names[0])
                batch = spec.input_shape[0] if spec.input_shape else -1
                if producer_shape:
                    batch = producer_shape[0]
                spec.input_shape = [int(batch) if int(batch) > 0 else 1, int(spec.gemm_in_features)]
            else:
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


# --- INT4/INT8 WEIGHT quantization (Scheme A: INT4 weights, INT8 activations) ---
# Activations stay INT8 (_safe_scale, /127). Only WEIGHTS switch width via
# NN2RTL_WEIGHT_BITS (8 default, 4 for INT4). The composite scale_factor and the
# requant pipeline are unchanged — they operate on the 32-bit accumulator and
# pick up the new (coarser) weight_scale automatically.
WEIGHT_BITS = int(os.environ.get("NN2RTL_WEIGHT_BITS", "8"))
_WQMAX = (1 << (WEIGHT_BITS - 1)) - 1   # INT8: 127, INT4: 7
_WQMIN = -(1 << (WEIGHT_BITS - 1))      # INT8: -128, INT4: -8


def _safe_weight_scale(max_abs: float) -> float:
    return max_abs / float(_WQMAX) if max_abs > 0.0 else 1.0


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
        # Output scale. For relu layers carrying a ReLU6-style clip ceiling,
        # use max(observed, clip_max) as the calibration upper bound so the
        # scale is anchored on the activation's theoretical range, not on
        # whatever the synthetic calibration data happened to excite. Without
        # this, calibration that never saturates the clip produces a scale
        # tighter than 6/128, and real inputs near the clip saturate to INT8
        # 127 prematurely.
        observed = stats.get(spec.output_tensor_name, 128.0)
        if spec.op_type == "relu" and spec.clip_max is not None:
            observed = max(float(observed), float(spec.clip_max))
        spec.output_scale = _safe_scale(observed)

        if spec.op_type == "conv2d" and spec.weight is not None:
            _dw = (int(spec.groups) == int(spec.weight.shape[0])) and \
                  (int(spec.weight.shape[1]) == 1)
            if _dw and os.environ.get("NN2RTL_DW_PER_CHANNEL", "1") != "0":
                # [ACCURACY 2026-06-08] per-CHANNEL INT8 for DEPTHWISE convs. DEFAULT ON: VERIFIED
                # +4.00% deployed top-1 (67.27 -> 71.27 on 1500 imgs, all trust gates) AND byte-exact
                # e2e 8/8 with the per-OC scale-ROM RTL (node_conv_8XX scale_rom + scale.mem). Per-
                # tensor depthwise quant was the dominant MBv2 INT8 penalty. Plain per-OUTPUT-CHANNEL
                # max/qmax (INT8 is forgiving; no GPTQ/Hessian). Sets weight_scale_per_oc + gptq_qweight
                # so the EXISTING per-OC downstream (_spec_int_weight_and_scale / _spec_bias_int /
                # _composite_conv_scale_per_oc / layer_ir export) applies. Pointwise/engine convs stay
                # per-tensor (the engine already requants per-OC). Set NN2RTL_DW_PER_CHANNEL=0 to revert.
                _W = spec.weight
                _per_oc = np.maximum(
                    np.abs(_W).reshape(_W.shape[0], -1).max(axis=1) / _WQMAX, 1e-12)
                spec.weight_scale_per_oc = _per_oc.astype(np.float64)
                spec.gptq_qweight = np.clip(
                    np.round(_W / _per_oc.reshape((-1,) + (1,) * (_W.ndim - 1))),
                    _WQMIN, _WQMAX).astype(np.int8)
            else:
                spec.weight_scale = _safe_weight_scale(float(np.abs(spec.weight).max()))

        if spec.op_type == "gemm" and spec.weight is not None:
            spec.weight_scale = _safe_weight_scale(float(np.abs(spec.weight).max()))

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
    # Name kept for call-site stability; clamp range follows WEIGHT_BITS
    # (INT8: [-128,127], INT4: [-8,7]). Stored in int8 either way (INT4 values
    # fit; the nibble-packing happens later in repack_weights_wide.py).
    return np.clip(np.round(weight / scale), _WQMIN, _WQMAX).astype(np.int8)


def _layer_ir_scale_factor(spec: OnnxLayerSpec) -> float:
    """Per-op LayerIR `scale_factor` field. See call site for the per-op
    semantics; this helper centralizes the dispatch so the frontend stays
    consistent with the pattern docs and the Int8 modules."""
    if spec.op_type in ("conv2d", "gemm"):
        return _composite_conv_scale(spec)
    if spec.op_type == "global_avg_pool":
        h, w = spec.gap_spatial[0], spec.gap_spatial[1]
        hw = max(1, int(h) * int(w))
        if spec.output_scale == 0.0:
            return float(spec.input_scale) / hw
        return float(spec.input_scale) / float(spec.output_scale) / hw
    if spec.op_type == "relu" and spec.clip_max is not None:
        if spec.output_scale == 0.0:
            return float(spec.input_scale)
        return float(spec.input_scale) / float(spec.output_scale)
    return float(spec.output_scale)


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
# Per-output-channel INT4 (GPTQ) helpers — Scheme A' (INT4 w / INT8 act,
# per-output-channel weight scale). Gated by WEIGHT_BITS<8; naive per-tensor
# INT4 collapses ResNet-50 accuracy (0%), GPTQ per-OC recovers it (~77%).
# ---------------------------------------------------------------------------
USE_GPTQ = (WEIGHT_BITS < 8) and (os.environ.get("NN2RTL_GPTQ", "1") == "1")


def _spec_int_weight_and_scale(spec: "OnnxLayerSpec"):
    """(int_weight_np [shape of spec.weight], scale) — scale scalar (per-tensor)
    or np[OC] (per-OC GPTQ, when spec.weight_scale_per_oc was set)."""
    poc = getattr(spec, "weight_scale_per_oc", None)
    if poc is not None:
        return getattr(spec, "gptq_qweight"), poc
    return _quantize_weights_int8(spec.weight, spec.weight_scale), spec.weight_scale


def _spec_bias_int(spec: "OnnxLayerSpec") -> np.ndarray:
    poc = getattr(spec, "weight_scale_per_oc", None)
    if poc is not None:                      # per-OC: bias_int[oc]=round(b[oc]/(in*ws[oc]))
        acc = spec.input_scale * np.asarray(poc, dtype=np.float64)
        acc = np.where(acc == 0.0, 1.0, acc)
        return np.round(np.asarray(spec.bias, dtype=np.float64) / acc).astype(np.int32)
    return _quantize_bias_int32(spec.bias, spec.input_scale, spec.weight_scale)


def _composite_conv_scale_per_oc(spec: "OnnxLayerSpec"):
    """np[OC] composite requant scale per output channel, or None if per-tensor."""
    poc = getattr(spec, "weight_scale_per_oc", None)
    if poc is None:
        return None
    acc = spec.input_scale * np.asarray(poc, dtype=np.float64)
    return acc if spec.output_scale == 0.0 else acc / spec.output_scale


def _int8_ref_conv_module(spec):
    """Build a genuine INT8-weight conv module (8-bit, ws=max/127) for realistic
    Hessian-capture activations — naive-INT4 deep-layer activations are garbage
    and ruin GPTQ. INT8-weight network ~75% acc -> good activations."""
    W = np.asarray(spec.weight, dtype=np.float64)
    ws8 = float(np.abs(W).max()) / 127.0 or 1.0
    w8 = np.clip(np.round(W / ws8), -128, 127).astype(np.float32)
    acc8 = spec.input_scale * ws8
    b8 = np.round(np.asarray(spec.bias, dtype=np.float64) / (acc8 or 1.0)).astype(np.float32)
    comp8 = acc8 if spec.output_scale == 0.0 else acc8 / spec.output_scale
    return Int8Conv2d(
        weight=torch.tensor(w8), bias=torch.tensor(b8),
        stride=spec.stride, padding=spec.padding, dilation=spec.dilation,
        groups=spec.groups, scale_factor=float(comp8), rtl_compat=False)


def _gptq_quantize_convs(specs, modules, gptq_inputs) -> None:
    """Set spec.gptq_qweight (INT4 ints) + spec.weight_scale_per_oc (per-OC) on
    every conv2d, via GPTQ using per-conv input Hessians. Hessians are captured
    from an INT8-WEIGHT reference forward (good activations), not the naive-INT4
    modules (whose deep activations are garbage)."""
    import torch.nn.functional as _F
    import gptq_core
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    conv_specs = [s for s in specs if s.op_type == "conv2d" and s.weight is not None]
    H = {s.module_id: None for s in conv_specs}
    cnt = {s.module_id: 0 for s in conv_specs}
    ref = {s.module_id: (_int8_ref_conv_module(s) if (s.op_type == "conv2d" and s.weight is not None)
                         else modules[s.module_id]) for s in specs}
    gmods = {k: m.to(dev) for k, m in ref.items()}
    with torch.no_grad():
        for inp in gptq_inputs:
            tmap = run_int8_network(specs, gmods, inp.to(dev))
            for s in conv_specs:
                x = tmap[s.input_tensor_names[0]].to(dev).float()
                kh, kw = int(s.weight.shape[2]), int(s.weight.shape[3])
                xu = _F.unfold(x, (kh, kw), dilation=tuple(s.dilation),
                               padding=tuple(s.padding), stride=tuple(s.stride))
                xu = xu.transpose(1, 2).reshape(-1, xu.shape[1])
                h = (xu.t() @ xu).cpu()
                H[s.module_id] = h if H[s.module_id] is None else H[s.module_id] + h
                cnt[s.module_id] += xu.shape[0]
    del gmods, ref  # free GPU ref modules
    # [INT3-MIXED 2026-05-30] Per-layer bit-width. Module_ids listed in the
    # NN2RTL_INT3_LAYERS env var (comma-separated) are quantized at INT3
    # (qmax=3, qmin=-4) using the SAME Hessian-aware GPTQ as INT4 (gptq_core,
    # error-compensated — NOT a naive round), so the mixed-precision accuracy
    # matches the measured sweep. Everything else stays INT4 (_WQMAX/_WQMIN).
    # Each conv records s.weight_bits (3|4) for the downstream packers
    # (repack 3-bit, engine banks 3-bit) and the layer_ir export.
    import os as _os
    _int3 = {m for m in _os.environ.get("NN2RTL_INT3_LAYERS", "").split(",") if m}
    n_int3 = 0
    for s in conv_specs:
        is_int3 = s.module_id in _int3
        qmax, qmin = (3, -4) if is_int3 else (_WQMAX, _WQMIN)
        W = torch.tensor(np.asarray(s.weight, dtype=np.float32), device=dev)
        OC = W.shape[0]
        W2d = W.reshape(OC, -1)
        scale = gptq_core.per_oc_scale(W2d, qmax)             # [OC,1]
        Hm = (H[s.module_id] / max(1, cnt[s.module_id]))
        Qint = gptq_core.gptq_int_weights(W2d, Hm, scale, qmin, qmax)
        s.gptq_qweight = Qint.reshape(W.shape).cpu().numpy().astype(np.int8)
        s.weight_scale_per_oc = scale.squeeze(1).cpu().numpy().astype(np.float64)
        s.weight_bits = 3 if is_int3 else 4
        n_int3 += int(is_int3)
    print(f"[gptq] quantized {len(conv_specs)} conv layers per-output-channel "
          f"({n_int3} INT3, {len(conv_specs) - n_int3} INT4)")


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
        w_int8, _wsc = _spec_int_weight_and_scale(spec)
        w_tensor = torch.tensor(w_int8.astype(np.float32))
        b_int32 = _spec_bias_int(spec)
        b_tensor = torch.tensor(b_int32.astype(np.float32))
        _per_oc = _composite_conv_scale_per_oc(spec)
        if _per_oc is not None:
            return Int8Conv2d(
                weight=w_tensor, bias=b_tensor,
                stride=spec.stride, padding=spec.padding,
                dilation=spec.dilation, groups=spec.groups,
                scale_factor=float(_per_oc.mean()),  # metadata only; per-OC used
                rtl_compat=rtl_compat_conv,
                scale_factor_per_oc=[float(v) for v in _per_oc.tolist()],
            )
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
        return Int8ReLU(
            input_scale=spec.input_scale,
            output_scale=spec.output_scale,
        )
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
    if spec.op_type == "global_avg_pool":
        # GAP carries no weights; the composite scale is built inside the
        # module from input_scale / output_scale / (H*W).
        return Int8GlobalAveragePool(
            input_scale=spec.input_scale,
            output_scale=spec.output_scale,
        )
    if spec.op_type == "gemm":
        assert spec.weight is not None
        w_int8 = _quantize_weights_int8(spec.weight, spec.weight_scale)
        w_tensor = torch.tensor(w_int8.astype(np.int64), dtype=torch.int32)
        if spec.bias is not None:
            b_int32 = _quantize_bias_int32(spec.bias, spec.input_scale, spec.weight_scale)
            b_tensor = torch.tensor(b_int32, dtype=torch.int32)
        else:
            b_tensor = None
        acc_scale = spec.input_scale * spec.weight_scale
        composite_scale = acc_scale if spec.output_scale == 0.0 else acc_scale / spec.output_scale
        return Int8Gemm(
            weight_int8=w_tensor,
            bias_int32=b_tensor,
            scale_factor=float(composite_scale),
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
    w_int8, _wsc = _spec_int_weight_and_scale(spec)   # GPTQ INT4 ints when per-OC
    b_int32 = _spec_bias_int(spec)

    weight_values = [int(v) for v in w_int8.reshape(-1).tolist()]
    bias_values = [int(v) for v in b_int32.reshape(-1).tolist()]

    weights_path, bias_path = get_weight_artifact_paths(repo_root, spec.module_id)
    write_signed_int8_hex(weight_values, weights_path)
    write_signed_int32_hex(bias_values, bias_path)
    return weight_values, bias_values


def _write_empty_weight_file(repo_root: Path, module_id: str) -> None:
    """Write an empty placeholder hex file for ops without weights (relu, maxpool, add, global_avg_pool)."""
    weights_path, _ = get_weight_artifact_paths(repo_root, module_id)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_text("", encoding="utf8")


def _write_gemm_hex_artifacts(
    spec: OnnxLayerSpec,
    repo_root: Path,
) -> tuple[list[int], list[int]]:
    """Write weight ([M, K] row-major) and bias ([M]) hex files for a Gemm spec.

    Layout: weights are emitted row-major (output feature is the outer index),
    matching how the RTL is expected to iterate — one output feature at a time,
    K weights sequential per output.
    """
    assert spec.weight is not None
    w_int8 = _quantize_weights_int8(spec.weight, spec.weight_scale)
    weight_values = [int(v) for v in w_int8.reshape(-1).tolist()]
    if spec.bias is not None:
        b_int32 = _quantize_bias_int32(spec.bias, spec.input_scale, spec.weight_scale)
        bias_values = [int(v) for v in b_int32.reshape(-1).tolist()]
    else:
        bias_values = []
    weights_path, bias_path = get_weight_artifact_paths(repo_root, spec.module_id)
    write_signed_int8_hex(weight_values, weights_path)
    if bias_values:
        write_signed_int32_hex(bias_values, bias_path)
    else:
        bias_path.parent.mkdir(parents=True, exist_ok=True)
        bias_path.write_text("", encoding="utf8")
    return weight_values, bias_values


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
    if spec.op_type == "gemm":
        tensors = [_gemm_input_as_4d(t, spec) for t in tensors]
    else:
        tensors = [_as_4d_tensor(t) for t in tensors]
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
    tensors = [_as_4d_tensor(t) for t in tensors]
    return pack_tensor_vectors_to_bus_words(tensors, out_bits, context=f"{spec.module_id}.goldout")


def _as_4d_tensor(t: torch.Tensor) -> torch.Tensor:
    """Pack helpers expect [1, C, H, W]. For 2D activations ([N, K] — e.g. the
    output of a Gemm or the post-Flatten input) we surface [1, K, 1, 1] so
    each channel becomes a single beat on the bus. For 4D inputs the tensor
    passes through unchanged."""
    if t.dim() == 4:
        return t
    if t.dim() == 2:
        n, k = t.shape
        return t.view(n, k, 1, 1)
    if t.dim() == 3:
        # [N, C, L] — fold L into width
        n, c, l = t.shape
        return t.view(n, c, 1, l)
    raise GoldenGenerationError(
        f"Cannot pack tensor of rank {t.dim()} (shape {tuple(t.shape)}) for goldin/goldout."
    )


def _gemm_input_as_4d(t: torch.Tensor, spec: OnnxLayerSpec) -> torch.Tensor:
    """Pack a Gemm input as one flat [N, K, 1, 1] bus beat.

    The tensor may still be the aliased upstream producer output, e.g.
    Conv->[N,C,H,W] before a folded Flatten. Gemm's contract consumes the
    flattened K vector, not spatial samples of C channels.
    """
    if t.dim() < 2:
        raise GoldenGenerationError(
            f"Gemm layer '{spec.module_id}' input must have batch plus feature dims, got shape {tuple(t.shape)}."
        )
    batch = int(t.shape[0])
    k_observed = int(torch.tensor(t.shape[1:]).prod().item())
    k_expected = int(spec.gemm_in_features)
    if k_observed != k_expected:
        raise GoldenGenerationError(
            f"Gemm layer '{spec.module_id}' input shape {tuple(t.shape)} flattens to K={k_observed}, "
            f"but gemm_in_features={k_expected}."
        )
    return t.reshape(batch, k_expected, 1, 1)


def _shape_as_nchw(shape: Sequence[int]) -> list[int]:
    """Coerce a possibly-2D shape ([N, K]) into [N, K, 1, 1] so the channel-bus
    helpers — which assume [N, C, H, W] — can still compute the right width.

    Gemm inputs/outputs are 2D in ONNX; we surface them as single-beat
    channel-bus rows downstream."""
    s = list(shape)
    if len(s) == 2:
        return [s[0], s[1], 1, 1]
    return s


def _spec_input_width_bits(spec: OnnxLayerSpec) -> int:
    if spec.op_type == "add":
        # Packed wide bus: [lhs_channels*8 + rhs_channels*8]
        return channel_bus_bits_from_shape(_shape_as_nchw(spec.input_shape), context=f"{spec.module_id}.input_shape") * 2
    return channel_bus_bits_from_shape(_shape_as_nchw(spec.input_shape), context=f"{spec.module_id}.input_shape")


def _spec_output_width_bits(spec: OnnxLayerSpec) -> int:
    return channel_bus_bits_from_shape(_shape_as_nchw(spec.output_shape), context=f"{spec.module_id}.output_shape")


# ---------------------------------------------------------------------------
# Pipeline latency
# ---------------------------------------------------------------------------

# [THROUGHPUT A2 2026-06-03] Per-module MP override for the 3x3 depthwises (byte-exact:
# depthwise is per-channel, MP only sets lane parallelism; reg widths derived from MP).
# The on-disk node_conv_*.v were edited to these MP values + e2e-verified byte-exact
# (mismatch_bytes=0, 6,823,395 cyc). 15 DW at MP=16; the two 2-BEAT modules (884, 908)
# stay at MP=4 -- their lo/hi beat emission de-syncs the consumer at MP>4 (deadlock), so
# they are GATED on a beat-timing/skid fix (Phase B). This keeps compute_conv2d_latency_cycles
# consistent with the live RTL for a future regen. See [[project-mbv2-throughput-corrected]].
_A2_MP_OVERRIDE = {
    "node_conv_812": 16, "node_conv_818": 16, "node_conv_824": 16, "node_conv_830": 16,
    "node_conv_836": 16, "node_conv_842": 16, "node_conv_848": 16, "node_conv_854": 16,
    "node_conv_860": 16, "node_conv_866": 16, "node_conv_872": 16, "node_conv_878": 16,
    "node_conv_890": 16, "node_conv_896": 16, "node_conv_902": 16,
    # STEM conv_810 (cross-channel conv_datapath_mp_k): MP=16 unlocked by a symmetric
    # LHS skid on node_add_198 (top-wrapper patch, NOT regen-able from here; see
    # backups/stem_mp16_lhsskid_6446k_20260603). 6,446,347 cyc byte-exact.
    "node_conv_810": 16,
    # node_conv_884, node_conv_908 (2-beat): MP=4 (default) -- separate beat-timing issue.
}


def _conv_mac_parallelism(spec: OnnxLayerSpec) -> int:
    """Accumulator-group size for an ONNX conv layer: min(OC, MAX_PARALLEL_MACS)."""
    if spec.weight is None or len(spec.weight.shape) < 1:
        return 1
    ov = _A2_MP_OVERRIDE.get(getattr(spec, "module_id", None))
    if ov is not None:
        return ov
    return conv_mac_parallelism(int(spec.weight.shape[0]))


def _conv_mp_k(spec: OnnxLayerSpec) -> int:
    """A2: the 3x3 spatial path (MBv2 stem + 17 depthwise) uses the MP_K=9
    tap-parallel datapath; everything else stays tap-serial (mp_k=1)."""
    if spec.weight is not None and len(spec.weight.shape) >= 4:
        kh, kw = int(spec.weight.shape[2]), int(spec.weight.shape[3])
        if kh == 3 and kw == 3:
            return 9
    return 1


def _pipeline_latency(spec: OnnxLayerSpec) -> int:
    if spec.op_type == "conv2d":
        weight_shape = list(spec.weight.shape) if spec.weight is not None else [1, 1, 1, 1]
        return compute_conv2d_latency_cycles(
            weight_shape,
            input_shape=spec.input_shape,
            stride=spec.stride,
            padding=spec.padding,
            mac_parallelism=_conv_mac_parallelism(spec),
            mp_k=_conv_mp_k(spec),
        )
    if spec.op_type == "maxpool":
        return compute_maxpool_latency_cycles(spec)
    if spec.op_type == "add":
        return compute_add_latency_cycles(spec.output_shape)
    if spec.op_type == "global_avg_pool":
        # Per-channel reduction over H*W cells, one cell per cycle, plus a
        # 3-stage requantize tail (BIAS pass-through + SCALE + CLAMP/OUTPUT).
        h, w = spec.gap_spatial[0], spec.gap_spatial[1]
        return max(1, int(h) * int(w)) + 3
    if spec.op_type == "gemm":
        # K-deep dot product per output, serialized across mac_parallelism
        # lanes; M outputs emitted one per cycle in the requantize tail.
        k = max(1, int(spec.gemm_in_features))
        m = max(1, int(spec.gemm_out_features))
        mp = max(1, _conv_mac_parallelism(spec))
        mac_cycles_per_output = -(-k // mp)  # ceil(k / mp)
        return mac_cycles_per_output + m + 3
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

    # --- Step 4: Build calibration inputs -----------------------------------
    # NN2RTL_IMAGENET_CALIB=N feeds N real preprocessed ImageNet images (float,
    # resize256+crop224+normalize) instead of synthetic INT8-range noise. With
    # real floats, calibrate_onnx records true activation ranges -> proper scales
    # (input_scale becomes ~range/127, not ~1.0). The golden input is then
    # int8-quantized by input_scale at Step 7 (see use_real_input_scale).
    imagenet_calib = int(os.environ.get("NN2RTL_IMAGENET_CALIB", "0"))
    use_real_input_scale = imagenet_calib > 0
    if imagenet_calib > 0:
        import sys as _sys
        _sd = str((repo_root / "scripts").resolve())
        if _sd not in _sys.path:
            _sys.path.insert(0, _sd)
        import imagenet_util as _iu
        _imgs, _labels = _iu.load_batch(imagenet_calib)  # (N,3,224,224) float32
        num_calibration_samples = imagenet_calib
        cal_feeds = [
            {network_input_name: _imgs[i:i + 1].astype(np.float32)}
            for i in range(imagenet_calib)
        ]
        warnings.warn(
            f"ImageNet calibration: {imagenet_calib} real images "
            f"(WEIGHT_BITS={WEIGHT_BITS}).", RuntimeWarning, stacklevel=2)
    else:
        rng = np.random.default_rng(calibration_seed)
        cal_feeds = [
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

    # input scale for golden quantization (needed early for GPTQ calib inputs)
    _in_scale = float(specs[0].input_scale) if use_real_input_scale else 1.0
    if not _in_scale:
        _in_scale = 1.0

    def _golden_input(i):
        return quantize_tensor_to_int8_range(
            torch.tensor(cal_feeds[i][network_input_name]).reshape(first_input_shape)
            / _in_scale)

    # --- Step 6: Build INT8 modules -----------------------------------------
    modules: dict[str, nn.Module] = {
        s.module_id: _build_int8_module(s, rtl_compat_conv=rtl_compat_conv) for s in specs
    }

    # --- Step 6b: GPTQ INT4 (per-output-channel) -----------------------------
    # Naive INT4 collapses accuracy; GPTQ per-OC recovers it. Quantize convs to
    # INT4 using calibration Hessians, then rebuild the modules per-OC.
    if USE_GPTQ:
        n_gptq = min(num_calibration_samples,
                     int(os.environ.get("NN2RTL_GPTQ_CALIB", "128")))
        gptq_inputs = [_golden_input(i) for i in range(n_gptq)]
        _gptq_quantize_convs(specs, modules, gptq_inputs)
        modules = {
            s.module_id: _build_int8_module(s, rtl_compat_conv=rtl_compat_conv) for s in specs
        }

    # --- Step 7: Run INT8 simulation to capture golden activations ----------
    torch.manual_seed(calibration_seed)
    # Calibration uses ALL N feeds (for scale stats), but the GOLDENS only need a
    # few vectors for RTL verification. NN2RTL_GOLDEN_VECTORS caps the golden
    # forward so 256-image calibration doesn't emit 256-vector (GB-sized) goldens.
    _gv_env = int(os.environ.get("NN2RTL_GOLDEN_VECTORS", "0"))
    golden_vectors = (_gv_env if _gv_env > 0
                      else (2 if use_real_input_scale else num_calibration_samples))
    golden_vectors = max(1, min(golden_vectors, num_calibration_samples))
    input_tensors = [_golden_input(i) for i in range(golden_vectors)]

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
        elif spec.op_type == "gemm":
            weight_values, bias_values = _write_gemm_hex_artifacts(spec, repo_root)
            weight_shape = list(spec.weight.shape)  # [M, K]
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
            "bias_path": bias_path_obj.resolve().as_posix() if spec.op_type in ("conv2d", "gemm") and spec.bias is not None else None,
            "weight_shape": weight_shape,
            "num_weights": num_weights,
            # Per-op scale semantics:
            # * conv2d / gemm: composite (input_scale * weight_scale / output_scale)
            #   that the RTL multiplies (acc + bias_int32) by to requantize to INT8.
            # * global_avg_pool: composite (input_scale / output_scale / (H*W))
            #   that folds the spatial divisor into SCALE_MULT / SCALE_SHIFT.
            # * relu with clip_max (ReLU6): composite (input_scale / output_scale)
            #   to rescale the upstream Conv's INT8 stream into the post-clip scale.
            #   For plain relu (no clip_max) we keep emitting output_scale to
            #   preserve the historical LayerIR shape used by ResNet-50.
            # * add / maxpool / plain relu: output activation scale, used by the
            #   quantized-add formula and as a metadata anchor for other ops.
            "scale_factor": _layer_ir_scale_factor(spec),
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

        if spec.op_type == "relu" and spec.clip_max is not None:
            layer_payload["clip_max"] = float(spec.clip_max)

        if spec.op_type in ("conv2d", "gemm"):
            # Scheme A': weights WEIGHT_BITS-wide (4=INT4), activations INT8.
            # [INT3-MIXED] per-layer override: convs quantized at INT3 (set in
            # _gptq_quantize_convs via NN2RTL_INT3_LAYERS) carry weight_bits=3 so
            # the spatial repack / engine bank packers know to use 3-bit fields.
            layer_payload["weight_bits"] = int(getattr(spec, "weight_bits", WEIGHT_BITS))
            layer_payload["activation_bits"] = 8
            _per_oc = _composite_conv_scale_per_oc(spec) if spec.op_type == "conv2d" else None
            if _per_oc is not None:
                # Per-output-channel requant: one composite scale per OC. The RTL
                # builds a per-lane SCALE_MULT/SHIFT ROM from these.
                layer_payload["scale_factor_per_oc"] = [float(v) for v in _per_oc.tolist()]
                layer_payload["weight_scale_per_oc"] = [
                    float(v) for v in np.asarray(spec.weight_scale_per_oc).tolist()]

        if spec.op_type == "conv2d":
            layer_payload["stride"] = list(spec.stride)
            layer_payload["padding"] = list(spec.padding)
            layer_payload["dilation"] = list(spec.dilation)
            layer_payload["groups"] = int(spec.groups)
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

        if spec.op_type == "global_avg_pool":
            # H * W is the divisor that the requantize tail folds into
            # SCALE_MULT / SCALE_SHIFT. Foundry reads gap_spatial to size the
            # accumulator and the spatial counter.
            layer_payload["gap_spatial"] = [int(spec.gap_spatial[0]), int(spec.gap_spatial[1])]

        if spec.op_type == "gemm":
            # M, K let Foundry size the weight memory and the K-deep MAC loop
            # without re-parsing weight_shape.
            layer_payload["gemm_in_features"] = int(spec.gemm_in_features)
            layer_payload["gemm_out_features"] = int(spec.gemm_out_features)

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
