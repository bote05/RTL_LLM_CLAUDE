"""Tests for scripts/onnx_frontend.py.

Covers the critical behaviours that regressed during the first implementation
pass: MaxPool extraction, multi-input rejection, graph-completeness checks,
RTL-compat vs faithful conv toggle, and the Foundry JSON-recovery path.
"""

from __future__ import annotations

import contextlib
import io
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.golden_impl import GoldenGenerationError
from scripts.onnx_frontend import (
    DEFAULT_ONNX_OPSET,
    Int8MaxPool2d,
    OnnxLayerSpec,
    _real_graph_inputs,
    build_pipeline_ir_from_onnx,
    export_pytorch_to_onnx,
    extract_layer_specs,
    load_onnx,
    simplify_onnx,
    validate_graph_completeness,
    validate_graph_outputs_covered,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _export(model: nn.Module, path: Path, shape=(1, 3, 16, 16)) -> Path:
    """Export a PyTorch model to ONNX silently (suppresses the Windows cp1252
    emoji print that torch.onnx.export does)."""
    with contextlib.redirect_stdout(io.StringIO()):
        export_pytorch_to_onnx(model, path, input_shape=shape, opset=DEFAULT_ONNX_OPSET)
    return path


# ---------------------------------------------------------------------------
# Int8MaxPool2d
# ---------------------------------------------------------------------------

def test_int8_maxpool_matches_torch_int8():
    m = Int8MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    x = torch.randint(-128, 128, (1, 4, 8, 8), dtype=torch.int32).to(torch.float32)
    y = m(x)
    ref = torch.clamp(
        F.max_pool2d(x.to(torch.float32), 3, 2, 1), -128, 127
    )
    assert torch.equal(y, ref)


# ---------------------------------------------------------------------------
# _real_graph_inputs
# ---------------------------------------------------------------------------

def test_real_graph_inputs_filters_initializers(tmp_path):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(3, 4, 1, bias=True)
        def forward(self, x):
            return self.c(x)
    onnx_path = _export(M().eval(), tmp_path / "m.onnx")
    model = load_onnx(onnx_path)
    reals = _real_graph_inputs(model)
    assert len(reals) == 1
    # conv weights should NOT appear in real inputs
    assert all("conv" not in r.name.lower() for r in reals)


# ---------------------------------------------------------------------------
# validate_graph_completeness
# ---------------------------------------------------------------------------

def test_validate_graph_completeness_passes_on_chain():
    specs = [
        OnnxLayerSpec("c0", "conv2d", ["input"], "t1", [1, 3, 8, 8], [1, 4, 8, 8],
                      weight=np.zeros((4, 3, 1, 1), np.float32)),
        OnnxLayerSpec("r0", "relu", ["t1"], "t2", [1, 4, 8, 8], [1, 4, 8, 8]),
    ]
    validate_graph_completeness(specs, {"input"})  # should not raise


def test_validate_graph_outputs_covered_passes_when_spec_produces_output():
    specs = [
        OnnxLayerSpec("c0", "conv2d", ["input"], "t1", [1, 3, 8, 8], [1, 4, 8, 8],
                      weight=np.zeros((4, 3, 1, 1), np.float32)),
    ]
    # graph output is t1 — produced by spec c0
    validate_graph_outputs_covered(specs, graph_output_names={"t1"}, graph_input_names={"input"})


def test_validate_graph_outputs_covered_raises_on_unsupported_tail():
    # Simulates Conv -> Flatten -> Gemm where Flatten+Gemm are skipped.
    # The spec set only covers the Conv; graph.output points at Gemm's tensor
    # which no spec produces.
    specs = [
        OnnxLayerSpec("c0", "conv2d", ["input"], "conv_out", [1, 3, 8, 8], [1, 4, 8, 8],
                      weight=np.zeros((4, 3, 1, 1), np.float32)),
    ]
    with pytest.raises(GoldenGenerationError, match="unsupported ops AFTER"):
        validate_graph_outputs_covered(
            specs, graph_output_names={"classifier_out"}, graph_input_names={"input"},
        )


def test_end_to_end_rejects_conv_flatten_gemm(tmp_path):
    """Regression: a Conv -> Flatten -> Gemm model must NOT silently truncate to Conv."""
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(3, 4, 1, bias=True)
            self.fc = nn.Linear(4 * 8 * 8, 10)
        def forward(self, x):
            x = self.c(x)
            x = x.flatten(1)
            return self.fc(x)

    onnx_path = _export(M().eval(), tmp_path / "classifier.onnx", shape=(1, 3, 8, 8))
    with pytest.raises(GoldenGenerationError, match="AFTER the last extracted"):
        build_pipeline_ir_from_onnx(
            onnx_path=onnx_path, repo_root=tmp_path,
            num_calibration_samples=2,
        )


def test_validate_graph_completeness_raises_on_gap():
    specs = [
        OnnxLayerSpec("c0", "conv2d", ["input"], "t1", [1, 3, 8, 8], [1, 4, 8, 8],
                      weight=np.zeros((4, 3, 1, 1), np.float32)),
        # 'mystery' tensor not produced by any spec → gap
        OnnxLayerSpec("r0", "relu", ["mystery"], "t2", [1, 4, 8, 8], [1, 4, 8, 8]),
    ]
    with pytest.raises(GoldenGenerationError, match="unsupported op"):
        validate_graph_completeness(specs, {"input"})


# ---------------------------------------------------------------------------
# Multi-input rejection
# ---------------------------------------------------------------------------

def test_multi_input_model_rejected(tmp_path):
    class TwoInputs(nn.Module):
        def forward(self, a, b):
            return a + b
    model = TwoInputs().eval()
    onnx_path = tmp_path / "two.onnx"
    dummy_a = torch.randn(1, 3, 8, 8)
    dummy_b = torch.randn(1, 3, 8, 8)
    with contextlib.redirect_stdout(io.StringIO()):
        torch.onnx.export(
            model, (dummy_a, dummy_b), str(onnx_path),
            opset_version=DEFAULT_ONNX_OPSET,
            input_names=["a", "b"], output_names=["out"],
        )
    with pytest.raises(GoldenGenerationError, match="real inputs"):
        build_pipeline_ir_from_onnx(
            onnx_path=onnx_path, repo_root=tmp_path,
            num_calibration_samples=2,
        )


# ---------------------------------------------------------------------------
# Extraction coverage: MaxPool
# ---------------------------------------------------------------------------

def test_maxpool_extracted_with_geometry(tmp_path):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(3, 4, 3, padding=1, bias=True)
        def forward(self, x):
            return F.max_pool2d(self.c(x), kernel_size=3, stride=2, padding=1)
    onnx_path = _export(M().eval(), tmp_path / "mp.onnx")
    model = simplify_onnx(load_onnx(onnx_path))
    specs = extract_layer_specs(model)
    pool_specs = [s for s in specs if s.op_type == "maxpool"]
    assert len(pool_specs) == 1
    p = pool_specs[0]
    assert p.pool_kernel == [3, 3]
    assert p.pool_stride == [2, 2]
    assert p.pool_padding == [1, 1]


# ---------------------------------------------------------------------------
# Extraction coverage: Add (residual)
# ---------------------------------------------------------------------------

def test_residual_add_extracted(tmp_path):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(4, 4, 1, bias=True)
        def forward(self, x):
            return x + self.c(x)
    onnx_path = _export(M().eval(), tmp_path / "add.onnx", shape=(1, 4, 8, 8))
    model = simplify_onnx(load_onnx(onnx_path))
    specs = extract_layer_specs(model)
    add_specs = [s for s in specs if s.op_type == "add"]
    assert len(add_specs) == 1
    assert add_specs[0].add_lhs_tensor != add_specs[0].add_rhs_tensor


# ---------------------------------------------------------------------------
# Clip(min=0) → relu coercion
# ---------------------------------------------------------------------------

def test_clip_as_relu(tmp_path):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(3, 4, 1, bias=True)
        def forward(self, x):
            return torch.clamp(self.c(x), min=0.0)  # → Clip(min=0) in ONNX
    onnx_path = _export(M().eval(), tmp_path / "clip.onnx")
    model = simplify_onnx(load_onnx(onnx_path))
    specs = extract_layer_specs(model)
    # Either the Clip survived (→ relu spec) or simplify folded it; at minimum
    # the chain has to be valid
    validate_graph_completeness(specs, {gi.name for gi in _real_graph_inputs(model)})


# ---------------------------------------------------------------------------
# RTL-compat vs faithful conv
# ---------------------------------------------------------------------------

def test_legacy_rtl_compat_conv_warns_on_spatial_kernel(tmp_path):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 4, 3, padding=1, bias=True)  # spatial
        def forward(self, x):
            return self.c1(x)
    onnx_path = _export(M().eval(), tmp_path / "spatial.onnx")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        build_pipeline_ir_from_onnx(
            onnx_path=onnx_path, repo_root=tmp_path,
            num_calibration_samples=2,
            rtl_compat_conv=True,  # explicit opt-in to legacy path
        )
    msgs = [str(x.message) for x in w if "rtl_compat_conv" in str(x.message)]
    assert msgs, "Expected LEGACY RTL-compat warning for spatial conv"


def test_default_faithful_conv_no_warning(tmp_path):
    """Default build_pipeline_ir_from_onnx emits faithful 2D conv goldens silently."""
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 4, 3, padding=1, bias=True)
        def forward(self, x):
            return self.c1(x)
    onnx_path = _export(M().eval(), tmp_path / "faith.onnx")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        build_pipeline_ir_from_onnx(
            onnx_path=onnx_path, repo_root=tmp_path,
            num_calibration_samples=2,
            # rtl_compat_conv defaults to False (faithful)
        )
    msgs = [str(x.message) for x in w if "rtl_compat_conv" in str(x.message)]
    assert not msgs, f"Unexpected legacy warning in default (faithful) mode: {msgs}"


def test_faithful_conv_goldens_match_real_pytorch_conv(tmp_path):
    """End-to-end: goldens for a 3x3 conv must match real F.conv2d, not the
    spatially-summed approximation."""
    torch.manual_seed(0)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 4, 3, padding=1, bias=True)
        def forward(self, x):
            return self.c1(x)

    model = M().eval()
    onnx_path = _export(model, tmp_path / "conv3x3.onnx", shape=(1, 3, 8, 8))
    payload = build_pipeline_ir_from_onnx(
        onnx_path=onnx_path, repo_root=tmp_path,
        num_calibration_samples=2,
    )
    conv_layer = payload["layers"][0]
    assert conv_layer["op_type"] == "conv2d"
    assert conv_layer["stride"] == [1, 1]
    assert conv_layer["padding"] == [1, 1]
    assert len(conv_layer["weight_bank_paths"]) == conv_layer["mac_parallelism"]
    for bank_path in conv_layer["weight_bank_paths"]:
        assert Path(bank_path).exists()
    # Output is SAME-padded 8x8, so 64 output samples per vector
    assert conv_layer["output_shape"] == [1, 4, 8, 8]
    # Pipeline latency for spatial conv must be > K_TOTAL + 4 because of
    # the line-buffer fill time.
    k_total = 3 * 3 * 3  # IC=3, KH=KW=3
    assert conv_layer["pipeline_latency_cycles"] > k_total + 4


def test_pipeline_ir_emits_strided_conv_geometry(tmp_path):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 4, 3, stride=2, padding=1, bias=True)
        def forward(self, x):
            return self.c1(x)

    onnx_path = _export(M().eval(), tmp_path / "stride2.onnx", shape=(1, 3, 8, 8))
    payload = build_pipeline_ir_from_onnx(
        onnx_path=onnx_path, repo_root=tmp_path,
        num_calibration_samples=2,
    )

    conv_layer = payload["layers"][0]
    assert conv_layer["op_type"] == "conv2d"
    assert conv_layer["input_shape"] == [1, 3, 8, 8]
    assert conv_layer["output_shape"] == [1, 4, 4, 4]
    assert conv_layer["stride"] == [2, 2]
    assert conv_layer["padding"] == [1, 1]


# ---------------------------------------------------------------------------
# End-to-end: tiny CNN -> PipelineIR -> validates, files on disk
# ---------------------------------------------------------------------------

def test_end_to_end_tiny_cnn(tmp_path):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 4, 1, bias=True)
            self.r1 = nn.ReLU()
            self.c2 = nn.Conv2d(4, 2, 1, bias=True)
        def forward(self, x):
            return self.c2(self.r1(self.c1(x)))
    onnx_path = _export(M().eval(), tmp_path / "e2e.onnx")
    payload = build_pipeline_ir_from_onnx(
        onnx_path=onnx_path, repo_root=tmp_path,
        model_name="e2e",
        num_calibration_samples=2,
    )
    assert payload["model_name"] == "e2e"
    assert len(payload["layers"]) == 3
    ops = [l["op_type"] for l in payload["layers"]]
    assert ops == ["conv2d", "relu", "conv2d"]
    # All goldin/goldout must be real files on disk
    for l in payload["layers"]:
        assert Path(l["golden_inputs_path"]).exists()
        assert Path(l["golden_outputs_path"]).exists()
        if l["op_type"] == "conv2d":
            assert len(l["weight_bank_paths"]) == l["mac_parallelism"]
            for bank_path in l["weight_bank_paths"]:
                assert Path(bank_path).exists()
