"""Importable helpers for generate_golden.py."""

from __future__ import annotations

import array
import json
import os
import operator
import struct
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import fx, nn

from scripts.quantize_impl import create_toy_model, load_quantized_checkpoint, run_toy_model


LAYER_IR_FILE_NAME = "layer_ir.json"
LEGACY_GOLDEN_FILE_NAME = "golden_vectors.json"
SIGNAL_LITERALS = {
    "clock_signal": "clk",
    "reset_signal": "rst_n",
    "valid_in_signal": "valid_in",
    "valid_out_signal": "valid_out",
    "ready_in_signal": "ready_in",
    "data_in_signal": "data_in",
    "data_out_signal": "data_out",
}
# Add latency is shape-dependent. The resource-bounded add template captures
# the packed pixel, then streams one channel per cycle through a 3-stage
# multiply/sum/saturate pipe. This avoids the old 512-parallel-multiplier
# residual add that consumed all 240 Artix-7 DSPs and spilled the rest into
# LUTs at OC=256.
PIPELINE_LATENCY_CYCLES = {
    "conv2d": 2,
    "relu": 1,
    "add": 1,
    "maxpool": 1,
    # Per-channel reduction over H*W feature-map cells. The pipeline can issue
    # one accumulator update per cycle until all spatial cells are consumed,
    # then divide by H*W and emit. Foundry computes the exact figure from
    # spatial / mac_parallelism geometry; this default is the floor.
    "global_avg_pool": 2,
    # K-way dot product per output feature. Floor of 2 stages matches the
    # bias/scale tail of conv; the real value scales with `gemm_in_features /
    # mac_parallelism` and is filled by the latency model.
    "gemm": 2,
}
SUPPORTED_OP_TYPES = frozenset(PIPELINE_LATENCY_CYCLES)

# Number of register stages wrapped around the output-stationary conv
# datapath. Foundry allocates four distinct registered stages after the input
# latch and MAC loop:
#   BIAS   — acc[oc] + bias[oc]          (32-bit add only)
#   SCALE  — biased[oc] * SCALE_MULT     (32-bit × 16-bit multiply only)
#   OUTPUT — >>> SCALE_SHIFT, saturate, pack to data_out
# Separating BIAS from SCALE halves the critical-path logic depth compared to
# combining them, which is the primary lever for Vivado timing on Artix-7.
CONV_PIPELINE_STAGES = 6

# Cap on the number of MAC accumulator lanes Foundry
# instantiates per conv layer. Must stay in sync with PIPELINE_CONFIG.
# MAX_PARALLEL_MACS in sdk/config.ts.
#
# Note: with the serialized-weight-read architecture (see foundry.md), MP
# is the ACCUMULATOR count, not a parallel-reads count. Each ST_RUNNING
# cycle does ONE weight read / ONE multiply / ONE accumulate; lane_counter
# rotates through MP accumulators. MP therefore controls:
#   - OC_PASSES = ceil(OC / MP) — how many groups per output pixel
#   - MAC cycles per pass = MP * K_TOTAL
#   - per-pixel latency ≈ MP * K_TOTAL * OC_PASSES ≈ K_TOTAL * OC (independent of MP!)
#
# Practical effect of MP: it trades accumulator storage (acc/biased/scaled
# arrays sized [0:MP-1]) against OC_PASSES count. Smaller MP → fewer
# accumulator registers but more passes. Larger MP → more regs, fewer
# passes, but wider groups need corresponding BRAM banking and routing budget.
# 4 is the current conservative point for the existing serialized datapath.
MAX_PARALLEL_MACS = 4


def compute_add_latency_cycles(output_shape: Sequence[int]) -> int:
    if len(output_shape) < 2:
        raise GoldenGenerationError(
            f"Add output_shape must include a channel dimension, got {list(output_shape)}."
        )
    output_channels = int(output_shape[1])
    if output_channels <= 0:
        raise GoldenGenerationError(
            f"Add output channel count must be positive, got {output_channels}."
        )
    return output_channels + 3


def conv_mac_parallelism(output_channels: int) -> int:
    """Return the mac_parallelism value a conv layer should be generated with.

    The rule is ``min(OC, MAX_PARALLEL_MACS)``. For very small-OC layers
    (OC < MAX_PARALLEL_MACS) we generate OC lanes, which keeps the existing
    trivial synthesis path for pointwise 1×1 with few channels. Above the cap
    we group output channels into ``ceil(OC / MAX_PARALLEL_MACS)`` passes.
    In the current serialized-read datapath, lane_counter still issues only
    one weight read / multiply per cycle; MP is an accumulator-group size,
    not a promise of MP cycle-parallel BRAM reads.
    """
    if output_channels <= 0:
        return 1
    return min(int(output_channels), MAX_PARALLEL_MACS)


def compute_conv2d_latency_cycles(
    weight_shape: list[int],
    input_shape: list[int] | None = None,
    stride: Sequence[int] | None = None,
    padding: Sequence[int] | None = None,
    mac_parallelism: int | None = None,
) -> int:
    """Return the cycle count from first valid_in to first valid_out.

    Two latency shapes depending on the kernel size:

    * **Pointwise (KH = KW = 1)** — the RTL is a single-pixel output-
      stationary MAC group. The current verified contract serializes the
      MP accumulator lanes through a 3-stage MAC pipeline (synchronous
      weight ROM read → registered DSP multiply → indexed accumulator
      add). One OC pass issues ``MP * IC * 1 * 1`` MAC cycles, then 2
      trailing drain cycles for stages 2 and 3, then BIAS / SCALE / OUTPUT.
      Formula: ``1 + OC_PASSES * (MP * IC * KH * KW + CONV_PIPELINE_STAGES)``
      with ``CONV_PIPELINE_STAGES = 6``.

    * **Spatial (KH*KW > 1)** — the RTL uses a line buffer + sliding-
      window datapath (see ``nn2rtl-plugin/agents/foundry.md § Spatial
      conv datapath``). The first valid_out only fires once the first
      complete receptive-field window has been streamed in, plus the MAC
      and the post-MAC pipeline:
      ``max(KH - 1 - PH, 0) * (IW + PW) + max(KW - PW, 1) +
       OC_PASSES * (MP*K_TOTAL + CONV_PIPELINE_STAGES)``
      where ``IW = input_shape[3]`` and ``PH, PW`` come from the layer's
      padding. If the frontend passed ``input_shape`` and ``padding``, we
      use that formula; otherwise we fall back to the pointwise shape
      with a conservative warning (the caller just gets the pointwise-
      shaped ``OC_PASSES * (MP*K_TOTAL + CONV_PIPELINE_STAGES)`` term,
      without window fill). That will mismatch actual spatial RTL —
      always pass the full layer geometry.
    """
    if len(weight_shape) < 4:
        return PIPELINE_LATENCY_CYCLES["conv2d"]
    oc, ic, kh, kw = weight_shape[:4]
    oc_i, ic_i, kh_i, kw_i = int(oc), int(ic), int(kh), int(kw)
    k_total = ic_i * kh_i * kw_i

    # OC-group iteration with serialized accumulator lanes and a registered
    # multiplier output (DSP48E1 MREG=1, Vivado-inferred):
    # Per OC pass, ST_RUNNING runs MP*K_TOTAL issue cycles. Each issue
    # selects one lane with lane_counter and reads one flat weight; the
    # MAC pipeline is then 3 stages deep (weight_q ROM read → mul_q
    # registered product → acc accumulate). After the last issue, TWO
    # trailing-consume cycles drain stages 2 and 3, then ST_BIAS (1) +
    # ST_SCALE (1) + ST_OUTPUT (1) wrap up the pass — giving
    # MP*K_TOTAL + 6 cycles per pass (CONV_PIPELINE_STAGES = 6). The
    # LAST pass's ST_OUTPUT asserts valid_out; the TB sampling offset is
    # absorbed into the formula.
    mp = int(mac_parallelism) if mac_parallelism and mac_parallelism > 0 else oc_i
    mp = min(mp, oc_i) if oc_i > 0 else mp
    oc_passes = (oc_i + mp - 1) // mp if mp > 0 and oc_i > 0 else 1
    pass_cycles = mp * k_total + CONV_PIPELINE_STAGES

    if kh_i * kw_i <= 1:
        # Pointwise — no window fill. First input triggers output_fires
        # immediately → ST_RUNNING starts one cycle later.
        # first MAC at cycle 1, last OUTPUT at OC_PASSES * pass_cycles,
        # valid_out observed at OC_PASSES * pass_cycles + 1.
        return 1 + oc_passes * pass_cycles

    # Spatial — line-buffer datapath. ST_STREAM wraps in_col at IW-1+PW
    # (handles right-edge padding outputs inline). Each fill row therefore
    # takes IW+PW cycles, not IW.
    iw = int(input_shape[3]) if input_shape and len(input_shape) >= 4 else 0
    ph = int(padding[0]) if padding and len(padding) >= 1 else 0
    pw = int(padding[1]) if padding and len(padding) >= 2 else 0

    if iw <= 0:
        # No geometry supplied — fall back to pointwise shape.
        return 1 + oc_passes * pass_cycles

    fill_rows = max(kh_i - 1 - ph, 0)
    fill_cols = max(kw_i - pw, 1)
    # Spatial path has an extra +1 over the pointwise per-pass count: the
    # coord_scheduler registers `output_fires` as a one-cycle pulse, and
    # `conv_datapath` then takes one more cycle to transition ST_IDLE →
    # ST_MAC on it. The pointwise reference does not go through
    # `output_fires` / ST_IDLE — it transitions ST_STREAM → ST_RUNNING in
    # a single state-register update — so this offset is spatial-only.
    # Verified empirically against `layer1_0_conv2` at +1 (37076 vs 37075).
    return fill_rows * (iw + pw) + fill_cols + oc_passes * pass_cycles + 1


class GoldenGenerationError(ValueError):
    """Raised when golden generation cannot satisfy the repo contract."""


class Int8Conv2d(nn.Module):
    """INT8 Conv2d used to produce golden vectors.

    The class supports two simulation modes:

    * ``rtl_compat=False`` (default as of the real-2D-conv RTL datapath) —
      performs a faithful INT8 2D convolution with the full ``KH × KW``
      receptive field, matching the ONNX model and the line-buffer RTL
      described in ``nn2rtl-plugin/agents/foundry.md § Spatial conv datapath``.

    * ``rtl_compat=True`` (deprecated) — the legacy spatially-summed 1×1
      approximation that matched the old single-pixel-MAC RTL, i.e.
      ``acc[oc] += w[oc,k] * in_latch[k / (KH*KW)]`` which is equivalent to
      using ``w.sum(dim=(2,3))``. Only useful when regenerating goldens
      against RTL that still uses the old datapath. New RTL must match
      ``rtl_compat=False``.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        stride: Sequence[int] = (1, 1),
        padding: Sequence[int] = (0, 0),
        dilation: Sequence[int] = (1, 1),
        groups: int = 1,
        scale_factor: float = 1.0,
        rtl_compat: bool = False,
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight.to(torch.float32))
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.to(torch.float32))
        self.stride = tuple(int(v) for v in stride)
        self.padding = tuple(int(v) for v in padding)
        self.dilation = tuple(int(v) for v in dilation)
        self.groups = int(groups)
        self.scale_factor = float(scale_factor)
        self.rtl_compat = bool(rtl_compat)

    def forward(self, x):
        weight = self.weight
        padding = self.padding
        if self.rtl_compat and weight.shape[2] * weight.shape[3] > 1:
            # RTL-compat: current single-MAC datapath can only implement the
            # spatially-summed 1×1 approximation.  Collapse the kernel so the
            # golden matches the RTL bitwise.
            weight = weight.sum(dim=(2, 3), keepdim=True)
            padding = (0, 0)
        y = F.conv2d(
            x.to(torch.float32),
            weight,
            self.bias,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return requantize_tensor_with_scale(y, self.scale_factor)


class Int8FusedStemConv2d(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        stride: Sequence[int] = (1, 1),
        padding: Sequence[int] = (0, 0),
        dilation: Sequence[int] = (1, 1),
        groups: int = 1,
        scale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight.to(torch.float32))
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.to(torch.float32))
        self.stride = tuple(int(v) for v in stride)
        self.padding = tuple(int(v) for v in padding)
        self.dilation = tuple(int(v) for v in dilation)
        self.groups = int(groups)
        self.scale_factor = float(scale_factor)

    def forward(self, x):
        y = F.conv2d(
            x.to(torch.float32),
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        y = torch.relu(y)
        y = F.max_pool2d(y, kernel_size=3, stride=2, padding=1)
        return requantize_tensor_with_scale(y, self.scale_factor)


class Int8ReLU(nn.Module):
    """INT8 ReLU with optional rescale + upper clip.

    For unbounded ReLU (the historical case, used by every ResNet-50 layer),
    `input_scale == output_scale` and the module is a pure max(0, x) on the
    INT8 stream. For ReLU6 (MobileNet et al.), the previous Conv's output
    was calibrated against the float pre-clip range, but the ReLU6's output
    is calibrated against [0, clip_max]. The two scales differ, so we must
    requantize from input_scale to output_scale here — otherwise the next
    layer interprets the INT8 stream with the wrong float scale.
    """

    def __init__(
        self,
        input_scale: float = 1.0,
        output_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_scale = float(input_scale)
        self.output_scale = float(output_scale)

    def forward(self, x):
        x_relu = torch.relu(x.to(torch.float32))
        if self.output_scale == 0.0:
            scale = 1.0
        else:
            scale = self.input_scale / self.output_scale
        # When input_scale == output_scale (unbounded ReLU), scale == 1 and
        # this is identical to the previous pass-through behavior. When
        # they differ (ReLU6), the multiply requantizes the INT8 stream so
        # downstream layers see values in the post-clip scale. Saturation
        # at INT8 127 is then the in-domain ReLU6 ceiling.
        rescaled = round_half_up_toward_pos_inf(x_relu * scale)
        return torch.clamp(rescaled, -128, 127)


class Int8Add(nn.Module):
    def __init__(
        self,
        lhs_scale_factor: float,
        rhs_scale_factor: float,
        output_scale_factor: float,
    ) -> None:
        super().__init__()
        self.lhs_scale_factor = float(lhs_scale_factor)
        self.rhs_scale_factor = float(rhs_scale_factor)
        self.output_scale_factor = float(output_scale_factor)

    def forward(self, lhs, rhs):
        # lhs/rhs are int8 values; multiplying by their scale factors converts
        # to real-valued domain. To get back to output int8 we divide by the
        # output scale (real → int). This is different from the conv path where
        # the accumulator stays in integer domain and the multiplier IS the
        # scale_factor field.
        summed = (
            lhs.to(torch.float32) * self.lhs_scale_factor
            + rhs.to(torch.float32) * self.rhs_scale_factor
        )
        # Match the RTL requantize stage: round-half-toward-+infinity, not
        # PyTorch's banker's rounding. See round_half_up_toward_pos_inf for
        # the semantics and the asymmetry it fixes.
        rescaled = round_half_up_toward_pos_inf(summed / float(self.output_scale_factor))
        return torch.clamp(rescaled, -128, 127)


class Int8GlobalAveragePool(nn.Module):
    """Per-channel reduction over H*W cells, divided by H*W, requantized.

    INT8 input → float accumulator over the spatial domain → divide by H*W →
    rescale by the composite (input_scale / output_scale) → clamp to INT8.

    The RTL form mirrors this exactly: an INT32 accumulator per channel
    consumes one cell per cycle, and when the spatial counter wraps the result
    is multiplied by SCALE_MULT/SCALE_SHIFT (composite scale folds in the
    1/(H*W) divisor) and clamped to INT8.
    """

    def __init__(self, input_scale: float, output_scale: float) -> None:
        super().__init__()
        self.input_scale = float(input_scale)
        self.output_scale = float(output_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise GoldenGenerationError(
                f"Int8GlobalAveragePool expected 4D input, got shape {tuple(x.shape)}"
            )
        n, c, h, w = x.shape
        if h <= 0 or w <= 0:
            raise GoldenGenerationError(
                f"Int8GlobalAveragePool input has non-positive spatial dim H={h} W={w}"
            )
        # Accumulate in float to avoid INT16 overflow on large H*W (e.g.
        # 224*224 = 50176 cells fits in INT16 sum-of-INT8s but H*W=14*14 is
        # the typical classifier-head case and INT32 stays safe).
        acc = x.to(torch.float32).sum(dim=(2, 3))
        # Composite scale: input_scale * (1 / output_scale) / (H*W) is the
        # equivalent multiplier from accumulator-INT to output-INT8. We split
        # the divide-by-(H*W) into the composite so the RTL doesn't need a
        # divider; SCALE_MULT/SHIFT carries it.
        composite = self.input_scale / float(self.output_scale) / float(h * w)
        rescaled = round_half_up_toward_pos_inf(acc * composite)
        clamped = torch.clamp(rescaled, -128, 127)
        # Restore the [N, C, 1, 1] shape that ONNX GlobalAveragePool emits;
        # downstream Flatten/Reshape is folded out at frontend extraction time.
        return clamped.unsqueeze(-1).unsqueeze(-1)


class Int8Gemm(nn.Module):
    """INT8 fully-connected layer: `out = W @ x + b`.

    Weights are stored as INT8 [M, K] (pre-quantized); bias is INT32 in
    accumulator domain. The composite scale (input_scale * weight_scale /
    output_scale) is applied after the accumulator + bias, matching the
    conv requantize tail.
    """

    def __init__(
        self,
        weight_int8: torch.Tensor,
        bias_int32: Optional[torch.Tensor],
        scale_factor: float,
    ) -> None:
        super().__init__()
        if weight_int8.dim() != 2:
            raise GoldenGenerationError(
                f"Int8Gemm expects 2D weight [M, K], got shape {tuple(weight_int8.shape)}"
            )
        self.register_buffer("weight", weight_int8.to(torch.int32))
        if bias_int32 is not None:
            self.register_buffer("bias", bias_int32.to(torch.int32))
        else:
            self.bias = None
        self.scale_factor = float(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept any input rank. nn.Linear-style: matmul against the last
        # K dims combined. ONNX models can route either [N, K] (post-Flatten)
        # or [N, C, 1, 1] (post-GlobalAveragePool, no Flatten) into Gemm; both
        # collapse to [N, K] here. The K-dim product check guarantees the
        # weight matrix and input are compatible.
        if x.dim() > 2:
            n = x.shape[0]
            k_observed = int(torch.tensor(x.shape[1:]).prod().item())
            k_weight = int(self.weight.shape[1])
            if k_observed != k_weight:
                raise GoldenGenerationError(
                    f"Int8Gemm input shape {tuple(x.shape)} has flattened K={k_observed} "
                    f"but weight expects K={k_weight}."
                )
            x = x.reshape(n, k_weight)
        x_int32 = x.to(torch.int32)
        # x: [N, K], weight: [M, K] → matmul over K
        acc = x_int32 @ self.weight.t()
        if self.bias is not None:
            acc = acc + self.bias.view(1, -1)
        # Same requantize semantics as conv: round half toward +inf, clamp to
        # INT8.
        rescaled = round_half_up_toward_pos_inf(acc.to(torch.float32) * self.scale_factor)
        return torch.clamp(rescaled, -128, 127)


class ResidualStackTracer(fx.Tracer):
    def is_leaf_module(self, module: nn.Module, qualname: str) -> bool:
        if isinstance(module, (
            Int8Conv2d, Int8FusedStemConv2d, Int8ReLU, Int8Add,
            Int8GlobalAveragePool, Int8Gemm,
        )):
            return True
        return super().is_leaf_module(module, qualname)


class ActivationCaptureInterpreter(fx.Interpreter):
    def __init__(self, graph_module: fx.GraphModule) -> None:
        super().__init__(graph_module)
        self.captured_outputs: dict[str, Any] = {}

    def run_node(self, node: fx.Node) -> Any:
        result = super().run_node(node)
        self.captured_outputs[node.name] = result
        return result


class CheckpointResidualStack(nn.Module):
    def __init__(
        self,
        operations: Sequence[Mapping[str, Any]],
        layers: Mapping[str, Mapping[str, Any]],
        output_module_id: str | None = None,
        input_name: str = "input",
    ) -> None:
        super().__init__()
        self.operations = [dict(operation) for operation in operations]
        self.output_module_id = output_module_id
        self.input_name = input_name

        for operation in self.operations:
            module_id = require_string(operation, "module_id", "operation")
            op_type = require_op_type(operation, "operation")
            if module_id not in layers:
                raise GoldenGenerationError(
                    f"Checkpoint graph references unknown layer '{module_id}'."
                )
            metadata = layers[module_id]
            if op_type == "conv2d":
                weight_tensor, bias_tensor = resolve_layer_parameters(metadata)
                if metadata.get("batch_norm") is not None:
                    weight_tensor, bias_tensor = fold_batch_norm_from_metadata(
                        weight_tensor,
                        bias_tensor,
                        metadata["batch_norm"],
                    )
                conv_cls = Int8Conv2d
                module = conv_cls(
                    weight=weight_tensor,
                    bias=bias_tensor,
                    stride=coerce_int_sequence(operation.get("stride", [1, 1]), "stride"),
                    padding=coerce_int_sequence(
                        operation.get("padding", [0, 0]),
                        "padding",
                        allow_zero=True,
                    ),
                    dilation=coerce_int_sequence(operation.get("dilation", [1, 1]), "dilation"),
                    groups=int(operation.get("groups", 1)),
                    scale_factor=float(metadata.get("scale_factor", 1.0)),
                )
            elif op_type == "relu":
                module = Int8ReLU()
            else:
                module = Int8Add(
                    lhs_scale_factor=float(metadata.get("lhs_scale_factor", 1.0)),
                    rhs_scale_factor=float(metadata.get("rhs_scale_factor", 1.0)),
                    output_scale_factor=float(metadata.get("scale_factor", 1.0)),
                )
            register_module_path(self, module_id, module)

    def forward(self, x):
        values: dict[str, torch.Tensor] = {self.input_name: quantize_tensor_to_int8_range(x)}
        previous_id = self.input_name

        for operation in self.operations:
            module_id = require_string(operation, "module_id", "operation")
            op_type = require_op_type(operation, "operation")
            module = get_module_by_path(self, module_id)

            if op_type == "add":
                lhs = require_string(operation, "lhs", f"operation '{module_id}'")
                rhs = require_string(operation, "rhs", f"operation '{module_id}'")
                output = module(values[lhs], values[rhs])
            else:
                input_ref = str(operation.get("input", previous_id))
                if input_ref not in values:
                    raise GoldenGenerationError(
                        f"Operation '{module_id}' references unknown input '{input_ref}'."
                    )
                output = module(values[input_ref])

            values[module_id] = quantize_tensor_to_int8_range(output)
            previous_id = module_id

        final_id = self.output_module_id or previous_id
        if final_id not in values:
            raise GoldenGenerationError(
                f"Checkpoint output_module_id '{final_id}' is not produced by the graph."
            )
        return values[final_id]


def resolve_output_dir(repo_root: Path) -> Path:
    raw = os.environ.get("NN2RTL_OUTPUT_DIR")
    if raw:
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else repo_root / candidate
    return repo_root / "output"


def get_output_paths(repo_root: Path) -> tuple[Path, Path, Path]:
    output_dir = resolve_output_dir(repo_root)
    layer_ir_path = output_dir / LAYER_IR_FILE_NAME
    legacy_output_path = output_dir / LEGACY_GOLDEN_FILE_NAME
    weights_dir = output_dir / "weights"
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    return layer_ir_path, legacy_output_path, weights_dir


def get_goldens_dir(repo_root: Path) -> Path:
    goldens_dir = resolve_output_dir(repo_root) / "goldens"
    goldens_dir.mkdir(parents=True, exist_ok=True)
    return goldens_dir


def get_golden_artifact_paths(repo_root: Path, module_id: str) -> tuple[Path, Path]:
    goldens_dir = get_goldens_dir(repo_root)
    return (
        goldens_dir / f"{module_id}.goldin",
        goldens_dir / f"{module_id}.goldout",
    )


# --- Binary vector file format (.goldin / .goldout) -----------------------
# Full ResNet-50 feature maps inflate inline JSON LayerIR to multi-GB. Storing
# golden vectors as binary sidecar files keeps the LayerIR itself small (so
# Node's readFileSync 512 MB string cap and MCP argument-size limits don't
# bite) while preserving per-module verification coverage.
#
# Layout (all little-endian):
#   [ 0..4)  magic            : 4 bytes, ASCII "NN2V"
#   [ 4..8)  version          : uint32, current=2
#   [ 8..12) num_vectors      : uint32
#   [12..16) samples_per_vector : uint32
#   [16..20) bytes_per_sample : uint32
#   [20..)   data             : num_vectors * samples_per_vector *
#                               ceil(bytes_per_sample / 4) int32 words
#
# Each logical sample is a packed bus value for one cycle. Within a sample the
# bytes are little-endian: byte 0 is the least-significant 8 bits of word 0,
# byte 1 the next 8 bits, etc. `data_in[i*8 +: 8]` / `data_out[i*8 +: 8]`
# therefore correspond to channel i.
GOLDEN_FILE_MAGIC = b"NN2V"
GOLDEN_FILE_VERSION = 2
GOLDEN_FILE_HEADER_STRUCT = struct.Struct("<4sIIII")


def bus_bytes_for_bits(bus_bits: int, *, context: str) -> int:
    if not isinstance(bus_bits, int) or isinstance(bus_bits, bool):
        raise GoldenGenerationError(f"{context} must be an integer, got {bus_bits!r}.")
    if bus_bits <= 0 or bus_bits % 8 != 0:
        raise GoldenGenerationError(
            f"{context} must be a positive multiple of 8, got {bus_bits}."
        )
    return bus_bits // 8


def int32_words_for_bus_bytes(bus_bytes: int) -> int:
    return (bus_bytes + 3) // 4


def channel_bus_bits_from_shape(shape: Sequence[int], *, context: str) -> int:
    resolved = coerce_shape(shape, context)
    if len(resolved) < 2:
        raise GoldenGenerationError(
            f"{context} must have at least batch and channel dimensions, got {resolved}."
        )
    return int(resolved[1]) * 8


def normalize_layer_bus_width_bits(
    *,
    module_id: str,
    field_name: str,
    stored_value: Any,
    expected_value: int,
    legacy_value: int,
) -> int:
    if stored_value is None:
        return expected_value
    actual_value = coerce_int(stored_value, f"{module_id}.{field_name}")
    if actual_value in {expected_value, legacy_value}:
        return expected_value
    raise GoldenGenerationError(
        f"Layer '{module_id}' field '{field_name}' must be {expected_value} "
        f"(channel-packed) or legacy {legacy_value}, got {actual_value}."
    )


def pack_word_to_signed_int32(word: int) -> int:
    return word - 2**32 if word >= 2**31 else word


def tensor_to_bus_samples(
    tensor: torch.Tensor,
    *,
    context: str,
) -> list[list[int]]:
    quantized = quantize_tensor_to_int8_range(tensor).to(torch.int32)
    if quantized.ndim != 4 or int(quantized.shape[0]) != 1:
        raise GoldenGenerationError(
            f"{context} must have shape [1, C, H, W], got {list(quantized.shape)}."
        )
    _, channels, height, width = quantized.shape
    flattened = (
        quantized.squeeze(0)
        .permute(1, 2, 0)
        .contiguous()
        .reshape(int(height) * int(width), int(channels))
    )
    return [[int(value) for value in sample] for sample in flattened.tolist()]


def pack_bus_sample_words(
    sample_bytes: Sequence[int],
    bus_bits: int,
    *,
    context: str,
) -> list[int]:
    bus_bytes = bus_bytes_for_bits(bus_bits, context=context)
    if len(sample_bytes) != bus_bytes:
        raise GoldenGenerationError(
            f"{context} must provide exactly {bus_bytes} bytes for bus_bits={bus_bits}, "
            f"got {len(sample_bytes)}."
        )

    words: list[int] = []
    for byte_start in range(0, bus_bytes, 4):
        word = 0
        for byte_offset, value in enumerate(sample_bytes[byte_start : byte_start + 4]):
            word |= (int(value) & 0xFF) << (8 * byte_offset)
        words.append(pack_word_to_signed_int32(word))
    return words


def pack_tensor_vectors_to_bus_words(
    tensors: Sequence[torch.Tensor],
    bus_bits: int,
    *,
    context: str,
) -> list[list[int]]:
    packed_vectors: list[list[int]] = []
    for vector_index, tensor in enumerate(tensors):
        vector_words: list[int] = []
        for sample_index, sample_bytes in enumerate(
            tensor_to_bus_samples(tensor, context=f"{context} vector[{vector_index}]")
        ):
            vector_words.extend(
                pack_bus_sample_words(
                    sample_bytes,
                    bus_bits,
                    context=f"{context} vector[{vector_index}] sample[{sample_index}]",
                )
            )
        packed_vectors.append(vector_words)
    return packed_vectors


def pack_paired_tensors_to_bus_words(
    lhs_tensors: Sequence[torch.Tensor],
    rhs_tensors: Sequence[torch.Tensor],
    bus_bits: int,
    *,
    context: str,
) -> list[list[int]]:
    if len(lhs_tensors) != len(rhs_tensors):
        raise GoldenGenerationError(
            f"{context} requires matching vector counts, got {len(lhs_tensors)} and {len(rhs_tensors)}."
        )

    packed_vectors: list[list[int]] = []
    for vector_index, (lhs_tensor, rhs_tensor) in enumerate(zip(lhs_tensors, rhs_tensors)):
        lhs_samples = tensor_to_bus_samples(
            lhs_tensor,
            context=f"{context} lhs vector[{vector_index}]",
        )
        rhs_samples = tensor_to_bus_samples(
            rhs_tensor,
            context=f"{context} rhs vector[{vector_index}]",
        )
        if len(lhs_samples) != len(rhs_samples):
            raise GoldenGenerationError(
                f"{context} vector[{vector_index}] requires matching sample counts, "
                f"got {len(lhs_samples)} and {len(rhs_samples)}."
            )

        vector_words: list[int] = []
        for sample_index, (lhs_sample, rhs_sample) in enumerate(zip(lhs_samples, rhs_samples)):
            vector_words.extend(
                pack_bus_sample_words(
                    [*lhs_sample, *rhs_sample],
                    bus_bits,
                    context=f"{context} vector[{vector_index}] sample[{sample_index}]",
                )
            )
        packed_vectors.append(vector_words)
    return packed_vectors


def write_golden_vector_file(
    values: Sequence[Sequence[int]],
    file_path: Path,
    bus_bits: int,
) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    bytes_per_sample = bus_bytes_for_bits(bus_bits, context=f"{file_path}.bus_bits")
    words_per_sample = int32_words_for_bus_bytes(bytes_per_sample)
    num_vectors = len(values)
    samples_per_vector = len(values[0]) // words_per_sample if num_vectors > 0 else 0
    for vector in values:
        if len(vector) % words_per_sample != 0:
            raise ValueError(
                f"Golden vector file '{file_path}' requires rows to be a multiple of "
                f"{words_per_sample} int32 words per bus sample; found {len(vector)} words."
            )
        if len(vector) != samples_per_vector * words_per_sample:
            raise ValueError(
                f"Golden vector file '{file_path}' requires every row to have "
                f"{samples_per_vector * words_per_sample} int32 words "
                f"({samples_per_vector} samples at {words_per_sample} words/sample); "
                f"found {len(vector)}."
            )

    header = GOLDEN_FILE_HEADER_STRUCT.pack(
        GOLDEN_FILE_MAGIC,
        GOLDEN_FILE_VERSION,
        num_vectors,
        samples_per_vector,
        bytes_per_sample,
    )

    total_words = num_vectors * samples_per_vector * words_per_sample
    flat_array = array.array("i")
    flat_array.extend(int(v) for vector in values for v in vector)
    if len(flat_array) != total_words:
        raise ValueError(
            f"Expected {total_words} int32 words in '{file_path}', "
            f"got {len(flat_array)}."
        )

    with file_path.open("wb") as fh:
        fh.write(header)
        flat_array.tofile(fh)


def read_golden_vector_file(file_path: Path, bus_bits: int) -> list[list[int]]:
    expected_bytes_per_sample = bus_bytes_for_bits(
        bus_bits,
        context=f"{file_path}.bus_bits",
    )
    with file_path.open("rb") as fh:
        header_bytes = fh.read(GOLDEN_FILE_HEADER_STRUCT.size)
        if len(header_bytes) != GOLDEN_FILE_HEADER_STRUCT.size:
            raise GoldenGenerationError(f"Golden vector file '{file_path}' is truncated.")
        magic, version, num_vectors, samples_per_vector, bytes_per_sample = GOLDEN_FILE_HEADER_STRUCT.unpack(header_bytes)
        if magic != GOLDEN_FILE_MAGIC:
            raise GoldenGenerationError(
                f"Golden vector file '{file_path}' has wrong magic: {magic!r}."
            )
        if version != GOLDEN_FILE_VERSION:
            raise GoldenGenerationError(
                f"Golden vector file '{file_path}' version {version} unsupported."
            )
        if bytes_per_sample != expected_bytes_per_sample:
            raise GoldenGenerationError(
                f"Golden vector file '{file_path}' bytes_per_sample={bytes_per_sample} "
                f"does not match bus_bits={bus_bits} ({expected_bytes_per_sample} bytes)."
            )

        words_per_sample = int32_words_for_bus_bytes(bytes_per_sample)
        flat = array.array("i")
        flat.fromfile(fh, num_vectors * samples_per_vector * words_per_sample)

    result: list[list[int]] = []
    for v in range(num_vectors):
        start = v * samples_per_vector * words_per_sample
        stop = start + samples_per_vector * words_per_sample
        result.append(list(flat[start:stop]))
    return result


def get_legacy_output_path(repo_root: Path) -> Path:
    return resolve_output_dir(repo_root) / LEGACY_GOLDEN_FILE_NAME


def get_weight_artifact_paths(repo_root: Path, module_id: str) -> tuple[Path, Path]:
    _, _, weights_dir = get_output_paths(repo_root)
    return (
        weights_dir / f"{module_id}_weights.hex",
        weights_dir / f"{module_id}_bias.hex",
    )


def get_weight_bank_artifact_paths(
    repo_root: Path,
    module_id: str,
    mac_parallelism: int,
) -> list[Path]:
    _, _, weights_dir = get_output_paths(repo_root)
    mp = max(1, int(mac_parallelism))
    return [weights_dir / f"{module_id}_weights_bank{lane}.hex" for lane in range(mp)]


def bank_weight_values_for_mac_lanes(
    weight_values: Sequence[int],
    weight_shape: Sequence[int],
    mac_parallelism: int,
) -> list[list[int]]:
    if len(weight_shape) < 4:
        raise GoldenGenerationError("weight_shape must be [OC, IC, KH, KW] to bank conv weights.")
    oc, ic, kh, kw = (int(v) for v in weight_shape[:4])
    if oc <= 0 or ic <= 0 or kh <= 0 or kw <= 0:
        raise GoldenGenerationError(f"Invalid conv weight_shape for banking: {list(weight_shape)}.")
    mp = max(1, min(int(mac_parallelism), oc))
    k_total = ic * kh * kw
    expected = oc * k_total
    if len(weight_values) != expected:
        raise GoldenGenerationError(
            f"Cannot bank {len(weight_values)} weights for shape {list(weight_shape)}; expected {expected}."
        )

    oc_passes = (oc + mp - 1) // mp
    banks: list[list[int]] = [[] for _ in range(mp)]
    for oc_group in range(oc_passes):
        for lane in range(mp):
            oc_index = oc_group * mp + lane
            if oc_index < oc:
                start = oc_index * k_total
                banks[lane].extend(int(v) for v in weight_values[start:start + k_total])
            else:
                banks[lane].extend(0 for _ in range(k_total))
    return banks


def write_weight_bank_hex_files(
    weight_values: Sequence[int],
    weight_shape: Sequence[int],
    mac_parallelism: int,
    repo_root: Path,
    module_id: str,
) -> list[Path]:
    # The legacy on-chip datapaths consume bank files, but the
    # dram-backed-weights AXI testbench consumes the flat byte stream at
    # <module>_weights.hex. Keep both artifacts in sync whenever banking runs.
    flat_weights_path, _ = get_weight_artifact_paths(repo_root, module_id)
    write_signed_int8_hex(weight_values, flat_weights_path)

    bank_values = bank_weight_values_for_mac_lanes(
        weight_values,
        weight_shape,
        mac_parallelism,
    )
    bank_paths = get_weight_bank_artifact_paths(repo_root, module_id, len(bank_values))
    for values, bank_path in zip(bank_values, bank_paths):
        write_signed_int8_hex(values, bank_path)
    return bank_paths


def int8_to_hex(value: int) -> str:
    if value < -128 or value > 127:
        raise ValueError(f"INT8 value out of range: {value}")
    return f"{value & 0xFF:02X}"


def int32_to_hex(value: int) -> str:
    if value < -(2**31) or value > 2**31 - 1:
        raise ValueError(f"INT32 value out of range: {value}")
    return f"{value & 0xFFFFFFFF:08X}"


def write_signed_int8_hex(values: Iterable[int], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        "".join(f"{int8_to_hex(int(value))}\n" for value in values),
        encoding="utf8",
        newline="",
    )


def write_signed_int32_hex(values: Iterable[int], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        "".join(f"{int32_to_hex(int(value))}\n" for value in values),
        encoding="utf8",
        newline="",
    )


def fold_batch_norm_into_conv(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bias is None:
        bias = torch.zeros_like(running_mean)

    scale = bn_weight / torch.sqrt(running_var + eps)
    folded_weight = weight * scale.reshape(-1, 1, 1, 1)
    folded_bias = (bias - running_mean) * scale + bn_bias
    return folded_weight, folded_bias


def quantize_tensor_to_int8_range(tensor: torch.Tensor) -> torch.Tensor:
    working = tensor.detach() if isinstance(tensor, torch.Tensor) else tensor
    if isinstance(working, torch.Tensor) and working.is_quantized:
        working = working.dequantize()
    return torch.clamp(working.to(torch.float32).round(), -128, 127)


def compute_scale_approx(scale_factor: float) -> tuple[int, int]:
    """Pick (SCALE_MULT, SCALE_SHIFT) that approximate ``scale_factor`` with
    minimum relative error, identical to ``computeScaleApprox`` in
    ``sdk/orchestrate.ts``. Both must agree because the SDK is the one that
    embeds these constants into the generated Verilog's ``localparam``
    block; the golden model must use the same approximation, not the true
    float ``scale_factor``, to be bit-equivalent to RTL.

    Search range ``shift ∈ [0, 23]`` and ``1 ≤ mult < 32768`` matches the
    SDK exactly. Layers with scale_factor > 128 (e.g. node_relu_14 at
    283.33) need shift < 8 to fit mult inside the 15-bit cap; clamping
    Python to shift ≥ 8 left those layers with the (1, 8) sentinel while
    the SDK picked a real (mult, shift) — the resulting golden vs RTL
    constants disagreed silently. Tie-break: the SDK keeps the FIRST shift
    that improves the error (strict ``<``), so we replicate that ordering.
    """
    if scale_factor <= 0.0:
        raise GoldenGenerationError(f"scale_factor must be positive, got {scale_factor}.")
    best_mult, best_shift, best_err = 1, 0, float("inf")
    for shift in range(0, 24):
        mult = round(scale_factor * (2 ** shift))
        if mult < 1 or mult >= 32768:
            continue
        err = abs(mult / (2 ** shift) - scale_factor) / scale_factor
        if err < best_err:
            best_mult, best_shift, best_err = mult, shift, err
    return best_mult, best_shift


def requantize_fixed_point_int(value: int, scale_factor: float) -> int:
    """Bit-exact mirror of the RTL requantize stage:

        scaled = (value * SCALE_MULT + (1 << (SCALE_SHIFT - 1))) >>> SCALE_SHIFT
        out    = clamp(scaled, -128, 127)

    Python's ``>>`` on negative ints already floors toward -inf, matching
    Verilog's arithmetic ``>>>`` on a signed reg; combined with the
    positive ROUND_BIAS this realises round-half-toward-+inf without ever
    leaving integer arithmetic. The result is bit-identical to the RTL
    output (subject to the same ``compute_scale_approx`` constants). Use
    this in place of the float-domain ``round(x * scale_factor)`` whenever
    bit-equivalence with RTL matters.
    """
    mult, shift = compute_scale_approx(scale_factor)
    raw = int(value) * mult + (1 << (shift - 1))
    scaled = raw >> shift  # arithmetic shift right on signed Python int
    if scaled > 127:
        return 127
    if scaled < -128:
        return -128
    return scaled


def round_half_up_toward_pos_inf(tensor: torch.Tensor) -> torch.Tensor:
    """Round-half-toward-+infinity, matching the RTL requantize stage.

    The RTL implements requantization as
        v_tmp = (scaled + (1 << (SCALE_SHIFT - 1))) >>> SCALE_SHIFT;
    an arithmetic right shift after adding a fixed positive half-LSB bias.
    For a real value ``x``, this computes ``floor(x + 0.5)``. Crucially:

      *  +N.5  ->  N+1   (rounds up)
      *  -N.5  ->  -N    (rounds up — i.e. toward +inf, not away from 0)

    PyTorch's ``torch.round`` uses banker's rounding (round-half-to-even),
    so it disagrees with the RTL on every exact .5 tie. We reproduce the
    RTL behavior here so the goldens are bit-equivalent rather than just
    within tolerance. ``floor(x + 0.5)`` works in float32 for the value
    ranges this pipeline produces (accumulator products fit comfortably in
    float64; we keep float32 for speed and match Python's tensor-default
    promotion).
    """
    return torch.floor(tensor.to(torch.float32) + 0.5)


def requantize_tensor_with_scale(tensor: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Bit-exact mirror of the RTL requantize stage.

    In our INT8 pipeline the accumulator holds raw integer dot-products
    (int8 × int8 summed over IC channels) plus the INT32 bias. The RTL
    requantizes via integer multiply-and-shift using the same constants
    ``computeScaleApprox`` (TS, in ``sdk/orchestrate.ts``) and
    ``compute_scale_approx`` (Python, here) pick:

        scaled = (acc_plus_bias * SCALE_MULT + ROUND_BIAS) >>> SCALE_SHIFT
        out    = clamp(scaled, -128, 127)

    To produce goldens that are BIT-IDENTICAL to RTL — not merely within
    the testbench's ``max_error <= 3`` tolerance — we do the same integer
    arithmetic here. Using the true float ``scale_factor`` would leave a
    residual ±1 LSB on values where the fixed-point approximation rounds
    differently from float multiplication. That residual is what the
    earlier float-path implementations of this function produced.

    The input ``tensor`` is the post-bias accumulator value (the
    convolution sum + bias added in float domain). We round it back to an
    integer first (the float input represents an integer value because
    int8 weights × int8 inputs + int32 bias is an exact integer; we trust
    it), then apply the fixed-point requantize per element.
    """
    if scale_factor <= 0.0:
        raise GoldenGenerationError(f"scale_factor must be positive, got {scale_factor}.")
    working = tensor.detach() if isinstance(tensor, torch.Tensor) else tensor
    if isinstance(working, torch.Tensor) and working.is_quantized:
        working = working.dequantize()
    # Compute (mult, shift) ONCE per call — same constants the SDK embeds
    # into the generated Verilog's localparams. Then do the fixed-point
    # multiply-add-shift on the whole tensor in vectorised PyTorch (NOT
    # a Python loop): for an int64 tensor, ``//`` is floor division which
    # matches Verilog's arithmetic ``>>>`` on signed regs (both round
    # toward -inf for negative values).
    mult, shift = compute_scale_approx(float(scale_factor))
    round_bias = 1 << (shift - 1)
    divisor = 1 << shift
    # Snap the float input back to its underlying integer (acc+bias is an
    # exact integer; we trust the host computed it correctly).
    integer_input = working.to(torch.float64).round().to(torch.int64)
    raw = integer_input * mult + round_bias
    scaled = torch.div(raw, divisor, rounding_mode="floor")
    return torch.clamp(scaled.to(torch.float32), -128, 127)


def tensor_to_int8_list(tensor: torch.Tensor) -> list[int]:
    return [int(value) for value in quantize_tensor_to_int8_range(tensor).reshape(-1).tolist()]


def tensor_to_int32_list(tensor: torch.Tensor) -> list[int]:
    working = tensor.detach() if isinstance(tensor, torch.Tensor) else tensor
    if isinstance(working, torch.Tensor) and working.is_quantized:
        working = working.dequantize()
    int32_min, int32_max = -(2**31), 2**31 - 1
    clamped = torch.clamp(working.to(torch.float64).round(), int32_min, int32_max)
    return [int(value) for value in clamped.reshape(-1).tolist()]


def utc_now_iso8601() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def summarize_pipeline_ir(
    payload: Mapping[str, Any],
    checkpoint_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "model_name": payload["model_name"],
        "num_layers": len(payload["layers"]),
        "checkpoint_path": checkpoint_path.resolve().as_posix(),
        "pipeline_ir_path": output_path.resolve().as_posix(),
    }


def build_pipeline_ir_payload(
    checkpoint_path: Path,
    repo_root: Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(raw_checkpoint, dict):
        raise GoldenGenerationError("Checkpoint payload must deserialize to a dict.")

    format_version = int(raw_checkpoint.get("format_version", 1))
    if format_version == 1:
        checkpoint = load_quantized_checkpoint(checkpoint_path)
        payload = build_legacy_pipeline_ir_payload(checkpoint, repo_root)
    elif format_version == 2:
        # The fx path needs either a pickled nn.Module or a residual_stack_spec
        # describing how the `layers` dict wires together. A v2 checkpoint that
        # ships only a flat `layers` dict cannot be traced and cannot produce
        # real golden activations — earlier revisions silently fell back to
        # empty golden_inputs/golden_outputs, which passed schema validation
        # but made every downstream Assayer verification a false positive.
        if not _v2_has_traceable_spec(raw_checkpoint):
            raise GoldenGenerationError(
                "format_version=2 checkpoint lacks a traceable model spec. "
                "The `layers` dict alone is not sufficient to capture per-module "
                "golden activations — the checkpoint must also embed one of: "
                "a pickled nn.Module under `quantized_residual_stack` / "
                "`residual_stack` / `model`, OR a `residual_stack_spec` / "
                "`model_spec` / `graph` describing how the layers wire together "
                "(see CheckpointResidualStack for the expected operations "
                "schema). See ARCHITECTURE.md for why this is a hard error."
            )
        # Route v2 checkpoints through the strict ResNet-50 validator in
        # quantize_impl.py so the same shape/scope checks apply to the main
        # golden-generation path. Without this a handcrafted bias_int32=[300]
        # checkpoint would be accepted here even though load_quantized_checkpoint
        # would reject it. Pass the validated payload onward.
        load_quantized_checkpoint(checkpoint_path)
        payload = build_fx_pipeline_ir_payload(
            checkpoint=raw_checkpoint,
            repo_root=repo_root,
            generated_at=generated_at or utc_now_iso8601(),
        )
    else:
        raise GoldenGenerationError(f"Unsupported checkpoint format_version: {format_version}")

    validate_pipeline_ir_payload(payload)
    return payload


def write_pipeline_ir(
    repo_root: Path,
    checkpoint_path: Path,
    generated_at: str | None = None,
) -> Path:
    layer_ir_path, legacy_output_path, _ = get_output_paths(repo_root)
    payload = build_pipeline_ir_payload(checkpoint_path, repo_root, generated_at=generated_at)
    encoded = json.dumps(payload, indent=2) + "\n"
    layer_ir_path.write_text(encoded, encoding="utf8", newline="")
    legacy_output_path.write_text(encoded, encoding="utf8", newline="")
    return layer_ir_path


def validate_pipeline_ir_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("quantization") != "int8_symmetric_per_tensor":
        raise GoldenGenerationError(
            "PipelineIR quantization must be 'int8_symmetric_per_tensor'."
        )

    generated_at = payload.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at.endswith("Z"):
        raise GoldenGenerationError("PipelineIR generated_at must be an ISO-8601 UTC string.")

    layers = payload.get("layers")
    if not isinstance(layers, list) or not layers:
        raise GoldenGenerationError("PipelineIR must contain at least one layer.")

    for layer in layers:
        if not isinstance(layer, Mapping):
            raise GoldenGenerationError("Each layer must be a dict.")
        module_id = require_string(layer, "module_id", "LayerIR")
        require_op_type(layer, f"LayerIR '{module_id}'")
        coerce_shape(layer.get("input_shape"), f"{module_id}.input_shape")
        coerce_shape(layer.get("output_shape"), f"{module_id}.output_shape")
        coerce_shape(layer.get("weight_shape"), f"{module_id}.weight_shape")
        coerce_nonnegative_int(layer.get("num_weights"), f"{module_id}.num_weights")
        coerce_int(layer.get("zero_point"), f"{module_id}.zero_point")

        for signal_name, literal in SIGNAL_LITERALS.items():
            if layer.get(signal_name) != literal:
                raise GoldenGenerationError(
                    f"Layer '{module_id}' must set {signal_name}='{literal}'."
                )

        weights_path = layer.get("weights_path")
        if not isinstance(weights_path, str) or not is_absolute_posix_path(weights_path):
            raise GoldenGenerationError(
                f"Layer '{module_id}' weights_path must be an absolute POSIX path."
            )

        bias_path = layer.get("bias_path")
        if bias_path is not None and (
            not isinstance(bias_path, str) or not is_absolute_posix_path(bias_path)
        ):
            raise GoldenGenerationError(
                f"Layer '{module_id}' bias_path must be null or an absolute POSIX path."
            )

        for golden_field in ("golden_inputs_path", "golden_outputs_path"):
            path_value = layer.get(golden_field)
            if not isinstance(path_value, str) or not is_absolute_posix_path(path_value):
                raise GoldenGenerationError(
                    f"Layer '{module_id}' {golden_field} must be an absolute POSIX path."
                )
            if not Path(path_value).exists():
                raise GoldenGenerationError(
                    f"Layer '{module_id}' {golden_field} '{path_value}' does not exist on disk."
                )

        if layer["op_type"] == "add":
            for field_name in ("lhs_scale_factor", "rhs_scale_factor"):
                value = layer.get(field_name)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0.0:
                    raise GoldenGenerationError(
                        f"Layer '{module_id}' field '{field_name}' must be a positive number for add layers."
                    )
        elif layer["op_type"] == "conv2d":
            for field_name, allow_zero in (("stride", False), ("padding", True)):
                value = layer.get(field_name)
                if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) < 2:
                    raise GoldenGenerationError(
                        f"Layer '{module_id}' field '{field_name}' must be a 2-element sequence for conv2d layers."
                    )
                try:
                    coerced = coerce_int_sequence(
                        value,
                        f"{module_id}.{field_name}",
                        allow_zero=allow_zero,
                    )
                except GoldenGenerationError as exc:
                    raise GoldenGenerationError(str(exc)) from exc
                if len(coerced) != 2:
                    raise GoldenGenerationError(
                        f"Layer '{module_id}' field '{field_name}' must be exactly [H, W] for conv2d layers."
                    )
            weight_bank_paths = layer.get("weight_bank_paths")
            if weight_bank_paths is not None:
                mac_parallelism = coerce_int(
                    layer.get("mac_parallelism"),
                    f"{module_id}.mac_parallelism",
                )
                if (
                    isinstance(weight_bank_paths, (str, bytes))
                    or not isinstance(weight_bank_paths, Sequence)
                    or len(weight_bank_paths) != mac_parallelism
                ):
                    raise GoldenGenerationError(
                        f"Layer '{module_id}' weight_bank_paths must contain one path per "
                        f"mac_parallelism lane ({mac_parallelism})."
                    )
                for bank_path in weight_bank_paths:
                    if not isinstance(bank_path, str) or not is_absolute_posix_path(bank_path):
                        raise GoldenGenerationError(
                            f"Layer '{module_id}' weight_bank_paths entries must be absolute POSIX paths."
                        )
                    if not Path(bank_path).exists():
                        raise GoldenGenerationError(
                            f"Layer '{module_id}' weight bank file '{bank_path}' does not exist on disk."
                        )


def build_legacy_pipeline_ir_payload(
    checkpoint: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    module_id = require_string(checkpoint, "module_id", "checkpoint")
    weights_path, bias_path = get_weight_artifact_paths(repo_root, module_id)
    model = create_toy_model(checkpoint)

    input_stream = [int(value) for value in checkpoint["golden_input_stream"]]
    output_stream = run_toy_model(model, input_stream)

    conv_weight = torch.tensor(checkpoint["conv_weight"], dtype=torch.float32).reshape(
        checkpoint["weight_shape"]
    )
    conv_bias = torch.tensor(checkpoint["conv_bias"], dtype=torch.float32)
    batch_norm = checkpoint["batch_norm"]
    folded_weight, folded_bias = fold_batch_norm_into_conv(
        conv_weight,
        conv_bias,
        torch.tensor(batch_norm["weight"], dtype=torch.float32),
        torch.tensor(batch_norm["bias"], dtype=torch.float32),
        torch.tensor(batch_norm["running_mean"], dtype=torch.float32),
        torch.tensor(batch_norm["running_var"], dtype=torch.float32),
        float(batch_norm["eps"]),
    )

    write_signed_int8_hex(tensor_to_int8_list(folded_weight), weights_path)
    write_signed_int8_hex(tensor_to_int8_list(folded_bias), bias_path)

    stream_shape = [1, 1, 1, len(input_stream)]

    goldin_path, goldout_path = get_golden_artifact_paths(repo_root, module_id)
    write_golden_vector_file(
        [list(input_stream)],
        goldin_path,
        bus_bits=int(checkpoint["input_width_bits"]),
    )
    write_golden_vector_file(
        [list(output_stream)],
        goldout_path,
        bus_bits=int(checkpoint["output_width_bits"]),
    )

    return {
        "model_name": checkpoint["model_name"],
        "quantization": checkpoint["quantization"],
        "generated_at": checkpoint["generated_at"],
        "layers": [
            {
                "module_id": module_id,
                "op_type": "conv2d",
                "input_shape": stream_shape,
                "output_shape": stream_shape,
                "weights_path": weights_path.resolve().as_posix(),
                "bias_path": bias_path.resolve().as_posix(),
                "weight_shape": checkpoint["weight_shape"],
                "num_weights": 1,
                "scale_factor": checkpoint["scale_factor"],
                "zero_point": checkpoint["zero_point"],
                "pipeline_latency_cycles": checkpoint["pipeline_latency_cycles"],
                "clock_period_ns": checkpoint["clock_period_ns"],
                "input_width_bits": checkpoint["input_width_bits"],
                "output_width_bits": checkpoint["output_width_bits"],
                "stride": [1, 1],
                "padding": [0, 0],
                **SIGNAL_LITERALS,
                "golden_inputs_path": goldin_path.resolve().as_posix(),
                "golden_outputs_path": goldout_path.resolve().as_posix(),
            }
        ],
    }


def _v2_has_traceable_spec(checkpoint: Mapping[str, Any]) -> bool:
    # build_fx_pipeline_ir_payload needs one of these to reconstruct the
    # network for tracing + golden activation capture. A flat `layers` dict
    # alone is not enough.
    return any(
        key in checkpoint
        for key in (
            "quantized_residual_stack",
            "residual_stack",
            "model",
            "residual_stack_spec",
            "model_spec",
            "graph",
        )
    )


def build_fx_pipeline_ir_payload(
    checkpoint: Mapping[str, Any],
    repo_root: Path,
    generated_at: str,
) -> dict[str, Any]:
    layers = require_layer_mapping(checkpoint)
    model = load_residual_stack_model(checkpoint, layers)
    traced_model = trace_residual_stack(model)
    trace_layers = collect_trace_layers(traced_model, layers)
    if not trace_layers:
        raise GoldenGenerationError("No conv2d/relu/add nodes were found in the fx trace.")

    first_input_shape = coerce_shape(
        layers[trace_layers[0]["module_id"]]["input_shape"],
        f"{trace_layers[0]['module_id']}.input_shape",
    )
    input_stream = build_deterministic_input_stream(first_input_shape)
    captured_outputs = capture_golden_outputs(traced_model, trace_layers, layers, input_stream)
    operation_map, input_name = extract_operation_map(checkpoint)
    layer_payloads: list[dict[str, Any]] = []

    for index, trace_layer in enumerate(trace_layers):
        module_id = trace_layer["module_id"]
        op_type = trace_layer["op_type"]
        metadata = layers[module_id]
        weight_values, bias_values = write_layer_hex_artifacts(
            repo_root=repo_root,
            module_id=module_id,
            op_type=op_type,
            metadata=metadata,
            traced_model=traced_model,
        )

        weights_path, bias_path = get_weight_artifact_paths(repo_root, module_id)
        expected_num_weights = coerce_nonnegative_int(metadata["num_weights"], f"{module_id}.num_weights")
        if expected_num_weights != len(weight_values):
            raise GoldenGenerationError(
                f"Layer '{module_id}' num_weights={expected_num_weights} but serialized {len(weight_values)} values."
            )

        expected_input_width_bits = (
            2
            * channel_bus_bits_from_shape(
                metadata["input_shape"],
                context=f"{module_id}.input_shape",
            )
            if op_type == "add"
            else channel_bus_bits_from_shape(
                metadata["input_shape"],
                context=f"{module_id}.input_shape",
            )
        )
        expected_output_width_bits = channel_bus_bits_from_shape(
            metadata["output_shape"],
            context=f"{module_id}.output_shape",
        )
        input_width_bits = normalize_layer_bus_width_bits(
            module_id=module_id,
            field_name="input_width_bits",
            stored_value=metadata.get("input_width_bits"),
            expected_value=expected_input_width_bits,
            legacy_value=16 if op_type == "add" else 8,
        )
        output_width_bits = normalize_layer_bus_width_bits(
            module_id=module_id,
            field_name="output_width_bits",
            stored_value=metadata.get("output_width_bits"),
            expected_value=expected_output_width_bits,
            legacy_value=8,
        )

        layer_input_vectors = resolve_golden_inputs(
            module_id=module_id,
            op_type=op_type,
            trace_layers=trace_layers,
            trace_index=index,
            operation_map=operation_map,
            input_name=input_name,
            input_tensors=input_stream,
            captured_outputs=captured_outputs,
            input_width_bits=input_width_bits,
        )
        layer_payload = {
            "module_id": module_id,
            "op_type": op_type,
            "input_shape": coerce_shape(metadata["input_shape"], f"{module_id}.input_shape"),
            "output_shape": coerce_shape(metadata["output_shape"], f"{module_id}.output_shape"),
            "weights_path": weights_path.resolve().as_posix(),
            "bias_path": None if op_type != "conv2d" else bias_path.resolve().as_posix(),
            "weight_shape": coerce_shape(metadata["weight_shape"], f"{module_id}.weight_shape"),
            "num_weights": expected_num_weights,
            "scale_factor": float(metadata["scale_factor"]),
            "zero_point": coerce_int(metadata["zero_point"], f"{module_id}.zero_point"),
            "clock_period_ns": 20,
            "input_width_bits": input_width_bits,
            "output_width_bits": output_width_bits,
            **SIGNAL_LITERALS,
        }
        if op_type == "conv2d":
            conv_weight_shape = coerce_shape(metadata["weight_shape"], f"{module_id}.weight_shape")
            conv_input_shape = coerce_shape(metadata["input_shape"], f"{module_id}.input_shape")
            conv_operation = operation_map.get(module_id, {})
            conv_stride = list(conv_operation.get("stride", [1, 1]))
            conv_padding = list(conv_operation.get("padding", [0, 0]))
            conv_dilation = list(conv_operation.get("dilation", [1, 1]))
            conv_groups = int(conv_operation.get("groups", 1))
            conv_oc = int(conv_weight_shape[0]) if conv_weight_shape else 0
            conv_mp = conv_mac_parallelism(conv_oc)
            weight_bank_paths = write_weight_bank_hex_files(
                weight_values,
                conv_weight_shape,
                conv_mp,
                repo_root,
                module_id,
            )
            layer_payload["stride"] = conv_stride
            layer_payload["padding"] = conv_padding
            layer_payload["dilation"] = conv_dilation
            layer_payload["groups"] = conv_groups
            layer_payload["mac_parallelism"] = conv_mp
            layer_payload["weight_bank_paths"] = [
                bank_path.resolve().as_posix() for bank_path in weight_bank_paths
            ]
            layer_payload["pipeline_latency_cycles"] = compute_conv2d_latency_cycles(
                conv_weight_shape,
                input_shape=conv_input_shape,
                stride=conv_stride,
                padding=conv_padding,
                mac_parallelism=conv_mp,
            )
        else:
            layer_payload["pipeline_latency_cycles"] = (
                compute_add_latency_cycles(layer_payload["output_shape"])
                if op_type == "add"
                else PIPELINE_LATENCY_CYCLES[op_type]
            )
        if op_type == "conv2d" and not bias_values:
            raise GoldenGenerationError(f"Layer '{module_id}' did not serialize a bias vector.")
        if op_type == "add":
            layer_payload["lhs_scale_factor"] = float(metadata["lhs_scale_factor"])
            layer_payload["rhs_scale_factor"] = float(metadata["rhs_scale_factor"])

        # Per-layer vector cap. Pointwise conv2d (KH=KW=1) and the other
        # cheap ops (relu, add, maxpool) keep all deterministic vectors —
        # the inter-frame coverage costs nothing because their per-pixel
        # latency is small. Spatial conv2d (KH*KW > 1) caps at one vector
        # because each pixel costs ~MP*K_TOTAL+6 cycles and an 8-vector
        # 100k-pixel sim runs ~50 minutes at Verilator's ~1.3 MHz, blowing
        # the VERILATOR_SIM_TIMEOUT_MS budget. One vector still covers all
        # output channels and produces a max_error / first_mismatch
        # measurement; multi-frame stress is exercised on the cheap
        # pointwise / add / relu paths instead.
        if op_type == "conv2d":
            conv_kh = int(coerce_shape(metadata["weight_shape"], f"{module_id}.weight_shape")[2])
            conv_kw = int(coerce_shape(metadata["weight_shape"], f"{module_id}.weight_shape")[3])
            golden_vector_count = 1 if conv_kh * conv_kw > 1 else len(layer_input_vectors)
        else:
            golden_vector_count = len(layer_input_vectors)
        capped_inputs = layer_input_vectors[:golden_vector_count]
        capped_outputs_tensors = list(captured_outputs[module_id])[:golden_vector_count]

        goldin_path, goldout_path = get_golden_artifact_paths(repo_root, module_id)
        write_golden_vector_file(
            capped_inputs,
            goldin_path,
            bus_bits=layer_payload["input_width_bits"],
        )
        write_golden_vector_file(
            pack_tensor_vectors_to_bus_words(
                capped_outputs_tensors,
                layer_payload["output_width_bits"],
                context=f"{module_id}.goldout",
            ),
            goldout_path,
            bus_bits=layer_payload["output_width_bits"],
        )
        layer_payload["golden_inputs_path"] = goldin_path.resolve().as_posix()
        layer_payload["golden_outputs_path"] = goldout_path.resolve().as_posix()

        layer_payloads.append(layer_payload)

    return {
        "model_name": "resnet50",
        "quantization": "int8_symmetric_per_tensor",
        "generated_at": generated_at,
        "layers": layer_payloads,
    }


def load_residual_stack_model(
    checkpoint: Mapping[str, Any],
    layers: Mapping[str, Mapping[str, Any]],
) -> nn.Module:
    for key in ("quantized_residual_stack", "residual_stack", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, nn.Module):
            return candidate.eval()

    graph_spec = checkpoint.get("residual_stack_spec") or checkpoint.get("model_spec") or checkpoint.get("graph")
    if graph_spec is None:
        raise GoldenGenerationError(
            "Checkpoint v2 must contain a residual_stack/model nn.Module or a residual_stack_spec graph."
        )

    if isinstance(graph_spec, Mapping):
        operations = graph_spec.get("operations")
        if not isinstance(operations, list) or not operations:
            raise GoldenGenerationError("Checkpoint residual_stack_spec.operations must be a non-empty list.")
        output_module_id = graph_spec.get("output_module_id")
        input_name = str(graph_spec.get("input_name", "input"))
    elif isinstance(graph_spec, list):
        operations = graph_spec
        output_module_id = None
        input_name = "input"
    else:
        raise GoldenGenerationError("Checkpoint residual_stack_spec must be a dict or list.")

    return CheckpointResidualStack(
        operations=operations,
        layers=layers,
        output_module_id=None if output_module_id is None else str(output_module_id),
        input_name=input_name,
    ).eval()


def trace_residual_stack(model: nn.Module) -> fx.GraphModule:
    tracer = ResidualStackTracer()
    graph = tracer.trace(model)
    for node in graph.nodes:
        if isinstance(node.type, str):
            node.type = None
    return fx.GraphModule(model, graph)


def collect_trace_layers(
    traced_model: fx.GraphModule,
    layers: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    named_modules = dict(traced_model.named_modules())
    used_module_ids: set[str] = set()
    trace_layers: list[dict[str, str]] = []

    for node in traced_model.graph.nodes:
        module_id: str | None = None
        op_type: str | None = None

        if node.op == "call_module":
            module_id = str(node.target)
            module = named_modules[module_id]
            if isinstance(module, (Int8Conv2d, Int8FusedStemConv2d, nn.Conv2d)):
                op_type = "conv2d"
            elif isinstance(module, (Int8ReLU, nn.ReLU, nn.ReLU6)):
                op_type = "relu"
            elif isinstance(module, Int8Add) or module.__class__.__name__.lower() in {"residualadd", "int8add"}:
                op_type = "add"
        elif node.op == "call_function":
            if node.target in (operator.add, torch.add):
                op_type = "add"
            elif node.target in (torch.relu, F.relu):
                op_type = "relu"
        elif node.op == "call_method":
            if node.target == "add":
                op_type = "add"
            elif node.target == "relu":
                op_type = "relu"

        if op_type is None:
            continue
        if module_id is None:
            module_id = resolve_functional_module_id(node, op_type, layers, used_module_ids)
        if module_id not in layers:
            raise GoldenGenerationError(
                f"fx node '{node.name}' resolved to '{module_id}', which is missing from checkpoint['layers']."
            )
        trace_layers.append({"module_id": module_id, "op_type": op_type, "node_name": node.name})
        used_module_ids.add(module_id)

    return trace_layers


def build_deterministic_input_stream(shape: Sequence[int], count: int = 8) -> list[torch.Tensor]:
    # These inline vectors are convenient while the pipeline is still small, but
    # they grow quickly for real ResNet activations. If MCP argument-size limits
    # become a problem, move golden vectors to per-layer binary artifacts and
    # store `golden_inputs_path` / `golden_outputs_path` references in LayerIR
    # instead of embedding the arrays inline.
    torch.manual_seed(0)
    return [
        quantize_tensor_to_int8_range(torch.randint(-128, 128, tuple(shape), dtype=torch.int32))
        for _ in range(count)
    ]


def capture_golden_outputs(
    traced_model: fx.GraphModule,
    trace_layers: Sequence[Mapping[str, str]],
    layers: Mapping[str, Mapping[str, Any]],
    input_stream: Sequence[torch.Tensor],
) -> dict[str, list[torch.Tensor]]:
    outputs: dict[str, list[torch.Tensor]] = {
        trace_layer["module_id"]: [] for trace_layer in trace_layers
    }

    for input_tensor in input_stream:
        interpreter = ActivationCaptureInterpreter(traced_model)
        interpreter.run(input_tensor.clone())
        for trace_layer in trace_layers:
            module_id = trace_layer["module_id"]
            node_name = trace_layer["node_name"]
            if node_name not in interpreter.captured_outputs:
                raise GoldenGenerationError(f"fx node '{node_name}' did not produce a runtime activation.")
            value = interpreter.captured_outputs[node_name]
            if not isinstance(value, torch.Tensor):
                raise GoldenGenerationError(f"fx node '{node_name}' returned a non-tensor output.")
            quantized_value = quantize_tensor_to_int8_range(value)
            actual_shape = [int(dim) for dim in quantized_value.shape]
            expected_shape = coerce_shape(layers[module_id]["output_shape"], f"{module_id}.output_shape")
            if actual_shape != expected_shape:
                raise GoldenGenerationError(
                    f"Layer '{module_id}' produced shape {actual_shape}, expected {expected_shape}."
                )
            outputs[module_id].append(quantized_value.clone())

    return outputs


def write_layer_hex_artifacts(
    repo_root: Path,
    module_id: str,
    op_type: str,
    metadata: Mapping[str, Any],
    traced_model: fx.GraphModule,
) -> tuple[list[int], list[int]]:
    weights_path, bias_path = get_weight_artifact_paths(repo_root, module_id)
    weight_values: list[int] = []
    bias_values: list[int] = []

    if op_type == "conv2d":
        weight_tensor, bias_tensor = resolve_layer_parameters(
            metadata,
            traced_model=traced_model,
            module_id=module_id,
        )
        if metadata.get("batch_norm") is not None and has_serialized_weight_values(metadata):
            weight_tensor, bias_tensor = fold_batch_norm_from_metadata(
                weight_tensor,
                bias_tensor,
                metadata["batch_norm"],
            )
        if bias_tensor is None:
            bias_tensor = torch.zeros(weight_tensor.shape[0], dtype=torch.float32)
        weight_values = tensor_to_int8_list(weight_tensor)
        # Folded conv bias is an INT32 accumulator-width quantity per the
        # checkpoint schema (`bias_int32`); writing it through the INT8 path
        # would silently clip any non-trivial folded bias before Foundry sees
        # it. Widths differ between weights (INT8) and bias (INT32).
        bias_values = tensor_to_int32_list(bias_tensor)

    write_signed_int8_hex(weight_values, weights_path)
    write_signed_int32_hex(bias_values, bias_path)
    return weight_values, bias_values


def extract_operation_map(
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], str]:
    graph_spec = checkpoint.get("residual_stack_spec") or checkpoint.get("model_spec") or checkpoint.get("graph")
    if graph_spec is None:
        return {}, "input"

    if isinstance(graph_spec, Mapping):
        operations = graph_spec.get("operations")
        if not isinstance(operations, list):
            raise GoldenGenerationError("Checkpoint residual_stack_spec.operations must be a list.")
        input_name = str(graph_spec.get("input_name", "input"))
    elif isinstance(graph_spec, list):
        operations = graph_spec
        input_name = "input"
    else:
        raise GoldenGenerationError("Checkpoint residual_stack_spec must be a dict or list.")

    operation_map: dict[str, dict[str, Any]] = {}
    for operation in operations:
        if not isinstance(operation, Mapping):
            raise GoldenGenerationError("Checkpoint graph operations must be dicts.")
        module_id = require_string(operation, "module_id", "operation")
        operation_map[module_id] = dict(operation)
    return operation_map, input_name


def pack_int8_pair(lhs: int, rhs: int) -> int:
    packed = (int(lhs) & 0xFF) | ((int(rhs) & 0xFF) << 8)
    if packed >= 2**15:
        packed -= 2**16
    return packed


def resolve_tensor_ref(
    ref: str,
    *,
    input_name: str,
    input_tensors: Sequence[torch.Tensor],
    captured_outputs: Mapping[str, list[torch.Tensor]],
) -> list[torch.Tensor]:
    if ref == input_name:
        return [tensor.clone() for tensor in input_tensors]
    if ref not in captured_outputs:
        raise GoldenGenerationError(f"Checkpoint graph references unknown tensor source '{ref}'.")
    return [tensor.clone() for tensor in captured_outputs[ref]]


def resolve_golden_inputs(
    *,
    module_id: str,
    op_type: str,
    trace_layers: Sequence[Mapping[str, str]],
    trace_index: int,
    operation_map: Mapping[str, Mapping[str, Any]],
    input_name: str,
    input_tensors: Sequence[torch.Tensor],
    captured_outputs: Mapping[str, list[torch.Tensor]],
    input_width_bits: int,
) -> list[list[int]]:
    operation = operation_map.get(module_id)
    if operation is None:
        if op_type == "add":
            raise GoldenGenerationError(
                f"Layer '{module_id}' requires graph wiring metadata to build packed add inputs."
            )
        if trace_index == 0:
            return pack_tensor_vectors_to_bus_words(
                input_tensors,
                input_width_bits,
                context=f"{module_id}.goldin",
            )
        return pack_tensor_vectors_to_bus_words(
            captured_outputs[trace_layers[trace_index - 1]["module_id"]],
            input_width_bits,
            context=f"{module_id}.goldin",
        )

    if op_type == "add":
        lhs = require_string(operation, "lhs", f"operation '{module_id}'")
        rhs = require_string(operation, "rhs", f"operation '{module_id}'")
        lhs_tensors = resolve_tensor_ref(
            lhs,
            input_name=input_name,
            input_tensors=input_tensors,
            captured_outputs=captured_outputs,
        )
        rhs_tensors = resolve_tensor_ref(
            rhs,
            input_name=input_name,
            input_tensors=input_tensors,
            captured_outputs=captured_outputs,
        )
        return pack_paired_tensors_to_bus_words(
            lhs_tensors,
            rhs_tensors,
            input_width_bits,
            context=f"{module_id}.goldin",
        )

    input_ref = require_string(operation, "input", f"operation '{module_id}'")
    return pack_tensor_vectors_to_bus_words(
        resolve_tensor_ref(
            input_ref,
            input_name=input_name,
            input_tensors=input_tensors,
            captured_outputs=captured_outputs,
        ),
        input_width_bits,
        context=f"{module_id}.goldin",
    )


def resolve_layer_parameters(
    metadata: Mapping[str, Any],
    traced_model: fx.GraphModule | None = None,
    module_id: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    weight_shape = coerce_shape(metadata["weight_shape"], "weight_shape")
    # `weight_int8` / `bias_int32` is the v2 real-PTQ schema emitted by
    # scripts/quantize_impl.py. The legacy keys (`weights`, `weight`,
    # `conv_weight`, `bias`, `conv_bias`) are kept for the toy v1 checkpoint
    # and for hand-crafted test fixtures.
    weight_key = next(
        (key for key in ("weight_int8", "weights", "weight", "conv_weight") if key in metadata),
        None,
    )
    bias_key = next(
        (key for key in ("bias_int32", "bias", "conv_bias") if key in metadata),
        None,
    )

    if weight_key is not None:
        weight_tensor = tensor_from_numeric_values(metadata[weight_key], weight_shape, weight_key)
        bias_tensor = None
        if bias_key is not None:
            bias_tensor = tensor_from_numeric_values(metadata[bias_key], [weight_shape[0]], bias_key)
        return weight_tensor, bias_tensor

    if traced_model is None or module_id is None:
        raise GoldenGenerationError(
            "Checkpoint layer metadata is missing serialized weights and no traced module was provided."
        )

    module = get_module_by_path(traced_model, module_id)
    raw_weight = getattr(module, "weight", None)
    if not isinstance(raw_weight, torch.Tensor):
        raise GoldenGenerationError(f"Layer '{module_id}' does not expose a tensor weight.")
    raw_bias = getattr(module, "bias", None)
    bias_tensor = raw_bias.detach().to(torch.float32) if isinstance(raw_bias, torch.Tensor) else None
    return raw_weight.detach().to(torch.float32).reshape(tuple(weight_shape)), bias_tensor


def fold_batch_norm_from_metadata(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    batch_norm: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch_norm, Mapping):
        raise GoldenGenerationError("batch_norm metadata must be a dict.")
    out_channels = weight.shape[0]
    return fold_batch_norm_into_conv(
        weight,
        bias,
        tensor_from_numeric_values(batch_norm.get("weight"), [out_channels], "batch_norm.weight"),
        tensor_from_numeric_values(batch_norm.get("bias"), [out_channels], "batch_norm.bias"),
        tensor_from_numeric_values(batch_norm.get("running_mean"), [out_channels], "batch_norm.running_mean"),
        tensor_from_numeric_values(batch_norm.get("running_var"), [out_channels], "batch_norm.running_var"),
        float(batch_norm.get("eps", 1e-5)),
    )


def require_layer_mapping(checkpoint: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_layers = checkpoint.get("layers")
    if not isinstance(raw_layers, Mapping) or not raw_layers:
        raise GoldenGenerationError("Checkpoint v2 must contain a non-empty 'layers' mapping.")
    layers: dict[str, dict[str, Any]] = {}
    for module_id, metadata in raw_layers.items():
        if not isinstance(module_id, str):
            raise GoldenGenerationError("Checkpoint layer keys must be strings.")
        if not isinstance(metadata, Mapping):
            raise GoldenGenerationError(f"Checkpoint layer '{module_id}' must map to a dict.")
        for field in ("input_shape", "output_shape", "weight_shape", "num_weights", "scale_factor", "zero_point"):
            if field not in metadata:
                raise GoldenGenerationError(
                    f"Checkpoint layer '{module_id}' is missing required field '{field}'."
                )
        layers[module_id] = dict(metadata)
    return layers


def resolve_functional_module_id(
    node: fx.Node,
    op_type: str,
    layers: Mapping[str, Mapping[str, Any]],
    used_module_ids: set[str],
) -> str:
    if node.name in layers and node.name not in used_module_ids:
        return node.name

    for module_id, metadata in layers.items():
        if module_id in used_module_ids:
            continue
        if metadata.get("fx_node_name") == node.name:
            return module_id

    candidates = [
        module_id
        for module_id, metadata in layers.items()
        if module_id not in used_module_ids and metadata.get("op_type") == op_type
    ]
    if len(candidates) == 1:
        return candidates[0]

    raise GoldenGenerationError(
        f"Unable to map fx node '{node.name}' ({op_type}) to a unique checkpoint layer key."
    )


def register_module_path(root: nn.Module, module_path: str, module: nn.Module) -> None:
    parts = module_path.split(".")
    container: nn.Module = root
    for part in parts[:-1]:
        child = container._modules.get(part)
        if child is None:
            child = nn.Module()
            container.add_module(part, child)
        container = child
    container.add_module(parts[-1], module)


def get_module_by_path(root: nn.Module, module_path: str) -> nn.Module:
    current: nn.Module = root
    for part in module_path.split("."):
        current = getattr(current, part)
    return current


def tensor_from_numeric_values(value: Any, shape: Sequence[int], field_name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(torch.float32)
    elif isinstance(value, list):
        expected_length = 1
        for dimension in shape:
            expected_length *= int(dimension)
        if len(value) != expected_length:
            raise GoldenGenerationError(
                f"Field '{field_name}' must contain {expected_length} values, got {len(value)}."
            )
        tensor = torch.tensor(value, dtype=torch.float32)
    else:
        raise GoldenGenerationError(f"Field '{field_name}' must be a tensor or numeric list.")
    return tensor.reshape(tuple(int(dimension) for dimension in shape))


def has_serialized_weight_values(metadata: Mapping[str, Any]) -> bool:
    return any(key in metadata for key in ("weights", "weight", "conv_weight"))


def require_string(mapping: Mapping[str, Any], field: str, context: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise GoldenGenerationError(f"{context} field '{field}' must be a non-empty string.")
    return value


def require_op_type(mapping: Mapping[str, Any], context: str) -> str:
    value = mapping.get("op_type")
    if value not in SUPPORTED_OP_TYPES:
        raise GoldenGenerationError(
            f"{context} field 'op_type' must be one of {sorted(SUPPORTED_OP_TYPES)}, got {value!r}."
        )
    return str(value)


def coerce_shape(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise GoldenGenerationError(f"Field '{field_name}' must be a non-empty list.")
    if not all(isinstance(item, int) and item > 0 for item in value):
        raise GoldenGenerationError(f"Field '{field_name}' must contain positive integers only.")
    return [int(item) for item in value]


def coerce_int_sequence(value: Any, field_name: str, *, allow_zero: bool = False) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value or not all(isinstance(item, int) for item in value):
        raise GoldenGenerationError(f"Field '{field_name}' must be a non-empty integer sequence.")
    min_value = 0 if allow_zero else 1
    if not all(int(item) >= min_value for item in value):
        comparator = "non-negative" if allow_zero else "positive"
        raise GoldenGenerationError(f"Field '{field_name}' must contain {comparator} integers only.")
    return [int(item) for item in value]


def coerce_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int):
        raise GoldenGenerationError(f"Field '{field_name}' must be an integer.")
    return int(value)


def coerce_nonnegative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise GoldenGenerationError(f"Field '{field_name}' must be a non-negative integer.")
    return int(value)


def is_absolute_posix_path(path_value: str) -> bool:
    # Must use forward slashes only (POSIX formatting).
    if "\\" in path_value:
        return False
    # Native POSIX absolute (`/home/...`).
    if PurePosixPath(path_value).is_absolute():
        return True
    # Windows drive-rooted path emitted by Path.resolve().as_posix() on
    # Windows (e.g. `C:/Users/...`). The downstream TypeScript code uses
    # path.isAbsolute() which treats these as absolute on win32, and the
    # existing smoke fixtures use this exact form — accept it here so the
    # Python validator matches the rest of the pipeline.
    if (
        len(path_value) >= 3
        and path_value[1] == ":"
        and path_value[2] == "/"
        and path_value[0].isalpha()
    ):
        return True
    return False
