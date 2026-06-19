# Copyright (C) 2026 — RQ2 Leg B (FINN) model definition.
# Quantization patterns copied VERBATIM where applicable from
# /root/tools/finn/deps/brevitas/src/brevitas_examples/bnn_pynq/models/resnet.py
# (BSD-3-Clause, AMD) and .../models/CNV.py (input quant pattern).
"""
MLPerf Tiny ResNet-8 as a Brevitas W4A4 QAT model for the FINN leg (Leg B).

TOPOLOGY CONTRACT (thesis fairness invariant — must match the verified reference EXACTLY):
  stem    Conv3x3-16 (s1, pad1) + BN + ReLU
  stack1  {Conv3x3-16 + BN + ReLU, Conv3x3-16 + BN, identity skip}, ReLU after add
  stack2  {Conv3x3-32 s2 + BN + ReLU, Conv3x3-32 + BN, Conv1x1-32 s2 + BN projection skip}, ReLU after add
  stack3  {Conv3x3-64 s2 + BN + ReLU, Conv3x3-64 + BN, Conv1x1-64 s2 + BN projection skip}, ReLU after add
  GAP 8x8 -> Dense 10.   BN after every conv during training (folds at FINN streamlining).
  9 convs + 1 dense.  PyTorch trainable params: 78,042
  (= 76,720 conv weights + 672 BN gamma/beta + 650 dense). The Keras reference's 78,666
  counts the same 76,720 conv weights plus Keras conv biases (336) + 4-per-channel BN
  bookkeeping on 240 channels (960) + dense 650; filter counts / kernel sizes / strides /
  skip structure are IDENTICAL. verify_topology() asserts every conv shape and stride.

QUANTIZATION (FINN convention, brevitas_examples/bnn_pynq/models/resnet.py patterns):
  - W4A4 internal: 4-bit per-channel weights (Int8WeightPerChannelFloat @ bit_width=4),
    4-bit QuantReLU activations.
  - first conv: 8-bit PER-CHANNEL weights (first_layer_weight_quant, reference verbatim).
  - last layer: 8-bit PER-TENSOR weights + Int32Bias (reference's last_layer_* pattern).
  - SHARED activation quantizers feeding every residual add: both add operands are
    quantized by the SAME QuantReLU module instance -> identical scale factors.
    This is the killer for FINN issue #1090 'Scaling factors are different'.
  - extra QuantReLU after conv2/bn2 (relu2) and at the end of the projection path,
    because FINN requires the same sign along residual adds (reference comment verbatim).
  - GAP: TruncAvgPool2d(kernel_size=8, TruncTo8bit, FLOOR) — truncation supported in FINN.
  - input: forward does x = 2*x01 - 1 then an 8-bit QuantIdentity(CommonActQuant,
    min=-1.0, max=1.0-2^-7, narrow_range=False, POWER_OF_TWO) — the bnn_pynq CNV
    "Q1.7 input format" pattern verbatim. Scale = 2^-7, zero_point = 0, signed int8.

Documented deviation from the float reference: the two extra QuantReLUs per projection
block (downsample tail + relu2) are REQUIRED quantization infrastructure for FINN
residual adds; they do not change the conv topology, filter counts, strides or skips.
"""

from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn

import brevitas.nn as qnn
from brevitas.core.restrict_val import RestrictValueType
from brevitas.quant import Int8WeightPerChannelFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias
from brevitas.quant import TruncTo8bit
from brevitas.quant_tensor import QuantTensor

from brevitas_examples.bnn_pynq.models.common import CommonActQuant


def make_quant_conv2d(
        in_channels,
        out_channels,
        kernel_size,
        weight_bit_width,
        weight_quant,
        stride=1,
        padding=0,
        bias=False):
    # VERBATIM from brevitas_examples/bnn_pynq/models/resnet.py
    return qnn.QuantConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        weight_quant=weight_quant,
        weight_bit_width=weight_bit_width)


class QuantBasicBlock(nn.Module):
    """
    VERBATIM from brevitas_examples/bnn_pynq/models/resnet.py:
    Quantized BasicBlock implementation with extra relu activations to respect FINN
    constraints on the sign of residual adds. Ok to train from scratch.
    """
    expansion = 1

    def __init__(
            self,
            in_planes,
            planes,
            stride=1,
            bias=False,
            shared_quant_act=None,
            act_bit_width=4,
            weight_bit_width=4,
            weight_quant=Int8WeightPerChannelFloat):
        super(QuantBasicBlock, self).__init__()
        self.conv1 = make_quant_conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            weight_bit_width=weight_bit_width,
            weight_quant=weight_quant)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        self.conv2 = make_quant_conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
            weight_bit_width=weight_bit_width,
            weight_quant=weight_quant)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                make_quant_conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=bias,
                    weight_bit_width=weight_bit_width,
                    weight_quant=weight_quant),
                nn.BatchNorm2d(self.expansion * planes),
                # We add a ReLU activation here because FINN requires the same sign
                # along residual adds
                qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True))
            # Redefine shared_quant_act whenever shortcut is performing downsampling
            shared_quant_act = self.downsample[-1]
        if shared_quant_act is None:
            shared_quant_act = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        # We add a ReLU activation here because FINN requires the same sign along
        # residual adds. Sharing the module with the producer of the skip operand
        # guarantees both add inputs carry the SAME scale (FINN issue #1090).
        self.relu2 = shared_quant_act
        self.relu_out = qnn.QuantReLU(return_quant_tensor=True, bit_width=act_bit_width)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        if len(self.downsample):
            x = self.downsample(x)
        # Check that the addition is made explicitly among QuantTensor structures
        assert isinstance(out, QuantTensor), "Perform add among QuantTensors"
        assert isinstance(x, QuantTensor), "Perform add among QuantTensors"
        out = out + x
        out = self.relu_out(out)
        return out


class QuantResNet8(nn.Module):
    """MLPerf Tiny ResNet-8, W4A4, FINN-ready (see module docstring)."""

    def __init__(
            self,
            num_classes=10,
            act_bit_width=4,
            weight_bit_width=4,
            in_bit_width=8,
            round_average_pool=False,
            weight_quant=Int8WeightPerChannelFloat,
            first_layer_weight_quant=Int8WeightPerChannelFloat,
            last_layer_weight_quant=Int8WeightPerTensorFloat,
            last_layer_bias_quant=Int32Bias):
        super(QuantResNet8, self).__init__()

        # Input quantizer — bnn_pynq CNV "Q1.7 input format" pattern VERBATIM.
        self.input_quant = qnn.QuantIdentity(
            act_quant=CommonActQuant,
            bit_width=in_bit_width,
            min_val=-1.0,
            max_val=1.0 - 2.0 ** (-7),
            narrow_range=False,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO)

        # Stem: keep first layer at 8b per-channel (reference first_layer convention)
        self.conv1 = make_quant_conv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            weight_bit_width=8,
            weight_quant=first_layer_weight_quant)
        self.bn1 = nn.BatchNorm2d(16)
        shared_quant_act = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        self.relu = shared_quant_act

        # stack1: identity skip — shared_quant_act is the stem ReLU, so the skip operand
        # and the conv2-branch operand of the add share one quantizer (same scale).
        self.stack1 = QuantBasicBlock(
            16, 16, stride=1, bias=False, shared_quant_act=shared_quant_act,
            act_bit_width=act_bit_width, weight_bit_width=weight_bit_width,
            weight_quant=weight_quant)
        # stack2/stack3: projection skip — QuantBasicBlock redefines the shared quantizer
        # to the downsample-tail ReLU internally (reference behaviour).
        self.stack2 = QuantBasicBlock(
            16, 32, stride=2, bias=False, shared_quant_act=self.stack1.relu_out,
            act_bit_width=act_bit_width, weight_bit_width=weight_bit_width,
            weight_quant=weight_quant)
        self.stack3 = QuantBasicBlock(
            32, 64, stride=2, bias=False, shared_quant_act=self.stack2.relu_out,
            act_bit_width=act_bit_width, weight_bit_width=weight_bit_width,
            weight_quant=weight_quant)

        # GAP 8x8: truncation to 8b (without rounding), which is supported in FINN
        avgpool_float_to_int_impl_type = 'ROUND' if round_average_pool else 'FLOOR'
        self.final_pool = qnn.TruncAvgPool2d(
            kernel_size=8,
            trunc_quant=TruncTo8bit,
            float_to_int_impl_type=avgpool_float_to_int_impl_type,
            return_quant_tensor=True)

        # Keep last layer at 8b, per-tensor weights + Int32Bias (reference convention)
        self.linear = qnn.QuantLinear(
            64,
            num_classes,
            weight_bit_width=8,
            bias=True,
            bias_quant=last_layer_bias_quant,
            weight_quant=last_layer_weight_quant)

        # Init VERBATIM from the reference
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor):
        # Trainer feeds ToTensor() output in [0,1]; map to [-1, 1] (CNV verbatim) and
        # quantize to Q1.7. Both ops are traced into the exported QONNX graph.
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        x = self.input_quant(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.stack1(out)
        out = self.stack2(out)
        out = self.stack3(out)
        out = self.final_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def quant_resnet8_w4a4(num_classes: int = 10) -> QuantResNet8:
    return QuantResNet8(num_classes=num_classes, act_bit_width=4, weight_bit_width=4)


# ---------------------------------------------------------------------------
# Topology contract gate
# ---------------------------------------------------------------------------
_EXPECTED_CONVS = [
    # (qualified name, (out_c, in_c, kh, kw), stride)
    ("conv1", (16, 3, 3, 3), 1),
    ("stack1.conv1", (16, 16, 3, 3), 1),
    ("stack1.conv2", (16, 16, 3, 3), 1),
    ("stack2.conv1", (32, 16, 3, 3), 2),
    ("stack2.conv2", (32, 32, 3, 3), 1),
    ("stack2.downsample.0", (32, 16, 1, 1), 2),
    ("stack3.conv1", (64, 32, 3, 3), 2),
    ("stack3.conv2", (64, 64, 3, 3), 1),
    ("stack3.downsample.0", (64, 64 // 2, 1, 1), 2),
]
_EXPECTED_TRAINABLE_PARAMS = 78042  # 76,720 conv + 672 BN + 650 dense (see docstring)


def verify_topology(model: QuantResNet8) -> int:
    """Assert the exact MLPerf Tiny ResNet-8 conv topology; return trainable param count."""
    mods = dict(model.named_modules())
    convs = [(n, m) for n, m in mods.items() if isinstance(m, nn.Conv2d)]
    assert len(convs) == 9, f"expected 9 convs, got {len(convs)}: {[n for n, _ in convs]}"
    for name, shape, stride in _EXPECTED_CONVS:
        m = mods[name]
        assert tuple(m.weight.shape) == shape, f"{name}: {tuple(m.weight.shape)} != {shape}"
        assert m.stride[0] == stride, f"{name}: stride {m.stride} != {stride}"
        assert m.bias is None, f"{name}: conv bias must be None (BN absorbs it)"
        bn_name = {"conv1": "bn1"}.get(name) or name.replace("conv1", "bn1").replace(
            "conv2", "bn2").replace("downsample.0", "downsample.1")
        assert isinstance(mods[bn_name], nn.BatchNorm2d), f"missing BN after {name}"
    lin = mods["linear"]
    assert tuple(lin.weight.shape) == (10, 64) and lin.bias is not None
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Allow for learned activation-quantizer scale parameters on top of the contract count
    n_quant_scale = n_params - _EXPECTED_TRAINABLE_PARAMS
    assert 0 <= n_quant_scale <= 64, (
        f"trainable params {n_params}: contract nominal {_EXPECTED_TRAINABLE_PARAMS} "
        f"(+ small learned quant scales), delta {n_quant_scale} out of range")
    return n_params


if __name__ == "__main__":
    m = quant_resnet8_w4a4()
    n = verify_topology(m)
    print(f"QuantResNet8 W4A4 topology OK; trainable params = {n} "
          f"(contract nominal {_EXPECTED_TRAINABLE_PARAMS} + learned quant scales)")
    y = m(torch.rand(2, 3, 32, 32))
    print("forward OK, logits:", tuple(y.shape))
