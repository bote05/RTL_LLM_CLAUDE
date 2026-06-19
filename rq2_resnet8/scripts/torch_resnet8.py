#!/usr/bin/env python3
"""PyTorch port of the MLPerf Tiny ResNet-8 (CIFAR-10) from resnet8_folded.onnx.

Builds two functionally identical BN-free NCHW models, both ending at the 10
LOGITS (no Softmax):

* build_ref()    -- ResNet8 with _RefConv: every Conv applies its ONNX node's
                    LITERAL pads (possibly ASYMMETRIC, e.g. [0,0,1,1] from the
                    keras 'same' stride-2 export) via F.pad, then a padding-0
                    Conv2d.  This is the exact replica used for the 10k parity
                    gate against onnxruntime.
* build_export() -- ResNet8 with _export_conv: symmetric-pad-only convs so the
                    ONNX re-export contains NO Pad nodes and NO asymmetric Conv
                    pads (both are hard-rejected by nn2rtl's onnx_frontend).
                    Asymmetric-pad convs are reformulated EXACTLY by embedding
                    the KxK kernel into an enlarged kernel zero-filled on the
                    short-pad side(s), with symmetric padding:
                        pads (t,l,b,r), P_h=max(t,b), P_w=max(l,r)
                        K'_h = K_h + (P_h-t) + (P_h-b);  K'_w analogous
                        W'[:, :, P_h-t : P_h-t+K_h, P_w-l : P_w-l+K_w] = W
                    Every output pixel then reads the identical receptive
                    window (the extra kernel rows/cols multiply only zeros),
                    so the reformulation is mathematically exact.
                    For ResNet-8: the two 3x3 stride-2 convs with pads
                    [0,0,1,1] become 4x4 stride-2 padding-1 convs.

Weights are loaded DIRECTLY from the ONNX initializers:
  - Conv weights are already OIHW float32 (BN pre-folded upstream), used as-is.
  - The head MatMul weight [64,10] becomes Linear.weight via transpose [10,64].
  - The head bias comes from the trailing Add initializer.
Pads/strides are read from each Conv node's actual attributes -- never assumed.
The residual-add wiring is recovered from the graph edges and asserted
(block 1: identity skip; blocks 2-3: 1x1 stride-2 conv shortcut).
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
DEFAULT_ONNX = HERE.parent / "model" / "resnet8_folded.onnx"
DEFAULT_CIFAR_DIR = HERE.parent / "data" / "cifar-10-batches-py"


# ---------------------------------------------------------------------------
# ONNX parsing
# ---------------------------------------------------------------------------

def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.INTS:
                return [int(v) for v in a.ints]
            if a.type == onnx.AttributeProto.INT:
                return int(a.i)
            if a.type == onnx.AttributeProto.FLOAT:
                return float(a.f)
    return default


def _only(nodes, op_type):
    matches = [n for n in nodes if n.op_type == op_type]
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one {op_type} consumer, got {[n.op_type for n in nodes]}")
    return matches[0]


def parse_resnet8(onnx_path: Path = DEFAULT_ONNX) -> dict:
    """Trace resnet8_folded.onnx and return weights + per-conv attrs + wiring.

    Returns {"stem": conv, "blocks": [b1,b2,b3], "dense_w": [64,10],
    "dense_b": [10]} where conv = {"name","weight","bias","pads","strides"}
    (pads in ONNX order [t,l,b,r]) and block = {"c1","c2","sc"} (sc=None for
    the identity-skip block).  Raises AssertionError if the graph does not
    have the expected ResNet-8 topology.
    """
    model = onnx.load(str(onnx_path))
    g = model.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}

    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    def conv_params(node):
        assert node.op_type == "Conv", node.op_type
        w = np.asarray(inits[node.input[1]], dtype=np.float32)
        b = np.asarray(inits[node.input[2]], dtype=np.float32)
        pads = _attr(node, "pads", [0, 0, 0, 0])
        strides = _attr(node, "strides", [1, 1])
        assert _attr(node, "dilations", [1, 1]) == [1, 1], node.name
        assert _attr(node, "group", 1) == 1, node.name
        ks = _attr(node, "kernel_shape", list(w.shape[2:]))
        assert ks == list(w.shape[2:]), (node.name, ks, w.shape)
        return {"name": node.name, "weight": w, "bias": b,
                "pads": [int(p) for p in pads], "strides": [int(s) for s in strides]}

    # --- input -> Transpose(NHWC->NCHW) -> stem Conv -> Relu ---------------
    real_inputs = [i for i in g.input if i.name not in inits]
    assert len(real_inputs) == 1, [i.name for i in real_inputs]
    tr = _only(consumers[real_inputs[0].name], "Transpose")
    assert _attr(tr, "perm") == [0, 3, 1, 2], _attr(tr, "perm")
    stem_node = _only(consumers[tr.output[0]], "Conv")
    stem_relu = _only(consumers[stem_node.output[0]], "Relu")

    def trace_block(block_in: str):
        cons = consumers[block_in]
        convs = [n for n in cons if n.op_type == "Conv"]
        adds = [n for n in cons if n.op_type == "Add"]
        if len(convs) == 1 and len(adds) == 1:        # identity-skip block
            c1, sc, skip_tensor = convs[0], None, block_in
        else:                                          # conv-shortcut block
            assert len(convs) == 2 and not adds, [n.op_type for n in cons]
            c1 = _only(convs, "Conv") if len(convs) == 1 else \
                next(n for n in convs if _attr(n, "kernel_shape") == [3, 3])
            sc = next(n for n in convs if _attr(n, "kernel_shape") == [1, 1])
            assert sc is not c1
            skip_tensor = sc.output[0]
        r1 = _only(consumers[c1.output[0]], "Relu")
        c2 = _only(consumers[r1.output[0]], "Conv")
        add_node = _only(consumers[c2.output[0]], "Add")
        assert set(add_node.input) == {skip_tensor, c2.output[0]}, \
            (add_node.name, list(add_node.input))
        r_out = _only(consumers[add_node.output[0]], "Relu")
        return ({"c1": conv_params(c1), "c2": conv_params(c2),
                 "sc": conv_params(sc) if sc is not None else None},
                r_out.output[0])

    blocks = []
    t = stem_relu.output[0]
    for _ in range(3):
        bp, t = trace_block(t)
        blocks.append(bp)

    # --- head: AveragePool(8x8 == global) -> Reshape -> MatMul -> Add(bias)
    ap = _only(consumers[t], "AveragePool")
    assert _attr(ap, "kernel_shape") == [8, 8] and _attr(ap, "strides") == [8, 8]
    rs = _only(consumers[ap.output[0]], "Reshape")
    mm = _only(consumers[rs.output[0]], "MatMul")
    dense_w = np.asarray(inits[mm.input[1]], dtype=np.float32)        # [64, 10]
    badd = _only(consumers[mm.output[0]], "Add")
    bias_name = badd.input[1] if badd.input[1] in inits else badd.input[0]
    dense_b = np.asarray(inits[bias_name], dtype=np.float32)          # [10]
    assert dense_w.shape == (64, 10) and dense_b.shape == (10,)
    sm = _only(consumers[badd.output[0]], "Softmax")
    assert sm.output[0] in {o.name for o in g.output}

    return {"stem": conv_params(stem_node), "blocks": blocks,
            "dense_w": dense_w, "dense_b": dense_b}


# ---------------------------------------------------------------------------
# Conv constructors
# ---------------------------------------------------------------------------

class _RefConv(nn.Module):
    """Exact ONNX Conv replica: literal (possibly asymmetric) pads via F.pad."""

    def __init__(self, p: dict):
        super().__init__()
        w = torch.from_numpy(np.array(p["weight"]))
        o, i, kh, kw = w.shape
        self.pads = tuple(p["pads"])                  # ONNX order: t, l, b, r
        self.conv = nn.Conv2d(i, o, (kh, kw), stride=tuple(p["strides"]),
                              padding=0, bias=True)
        with torch.no_grad():
            self.conv.weight.copy_(w)
            self.conv.bias.copy_(torch.from_numpy(np.array(p["bias"])))

    def forward(self, x):
        t, l, b, r = self.pads
        if t or l or b or r:
            x = F.pad(x, (l, r, t, b))                # F.pad order: l, r, t, b
        return self.conv(x)


def _export_conv(p: dict) -> nn.Conv2d:
    """Symmetric-padding-only Conv2d; asymmetric pads zero-embedded (exact)."""
    w = np.array(p["weight"])
    o, i, kh, kw = w.shape
    t, l, b, r = p["pads"]
    if t == b and l == r:                             # already symmetric
        conv = nn.Conv2d(i, o, (kh, kw), stride=tuple(p["strides"]),
                         padding=(t, l), bias=True)
        wt = torch.from_numpy(w)
    else:                                             # zero-embed enlargement
        ph, pw = max(t, b), max(l, r)
        nkh = kh + (ph - t) + (ph - b)
        nkw = kw + (pw - l) + (pw - r)
        wn = np.zeros((o, i, nkh, nkw), dtype=np.float32)
        wn[:, :, ph - t:ph - t + kh, pw - l:pw - l + kw] = w
        conv = nn.Conv2d(i, o, (nkh, nkw), stride=tuple(p["strides"]),
                         padding=(ph, pw), bias=True)
        wt = torch.from_numpy(wn)
    with torch.no_grad():
        conv.weight.copy_(wt)
        conv.bias.copy_(torch.from_numpy(np.array(p["bias"])))
    return conv


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _Block(nn.Module):
    def __init__(self, bp: dict, make_conv):
        super().__init__()
        self.c1 = make_conv(bp["c1"])
        self.c2 = make_conv(bp["c2"])
        self.sc = make_conv(bp["sc"]) if bp["sc"] is not None else None

    def forward(self, x):
        skip = self.sc(x) if self.sc is not None else x
        y = F.relu(self.c1(x))
        y = self.c2(y)
        return F.relu(skip + y)


class ResNet8(nn.Module):
    """BN-free ResNet-8, NCHW [N,3,32,32] raw 0..255 in, 10 logits out."""

    def __init__(self, params: dict, make_conv):
        super().__init__()
        self.stem = make_conv(params["stem"])
        self.blocks = nn.ModuleList(_Block(bp, make_conv) for bp in params["blocks"])
        self.pool = nn.AdaptiveAvgPool2d(1)           # exports as global avg pool
        self.fc = nn.Linear(params["dense_w"].shape[0], params["dense_w"].shape[1])
        with torch.no_grad():
            self.fc.weight.copy_(torch.from_numpy(np.array(params["dense_w"]).T))
            self.fc.bias.copy_(torch.from_numpy(np.array(params["dense_b"])))

    def forward(self, x):
        x = F.relu(self.stem(x))
        for blk in self.blocks:
            x = blk(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)                             # logits, NO softmax


def build_ref(onnx_path: Path = DEFAULT_ONNX) -> ResNet8:
    return ResNet8(parse_resnet8(onnx_path), lambda p: _RefConv(p)).eval()


def build_export(onnx_path: Path = DEFAULT_ONNX) -> ResNet8:
    return ResNet8(parse_resnet8(onnx_path), _export_conv).eval()


# ---------------------------------------------------------------------------
# CIFAR-10 loading (raw 0..255, NCHW)
# ---------------------------------------------------------------------------

def load_cifar10_test_nchw(data_dir: Path | str | None = None):
    """(10000,3,32,32) float32 raw 0..255 NCHW + (10000,) int64 labels.

    CIFAR python batches store rows as 1024 R + 1024 G + 1024 B, so
    reshape(N,3,32,32) IS the CHW layout (same values eval_cifar10.py rolls
    to NHWC for the NHWC reference model).
    """
    data_dir = Path(data_dir or os.environ.get("NN2RTL_CIFAR10_DIR", DEFAULT_CIFAR_DIR))
    with open(data_dir / "test_batch", "rb") as fo:
        d = pickle.load(fo, encoding="bytes")
    x = d[b"data"].reshape((-1, 3, 32, 32)).astype(np.float32)
    y = np.array(d[b"labels"], dtype=np.int64)
    return x, y


if __name__ == "__main__":
    params = parse_resnet8()
    convs = [params["stem"]] + [c for b in params["blocks"]
                                for c in (b["c1"], b["c2"], b["sc"]) if c is not None]
    for c in convs:
        print(f"{c['name']:<32} w{list(c['weight'].shape)} strides={c['strides']} "
              f"pads(t,l,b,r)={c['pads']}")
    ref, exp = build_ref(), build_export()
    x, y = load_cifar10_test_nchw()
    xb = torch.from_numpy(x[:8])
    with torch.no_grad():
        lr, le = ref(xb), exp(xb)
    print("ref argmax  :", lr.argmax(1).tolist(), "labels:", y[:8].tolist())
    print("ref==export max|dlogit|:", float((lr - le).abs().max()))
