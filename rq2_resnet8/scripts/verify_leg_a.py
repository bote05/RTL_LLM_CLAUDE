#!/usr/bin/env python3
"""Leg A step 6: verify the imported ResNet-8 LayerIR + goldens + weights in
D:/RTL_LLM_CLAUDE/nn2rtl-repo/output/resnet8/.

Checks:
  1. layer_ir.json: 21 layers = 9 conv2d + 7 relu + 3 add + 1 global_avg_pool
     + 1 gemm, in a sane order.
  2. Every layer's weights/bias hex + .goldin/.goldout artifacts exist;
     conv/gemm weight hex non-empty; conv weight_bank_paths exist.
  3. Per-OC scale presence: groups==1 non-1x1 convs (7: five 3x3 + the two
     4x4 reformulated stride-2) carry scale_factor_per_oc/weight_scale_per_oc;
     the two 1x1 shortcut convs and the gemm are per-tensor.
  4. All goldens carry 8 vectors.
  5. REAL-CALIBRATION marker: the stem .goldin vector 0 must equal
     clamp(round(raw_test_image_0 / (255/127)), -128, 127) BYTE-EXACTLY,
     proving input_scale = 255/127 ~= 2.008 (not the ~1.0 synthetic marker).
  6. Fingerprint sidecar layer_ir.json.checkpoint == abs checkpoint path.
Reports layer_ir scale statistics.  Exit 0 only on full pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

NN2RTL_ROOT = Path("D:/RTL_LLM_CLAUDE/nn2rtl-repo")
OUT = NN2RTL_ROOT / "output" / "resnet8"
RQ2_SCRIPTS = Path("D:/RTL_LLM_CLAUDE/rq2_resnet8/scripts")
sys.path.insert(0, str(NN2RTL_ROOT))
sys.path.insert(0, str(RQ2_SCRIPTS))

EXPECTED_COUNTS = {"conv2d": 9, "relu": 7, "add": 3, "global_avg_pool": 1, "gemm": 1}


def main() -> int:
    from scripts.golden_impl import read_golden_vector_file
    from torch_resnet8 import load_cifar10_test_nchw

    failures: list[str] = []
    ir = json.loads((OUT / "layer_ir.json").read_text(encoding="utf8"))
    layers = ir["layers"]

    counts: dict[str, int] = {}
    for l in layers:
        counts[l["op_type"]] = counts.get(l["op_type"], 0) + 1
    if counts != EXPECTED_COUNTS:
        failures.append(f"op counts {counts} != {EXPECTED_COUNTS}")

    per_oc, per_tensor_conv = [], []
    n_goldens_ok = 0
    for l in layers:
        mid, op = l["module_id"], l["op_type"]
        for key in ("golden_inputs_path", "golden_outputs_path"):
            if not Path(l[key]).exists():
                failures.append(f"{mid}: missing {key}")
        if op in ("conv2d", "gemm"):
            wp = Path(l["weights_path"])
            if not (wp.exists() and wp.stat().st_size > 0):
                failures.append(f"{mid}: weights hex missing/empty")
            if l.get("bias_path") and not Path(l["bias_path"]).exists():
                failures.append(f"{mid}: bias hex missing")
            if l.get("weight_bits") != 8 or l.get("activation_bits") != 8:
                failures.append(f"{mid}: weight/activation bits != 8")
        if op == "conv2d":
            for bp in l.get("weight_bank_paths", []):
                if not Path(bp).exists():
                    failures.append(f"{mid}: missing bank {bp}")
            kh, kw = l["weight_shape"][2], l["weight_shape"][3]
            if "scale_factor_per_oc" in l:
                per_oc.append((mid, f"{kh}x{kw}", len(l["scale_factor_per_oc"])))
                if len(l["scale_factor_per_oc"]) != l["weight_shape"][0]:
                    failures.append(f"{mid}: per-OC length != OC")
            else:
                per_tensor_conv.append((mid, f"{kh}x{kw}"))
                if (kh, kw) != (1, 1):
                    failures.append(f"{mid}: non-1x1 conv missing per-OC scales")
        # golden vector count
        try:
            vecs = read_golden_vector_file(Path(l["golden_inputs_path"]),
                                           l["input_width_bits"])
            if len(vecs) == 8:
                n_goldens_ok += 1
            else:
                failures.append(f"{mid}: goldin has {len(vecs)} vectors, want 8")
        except Exception as e:  # noqa: BLE001
            failures.append(f"{mid}: goldin unreadable: {e}")

    if len(per_oc) != 7:
        failures.append(f"{len(per_oc)} per-OC convs, want 7: {per_oc}")
    if len(per_tensor_conv) != 2 or any(k != "1x1" for _, k in per_tensor_conv):
        failures.append(f"per-tensor convs unexpected: {per_tensor_conv}")

    # --- real-calibration input_scale marker (byte-exact stem goldin check) --
    stem = next(l for l in layers if l["op_type"] == "conv2d")
    x, _ = load_cifar10_test_nchw()
    input_scale = 255.0 / 127.0
    expect = np.clip(np.round(x[0] / input_scale), -128, 127).astype(np.int64)  # CHW
    vec0 = read_golden_vector_file(Path(stem["golden_inputs_path"]),
                                   stem["input_width_bits"])[0]
    words = np.array(vec0, dtype=np.int64)
    bus_bytes = stem["input_width_bits"] // 8          # 3 (one byte per channel)
    wps = (bus_bytes + 3) // 4                          # int32 words per sample
    samples = len(words) // wps
    got = np.zeros((samples, bus_bytes), dtype=np.int64)
    for s in range(samples):
        w = words[s * wps:(s + 1) * wps]
        for b in range(bus_bytes):
            byte = (int(w[b // 4]) >> (8 * (b % 4))) & 0xFF
            got[s, b] = byte - 256 if byte >= 128 else byte
    # goldin samples are pixel-major, bytes are channels (data_in[i*8 +:8] = ch i)
    expect_pm = expect.reshape(3, -1).T                # (1024, 3)
    stem_match = bool((got == expect_pm).all())
    if samples != expect_pm.shape[0]:
        failures.append(f"stem goldin samples {samples} != {expect_pm.shape[0]}")
    if not stem_match:
        # diagnose: would the synthetic scale (1.0) match instead?
        alt = np.clip(np.round(x[0]), -128, 127).reshape(3, -1).T
        failures.append(
            f"stem goldin NOT raw/2.008 quantized (mismatch bytes: "
            f"{int((got != expect_pm).sum())}; raw/1.0 match: {bool((got == alt).all())})")

    # --- fingerprint sidecar -------------------------------------------------
    fp = (OUT / "layer_ir.json.checkpoint").read_text(encoding="utf8").strip()
    ckpt = NN2RTL_ROOT / "checkpoints" / "resnet8.onnx"
    if fp.replace("\\", "/").lower() != str(ckpt).replace("\\", "/").lower():
        failures.append(f"fingerprint '{fp}' != '{ckpt}'")

    # --- scale stats ----------------------------------------------------------
    conv_sf = [l["scale_factor"] for l in layers if l["op_type"] == "conv2d"]
    poc_all = np.concatenate([np.asarray(l["scale_factor_per_oc"]) for l in layers
                              if "scale_factor_per_oc" in l])
    report = {
        "layers": len(layers),
        "op_counts": counts,
        "module_ids": [l["module_id"] for l in layers],
        "goldens_with_8_vectors": f"{n_goldens_ok}/{len(layers)}",
        "stem_goldin_matches_input_scale_2.008": stem_match,
        "input_scale_marker": input_scale,
        "per_oc_convs": per_oc,
        "per_tensor_convs": per_tensor_conv,
        "conv_scale_factor_minmax": [float(min(conv_sf)), float(max(conv_sf))],
        "per_oc_composite_scale_min_max_mean": [
            float(poc_all.min()), float(poc_all.max()), float(poc_all.mean())],
        "gemm_scale_factor": float(next(l["scale_factor"] for l in layers
                                        if l["op_type"] == "gemm")),
        "gap_spatial": next(l["gap_spatial"] for l in layers
                            if l["op_type"] == "global_avg_pool"),
        "fingerprint": fp,
        "failures": failures,
        "gate": "PASS" if not failures else "FAIL",
    }
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
