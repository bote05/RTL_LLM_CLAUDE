# Leg A status — MLPerf Tiny ResNet-8 (CIFAR-10) into the nn2rtl pipeline

Date: 2026-06-12. Scope: steps 1-3 of Leg A — PyTorch port, parity gate,
ONNX export, CIFAR calibration shim, IR+golden import, verification.
NO RTL generation, NO Vivado, NO training. All gates PASS.

## Result summary

| Gate | Result |
|---|---|
| Torch port top-1, FULL 10k test set | **8719/10000 = 87.19%** (== reference) |
| Argmax agreement torch vs ort(resnet8_folded.onnx), 10k | **10000/10000** |
| max abs(softmax(torch logits) − ort probs), 10k | **2.95e-06** |
| Export-variant vs ref (10k) | argmax 10000/10000, max abs logit diff **0.0** (bit-exact) |
| Exported resnet8.onnx re-score (ort vs torch, 1000 imgs) | argmax **1000/1000** (ort top-1 881/1000) |
| layer_ir.json | **21 layers**: 9 conv2d + 7 relu + 3 add + 1 global_avg_pool + 1 gemm |
| Goldens | 21/21 modules have .goldin/.goldout with **8 vectors** each |
| input_scale (real-calibration marker) | **255/127 = 2.0079** — stem .goldin vector 0 is byte-exactly `clamp(round(raw/2.0079))` (NOT the ≈1.0 synthetic marker) |
| INT8 golden chain (node_linear.goldout argmax, 8 vecs) | `[3,8,8,8,6,6,1,6]` = torch float preds 8/8 (labels 7/8 — same miss as float) |
| nn2rtl-repo existing files | **zero modified** (git: only new untracked files) |
| vivado.exe pid 6004 | untouched, still running |

## Files created

In `D:/RTL_LLM_CLAUDE/rq2_resnet8/`:
- `scripts/torch_resnet8.py` — BN-free PyTorch ResNet-8 loaded directly from
  `model/resnet8_folded.onnx` initializers (Conv OIHW as-is; MatMul [64,10] →
  Linear via transpose; bias from the trailing Add). Pads/strides read from
  each Conv node's attributes; residual wiring recovered + asserted from
  graph edges (block 1: identity skip; blocks 2-3: 1x1-stride-2 conv skip).
  Builds two variants (see "asymmetric padding" below): `build_ref()` and
  `build_export()`. Ends at 10 logits (no Softmax); AdaptiveAvgPool2d(1).
- `scripts/parity_gate.py` — the 10k parity gate (table above).
- `scripts/export_to_nn2rtl.py` — drives nn2rtl's own
  `scripts.onnx_frontend.export_pytorch_to_onnx` (input_shape=(1,3,32,32)),
  verifies the exported op set + symmetric pads + 1000-image ort re-score.
- `scripts/verify_leg_a.py` — step-6 verification gate (layer counts,
  artifacts, per-OC presence, 8-vector goldens, byte-exact input-scale
  marker, fingerprint sidecar, scale stats).
- `LEG_A_STATUS.md` — this file.

In `D:/RTL_LLM_CLAUDE/nn2rtl-repo/` (NEW files only):
- `checkpoints/resnet8.onnx` — torch 2.12 dynamo export, opset 18 (the repo
  helper requests 17; torch auto-raises). Graph: 9 Conv / 7 Relu / 3 Add /
  ReduceMean(axes=[-1,-2],keepdims=1 → frontend maps to GlobalAveragePool) /
  Gemm(transB=1, alpha=beta=1) + a Shape→Concat→Reshape dynamic-batch chain
  whose outputs feed ONLY the Reshape shape input (Reshape is an
  alias-passthrough in the frontend; the skipped Shape/Concat never become
  spec inputs). No Softmax/Transpose/BN/Pad/MatMul. Input `input`
  [batch,3,32,32] float32 raw 0..255 NCHW.
- `scripts/cifar10_util.py` — CIFAR-10 stand-in for the `imagenet_util` API
  (`load_batch(limit, skip)` → (N,3,32,32) float32 raw 0..255 + labels;
  `iter_images`/`count_rows` mirrored). Paths overridable via
  `NN2RTL_CIFAR10_DIR` / `NN2RTL_CIFAR10_CALIB_IDX`.
- `scripts/generate_golden_resnet8.py` — launcher that seeds
  `sys.modules['imagenet_util'] = scripts.cifar10_util` BEFORE importing
  `scripts.onnx_frontend` (the frontend does a plain `import imagenet_util`
  at step 4; sys.modules wins), then drives the exact generate_golden.py
  ONNX code path (`build_pipeline_ir_from_onnx` +
  `generate_golden._write_layer_ir_json`) and writes the registration
  fingerprint + provenance (below).
- `output/resnet8/**` — `layer_ir.json` (+ legacy `golden_vectors.json`),
  `layer_ir.json.checkpoint`, `golden_labels.json`, `weights/*.hex`
  (per-module weights/bias + conv `*_weights_bank*.hex`), `goldens/*.goldin|
  .goldout`. Total ≈ 4.2 MB.

## Quantization configuration (as imported)

- `NN2RTL_WEIGHT_BITS=8` → INT8 weights, INT8 activations; `USE_GPTQ` is
  auto-OFF at 8 bits (it gates on `WEIGHT_BITS < 8`).
- `NN2RTL_STEM_PER_CHANNEL=1` (repo default, kept): per-output-channel
  weight scales for ALL groups==1 non-1x1 convs → 7 convs per-OC
  (node_conv2d, _1, _2, _5, _8 [3x3]; _4, _7 [4x4 reformulated]), each with
  `scale_factor_per_oc` + `weight_scale_per_oc` of length OC.
- `NN2RTL_PW_PER_CHANNEL=0` (repo default, KEPT and documented): the two
  1x1 stride-2 shortcut convs (node_conv2d_3, node_conv2d_6) are
  per-TENSOR. The repo keeps this default OFF because enabling it broke
  byte-exact e2e on the engine path (see onnx_frontend.py "improvement E"
  note); ResNet-8's shortcuts are tiny (16→32, 32→64 1x1) so the accuracy
  exposure is small. Gemm is per-tensor by construction.
- `NN2RTL_IMAGENET_CALIB=256`, `NN2RTL_GOLDEN_VECTORS=8`.
- Scale stats: per-OC composite scales in [4.56e-4, 5.28e-3] (mean 2.24e-3);
  conv per-tensor `scale_factor` field min/max [4.90e-3, 31.6] (the large
  values are the per-OC layers' legacy composite field; the per-OC arrays
  are what the RTL scale ROMs consume); gemm composite 6.07e-3;
  gap_spatial [8,8].

## Calibration + golden provenance (IMPORTANT deviation, documented)

The task brief assumed `calibration_samples_idxs.npy` indexes the TRAIN set.
Upstream evidence says otherwise: `scripts/upstream_refs/model_converter.py`
(mlcommons/tiny reference) feeds `test_data[i]` for those indices
(`representative_dataset_generator`), and the 500 indices are in [18, 9950]
(< 10000). **The official MLPerf Tiny INT8 calibration set is drawn from the
CIFAR-10 TEST set.** Leg A mirrors the official scheme exactly rather than
inventing a train-set variant:

- Feed rows 0..7 = test images 0..7 (raw 0..255) — these become the 8 golden
  vectors (the frontend takes the FIRST `NN2RTL_GOLDEN_VECTORS` calibration
  feeds as goldens). Labels [3,8,8,0,6,6,1,6] recorded in
  `output/resnet8/golden_labels.json`.
- Feed rows 8..255 = the first 248 entries of the official
  `calibration_samples_idxs.npy`, in order.
- All 256 feeds contribute to activation-range stats (max-abs / 127).
  Consequence: calibration uses test-set images — identical in kind to the
  upstream reference flow, but worth stating in the thesis text. The 8
  golden images necessarily also appear in the calibration stats (the
  frontend derives goldens from calibration feeds by design).

## Asymmetric padding: the one model transformation (exact)

`resnet8_folded.onnx`'s two stride-2 3x3 convs (conv2d_3→node_conv2d_4,
conv2d_6→node_conv2d_7 main-path convs) carry keras-'same' ASYMMETRIC pads
[t,l,b,r]=[0,0,1,1]. The nn2rtl frontend hard-rejects asymmetric Conv pads
(onnx_frontend.py `_extract_conv`) and does not support Pad nodes (graph-
completeness hard-fail), so neither literal form can pass. Fix used:
**zero-embedded kernel enlargement** — embed the 3x3 kernel in a 4x4 kernel
(zero first row + first col) with symmetric padding=1, stride 2:

    W'[:, :, 1:4, 1:4] = W;  out = floor((32+2−4)/2)+1 = 16  (unchanged)

Every output pixel reads the identical receptive window (the extra kernel
row/col multiplies only zeros/padding), so the reformulation is
mathematically exact — and empirically bit-exact in float32 on all 10k test
images (max logit diff 0.0 vs the F.pad reference). Cost: those two convs
carry 16/9 ≈ 1.78× weights (still tiny) and Leg B will see K=4 windows.

## Registration (how it was made to land)

`sdk/main.ts import_network` does three things: (1) upsert `networks.json`,
(2) shell `scripts/generate_golden.py` via the `PYTHON` env binary,
(3) write the `<output>/layer_ir.json.checkpoint` fingerprint sidecar that
`ensureLayerIr` (sdk/orchestrate.ts) requires. Step (1) modifies an existing
repo file (forbidden for this leg) and step (2) cannot inject the CIFAR shim
without editing generate_golden.py (a `.bat` PYTHON shim fails under Node's
execFile on Windows). Per the brief's allowance, steps (2)+(3) were
replicated exactly instead: `generate_golden_resnet8.py` runs the identical
code path and writes the fingerprint (absolute checkpoint path;
`pathFingerprintKey` is separator/case-insensitive — verified format).
Deferred to the user (one command, when modifying networks.json is
acceptable — it will regenerate via plain generate_golden.py UNLESS run with
--no-prepare, so prefer):

    cd D:/RTL_LLM_CLAUDE/nn2rtl-repo
    npx tsx sdk/main.ts import_network --id resnet8 --checkpoint checkpoints/resnet8.onnx --model-name resnet8 --no-prepare

(--no-prepare keeps the CIFAR-calibrated IR; the readiness report +
contract_id annotation pass then runs against the existing layer_ir.json.)

## Layer inventory (layer_ir.json order)

| module_id | op | shape notes |
|---|---|---|
| node_conv2d | conv2d 3x3 s1 p1 | 3→16, 32x32, per-OC(16) |
| node_relu | relu | |
| node_conv2d_1 | conv2d 3x3 s1 p1 | 16→16, per-OC(16) |
| node_relu_1 | relu | |
| node_conv2d_2 | conv2d 3x3 s1 p1 | 16→16, per-OC(16) |
| node_add_25 | add | identity skip |
| node_relu_2 | relu | |
| node_conv2d_3 | conv2d 1x1 s2 | 16→32 shortcut, per-TENSOR |
| node_conv2d_4 | conv2d 4x4 s2 p1 | 16→32 (reformulated 3x3 asym), per-OC(32) |
| node_relu_3 | relu | |
| node_conv2d_5 | conv2d 3x3 s1 p1 | 32→32, per-OC(32) |
| node_add_56 | add | |
| node_relu_4 | relu | |
| node_conv2d_6 | conv2d 1x1 s2 | 32→64 shortcut, per-TENSOR |
| node_conv2d_7 | conv2d 4x4 s2 p1 | 32→64 (reformulated), per-OC(64) |
| node_relu_5 | relu | |
| node_conv2d_8 | conv2d 3x3 s1 p1 | 64→64, per-OC(64) |
| node_add_87 | add | |
| node_relu_6 | relu | |
| node_mean | global_avg_pool | gap_spatial [8,8] |
| node_linear | gemm | 64→10, in 512b out 80b |

## Repro commands

    cd D:/RTL_LLM_CLAUDE/rq2_resnet8/scripts
    python torch_resnet8.py                         # port smoke + wiring assert
    python parity_gate.py                           # 10k gate (PASS required)
    PYTHONUTF8=1 python export_to_nn2rtl.py         # export + graph + 1000-img gate
    cd D:/RTL_LLM_CLAUDE/nn2rtl-repo
    PYTHONUTF8=1 python scripts/generate_golden_resnet8.py   # IR+goldens+fingerprint
    cd D:/RTL_LLM_CLAUDE/rq2_resnet8/scripts && PYTHONUTF8=1 python verify_leg_a.py

(Windows python = C:\Python313, torch 2.12.0+cu126 / ort 1.23.0 / onnx 1.21.0.
PYTHONUTF8=1 needed: the torch dynamo exporter prints a U+2705 that crashes a
cp1252 console. Peak RAM well under 2 GB; Vivado route untouched throughout.)

## Known caveats for Leg B

- Two convs have K=4 (even kernel) windows; line-buffer datapath must handle
  K=4 (window/pad logic is K-generic in the patterns, but verify).
- The exported graph's Shape/Concat/Reshape chain is skipped harmlessly by
  the frontend (verified end-to-end by the successful 21-layer extraction).
- `quantization` field in layer_ir.json says `int8_symmetric_per_tensor`
  (historic label) while 7/9 convs are per-OC — same convention as the
  ResNet-50/MBV2 imports.
- opset 18 in the ONNX file (torch floor for dynamo exports), vs 17 for the
  older repo checkpoints; onnxruntime + the frontend handle it fine.
