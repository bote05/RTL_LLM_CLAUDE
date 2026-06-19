# RQ2 ResNet-8 Acquisition Record

Date: 2026-06-12. Workspace: `D:/RTL_LLM_CLAUDE/rq2_resnet8/` (new files only; nn2rtl-repo untouched).

## 1. Provenance

- Source repo: https://github.com/mlcommons/tiny — `benchmark/training/image_classification/`
- Pinned master commit at fetch time: `1afd2c9820f795965a6134facd0b4dfae41ef23f` (2026-05-12T16:20:13Z)
- Pretrained model: `trained_models/pretrainedResnet.h5` (training curves in repo dated 2020-12-18)
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz (server throttled ~2.5 MB/min
  per connection; fetched via 16 parallel HTTP range requests, then reassembled; checksum-verified)

### Architecture caveat (important)

Current-master `keras_model.py` defaults to `resnet_v1_eembc(conv_filters=26)` and contains later
EfficientNetV2S/distillation additions. The **pretrained h5 predates this** and embeds its own
architecture: the original ResNet-8 with **16/32/64 filters, 78,666 params (78,186 trainable)**.
The h5 was loaded directly (`tf.keras.models.load_model`), so the keras_model.py drift is
irrelevant to the converted ONNX. Conv stack confirmed from the loaded model:

| layer | k | stride | filters |
|---|---|---|---|
| conv2d (stem) | 3x3 | 1 | 16 |
| conv2d_1, conv2d_2 (stack1) | 3x3 | 1 | 16 |
| conv2d_3 (stack2) | 3x3 | 2 | 32 |
| conv2d_4 (stack2) | 3x3 | 1 | 32 |
| conv2d_5 (stack2 shortcut) | 1x1 | 2 | 32 |
| conv2d_6 (stack3) | 3x3 | 2 | 64 |
| conv2d_7 (stack3) | 3x3 | 1 | 64 |
| conv2d_8 (stack3 shortcut) | 1x1 | 2 | 64 |

then AveragePooling2D(8) -> Flatten -> Dense(10, softmax). 7 BatchNorm layers in the h5
(stem + 6 stack convs; the two 1x1 shortcut convs have no BN).

## 2. Files

| file | bytes | sha256 |
|---|---|---|
| model/pretrainedResnet.h5 | 1,116,712 | 5f938a8eea605438cc7360aa13c987bc6d7bd0c8d931b2917361ae7ae4eeda2f |
| model/keras_model.py | 11,448 | 52a7564de25828abba7aa4aa785f1355513ad86079bad1ef9f26167073791f45 |
| model/resnet8_ref.onnx | 316,217 | b3996cbf8959644aa600b11627a8c49141eb4467306d0a01dc3e077ff5fd3c27 |
| model/resnet8_folded.onnx | 316,217 | b3996cbf8959644aa600b11627a8c49141eb4467306d0a01dc3e077ff5fd3c27 (byte-identical to ref, see 3) |
| data/cifar-10-python.tar.gz | 170,498,071 | 6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce — md5 c58f30108f718f92721af3b95e74349a (matches canonical) |
| scripts/upstream_refs/train.py | 6,228 | 4c238829dc8f89961fb136d184565693350dc17ed9b3466b279a564f02d9c592 |
| scripts/upstream_refs/test.py | 2,270 | 741e74c9c51823cf55f03e3de2e5c97e1ce169bde663e4919f4bb97913193aca |
| scripts/upstream_refs/perf_samples_idxs.npy | 1,728 | 3bd4a88eeb4c50fad652d0f24c8af13bc9219ba2878aea47c6536bfbeb43024d |

Also kept for reference (scripts/upstream_refs/): eval_functions_eembc.py, README.md,
requirements.txt, model_converter.py. CIFAR-10 extracted to `data/cifar-10-batches-py/`
(data_batch_1..5, test_batch, batches.meta).

## 3. Conversion (h5 -> ONNX)

Environment: WSL Ubuntu venv `/root/rq2_venv` —
tensorflow-cpu 2.15.1 / keras 2.15.0, tf2onnx 1.16.1, onnx 1.16.2, onnxruntime 1.23.2,
numpy 1.26.4, onnxoptimizer 0.4.2, protobuf 3.20.3. (TF pinned to 2.15: tf2onnx is
incompatible with Keras 3 / TF>=2.16.)

- `scripts/convert_h5_to_onnx.py` — tf2onnx `convert.from_keras`, **opset 13**, input
  signature (None,32,32,3) float32 -> `model/resnet8_ref.onnx`.
- **tf2onnx already folded all 7 BatchNorm layers into the Conv weights during conversion**:
  `resnet8_ref.onnx` contains **0 BatchNormalization nodes** (verified by node count).
- `scripts/fold_bn.py` — `onnxoptimizer fuse_bn_into_conv` pass (manual numpy fold as
  fallback, not needed) -> `model/resnet8_folded.onnx`. The pass was a no-op; the saved file
  is **byte-identical** to the ref (same sha256). Parity check ref vs folded on 64 random
  raw-pixel inputs: max |diff| = 0.0, argmax agreement 64/64.
- Keras vs ONNX parity on 16 random raw-pixel inputs: max |diff| = 1.37e-06, argmax 16/16.

### ONNX node inventory (identical for both variants)

| op | count |
|---|---|
| Conv | 9 |
| Relu | 7 |
| Add | 4 (3 residual adds + 1 Dense-bias add; Dense exported as MatMul+Add, not Gemm) |
| MatMul | 1 |
| AveragePool | 1 |
| Softmax | 1 |
| Transpose | 1 (input NHWC->NCHW) |
| Reshape | 1 (flatten) |
| BatchNormalization | **0** |

Note vs the "8 Conv" expectation: the graph genuinely has **9 Conv nodes** — 7 main-path
convs + 2 downsample 1x1 shortcut convs. The "ResNet-8" name counts 8 *weight layers*
(stem + 6 stack convs + Dense), excluding the shortcut convs.

### IO tensors

- Input: `input`, float32, [N, 32, 32, 3] (NHWC, batch dim symbolic `unk__101`)
- Output: `dense`, float32, [N, 10] — softmax probabilities (Softmax is inside the graph)

## 4. Preprocessing (exact upstream replication)

From `train.py:load_cifar_10_data` (negatives=False path, lines 108-119) as consumed by
`test.py` (`model.evaluate` directly on the returned arrays):

```
x = test_batch[b'data']            # uint8 (10000, 3072), row-major R,G,B planes
x = x.reshape(N, 3, 32, 32)
x = np.rollaxis(x, 1, 4)           # -> NHWC (N, 32, 32, 3), still uint8
x = x.astype(np.float32)           # RAW 0..255  — NO /255, NO mean/std anywhere
```

The model consumes **raw pixel values 0..255 as float32**. The training
`ImageDataGenerator` has no `rescale`/featurewise options enabled (augmentation only), and
no normalization exists anywhere in train.py/test.py. Labels: `test_batch[b'labels']`,
top-1 = argmax over the 10 outputs (same as upstream `eval_functions_eembc.calculate_accuracy`).

## 5. Accuracy validation

Full CIFAR-10 test set, all 10,000 images, preprocessing as above
(`scripts/eval_cifar10.py`, ORT CPUExecutionProvider / TF-CPU):

| model | full 10,000 top-1 | official perf-200 subset |
|---|---|---|
| pretrainedResnet.h5 (TF 2.15) | **0.8719 (87.19%)** | 0.8700 (87.00%) |
| resnet8_ref.onnx (ORT) | **0.8719 (87.19%)** | 0.8700 (87.00%) |
| resnet8_folded.onnx (ORT) | **0.8719 (87.19%)** | 0.8700 (87.00%) |

All three agree exactly (identical 8719/10000 correct). 87.19% sits at the top of the
expected 85-87% band and matches the MLPerf Tiny float reference (the published ~85% closed
figure is the int8 .tflite, not the float model).

## 6. Reproduce

```
# in WSL Ubuntu (venv /root/rq2_venv already provisioned)
cd /mnt/d/RTL_LLM_CLAUDE/rq2_resnet8
/root/rq2_venv/bin/python scripts/convert_h5_to_onnx.py     # h5 -> resnet8_ref.onnx (opset 13)
/root/rq2_venv/bin/python scripts/fold_bn.py                # -> resnet8_folded.onnx + parity
/root/rq2_venv/bin/python scripts/eval_cifar10.py --h5 \
    --onnx model/resnet8_ref.onnx --onnx model/resnet8_folded.onnx
```
