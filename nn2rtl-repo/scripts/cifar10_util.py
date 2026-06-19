#!/usr/bin/env python3
"""CIFAR-10 calibration / golden-vector feed for the nn2rtl ONNX frontend
(MLPerf Tiny ResNet-8).  NEW module: drop-in replacement for the
imagenet_util API surface the frontend uses (load_batch; iter_images and
count_rows mirrored for completeness).  It is injected via
``sys.modules['imagenet_util'] = cifar10_util`` by
scripts/generate_golden_resnet8.py BEFORE scripts.onnx_frontend is imported,
so onnx_frontend's NN2RTL_IMAGENET_CALIB branch (``import imagenet_util as
_iu; _iu.load_batch(N)``) transparently receives CIFAR-10 data.

Feed layout returned by load_batch(N) -- the frontend turns the FIRST
NN2RTL_GOLDEN_VECTORS calibration feeds into the golden vectors, so:

  rows 0..G-1 (G = NN2RTL_GOLDEN_VECTORS, default 8):
      CIFAR-10 TEST images 0..G-1 (test_batch order) -> goldens come from
      the test set with known labels (see golden_head_labels()).
  rows G..N-1:
      the OFFICIAL MLPerf Tiny calibration samples, in
      calibration_samples_idxs.npy order.  NOTE: upstream
      mlcommons/tiny model_converter.py provably indexes these into the
      TEST set (representative_dataset_generator yields test_data[i]),
      not the train set -- max index 9950 < 10000 confirms.  Using them
      identically mirrors the reference INT8 TFLite calibration.

Pixel format: raw 0..255 float32 (NO /255, NO mean/std), NCHW [N,3,32,32] --
the exact upstream MLPerf Tiny preprocessing (negatives=False) modulo the
NHWC->NCHW transpose (checkpoints/resnet8.onnx is an NCHW export).  CIFAR
python batches store each row as 1024 R + 1024 G + 1024 B bytes, so
reshape(N,3,32,32) is already CHW.

With this feed, calibrate_onnx observes the true input range (max_abs 255)
and the frontend's input_scale becomes 255/127 ~= 2.008 (the "real
calibration" marker), instead of ~1.0 from synthetic INT8 noise.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Iterator

import numpy as np

CIFAR_DIR = Path(os.environ.get(
    "NN2RTL_CIFAR10_DIR",
    r"D:/RTL_LLM_CLAUDE/rq2_resnet8/data/cifar-10-batches-py"))
CALIB_IDX_PATH = Path(os.environ.get(
    "NN2RTL_CIFAR10_CALIB_IDX",
    r"D:/RTL_LLM_CLAUDE/rq2_resnet8/scripts/upstream_refs/calibration_samples_idxs.npy"))
GOLDEN_HEAD = int(os.environ.get("NN2RTL_GOLDEN_VECTORS", "8"))

_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _load_test_set() -> tuple[np.ndarray, np.ndarray]:
    """(10000,3,32,32) float32 raw 0..255 NCHW + (10000,) int64 labels."""
    if "test" not in _cache:
        with open(CIFAR_DIR / "test_batch", "rb") as fo:
            d = pickle.load(fo, encoding="bytes")
        x = d[b"data"].reshape((-1, 3, 32, 32)).astype(np.float32)
        y = np.array(d[b"labels"], dtype=np.int64)
        _cache["test"] = (x, y)
    return _cache["test"]


def _feed() -> tuple[np.ndarray, np.ndarray]:
    """Full feed: GOLDEN_HEAD test images then the 500 official calib samples."""
    if "feed" not in _cache:
        x, y = _load_test_set()
        calib_idx = np.load(CALIB_IDX_PATH).astype(np.int64)   # 500 TEST indices
        head_x, head_y = x[:GOLDEN_HEAD], y[:GOLDEN_HEAD]
        tail_x, tail_y = x[calib_idx], y[calib_idx]
        _cache["feed"] = (np.concatenate([head_x, tail_x]).astype(np.float32),
                          np.concatenate([head_y, tail_y]))
    return _cache["feed"]


def load_batch(limit: int, skip: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (N,3,32,32) float32 raw 0..255 + (N,) int64 labels.

    Same signature/semantics as imagenet_util.load_batch (skip discards the
    first `skip` feed rows).  Raises if more rows are requested than the
    GOLDEN_HEAD + 500 official calibration samples provide.
    """
    x, y = _feed()
    if skip + limit > len(x):
        raise ValueError(
            f"load_batch(limit={limit}, skip={skip}) exceeds the available "
            f"{len(x)} rows ({GOLDEN_HEAD} golden-head test images + "
            f"{len(x) - GOLDEN_HEAD} official MLPerf Tiny calibration samples).")
    return x[skip:skip + limit].copy(), y[skip:skip + limit].copy()


def iter_images(limit: int | None = None, skip: int = 0
                ) -> Iterator[tuple[np.ndarray, int]]:
    """Yield (chw_float32[3,32,32], label) across the feed, imagenet_util-style."""
    x, y = _feed()
    end = len(x) if limit is None else min(len(x), skip + limit)
    for i in range(skip, end):
        yield x[i], int(y[i])


def count_rows() -> int:
    return len(_feed()[0])


def golden_head_labels() -> list[int]:
    """Labels of the first GOLDEN_HEAD test images (the golden vectors)."""
    return [int(v) for v in _load_test_set()[1][:GOLDEN_HEAD]]


if __name__ == "__main__":
    x, y = load_batch(256)
    print("feed rows:", count_rows())
    print("batch:", x.shape, x.dtype, "range:", float(x.min()), float(x.max()))
    print("golden head labels:", golden_head_labels())
    print("first calib-tail labels:", y[GOLDEN_HEAD:GOLDEN_HEAD + 8].tolist())
