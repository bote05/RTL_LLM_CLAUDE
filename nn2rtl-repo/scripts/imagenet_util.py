#!/usr/bin/env python3
"""ImageNet validation-set utilities for nn2rtl calibration + accuracy.

Reads the local parquet shards (image=JPEG bytes + int label) at
C:\\Users\\User\\Desktop\\RTL_LLM_CLAUDE\\imagenet-val\\data\\validation-*.parquet,
decodes + preprocesses with the standard torchvision eval transform
(resize 256 -> center-crop 224 -> normalize ImageNet mean/std), and yields
float CHW tensors + labels.

Used by:
  - the INT4 ImageNet calibration feed (onnx_frontend, ~256-1024 images)
  - the Phase 4 accuracy sweep (all 50k)
  - the Phase 1.3 label-order / preprocessing sanity check
"""
from __future__ import annotations
import io
import os
from pathlib import Path
from typing import Iterator

import numpy as np

# Data migrated Desktop -> D: ; allow env override, default to the D: location.
IMAGENET_DIR = Path(os.environ.get("NN2RTL_IMAGENET_DIR", r"D:\RTL_LLM_CLAUDE\imagenet-val\data"))
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def shard_paths() -> list[Path]:
    paths = sorted(IMAGENET_DIR.glob("validation-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no validation parquet shards in {IMAGENET_DIR}")
    return paths


def _preprocess_pil(img) -> np.ndarray:
    """PIL.Image -> float32 CHW [3,224,224], resize256+centercrop224+normalize."""
    from PIL import Image
    img = img.convert("RGB")
    # resize shorter side to 256 (bilinear), matching torchvision Resize(256)
    w, h = img.size
    if w < h:
        nw, nh = 256, int(round(256 * h / w))
    else:
        nw, nh = int(round(256 * w / h)), 256
    img = img.resize((nw, nh), Image.BILINEAR)
    # center crop 224
    left = (nw - 224) // 2
    top = (nh - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC [0,1]
    arr = (arr - MEAN) / STD
    return np.transpose(arr, (2, 0, 1)).copy()  # CHW


def iter_images(limit: int | None = None, skip: int = 0,
                shards: list[Path] | None = None
                ) -> Iterator[tuple[np.ndarray, int]]:
    """Yield (chw_float32[3,224,224], label) across shards, in file order.
    `skip` discards the first `skip` rows (to separate calib vs val sets)."""
    import pyarrow.parquet as pq
    from PIL import Image
    seen = 0
    n = 0
    for sp in (shards or shard_paths()):
        tbl = pq.read_table(sp, columns=["image", "label"])
        imgs = tbl.column("image").to_pylist()
        labels = tbl.column("label").to_pylist()
        for rec, lab in zip(imgs, labels):
            if seen < skip:
                seen += 1
                continue
            b = rec["bytes"] if isinstance(rec, dict) else rec
            pil = Image.open(io.BytesIO(b))
            yield _preprocess_pil(pil), int(lab)
            n += 1
            if limit is not None and n >= limit:
                return


def load_batch(limit: int, skip: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (N,3,224,224) float32 + (N,) int labels, skipping the first `skip`."""
    xs, ys = [], []
    for x, y in iter_images(limit=limit, skip=skip):
        xs.append(x); ys.append(y)
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)


def count_rows() -> int:
    import pyarrow.parquet as pq
    return sum(pq.ParquetFile(sp).metadata.num_rows for sp in shard_paths())


if __name__ == "__main__":
    print("shards:", len(shard_paths()))
    print("total rows:", count_rows())
    x, y = load_batch(4)
    print("batch:", x.shape, x.dtype, "labels:", y.tolist(),
          "x range:", round(float(x.min()), 3), round(float(x.max()), 3))
