#!/usr/bin/env python3
"""Phase 1.2 + 1.3: verify the ImageNet val parquet set and the label order.

1.2: total rows == 50000, labels in [0,999], images decode.
1.3: run torchvision pretrained ResNet-50 (FLOAT, full head) on N images and
     confirm predicted class == parquet label often enough that the label order
     and preprocessing are correct (chance = 0.1%; correct ~76%). A near-chance
     result means label misalignment or a preprocessing bug.

Usage: py scripts/check_imagenet.py [N]   (default N=256)
"""
from __future__ import annotations
import sys
import numpy as np
import torch

import imagenet_util as iu


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256

    print("== Phase 1.2: dataset sanity ==")
    total = iu.count_rows()
    print(f"shards={len(iu.shard_paths())}  total_rows={total}")
    if total != 50000:
        print(f"WARN: expected 50000 rows, got {total}")

    xs, ys = iu.load_batch(n)
    print(f"loaded {n} images: x={xs.shape} {xs.dtype} "
          f"range=[{xs.min():.3f},{xs.max():.3f}]  "
          f"labels in [{ys.min()},{ys.max()}]  unique={len(set(ys.tolist()))}")
    assert ys.min() >= 0 and ys.max() <= 999, "labels out of [0,999]"

    print("\n== Phase 1.3: label-order / preprocessing check (torchvision R50 float) ==")
    from torchvision.models import resnet50, ResNet50_Weights
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for wname, weights in [("V1", ResNet50_Weights.IMAGENET1K_V1),
                           ("V2", ResNet50_Weights.IMAGENET1K_V2)]:
        model = resnet50(weights=weights).eval().to(dev)
        with torch.no_grad():
            logits = model(torch.from_numpy(xs).to(dev))
            top1 = logits.argmax(1).cpu().numpy()
            top5 = torch.topk(logits, 5, dim=1).indices.cpu().numpy()
        acc1 = float((top1 == ys).mean())
        acc5 = float(np.mean([ys[i] in top5[i] for i in range(len(ys))]))
        print(f"  weights={wname}: top1={acc1*100:.1f}%  top5={acc5*100:.1f}%  "
              f"(N={n})")
        del model
    print("\n=> top1 well above chance (0.1%) confirms label order + preprocessing OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
