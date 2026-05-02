"""Export the full torchvision ResNet-50 to ONNX.

The legacy `quantize_model.py` flow is hard-wired to ResNet-50 Layer 1 (stem
+ 3 bottlenecks). To exercise the rest of the pipeline — in particular the
tiled-streaming contract path that needs >512-channel layers to fire — we
need every layer in the graph, not just Layer 1.

Output:
    checkpoints/resnet50_full.onnx
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torchvision


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / "checkpoints" / "resnet50_full.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Feature-extractor mode: the nn2rtl ONNX frontend supports
    # Conv / Relu / MaxPool / Add / Clip(min=0). torchvision's ResNet-50
    # forward also runs `avgpool → flatten → fc`, none of which the
    # frontend handles. Replacing those modules with `nn.Identity()` does
    # not help because the explicit `torch.flatten(x, 1)` in the parent
    # `forward` survives as a `Flatten` ONNX node and the Identity wraps
    # become `Identity` ops — both unsupported. The clean fix is a thin
    # wrapper module whose `forward` ends at `layer4`'s last activation.
    import torch.nn as nn

    base = torchvision.models.resnet50(weights="DEFAULT").eval()

    class ResNet50FeatureExtractor(nn.Module):
        """ResNet-50 stem + 4 stages, no classifier head.

        Output tensor is the post-relu activation of the last residual block
        in `layer4`, shape `[N, 2048, 7, 7]` for the standard 224x224 input.
        That is exactly the boundary the nn2rtl ONNX frontend can extract.
        """

        def __init__(self, base: torchvision.models.ResNet) -> None:
            super().__init__()
            self.conv1 = base.conv1
            self.bn1 = base.bn1
            self.relu = base.relu
            self.maxpool = base.maxpool
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            return x

    model = ResNet50FeatureExtractor(base).eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Wrote {out_path}")
    print(f"Size: {out_path.stat().st_size / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
