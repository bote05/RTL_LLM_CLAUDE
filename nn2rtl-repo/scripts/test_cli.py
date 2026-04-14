from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def make_fake_torchvision_package(root: Path) -> Path:
    package_root = root / "fake_site"
    models_dir = package_root / "torchvision" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    (package_root / "torchvision" / "__init__.py").write_text(
        "from . import models\n__all__ = ['models']\n__version__ = '0.fake'\n",
        encoding="utf8",
    )
    (models_dir / "__init__.py").write_text(
        textwrap.dedent(
            """
            import torch


            class FakeBottleneck(torch.nn.Module):
                def __init__(self, in_channels, bottleneck_channels, out_channels, use_downsample):
                    super().__init__()
                    self.conv1 = torch.nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
                    self.bn1 = torch.nn.BatchNorm2d(bottleneck_channels)
                    self.conv2 = torch.nn.Conv2d(
                        bottleneck_channels,
                        bottleneck_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    )
                    self.bn2 = torch.nn.BatchNorm2d(bottleneck_channels)
                    self.conv3 = torch.nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
                    self.bn3 = torch.nn.BatchNorm2d(out_channels)
                    self.relu = torch.nn.ReLU(inplace=False)
                    if use_downsample:
                        self.downsample = torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                            torch.nn.BatchNorm2d(out_channels),
                        )
                    else:
                        self.downsample = None


            class FakeResNet50(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1, bias=False)
                    self.bn1 = torch.nn.BatchNorm2d(4)
                    self.relu = torch.nn.ReLU(inplace=False)
                    self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                    self.layer1 = torch.nn.Sequential(
                        FakeBottleneck(4, 2, 4, True),
                        FakeBottleneck(4, 2, 4, False),
                        FakeBottleneck(4, 2, 4, False),
                    )


            class ResNet50_Weights:
                DEFAULT = object()


            def resnet50(*, weights):
                if weights is not ResNet50_Weights.DEFAULT:
                    raise ValueError("Expected ResNet50_Weights.DEFAULT")
                return FakeResNet50()
            """
        ).strip()
        + "\n",
        encoding="utf8",
    )
    return package_root


def run_script(
    tmp_path: Path,
    script_name: str,
    *args: str,
    pythonpath: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["NN2RTL_REPO_ROOT"] = str(tmp_path)
    if pythonpath is not None:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(pythonpath) if not existing else f"{pythonpath}{os.pathsep}{existing}"
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script_name), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


@pytest.mark.full
def test_quantize_model_cli_writes_a_real_checkpoint_and_summary(tmp_path: Path) -> None:
    fake_torchvision = make_fake_torchvision_package(tmp_path)
    result = run_script(tmp_path, "quantize_model.py", pythonpath=fake_torchvision)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert Path(payload["checkpoint_path"]).as_posix().endswith(
        "checkpoints/resnet50_int8.pth"
    )
    assert payload["export_scope"] == "stem_plus_layer1"
    assert "synthetic tensors" in payload["notes"][0]
    assert len(payload["layers"]) == 16
    assert payload["layers"]["layer0_0_conv1"]["op_type"] == "conv2d"
    assert (tmp_path / "checkpoints" / "resnet50_int8.pth").exists()


@pytest.mark.full
def test_generate_golden_cli_rejects_flat_v2_checkpoint_from_real_ptq(tmp_path: Path) -> None:
    # scripts/quantize_model.py currently writes a flat format_version=2
    # checkpoint (a `layers` dict only, no nn.Module or residual_stack_spec).
    # scripts/generate_golden.py cannot fx-trace that, so it must fail loudly
    # with the documented error rather than quietly emitting empty
    # golden_inputs/golden_outputs (the old behavior was a silent false-pass
    # on the downstream Assayer stage). This test locks in the loud-failure
    # contract until Prompt 1 is extended to emit a traceable model spec.
    # See ARCHITECTURE.md "PTQ ↔ fx golden capture wiring" for the gap.
    fake_torchvision = make_fake_torchvision_package(tmp_path)
    quantize_result = run_script(tmp_path, "quantize_model.py", pythonpath=fake_torchvision)
    assert quantize_result.returncode == 0, quantize_result.stderr

    result = run_script(
        tmp_path,
        "generate_golden.py",
        "checkpoints/resnet50_int8.pth",
        pythonpath=fake_torchvision,
    )

    assert result.returncode != 0
    assert "lacks a traceable model spec" in result.stderr


@pytest.mark.full
def test_generate_golden_cli_fails_meaningfully_for_missing_checkpoint(tmp_path: Path) -> None:
    result = run_script(tmp_path, "generate_golden.py", "checkpoints/missing.pth")

    assert result.returncode != 0
    assert "Checkpoint not found" in result.stderr or "No such file or directory" in result.stderr
