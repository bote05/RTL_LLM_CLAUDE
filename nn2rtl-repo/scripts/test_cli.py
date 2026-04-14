from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def run_script(tmp_path: Path, script_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["NN2RTL_REPO_ROOT"] = str(tmp_path)
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
    result = run_script(tmp_path, "quantize_model.py")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["checkpoint_path"].endswith("checkpoints/resnet50_int8.pth")
    assert payload["layers"]["toy_conv1x1"]["scale_factor"] == 0.125
    assert (tmp_path / "checkpoints" / "resnet50_int8.pth").exists()


@pytest.mark.full
def test_generate_golden_cli_writes_pipeline_ir_and_weight_artifacts(tmp_path: Path) -> None:
    quantize_result = run_script(tmp_path, "quantize_model.py")
    assert quantize_result.returncode == 0, quantize_result.stderr

    result = run_script(tmp_path, "generate_golden.py", "checkpoints/resnet50_int8.pth")

    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "output" / "golden_vectors.json").read_text(encoding="utf8"))
    layer = payload["layers"][0]
    assert json.loads(result.stdout)["status"] == "ok"
    assert layer["ready_in_signal"] == "ready_in"
    assert Path(layer["weights_path"]).exists()
    assert Path(layer["bias_path"]).exists()


@pytest.mark.full
def test_generate_golden_cli_fails_meaningfully_for_missing_checkpoint(tmp_path: Path) -> None:
    result = run_script(tmp_path, "generate_golden.py", "checkpoints/missing.pth")

    assert result.returncode != 0
    assert "Checkpoint not found" in result.stderr
