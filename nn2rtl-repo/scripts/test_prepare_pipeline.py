from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import prepare_pipeline as prepare_pipeline_module


REPO_ROOT = Path(__file__).resolve().parent.parent


def run_prepare_pipeline(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["NN2RTL_REPO_ROOT"] = str(tmp_path)
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "prepare_pipeline.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


@pytest.mark.full
def test_prepare_pipeline_cli_runs_full_frontend_and_prints_table(tmp_path: Path) -> None:
    result = run_prepare_pipeline(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "module_id" in result.stdout
    assert "pipeline_latency_cycles" in result.stdout

    layer_ir_path = tmp_path / "output" / "layer_ir.json"
    payload = json.loads(layer_ir_path.read_text(encoding="utf8"))
    layer = payload["layers"][0]

    assert (tmp_path / "checkpoints" / "resnet50_int8.pth").exists()
    assert layer["module_id"] in result.stdout
    assert "conv2d" in result.stdout
    assert layer["op_type"] == "conv2d"
    assert Path(layer["weights_path"]).exists()
    assert Path(layer["bias_path"]).exists()


@pytest.mark.full
def test_prepare_pipeline_fails_when_checkpoint_disappears_between_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_run_python_script = prepare_pipeline_module.run_python_script

    def run_and_remove_checkpoint(
        source_repo_root: Path,
        runtime_repo_root: Path,
        script_name: str,
    ) -> subprocess.CompletedProcess[str]:
        result = original_run_python_script(source_repo_root, runtime_repo_root, script_name)
        if script_name == "quantize_model.py":
            (runtime_repo_root / "checkpoints" / "resnet50_int8.pth").unlink()
        return result

    monkeypatch.setattr(prepare_pipeline_module, "run_python_script", run_and_remove_checkpoint)

    with pytest.raises(prepare_pipeline_module.PreparePipelineError, match="Checkpoint not found"):
        prepare_pipeline_module.prepare_pipeline(runtime_repo_root=tmp_path)


@pytest.mark.full
def test_validate_pipeline_ir_rejects_schema_invalid_output(tmp_path: Path) -> None:
    prepare_pipeline_module.prepare_pipeline(runtime_repo_root=tmp_path)
    layer_ir_path = tmp_path / "output" / "layer_ir.json"
    payload = json.loads(layer_ir_path.read_text(encoding="utf8"))
    payload["layers"][0]["clock_signal"] = "clock"
    layer_ir_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")

    with pytest.raises(
        prepare_pipeline_module.PreparePipelineError,
        match="PipelineIR schema validation failed",
    ):
        prepare_pipeline_module.validate_pipeline_ir(REPO_ROOT, layer_ir_path)
