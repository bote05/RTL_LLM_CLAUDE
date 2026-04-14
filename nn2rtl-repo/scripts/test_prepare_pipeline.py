from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import prepare_pipeline as prepare_pipeline_module
from scripts.test_cli import make_fake_torchvision_package


REPO_ROOT = Path(__file__).resolve().parent.parent


def run_prepare_pipeline(
    tmp_path: Path,
    *,
    pythonpath: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["NN2RTL_REPO_ROOT"] = str(tmp_path)
    if pythonpath is not None:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(pythonpath) if not existing else f"{pythonpath}{os.pathsep}{existing}"
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "prepare_pipeline.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def parse_summary_table(stdout: str) -> dict[str, dict[str, str]]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    data_lines = lines[2:]
    rows: dict[str, dict[str, str]] = {}
    for line in data_lines:
        module_id, op_type, shape, num_weights, pipeline_latency_cycles = [
            cell.strip() for cell in line.split(" | ")
        ]
        rows[module_id] = {
            "op_type": op_type,
            "shape": shape,
            "num_weights": num_weights,
            "pipeline_latency_cycles": pipeline_latency_cycles,
        }
    return rows


@pytest.mark.full
def test_prepare_pipeline_cli_runs_full_frontend_and_prints_all_17_modules(
    tmp_path: Path,
) -> None:
    fake_torchvision = make_fake_torchvision_package(tmp_path)
    result = run_prepare_pipeline(tmp_path, pythonpath=fake_torchvision)

    assert result.returncode == 0, result.stderr

    payload = json.loads((tmp_path / "output" / "layer_ir.json").read_text(encoding="utf8"))
    assert len(payload["layers"]) == 17

    table_rows = parse_summary_table(result.stdout)
    assert len(table_rows) == 17

    for layer in payload["layers"]:
        assert layer["module_id"] in table_rows
        row = table_rows[layer["module_id"]]
        assert row["op_type"] == layer["op_type"]
        assert row["pipeline_latency_cycles"] == str(layer["pipeline_latency_cycles"])

    assert table_rows["layer0_0_conv1"]["op_type"] == "conv2d"
    assert table_rows["layer1_0_downsample"]["op_type"] == "conv2d"
    assert table_rows["layer1_0_add"]["pipeline_latency_cycles"] == "1"
    assert table_rows["layer1_2_post_add_relu"]["pipeline_latency_cycles"] == "1"


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
    layer_ir_path = tmp_path / "layer_ir.json"
    fixture_path = REPO_ROOT / "test" / "fixtures" / "pipeline_ir.json"
    payload = json.loads(fixture_path.read_text(encoding="utf8"))
    payload["layers"][0]["clock_signal"] = "clock"
    layer_ir_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")

    with pytest.raises(
        prepare_pipeline_module.PreparePipelineError,
        match="PipelineIR schema validation failed",
    ):
        prepare_pipeline_module.validate_pipeline_ir(REPO_ROOT, layer_ir_path)
