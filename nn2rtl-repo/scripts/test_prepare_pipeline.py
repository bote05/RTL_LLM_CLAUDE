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
def test_prepare_pipeline_cli_surfaces_flat_v2_failure_instead_of_silent_empty_goldens(
    tmp_path: Path,
) -> None:
    # Same contract as test_generate_golden_cli_rejects_flat_v2_checkpoint_from_real_ptq:
    # the prepare_pipeline smoke harness must propagate the loud flat-v2
    # failure from generate_golden.py instead of producing a LayerIR with
    # empty golden vectors. Remove this test (and restore the success-path
    # assertion) once Prompt 1 emits a residual_stack_spec and the fx path
    # succeeds end-to-end.
    result = run_prepare_pipeline(tmp_path)

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "lacks a traceable model spec" in combined or "generate_golden.py failed" in combined


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
    # prepare_pipeline() currently fails at the generate_golden step for the
    # flat v2 checkpoint (see the test above), so we exercise the schema
    # validator directly with a hand-crafted LayerIR derived from the
    # project's shared test fixture instead of going through the full CLI
    # chain. This keeps coverage of the node-side Zod validation while the
    # PTQ ↔ fx wiring gap exists.
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
