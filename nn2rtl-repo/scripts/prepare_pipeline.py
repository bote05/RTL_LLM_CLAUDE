#!/usr/bin/env python3
"""Run the deterministic frontend smoke flow and print a PipelineIR summary table."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from scripts.paths import detect_repo_root


SOURCE_REPO_ROOT = Path(__file__).resolve().parent.parent
LAYER_IR_RELATIVE_PATH = Path("output") / "layer_ir.json"

PIPELINE_IR_VALIDATION_SOURCE = """
import { readFileSync } from "node:fs";
import { pipelineIrSchema } from "./mcp/schemas.ts";

const filePath = process.argv[1];
const parsed = JSON.parse(readFileSync(filePath, "utf8"));
const validated = pipelineIrSchema.safeParse(parsed);

if (!validated.success) {
  console.error(JSON.stringify(validated.error.issues, null, 2));
  process.exit(1);
}
"""


class PreparePipelineError(RuntimeError):
    """Raised when a smoke-harness step fails."""


def _format_process_failure(step_name: str, result: subprocess.CompletedProcess[str]) -> str:
    message = [f"{step_name} failed with exit code {result.returncode}."]
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if stderr:
        message.append(stderr)
    elif stdout:
        message.append(stdout)
    return "\n".join(message)


def _build_runtime_env(runtime_repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["NN2RTL_REPO_ROOT"] = str(runtime_repo_root)
    return env


def run_python_script(
    source_repo_root: Path,
    runtime_repo_root: Path,
    script_name: str,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(source_repo_root / "scripts" / script_name)],
        cwd=source_repo_root,
        capture_output=True,
        text=True,
        env=_build_runtime_env(runtime_repo_root),
        check=False,
    )
    if result.returncode != 0:
        raise PreparePipelineError(_format_process_failure(script_name, result))
    return result


def validate_pipeline_ir(source_repo_root: Path, layer_ir_path: Path) -> dict[str, Any]:
    if not layer_ir_path.exists():
        raise PreparePipelineError(f"Expected PipelineIR at '{layer_ir_path}', but the file was not created.")

    result = subprocess.run(
        [
            "node",
            "--disable-warning=ExperimentalWarning",
            "--experimental-strip-types",
            "--input-type=module",
            "-e",
            PIPELINE_IR_VALIDATION_SOURCE,
            str(layer_ir_path),
        ],
        cwd=source_repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise PreparePipelineError(
            f"PipelineIR schema validation failed for '{layer_ir_path}':\n"
            f"{result.stderr.strip() or result.stdout.strip()}"
        )

    return json.loads(layer_ir_path.read_text(encoding="utf8"))


def _format_shape(shape: Sequence[object]) -> str:
    return "x".join(str(dimension) for dimension in shape)


def format_pipeline_table(layers: Sequence[Mapping[str, Any]]) -> str:
    headers = (
        "module_id",
        "op_type",
        "shape",
        "num_weights",
        "pipeline_latency_cycles",
    )
    rows = [
        (
            str(layer["module_id"]),
            str(layer["op_type"]),
            _format_shape(layer["output_shape"]),
            str(layer["num_weights"]),
            str(layer["pipeline_latency_cycles"]),
        )
        for layer in layers
    ]
    widths = [
        max([len(header), *(len(row[index]) for row in rows)])
        for index, header in enumerate(headers)
    ]

    def render_row(row: Sequence[str]) -> str:
        return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    lines = [render_row(headers), separator]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def prepare_pipeline(
    source_repo_root: Path = SOURCE_REPO_ROOT,
    runtime_repo_root: Path | None = None,
) -> dict[str, Any]:
    resolved_runtime_repo_root = (runtime_repo_root or detect_repo_root(__file__)).resolve()
    run_python_script(source_repo_root, resolved_runtime_repo_root, "quantize_model.py")
    run_python_script(source_repo_root, resolved_runtime_repo_root, "generate_golden.py")
    return validate_pipeline_ir(
        source_repo_root,
        resolved_runtime_repo_root / LAYER_IR_RELATIVE_PATH,
    )


def main() -> None:
    payload = prepare_pipeline()
    print(format_pipeline_table(payload["layers"]))


if __name__ == "__main__":
    try:
        main()
    except PreparePipelineError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error
