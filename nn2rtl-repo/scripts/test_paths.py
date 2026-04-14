from __future__ import annotations

from pathlib import Path

from scripts.paths import detect_repo_root


def test_detect_repo_root_uses_env_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NN2RTL_REPO_ROOT", str(tmp_path))
    assert detect_repo_root(__file__) == tmp_path.resolve()


def test_detect_repo_root_defaults_to_parent_repo() -> None:
    repo_root = detect_repo_root(__file__)
    assert (repo_root / "README.md").exists()
