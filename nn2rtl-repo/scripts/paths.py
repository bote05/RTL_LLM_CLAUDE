"""Path helpers shared by the nn2rtl preprocessing scripts."""

from __future__ import annotations

import os
from pathlib import Path


def detect_repo_root(current_file: str | Path) -> Path:
    """Resolve the repository root, with an env override for tests."""
    override = os.environ.get("NN2RTL_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(current_file).resolve().parent.parent
