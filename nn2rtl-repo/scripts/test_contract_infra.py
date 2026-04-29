from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ROOT = REPO_ROOT / "contracts"
CONTRACT_IDS = [
    "flat-bus",
    "tiled-streaming",
    "dram-backed-weights",
    "activation-double-buffering",
    "weight-tiling",
]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_folders_are_complete() -> None:
    for contract_id in CONTRACT_IDS:
        contract_dir = CONTRACT_ROOT / contract_id
        metadata = json.loads((contract_dir / "metadata.json").read_text(encoding="utf8"))
        assert metadata["name"] == contract_id
        assert metadata["complexity_rank"] >= 0
        assert metadata["interface_signals"]
        assert metadata["fit_constraints"]["max_bus_width_bits"] > 0
        assert (contract_dir / "testbench.cpp").read_text(encoding="utf8")
        assert "generate_contract_vectors" in (contract_dir / "golden.py").read_text(encoding="utf8")
        assert "expectedLatencyCycles" in (contract_dir / "latency.ts").read_text(encoding="utf8")


def test_contract_golden_generators_are_importable() -> None:
    samples = [[1, 2, 3, 4], [5, 6, 7, 8]]
    expected = [[9, 10, 11, 12]]

    flat = load_module(CONTRACT_ROOT / "flat-bus" / "golden.py")
    assert flat.generate_contract_vectors(samples, expected) == (samples, expected)

    tiled = load_module(CONTRACT_ROOT / "tiled-streaming" / "golden.py")
    assert tiled.generate_contract_vectors(samples, expected, channel_tile=2) == (
        [[1, 2], [3, 4], [5, 6], [7, 8]],
        [[9, 10], [11, 12]],
    )

    dram = load_module(CONTRACT_ROOT / "dram-backed-weights" / "golden.py")
    assert dram.pack_weight_memory([1, 2, 3], data_width_bits=16) == [0x0201, 0x0003]

    weight_tiling = load_module(CONTRACT_ROOT / "weight-tiling" / "golden.py")
    assert weight_tiling.split_weight_indices(10, 3) == [(0, 4), (4, 8), (8, 10)]
