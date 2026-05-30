#!/usr/bin/env python3
"""Emit a tiled-streaming sidecar for a conv whose module was re-architected from
the dram-backed-weights contract to the on-chip split-arch (tiled-streaming) ABI.

The GOLDEN vectors (goldin/goldout) are contract-independent here: both contracts
use 256-bit channel-tiled beats with CHANNEL_TILE=32, so the same byte streams
apply. We just swap the contract/testbench fields so equiv_one.ts runs the
streaming TB (which drives valid/ready, not AXI weight ports) and drop the AXI
weight fields. Output: output/tb/<mod>.streaming.sidecar.json (original untouched).

USAGE:
  python scripts/make_streaming_sidecar.py node_conv_284 [node_conv_292 ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TB = Path("output/tb")
REPO = Path.cwd()

# Contract fields copied verbatim from a working tiled-streaming sidecar
# (node_conv_220). Paths are absolute, shared by all tiled-streaming modules.
STREAMING = {
    "testbench_template_path": str(REPO / "contracts" / "tiled-streaming" / "testbench.cpp"),
    "contract_id": "tiled-streaming",
    "contract_name": "Tiled Streaming (Canonical ResNet-50 ABI)",
    "contract_metadata_path": str(REPO / "contracts" / "tiled-streaming" / "metadata.json"),
}
AXI_FIELDS = ["weights_path", "weight_bank_paths", "axi_weight_data_width_bits"]


def main() -> None:
    mods = sys.argv[1:]
    if not mods:
        raise SystemExit("usage: make_streaming_sidecar.py <module> [<module> ...]")
    for mod in mods:
        src = TB / f"{mod}.sidecar.json"
        sc = json.loads(src.read_text())
        sc.update(STREAMING)
        for f in AXI_FIELDS:
            sc.pop(f, None)
        out = TB / f"{mod}.streaming.sidecar.json"
        out.write_text(json.dumps(sc, indent=2) + "\n")
        print(f"[ok] {out}  (beats_in={sc.get('beats_per_input_sample')} "
              f"beats_out={sc.get('beats_per_output_sample')})")


if __name__ == "__main__":
    main()
