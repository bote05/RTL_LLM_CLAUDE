#!/usr/bin/env python3
"""apply_resnet_engpipe.py — [ENG-PIPE-RN 2026-06-11] enable the pipelined
(pixel, oc_pass) engine issue (ENG_PIPE=1) on the ResNet-50 top.

The ENG_PIPE machinery itself (shared_engine_skeleton.v `g_ep` branch:
ST_GAP issue pipelining + per-pass capture registers + event-driven retire)
was built, proven and shipped for MBV2 (docs/agent_tasks/ENG_PIPE_ANALYSIS.md,
21/21 ISO byte-exact incl. LFSR-throttled backpressure, MBV2 8/8 e2e).
ResNet elaborated ENG_PIPE=0 (verbatim legacy stop-and-wait FSM) until now.

THIS APPLIER CHANGES ONLY output/rtl/nn2rtl_top.v (ResNet-own file): the
shared_engine instantiation gains `.ENG_PIPE(1)`. No shared file is touched.

Why it is safe on ResNet:
* ResNet leaves ENABLE_OUTPUT_BACKPRESSURE at 0 -> eff_out_ready==1'b1: the
  bridge write drains every beat in one cycle, so the g_ep FIRE gating
  (`!in_pipe && !act_out_wr_en`) never sees a held beat; the pend queue
  stays <= 2 by construction exactly as on MBV2.
* ResNet's engine output NEVER enters the spatial stream directly — it goes
  through the 4096-deep engine_output_fifo and the per-dispatch
  engine_output_bridge shims. The MBV2 ADD-JOIN class (faster engine write
  cadence exposing an accept-vs-pop race at residual joins) has no ResNet
  analog on this path; the e2e byte-exact gate is the authority.
* Per-dispatch output volume max = 784 beats (conv_250) << 4096 FIFO depth,
  so the (unconnected-out_ready) FIFO cannot overflow even at the ~6-cycle
  pipelined retire cadence.

Expected effect: per-(pixel, oc_pass) issue bubble 12 (pixel) / 10
(intermediate) -> 3 across the 17 ResNet dispatches.

Gates (run after applying):
  * lint (verilator --lint-only) — 0 errors
  * ResNet e2e vec0+vec1: PASS 0/100352 (cycles drop vs 5,664,715)

Usage: python scripts/apply_resnet_engpipe.py [--check]
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOP = REPO / "output" / "rtl" / "nn2rtl_top.v"

_backed_up: set[Path] = set()


def patch(path: Path, old: str, new: str, tag: str, count: int = 1,
          probe: str | None = None) -> None:
    """Anchor-asserted replace. Idempotent: presence of `probe` (or `new`)
    == applied. `probe` must be a marker LATER appliers cannot disturb
    (apply_resnet_waddr_rep.py appends a parameter right after this hunk,
    so the full `new` text does not survive the rest of the bundle)."""
    text = path.read_text(encoding="utf-8")
    if (probe or new) in text:
        print(f"  [skip] {path.name}: {tag} already applied")
        return
    n = text.count(old)
    if n != count:
        raise SystemExit(f"ANCHOR FAIL {path.name} / {tag}: found {n}, want {count}")
    if path not in _backed_up:
        bak = path.with_name(path.name + ".preengpiper")
        if not bak.exists():
            bak.write_text(text, encoding="utf-8", newline="\n")
        _backed_up.add(path)
    path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")
    print(f"  [ok]   {path.name}: {tag}")


def main() -> int:
    if "--check" in sys.argv:
        t = TOP.read_text(encoding="utf-8")
        print(f"{TOP.name}: ENG-PIPE-RN markers = {t.count('[ENG-PIPE-RN')}")
        return 0
    print("[engpipe-rn] patching nn2rtl_top.v (ResNet shared_engine inst) ...")
    patch(TOP, """        .K_PAR(ENGINE_K_PAR)
    ) u_shared_engine (
""", """        .K_PAR(ENGINE_K_PAR),
        // [ENG-PIPE-RN 2026-06-11] pipelined (pixel, oc_pass) issue: per-pass
        // bubble 12 (pixel) / 10 (intermediate) -> 3. Machinery proven on
        // MBV2 (ENG_PIPE_ANALYSIS.md); ResNet keeps backpressure disabled
        // (eff_out_ready==1) and drains via the 4096-deep engine FIFO.
        .ENG_PIPE(1)
    ) u_shared_engine (
""", "ENG_PIPE=1 on u_shared_engine", probe=".ENG_PIPE(1)")
    print("[engpipe-rn] done. Backup: nn2rtl_top.v.preengpiper. Re-run is a no-op.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
