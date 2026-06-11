#!/usr/bin/env python3
"""apply_resnet_kpar8.py — [KPAR8-RN 2026-06-11] ENGINE K-PARALLEL P=8 for
the ResNet-50 top. Anchor-asserted + idempotent; .prekp8r backups.

The SHARED core already contains the complete K_PAR==8 elaboration
(g_p8 / g_walk_kpar8 / g_waddr_kpar8 / g_ktap_kpar8 — shipped for MBV2,
docs/agent_tasks/KPAR8_ANALYSIS.md) INCLUDING the ResNet pos-major dense-KxK
fast walk (the [KPAR4-RN] extension was merged into the kpar8 branch: the
g_walk_kpar8 ic-wrap advances kw/kh at ic_cnt == IC-8). NO shared file is
touched by this applier — it only re-parameterizes the ResNet top:

* ENGINE_K_PAR 4 -> 8 (ENGINE_WBUS_W auto-scales to 8*768 = 6144).
* Banks: 384b x 16768 (_kp4) -> 768b x 8384 (_kp8), ADDR_W 15 -> 14.
  Bits/bank IDENTICAL (67072 % 8 == 0, zero pad) => BRAM-neutral.
* weight_bank_rd_addr takes the engine's GROUP address export (old>>3).

ELIGIBILITY (proof P0 in scripts/repack_resnet_kpar8_banks.py, parsed from
the deployed scheduler): all 17 dispatch bases {0, 2304, 4352, 6656, 8960,
9984, 12288, 14592, 16896, 18944, 28160, 32256, 36352, 45568, 49664, 53760,
62976} are %8==0 and every IC in {256,512,1024,2048} is %8==0 — NO
relocation pad needed (unlike MBV2's FC base 13413 -> 13416 [FC-PAD]).

Run AFTER: python scripts/repack_resnet_kpar8_banks.py (gitignored _kp8.mem
banks must exist in the target checkout; proofs P0-P4 rerun every time).

Gates: lint 0; ISO A/B (run_resnet_engine_iso_kpar8.sh: 246 3x3, 250 1x1,
284 3x3-stride2, vec0+vec1, KPAR8==LEGACY byte-identical); ResNet e2e
vec0+vec1 PASS 0/100352.

Usage: python scripts/apply_resnet_kpar8.py [--check]
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
    == applied. `probe` is needed where a LATER bundle applier
    (apply_resnet_waddr_rep.py) rewrites this hunk's text."""
    text = path.read_text(encoding="utf-8")
    if (probe or new) in text:
        print(f"  [skip] {path.name}: {tag} already applied")
        return
    n = text.count(old)
    if n != count:
        raise SystemExit(f"ANCHOR FAIL {path.name} / {tag}: found {n}, want {count}")
    if path not in _backed_up:
        bak = path.with_name(path.name + ".prekp8r")
        if not bak.exists():
            bak.write_text(text, encoding="utf-8", newline="\n")
        _backed_up.add(path)
    path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")
    print(f"  [ok]   {path.name}: {tag}")


def main() -> int:
    if "--check" in sys.argv:
        t = TOP.read_text(encoding="utf-8")
        print(f"{TOP.name}: KPAR8-RN markers = {t.count('[KPAR8-RN')}")
        return 0

    print("[kpar8-rn] patching nn2rtl_top.v (ResNet engine geometry) ...")

    # 1. ENGINE_K_PAR 4 -> 8 (+ the doc comment above it).
    patch(TOP, """    // [KPAR4-RN 2026-06-10] ENGINE K-PARALLEL P=4: the weight bus carries
    // ENGINE_K_PAR tap-major 768b words per (group-addressed) repacked bank
    // line (scripts/repack_resnet_kpar4_banks.py: 96b->384b lines, depth /4
    // = 16768, dense-3x3 regions transposed pos-major). All 17 dispatches
    // are fast-eligible (every base and IC %4==0 — proof P0 in the repack).
    localparam integer ENGINE_K_PAR  = 4;
    localparam integer ENGINE_WBUS_W = ENGINE_K_PAR * 8 * ENGINE_LANE_B;  // weight bus width (4 taps x 768)
""", """    // [KPAR8-RN 2026-06-11] ENGINE K-PARALLEL P=8: the weight bus carries
    // ENGINE_K_PAR tap-major 768b words per (group-addressed) repacked bank
    // line (scripts/repack_resnet_kpar8_banks.py: 96b->768b lines, depth /8
    // = 8384, dense-3x3 regions transposed pos-major — same permutation as
    // the KPAR4 lineage, only the packing width changed). All 17 dispatches
    // are fast-eligible (every base and IC %8==0 — proof P0 in the repack).
    localparam integer ENGINE_K_PAR  = 8;
    localparam integer ENGINE_WBUS_W = ENGINE_K_PAR * 8 * ENGINE_LANE_B;  // weight bus width (8 taps x 768)
""", "ENGINE_K_PAR 4 -> 8")

    # 2. GROUP addressing: engine exports old>>3; 8384 lines need 14 bits.
    patch(TOP, """    wire [14:0] weight_bank_rd_addr = engine_weight_rd_addr[14:0];  // [KPAR4-RN] GROUP address (engine exports old>>2; 16768 wide lines)
""", """    wire [13:0] weight_bank_rd_addr = engine_weight_rd_addr[13:0];  // [KPAR8-RN] GROUP address (engine exports old>>3; 8384 wide lines)
""", "group address 15b -> 14b", probe="engine exports old>>3; 8384 wide lines")

    # 3. bank rd_data wire comments (8 occurrences; width expr is generic).
    patch(TOP, "_rd_data;  // [KPAR4-RN] 4 x 96b tap-major\n",
               "_rd_data;  // [KPAR8-RN] 8 x 96b tap-major\n",
          "bank rd_data wire comments", count=8)

    # 4. tap-major bus comment (generate loop itself is K_PAR-generic).
    patch(TOP, """    // [KPAR4-RN] MAC bus = ENGINE_K_PAR tap-major words. Each repacked bank
    // line carries 4 old words (tap j at [j*ENGINE_BANK_W +: ENGINE_BANK_W]);
""", """    // [KPAR8-RN] MAC bus = ENGINE_K_PAR tap-major words. Each repacked bank
    // line carries 8 old words (tap j at [j*ENGINE_BANK_W +: ENGINE_BANK_W]);
""", "tap-major bus comment")

    # 5. the 8 bank instantiations -> _kp8 geometry.
    for b in range(8):
        patch(TOP, f"""    uram_weight_bank #(
        .DEPTH(16768),          // [KPAR4-RN] 67072/4 wide lines
        .ADDR_W(15),
        .WORD_W(ENGINE_K_PAR*ENGINE_BANK_W),   // [KPAR4-RN] 4 x 96b tap-major
        .MEM_INIT_FILE("output/weights/uram_weights_bank{b}_kp4.mem")
""", f"""    uram_weight_bank #(
        .DEPTH(8384),           // [KPAR8-RN] 67072/8 wide lines
        .ADDR_W(14),
        .WORD_W(ENGINE_K_PAR*ENGINE_BANK_W),   // [KPAR8-RN] 8 x 96b tap-major
        .MEM_INIT_FILE("output/weights/uram_weights_bank{b}_kp8.mem")
""", f"bank{b} _kp8 geometry")

    # 6. engine instantiation comment (param expr already ENGINE_K_PAR).
    patch(TOP, """        // [KPAR4-RN] 4 taps/cycle/lane on ALL 17 dispatches (8 dense 1x1 +
        // 9 dense 3x3 via the pos-major transposed _kp4 banks).
""", """        // [KPAR8-RN] 8 taps/cycle/lane on ALL 17 dispatches (8 dense 1x1 +
        // 9 dense 3x3 via the pos-major transposed _kp8 banks).
""", "engine inst comment")

    print("[kpar8-rn] done. Backup: nn2rtl_top.v.prekp8r. Re-run is a no-op.")
    print("[kpar8-rn] REMINDER: regenerate banks via "
          "`python scripts/repack_resnet_kpar8_banks.py` before any sim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
