#!/usr/bin/env python3
"""Parallelize the 15 TRUE tiled-streaming serial-MAC 1x1 convs to the split-arch
backpressured datapath (coord_scheduler + line_buf_window + conv_datapath_mp_k),
MP=16 MP_K=8. Adds output backpressure (the streamer holds valid_out until
ready_out) so they stop dropping beats under a fast/slow imbalance.

Authoritative dims come from each module's sidecar conv signature
(conv2d_ICxOCxKHxKW_sIHxIW_stSHxSW_pPHxPW), so stride-2 (conv_224) is handled
correctly. SCALE_MULT/SCALE_SHIFT are read robustly from the current .v
(handles signed/sized literals and the SCALE_MULT_CONST alias); the equiv check
(tie-high) is the ground-truth verification that the chosen scale is byte-exact.

NOT for: conv_284/288/292/298 (dram-backed-weights contract, different arch;
284/292/298 are actually 3x3). Those are handled separately.

USAGE:
  python scripts/apply_parallel_1x1.py                 # regenerate all 15
  python scripts/apply_parallel_1x1.py --only 204,224
  python scripts/apply_parallel_1x1.py --equiv 224     # tie-high temp for equiv
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# import the proven general template
sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_3x3_backpressure import gen_wrapper  # noqa: E402

RTL = Path("output/rtl")
TB = Path("output/tb")
WEIGHTS = Path("output/weights")

TARGETS = [204, 206, 210, 212, 216, 224, 226, 230, 232, 238,
           248, 256, 268, 274, 280]

MP = 16
MP_K = 8


def grab_scale(txt: str, name: str) -> int | None:
    # tolerate signed/sized literals: 16'sd23595, 32'd6427, 8'hFF, plain 21
    for nm in (name, name + "_CONST"):
        m = re.search(rf"\b{nm}\b\s*=\s*\d+'s?([dh])([0-9a-fA-F]+)", txt)
        if m:
            return int(m.group(2), 16 if m.group(1) == "h" else 10)
        # tolerate localparam terminators: `=27623;` (orig) and `=27623,` (regenerated)
        m = re.search(rf"\b{nm}\b\s*=\s*(\d+)\s*[;,]", txt)
        if m:
            return int(m.group(1))
    return None


def sig_dims(n: int) -> dict:
    j = json.load(open(TB / f"node_conv_{n}.sidecar.json"))
    contract = j["contract_id"]
    gp = j["golden_inputs_path"]
    m = re.search(
        r"conv2d_(\d+)x(\d+)x(\d+)x(\d+)_s(\d+)x(\d+)_st(\d+)x(\d+)_p(\d+)x(\d+)", gp)
    if not m:
        raise SystemExit(f"conv_{n}: cannot parse signature from {gp}")
    ic, oc, kh, kw, ih, iw, sh, sw, ph, pw = map(int, m.groups())
    oh = (ih + 2 * ph - kh) // sh + 1
    ow = (iw + 2 * pw - kw) // sw + 1
    return dict(contract=contract, ic=ic, oc=oc, kh=kh, kw=kw,
                ih=ih, iw=iw, oh=oh, ow=ow, sh=sh, sw=sw, ph=ph, pw=pw)


def build_params(n: int) -> dict:
    d = sig_dims(n)
    if d["contract"] != "tiled-streaming":
        raise SystemExit(f"conv_{n}: contract={d['contract']} is NOT tiled-streaming; skip")
    if d["kh"] != 1 or d["kw"] != 1:
        raise SystemExit(f"conv_{n}: kernel {d['kh']}x{d['kw']} != 1x1; skip")
    src = (RTL / f"node_conv_{n}.v").read_text()
    smult = grab_scale(src, "SCALE_MULT")
    sshift = grab_scale(src, "SCALE_SHIFT")
    if smult is None or sshift is None:
        raise SystemExit(f"conv_{n}: failed to read SCALE_MULT={smult} SCALE_SHIFT={sshift}")
    wpath = (Path.cwd() / WEIGHTS / f"node_conv_{n}_weights_mp_k_{MP_K}.hex").as_posix()
    bpath = (Path.cwd() / WEIGHTS / f"node_conv_{n}_bias.hex").as_posix()
    return dict(ic=d["ic"], oc=d["oc"], ih=d["ih"], iw=d["iw"], oh=d["oh"], ow=d["ow"],
                kh=1, kw=1, sh=d["sh"], sw=d["sw"], ph=0, pw=0,
                mp=MP, mp_k=MP_K, smult=smult, sshift=sshift,
                wpath=wpath, bpath=bpath)


def repack(n: int, p: dict) -> bool:
    inp = WEIGHTS / f"node_conv_{n}_weights.hex"
    outp = WEIGHTS / f"node_conv_{n}_weights_mp_k_{MP_K}.hex"
    cmd = [sys.executable, "scripts/repack_weights_wide.py",
           "--input", str(inp), "--output", str(outp),
           "--oc", str(p["oc"]), "--k-total", str(p["ic"]),
           "--mp", str(MP), "--mp-k", str(MP_K)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[FAIL] conv_{n}: repack failed: {r.stderr.strip()[:160]}")
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated conv ids")
    ap.add_argument("--equiv", help="emit tie-high (no-port) variant to node_conv_<id>.v for ONE id")
    args = ap.parse_args()

    if args.equiv:
        n = int(args.equiv)
        p = build_params(n)
        repack(n, p)
        (RTL / f"node_conv_{n}.v").write_text(gen_wrapper(n, p, tie_ready_high=True))
        print(f"[equiv] node_conv_{n}.v overwritten (ready_out tied high) "
              f"IC={p['ic']} OC={p['oc']} {p['ih']}x{p['iw']}->{p['oh']}x{p['ow']} s{p['sh']}")
        return

    only = set(int(x) for x in args.only.split(",")) if args.only else set(TARGETS)
    for n in TARGETS:
        if n not in only:
            continue
        p = build_params(n)
        if not repack(n, p):
            continue
        (RTL / f"node_conv_{n}.v").write_text(gen_wrapper(n, p))
        print(f"[ok] conv_{n}: IC={p['ic']} OC={p['oc']} {p['ih']}x{p['iw']}->{p['oh']}x{p['ow']} "
              f"s{p['sh']} SM={p['smult']} SS={p['sshift']} MP={MP} MP_K={MP_K} (DSP={MP*MP_K})")


if __name__ == "__main__":
    main()
