#!/usr/bin/env python3
"""Parallelize the 3 stage-4 DRAM-backed 3x3 convs (284/292/298) to the proven
on-chip split-arch (coord_scheduler + line_buf_window + conv_datapath_mp_k) with
a BACKPRESSURED output streamer -- the SAME template the 7 spatial 3x3 convs use
(scripts/apply_3x3_backpressure.py.gen_wrapper). In serial Verilator the DRAM
variant re-streams its whole weight set per output pixel (~40-50M cyc each), so a
frame never finishes in kMaxCycles; the on-chip variant holds weights in host RAM
via $readmemh and runs at split-arch speed.

These are NOT in apply_3x3_backpressure.py's TARGETS and the DRAM module shape has
no MP_K/WEIGHTS_PATH to scrape, so params are given EXPLICITLY here (read off the
DRAM module: IC/OC/KH/KW/IH/IW/OH/OW/SH/SW/PH/PW/SCALE_*). MP=4, MP_K=9 to match
the working spatial 3x3 convs (e.g. conv_220 s2). Weights repacked to
node_conv_N_weights_mp_k_9.hex via repack_weights_wide.py --mp 4 --mp-k 9.

Byte-exact verified by tying ready_out high (--equiv) then run equiv_one.ts.

USAGE:
  python scripts/apply_dram_conv3x3.py                # regenerate all 3 (real port)
  python scripts/apply_dram_conv3x3.py --only 284
  python scripts/apply_dram_conv3x3.py --equiv 284    # tie ready_out high for equiv
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Reuse the EXACT proven backpressured split-arch template.
from apply_3x3_backpressure import gen_wrapper

RTL = Path("output/rtl")
WEIGHTS = Path("output/weights")

# Params read off each DRAM module on disk (IC/OC/KH/KW/IH/IW/OH/OW/SH/SW/PH/PW
# + the requant SCALE_MULT/SCALE_SHIFT). MP=4, MP_K=9 (match working spatial 3x3).
# MP bumped 4->16 (perf): these 3x3 convs are on the serial stage-4 critical path;
# MP=16 (144 DSP each, +324 total) ~4x faster. Byte-exact (equiv-verified). MP_K=9 (taps).
PARAMS = {
    284: dict(ic=512, oc=512, ih=14, iw=14, oh=7, ow=7, kh=3, kw=3,
              sh=2, sw=2, ph=1, pw=1, mp=16, mp_k=9, smult=18735, sshift=23),
    292: dict(ic=512, oc=512, ih=7,  iw=7,  oh=7, ow=7, kh=3, kw=3,
              sh=1, sw=1, ph=1, pw=1, mp=16, mp_k=9, smult=24577, sshift=22),
    298: dict(ic=512, oc=512, ih=7,  iw=7,  oh=7, ow=7, kh=3, kw=3,
              sh=1, sw=1, ph=1, pw=1, mp=16, mp_k=9, smult=28241, sshift=22),
}


def build(n: int) -> dict:
    p = dict(PARAMS[n])
    p["wpath"] = (Path.cwd() / WEIGHTS / f"node_conv_{n}_weights_mp_k_9.hex").as_posix()
    p["bpath"] = (Path.cwd() / WEIGHTS / f"node_conv_{n}_bias.hex").as_posix()
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated conv ids")
    ap.add_argument("--equiv", help="emit ready_out-tied-high variant for ONE id")
    args = ap.parse_args()

    if args.equiv:
        n = int(args.equiv)
        (RTL / f"node_conv_{n}.v").write_text(gen_wrapper(n, build(n), tie_ready_high=True))
        print(f"[equiv] node_conv_{n}.v overwritten with ready_out tied high")
        return

    only = set(int(x) for x in args.only.split(",")) if args.only else set(PARAMS)
    for n in sorted(PARAMS):
        if n not in only:
            continue
        p = build(n)
        (RTL / f"node_conv_{n}.v").write_text(gen_wrapper(n, p))
        print(f"[ok] conv_{n}: IC={p['ic']} OC={p['oc']} {p['ih']}x{p['iw']}->{p['oh']}x{p['ow']} "
              f"s{p['sh']} MP={p['mp']} MP_K={p['mp_k']} SCALE={p['smult']}/{p['sshift']} (parallel split-arch)")


if __name__ == "__main__":
    main()
