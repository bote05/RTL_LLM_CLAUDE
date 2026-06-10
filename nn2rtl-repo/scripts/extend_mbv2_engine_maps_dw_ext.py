#!/usr/bin/env python3
"""extend_mbv2_engine_maps_dw_ext.py — DW-ENGINE STRIDE-1 EXTENSION (2026-06-10)

Appends the 9 remaining STRIDE-1 MobileNetV2 depthwise convs to the MBV2
shared-engine weight banks + bias/scale ROM images, so they can run as engine
DEPTHWISE dispatches (per-lane activation; engine-core support landed in
DW-ENGINE P1 — see output/rtl/engine/* [DW-ENGINE P1] markers).

Extends the P1 state (banks 13260 deep, bias/scale 70 words — built by
scripts/extend_mbv2_engine_maps_dw.py for conv_896/902/908) APPEND-ONLY:

  conv   C    HxW    oc_passes  wgt words  wgt base   bias/scale words @ base
  824   144  56x56       1          9       13260            1 @ 70
  836   192  28x28       1          9       13269            1 @ 71
  842   192  28x28       1          9       13278            1 @ 72
  854   384  14x14       2         18       13287            2 @ 73
  860   384  14x14       2         18       13305            2 @ 75
  866   384  14x14       2         18       13323            2 @ 77
  872   384  14x14       2         18       13341            2 @ 79
  878   576  14x14       3         27       13359            3 @ 81
  884   576  14x14       3         27       13386            3 @ 84
                        sum:      153 -> banks 13413;  17 -> bias/scale 87

Layouts are IDENTICAL to P1 (same encoder functions):
  * WEIGHT BANKS: per conv, word (base + p*9 + t) lane L (bank L/32, byte
    L%32) = weights[(256p+L)*9 + t]; zero for 256p+L >= C (dead lanes).
    Matches the depthwise address_generator walk weight_word = base +
    oc_pass*K_TOTAL + (kh*KW+kw), K_TOTAL = 9.
  * BIAS / SCALE: word (base + p) slot L = biases/scale'[256p+L]; scale slot
    = the per-channel CONSTANT-SHIFT slot from node_conv_*_scale.mem
    (slot[30:0] = mult' = mult << (23-shift); FIXED_SHIFT=23 both sides).

PROOF + idempotency: same two-pass scheme as P1 — derive (pass-major),
verify with an INDEPENDENT lane-major walk, append with prefix/suffix
re-read asserts; if already extended, re-derive + byte-compare + exit 0.

Run from anywhere: paths are Path(__file__)-relative (worktree-safe).
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WDIR = REPO / "output" / "mobilenet-v2" / "weights"

# (module_id, C). Dispatch/append order. oc_passes = ceil(C/256).
EXT_CONVS = [
    ("node_conv_824", 144),
    ("node_conv_836", 192),
    ("node_conv_842", 192),
    ("node_conv_854", 384),
    ("node_conv_860", 384),
    ("node_conv_866", 384),
    ("node_conv_872", 384),
    ("node_conv_878", 576),
    ("node_conv_884", 576),
]
K_TOTAL = 9
MAC = 256
NUM_BANKS = 8
BANK_BASE = 13260     # P1-extended bank depth (words)
BIAS_BASE = 70        # P1-extended bias/scale word count


def passes(c: int) -> int:
    return (c + MAC - 1) // MAC


def read_hex_lines(path: Path) -> list[int]:
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        t = ln.strip()
        if not t or t.startswith("//"):
            continue
        out.append(int(t, 16))
    return out


def load_conv(mid: str, c: int):
    w = read_hex_lines(WDIR / f"{mid}_weights.hex")
    b = read_hex_lines(WDIR / f"{mid}_bias.hex")
    s = read_hex_lines(WDIR / f"{mid}_scale.mem")
    assert len(w) == c * K_TOTAL, f"{mid}: weights {len(w)} != {c*K_TOTAL}"
    assert len(b) == c, f"{mid}: bias {len(b)} != {c}"
    assert len(s) == c, f"{mid}: scale {len(s)} != {c}"
    for v in s:
        assert 0 <= v < (1 << 31), f"{mid}: scale slot {v:#x} not a [30:0] mult'"
    return w, b, s


def encode_bank_line(bank_bytes: list[int]) -> str:
    """32 INT8 bytes -> 288b line ('00000000' + reversed-bytes hex) —
    identical to build_weight_memory_map.py / extend_mbv2_engine_maps_dw.py."""
    assert len(bank_bytes) == 32
    return "00000000" + bytes(x & 0xFF for x in reversed(bank_bytes)).hex()


def pack_wide_word(slots32: list[int]) -> str:
    """256 x 32b slots -> 2048-hex-char line, slot 255 first, BE per slot."""
    assert len(slots32) == MAC
    raw = bytearray()
    for s in range(MAC - 1, -1, -1):
        raw.extend(struct.pack(">I", slots32[s] & 0xFFFFFFFF))
    return raw.hex()


def derive_appended():
    bank_app: list[list[str]] = [[] for _ in range(NUM_BANKS)]
    bias_app: list[str] = []
    scale_app: list[str] = []
    for mid, c in EXT_CONVS:
        w, b, s = load_conv(mid, c)
        for p in range(passes(c)):
            for t in range(K_TOTAL):
                for bank in range(NUM_BANKS):
                    bb = []
                    for slot in range(32):
                        ch = p * MAC + bank * 32 + slot
                        bb.append(w[ch * K_TOTAL + t] if ch < c else 0)
                    bank_app[bank].append(encode_bank_line(bb))
        for p in range(passes(c)):
            bias_app.append(pack_wide_word(
                [(b[p * MAC + L] if p * MAC + L < c else 0) for L in range(MAC)]))
            scale_app.append(pack_wide_word(
                [(s[p * MAC + L] if p * MAC + L < c else 0) for L in range(MAC)]))
    return bank_app, bias_app, scale_app


def independent_check(bank_app, bias_app, scale_app) -> None:
    """Second-pass verification with INDEPENDENT indexing (lane-major)."""
    woff = 0   # appended weight words consumed so far (per bank)
    boff = 0   # appended bias/scale words consumed so far
    for mid, c in EXT_CONVS:
        w, b, s = load_conv(mid, c)
        np_ = passes(c)
        for L in range(MAC * np_):                # global lane walk
            p, lane = divmod(L, MAC)
            bank, slot = divmod(lane, 32)
            ch = p * MAC + lane
            bword = bias_app[boff + p]
            sword = scale_app[boff + p]
            off = (MAC - 1 - lane) * 8
            bgot = int(bword[off:off + 8], 16)
            sgot = int(sword[off:off + 8], 16)
            bexp = (b[ch] & 0xFFFFFFFF) if ch < c else 0
            sexp = (s[ch] & 0xFFFFFFFF) if ch < c else 0
            assert bgot == bexp, f"{mid} bias lane {lane} pass {p}: {bgot:#x}!={bexp:#x}"
            assert sgot == sexp, f"{mid} scale lane {lane} pass {p}: {sgot:#x}!={sexp:#x}"
            for t in range(K_TOTAL):
                line = bank_app[bank][woff + p * K_TOTAL + t]
                hoff = len(line) - (slot + 1) * 2
                got = int(line[hoff:hoff + 2], 16)
                exp = (w[ch * K_TOTAL + t] & 0xFF) if ch < c else 0
                assert got == exp, f"{mid} w lane {lane} p{p} t{t}: {got:#x}!={exp:#x}"
        woff += np_ * K_TOTAL
        boff += np_
    print("[maps-dw-ext] independent lane-major verification PASS "
          f"({woff} wgt words, {boff} bias/scale words, 9 convs)")


def extend_file(path: Path, base_count: int, appended: list[str], what: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) == base_count + len(appended):
        assert lines[base_count:] == appended, f"{what}: extended but content differs!"
        print(f"[maps-dw-ext] {what}: already extended + content verified ({len(lines)} lines)")
        return
    assert len(lines) == base_count, f"{what}: expected {base_count} lines, got {len(lines)}"
    pre = list(lines)
    out = lines + appended
    path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    post = path.read_text(encoding="utf-8").splitlines()
    assert post[:base_count] == pre, f"{what}: pre-existing lines changed!"
    assert post[base_count:] == appended, f"{what}: appended block mismatch!"
    print(f"[maps-dw-ext] {what}: {base_count} -> {len(post)} lines (appended {len(appended)})")


def update_json(path: Path, key_layers: str, new_entries: list[dict],
                totals: dict | None) -> None:
    if not path.is_file():
        print(f"[maps-dw-ext] (skip sidecar, missing: {path.name})")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    layers = data.get(key_layers)
    if layers is None:
        print(f"[maps-dw-ext] (skip sidecar, no '{key_layers}': {path.name})")
        return
    have = {L.get("module_id") for L in layers}
    changed = False
    for e in new_entries:
        if e["module_id"] not in have:
            layers.append(e)
            changed = True
    if totals:
        for k, v in totals.items():
            if data.get(k) != v:
                data[k] = v
                changed = True
    if changed:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"[maps-dw-ext] sidecar updated: {path.name}")
    else:
        print(f"[maps-dw-ext] sidecar already current: {path.name}")


def main() -> int:
    bank_app, bias_app, scale_app = derive_appended()
    independent_check(bank_app, bias_app, scale_app)

    for bank in range(NUM_BANKS):
        extend_file(WDIR / f"uram_weights_bank{bank}.mem", BANK_BASE,
                    bank_app[bank], f"bank{bank}")
    extend_file(WDIR / "bias.mem", BIAS_BASE, bias_app, "bias.mem")
    extend_file(WDIR / "scale.mem", BIAS_BASE, scale_app, "scale.mem")

    # sidecars (+ print the per-conv base table the applier/iso-cfg consume)
    wmap_entries, bmap_entries = [], []
    wbase, bbase = BANK_BASE, BIAS_BASE
    print("[maps-dw-ext] base table (conv: wgt_base bias/scale_base oc_passes):")
    for mid, c in EXT_CONVS:
        np_ = passes(c)
        print(f"  {mid}: w={wbase} b/s={bbase} passes={np_}")
        wmap_entries.append({
            "module_id": mid,
            "weight_shape": [c, 1, 3, 3],
            "oc_passes": np_,
            "base_mac_cycle": wbase,
            "size_mac_cycles": np_ * K_TOTAL,
            "depthwise": True,
        })
        bmap_entries.append({
            "module_id": mid,
            "oc": c,
            "oc_passes": np_,
            "base_word": bbase,
            "depthwise": True,
        })
        wbase += np_ * K_TOTAL
        bbase += np_
    update_json(WDIR / "weight_memory_map.json", "layers", wmap_entries,
                {"total_mac_cycles": wbase})
    update_json(WDIR / "bias_memory_map.json", "layers", bmap_entries, None)
    update_json(WDIR / "scale_memory_map.json", "layers", bmap_entries, None)

    print(f"[maps-dw-ext] DONE: banks {BANK_BASE}->{wbase}, bias/scale {BIAS_BASE}->{bbase}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
