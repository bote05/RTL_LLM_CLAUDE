#!/usr/bin/env python3
"""extend_mbv2_engine_maps_dw.py — DW-ENGINE P1 (2026-06-10)

Appends the 3 MobileNetV2 wide depthwise convs (node_conv_896/902/908:
C=960, 3x3, stride 1, pad 1, 7x7) to the MBV2 shared-engine weight banks +
bias/scale ROM images, so they can run as engine dispatches 28/31/34 in
DEPTHWISE mode (per-lane activation; see output/rtl/engine/mac_array.v
[DW-ENGINE P1]).

Engine-side layout appended (per conv, in dispatch order 896, 902, 908):

  WEIGHT BANKS (uram_weights_bank0..7.mem, 288b lines, low 256b used):
    36 words per conv = oc_pass p in 0..3 (outer) x tap t in 0..8 (inner,
    t = kh*3 + kw). Word (base + p*9 + t) lane L (bank L/32, byte L%32) =
    weights[(256p+L)*9 + t] from node_conv_*_weights.hex ([oc][kh][kw],
    one signed INT8 byte per line — the EXACT image the spatial wrapper
    $readmemh'd), zero for 256p+L >= 960. This matches the depthwise
    address_generator walk: weight_word = base + oc_pass*K_TOTAL + (kh*KW+kw)
    with K_TOTAL = KH*KW = 9.

  BIAS (bias.mem, 8192b lines = 256 x INT32, slot255-first BE — the
  build_bias_memory_map.py convention):
    4 words per conv: word (base + p) slot L = biases[256p+L] from
    node_conv_*_bias.hex (INT32 per line), zero-padded past channel 959.

  SCALE (scale.mem, same word geometry as bias; read at the bias address):
    4 words per conv: word (base + p) slot L = the per-channel CONSTANT-SHIFT
    slot from node_conv_*_scale.mem (slot[30:0] = mult' = mult << (23-shift)
    — EXACTLY the slots the byte-exact spatial DW RTL consumes, and exactly
    the format output/rtl/engine/requant_pipeline.v consumes since the
    FIT-FIX 2026-06-07 constant-shift rework; FIXED_SHIFT=23 on both sides).
    Zero past channel 959 (those lanes also have zero weights+bias -> emit 0,
    and the output bridge never emits bytes 192..255 of the last beat).

Bases (asserted): weights 13152/13188/13224 (banks were 13152 deep);
bias/scale word 58/62/66 (maps were 58 words). New totals: banks 13260,
bias/scale 70 words (<= 256 capacity).

PROOF + idempotency:
  * pre-existing lines are NEVER touched: the script asserts the exact
    pre-extension line counts (or, if already extended, re-derives the
    appended block and asserts it matches byte-for-byte, then exits 0).
  * every appended line is re-derived from the per-layer hex files by an
    INDEPENDENT second pass (different indexing order) and compared.
  * sidecar JSONs (weight/bias/scale memory maps) get the 3 layer entries
    appended + totals updated.

Run from anywhere: paths are Path(__file__)-relative (worktree-safe).
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WDIR = REPO / "output" / "mobilenet-v2" / "weights"

CONVS = ["node_conv_896", "node_conv_902", "node_conv_908"]
C = 960               # channels (= IC = OC, depthwise)
K_TOTAL = 9           # 3x3 taps, t = kh*3 + kw
OC_PASSES = 4         # ceil(960/256)
MAC = 256
NUM_BANKS = 8
BANK_BASE = 13152     # pre-extension bank depth (words)
BIAS_BASE = 58        # pre-extension bias/scale word count
WORDS_PER_CONV_W = OC_PASSES * K_TOTAL    # 36
WORDS_PER_CONV_B = OC_PASSES              # 4


def read_hex_lines(path: Path) -> list[int]:
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        t = ln.strip()
        if not t or t.startswith("//"):
            continue
        out.append(int(t, 16))
    return out


def load_conv(mid: str):
    w = read_hex_lines(WDIR / f"{mid}_weights.hex")
    b = read_hex_lines(WDIR / f"{mid}_bias.hex")
    s = read_hex_lines(WDIR / f"{mid}_scale.mem")
    assert len(w) == C * K_TOTAL, f"{mid}: weights {len(w)} != {C*K_TOTAL}"
    assert len(b) == C, f"{mid}: bias {len(b)} != {C}"
    assert len(s) == C, f"{mid}: scale {len(s)} != {C}"
    for v in s:
        assert 0 <= v < (1 << 31), f"{mid}: scale slot {v:#x} not a [30:0] mult'"
    return w, b, s


def encode_bank_line(bank_bytes: list[int]) -> str:
    """32 INT8 bytes -> 288b line: '00000000' + reversed-bytes hex
    (byte 0 at bits[7:0]) — identical to build_weight_memory_map.py."""
    assert len(bank_bytes) == 32
    return "00000000" + bytes(b & 0xFF for b in reversed(bank_bytes)).hex()


def pack_wide_word(slots32: list[int]) -> str:
    """256 x 32b slots -> 2048-hex-char line, slot 255 first, BE per slot —
    identical to build_bias_memory_map.py / build_scale_memory_map.py."""
    assert len(slots32) == MAC
    raw = bytearray()
    for s in range(MAC - 1, -1, -1):
        raw.extend(struct.pack(">I", slots32[s] & 0xFFFFFFFF))
    return raw.hex()


def derive_appended() -> tuple[list[list[str]], list[str], list[str]]:
    """-> (per-bank weight lines, bias lines, scale lines) for the 3 convs."""
    bank_app: list[list[str]] = [[] for _ in range(NUM_BANKS)]
    bias_app: list[str] = []
    scale_app: list[str] = []
    for mid in CONVS:
        w, b, s = load_conv(mid)
        for p in range(OC_PASSES):
            for t in range(K_TOTAL):
                for bank in range(NUM_BANKS):
                    bb = []
                    for slot in range(32):
                        ch = p * MAC + bank * 32 + slot
                        bb.append(w[ch * K_TOTAL + t] if ch < C else 0)
                    bank_app[bank].append(encode_bank_line(bb))
        for p in range(OC_PASSES):
            bias_app.append(pack_wide_word(
                [(b[p * MAC + L] if p * MAC + L < C else 0) for L in range(MAC)]))
            scale_app.append(pack_wide_word(
                [(s[p * MAC + L] if p * MAC + L < C else 0) for L in range(MAC)]))
    return bank_app, bias_app, scale_app


def independent_check(bank_app, bias_app, scale_app) -> None:
    """Second-pass verification with INDEPENDENT indexing (lane-major)."""
    for ci, mid in enumerate(CONVS):
        w, b, s = load_conv(mid)
        for L in range(MAC * OC_PASSES):          # global lane walk
            p, lane = divmod(L, MAC)
            bank, slot = divmod(lane, 32)
            ch = p * MAC + lane
            # bias/scale slot check
            bword = bias_app[ci * OC_PASSES + p]
            sword = scale_app[ci * OC_PASSES + p]
            off = (MAC - 1 - lane) * 8            # hex offset of this slot
            bgot = int(bword[off:off + 8], 16)
            sgot = int(sword[off:off + 8], 16)
            bexp = (b[ch] & 0xFFFFFFFF) if ch < C else 0
            sexp = (s[ch] & 0xFFFFFFFF) if ch < C else 0
            assert bgot == bexp, f"{mid} bias lane {lane} pass {p}: {bgot:#x}!={bexp:#x}"
            assert sgot == sexp, f"{mid} scale lane {lane} pass {p}: {sgot:#x}!={sexp:#x}"
            for t in range(K_TOTAL):
                line = bank_app[bank][ci * WORDS_PER_CONV_W + p * K_TOTAL + t]
                hoff = len(line) - (slot + 1) * 2  # byte `slot` at bits[slot*8 +: 8]
                got = int(line[hoff:hoff + 2], 16)
                exp = (w[ch * K_TOTAL + t] & 0xFF) if ch < C else 0
                assert got == exp, f"{mid} w lane {lane} p{p} t{t}: {got:#x}!={exp:#x}"
    print("[maps-dw] independent lane-major verification PASS")


def extend_file(path: Path, base_count: int, appended: list[str], what: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) == base_count + len(appended):
        assert lines[base_count:] == appended, f"{what}: extended but content differs!"
        assert lines[:base_count] == lines[:base_count]  # no-op; counts asserted
        print(f"[maps-dw] {what}: already extended + content verified ({len(lines)} lines)")
        return
    assert len(lines) == base_count, f"{what}: expected {base_count} lines, got {len(lines)}"
    pre = list(lines)
    out = lines + appended
    path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    # PROOF: re-read; prefix unchanged, suffix == appended.
    post = path.read_text(encoding="utf-8").splitlines()
    assert post[:base_count] == pre, f"{what}: pre-existing lines changed!"
    assert post[base_count:] == appended, f"{what}: appended block mismatch!"
    print(f"[maps-dw] {what}: {base_count} -> {len(post)} lines (appended {len(appended)})")


def update_json(path: Path, key_layers: str, new_entries: list[dict],
                totals: dict | None) -> None:
    if not path.is_file():
        print(f"[maps-dw] (skip sidecar, missing: {path.name})")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    layers = data.get(key_layers)
    if layers is None:
        print(f"[maps-dw] (skip sidecar, no '{key_layers}': {path.name})")
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
        print(f"[maps-dw] sidecar updated: {path.name}")
    else:
        print(f"[maps-dw] sidecar already current: {path.name}")


def main() -> int:
    bank_app, bias_app, scale_app = derive_appended()
    independent_check(bank_app, bias_app, scale_app)

    for bank in range(NUM_BANKS):
        extend_file(WDIR / f"uram_weights_bank{bank}.mem", BANK_BASE,
                    bank_app[bank], f"bank{bank}")
    extend_file(WDIR / "bias.mem", BIAS_BASE, bias_app, "bias.mem")
    extend_file(WDIR / "scale.mem", BIAS_BASE, scale_app, "scale.mem")

    # sidecars
    wmap_entries = []
    bmap_entries = []
    for i, mid in enumerate(CONVS):
        wmap_entries.append({
            "module_id": mid,
            "weight_shape": [C, 1, 3, 3],
            "oc_passes": OC_PASSES,
            "base_mac_cycle": BANK_BASE + i * WORDS_PER_CONV_W,
            "size_mac_cycles": WORDS_PER_CONV_W,
            "depthwise": True,
        })
        bmap_entries.append({
            "module_id": mid,
            "oc": C,
            "oc_passes": OC_PASSES,
            "base_word": BIAS_BASE + i * WORDS_PER_CONV_B,
            "depthwise": True,
        })
    update_json(WDIR / "weight_memory_map.json", "layers", wmap_entries,
                {"total_mac_cycles": BANK_BASE + 3 * WORDS_PER_CONV_W})
    update_json(WDIR / "bias_memory_map.json", "layers", bmap_entries, None)
    update_json(WDIR / "scale_memory_map.json", "layers", bmap_entries, None)

    print(f"[maps-dw] DONE: banks {BANK_BASE}->{BANK_BASE + 3*WORDS_PER_CONV_W}, "
          f"bias/scale {BIAS_BASE}->{BIAS_BASE + 3*WORDS_PER_CONV_B} "
          f"(bases: w 13152/13188/13224, b/s 58/62/66)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
