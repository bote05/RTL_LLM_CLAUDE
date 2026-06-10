#!/usr/bin/env python3
"""extend_mbv2_engine_maps_dw_quartet.py — DW-ENGINE STRIDE-2 QUARTET (2026-06-10)

Appends the 4 STRIDE-2 MobileNetV2 depthwise convs (818/830/848/890 — the
last spatial DW convs besides the stem-zone 812) to the MBV2 shared-engine
weight banks + bias/scale ROM images, so they can run as engine DEPTHWISE
dispatches. Stride-2 needs NO new weight layout: the depthwise walk reads
weight word = base + oc_pass*9 + (kh*KW+kw) exactly as for stride-1 (stride
only affects ACT addressing, which address_generator.v already computes from
cfg_stride_h/w with generic bounds).

Extends the FC/KPAR8 state APPEND-ONLY in BOTH bank domains:

  * OLD-domain banks  uram_weights_bank{0..7}.mem : 18533 -> 18587 lines
  * KP8 banks         uram_weights_bank{0..7}_kp8.mem : 2317 -> 2324 lines
      (relocated domain = old + 3 FC-pad for words >= 13413; appended words
       start at relocated 18536 = 8*2317 EXACTLY, i.e. on a fresh kp8 line;
       54 words -> 7 lines, last line tail-padded with 2 zero words. The DW
       serial walk reads tap addr&7 via the skeleton subword select, so no
       alignment requirement applies.)
  * bias.mem / scale.mem : 91 -> 97 words

  conv   C    IHxIH->OHxOH  passes  wgt wds  OLD base  KP8(reloc) base  b/s @ base
  818    96   112 -> 56        1       9      18533        18536          1 @ 91
  830   144    56 -> 28        1       9      18542        18545          1 @ 92
  848   192    28 -> 14        1       9      18551        18554          1 @ 93
  890   576    14 ->  7        3      27      18560        18563          3 @ 94
                              sum:    54  -> old 18587 / kp8 2324; b/s 97

Layouts IDENTICAL to P1/EXT (same encoder functions):
  * WEIGHT BANKS (old domain): per conv, word (base + p*9 + t) lane L
    (bank L//32, byte L%32) = weights[(256p+L)*9 + t]; zero for dead lanes.
  * KP8 lines: line g = {reloc[8g+7], ..., reloc[8g]} text-concatenated
    tap7-first — byte-identical construction to repack_mbv2_kpar8_banks.py.
  * BIAS / SCALE: word (base + p) slot L = biases/scale'[256p+L]; scale slot
    = per-channel CONSTANT-SHIFT slot from node_conv_*_scale.mem ([30:0]).

PROOF + idempotency: derive (pass-major) -> verify with an INDEPENDENT
lane-major walk -> verify the kp8 append re-expands to the old append ->
append with prefix/suffix re-read asserts; if already extended, re-derive +
byte-compare + exit 0. ALL FOUR convs are appended in one run regardless of
which stage the RTL applier wires (unwired map words are never read).
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WDIR = REPO / "output" / "mobilenet-v2" / "weights"

# (module_id, C, IH, OH). Append order = dispatch order.
QUARTET = [
    ("node_conv_818", 96, 112, 56),
    ("node_conv_830", 144, 56, 28),
    ("node_conv_848", 192, 28, 14),
    ("node_conv_890", 576, 14, 7),
]
K_TOTAL = 9
MAC = 256
NUM_BANKS = 8

OLD_BASE = 18533        # FC-extended old-domain bank depth (words)
FC_PAD = 3              # [KPAR8 FC-PAD] zero words at 13413..13415
RELOC_APPEND_BASE = OLD_BASE + FC_PAD   # 18536 == 8*2317 (fresh kp8 line)
KP8_BASE_LINES = 2317
OLD_HEX = 72            # 288b line
KP8_TAPS = 8
BIAS_BASE = 91          # FC-extended bias/scale word count


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
    """32 INT8 bytes -> 288b line — identical to extend_mbv2_engine_maps_dw*.py."""
    assert len(bank_bytes) == 32
    return "00000000" + bytes(x & 0xFF for x in reversed(bank_bytes)).hex()


def pack_wide_word(slots32: list[int]) -> str:
    assert len(slots32) == MAC
    raw = bytearray()
    for s in range(MAC - 1, -1, -1):
        raw.extend(struct.pack(">I", slots32[s] & 0xFFFFFFFF))
    return raw.hex()


def derive_appended():
    bank_app: list[list[str]] = [[] for _ in range(NUM_BANKS)]
    bias_app: list[str] = []
    scale_app: list[str] = []
    for mid, c, _ih, _oh in QUARTET:
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
    woff = 0
    boff = 0
    for mid, c, _ih, _oh in QUARTET:
        w, b, s = load_conv(mid, c)
        np_ = passes(c)
        for L in range(MAC * np_):
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
    print("[maps-dw-q] independent lane-major verification PASS "
          f"({woff} wgt words, {boff} bias/scale words, 4 convs)")


def build_kp8_append(old_app: list[str]) -> list[str]:
    """Appended old-domain words start at relocated 18536 = 8*2317 exactly,
    so the kp8 append is simply groups of 8 appended words on fresh lines
    (tail zero-padded) — same text construction as repack_mbv2_kpar8_banks.py:
    line g = tap7 || tap6 || ... || tap0."""
    zero = "0" * OLD_HEX
    n = len(old_app)
    lines = []
    g = 0
    while g * KP8_TAPS < n:
        taps = [(old_app[g * KP8_TAPS + j] if g * KP8_TAPS + j < n else zero)
                for j in range(KP8_TAPS)]
        lines.append("".join(reversed(taps)))
        g += 1
    # re-expansion proof: every tap slice maps back to the appended word
    for gi, ln in enumerate(lines):
        assert len(ln) == KP8_TAPS * OLD_HEX
        for j in range(KP8_TAPS):
            back = ln[(KP8_TAPS - 1 - j) * OLD_HEX:(KP8_TAPS - j) * OLD_HEX]
            w = gi * KP8_TAPS + j
            exp = old_app[w] if w < n else zero
            assert back == exp, f"kp8 re-expansion FAIL line {gi} tap {j}"
    print(f"[maps-dw-q] kp8 append re-expansion PASS ({n} words -> {len(lines)} "
          f"lines, {len(lines)*KP8_TAPS - n} zero tail words)")
    return lines


def extend_file(path: Path, base_count: int, appended: list[str], what: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) == base_count + len(appended):
        assert lines[base_count:] == appended, f"{what}: extended but content differs!"
        print(f"[maps-dw-q] {what}: already extended + content verified ({len(lines)} lines)")
        return
    assert len(lines) == base_count, f"{what}: expected {base_count} lines, got {len(lines)}"
    pre = list(lines)
    out = lines + appended
    path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    post = path.read_text(encoding="utf-8").splitlines()
    assert post[:base_count] == pre, f"{what}: pre-existing lines changed!"
    assert post[base_count:] == appended, f"{what}: appended block mismatch!"
    print(f"[maps-dw-q] {what}: {base_count} -> {len(post)} lines (appended {len(appended)})")


def update_json(path: Path, key_layers: str, new_entries: list[dict],
                totals: dict | None) -> None:
    if not path.is_file():
        print(f"[maps-dw-q] (skip sidecar, missing: {path.name})")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    layers = data.get(key_layers)
    if layers is None:
        print(f"[maps-dw-q] (skip sidecar, no '{key_layers}': {path.name})")
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
        print(f"[maps-dw-q] sidecar updated: {path.name}")
    else:
        print(f"[maps-dw-q] sidecar already current: {path.name}")


def main() -> int:
    bank_app, bias_app, scale_app = derive_appended()
    independent_check(bank_app, bias_app, scale_app)
    kp8_app = [build_kp8_append(bank_app[b]) for b in range(NUM_BANKS)]

    for bank in range(NUM_BANKS):
        extend_file(WDIR / f"uram_weights_bank{bank}.mem", OLD_BASE,
                    bank_app[bank], f"bank{bank}")
        extend_file(WDIR / f"uram_weights_bank{bank}_kp8.mem", KP8_BASE_LINES,
                    kp8_app[bank], f"bank{bank}_kp8")
    extend_file(WDIR / "bias.mem", BIAS_BASE, bias_app, "bias.mem")
    extend_file(WDIR / "scale.mem", BIAS_BASE, scale_app, "scale.mem")

    wmap_entries, bmap_entries = [], []
    wbase, bbase = OLD_BASE, BIAS_BASE
    print("[maps-dw-q] base table (conv: old_wbase RELOC_wbase b/s_base oc_passes):")
    for mid, c, ih, oh in QUARTET:
        np_ = passes(c)
        print(f"  {mid}: w_old={wbase} w_reloc={wbase + FC_PAD} b/s={bbase} passes={np_}")
        wmap_entries.append({
            "module_id": mid,
            "weight_shape": [c, 1, 3, 3],
            "oc_passes": np_,
            "base_mac_cycle": wbase,
            "base_word_kp8_reloc": wbase + FC_PAD,
            "size_mac_cycles": np_ * K_TOTAL,
            "depthwise": True,
            "stride": 2, "ih": ih, "oh": oh,
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

    print(f"[maps-dw-q] DONE: banks {OLD_BASE}->{wbase} (kp8 {KP8_BASE_LINES}->"
          f"{KP8_BASE_LINES + len(kp8_app[0])}), bias/scale {BIAS_BASE}->{bbase}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
