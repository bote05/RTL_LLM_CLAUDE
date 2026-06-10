#!/usr/bin/env python3
"""extend_mbv2_engine_maps_fc.py — MBV2 "FC-ON-ENGINE" map extension (2026-06-10)

Appends node_linear (the FC/Gemm classifier: M=1000 outputs x K=1280 inputs)
to the MBV2 shared-engine weight banks + bias/scale ROM images so it can run
as ONE dense engine dispatch (a 1x1 conv over a 1x1 "image": IC=1280, OC=1000,
oc_passes=4, k_total=1280).

Extends the DW-EXT state (banks 13413 deep, bias/scale 87 words — built by
scripts/extend_mbv2_engine_maps_dw_ext.py) APPEND-ONLY:

  module        IC    OC   oc_passes  wgt words  wgt base   bias/scale @ base
  node_linear  1280  1000      4        5120      13413          4 @ 87
                              sum:      5120 -> banks 18533;  4 -> bias/scale 91

Layouts are IDENTICAL to the dense dispatches (same encoders as the P1/EXT
scripts / build_weight_memory_map.py):
  * WEIGHT BANKS: word (13413 + p*1280 + k), lane L (bank L/32, byte L%32) =
    W[oc=256p+L][ic=k] = node_linear_weights.hex row-major flat[(256p+L)*1280+k];
    ZERO for 256p+L >= 1000 (24 dead lanes on pass 3). Matches the DENSE
    address_generator walk for KH=KW=1: weight_word = base + oc_pass*K_TOTAL
    + ic, K_TOTAL = cfg_ic = 1280.
  * BIAS: word (87 + p) slot L = node_linear_bias.hex[256p+L] (INT32), dead = 0.
  * SCALE: word (87 + p) slot L = mult' = SCALE_MULT << (FIXED_SHIFT -
    SCALE_SHIFT) = 4071 << (23 - 20) = 32568 (constant per-TENSOR scale from
    node_linear.v MULT=4071/SHIFT=20, pre-widened for the engine's
    constant-shift requant >>>23), dead = 0.

REQUANT IDENTITY (proven in docs/agent_tasks/FC_ENGINE_ANALYSIS.md and
re-asserted empirically below with --verify-requant): for every integer B,
  (B*4071 + 2^19) >>> 20  ==  (B*(4071<<3) + 2^22) >>> 23      (exact)
so the engine's requant_pipeline output is byte-identical to node_linear.v's.

PROOF + idempotency: same two-pass scheme as P1/EXT — derive (pass-major),
verify with an INDEPENDENT lane-major walk, append with prefix/suffix re-read
asserts; if already extended, re-derive + byte-compare + exit 0.

Run from anywhere: paths are Path(__file__)-relative (worktree-safe).
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WDIR = REPO / "output" / "mobilenet-v2" / "weights"

MID = "node_linear"
K = 1280                  # input features  (= engine k_total, dense 1x1)
M = 1000                  # output logits   (= cfg_oc; lanes 1000..1023 dead)
MAC = 256
NUM_BANKS = 8
OC_PASSES = (M + MAC - 1) // MAC     # 4
BANK_BASE = 13413         # DW-EXT-extended bank depth (words)
BIAS_BASE = 87            # DW-EXT-extended bias/scale word count
WGT_WORDS = OC_PASSES * K  # 5120 -> banks 18533

# node_linear.v requant constants (per-TENSOR): (x*4071 + 2^19) >>> 20.
SCALE_MULT = 4071
SCALE_SHIFT = 20
FIXED_SHIFT = 23          # engine requant_pipeline FIXED_SHIFT
MULT_PRIME = SCALE_MULT << (FIXED_SHIFT - SCALE_SHIFT)   # 32568
assert 0 < MULT_PRIME < (1 << 31), "mult' must fit the [30:0] scale slot"


def read_hex_lines(path: Path) -> list[int]:
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        t = ln.strip()
        if not t or t.startswith("//"):
            continue
        out.append(int(t, 16))
    return out


def load_fc():
    w = read_hex_lines(WDIR / f"{MID}_weights.hex")   # row-major [M][K] INT8
    b = read_hex_lines(WDIR / f"{MID}_bias.hex")      # [M] INT32
    assert len(w) == M * K, f"{MID}: weights {len(w)} != {M*K}"
    assert len(b) == M, f"{MID}: bias {len(b)} != {M}"
    return w, b


def encode_bank_line(bank_bytes: list[int]) -> str:
    """32 INT8 bytes -> 288b line ('00000000' + reversed-bytes hex) —
    identical to build_weight_memory_map.py / extend_mbv2_engine_maps_dw*.py."""
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
    w, b = load_fc()
    bank_app: list[list[str]] = [[] for _ in range(NUM_BANKS)]
    # DENSE walk order: word index = p*K + k (oc_pass-major, ic inner) —
    # exactly address_generator's weight_offset for KH=KW=1.
    for p in range(OC_PASSES):
        for k in range(K):
            for bank in range(NUM_BANKS):
                bb = []
                for slot in range(32):
                    oc = p * MAC + bank * 32 + slot
                    bb.append(w[oc * K + k] if oc < M else 0)
                bank_app[bank].append(encode_bank_line(bb))
    bias_app = [pack_wide_word(
        [(b[p * MAC + L] if p * MAC + L < M else 0) for L in range(MAC)])
        for p in range(OC_PASSES)]
    scale_app = [pack_wide_word(
        [(MULT_PRIME if p * MAC + L < M else 0) for L in range(MAC)])
        for p in range(OC_PASSES)]
    return bank_app, bias_app, scale_app


def independent_check(bank_app, bias_app, scale_app) -> None:
    """Second-pass verification with INDEPENDENT indexing (lane-major)."""
    w, b = load_fc()
    for L in range(MAC * OC_PASSES):              # global lane walk 0..1023
        p, lane = divmod(L, MAC)
        bank, slot = divmod(lane, 32)
        oc = p * MAC + lane
        bword = bias_app[p]
        sword = scale_app[p]
        off = (MAC - 1 - lane) * 8
        bgot = int(bword[off:off + 8], 16)
        sgot = int(sword[off:off + 8], 16)
        bexp = (b[oc] & 0xFFFFFFFF) if oc < M else 0
        sexp = MULT_PRIME if oc < M else 0
        assert bgot == bexp, f"bias lane {lane} pass {p}: {bgot:#x}!={bexp:#x}"
        assert sgot == sexp, f"scale lane {lane} pass {p}: {sgot:#x}!={sexp:#x}"
        # lane-major weight walk: spot the FULL k range for this lane.
        for k in range(K):
            line = bank_app[bank][p * K + k]
            hoff = len(line) - (slot + 1) * 2
            got = int(line[hoff:hoff + 2], 16)
            exp = (w[oc * K + k] & 0xFF) if oc < M else 0
            assert got == exp, f"w lane {lane} p{p} k{k}: {got:#x}!={exp:#x}"
    print(f"[maps-fc] independent lane-major verification PASS "
          f"({WGT_WORDS} wgt words, {OC_PASSES} bias/scale words, 1024 lanes x {K} taps)")


def verify_requant_identity() -> None:
    """Empirical requant-identity re-assert vs the INTEGER golden (8 vectors):
    node_linear.v's (acc+bias)*4071 +2^19 >>>20 clamp == the engine's
    (acc+bias)*mult' +2^22 >>>23 clamp == node_linear.goldout, byte-exact."""
    gdir = REPO / "output" / "mobilenet-v2" / "goldens"
    def load_nn2v(p: Path):
        d = p.read_bytes()
        _, nvec, samples, bps = struct.unpack("<4I", d[4:20])
        pay = d[20:]
        return [pay[v * samples * bps:(v + 1) * samples * bps] for v in range(nvec)]
    gin = load_nn2v(gdir / "node_linear.goldin")
    gout = load_nn2v(gdir / "node_linear.goldout")
    w, b = load_fc()
    ws = [x - 256 if x >= 128 else x for x in w]
    bs = [x - (1 << 32) if x >= (1 << 31) else x for x in b]
    clamp = lambda v: 127 if v > 127 else (-128 if v < -128 else v)
    s8 = lambda x: x - 256 if x >= 128 else x
    for v in range(len(gin)):
        x = [s8(by) for by in gin[v]]
        ref = [s8(by) for by in gout[v]]
        for m in range(M):
            acc = sum(x[k] * ws[m * K + k] for k in range(K))
            biased = acc + bs[m]
            v_nl = (biased * SCALE_MULT + (1 << (SCALE_SHIFT - 1))) >> SCALE_SHIFT
            v_en = (biased * MULT_PRIME + (1 << (FIXED_SHIFT - 1))) >> FIXED_SHIFT
            assert v_nl == v_en, f"identity break vec{v} m{m}: {v_nl} != {v_en}"
            assert clamp(v_en) == ref[m], f"vs golden vec{v} m{m}: {clamp(v_en)} != {ref[m]}"
    print(f"[maps-fc] REQUANT IDENTITY re-asserted: engine formula == node_linear "
          f"formula == integer golden, {len(gin)} vectors x {M} logits, byte-exact")


def extend_file(path: Path, base_count: int, appended: list[str], what: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) == base_count + len(appended):
        assert lines[base_count:] == appended, f"{what}: extended but content differs!"
        print(f"[maps-fc] {what}: already extended + content verified ({len(lines)} lines)")
        return
    assert len(lines) == base_count, f"{what}: expected {base_count} lines, got {len(lines)}"
    pre = list(lines)
    out = lines + appended
    path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    post = path.read_text(encoding="utf-8").splitlines()
    assert post[:base_count] == pre, f"{what}: pre-existing lines changed!"
    assert post[base_count:] == appended, f"{what}: appended block mismatch!"
    print(f"[maps-fc] {what}: {base_count} -> {len(post)} lines (appended {len(appended)})")


def update_json(path: Path, key_layers: str, new_entries: list[dict],
                totals: dict | None) -> None:
    if not path.is_file():
        print(f"[maps-fc] (skip sidecar, missing: {path.name})")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    layers = data.get(key_layers)
    if layers is None:
        print(f"[maps-fc] (skip sidecar, no '{key_layers}': {path.name})")
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
        print(f"[maps-fc] sidecar updated: {path.name}")
    else:
        print(f"[maps-fc] sidecar already current: {path.name}")


def main() -> int:
    if "--verify-requant" in sys.argv:
        verify_requant_identity()

    bank_app, bias_app, scale_app = derive_appended()
    independent_check(bank_app, bias_app, scale_app)

    for bank in range(NUM_BANKS):
        extend_file(WDIR / f"uram_weights_bank{bank}.mem", BANK_BASE,
                    bank_app[bank], f"bank{bank}")
    extend_file(WDIR / "bias.mem", BIAS_BASE, bias_app, "bias.mem")
    extend_file(WDIR / "scale.mem", BIAS_BASE, scale_app, "scale.mem")

    wmap_entry = {
        "module_id": MID,
        "weight_shape": [M, K, 1, 1],
        "oc_passes": OC_PASSES,
        "base_mac_cycle": BANK_BASE,
        "size_mac_cycles": WGT_WORDS,
        "depthwise": False,
        "fc": True,
    }
    bmap_entry = {
        "module_id": MID,
        "oc": M,
        "oc_passes": OC_PASSES,
        "base_word": BIAS_BASE,
        "depthwise": False,
        "fc": True,
    }
    print(f"[maps-fc] base table: {MID}: w={BANK_BASE} b/s={BIAS_BASE} "
          f"passes={OC_PASSES} mult'={MULT_PRIME}")
    update_json(WDIR / "weight_memory_map.json", "layers", [wmap_entry],
                {"total_mac_cycles": BANK_BASE + WGT_WORDS})
    update_json(WDIR / "bias_memory_map.json", "layers", [bmap_entry], None)
    update_json(WDIR / "scale_memory_map.json", "layers", [bmap_entry], None)

    print(f"[maps-fc] DONE: banks {BANK_BASE}->{BANK_BASE + WGT_WORDS}, "
          f"bias/scale {BIAS_BASE}->{BIAS_BASE + OC_PASSES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
