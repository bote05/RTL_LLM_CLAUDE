#!/usr/bin/env python3
"""check_mbv2_act_region_hazards_ext.py — static act-BRAM hazard prover for the
MBV2 engine top under ENGINE/SPATIAL OVERLAP, for the DW-ENGINE STRIDE-1
EXTENSION (up to 46 dispatches; 9 stride-1 DW convs + the P1 trio).

Same two-part proof as scripts/check_mbv2_act_region_hazards.py (P1), with the
BASELINE moved forward to the e2e-proven P1 state (.preext backups, 37
dispatches):

  PART A (strict): every concurrency pair the EXT change TOUCHES — the new DW
    dispatches' read/write regions, their input loaders, their scratch out
    regions, and the fills concurrent with them — must be STRICTLY DISJOINT.

  PART B (equivalence): every INHERITED pair (including the renumbered P1 DW
    trio) must be byte-for-byte the same as in the P1 baseline under the
    dispatch renumber map — or have become strictly disjoint/safer. Their
    safety transfers from P1's green 8/8 e2e.

  C5 lifetime: the new DW regions live in [9368, 24264) — inside the
    frame-start stem/816 loader regions, whose consumers (d0/d1) retire long
    before the first DW fill opens (the SAME lifetime argument that placed
    P1's regions at 8192+). All 24 DW in/out regions are pairwise STRICTLY
    disjoint.

Concurrency model (asserted): while dispatch d runs/drains, the ONLY act-mem
loader receiving beats is dispatch d+1's loader; loaders latch `loaded` after
their full region and never write again (single frame).

Checks: C1 loader==read region; C2 write-vs-fill; C3 read-vs-fill; C4
write-vs-own-read; C5 bounds + DW-region lifetime/disjointness.
Exit 0 = PASS, 1 = FAIL.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOP = REPO / "output" / "mobilenet-v2" / "rtl" / "nn2rtl_top_engine.v"
SCHED = REPO / "output" / "mobilenet-v2" / "rtl" / "nn2rtl_scheduler.v"
TOP_BASE = TOP.with_name("nn2rtl_top_engine.v.preext")
SCHED_BASE = SCHED.with_name("nn2rtl_scheduler.v.preext")

ACT_DEPTH = 25600

MODULES_P1 = [
    "node_conv_814", "node_conv_816", "node_conv_820", "node_conv_822",
    "node_conv_826", "node_conv_828", "node_conv_832", "node_conv_834",
    "node_conv_838", "node_conv_840", "node_conv_844", "node_conv_846",
    "node_conv_850", "node_conv_852", "node_conv_856", "node_conv_858",
    "node_conv_862", "node_conv_864", "node_conv_868", "node_conv_870",
    "node_conv_874", "node_conv_876", "node_conv_880", "node_conv_882",
    "node_conv_886", "node_conv_888", "node_conv_892", "node_conv_894",
    "node_conv_896", "node_conv_898", "node_conv_900", "node_conv_902",
    "node_conv_904", "node_conv_906", "node_conv_908", "node_conv_910",
    "node_conv_912",
]
# EXT: each stride-1 DW conv inserted before its project consumer.
EXT_INSERTS = {                      # succ module -> DW module
    "node_conv_826": "node_conv_824",
    "node_conv_838": "node_conv_836",
    "node_conv_844": "node_conv_842",
    "node_conv_856": "node_conv_854",
    "node_conv_862": "node_conv_860",
    "node_conv_868": "node_conv_866",
    "node_conv_874": "node_conv_872",
    "node_conv_880": "node_conv_878",
    "node_conv_886": "node_conv_884",
}
MODULES_NEW: list[str] = []
for _m in MODULES_P1:
    if _m in EXT_INSERTS:
        MODULES_NEW.append(EXT_INSERTS[_m])
    MODULES_NEW.append(_m)

NEW9 = {i for i, m in enumerate(MODULES_NEW) if m in set(EXT_INSERTS.values())}
P1_DW_NEW = {i for i, m in enumerate(MODULES_NEW)
             if m in ("node_conv_896", "node_conv_902", "node_conv_908")}
DW_ALL = NEW9 | P1_DW_NEW

# old (P1) idx -> new (EXT) idx
OLD_TO_NEW = {}
for _oi, _m in enumerate(MODULES_P1):
    OLD_TO_NEW[_oi] = MODULES_NEW.index(_m)

# expected DW regions (must match the scheduler ROMs; re-asserted below)
DW_IN = {
    "node_conv_824": (9368, 3136),  "node_conv_836": (15640, 784),
    "node_conv_842": (17208, 784),  "node_conv_854": (18776, 392),
    "node_conv_860": (19560, 392),  "node_conv_866": (20344, 392),
    "node_conv_872": (21128, 392),  "node_conv_878": (21912, 588),
    "node_conv_884": (23088, 588),
    "node_conv_896": (8192, 196), "node_conv_902": (8388, 196),
    "node_conv_908": (8584, 196),
}
DW_OUT = {
    "node_conv_824": (12504, 3136), "node_conv_836": (16424, 784),
    "node_conv_842": (17992, 784),  "node_conv_854": (19168, 392),
    "node_conv_860": (19952, 392),  "node_conv_866": (20736, 392),
    "node_conv_872": (21520, 392),  "node_conv_878": (22500, 588),
    "node_conv_884": (23676, 588),
    "node_conv_896": (8780, 196), "node_conv_902": (8976, 196),
    "node_conv_908": (9172, 196),
}

fails: list[str] = []


def fail(msg: str) -> None:
    fails.append(msg)
    print(f"  FAIL: {msg}")


def overlap(a, b) -> bool:
    return a[0] < b[0] + b[1] and b[0] < a[0] + a[1]


def parse_rom(text: str, name: str, n: int) -> dict[int, int]:
    out = {int(i): int(v)
           for i, v in re.findall(rf"6'd(\d+): {name} = \d+'d(\d+);", text)}
    if len(out) != n:
        raise SystemExit(f"ROM {name}: {len(out)} entries != {n}")
    return out


def parse_loaders(top: str) -> dict[str, dict]:
    loaders: dict[str, dict] = {}
    pat = re.compile(
        r"(?:tiled_)?stream_to_act_bram_bridge #\((?P<params>.*?)"
        r"\)\s*(?P<inst>u_ldr_node_conv_\d+)\s*\(\s*"
        r"\.clk\(clk\), \.rst_n\(rst_n\),\s*"
        r"\.in_valid\((?P<invalid>[^)]*)\),", re.S)
    for m in pat.finditer(top):
        params = m.group("params")
        base = int(re.search(r"\.BRAM_BASE_ADDR\((\d+)\)", params).group(1))
        words = int(re.search(r"\.TOTAL_BRAM_WORDS\((\d+)\)", params).group(1))
        wr = re.search(r"\.wr_req\((\w+)_wr_req\)", top[m.start():m.start() + 2400])
        loaders[m.group("inst")] = {
            "base": base, "words": words,
            "in_valid": m.group("invalid").strip(),
            "wr_sig": wr.group(1) if wr else "?",
        }
    return loaders


def build_table(top_text: str, sched_text: str, modules: list[str]):
    n = len(modules)
    ic = parse_rom(sched_text, "channel_in_rom", n)
    oc = parse_rom(sched_text, "channel_out_rom", n)
    kh = parse_rom(sched_text, "kernel_h_rom", n)
    ih = parse_rom(sched_text, "input_h_rom", n)
    iw = parse_rom(sched_text, "input_w_rom", n)
    oh = parse_rom(sched_text, "output_h_rom", n)
    ow = parse_rom(sched_text, "output_w_rom", n)
    rbase = parse_rom(sched_text, "act_in_base_word_rom", n)
    wbase = parse_rom(sched_text, "act_out_base_word_rom", n)

    loaders = parse_loaders(top_text)
    disp_loader: dict[int, str | None] = {}
    for d in range(n):
        m = re.search(rf"assign all_loaded\[{d}\] = (\S+?);", top_text)
        if not m:
            raise SystemExit(f"all_loaded[{d}] missing")
        sig = m.group(1)
        if sig == "1'b1":
            disp_loader[d] = None
            continue
        prefix = sig.removesuffix("_loaded")
        inst = next((k for k, v in loaders.items() if v["wr_sig"] == prefix), None)
        if inst is None:
            raise SystemExit(f"all_loaded[{d}]={sig}: no matching loader instance")
        disp_loader[d] = inst

    table = {}
    for d in range(n):
        rwpp = math.ceil(ic[d] / 256)
        wwpp = math.ceil(oc[d] / 256)
        read = (rbase[d], ih[d] * iw[d] * rwpp)
        write = (wbase[d], oh[d] * ow[d] * wwpp)
        table[d] = {
            "module": modules[d], "kh": kh[d], "rwpp": rwpp, "wwpp": wwpp,
            "oh_ow": oh[d] * ow[d], "read": read, "write": write,
            "loader": disp_loader[d],
        }
    return table, loaders


def classify(d: int, t: dict, loaders: dict, n: int):
    e = t[d]
    if not overlap(e["write"], e["read"]):
        v_wr = "disjoint"
    elif e["kh"] == 1 and e["write"][0] == e["read"][0] and e["wwpp"] <= e["rwpp"]:
        v_wr = "in-place-1x1"
    else:
        v_wr = "RATE"
    fill = t[d + 1]["loader"] if d + 1 < n else None
    v_rf = v_wf = "none"
    if fill:
        L = loaders[fill]
        lrg = (L["base"], L["words"])
        if not overlap(e["read"], lrg):
            v_rf = "disjoint"
        elif (e["kh"] == 1 and lrg[0] == e["read"][0]
              and L["words"] % e["oh_ow"] == 0
              and L["words"] // e["oh_ow"] <= e["rwpp"]):
            v_rf = "lag-safe-1x1"
        else:
            v_rf = "RATE"
        if not overlap(e["write"], lrg):
            v_wf = "disjoint"
        elif e["write"] == lrg and f"{e['module']}_valid_out" in L["in_valid"]:
            v_wf = "redundant-copy"
        else:
            v_wf = "RATE"
    return v_wr, v_rf, v_wf, fill


def main() -> int:
    for p in [TOP, SCHED, TOP_BASE, SCHED_BASE]:
        if not p.is_file():
            raise SystemExit(f"missing artifact: {p}")
    new_t, new_l = build_table(TOP.read_text(encoding="utf-8"),
                               SCHED.read_text(encoding="utf-8"), MODULES_NEW)
    old_t, old_l = build_table(TOP_BASE.read_text(encoding="utf-8"),
                               SCHED_BASE.read_text(encoding="utf-8"), MODULES_P1)
    print(f"[hazard-ext] parsed: baseline (P1) 37 dispatches / {len(old_l)} loaders, "
          f"EXT {len(MODULES_NEW)} dispatches / {len(new_l)} loaders")

    n_new, n_old = len(MODULES_NEW), len(MODULES_P1)

    # sanity: loader naming + DW geometry + depthwise rom
    sched_text = SCHED.read_text(encoding="utf-8")
    dwbits = {int(i): int(v) for i, v in
              re.findall(r"6'd(\d+): depthwise_rom = 1'b(\d);", sched_text)}
    for d in range(n_new):
        want = 1 if d in DW_ALL else 0
        if dwbits.get(d, 0) != want:
            fail(f"depthwise_rom[{d}]={dwbits.get(d,0)} want {want}")
        ld = new_t[d]["loader"]
        if ld is not None and ld != f"u_ldr_{MODULES_NEW[d]}":
            fail(f"d{d} {MODULES_NEW[d]}: all_loaded maps to {ld}")
    d876 = MODULES_NEW.index("node_conv_876")
    if new_t[d876]["loader"] is not None:
        fail(f"d{d876} (conv_876) should be loader-less")
    for d in DW_ALL:
        mod = MODULES_NEW[d]
        e = new_t[d]
        if not (e["kh"] == 3 and e["read"] == DW_IN[mod] and e["write"] == DW_OUT[mod]):
            fail(f"DW d{d} {mod}: kh={e['kh']} read={e['read']} write={e['write']}")

    # C5: bounds; DW region lifetime-disjointness
    print("[hazard-ext] C5: bounds + DW region lifetime")
    for d in range(n_new):
        for tag in ["read", "write"]:
            b, w = new_t[d][tag]
            if not (0 <= b and b + w <= ACT_DEPTH):
                fail(f"d{d} {tag} [{b},+{w}) outside act mem")
    dwregs = list(DW_IN.values()) + list(DW_OUT.values())
    for i in range(len(dwregs)):
        for j in range(i + 1, len(dwregs)):
            if overlap(dwregs[i], dwregs[j]):
                fail(f"DW regions {dwregs[i]} / {dwregs[j]} overlap")
    # DW regions vs every loader region: allowed only when the other loader's
    # consumer dispatch RETIRES before the DW fill window opens (fill of DW
    # dispatch d starts during d-1).
    for dwd in sorted(DW_ALL):
        fill_start = dwd - 1
        mod = MODULES_NEW[dwd]
        for reg, kind in [(DW_IN[mod], "in"), (DW_OUT[mod], "out")]:
            for od in range(n_new):
                ld = new_t[od]["loader"]
                if ld is None or od in DW_ALL:
                    continue
                L = new_l[ld]
                if overlap(reg, (L["base"], L["words"])):
                    if od < fill_start:
                        continue   # consumed before the DW region is first written
                    fail(f"DW d{dwd} {kind} region {reg} overlaps live loader "
                         f"{ld} (consumer d{od})")

    # C1: loader region == read region
    print("[hazard-ext] C1: loader region == engine read region")
    for d in range(n_new):
        ld = new_t[d]["loader"]
        if ld is None:
            rr, wr_pred = new_t[d]["read"], new_t[d - 1]["write"]
            if not (wr_pred[0] <= rr[0] and rr[0] + rr[1] <= wr_pred[0] + wr_pred[1]):
                fail(f"d{d} (loader-less) read {rr} outside d{d-1} write {wr_pred}")
            continue
        L = new_l[ld]
        if (L["base"], L["words"]) != new_t[d]["read"]:
            fail(f"d{d}: loader [{L['base']},+{L['words']}) != read {new_t[d]['read']}")

    # model assertion: every DW-chain loader source is exactly as designed
    expect_src = {
        # EXT nine (producer relu of each moved conv)
        "u_ldr_node_conv_824": "n4_5_valid_out & spatial_run",
        "u_ldr_node_conv_836": "n4_9_valid_out & spatial_run",
        "u_ldr_node_conv_842": "n4_11_valid_out & spatial_run",
        "u_ldr_node_conv_854": "n4_15_valid_out & spatial_run",
        "u_ldr_node_conv_860": "n4_17_valid_out & spatial_run",
        "u_ldr_node_conv_866": "n4_19_valid_out & spatial_run",
        "u_ldr_node_conv_872": "n4_21_valid_out & spatial_run",
        "u_ldr_node_conv_878": "n4_23_valid_out & spatial_run",
        "u_ldr_node_conv_884": "n4_25_valid_out & spatial_run",
        # inherited P1 set
        "u_ldr_node_conv_896": "n4_29_valid_out & spatial_run",
        "u_ldr_node_conv_898": "n4_30_valid_out & spatial_run",
        "u_ldr_node_conv_902": "n4_31_valid_out & spatial_run",
        "u_ldr_node_conv_904": "n4_32_valid_out & spatial_run",
        "u_ldr_node_conv_908": "n4_33_valid_out & spatial_run",
        "u_ldr_node_conv_910": "n4_34_valid_out & spatial_run",
    }
    for inst, src in expect_src.items():
        if new_l[inst]["in_valid"] != src:
            fail(f"{inst} in_valid '{new_l[inst]['in_valid']}' != '{src}'")

    # PART A (strict for the NINE) + PART B (P1-baseline equivalence)
    print("[hazard-ext] C2/C3/C4 per-dispatch verdicts (A=strict for EXT-touched, "
          "B=P1-baseline-equivalent for inherited)")
    old_verdicts = {od: classify(od, old_t, old_l, n_old) for od in range(n_old)}
    new_to_old = {v: k for k, v in OLD_TO_NEW.items()}
    print(f"  {'d':>2} {'module':<10} {'read':>15} {'write':>15} {'fill(d+1)':<24} wxr / rxf / wxf")
    for d in range(n_new):
        v_wr, v_rf, v_wf, fill = classify(d, new_t, new_l, n_new)
        touched = (d in NEW9) or (fill is not None and d + 1 in NEW9)
        verdicts = (v_wr, v_rf, v_wf)
        if touched:
            ok_strict = {"disjoint", "none"}
            for tag, v in zip(["wxr", "rxf", "wxf"], verdicts):
                if v not in ok_strict:
                    fail(f"d{d} (EXT-touched) {tag} verdict '{v}' — must be disjoint")
            cls = "A"
        else:
            od = new_to_old.get(d)
            if od is None:
                fail(f"d{d}: no baseline mapping and not EXT-touched")
                cls = "?"
            else:
                base_v = old_verdicts[od]
                for tag, vn, vo in zip(["wxr", "rxf", "wxf"], verdicts, base_v[:3]):
                    if vn == vo or vn in ("disjoint", "none"):
                        continue
                    fail(f"d{d} {tag}: verdict '{vn}' differs from baseline d{od} '{vo}'")
                for k in ["read", "write", "kh", "rwpp", "wwpp"]:
                    if new_t[d][k] != old_t[od][k]:
                        fail(f"d{d} {k} {new_t[d][k]} != baseline d{od} {old_t[od][k]}")
                cls = "B"
        e = new_t[d]
        print(f"  {d:>2} {e['module'].removeprefix('node_'):<10} "
              f"[{e['read'][0]:>5},+{e['read'][1]:<5}) [{e['write'][0]:>5},+{e['write'][1]:<5}) "
              f"{(fill or '-'):<24} {v_wr} / {v_rf} / {v_wf}  [{cls}]")

    rate_pairs = [(d,) + classify(d, new_t, new_l, n_new)[:3] for d in range(n_new)]
    n_rate = sum(1 for r in rate_pairs if "RATE" in r[1:])
    print(f"[hazard-ext] rate-bounded inherited pairs: {n_rate} "
          f"(each verified identical to the e2e-proven P1 baseline)")

    if fails:
        print(f"\n[hazard-ext] RESULT: FAIL ({len(fails)} violations)")
        return 1
    print("\n[hazard-ext] RESULT: PASS — PART A: all EXT-touched pairs strictly disjoint;"
          "\n         PART B: all inherited pairs identical to the e2e-proven P1 baseline"
          "\n         (or strictly safer); C1/C5 region+lifetime invariants hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
