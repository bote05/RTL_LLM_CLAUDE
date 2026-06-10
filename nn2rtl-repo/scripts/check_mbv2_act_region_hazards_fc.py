#!/usr/bin/env python3
"""check_mbv2_act_region_hazards_fc.py — static act-BRAM hazard prover for the
MBV2 engine top after FC-ON-ENGINE (47 dispatches; node_linear appended as
dense dispatch 46).

Same two-part proof as the P1/EXT provers, with the BASELINE moved forward to
the e2e-proven DW-EXT state (.prefc backups, 46 dispatches):

  PART A (strict): every concurrency pair the FC change TOUCHES — the FC
    dispatch's read region [25088,+5), its scratch write [25093,+4), and its
    input-loader fill (which opens during dispatch 45) — must be STRICTLY
    DISJOINT from everything it can be concurrent with. This is trivially
    strong here: the FC regions sit ABOVE the GLOBAL act-mem maximum ever
    used (25088 = the frame-start d0-write/d1 region top, asserted from the
    baseline) and below ACT_DEPTH. NO lifetime argument is needed at all
    (unlike the P1/EXT DW regions, which overlay [12544,25088) and lean on
    the d0/d1-retire-before-fill lifetime rule).

  PART B (equivalence): every INHERITED pair (dispatches 0..45 — NO renumber:
    the FC is appended) must be byte-for-byte the same as in the DW-EXT
    baseline. Their safety transfers from EXT's green 8/8 e2e.

Concurrency model (asserted): while dispatch d runs/drains, the ONLY act-mem
loader receiving beats is dispatch d+1's loader; loaders latch `loaded` after
their full region and never write again (single frame). The FC loader's
producer (node_mean) only emits after dispatch 45's full output has drained
through n4_35 -> br_mean -> node_mean, so its fill can never precede d45.

Checks: C1 loader==read region; C2 write-vs-fill; C3 read-vs-fill; C4
write-vs-own-read; C5 bounds + FC-region above-all-disjointness.
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
TOP_BASE = TOP.with_name("nn2rtl_top_engine.v.prefc")
SCHED_BASE = SCHED.with_name("nn2rtl_scheduler.v.prefc")

ACT_DEPTH = 25600
FC_IN = (25088, 5)
FC_OUT = (25093, 4)

# The DW-EXT dispatch order (46 modules) — the FC is APPENDED (no renumber).
MODULES_EXT = [
    "node_conv_814", "node_conv_816", "node_conv_820", "node_conv_822",
    "node_conv_824", "node_conv_826", "node_conv_828", "node_conv_832",
    "node_conv_834", "node_conv_836", "node_conv_838", "node_conv_840",
    "node_conv_842", "node_conv_844", "node_conv_846", "node_conv_850",
    "node_conv_852", "node_conv_854", "node_conv_856", "node_conv_858",
    "node_conv_860", "node_conv_862", "node_conv_864", "node_conv_866",
    "node_conv_868", "node_conv_870", "node_conv_872", "node_conv_874",
    "node_conv_876", "node_conv_878", "node_conv_880", "node_conv_882",
    "node_conv_884", "node_conv_886", "node_conv_888", "node_conv_892",
    "node_conv_894", "node_conv_896", "node_conv_898", "node_conv_900",
    "node_conv_902", "node_conv_904", "node_conv_906", "node_conv_908",
    "node_conv_910", "node_conv_912",
]
MODULES_NEW = MODULES_EXT + ["node_linear"]
FC_D = 46
DW_IDX = {4, 9, 12, 17, 20, 23, 26, 29, 32, 37, 40, 43}   # EXT depthwise rows

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
        r"\)\s*(?P<inst>u_ldr_(?:node_conv_\d+|node_linear))\s*\(\s*"
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
                               SCHED_BASE.read_text(encoding="utf-8"), MODULES_EXT)
    print(f"[hazard-fc] parsed: baseline (DW-EXT) 46 dispatches / {len(old_l)} loaders, "
          f"FC {len(MODULES_NEW)} dispatches / {len(new_l)} loaders")

    n_new, n_old = len(MODULES_NEW), len(MODULES_EXT)

    # sanity: FC row geometry, depthwise rom, loader naming
    sched_text = SCHED.read_text(encoding="utf-8")
    dwbits = {int(i): int(v) for i, v in
              re.findall(r"6'd(\d+): depthwise_rom = 1'b(\d);", sched_text)}
    for d in range(n_new):
        want = 1 if d in DW_IDX else 0
        if dwbits.get(d, 0) != want:
            fail(f"depthwise_rom[{d}]={dwbits.get(d,0)} want {want} "
                 f"(FC must be DENSE)")
    e = new_t[FC_D]
    if not (e["kh"] == 1 and e["read"] == FC_IN and e["write"] == FC_OUT
            and e["rwpp"] == 5 and e["wwpp"] == 4 and e["oh_ow"] == 1):
        fail(f"FC d{FC_D}: kh={e['kh']} read={e['read']} write={e['write']} "
             f"rwpp={e['rwpp']} wwpp={e['wwpp']} px={e['oh_ow']}")
    if new_t[FC_D]["loader"] != "u_ldr_node_linear":
        fail(f"all_loaded[{FC_D}] maps to {new_t[FC_D]['loader']}")
    if new_l["u_ldr_node_linear"]["in_valid"] != "node_mean_valid_out & spatial_run":
        fail(f"FC loader in_valid '{new_l['u_ldr_node_linear']['in_valid']}'")

    # C5: bounds + FC regions strictly ABOVE every baseline-used word.
    print("[hazard-fc] C5: bounds + FC above-all strict disjointness")
    for d in range(n_new):
        for tag in ["read", "write"]:
            b, w = new_t[d][tag]
            if not (0 <= b and b + w <= ACT_DEPTH):
                fail(f"d{d} {tag} [{b},+{w}) outside act mem")
    base_max = 0
    for d in range(n_old):
        for tag in ["read", "write"]:
            b, w = old_t[d][tag]
            base_max = max(base_max, b + w)
    for L in old_l.values():
        base_max = max(base_max, L["base"] + L["words"])
    print(f"  baseline max used act word (exclusive) = {base_max}; "
          f"FC regions = [{FC_IN[0]},+{FC_IN[1]}) [{FC_OUT[0]},+{FC_OUT[1]})")
    if FC_IN[0] < base_max:
        fail(f"FC act_in base {FC_IN[0]} < baseline max {base_max}")
    if overlap(FC_IN, FC_OUT):
        fail("FC in/out regions overlap each other")
    # exhaustive: FC regions vs EVERY new-table region and loader
    for d in range(n_new - 1):
        for tag in ["read", "write"]:
            for reg in [FC_IN, FC_OUT]:
                if overlap(new_t[d][tag], reg):
                    fail(f"FC region {reg} overlaps d{d} {tag} {new_t[d][tag]}")
    for inst, L in new_l.items():
        if inst == "u_ldr_node_linear":
            continue
        for reg in [FC_IN, FC_OUT]:
            if overlap((L["base"], L["words"]), reg):
                fail(f"FC region {reg} overlaps loader {inst}")

    # C1: loader region == read region
    print("[hazard-fc] C1: loader region == engine read region")
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

    # PART A (strict for FC-touched pairs) + PART B (EXT-baseline equivalence)
    print("[hazard-fc] C2/C3/C4 per-dispatch verdicts (A=strict for FC-touched, "
          "B=EXT-baseline-identical for inherited; identity dispatch map)")
    old_verdicts = {od: classify(od, old_t, old_l, n_old) for od in range(n_old)}
    print(f"  {'d':>2} {'module':<12} {'read':>15} {'write':>15} {'fill(d+1)':<22} wxr / rxf / wxf")
    for d in range(n_new):
        v_wr, v_rf, v_wf, fill = classify(d, new_t, new_l, n_new)
        verdicts = (v_wr, v_rf, v_wf)
        touched = (d == FC_D) or (d + 1 == FC_D)
        if touched:
            # d=45: its OWN wxr verdict is inherited (its fill is the FC's);
            # the FC-introduced pairs are rxf/wxf (45 vs FC fill) and all of d=46.
            ok_strict = {"disjoint", "none"}
            check = ["wxr", "rxf", "wxf"] if d == FC_D else ["rxf", "wxf"]
            for tag, v in zip(["wxr", "rxf", "wxf"], verdicts):
                if tag in check and v not in ok_strict:
                    fail(f"d{d} (FC-touched) {tag} verdict '{v}' — must be disjoint")
            if d + 1 == FC_D:
                # inherited part of d=45 (wxr) must equal baseline
                if v_wr != old_verdicts[d][0] and v_wr not in ("disjoint",):
                    fail(f"d{d} wxr '{v_wr}' != baseline '{old_verdicts[d][0]}'")
                for k in ["read", "write", "kh", "rwpp", "wwpp"]:
                    if new_t[d][k] != old_t[d][k]:
                        fail(f"d{d} {k} {new_t[d][k]} != baseline {old_t[d][k]}")
            cls = "A"
        else:
            base_v = old_verdicts[d]
            for tag, vn, vo in zip(["wxr", "rxf", "wxf"], verdicts, base_v[:3]):
                if vn == vo or vn in ("disjoint", "none"):
                    continue
                fail(f"d{d} {tag}: verdict '{vn}' differs from baseline '{vo}'")
            for k in ["read", "write", "kh", "rwpp", "wwpp"]:
                if new_t[d][k] != old_t[d][k]:
                    fail(f"d{d} {k} {new_t[d][k]} != baseline {old_t[d][k]}")
            if new_t[d]["loader"] != old_t[d]["loader"]:
                fail(f"d{d} loader {new_t[d]['loader']} != baseline {old_t[d]['loader']}")
            cls = "B"
        e = new_t[d]
        print(f"  {d:>2} {e['module'].removeprefix('node_'):<12} "
              f"[{e['read'][0]:>5},+{e['read'][1]:<5}) [{e['write'][0]:>5},+{e['write'][1]:<5}) "
              f"{(fill or '-'):<22} {v_wr} / {v_rf} / {v_wf}  [{cls}]")

    if fails:
        print(f"\n[hazard-fc] RESULT: FAIL ({len(fails)} violations)")
        return 1
    print("\n[hazard-fc] RESULT: PASS — PART A: all FC-touched pairs strictly disjoint"
          "\n         (FC regions sit ABOVE the GLOBAL baseline act-mem maximum — no"
          "\n         lifetime argument needed); PART B: all 46 inherited dispatch rows"
          "\n         byte-identical to the e2e-proven DW-EXT baseline; C1/C5 hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
