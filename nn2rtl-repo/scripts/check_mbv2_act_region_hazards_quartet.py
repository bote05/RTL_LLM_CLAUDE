#!/usr/bin/env python3
"""check_mbv2_act_region_hazards_quartet.py — static act-BRAM hazard prover
for the MBV2 engine top after DW-QUARTET (stride-2 DW convs 818/830/848/890
inserted as engine dispatches; works for any applied subset — the applied set
is DETECTED from the deployed top).

Same two-part proof as the P1/EXT/FC provers, baseline moved forward to the
e2e-proven FC/ENG-PIPE state (.prequartet backups, 47 dispatches):

  PART A (touched pairs): for each inserted conv at slot s, the NEW pairs are
    (a) dispatch s-1's read/write vs the conv's loader fill (which opens
        during s-1's run/drain), and (b) ALL of dispatch s's own pairs.
    Verdicts must be STRICTLY DISJOINT, except the 818 in-place overlay:
      * d(816) read vs ldr_dw818 fill [12544,+12544): the established
        "lag-safe-1x1" class (fill word i derives from d's own beat i ->
        arrives only after the 1x1 walk read word i; e2e-proven rule).
      * d(816) write vs ldr_dw818 fill: NEW class "lag-safe-inplace-fill" —
        write region == read region == fill region, 1x1, and the loader's
        in_valid TRACES to d's own bridge stream (checked structurally
        through the relu stage). Engine act write of word i happens the SAME
        cycle as the FIFO push of beat i (engine_act_wr_commit =
        act_out_wr_en & eofifo_in_ready in the top), and the loader's write
        of word i flows through bridge->relu->arbiter >= 3 cycles later, so
        engine-write(i) < loader-write(i) ALWAYS; the final content is the
        relu'd copy dispatch s then reads. The engine never re-reads word i
        after pixel i (monotonic 1x1 walk) — same lag argument as
        lag-safe-1x1, applied to the write port.

  PART B (equivalence): every INHERITED dispatch row (renumbered by the
    insertion map) must be byte-for-byte identical to the FC baseline; its
    verdict may only change to something STRICTLY safer (disjoint/none).

Checks: C1 loader==read region; C2/C3/C4 the per-dispatch pair verdicts;
C5 bounds (ACT_DEPTH=25600) + stride-2 geometry sanity (ih=2*oh, read words
= IH*IW*chunks == loader TOTAL). Exit 0 = PASS, 1 = FAIL.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOP = REPO / "output" / "mobilenet-v2" / "rtl" / "nn2rtl_top_engine.v"
SCHED = REPO / "output" / "mobilenet-v2" / "rtl" / "nn2rtl_scheduler.v"
TOP_BASE = TOP.with_name("nn2rtl_top_engine.v.prequartet")
SCHED_BASE = SCHED.with_name("nn2rtl_scheduler.v.prequartet")

ACT_DEPTH = 25600

MODULES_FC = [
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
    "node_conv_910", "node_conv_912", "node_linear",
]
DW_FC = {"node_conv_824", "node_conv_836", "node_conv_842", "node_conv_854",
         "node_conv_860", "node_conv_866", "node_conv_872", "node_conv_878",
         "node_conv_884", "node_conv_896", "node_conv_902", "node_conv_908"}
Q_SUCC = {"node_conv_818": "node_conv_820", "node_conv_830": "node_conv_832",
          "node_conv_848": "node_conv_850", "node_conv_890": "node_conv_892"}
Q_ORDER = ["node_conv_818", "node_conv_830", "node_conv_848", "node_conv_890"]

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
    sh = parse_rom(sched_text, "stride_h_rom", n)
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
            "module": modules[d], "kh": kh[d], "sh": sh[d],
            "ih": ih[d], "oh": oh[d],
            "rwpp": rwpp, "wwpp": wwpp,
            "oh_ow": oh[d] * ow[d], "read": read, "write": write,
            "loader": disp_loader[d],
        }
    return table, loaders


def trace_fill_source(top_text: str, in_valid: str) -> str | None:
    """Return the valid_out NET driving the loader's in_valid, traced back
    through n4 relu stages to a node_conv_*_valid_out (or None)."""
    seen = 0
    sig = in_valid.replace("& spatial_run", "").strip()
    while seen < 4:
        m = re.match(r"(n4(?:_\d+)?)_valid_out$", sig)
        if not m:
            break
        inst = m.group(1)
        mm = re.search(rf"\b{inst} #\(.*?\) u_{inst} \(\s*[\s\S]{{0,400}}?"
                       rf"\.valid_in\(([^)]*)\),", top_text)
        if not mm:
            return None
        sig = mm.group(1).replace("& spatial_run", "").strip()
        seen += 1
    m = re.match(r"(node_conv_\d+|node_linear)_valid_out$", sig)
    return m.group(1) if m else None


def classify(d: int, t: dict, loaders: dict, n: int, top_text: str = ""):
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
        elif (e["kh"] == 1 and e["write"] == e["read"] == lrg
              and e["wwpp"] <= e["rwpp"] and top_text
              and trace_fill_source(top_text, L["in_valid"]) == e["module"]):
            # [DW-QUARTET] in-place region re-filled with the RELU'D copy of
            # this dispatch's own bridge stream (818 overlay; see header).
            v_wf = "lag-safe-inplace-fill"
        else:
            v_wf = "RATE"
    return v_wr, v_rf, v_wf, fill


def main() -> int:
    for p in [TOP, SCHED, TOP_BASE, SCHED_BASE]:
        if not p.is_file():
            raise SystemExit(f"missing artifact: {p}")
    top_text = TOP.read_text(encoding="utf-8")
    sched_text = SCHED.read_text(encoding="utf-8")

    # detect the applied subset from the deployed top
    applied = [q for q in Q_ORDER
               if f"u_ldr_{q.replace('node_conv_', 'node_conv_')} (" in top_text
               and f"ldr_dw{q.replace('node_conv_', '')}_loaded" in top_text]
    if not applied:
        raise SystemExit("no quartet loaders found in the top — nothing to prove")
    modules_new = []
    by_succ = {Q_SUCC[q]: q for q in applied}
    for m in MODULES_FC:
        if m in by_succ:
            modules_new.append(by_succ[m])
        modules_new.append(m)
    n_new, n_old = len(modules_new), len(MODULES_FC)
    q_slots = {modules_new.index(q): q for q in applied}
    touched = set()
    for s in q_slots:
        touched.add(s)
        touched.add(s - 1)

    new_t, new_l = build_table(top_text, sched_text, modules_new)
    old_t, old_l = build_table(TOP_BASE.read_text(encoding="utf-8"),
                               SCHED_BASE.read_text(encoding="utf-8"), MODULES_FC)
    print(f"[hazard-q] parsed: baseline (FC) {n_old} dispatches / {len(old_l)} loaders, "
          f"QUARTET {n_new} dispatches / {len(new_l)} loaders; applied = "
          + ", ".join(f"{q}@{modules_new.index(q)}" for q in applied))

    # depthwise rows: stride-1 12 + applied quartet; stride-2 rows = quartet only
    dwbits = {int(i): int(v) for i, v in
              re.findall(r"6'd(\d+): depthwise_rom = 1'b(\d);", sched_text)}
    for d in range(n_new):
        want = 1 if (modules_new[d] in DW_FC or modules_new[d] in applied) else 0
        if dwbits.get(d, 0) != want:
            fail(f"depthwise_rom[{d}]={dwbits.get(d, 0)} want {want}")
    for d in range(n_new):
        e = new_t[d]
        if modules_new[d] in applied:
            if not (e["sh"] == 2 and e["ih"] == 2 * e["oh"] and e["kh"] == 3):
                fail(f"d{d} {modules_new[d]}: stride-2 geometry sh={e['sh']} "
                     f"ih={e['ih']} oh={e['oh']}")
        elif e["sh"] != 1:
            fail(f"d{d} {modules_new[d]}: unexpected stride {e['sh']}")

    # C5: bounds
    print("[hazard-q] C5: act-region bounds (ACT_DEPTH=25600)")
    for d in range(n_new):
        for tag in ["read", "write"]:
            b, w = new_t[d][tag]
            if not (0 <= b and b + w <= ACT_DEPTH):
                fail(f"d{d} {tag} [{b},+{w}) outside act mem")
    for inst, L in new_l.items():
        if not (0 <= L["base"] and L["base"] + L["words"] <= ACT_DEPTH):
            fail(f"loader {inst} [{L['base']},+{L['words']}) outside act mem")

    # C1: loader region == read region
    print("[hazard-q] C1: loader region == engine read region")
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

    # PART A (touched pairs) + PART B (FC-baseline equivalence, renumbered)
    print("[hazard-q] C2/C3/C4 per-dispatch verdicts (A=strict/lag for touched, "
          "B=FC-baseline-identical for inherited; renumbered dispatch map)")
    old_verdicts = {od: classify(od, old_t, old_l, n_old) for od in range(n_old)}
    fc_idx = {m: i for i, m in enumerate(MODULES_FC)}
    OK_A = {"disjoint", "none", "lag-safe-1x1", "lag-safe-inplace-fill"}
    print(f"  {'d':>2} {'module':<12} {'read':>15} {'write':>15} {'fill(d+1)':<24} wxr / rxf / wxf")
    for d in range(n_new):
        v_wr, v_rf, v_wf, fill = classify(d, new_t, new_l, n_new, top_text)
        verdicts = (v_wr, v_rf, v_wf)
        mod = modules_new[d]
        if d in touched:
            if mod in applied:
                # the inserted dispatch itself: ALL pairs strictly disjoint
                for tag, v in zip(["wxr", "rxf", "wxf"], verdicts):
                    if v not in ("disjoint", "none"):
                        fail(f"d{d} ({mod} inserted) {tag} verdict '{v}' — must be disjoint")
            else:
                # predecessor of an inserted conv: rxf/wxf are the NEW pairs
                # (strict or the proven lag classes); wxr inherited.
                for tag, v in zip(["rxf", "wxf"], (v_rf, v_wf)):
                    if v not in OK_A:
                        fail(f"d{d} ({mod} pre-insert) {tag} verdict '{v}' — "
                             f"must be disjoint or a lag-safe class")
                od = fc_idx[mod]
                if v_wr != old_verdicts[od][0] and v_wr not in ("disjoint",):
                    fail(f"d{d} wxr '{v_wr}' != baseline '{old_verdicts[od][0]}'")
                for k in ["read", "write", "kh", "rwpp", "wwpp"]:
                    if new_t[d][k] != old_t[od][k]:
                        fail(f"d{d} {k} {new_t[d][k]} != baseline {old_t[od][k]}")
            cls = "A"
        else:
            od = fc_idx[mod]
            base_v = old_verdicts[od]
            for tag, vn, vo in zip(["wxr", "rxf", "wxf"], verdicts, base_v[:3]):
                if vn == vo or vn in ("disjoint", "none"):
                    continue
                fail(f"d{d} {tag}: verdict '{vn}' differs from baseline '{vo}'")
            for k in ["read", "write", "kh", "rwpp", "wwpp"]:
                if new_t[d][k] != old_t[od][k]:
                    fail(f"d{d} {k} {new_t[d][k]} != baseline {old_t[od][k]}")
            if new_t[d]["loader"] != old_t[od]["loader"]:
                fail(f"d{d} loader {new_t[d]['loader']} != baseline {old_t[od]['loader']}")
            cls = "B"
        e = new_t[d]
        print(f"  {d:>2} {e['module'].removeprefix('node_'):<12} "
              f"[{e['read'][0]:>5},+{e['read'][1]:<5}) [{e['write'][0]:>5},+{e['write'][1]:<5}) "
              f"{(fill or '-'):<24} {v_wr} / {v_rf} / {v_wf}  [{cls}]")

    # the 818 overlay (if applied) must come out as EXACTLY the two lag classes
    if "node_conv_818" in applied:
        d816 = modules_new.index("node_conv_816")
        v = classify(d816, new_t, new_l, n_new, top_text)
        if (v[1], v[2]) != ("lag-safe-1x1", "lag-safe-inplace-fill"):
            fail(f"d{d816} (816 overlay) verdicts {v[1]}/{v[2]} != "
                 "lag-safe-1x1/lag-safe-inplace-fill")
        else:
            print(f"  d{d816} (816) overlay verdicts CONFIRMED: rxf=lag-safe-1x1, "
                  "wxf=lag-safe-inplace-fill (fill source traced to node_conv_816)")

    if fails:
        print(f"\n[hazard-q] RESULT: FAIL ({len(fails)} violations)")
        return 1
    print("\n[hazard-q] RESULT: PASS — PART A: every touched pair strictly disjoint or"
          "\n         a proven lag class (818 overlay traced to its own bridge stream);"
          "\n         PART B: all inherited dispatch rows byte-identical to the"
          "\n         e2e-proven FC/ENG-PIPE baseline; C1/C5 + stride-2 geometry hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
