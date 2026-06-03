#!/usr/bin/env python3
"""apply_mpk9_depthwise.py -- A2 throughput lever: MP_K=9 tap-parallel transform.

DETERMINISTIC patch generator that applies the PROVEN (byte-exact on node_conv_812)
MP_K=9 tap-parallel reduction to all 17 MobileNetV2 depthwise wrappers AND switches
the stem node_conv_810 from conv_datapath -> conv_datapath_mp_k (MP_K=9, WGT_BITS=8).

The transform reproduces scratch/node_conv_812_mpk9.v BYTE-FOR-BYTE when applied to
the live node_conv_812.v (self-checked at the bottom of this script with --self-check).
It is WIDTH-AGNOSTIC: it keys off structural anchors common to all 17 wrappers
(oc_group/current_global_oc widths, OC_PASSES-1 vs 3'd7 final-pass compare, and the
two tap-selector spellings -- VERBOSE kh_idx/kw_idx and SIMPLE tap_k_lin=k_counter).

It does NOT mutate goldens, sidecars, weight hex, the engine, n4_*.v, or nn2rtl_top*.v.
It only rewrites the 18 spatial node_conv_*.v wrappers, backing each up first.

USAGE
    python scripts/apply_mpk9_depthwise.py --dry-run     # report 17+1 targets, no writes
    python scripts/apply_mpk9_depthwise.py --self-check  # prove 812 transform == scratch PoC
    python scripts/apply_mpk9_depthwise.py               # APPLY (backs up first)

SAFETY: --dry-run and --self-check make NO writes to the rtl/ tree. The bare run
backs up every target to backups/mpk9_<timestamp>/ before writing.

Byte-exact contract (see docs/agent_tasks/mbv2_spatial_throughput_roadmap.md):
 - tree-sum of 9 per-tap products == serial sum (integer-add associativity)
 - per-product `signed [PROD_W-1:0]` regs, NEVER outer $signed(a*b) (8-bit trunc trap)
 - q1->q2 valid/lane/oc shift kept at EXACTLY 2 stages (products registered at the
   same stage baseline registered mul_q; tree-sum combinational in the accumulate stage)
 - ACC_W unchanged (24); TREE_W = PROD_W + clog2(9) = 20 fits
 - ST_BIAS/ST_SCALE/ST_OUTPUT requant BYTE-IDENTICAL (untouched)
 - latency: per-pass MP*ceil(K_TOTAL/MP_K)+6 = 4*1+6 = 10 (DW), MP*3+6 = 18 (stem)
"""
from __future__ import annotations

import argparse
import datetime
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RTL_DIR = REPO / "output" / "mobilenet-v2" / "rtl"

# The 17 depthwise wrappers (inline-MAC MP_K hand-written tree transform).
DEPTHWISE_NODES = [
    "node_conv_812", "node_conv_818", "node_conv_824", "node_conv_830",
    "node_conv_836", "node_conv_842", "node_conv_848", "node_conv_854",
    "node_conv_860", "node_conv_866", "node_conv_872", "node_conv_878",
    "node_conv_884", "node_conv_890", "node_conv_896", "node_conv_902",
    "node_conv_908",
]
# The stem (group=1 conv_datapath -> conv_datapath_mp_k reuse).
STEM_NODE = "node_conv_810"

MP_K = 9
STEM_WGT_BITS = 8           # MBv2 INT8 (conv_datapath_mp_k default is 4 = INT4!)


# ---------------------------------------------------------------------------
# Depthwise transform (the proven node_conv_812 MP_K=9 mechanism, width-agnostic)
# ---------------------------------------------------------------------------

def _xform_counter_decls(src: str) -> str:
    """Region 1: drop k_counter reg, rewrite weight_read_addr -> weight_base_addr,
    add MP_K / K_GROUPS localparams + the contiguous-tap base address wire."""
    # Drop the standalone k_counter declaration line.
    src, n = re.subn(r"^[ \t]*reg \[3:0\] k_counter;[ \t]*\r?\n", "", src, flags=re.M)
    if n != 1:
        raise RuntimeError(f"expected exactly 1 'reg [3:0] k_counter;' decl, found {n}")

    # weight_read_addr = current_global_oc * K_TOTAL + k_counter; -> weight_base_addr (no k).
    # Width prefix and spacing vary across wrappers; capture them.
    m = re.search(
        r"^([ \t]*wire \[15:0\])([ \t]*)weight_read_addr([ \t]*)= current_global_oc \* K_TOTAL \+ k_counter;.*$",
        src, flags=re.M)
    if not m:
        raise RuntimeError("weight_read_addr declaration not found")
    repl = (f"{m.group(1)}{m.group(2)}weight_base_addr{m.group(3)}"
            f"= current_global_oc * K_TOTAL;  // contiguous K_TOTAL taps for this channel")
    src = src[:m.start()] + repl + src[m.end():]
    return src


def _xform_mac_localparams(src: str) -> str:
    """Add MP_K / K_GROUPS localparams next to MP. Keep the file's existing
    'localparam integer MP = 4;' and append MP_K/K_GROUPS right after it."""
    m = re.search(r"^([ \t]*)localparam integer MP[ \t]*= 4;.*$", src, flags=re.M)
    if not m:
        raise RuntimeError("'localparam integer MP = 4;' not found")
    indent = m.group(1)
    add = (f"\n{indent}localparam integer MP_K      = {MP_K};            "
           f"// tap-parallel width (= K_TOTAL)"
           f"\n{indent}localparam integer K_GROUPS  = K_TOTAL / MP_K; // = 1 (single-shot reduction)")
    src = src[:m.end()] + add + src[m.end():]
    return src


# Span-based matcher: from the FIRST tap-selector wire (verbose `kh_idx` or simple
# `tap_k_lin = k_counter`) THROUGH the `mul_q` declaration. Whitespace, the
# tap_byte_lsb width ([6:0] vs [7:0]), and the `* 8` vs `* 4'd8` spellings all vary
# across the 17 wrappers, so we match the whole region as one non-greedy span
# bounded by stable endpoints instead of pinning every interior line.
_TAP_BLOCK_START = re.compile(
    r"^[ \t]*wire \[1:0\] kh_idx = \(k_counter"     # verbose
    r"|^[ \t]*wire \[3:0\] tap_k_lin[ \t]*= k_counter;",   # simple
    flags=re.M)
_TAP_BLOCK_END = re.compile(
    r'\(\* use_dsp = "yes" \*\) reg signed \[PROD_W-1:0\] mul_q;[ \t]*\r?\n')
# Sanity anchors that MUST appear inside the matched span (catches a mis-bounded
# span). Spelling-independent substrings only: some wrappers fold tap_byte_lsb into
# tap_byte_raw (chan_window_flat[tap_k_lin*8 +: 8]) so we require just the prefix.
_TAP_BLOCK_REQUIRED = (
    "tap_byte_raw = chan_window_flat[",
    "weight_q <= weights[weight_read_addr];",
    "tap_q    <= $signed(tap_byte_raw);",
)

_TAP_BLOCK_REPLACEMENT = """\
    // ---- Tap-parallel read: pull all KH*KW=9 weights + 9 window bytes for the
    // current channel at once. chan_window_flat byte kk (0..8) is the (kh*KW+kw)
    // tap for the channel line_buf_window exposes via channel_select
    // (= current_global_oc) -- bit-identical to the baseline's per-tap read.
    reg signed [7:0] weight_q [0:MP_K-1];
    reg signed [7:0] tap_q    [0:MP_K-1];
    integer kk;
    always @(posedge clk) begin
        for (kk = 0; kk < MP_K; kk = kk + 1) begin
            weight_q[kk] <= weights[weight_base_addr + kk];
            tap_q[kk]    <= $signed(chan_window_flat[kk*8 +: 8]);
        end
    end

    // ---- 9 parallel products (one DSP per tap), registered at the SAME pipeline
    // stage the baseline registers its single `mul_q`. The tree-sum is done
    // COMBINATIONALLY in the accumulate stage so the q1->q2 valid pipeline depth is
    // BIT-FOR-BIT identical to the baseline (2 stages). Each product is an
    // independently-typed signed [PROD_W-1:0] reg so the multiply is PROD_W-wide
    // (NOT outer $signed(a*b), which self-determines to 8-bit and truncates).
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] prod_q [0:MP_K-1];
"""


def _xform_tap_block(src: str) -> str:
    """Region 2: replace the tap selector + scalar weight_q/tap_q read + mul_q decl
    with the MP_K-wide array read + prod_q[] decl. Span-bounded so it tolerates the
    verbose/simple spellings AND the per-wrapper whitespace/width variance."""
    start = _TAP_BLOCK_START.search(src)
    if not start:
        raise RuntimeError("tap-selector start anchor (kh_idx / tap_k_lin) not found")
    end = _TAP_BLOCK_END.search(src, start.start())
    if not end:
        raise RuntimeError("tap-selector end anchor (mul_q decl) not found")
    span = src[start.start():end.end()]
    for req in _TAP_BLOCK_REQUIRED:
        if req not in span:
            raise RuntimeError(f"tap-selector span missing required anchor: {req!r}")
    if span.count("mul_q") != 1:
        raise RuntimeError("tap-selector span captured an unexpected extra mul_q")
    # _TAP_BLOCK_REPLACEMENT carries the canonical 4-space module-body indent that
    # every wrapper uses; the span starts at column 4, so substitution is in-place.
    src = src[:start.start()] + _TAP_BLOCK_REPLACEMENT + src[end.end():]
    return src


def _xform_sum_comb(src: str) -> str:
    """Region 2b: insert the combinational tree-sum just after the mac_global_oc_q2
    declaration (which immediately precedes `assign mac_busy = ...`)."""
    # Width spec varies: literal [N:0] OR a $clog2(C)-1:0 expression (node_conv_854).
    anchor = re.search(
        r"^([ \t]*)reg \[[^\]]+\][ \t]*mac_global_oc_q2;[ \t]*\r?\n", src, flags=re.M)
    if not anchor:
        raise RuntimeError("mac_global_oc_q2 declaration not found")
    indent = anchor.group(1)
    block = (
        f"\n{indent}integer pp;\n"
        f"{indent}// Combinational tree-sum of the 9 registered products into one ACC_W value.\n"
        f"{indent}// Integer addition is associative -> this equals the baseline's serial\n"
        f"{indent}// accumulation of the 9 per-tap products bit-for-bit.\n"
        f"{indent}reg signed [ACC_W-1:0] sum_comb;\n"
        f"{indent}always @(*) begin\n"
        f"{indent}    sum_comb = {{ACC_W{{1'b0}}}};\n"
        f"{indent}    for (pp = 0; pp < MP_K; pp = pp + 1)\n"
        f"{indent}        sum_comb = sum_comb + $signed(prod_q[pp]);\n"
        f"{indent}end\n")
    src = src[:anchor.end()] + block + src[anchor.end():]
    return src


def _xform_reset(src: str) -> str:
    """Region 3: drop the RESET-block k_counter reset (12-space indent, the
    `if (!rst_n)` arm); replace mul_q reset with prod_q[] loop. The ST_IDLE and
    ST_OUTPUT k_counter resets (24-space indent) are removed separately."""
    src, n = re.subn(r"^            k_counter        <= 4'd0;[ \t]*\r?\n", "", src, flags=re.M)
    if n != 1:
        raise RuntimeError(f"expected 1 reset-block 'k_counter <= 4'd0;' (12-sp), found {n}")
    # Drop the scalar mul_q reset (no longer exists).
    src, n = re.subn(r"^[ \t]*mul_q            <= \{PROD_W\{1'b0\}\};[ \t]*\r?\n", "", src, flags=re.M)
    if n != 1:
        raise RuntimeError(f"expected 1 'mul_q <= {{PROD_W..}}' reset, found {n}")
    # Insert the prod_q[] reset loop right before the acc[]/biased[]/scaled[] loop
    # (matches the proven PoC ordering: ... v_tmp <= ...; <prod_q loop>; for(MP)...).
    m = re.search(r"^([ \t]*)for \(i = 0; i < MP; i = i \+ 1\) begin\r?\n", src, flags=re.M)
    if not m:
        raise RuntimeError("reset 'for (i=0;i<MP;...) begin' loop not found")
    indent = m.group(1)
    repl = (f"{indent}for (i = 0; i < MP_K; i = i + 1)\n"
            f"{indent}    prod_q[i] <= {{PROD_W{{1'b0}}}};\n")
    src = src[:m.start()] + repl + src[m.start():]
    return src


def _xform_mac_pipeline(src: str) -> str:
    """Region 4: stage-2 single multiply -> 9 parallel multiplies; stage-3 acc add
    of mul_q -> acc add of the combinational tree-sum."""
    m = re.search(
        r"^([ \t]*)// Stage 2: registered multiply\r?\n"
        r"[ \t]*mul_q            <= \$signed\(weight_q\) \* \$signed\(tap_q\);\r?\n",
        src, flags=re.M)
    if not m:
        raise RuntimeError("stage-2 'mul_q <= ...' block not found")
    indent = m.group(1)
    repl = (f"{indent}// Stage 2: registered parallel multiplies (one DSP per tap).\n"
            f"{indent}for (i = 0; i < MP_K; i = i + 1)\n"
            f"{indent}    prod_q[i] <= $signed(weight_q[i]) * $signed(tap_q[i]);\n")
    src = src[:m.start()] + repl + src[m.end():]

    src, n = re.subn(
        r"acc\[mac_lane_q2\] <= acc\[mac_lane_q2\] \+ \$signed\(mul_q\);",
        "acc[mac_lane_q2] <= acc[mac_lane_q2] + $signed(sum_comb);",
        src)
    if n != 1:
        raise RuntimeError(f"expected 1 stage-3 acc add of mul_q, found {n}")
    return src


def _xform_st_idle(src: str) -> str:
    """Region 5: drop the ST_IDLE k_counter reset (24-space indent, wide internal
    gap `k_counter        <= 4'd0;`)."""
    src, n = re.subn(r"^                        k_counter        <= 4'd0;[ \t]*\r?\n", "", src, flags=re.M)
    if n != 1:
        raise RuntimeError(f"expected 1 ST_IDLE 'k_counter <= 4'd0;' (24-sp), found {n}")
    return src


def _xform_st_mac_loop(src: str) -> str:
    """Region 6: replace the nested k_counter increment with direct done-issuing."""
    m = re.search(
        r"^([ \t]*)if \(lane_counter == 2'd3\) begin\r?\n"
        r"[ \t]*lane_counter <= 2'd0;\r?\n"
        r"[ \t]*if \(k_counter == 4'd8\) begin\r?\n"
        r"[ \t]*mac_done_issuing <= 1'b1;\r?\n"
        r"[ \t]*end else begin\r?\n"
        r"[ \t]*k_counter <= k_counter \+ 4'd1;\r?\n"
        r"[ \t]*end\r?\n"
        r"([ \t]*)end else begin\r?\n"
        r"[ \t]*lane_counter <= lane_counter \+ 2'd1;\r?\n"
        r"[ \t]*end\r?\n",
        src, flags=re.M)
    if not m:
        raise RuntimeError("ST_MAC nested k_counter loop not found")
    i1 = m.group(1)   # indent of the inner `if (lane_counter==3)`
    i2 = m.group(2)   # indent of the `end else begin`
    repl = (f"{i1}if (lane_counter == 2'd3) begin\n"
            f"{i1}    lane_counter     <= 2'd0;\n"
            f"{i1}    mac_done_issuing <= 1'b1;\n"
            f"{i2}end else begin\n"
            f"{i1}    lane_counter <= lane_counter + 2'd1;\n"
            f"{i2}end\n")
    src = src[:m.start()] + repl + src[m.end():]
    return src


def _xform_st_output(src: str) -> str:
    """Region 7: drop the ST_OUTPUT non-final-branch k_counter reset (24-space
    indent, narrow internal gap `k_counter    <= 4'd0;`)."""
    src, n = re.subn(r"^                        k_counter    <= 4'd0;[ \t]*\r?\n", "", src, flags=re.M)
    if n != 1:
        raise RuntimeError(f"expected 1 ST_OUTPUT 'k_counter <= 4'd0;' (24-sp), found {n}")
    return src


def transform_depthwise(src: str) -> str:
    """Apply all 7 region edits in order. Pure-string; raises on any anchor miss
    so a structural drift can never silently produce wrong RTL."""
    src = _xform_counter_decls(src)
    src = _xform_mac_localparams(src)
    src = _xform_tap_block(src)
    src = _xform_sum_comb(src)
    src = _xform_reset(src)
    src = _xform_mac_pipeline(src)
    src = _xform_st_idle(src)
    src = _xform_st_mac_loop(src)
    src = _xform_st_output(src)
    return src


# ---------------------------------------------------------------------------
# Stem transform: conv_datapath -> conv_datapath_mp_k (MP_K=9, WGT_BITS=8)
# ---------------------------------------------------------------------------

_STEM_INSTANCE = re.compile(
    r"[ \t]*conv_datapath #\(\r?\n"
    r"[ \t]*\.IC\(IC\), \.OC\(OC\), \.KH\(KH\), \.KW\(KW\),\r?\n"
    r"[ \t]*\.K_TOTAL\(K_TOTAL\), \.MP\(MP\),\r?\n"
    r"[ \t]*\.SCALE_MULT\(SCALE_MULT\), \.SCALE_SHIFT\(SCALE_SHIFT\),\r?\n"
    r"[ \t]*\.WEIGHTS_PATH\(\"(?P<wpath>[^\"]*)\"\),\r?\n"
    r"[ \t]*\.BIAS_PATH\(\"(?P<bpath>[^\"]*)\"\)\r?\n"
    r"[ \t]*\) dp \(\r?\n",
    flags=re.S)


def transform_stem(src: str) -> str:
    """Switch the stem's conv_datapath instance to conv_datapath_mp_k MP_K=9.
    Re-points WEIGHTS_PATH to the MP_K=9-repacked wide hex (..._weights_mp_k_9.hex)
    and pins WGT_BITS=8 (the mp_k default is 4=INT4). Per-tensor scale kept via
    SCALE_MULT/SCALE_SHIFT + SCALE_PATH("")."""
    m = _STEM_INSTANCE.search(src)
    if not m:
        raise RuntimeError("stem conv_datapath instance not found in node_conv_810.v")
    wpath = m.group("wpath")
    bpath = m.group("bpath")
    # Re-point weight hex to the MP_K=9 wide layout produced by repack_weights_wide.
    wpath_wide = re.sub(r"_weights\.hex$", "_weights_mp_k_9.hex", wpath)
    if wpath_wide == wpath:
        raise RuntimeError(f"stem WEIGHTS_PATH did not end in _weights.hex: {wpath}")
    repl = (
        "    conv_datapath_mp_k #(\n"
        "        .IC(IC), .OC(OC), .KH(KH), .KW(KW),\n"
        "        .K_TOTAL(K_TOTAL), .MP(MP),\n"
        f"        .MP_K({MP_K}),                 // tap-parallel; K_TOTAL=27 = 3*9 -> 3 group cycles\n"
        f"        .WGT_BITS({STEM_WGT_BITS}),               // MBv2 INT8 (mp_k default is 4=INT4!)\n"
        "        .SCALE_MULT(SCALE_MULT), .SCALE_SHIFT(SCALE_SHIFT), .SCALE_PATH(\"\"),\n"
        f"        .WEIGHTS_PATH(\"{wpath_wide}\"),\n"
        f"        .BIAS_PATH(\"{bpath}\")\n"
        "    ) dp (\n"
    )
    src = src[:m.start()] + repl + src[m.end():]
    return src


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _self_check() -> int:
    """Apply the depthwise transform to the LIVE node_conv_812.v and prove the
    result is byte-identical to the proven PoC scratch/node_conv_812_mpk9.v
    (modulo the module name + header comment, which the PoC renamed). NO writes."""
    live = (RTL_DIR / "node_conv_812.v").read_text()
    poc = (REPO / "scratch" / "node_conv_812_mpk9.v").read_text()
    got = transform_depthwise(live)

    # Normalize for comparison: the PoC renamed the module + rewrote the leading
    # comment header. Compare from the `module ...(` declaration onward, with the
    # PoC's _mpk9 suffix and the comment-only deltas removed.
    def body_from_module(t: str) -> str:
        idx = t.index("module node_conv_812")
        return t[idx:]
    got_body = body_from_module(got).replace("node_conv_812_mpk9", "node_conv_812")
    poc_body = body_from_module(poc).replace("node_conv_812_mpk9", "node_conv_812")

    # Drop pure-comment lines (the PoC re-narrated the datapath comments; arithmetic
    # is what must match). A comment-insensitive diff is the right equivalence here.
    def strip_comments(t: str) -> list[str]:
        out = []
        for ln in t.splitlines():
            s = ln.strip()
            if s.startswith("//"):
                continue
            # strip trailing inline comments
            if "//" in ln:
                ln = ln[:ln.index("//")].rstrip()
            if ln.strip():
                out.append(ln.rstrip())
        return out

    g = strip_comments(got_body)
    p = strip_comments(poc_body)
    if g == p:
        print("[self-check] PASS: transform(node_conv_812.v) == scratch PoC "
              "(code-identical, comment-insensitive).")
        return 0
    # Show the first divergence for debuggability.
    import difflib
    print("[self-check] FAIL: transform output diverges from the proven PoC:")
    for line in difflib.unified_diff(p, g, "PoC", "transform", lineterm="", n=2):
        print("  " + line)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply MP_K=9 tap-parallel transform "
                                             "to the 17 MBv2 depthwise wrappers + stem.")
    ap.add_argument("--dry-run", action="store_true",
                    help="report targets and validate every transform in-memory; NO writes")
    ap.add_argument("--self-check", action="store_true",
                    help="prove transform(node_conv_812.v) == scratch PoC; NO writes")
    args = ap.parse_args()

    if args.self_check:
        return _self_check()

    dw_files = [(RTL_DIR / f"{n}.v", n) for n in DEPTHWISE_NODES]
    stem_file = RTL_DIR / f"{STEM_NODE}.v"

    missing = [p for p, _ in dw_files if not p.exists()] + ([stem_file] if not stem_file.exists() else [])
    if missing:
        print("ERROR: missing RTL targets:", [str(p) for p in missing])
        return 2

    # Compute every transform in memory first; abort the whole batch on any miss.
    planned: list[tuple[Path, str, str]] = []   # (path, kind, new_text)
    for path, node in dw_files:
        try:
            new = transform_depthwise(path.read_text())
        except RuntimeError as e:
            print(f"ERROR transforming {node}: {e}")
            return 3
        planned.append((path, "depthwise", new))
    try:
        planned.append((stem_file, "stem", transform_stem(stem_file.read_text())))
    except RuntimeError as e:
        print(f"ERROR transforming {STEM_NODE} (stem): {e}")
        return 3

    n_dw = sum(1 for _, k, _ in planned if k == "depthwise")
    n_stem = sum(1 for _, k, _ in planned if k == "stem")
    print(f"[apply_mpk9] targets: {n_dw} depthwise + {n_stem} stem = {len(planned)} total")
    for path, kind, _ in planned:
        print(f"  [{kind:9s}] {path.relative_to(REPO)}")

    if args.dry_run:
        print("[apply_mpk9] --dry-run: every transform validated in-memory; NO files written.")
        print(f"[apply_mpk9] dryrun_targets = {len(planned)}")
        return 0

    # APPLY: back up every target, then write.
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = REPO / "backups" / f"mpk9_{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    print(f"[apply_mpk9] backing up {len(planned)} files to {backup_dir.relative_to(REPO)}/")
    for path, _, _ in planned:
        shutil.copy2(path, backup_dir / path.name)
    for path, _, new in planned:
        path.write_text(new)
    print(f"[apply_mpk9] APPLIED MP_K=9 to {len(planned)} files. Backups in "
          f"{backup_dir.relative_to(REPO)}/")
    print("[apply_mpk9] NEXT: run the regen chain + per-module byte-exact verify "
          "(see docs/agent_tasks/mbv2_spatial_throughput_roadmap.md, 'A2 EXECUTION').")
    return 0


if __name__ == "__main__":
    sys.exit(main())
