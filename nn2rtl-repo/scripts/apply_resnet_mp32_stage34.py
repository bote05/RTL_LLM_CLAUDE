#!/usr/bin/env python3
"""MP32 STAGE-3/4 (ResNet): raise the 11 stage-3/4 spatial 1x1 convs from
MP=16 to MP=32 lanes, with the deadlock-class hardening that the two prior
MP-increase attempts (B22, memory project_mp_increase_deadlock) lacked.

WHAT THIS DOES (all byte-exact / value-preserving; only timing moves):

1. MP 16->32 in output/rtl/node_conv_{248,252,256,258,262,268,270,274,276,
   280,288}.v  (localparam + header comment). All verified stage-3/4 spatial
   1x1 conv_datapath_mp_k instances at MP=16/MP_K=8; conv_288 is the
   1024->2048 s2 decimator wrapper and is INT3 (WGT_BITS=3).

2. Weight repack (MP-dependent!): conv_datapath_mp_k's ROM word is
   MP*MP_K*WGT_BITS bits, lane-major; doubling MP changes the packing. For
   each selected conv this repacks the flat node_conv_<id>_weights.hex into
   output/weights/node_conv_<id>_weights_mp32_k8.hex (NEW name, so the
   canonical mp_k_8 files for MP=16 are never shadowed/clobbered) and points
   the wrapper's WEIGHTS_PATH at it (repo-root-derived absolute path, the
   $readmemh convention in this tree). conv_288 repacks at --wgt-bits 3.
   scale.mem / bias.hex are per-absolute-OC -> MP-independent -> untouched.
   A full inverse self-check (unpack-and-compare vs flat) runs after each
   repack.

3. LHS skid on the add_8..12 joins (template: scripts/apply_conv202_lhs_skid.py,
   the PREPPED-but-never-built fix for the original MP-increase deadlock):
   conv_{256,262,268,274,280} currently drain DIRECTLY into their residual
   join with the circular ready tie `skip_valid & spatial_run & add_ready_in`.
   Each gets a symmetric DEPTH=512 skip_fifo (URAM FWFT, same as the RHS
   class), the join consumes the buffered arm, and the RHS pop gate swaps to
   the buffered valid. FIFO preserves value+order -> byte-exact.
   (add_7 and add_13 need nothing: their MP32-accelerated convs 248/288
   already drain into skip FIFOs - they ARE the buffered arm.)

4. Narrow-relu fork hardening (B20 class, TAIL_PIPE2_ANALYSIS.md #1b): the
   post-add relus 24/27/33/36 fork into a DEPTH=2 receiver
   (skid_node_conv_{252,258,270,276}) + the 8192 skip fifo. The narrow-relu
   streamer presents each pixel's LAST beat for exactly ONE cycle; a full
   DEPTH=2 receiver on that cycle silently drops the beat and wedges the
   downstream join (this retroactively explains B22). Bump those four
   receivers 2->64 (LUTRAM, value-preserving margin, TP2 precedent).

5. B20 drop DETECTORS (sim forensics): per-relu counters for the 14 stage-3/4
   narrow relus that flag the exact B20 signature (valid_out falls after an
   un-accepted offer cycle) + a final $display. $display-only sinks ->
   synth-pruned; same convention as the existing [fifo-peak] audit.

6. Atomic-arch rule: adds _MP32_STAGE34_OVERRIDE to scripts/onnx_frontend.py
   so compute_conv2d_latency_cycles sees MP=32/MP_K=8 for these 11 modules on
   any future regen (A2-override precedent). Goldens are VALUE streams and are
   NOT regenerated/touched by this script (asserted: nothing under
   output/goldens is written).

IDEMPOTENT (marker: "[MP32-S34]" / mp32_k8 filenames), ANCHOR-ASSERTED (every
old string must appear exactly once or the file is left untouched), BACKED UP
(backups/mp32_stage34/, first run only), REVERSIBLE (--revert), BISECTABLE
(--convs 248,252,...). Relative repo-root paths throughout.

USAGE:
  python scripts/apply_resnet_mp32_stage34.py --dry-run
  python scripts/apply_resnet_mp32_stage34.py                  # all 11 + joins + bumps + detectors + repack
  python scripts/apply_resnet_mp32_stage34.py --convs 248,256  # subset (joins follow their lhs conv)
  python scripts/apply_resnet_mp32_stage34.py --no-joins --no-bumps --no-detect
  python scripts/apply_resnet_mp32_stage34.py --revert
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "output/rtl"
WDIR = ROOT / "output/weights"
TOP = RTL / "nn2rtl_top.v"
BK = ROOT / "backups/mp32_stage34"

sys.path.insert(0, str(ROOT / "scripts"))
from repack_weights_wide import read_flat_weights, write_wide_weights  # noqa: E402

MARK = "[MP32-S34]"

# ---------------------------------------------------------------------------
# The 11 stage-3/4 spatial 1x1 convs (verified MP=16, MP_K=8 mp_k wrappers).
# join: the residual add whose LHS this conv feeds DIRECTLY (template fix);
# None = the conv already drains into a skip FIFO (buffered arm; no fix).
# ---------------------------------------------------------------------------
CONVS: dict[int, dict] = {
    248: {"oc": 1024, "kt": 256,  "wgt_bits": 4, "join": None},   # lhs of add_7 via u_skip_node_add_7 (already buffered)
    252: {"oc": 256,  "kt": 1024, "wgt_bits": 4, "join": None},   # mid-block 1024->256
    256: {"oc": 1024, "kt": 256,  "wgt_bits": 4, "join": 8},
    258: {"oc": 256,  "kt": 1024, "wgt_bits": 4, "join": None},
    262: {"oc": 1024, "kt": 256,  "wgt_bits": 4, "join": 9},
    268: {"oc": 1024, "kt": 256,  "wgt_bits": 4, "join": 10},
    270: {"oc": 256,  "kt": 1024, "wgt_bits": 4, "join": None},
    274: {"oc": 1024, "kt": 256,  "wgt_bits": 4, "join": 11},
    276: {"oc": 256,  "kt": 1024, "wgt_bits": 4, "join": None},
    280: {"oc": 1024, "kt": 256,  "wgt_bits": 4, "join": 12},
    288: {"oc": 2048, "kt": 1024, "wgt_bits": 3, "join": None},   # INT3! feeds u_skip_node_add_13 (already buffered)
}

# join id -> lhs conv id (for the add_8..12 template edits)
JOIN_LHS = {8: 256, 9: 262, 10: 268, 11: 274, 12: 280}

# Narrow-relu fork receivers to bump 2->64 (B20 last-beat-offer margin).
FORK_BUMPS = [252, 258, 270, 276]

# B20 drop detectors: (relu_id, accept-expression — the EXACT consume gate of
# that relu's output hop, anchor-asserted against the top before insertion).
DETECT: list[tuple[int, str]] = [
    (23, "skid_node_conv_248_ready & spatial_run"),
    (24, "node_relu_24_ready_out_combined"),
    (26, "skid_node_conv_256_ready & spatial_run"),
    (27, "node_relu_27_ready_out_combined"),
    (29, "skid_node_conv_262_ready & spatial_run"),
    (30, "node_relu_30_ready_out_combined"),
    (32, "skid_node_conv_268_ready & spatial_run"),
    (33, "node_relu_33_ready_out_combined"),
    (35, "skid_node_conv_274_ready & spatial_run"),
    (36, "node_relu_36_ready_out_combined"),
    (38, "skid_node_conv_280_ready & spatial_run"),
    (39, "node_relu_39_ready_out_combined"),
    (41, "node_relu_41_ready_out_combined"),
    (42, "node_relu_42_ready_out_combined"),
]


def die(msg: str) -> None:
    print(f"ERROR: {msg} — aborting, nothing further written.")
    sys.exit(1)


def backup(path: Path) -> None:
    BK.mkdir(parents=True, exist_ok=True)
    dst = BK / path.name
    if not dst.exists():
        shutil.copy(path, dst)
        print(f"  [backup] {path.name} -> {dst.relative_to(ROOT)}")


def replace_once(txt: str, old: str, new: str, what: str, fname: str) -> str:
    c = txt.count(old)
    if c != 1:
        die(f"anchor '{what}' found {c}x (expected 1) in {fname}")
    return txt.replace(old, new)


# ---------------------------------------------------------------------------
# 1+2. Per-conv: MP localparam + header + WEIGHTS_PATH + repack
# ---------------------------------------------------------------------------

def patch_conv(cid: int, dry: bool, skip_repack: bool, verify: bool) -> None:
    info = CONVS[cid]
    f = RTL / f"node_conv_{cid}.v"
    txt = f.read_text()
    new_hex = f"node_conv_{cid}_weights_mp32_k8.hex"
    if "MP=32" in txt and new_hex in txt:
        print(f"  [skip] conv_{cid}: already MP=32 + mp32_k8 path")
    else:
        old_lp = "localparam integer MP=16, MP_K=8;"
        new_lp = f"localparam integer MP=32, MP_K=8;  // {MARK} 16->32 (stage-3/4 lane doubling)"
        txt = replace_once(txt, old_lp, new_lp, "MP localparam", f.name)
        # header comment(s): remaining 'MP=16' occurrences are comments only now
        txt = txt.replace("MP=16", "MP=32")
        old_wp = (f'"C:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/output/weights/'
                  f'node_conv_{cid}_weights_mp_k_8.hex"')
        new_wp = f'"{ROOT.as_posix()}/output/weights/{new_hex}"'
        txt = replace_once(txt, old_wp, new_wp, "WEIGHTS_PATH", f.name)
        if "MP=16" in txt:
            die(f"{f.name}: residual 'MP=16' after patch")
        if dry:
            print(f"  [dry] conv_{cid}: MP 16->32 + WEIGHTS_PATH -> {new_hex}")
        else:
            backup(f)
            f.write_text(txt, newline="\n")
            print(f"  [ok] conv_{cid}: MP 16->32 + WEIGHTS_PATH -> {new_hex}")

    # ---- repack (MP=32, MP_K=8, per-conv wgt_bits) ----
    if skip_repack:
        return
    out = WDIR / new_hex
    flat = WDIR / f"node_conv_{cid}_weights.hex"
    if not flat.exists():
        die(f"flat weights missing: {flat} (copy from the deploy tree first)")
    if dry:
        print(f"  [dry] repack conv_{cid}: OC={info['oc']} KT={info['kt']} "
              f"MP=32 MP_K=8 WGT_BITS={info['wgt_bits']} -> {new_hex}")
        return
    weights = read_flat_weights(flat)
    if len(weights) != info["oc"] * info["kt"]:
        die(f"conv_{cid}: flat has {len(weights)} weights, expected {info['oc']*info['kt']}")
    entries, padded = write_wide_weights(out, weights, info["oc"], info["kt"], 32, 8, info["wgt_bits"])
    print(f"  [ok] repack conv_{cid}: {entries} entries (pad={padded}, wgt_bits={info['wgt_bits']})")
    if verify:
        verify_pack(out, weights, info["oc"], info["kt"], 32, 8, info["wgt_bits"], cid)


def verify_pack(out: Path, weights: list[int], oc: int, kt: int, mp: int, mpk: int,
                wgt_bits: int, cid: int) -> None:
    """Full inverse check: unpack every (oc,k) weight from the wide file and
    compare (masked) against the flat source. Mirrors the RTL slice
    weight_word[(lane*MP_K+kpos)*WGT_BITS +: WGT_BITS]."""
    mask = (1 << wgt_bits) - 1
    k_groups = kt // mpk
    lines = [ln.strip() for ln in out.read_text().splitlines() if ln.strip()]
    oc_passes = (oc + mp - 1) // mp
    if len(lines) != oc_passes * k_groups:
        die(f"conv_{cid} verify: {len(lines)} lines != {oc_passes*k_groups}")
    bad = 0
    for g in range(oc_passes):
        for kg in range(k_groups):
            word = int(lines[g * k_groups + kg], 16)
            for lane in range(mp):
                o = g * mp + lane
                if o >= oc:
                    continue
                for kpos in range(mpk):
                    got = (word >> ((lane * mpk + kpos) * wgt_bits)) & mask
                    exp = weights[o * kt + kg * mpk + kpos] & mask
                    if got != exp:
                        bad += 1
    if bad:
        die(f"conv_{cid} verify: {bad} mismatched weights in {out.name}")
    print(f"  [verify] conv_{cid}: inverse unpack == flat for all {oc*kt} weights")


# ---------------------------------------------------------------------------
# 3. LHS skid on add_8..12 (template: apply_conv202_lhs_skid.py)
# ---------------------------------------------------------------------------

def lhs_skid_edits(txt: str, join: int) -> str:
    cid = JOIN_LHS[join]
    fifo = f"u_skip_node_add_{join}_main"
    if fifo in txt:
        print(f"  [skip] add_{join}: lhs skid already present")
        return txt

    # (a) conv ready_out: drop the circular skip_valid&add_ready tie
    old_rdy = f".ready_out(node_add_{join}_skip_valid & spatial_run & node_add_{join}_ready_in),"
    new_rdy = f".ready_out(node_add_{join}_main_in_ready & spatial_run),  // {MARK} lhs-skid"
    txt = replace_once(txt, old_rdy, new_rdy, f"conv_{cid} ready_out", "nn2rtl_top.v")

    # (b) insert the lhs fifo immediately before the add instantiation
    add_anchor = f"    node_add_{join} u_node_add_{join} ("
    fifo_block = f"""    // {MARK} LHS elastic buffer (template: apply_conv202_lhs_skid.py).
    // conv_{cid} (MP32, ~2x faster) drains here instead of directly into the
    // synchronized join; absorbs the valid/ready phase slip that wedged the
    // MP-increase attempts (B22). FIFO preserves beat value+order -> byte-exact.
    wire node_add_{join}_main_in_ready;
    wire node_add_{join}_main_valid;
    wire [255:0] node_add_{join}_main_data;
    skip_fifo #(.WIDTH(256), .DEPTH(512)) {fifo} (
        .clk(clk), .rst_n(rst_n),
        .in_valid(node_conv_{cid}_valid_out & spatial_run & node_add_{join}_main_in_ready),
        .in_data(node_conv_{cid}_data_out[255:0]),
        .in_ready(node_add_{join}_main_in_ready),
        .out_valid(node_add_{join}_main_valid),
        .out_data(node_add_{join}_main_data),
        .out_ready(node_add_{join}_ready_in & node_add_{join}_skip_valid & spatial_run)
    );

"""
    txt = replace_once(txt, add_anchor, fifo_block + add_anchor, f"add_{join} inst", "nn2rtl_top.v")

    # (c) join consumes the buffered arm
    old_vi = f".valid_in(node_conv_{cid}_valid_out & node_add_{join}_skip_valid & spatial_run),"
    new_vi = f".valid_in(node_add_{join}_main_valid & node_add_{join}_skip_valid & spatial_run),  // {MARK}"
    txt = replace_once(txt, old_vi, new_vi, f"add_{join} valid_in", "nn2rtl_top.v")
    old_di = f".data_in({{node_add_{join}_skip_data, node_conv_{cid}_data_out[255:0]}}),"
    new_di = f".data_in({{node_add_{join}_skip_data, node_add_{join}_main_data}}),  // {MARK}"
    txt = replace_once(txt, old_di, new_di, f"add_{join} data_in", "nn2rtl_top.v")

    # (d) RHS skip fifo pops on the buffered lhs valid
    old_po = f".out_ready(node_add_{join}_ready_in & node_conv_{cid}_valid_out & spatial_run)"
    new_po = f".out_ready(node_add_{join}_ready_in & node_add_{join}_main_valid & spatial_run)  // {MARK}"
    txt = replace_once(txt, old_po, new_po, f"add_{join} skip out_ready", "nn2rtl_top.v")
    print(f"  [ok] add_{join}: lhs skid (DEPTH=512) on conv_{cid}")
    return txt


# ---------------------------------------------------------------------------
# 4. Fork-receiver depth bumps 2 -> 64
# ---------------------------------------------------------------------------

def fork_bump_edits(txt: str) -> str:
    for cid in FORK_BUMPS:
        old = f"skip_fifo #(.WIDTH(256), .DEPTH(2)) u_skid_node_conv_{cid} ("
        new = (f"// {MARK} 2->64: B20 last-beat-offer margin for the post-add relu fork\n"
               f"    skip_fifo #(.WIDTH(256), .DEPTH(64)) u_skid_node_conv_{cid} (")
        if f"DEPTH(64)) u_skid_node_conv_{cid}" in txt:
            print(f"  [skip] skid_conv_{cid}: already DEPTH=64")
            continue
        txt = replace_once(txt, old, new, f"skid_conv_{cid} depth", "nn2rtl_top.v")
        print(f"  [ok] skid_conv_{cid}: DEPTH 2->64")
    return txt


# ---------------------------------------------------------------------------
# 5. B20 drop detectors
# ---------------------------------------------------------------------------

def detector_edits(txt: str) -> str:
    if f"{MARK} B20 drop detectors" in txt:
        print("  [skip] detectors already present")
        return txt
    anchor = "    // ----- DEBUG INSTRUMENTATION (DEBUG_E2E) -----"
    lines = [f"    // {MARK} B20 drop detectors: the narrow-relu streamer offers each",
             "    // pixel's LAST beat for exactly one cycle; valid_out FALLING after an",
             "    // un-accepted offer == a silently dropped beat (the wedge seed).",
             "    // $display-only sinks -> pruned in synthesis (same as [fifo-peak])."]
    for rid, acc in DETECT:
        if acc not in txt:
            die(f"detector accept expr for relu_{rid} not found: '{acc}'")
        lines += [
            f"    reg [31:0] b20_drop_relu_{rid}; reg b20_vo_d_{rid}, b20_acc_d_{rid};",
            "    always @(posedge clk or negedge rst_n) begin",
            f"        if (!rst_n) begin b20_drop_relu_{rid} <= 0; b20_vo_d_{rid} <= 0; b20_acc_d_{rid} <= 0; end",
            "        else begin",
            f"            b20_vo_d_{rid}  <= node_relu_{rid}_valid_out;",
            f"            b20_acc_d_{rid} <= node_relu_{rid}_valid_out & ({acc});",
            f"            if (b20_vo_d_{rid} & ~b20_acc_d_{rid} & ~node_relu_{rid}_valid_out)",
            f"                b20_drop_relu_{rid} <= b20_drop_relu_{rid} + 1;",
            "        end",
            "    end",
            f'    final $display("[b20-drop] relu_{rid} drops=%0d", b20_drop_relu_{rid});',
        ]
    block = "\n".join(lines) + "\n\n"
    txt = replace_once(txt, anchor, block + anchor, "debug-instrumentation anchor", "nn2rtl_top.v")
    print(f"  [ok] B20 drop detectors on {len(DETECT)} stage-3/4 narrow relus")
    return txt


# ---------------------------------------------------------------------------
# 6. onnx_frontend MP/MP_K override (atomic-arch latency rule)
# ---------------------------------------------------------------------------

def frontend_edits(dry: bool) -> None:
    f = ROOT / "scripts/onnx_frontend.py"
    # preserve the file's original line-ending convention (it is CRLF in this
    # repo; a bare newline="\n" rewrite makes a whole-file spurious diff)
    nl = "\r\n" if b"\r\n" in f.read_bytes() else "\n"
    txt = f.read_text()
    if "_MP32_STAGE34_OVERRIDE" in txt:
        print("  [skip] onnx_frontend: override already present")
        return
    ids = ", ".join(f'"node_conv_{c}": 32' for c in sorted(CONVS))
    dict_block = f"""# {MARK} 2026-06-10: ResNet stage-3/4 spatial 1x1 convs raised to MP=32 lanes
# (conv_datapath_mp_k, MP_K=8) by scripts/apply_resnet_mp32_stage34.py. Keeps
# compute_conv2d_latency_cycles consistent with the live RTL for a future
# regen (same convention as _A2_MP_OVERRIDE above). These modules are DENSE
# conv_datapath_mp_k instances -> lane-parallel pass cycles (k_groups + stages).
_MP32_STAGE34_OVERRIDE = {{
    {ids},
}}


def _conv_mac_parallelism("""
    txt = replace_once(txt, "def _conv_mac_parallelism(", dict_block, "frontend mp fn", f.name)
    old_ov = ("""    ov = _A2_MP_OVERRIDE.get(getattr(spec, "module_id", None))
    if ov is not None:
        return ov""")
    new_ov = old_ov + f"""
    ov32 = _MP32_STAGE34_OVERRIDE.get(getattr(spec, "module_id", None))  # {MARK}
    if ov32 is not None:
        return ov32"""
    txt = replace_once(txt, old_ov, new_ov, "frontend A2 lookup", f.name)
    old_mpk = """    if spec.weight is not None and len(spec.weight.shape) >= 4:
        kh, kw = int(spec.weight.shape[2]), int(spec.weight.shape[3])
        if kh == 3 and kw == 3:
            return 9
    return 1"""
    new_mpk = f"""    if getattr(spec, "module_id", None) in _MP32_STAGE34_OVERRIDE:
        return 8  # {MARK} stage-3/4 1x1s use the MP_K=8 dense mp_k datapath
    if spec.weight is not None and len(spec.weight.shape) >= 4:
        kh, kw = int(spec.weight.shape[2]), int(spec.weight.shape[3])
        if kh == 3 and kw == 3:
            return 9
    return 1"""
    txt = replace_once(txt, old_mpk, new_mpk, "frontend mp_k fn", f.name)
    if dry:
        print("  [dry] onnx_frontend: + _MP32_STAGE34_OVERRIDE (MP=32, MP_K=8 for the 11)")
        return
    backup(f)
    f.write_text(txt, newline=nl)
    print("  [ok] onnx_frontend: + _MP32_STAGE34_OVERRIDE (MP=32, MP_K=8 for the 11)")


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--convs", default=",".join(str(c) for c in sorted(CONVS)),
                   help="comma list of conv ids to patch (bisection)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--revert", action="store_true")
    p.add_argument("--skip-repack", action="store_true")
    p.add_argument("--no-verify", action="store_true", help="skip the repack inverse check")
    p.add_argument("--no-joins", action="store_true")
    p.add_argument("--no-bumps", action="store_true")
    p.add_argument("--no-detect", action="store_true")
    p.add_argument("--no-frontend", action="store_true")
    args = p.parse_args()

    if args.revert:
        if not BK.exists():
            die(f"no backup dir at {BK}")
        for src in sorted(BK.iterdir()):
            dst = (ROOT / "scripts" / src.name) if src.suffix == ".py" else (RTL / src.name)
            shutil.copy(src, dst)
            print(f"[revert] {src.name} -> {dst.relative_to(ROOT)}")
        print("[revert] done (repacked mp32 hex files left in place; harmless).")
        return

    convs = sorted(int(c) for c in args.convs.split(",") if c.strip())
    for c in convs:
        if c not in CONVS:
            die(f"unknown conv id {c}")
    goldens_before = sorted((ROOT / "output/goldens").rglob("*")) if (ROOT / "output/goldens").exists() else []

    print(f"== MP32 STAGE-3/4 == convs={convs} dry={args.dry_run}")

    print("\n-- per-conv MP + weights --")
    for c in convs:
        patch_conv(c, args.dry_run, args.skip_repack, not args.no_verify)

    print("\n-- top wrapper --")
    txt = TOP.read_text()
    orig = txt
    if not args.no_joins:
        for j, lhs in sorted(JOIN_LHS.items()):
            if lhs in convs:
                txt = lhs_skid_edits(txt, j)
    if not args.no_bumps:
        txt = fork_bump_edits(txt)
    if not args.no_detect:
        txt = detector_edits(txt)
    if txt != orig:
        if args.dry_run:
            print(f"  [dry] nn2rtl_top.v: +{txt.count(chr(10)) - orig.count(chr(10))} lines (not written)")
        else:
            backup(TOP)
            TOP.write_text(txt, newline="\n")
            print(f"  [ok] nn2rtl_top.v written (+{txt.count(chr(10)) - orig.count(chr(10))} lines)")

    print("\n-- frontend latency override --")
    if not args.no_frontend:
        frontend_edits(args.dry_run)

    # safety: goldens must be untouched (they are VALUE streams; MP changes timing only)
    goldens_after = sorted((ROOT / "output/goldens").rglob("*")) if (ROOT / "output/goldens").exists() else []
    if goldens_before != goldens_after:
        die("output/goldens changed — this script must never touch goldens")

    print("\nNEXT: verilator lint, then the e2e byte-exact gate:")
    print("  NN2RTL_VALUE_THREADS=1 NN2RTL_VALUE_XINIT=0 npx tsx scripts/run_nn2rtl_top_value.ts 0")
    print("  NN2RTL_VALUE_RUNONLY=1 ... 1   (vec1)")
    print("Gate: result=PASS mismatch_bytes=0 BOTH vectors + [b20-drop] all 0.")


if __name__ == "__main__":
    main()
