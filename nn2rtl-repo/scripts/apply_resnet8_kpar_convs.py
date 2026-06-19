#!/usr/bin/env python3
"""Re-parallelize the ResNet-8 spatial convs with MP-lane + K_PAR-tap parallelism.

WHY
---
scripts/apply_resnet8_serialize_convs.py collapsed the oversized parallel-OC
convs to a strictly-serial MP=4 FSM (one weight*tap multiply per cycle) so the
design would FIT the ZCU104. That fit, but at 7,486,125 e2e cycles for a
~12.5M-MAC network (~1.7 MAC/cyc). The serial FSM issues exactly ONE multiply
per cycle regardless of MP -- per OC pass it costs K_TOTAL*MP+~6 cycles, so the
TOTAL MAC cycles per pixel = OC_PASSES*(K_TOTAL*MP) = OC*K_TOTAL, INDEPENDENT of
MP. MP only amortizes the per-pass BIAS/SCALE/OUTPUT overhead. The real lever is
multipliers-per-cycle.

The routed design has HUGE headroom: LUT 53.5%, DSP 47.7% (~900 free of 1728),
BRAM 0/312. This script restores per-cycle parallelism: each cycle computes
MP lanes * K_PAR taps = MP*K_PAR INT8 multiplies, tree-summed per lane and
accumulated. ST_MAC drops from K_TOTAL*MP cycles/pass to (K_TOTAL/K_PAR)*MP.
Per OC pass: (K_TOTAL/K_PAR)*MP + 6.

BYTE-EXACTNESS
--------------
Same products as the serial FSM, just MP*K_PAR of them per cycle instead of 1,
summed in the SAME accumulation order (per-lane tree-sum of K_PAR products,
accumulated across k_groups). Same per-OC requant ROMs (compute_scale_approx),
same round/saturate, same flat weight values. The weights are repacked into a
WIDE ROM (MP*K_PAR bytes/word, indexed [oc_group*K_GROUPS + k_group]) read one
word per cycle -- identical byte values, just laid out wide for parallel read.

CONSTRAINTS
-----------
  * K_PAR must divide K_TOTAL.
  * MP must divide OC (no OC padding in this rewrite).
  * DSP cost per conv = MP*K_PAR multipliers. Keep sum across all 6 convs under
    the free DSP budget (~900) OR let some map to LUT (107K free).

CONFIG (per-conv MP, K_PAR) -- tuned to the cycle profile + budget.

VERIFY: NN2RTL_VALUE_THREADS=1 NN2RTL_VALUE_XINIT=0 \
        npx tsx scripts/run_resnet8_top_value.ts 0   -> result=PASS mismatch_bytes=0
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.golden_impl import compute_scale_approx  # noqa: E402
from scripts.repack_weights_wide import read_flat_weights, write_wide_weights  # noqa: E402

RTL_DIR = REPO / "output" / "resnet8" / "rtl"
IR_PATH = REPO / "output" / "resnet8" / "layer_ir.json"
WEIGHTS_DIR = REPO / "output" / "resnet8" / "weights"
BACKUP_TAG = ".prekpar"

# Per-conv (MP, K_PAR). MP divides OC; K_PAR divides K_TOTAL = IC*KH*KW.
# Multipliers per conv = MP*K_PAR.  (geom: conv_1/2 OC16 KT144 pix1024; conv_4
# OC32 KT256 pix256; conv_5 OC32 KT288 pix256; conv_7 OC64 KT512 pix64; conv_8
# OC64 KT576 pix64.)
#
# Multipliers spread across DSP (1728 on xczu7ev) + LUT (USE_DSP=False below ->
# Vivado fills DSP then spills overflow to LUT; ~80-110K LUT headroom). The e2e
# is gated by the SLOWEST single conv (layers fully overlap after the frame->
# elastic-FIFO swap in apply_resnet8_overlap_fifos.py), so balance all convs.
CONFIG: dict[str, tuple[int, int]] = {
    # ROUTABLE K_PAR mix. The e2e is gated by the conv_1+conv_2 SERIAL chain (the
    # two 1024px convs run back-to-back through relu); the OTHER convs (256px /
    # 64px) overlap-HIDE UNDER that ~23K window. So ONLY conv_1/conv_2 need to be
    # fast -> K_PAR=16 (256-wide MAC tree). conv_4/5/7/8 stay at the proven-
    # routable K_PAR=8 (128-wide tree); at K8 they sit at ~20-21K which is still
    # < the conv_1+conv_2 gate, so e2e is UNCHANGED (~23K) but only TWO modules
    # carry the placement-hard 256-wide tree instead of six. (All-six-K16 was
    # placement-pathological: global-place stalled on the wide MAC-tree cones.)
    # CERTIFIED ROUTABLE config: only the two 1024px bottleneck convs at K_PAR=16
    # (their conv_1+conv_2 serial chain dominates the e2e). conv_4/5/7/8 at K8:
    # they overlap-hide, and adding them to K16 (3-4 K16 convs) made global-place
    # pathological while saving <1K cycles (conv_1/2/5 K16 measured 30,334 vs the
    # 31,058 of conv_1/2 K16 -- not worth the placement risk). All-six-K16 reaches
    # 22,963 cyc byte-exact but is placement-INFEASIBLE (16-deep DSP cascade
    # macros congest global placement).
    # [TREE_STAGES UNLOCK] With the pipelined balanced adder tree (TREE_STAGES
    # below) the 16-deep linear DSP cascade that made all-6-K16 placement-
    # infeasible is BROKEN (the certified critical path DSP_ALU x16 -> a short
    # CARRY8/LUT path). That + the freed DSP (tree adds move off DSP) re-opens the
    # all-K16 = 22,963-cyc config. ALL SIX 3x3 convs at K_PAR=16:
    # [DSP_PACK SPEND] DSP packing halves the DSP/mult on the 3x3 convs (each conv
    # now (MP/2)*K_PAR packed DSPs). That freed ~50% of the DSP array, which we
    # SPEND on the e2e-gating conv_1/conv_2 (1024px) by raising their K_PAR from 16
    # to 24 (K_GROUPS 9->6): cuts their MAC-issue cycles ~3/pixel. K_TOTAL=144=24*6.
    #   * WHY K_PAR=24 (not 48): the kpar FSM mis-accumulates the boundary k_groups
    #     when K_GROUPS < the MAC pipeline depth (n_valid=6) -- a PRE-EXISTING latent
    #     fill/drain limit, NOT a packing-math error (the packing is byte-exact for
    #     ANY K_PAR, proven by isolation). K_PAR=48 (KG3) and 36 (KG4) FAIL byte-exact;
    #     K_PAR=24 (KG6) and 16 (KG9) PASS. So K_PAR=24 is the largest byte-exact step
    #     for K_TOTAL=144 (divisors of 144 giving KG>=6: 24->6, 16->9).
    # packed DSP conv_1/2 = (16/2)*24 = 192 each (384 total); conv_4/5/7/8 stay K16
    # = 128 each (512). Total conv DSP = 896 + stem ~16 ~ 912 (~53% of 1728) --
    # WELL under the 100% wall the legacy 1-mult/DSP config hit (~1552/1728).
    "node_conv2d_1": (16, 24),   # PACK KG6 192 DSP -> bottleneck, K_PAR raised 16->24
    "node_conv2d_2": (16, 24),   # PACK KG6 192 DSP -> bottleneck, K_PAR raised 16->24
    "node_conv2d_4": (16, 16),   # PACK: 128 packed DSP, KG16 (overlap-hidden)
    "node_conv2d_5": (16, 16),   # PACK: 128 packed DSP, KG18 (overlap-hidden)
    "node_conv2d_7": (16, 16),   # PACK: 128 packed DSP, KG32 (overlap-hidden)
    "node_conv2d_8": (16, 16),   # PACK: 128 packed DSP, KG36 (overlap-hidden)
}

# ===========================================================================
# TREE_STAGES -- pipelined balanced binary adder tree for the K_PAR reduction.
# ===========================================================================
# The certified post-route critical path (resnet8_mix_retimed_c14_timing.rpt) is
#   Logic Levels: 35 (DSP_ALU=16 DSP_OUTPUT=15 ...)  Data Path Delay 13.975ns
# i.e. the K_PAR=16 reduction `sum_lane_w += prod_w` synthesised as a 16-DEEP
# linear DSP P-cascade (DSP_ALU x16 chained PCOUT->PCIN). That cascade is BOTH
# the Fmax wall (~72 MHz @13.9ns) AND the global-place blocker at >=3 K16 convs
# (the long DSP-macro cascade cones congest Phase 2.4).
#
# TREE_STAGES[mid] = number of PIPELINED register levels inserted into that conv's
# per-lane product reduction. The K_PAR products are reduced by a BALANCED BINARY
# tree (depth ceil(log2(K_PAR))) instead of a linear chain; each tree level is a
# register stage. log2(16)=4 -> a 4-level registered tree replaces the 16-deep
# cascade => combinational depth per stage drops from ~16 DSP-ALU hops to ~1-2.
#
# BYTE-EXACT: integer addition is associative+commutative, so a balanced-tree sum
# equals the linear sum. Every tree-stage sum is sized +1 bit per add level (no
# truncation, no overflow). The pipeline adds (1+TREE_STAGES) cycles of MAC-issue
# latency (vs the legacy 2); the valid chain + FSM drain are deepened to match, so
# each k_group's partial is still accumulated EXACTLY once in the same final sum.
# e2e cycles grow by a small constant per conv (drain fill) -- re-gated + recorded.
#
# 0 (default for any mid not listed) = legacy single-stage LINEAR reduction
# (byte- AND latency-identical to the pre-tree generator). Set per conv below.
TREE_STAGES: dict[str, int] = {
    # STEP 1 (de-risk on the certified-floor base): pipeline the two K_PAR=16
    # bottleneck convs' reductions into a 4-level balanced tree (16->8->4->2->1)
    # to break the 16-deep DSP cascade (the certified critical path: DSP_ALU x16).
    # NOTE: in DSP_PACK mode emit_packed_reduction IGNORES tree_stages (it builds
    # its own depth-4-unpack tree); these values just satisfy apply_one's
    # tree_stages==ceil_log2(K_PAR) assertion. conv_1/2 raised to K_PAR=24 -> 5.
    "node_conv2d_1": 5,  # ceil_log2(24)
    "node_conv2d_2": 5,  # ceil_log2(24)
    # STEP 2 (all-K16): every K_PAR=16 conv gets the 4-level tree (ceil(log2 16)).
    "node_conv2d_4": 4,
    "node_conv2d_5": 4,
    "node_conv2d_7": 4,
    "node_conv2d_8": 4,
}

# ===========================================================================
# DSP_PACK -- WP487 dual-INT8-MACC: TWO output channels per DSP48E2.
# ===========================================================================
# Each DSP48E2 has a 27x18 multiplier. We compute TWO INT8 products that SHARE
# the activation operand `a` (one OC-pair (m,n) at the same tap k):
#     A = (w_n <<< OFFSET) + w_m   (signed, packed into the 27-bit A port)
#     B = a                        (8-bit signed, B port)
#     P = A * B = (a*w_n) <<< OFFSET + (a*w_m)
# Accumulate over a SHALLOW chunk (depth 4) of packed products in the balanced
# tree, then UNPACK each depth-4 packed node into the two signed per-OC partials
# and accumulate those into the existing per-lane acc registers (so the FSM /
# requant / output path is byte-identical to the legacy reduction). The DSP
# multiplier COUNT halves: (MP/2)*K_PAR packed products instead of MP*K_PAR.
#
# OFFSET (a.k.a. S) -- the constraints (PROVEN in /tmp/pack_grid.py + pack_tree.py):
#   * A-port 27-bit signed:   max|A| = 128*2^S + 128 <= 2^26  ->  S <= 18.
#   * signed LO field, unpack depth D=4:  |Sum_4 a*w_m| <= 4*128*128 = 65536
#     must fit signed S-bit field range [-2^(S-1), 2^(S-1)-1].  65536 <= 2^17-1
#     (=131071) -> S >= 18.  => S == 18 is the UNIQUE feasible offset.
#   * UNPACK at the depth-4 tree node (NOT deeper): depth-8 overflows the signed
#     field by 1 at the all-(-128,-128) corner (4*16384=65536 ok; 8*16384=131072
#     > 131071). Hence the tree reduces packed products only to depth-4 nodes,
#     unpacks, then sums the unpacked signed partials.
# Byte-exact verified over EVERY real conv K_TOTAL (144..576) for ALL sign
# combinations: /tmp/pack_tree.py (random 20k/shape + 8 extreme corners, 0 bad)
# and the exact RTL bit-slice unpack form: /tmp/pack_rtl_unpack.py (0 bad/500k).
#
# Requirements: MP even (pairs of OCs) AND K_PAR % 4 == 0 (depth-4 chunks).
# Default OFF (mid absent) -> the legacy 1-mult/DSP reduction is emitted, byte-
# AND latency-identical. ON -> packed reduction, SAME data_latency (5) as TREE4.
DSP_PACK_OFFSET = 18
DSP_PACK: dict[str, bool] = {
    "node_conv2d_1": True,
    "node_conv2d_2": True,
    "node_conv2d_4": True,
    "node_conv2d_5": True,
    "node_conv2d_7": True,
    "node_conv2d_8": True,
}

# ===========================================================================
# DSP_PACK_PRIM -- FORCE the dual-INT8-MACC into an EXPLICIT DSP48E2 primitive.
# ===========================================================================
# WHY: the inferred DSP_PACK form (pp_q <= ((w_n<<<18)+w_m) * a) did NOT hold the
# packing through Vivado synthesis -- the synthesizer applied the
# shift-through-multiply strength-reduction identity and DISTRIBUTED it back into
# two separate DSP multiplies (a*w_n)<<<18 + (a*w_m). Routed DSP came back at ~95%
# (1,648/1,728) instead of the projected ~53%; the dual-MACC was effectively LOST.
#
# FIX: when DSP_PACK_PRIM[mid] is True (requires DSP_PACK[mid] True), the packed
# product is computed by an EXPLICITLY INSTANTIATED DSP48E2 (one per packed
# product). A hard primitive cannot be distributed by the optimizer, so the two
# OCs are GUARANTEED to share one DSP. The primitive is `ifdef NN2RTL_SYNTHESIS-
# guarded (Verilator can't simulate the Xilinx unisim DSP48E2); the sim path is
# the proven behavioral packed product (bit-identical). NN2RTL_SYNTHESIS is
# defined ONLY by the Vivado synth driver, never by Verilator. Default OFF.
DSP_PACK_PRIM: dict[str, bool] = {
    "node_conv2d_1": True,
    "node_conv2d_2": True,
    "node_conv2d_4": True,
    "node_conv2d_5": True,
    "node_conv2d_7": True,
    "node_conv2d_8": True,
}

# DSP_PACK_KEEP -- the LIGHTER touch (latency-neutral): mark the packed-A operand
# (* keep, dont_touch *) so the optimizer cannot push the <<<18 across the
# inferred multiply. Mutually exclusive with DSP_PACK_PRIM per mid. May or may not
# hold one DSP through Vivado (only synth confirms); prepared as a fallback. To
# use it INSTEAD of the primitive on a given conv, set PRIM False + KEEP True here.
DSP_PACK_KEEP: dict[str, bool] = {}


def _ceil_log2(n: int) -> int:
    s, v = 0, 1
    while v < n:
        v *= 2
        s += 1
    return s

# Per-conv DSP/LUT mapping of the MAC multiplier tree. At K_PAR=8 (128-wide)
# Vivado RELIABLY infers ~128 DSP/conv when use_dsp=yes (vs K_PAR=16 which it
# dumps to LUT). With ALL 8 convs on DSP the array hits 98% (1698/1728) which
# places slowly. The small / overlap-hidden convs (conv_7, conv_8 at 64px) do NOT
# gate the e2e (conv_1/conv_2 at 1024px do), so we map THEIR mults to LUT (32%
# LUT headroom) -- cycle-IDENTICAL (same products) and it frees ~256 DSP, taking
# the array to a comfortable ~85% for clean placement/routing.
# Only the e2e BOTTLENECK convs (conv_1, conv_2 at 1024px, ~24.5K cyc -- the
# critical path) keep their MAC multipliers on the DSP58 array. Every other conv
# is overlap-HIDDEN (its latency is absorbed behind conv_1/conv_2), so its MAC
# mults go to LUT (LUT has ~27% headroom). This drops the DSP array from a
# placement-straining 95% to a comfortable ~50-55% -- cycle-IDENTICAL (LUT and
# DSP compute the same products) but far faster/cleaner to place & route.
USE_DSP_DEFAULT = False
# At K_PAR=16 each 3x3 conv needs 256 multipliers; 6 convs = 1536 > the 1728-DSP
# array once the stem (~16), 1x1 convs (256), residual adds (~96) and node_linear
# also claim DSP. So map the four convs that are OVERLAP-HIDDEN behind the
# conv_1/conv_2 bottleneck (conv_4/5/7/8) to LUT, and keep ONLY the bottleneck
# pair on the DSP array. conv_1/conv_2 on DSP = 512; everything else on DSP fits
# comfortably (~1100/1728). The 1024 LUT-mapped multipliers (~30-40 LUT each)
# are paid for by the ~45K LUTs freed by the FIFO->BRAM conversion
# (apply_resnet8_fifo_bram.py). Cycle-IDENTICAL (LUT and DSP compute the same
# products); only the placement/timing/utilisation differ.
# REBALANCE (after the first kpar16 synth showed conv_2 spilled 256 mult to LUT
# = 20K LUT, and conv_5 LUT-mapped = 22K LUT, pushing logic-LUT to 90%). At
# K_PAR=16 each LUT-mapped conv costs ~17-29K LUT, so put the BIG ones on DSP and
# FREE the DSP they need by moving the residual adds OFF DSP (apply_resnet8
# _adds_to_lut.py): adds are tiny 3-cycle INT8*const, cheap on LUT, NOT on any
# critical path. DSP budget after that: convs 1/2/4/5/7/8 (256 each = 1536) +
# stem(~30) + 1x1(0) ~ 1566 <= 1728 -> ALL six 3x3 convs go on DSP -> the ~120K
# of LUT-mapped MACs collapse onto the DSP array. LUTRAM already freed by
# apply_resnet8_fifo_bram.py.
USE_DSP_PER_CONV: dict[str, bool] = {
    "node_conv2d_1": True,    # bottleneck (1024px) -> DSP
    "node_conv2d_2": True,    # bottleneck (1024px) -> DSP
    # BALANCED for ROUTABILITY: the all-6-on-DSP config (~85% DSP) was placement-
    # pathological (global-place stuck). Target the baseline's routable density
    # profile (~72% DSP / ~85% LUT) by splitting: the BIG LUT-cost convs on DSP
    # (conv_1/2 bottleneck, conv_5 was 29K LUT, conv_8 ~20K), the two smaller /
    # fully-hidden convs on LUT (conv_4, conv_7). DSP ~= 4*256+stem ~ 1054 (61%);
    # LUT ~= base + conv_4/7 MAC ~ 184K (80%).
    # conv_4/5/7/8 are now K_PAR=8 (128 mult each); put them on DSP (128*4=512,
    # reliably inferred at K8) so they cost ~0 LUT. conv_1/conv_2 K16 on DSP
    # (256*2=512). Total conv DSP ~1024 + stem ~ 1054 (61%); adds on LUT.
    # [TREE_STAGES all-K16 DSP/LUT REBALANCE] The first all-K16 attempt mapped
    # conv_7/8 products to LUT -> LUT-as-logic hit 105.26% (242,511/230,400),
    # UNPLACEABLE -- an 8x8 signed multiply on fabric costs ~60-70 LUT, so 512
    # LUT-mapped products = ~33K LUT. But the tree moved the ADDS off DSP, leaving
    # DSP at only 46% (798/1728). So put ALL six convs' PRODUCTS on the DSP array
    # (256x6 = 1536 product-DSPs + stem ~ 1566, 90% -- fits) and free the LUT. The
    # tree intermediate adds stay on CARRY8/LUT for every conv (never use_dsp).
    "node_conv2d_4": True,    # K16 256 mult -> DSP
    "node_conv2d_5": True,    # K16 256 mult -> DSP
    "node_conv2d_7": True,    # K16 256 mult -> DSP (was LUT; LUT was over budget)
    "node_conv2d_8": True,    # K16 256 mult -> DSP (was LUT)
}


def read_geom(mid: str) -> dict:
    txt = (RTL_DIR / f"{mid}.v").read_text()
    g = {}
    for key in ["IC", "OC", "IH", "IW", "OH", "OW", "KH", "KW", "SH", "SW", "PH", "PW"]:
        m = re.search(rf"localparam integer {key}\s*=\s*(\d+);", txt)
        if not m:
            raise SystemExit(f"{mid}: localparam {key} not found")
        g[key] = int(m.group(1))
    return g


def per_oc_pairs(mid: str):
    ir = json.loads(IR_PATH.read_text())
    layer = next(l for l in ir["layers"]
                 if l.get("module_id") == mid and l.get("op_type") == "conv2d")
    sf = layer["scale_factor_per_oc"]
    pairs = [compute_scale_approx(float(s)) for s in sf]
    oc = layer["output_shape"][1]
    if len(pairs) != oc:
        raise SystemExit(f"{mid}: {len(pairs)} scales vs OC={oc}")
    return pairs, oc


def backup(path: Path, tag: str = BACKUP_TAG):
    bak = path.with_suffix(path.suffix + tag)
    if not bak.exists():
        bak.write_bytes(path.read_bytes())
        print(f"  backup -> {bak.name}")


def build_wide_hex(mid: str, g: dict, mp: int, k_par: int) -> str:
    """Repack the flat weights into MP*K_PAR-byte-wide words; return hex path."""
    flat = WEIGHTS_DIR / f"{mid}_weights.hex"
    if not flat.exists():
        raise SystemExit(f"{mid}: flat weights hex not found: {flat}")
    weights = read_flat_weights(flat)
    oc = g["OC"]
    k_total = g["IC"] * g["KH"] * g["KW"]
    out = WEIGHTS_DIR / f"{mid}_weights_wide_mp{mp}_kp{k_par}.hex"
    # wgt_bits=8 (INT8), mp_k = K_PAR. Layout: bits[(lane*K_PAR+kpos)*8 +: 8]
    # = weight(oc=g*MP+lane, k=k_group*K_PAR+kpos). Matches the wide read below.
    entries, padded = write_wide_weights(out, weights, oc, k_total, mp, k_par, wgt_bits=8)
    print(f"  wide hex -> {out.name} ({entries} words, padded_zeros={padded})")
    return out.as_posix()


def emit_packed_reduction(mp: int, k_par: int, use_dsp: bool,
                          offset: int, prim: bool = False,
                          keep: bool = False) -> tuple[str, str, str, int]:
    """Emit the WP487 dual-INT8-MACC packed reduction (TWO OCs per DSP).

    Returns (decls, seq_body, final_partial_name, data_latency) with the SAME
    contract as emit_tree_reduction: final_partial[lane] holds this k_group's
    signed partial for output channel `lane`, so the FSM __STAGE3_ACCUM__
    (acc[lane] += final_partial[lane]) is byte-identical to the legacy path.

    Pairing: lane 2*pp = w_m (LOW field), lane 2*pp+1 = w_n (HIGH field). For each
    pair pp and tap k:  packed_A = (w_n <<< OFFSET) + w_m ; product = packed_A*tap.
    Tree-sum the K_PAR packed products to DEPTH-4 nodes, unpack each (signed LO =
    node[OFFSET-1:0]; signed HI = node[..:OFFSET] + node[OFFSET-1] borrow), then
    sum the K_PAR/4 unpacked LO/HI partials into lane_partial[2pp]/[2pp+1].

    Pipeline (data_latency from weight_word_q/tap_q, CONSTANT 5 for any K_PAR):
      L1: registered packed products pp_q       (DSP mult reg)          -> +1
      Ld: 2 registered tree levels (K_PAR -> K_PAR/2 -> K_PAR/4 nodes)  -> +2
      Lu: registered unpack (lo/hi per depth-4 node)                    -> +1
      Ls: registered sum of the K_PAR/4 unpacked nodes -> lane_partial  -> +1

    REALIZING THE PACKED MULTIPLY (the WP487 trap):
      prim=False, keep=False  (legacy): pp_q <= pack_a_comb * tap_q, INFERRED.
        Vivado synthesis DISTRIBUTES (w_n<<<18 + w_m)*a back into two separate DSP
        multiplies (a*w_n)<<<18 and a*w_m -> the dual-MACC packing is LOST (routed
        DSP came back ~95%, not the projected ~53%). The shift-through-multiply is
        a strength-reduction identity the optimizer sees through the inferred mult.

      keep=True (latency-neutral lighter touch): the packed-A operand is a
        (* keep="true",dont_touch="true" *) WIRE so the optimizer cannot push the
        <<<18 across the multiply -> it is FORCED to multiply the opaque 27-bit A.
        Same single-register product (pp_q), same data_latency. May or may not hold
        one DSP through Vivado -- only synth confirms (the optimizer can still
        re-derive A internally; the keep is on the operand node, not the mult).

      prim=True (the hard fix): the packed product is computed by an EXPLICITLY
        INSTANTIATED DSP48E2 primitive (USE_MULT="MULTIPLY", OPMODE=9'b000000101 ->
        P = A*B, no accumulation; AREG=0/BREG=0/MREG=1/PREG=0 -> ONE register stage
        on the product, identical latency to pp_q <= pp_comb). A=(w_n<<<18)+w_m
        sign-extended to 27b, B=a sign-extended to 18b. Vivado CANNOT distribute a
        hard primitive -> the dual-MACC packing is GUARANTEED to stay one DSP.
        DSP48E2 datapath proven == (a*w_n)<<<18 + (a*w_m) over 1,000,008 cases incl.
        all 8 extreme corners (scripts/dsp48e2_datapath_model.py).
        VERILATOR CANNOT simulate the Xilinx unisim DSP48E2 (tristate-in-top-level-IO
        + glbl.GSR hierarchical-ref errors), so the primitive is `ifdef
        NN2RTL_SYNTHESIS-guarded; the `ifndef sim path is the proven behavioral
        product pp_q <= pack_a_comb*tap_q (BIT-IDENTICAL to the legacy reduction).
        NN2RTL_SYNTHESIS is defined ONLY by Vivado synth (run_resnet8_synth.ts
        `-verilog_define NN2RTL_SYNTHESIS=1`); Verilator never defines it.
    """
    assert mp % 2 == 0, f"DSP_PACK needs even MP, got {mp}"
    assert k_par % 4 == 0, f"DSP_PACK needs K_PAR % 4 == 0, got {k_par}"
    assert k_par >= 8, (f"DSP_PACK needs K_PAR >= 8 (so >=1 tree level to depth-4 "
                        f"nodes); K_PAR=4 path not emitted. got {k_par}")
    assert not (prim and keep), "prim and keep are mutually exclusive packed variants"
    use_dsp_attr = '(* use_dsp = "yes" *) ' if (use_dsp and not prim) else ""
    PAIRS = mp // 2
    N4 = k_par // 4                       # number of depth-4 nodes per pair
    # Reduce the K_PAR packed products to N4 depth-4 nodes by halving the count
    # TWICE (K_PAR -> K_PAR/2 -> K_PAR/4 == N4). Each balanced-binary level pairs
    # adjacent nodes, so after 2 levels node j == sum of products [4j..4j+3] --
    # exactly the depth-4 group the unpack expects. K_PAR % 4 == 0 guarantees both
    # halvings are exact (no floor-pad), so this is ALWAYS 2 levels for any K_PAR.
    tree_levels = 2
    counts = [k_par, k_par // 2, k_par // 4]
    assert counts[-1] == N4, (k_par, N4, counts)

    dl: list[str] = []
    dl.append("    // [DSP_PACK] WP487 dual-INT8-MACC: TWO OCs per DSP48E2 (shared activation).")
    dl.append(f"    // OFFSET={offset}: A=(w_n<<<{offset})+w_m (27b), B=a (8b); P=(a*w_n)<<<{offset}+(a*w_m).")
    if prim:
        dl.append("    // [DSP_PACK_PRIM] packed product realized by EXPLICIT DSP48E2 primitive")
        dl.append("    // (`ifdef NN2RTL_SYNTHESIS) so Vivado cannot distribute the <<<OFFSET through")
        dl.append("    // the multiply; `ifndef path = proven behavioral product (Verilator sim).")
    elif keep:
        dl.append("    // [DSP_PACK_KEEP] packed-A operand is keep/dont_touch -> optimizer cannot")
        dl.append("    // push the <<<OFFSET across the inferred multiply (latency-neutral).")
    dl.append(f"    localparam integer PACK_OFFSET = {offset};")
    dl.append("    localparam integer PACK_A_W    = 27;          // DSP48E2 A port")
    dl.append("    localparam integer PACK_PROD_W = PACK_A_W + 8; // 27x8 packed product")
    # depth-4 packed node width: PACK_PROD_W + 2 (sum of 4). Generous = PACK_PROD_W+4.
    dl.append("    localparam integer PACK_NODE_W = PACK_PROD_W + 4;")
    dl.append("    localparam integer PACK_PAIRS  = MP / 2;")
    dl.append("    localparam integer PACK_N4     = K_PAR / 4;   // depth-4 nodes per pair")
    dl.append("    // Level 0a combinational: packed A operand + packed DSP product.")
    keep_attr = '(* keep = "true", dont_touch = "true" *) ' if keep else ""
    dl.append(f"    {keep_attr}reg signed [PACK_A_W-1:0]    pack_a_comb [0:PACK_PAIRS*K_PAR-1];")
    dl.append("    reg signed [PACK_PROD_W-1:0] pp_comb     [0:PACK_PAIRS*K_PAR-1];")
    if prim:
        # pp_q is driven by DSP48E2 P[34:0] (synth) or a registered behavioral
        # product (sim). Declare it as WIRE here; the synth/sim branches drive it.
        dl.append("    wire signed [PACK_PROD_W-1:0] pp_q [0:PACK_PAIRS*K_PAR-1];")
        dl.append("`ifndef NN2RTL_SYNTHESIS")
        dl.append("    // [DSP_PACK_PRIM][sim] behavioral registered packed product (bit-identical")
        dl.append("    // to the legacy reduction; proven byte-exact). Verilator path only.")
        dl.append("    reg signed [PACK_PROD_W-1:0] pp_q_beh [0:PACK_PAIRS*K_PAR-1];")
        dl.append("`endif")
    else:
        dl.append(f"    {use_dsp_attr}reg signed [PACK_PROD_W-1:0] pp_q [0:PACK_PAIRS*K_PAR-1];")
    for L in range(1, tree_levels + 1):
        n = counts[L]
        dl.append(f"    reg signed [PACK_NODE_W-1:0] ptree_l{L} [0:PACK_PAIRS*{n}-1];")
    # unpack outputs: N4 lo + N4 hi per pair; widths = TREE_W (final per-OC partial width)
    dl.append("    reg signed [TREE_W-1:0] un_lo [0:PACK_PAIRS*PACK_N4-1];")
    dl.append("    reg signed [TREE_W-1:0] un_hi [0:PACK_PAIRS*PACK_N4-1];")
    dl.append("    reg signed [TREE_W-1:0] lane_partial [0:MP-1];")
    dl.append("    integer cs_pair, cs_kpos;")
    decls = "\n".join(dl)

    cb: list[str] = []
    # ---- Level 0a (comb): pack A then multiply by the shared tap ----
    cb.append("    // [DSP_PACK] Level-0a: pack A=(w_n<<<OFFSET)+w_m, packed product = A*tap.")
    cb.append("    always @* begin")
    cb.append("        for (cs_pair = 0; cs_pair < PACK_PAIRS; cs_pair = cs_pair + 1)")
    cb.append("            for (cs_kpos = 0; cs_kpos < K_PAR; cs_kpos = cs_kpos + 1) begin")
    cb.append("                pack_a_comb[cs_pair*K_PAR + cs_kpos] =")
    cb.append("                    ($signed(weight_word_q[((2*cs_pair+1) * K_PAR + cs_kpos) * 8 +: 8]) <<< PACK_OFFSET) +")
    cb.append("                     $signed(weight_word_q[((2*cs_pair  ) * K_PAR + cs_kpos) * 8 +: 8]);")
    cb.append("                pp_comb[cs_pair*K_PAR + cs_kpos] =")
    cb.append("                    pack_a_comb[cs_pair*K_PAR + cs_kpos] * $signed(tap_q[cs_kpos]);")
    cb.append("            end")
    cb.append("    end")
    if prim:
        # ---- DSP48E2-primitive packed product (synth) / behavioral (sim) ----
        cb.append("    // [DSP_PACK_PRIM] one DSP48E2 per packed product: P = A*B, A=(w_n<<<OFFSET)+w_m")
        cb.append("    // (27b), B=tap (18b). AREG=0/BREG=0/MREG=1/PREG=0 -> ONE register stage on")
        cb.append("    // the product (== pp_q <= pp_comb latency). OPMODE=9'b000000101 (X=M,Y=M,Z=0),")
        cb.append("    // ALUMODE=0 -> P=A*B, no accumulation. Vivado cannot distribute a hard primitive.")
        cb.append("`ifdef NN2RTL_SYNTHESIS")
        cb.append("    genvar gpk;")
        cb.append("    generate")
        cb.append("        for (gpk = 0; gpk < PACK_PAIRS*K_PAR; gpk = gpk + 1) begin : g_dsp_pack")
        cb.append("            // tap index for this packed product = gpk % K_PAR (shared B across pairs)")
        cb.append("            wire signed [7:0]  tap_b   = $signed(tap_q[gpk % K_PAR]);")
        cb.append("            wire signed [29:0] dsp_a   = {{3{pack_a_comb[gpk][PACK_A_W-1]}}, pack_a_comb[gpk]};")
        cb.append("            wire signed [17:0] dsp_b   = {{10{tap_b[7]}}, tap_b};")
        cb.append("            wire        [47:0] dsp_p;")
        cb.append("            DSP48E2 #(")
        cb.append("                .AREG(0), .BREG(0), .MREG(1), .PREG(0), .ADREG(0), .DREG(0),")
        cb.append("                .ACASCREG(0), .BCASCREG(0), .CREG(0), .CARRYINREG(0),")
        cb.append("                .CARRYINSELREG(0), .ALUMODEREG(0), .INMODEREG(0), .OPMODEREG(0),")
        cb.append('                .USE_MULT("MULTIPLY"), .AMULTSEL("A"), .BMULTSEL("B"),')
        cb.append('                .USE_SIMD("ONE48"), .A_INPUT("DIRECT"), .B_INPUT("DIRECT")')
        cb.append("            ) u_dsp_pack (")
        cb.append("                .P(dsp_p), .PCOUT(), .ACOUT(), .BCOUT(), .CARRYCASCOUT(), .CARRYOUT(),")
        cb.append("                .MULTSIGNOUT(), .OVERFLOW(), .PATTERNBDETECT(), .PATTERNDETECT(),")
        cb.append("                .UNDERFLOW(), .XOROUT(),")
        cb.append("                .A(dsp_a), .ACIN(30'd0), .ALUMODE(4'b0000), .B(dsp_b), .BCIN(18'd0),")
        cb.append("                .C(48'd0), .CARRYCASCIN(1'b0), .CARRYIN(1'b0), .CARRYINSEL(3'b000),")
        cb.append("                .CEA1(1'b0), .CEA2(1'b0), .CEAD(1'b0), .CEALUMODE(1'b1), .CEB1(1'b0),")
        cb.append("                .CEB2(1'b0), .CEC(1'b0), .CECARRYIN(1'b0), .CECTRL(1'b1), .CED(1'b0),")
        cb.append("                .CEINMODE(1'b1), .CEM(1'b1), .CEP(1'b1), .CLK(clk), .D(27'd0),")
        cb.append("                .INMODE(5'b00000), .MULTSIGNIN(1'b0), .OPMODE(9'b000000101),")
        cb.append("                .PCIN(48'd0), .RSTA(1'b0), .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0),")
        cb.append("                .RSTB(1'b0), .RSTC(1'b0), .RSTCTRL(1'b0), .RSTD(1'b0), .RSTINMODE(1'b0),")
        cb.append("                .RSTM(1'b0), .RSTP(1'b0)")
        cb.append("            );")
        cb.append("            assign pp_q[gpk] = dsp_p[PACK_PROD_W-1:0];")
        cb.append("        end")
        cb.append("    endgenerate")
        cb.append("`else")
        cb.append("    // [DSP_PACK_PRIM][sim] behavioral registered packed product (proven byte-exact).")
        cb.append("    integer ppb_i;")
        cb.append("    always @(posedge clk) begin")
        cb.append("        for (ppb_i = 0; ppb_i < PACK_PAIRS*K_PAR; ppb_i = ppb_i + 1)")
        cb.append("            pp_q_beh[ppb_i] <= pp_comb[ppb_i];")
        cb.append("    end")
        cb.append("    genvar gpb;")
        cb.append("    generate")
        cb.append("        for (gpb = 0; gpb < PACK_PAIRS*K_PAR; gpb = gpb + 1) begin : g_pp_beh")
        cb.append("            assign pp_q[gpb] = pp_q_beh[gpb];")
        cb.append("        end")
        cb.append("    endgenerate")
        cb.append("`endif")
    # ---- Level 0b (reg DSP) + tree to depth-4 + unpack + sum (one sync block) ----
    cb.append("    // [DSP_PACK] Level-0b register (DSP packed mult) + tree->depth4 + unpack + sum.")
    cb.append("    integer pk_pair, pk_i, un_j;")
    cb.append("    always @(posedge clk) begin")
    if not prim:
        cb.append("        for (pk_i = 0; pk_i < PACK_PAIRS*K_PAR; pk_i = pk_i + 1)")
        cb.append("            pp_q[pk_i] <= pp_comb[pk_i];")
    cb.append("        for (pk_pair = 0; pk_pair < PACK_PAIRS; pk_pair = pk_pair + 1) begin")
    # registered tree levels to depth-4 nodes
    for L in range(1, tree_levels + 1):
        src = "pp_q" if L == 1 else f"ptree_l{L-1}"
        src_n = counts[L - 1]
        dst_n = counts[L]
        for j in range(dst_n):
            a = 2 * j
            b = 2 * j + 1
            dst = f"ptree_l{L}[pk_pair*{dst_n} + {j}]"
            if b < src_n:
                cb.append(f"            {dst} <= $signed({src}[pk_pair*{src_n} + {a}]) + $signed({src}[pk_pair*{src_n} + {b}]);")
            else:
                cb.append(f"            {dst} <= $signed({src}[pk_pair*{src_n} + {a}]);")
    cb.append("        end")
    # unpack stage (registered): each depth-4 node -> (lo, hi)
    node_arr = "pp_q" if tree_levels == 0 else f"ptree_l{tree_levels}"
    # NOTE tree_levels==0 means K_PAR==4 (N4==1): the depth-4 node IS pp_q[pair*4+..]
    # but that holds 4 SEPARATE products, not their sum. Guard: K_PAR%4==0 with
    # K_PAR==4 -> N4=1, tree_levels=ceil_log2(1)=0; we must SUM the 4 products first.
    cb.append("        // [DSP_PACK] unpack each depth-4 packed node into signed (lo,hi).")
    cb.append("        for (pk_pair = 0; pk_pair < PACK_PAIRS; pk_pair = pk_pair + 1)")
    cb.append("            for (un_j = 0; un_j < PACK_N4; un_j = un_j + 1) begin")
    cb.append(f"                un_lo[pk_pair*PACK_N4 + un_j] <= $signed({node_arr}[pk_pair*PACK_N4 + un_j][PACK_OFFSET-1:0]);")
    cb.append(f"                un_hi[pk_pair*PACK_N4 + un_j] <= $signed({node_arr}[pk_pair*PACK_N4 + un_j][PACK_NODE_W-1:PACK_OFFSET]) + {node_arr}[pk_pair*PACK_N4 + un_j][PACK_OFFSET-1];")
    cb.append("            end")
    # final sum stage (registered): sum the N4 unpacked lo/hi into lane_partial.
    cb.append("        // [DSP_PACK] sum the PACK_N4 unpacked partials -> lane_partial (per OC).")
    cb.append("        for (pk_pair = 0; pk_pair < PACK_PAIRS; pk_pair = pk_pair + 1) begin")
    # emit explicit sums (N4 is small: 1,2,4)
    lo_terms = " + ".join(f"$signed(un_lo[pk_pair*PACK_N4 + {j}])" for j in range(N4))
    hi_terms = " + ".join(f"$signed(un_hi[pk_pair*PACK_N4 + {j}])" for j in range(N4))
    cb.append(f"            lane_partial[2*pk_pair    ] <= {lo_terms};")
    cb.append(f"            lane_partial[2*pk_pair + 1] <= {hi_terms};")
    cb.append("        end")
    cb.append("    end")
    body = "\n".join(cb)

    # data_latency = 1 (pp_q) + tree_levels + 1 (unpack) + 1 (final sum)
    data_latency = 1 + tree_levels + 1 + 1
    return decls, body, "lane_partial", data_latency


def emit_tree_reduction(mp: int, k_par: int, tree_stages: int,
                        use_dsp: bool) -> tuple[str, str, str, int]:
    """Emit the per-lane reduction RTL.

    Returns (decls, comb_or_seq_body, final_partial_expr_per_lane, data_latency)
    where data_latency = number of register stages from `weight_word_q`/`tap_q`
    (Stage-1) to the final per-lane partial that feeds the accumulator.

    tree_stages == 0  -> legacy: combinational linear sum into a single register
                         `partial_q` (data_latency = 1, matches the pre-tree gen).
    tree_stages >= 1  -> balanced binary tree, each level registered. The number
                         of REGISTER levels = ceil(log2(k_par)); `tree_stages` must
                         equal that (asserted by the caller). data_latency =
                         ceil(log2(k_par)).

    BYTE-EXACT: products are identical; the per-lane sum is reassociated into a
    balanced tree (associativity of integer +). Each level's width grows +1 bit
    (no truncation). The final partial value per lane == the legacy linear sum.
    """
    # PROD_W=16, TREE_W = 16 + clog2(k_par). We size every tree node at TREE_W
    # (the final-sum width) for simplicity -- that is >= every intermediate sum's
    # required width (a level summing M terms of <=16+clog2(k_par)/... bits never
    # exceeds TREE_W since TREE_W already holds the sum of ALL k_par products),
    # so NO truncation at any node => byte-exact.
    use_dsp_attr = '(* use_dsp = "yes" *) ' if use_dsp else ""
    if tree_stages == 0:
        # Legacy linear reduction (byte- AND latency-identical to the pre-tree
        # generator). The combinational sum_lane_w feeds the single partial_q
        # register (the accumulate gates on mac_valid_q2 -> data_latency=1).
        decls = (
            f"    {use_dsp_attr}reg signed [TREE_W-1:0] partial_q [0:MP-1];\n"
            "    reg signed [TREE_W-1:0] sum_lane_w [0:MP-1];\n"
            "    reg signed [PROD_W-1:0] prod_w;\n"
            "    integer cs_lane, cs_kpos;"
        )
        comb = (
            "    always @* begin\n"
            "        for (cs_lane = 0; cs_lane < MP; cs_lane = cs_lane + 1) begin\n"
            "            sum_lane_w[cs_lane] = {TREE_W{1'b0}};\n"
            "            for (cs_kpos = 0; cs_kpos < K_PAR; cs_kpos = cs_kpos + 1) begin\n"
            "                prod_w = $signed(weight_word_q[(cs_lane * K_PAR + cs_kpos) * 8 +: 8]) *\n"
            "                         $signed(tap_q[cs_kpos]);\n"
            "                sum_lane_w[cs_lane] = sum_lane_w[cs_lane] + prod_w;\n"
            "            end\n"
            "        end\n"
            "    end\n"
            "    // Stage-2 register (single stage): MP partial sums.\n"
            "    integer p2reg_i;\n"
            "    always @(posedge clk) begin\n"
            "        for (p2reg_i = 0; p2reg_i < MP; p2reg_i = p2reg_i + 1)\n"
            "            partial_q[p2reg_i] <= sum_lane_w[p2reg_i];\n"
            "    end"
        )
        return decls, comb, "partial_q", 1

    levels = _ceil_log2(k_par)
    assert tree_stages == levels, (
        f"tree_stages={tree_stages} must equal ceil(log2(K_PAR={k_par}))={levels}")

    # ---- Declarations ----
    # Level 0a = combinational products: prod_comb[lane][k].
    # Level 0b = REGISTERED products: prod_q[lane][k]  (the DSP multiply register --
    #   registering the multiply output is what lets Vivado infer a DSP48 multiplier
    #   per product; a bare combinational 8x8 mult feeding a fabric adder tree is
    #   instead packed into LUTs ~60-70/mult, which at all-6-K16 overflowed the LUT
    #   array to 105%. Registering -> products map to the (46%-idle) DSP array).
    # Level L (1..levels) = registered balanced-tree sums: tree_l{L}[lane][j].
    # Width: prod = PROD_W (16); each tree level = TREE_W (final width, no trunc).
    dl: list[str] = []
    dl.append("    // [TREE_STAGES] pipelined balanced binary adder tree for the K_PAR reduction.")
    dl.append("    // Level 0a: combinational products. 0b: REGISTERED products (DSP mult reg).")
    dl.append("    // Levels 1..L: registered balanced-tree sums (each level halves the count).")
    dl.append(f"    reg signed [PROD_W-1:0] prod_comb [0:MP*K_PAR-1];")
    dl.append(f"    {use_dsp_attr}reg signed [PROD_W-1:0] prod_q [0:MP*K_PAR-1];")
    # node counts per level
    counts = [k_par]
    c = k_par
    for _ in range(levels):
        c = (c + 1) // 2
        counts.append(c)
    for L in range(1, levels + 1):
        n = counts[L]
        # Tree-add levels are short registered fabric adds (CARRY8/LUT); no use_dsp
        # (the DSP P-cascade we are breaking up is exactly these chained adds).
        dl.append(f"    reg signed [TREE_W-1:0] tree_l{L} [0:MP*{n}-1];")
    dl.append("    integer cs_lane, cs_kpos;")
    decls = "\n".join(dl)

    # ---- Level 0a: combinational products (always @*) ----
    cb: list[str] = []
    cb.append("    // [TREE_STAGES] Level-0a products (combinational, identical products to legacy).")
    cb.append("    always @* begin")
    cb.append("        for (cs_lane = 0; cs_lane < MP; cs_lane = cs_lane + 1)")
    cb.append("            for (cs_kpos = 0; cs_kpos < K_PAR; cs_kpos = cs_kpos + 1)")
    cb.append("                prod_comb[cs_lane*K_PAR + cs_kpos] =")
    cb.append("                    $signed(weight_word_q[(cs_lane * K_PAR + cs_kpos) * 8 +: 8]) *")
    cb.append("                    $signed(tap_q[cs_kpos]);")
    cb.append("    end")

    # ---- Level 0b (register products) + tree levels 1..L (one sync block) ----
    # prod_q registers the multiply output (DSP inference). Each tree level L:
    #   tree_l{L}[lane][j] <= (src 2j) + (src 2j+1)  [odd trailing node passes thru].
    #   src = prod_q for L==1 else tree_l{L-1}. Sync-only, no reset (K1/FDRE: every
    #   node is written every cycle before any reader).
    cb.append("    // [TREE_STAGES] Level-0b register (DSP mult) + registered balanced tree.")
    cb.append("    integer tr_lane, pr_i;")
    cb.append("    always @(posedge clk) begin")
    cb.append("        for (pr_i = 0; pr_i < MP*K_PAR; pr_i = pr_i + 1)")
    cb.append("            prod_q[pr_i] <= prod_comb[pr_i];")
    cb.append("        for (tr_lane = 0; tr_lane < MP; tr_lane = tr_lane + 1) begin")
    for L in range(1, levels + 1):
        src = "prod_q" if L == 1 else f"tree_l{L-1}"
        src_n = counts[L - 1]
        dst_n = counts[L]
        for j in range(dst_n):
            a = 2 * j
            b = 2 * j + 1
            dst = f"tree_l{L}[tr_lane*{dst_n} + {j}]"
            if b < src_n:
                ea = f"$signed({src}[tr_lane*{src_n} + {a}])"
                eb = f"$signed({src}[tr_lane*{src_n} + {b}])"
                cb.append(f"            {dst} <= {ea} + {eb};")
            else:
                ea = f"$signed({src}[tr_lane*{src_n} + {a}])"
                cb.append(f"            {dst} <= {ea};")
    cb.append("        end")
    cb.append("    end")
    body = "\n".join(cb)

    # final per-lane partial expression: tree_l{levels}[lane*1 + 0].
    # data_latency = 1 (prod_q register) + levels (tree register stages).
    return decls, body, f"tree_l{levels}", 1 + levels


def emit_kpar_fsm(mid: str, g: dict, pairs, mp: int, k_par: int, wide_hex: str,
                  use_dsp: bool, tree_stages: int = 0, dsp_pack: bool = False,
                  dsp_prim: bool = False, dsp_keep: bool = False) -> str:
    IC, OC = g["IC"], g["OC"]
    IH, IW, OH, OW = g["IH"], g["IW"], g["OH"], g["OW"]
    KH, KW, SH, SW, PH, PW = g["KH"], g["KW"], g["SH"], g["SW"], g["PH"], g["PW"]
    K_TOTAL = IC * KH * KW
    if K_TOTAL % k_par != 0:
        raise SystemExit(f"{mid}: K_TOTAL={K_TOTAL} not divisible by K_PAR={k_par}")
    if OC % mp != 0:
        raise SystemExit(f"{mid}: OC={OC} not divisible by MP={mp}")
    OC_PASSES = OC // mp
    K_GROUPS = K_TOTAL // k_par
    IN_W = IC * 8
    OUT_W = OC * 8
    WIDE_W = mp * k_par * 8

    b_hex = (WEIGHTS_DIR / f"{mid}_bias.hex").as_posix()
    if not (WEIGHTS_DIR / f"{mid}_bias.hex").exists():
        raise SystemExit(f"{mid}: bias hex not found: {b_hex}")

    mult_lines = "\n".join(
        f"        scale_mult_rom[{i}]  = 16'sd{m};" for i, (m, sh) in enumerate(pairs))
    shf_lines = "\n".join(
        f"        scale_shift_rom[{i}] = 6'd{sh};" for i, (m, sh) in enumerate(pairs))
    dsp_tag = "DSP" if use_dsp else "LUT"
    tree_tag = f" TREE{tree_stages}" if tree_stages else ""
    if dsp_pack and dsp_prim:
        pack_tag = " PACK2PRIM"
    elif dsp_pack and dsp_keep:
        pack_tag = " PACK2KEEP"
    elif dsp_pack:
        pack_tag = " PACK2"
    else:
        pack_tag = ""
    prim_note = ("// DSP_PACK_PRIM: the packed product is an EXPLICIT DSP48E2 primitive "
                 "(`ifdef\n// NN2RTL_SYNTHESIS, OPMODE=9'b000000101 P=A*B, MREG=1) so Vivado CANNOT "
                 "distribute\n// the <<<OFFSET back into two DSP multiplies (the inferred-mult trap "
                 "that returned\n// DSP ~95%). `ifndef path = proven behavioral product (Verilator "
                 "sim, bit-identical).\n") if (dsp_pack and dsp_prim) else ""
    keep_note = ("// DSP_PACK_KEEP: packed-A operand is (* keep, dont_touch *) so the optimizer "
                 "cannot\n// strength-reduce the <<<OFFSET across the inferred multiply "
                 "(latency-neutral fallback).\n") if (dsp_pack and dsp_keep) else ""
    pack_note = ("// DSP_PACK: WP487 dual-INT8-MACC -- TWO OCs share each DSP48E2 (A=(w_n<<<{off})"
                 "+w_m,\n// B=a; P=(a*w_n)<<<{off}+(a*w_m)). The K_PAR packed products tree-sum to "
                 "depth-4\n// nodes, then UNPACK (signed lo=node[{om1}:0], hi=node[..:{off}]+borrow) "
                 "into the two\n// per-OC signed partials -> halves the DSP multiplier count "
                 "((MP/2)*K_PAR packed\n// products). Byte-exact (OFFSET={off} is the unique offset: "
                 "A fits 27b @ S<=18, depth-4\n// LO field fits @ S>=18); same data_latency as TREE4 "
                 "(re-gated byte-exact).\n").format(
                     off=DSP_PACK_OFFSET, om1=DSP_PACK_OFFSET - 1) if dsp_pack else ""
    tree_note = ("// TREE_STAGES={ts}: K_PAR reduction is a {ts}-level PIPELINED balanced "
                 "binary adder\n// tree (breaks the {kp}-deep linear DSP cascade -> shorter "
                 "critical path + frees\n// global placement). Byte-exact (associative "
                 "integer adds, +1 bit/level, no trunc);\n// adds {ts} cycles of MAC-issue "
                 "latency (valid chain + drain deepened to match).\n").format(
                     ts=tree_stages, kp=k_par) if tree_stages else ""

    body = f"""// {mid} -- {KH}x{KW} stride-{SH} pad-{PH} conv (IC={IC}, OC={OC}, IH=IW={IH}, OH=OW={OH}).
// RE-PARALLELIZED: MP={mp} lanes x K_PAR={k_par} [{dsp_tag}]{tree_tag}{pack_tag} taps = {mp*k_par} INT8 multiplies/cycle.
// ST_MAC = K_GROUPS({K_GROUPS}) * MP({mp}) cycles/pass; OC_PASSES={OC_PASSES} passes/pixel.
// Byte-exact vs the serial MP=4 FSM (same products, same accumulation order, same
// per-OC requant compute_scale_approx, same round/saturate). Weights repacked WIDE
// (MP*K_PAR bytes/word) read one word/cycle.
{pack_note}{prim_note}{keep_note}{tree_note}

module {mid} (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       valid_in,
    output wire                       ready_in,
    input  wire [{IN_W-1}:0]               data_in,
    output wire                       valid_out,
    output wire [{OUT_W-1}:0]               data_out
);
    localparam integer IC          = {IC};
    localparam integer OC          = {OC};
    localparam integer IH          = {IH};
    localparam integer IW          = {IW};
    localparam integer OH          = {OH};
    localparam integer OW          = {OW};
    localparam integer KH          = {KH};
    localparam integer KW          = {KW};
    localparam integer SH          = {SH};
    localparam integer SW          = {SW};
    localparam integer PH          = {PH};
    localparam integer PW          = {PW};
    localparam integer K_TOTAL     = IC * KH * KW; // {K_TOTAL}
    localparam integer MP          = {mp};
    localparam integer K_PAR       = {k_par};
    localparam integer K_GROUPS    = K_TOTAL / K_PAR;  // {K_GROUPS}
    localparam integer OC_PASSES   = OC / MP;          // {OC_PASSES}
    localparam integer NUM_WIDE    = OC_PASSES * K_GROUPS; // {OC_PASSES*K_GROUPS}
    localparam integer WIDE_W      = MP * K_PAR * 8;   // {WIDE_W}

    localparam integer PROD_W       = 16;
    localparam integer TREE_W       = PROD_W + $clog2(K_PAR);
    localparam integer ACC_W        = TREE_W + $clog2(K_GROUPS > 1 ? K_GROUPS : 2);
    localparam integer BIAS_W       = 32;
    localparam integer BIASED_W     = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1;
    localparam integer SCALE_MULT_W = 16;
    localparam integer SCALED_W     = BIASED_W + SCALE_MULT_W;

    localparam integer KGROUP_W     = (K_GROUPS <= 1) ? 1 : $clog2(K_GROUPS);
    localparam integer OC_GROUP_W   = (OC_PASSES <= 1) ? 1 : $clog2(OC_PASSES);

    // ---- Per-OC requant ROMs: compute_scale_approx(scale_factor_per_oc[oc]) ----
    reg signed [SCALE_MULT_W-1:0] scale_mult_rom  [0:OC-1];
    reg        [5:0]              scale_shift_rom [0:OC-1];
    initial begin
{mult_lines}
{shf_lines}
    end

    reg started, start_pulse, pending_rearm;
    wire sched_out_frame_done;
    wire                              sched_needs_real_input;
    wire                              sched_ready_in;
    wire                              sched_output_fires;
    wire                              sched_advance;
    wire [$clog2(IH + PH + 1)-1:0]    sched_in_row;
    wire [$clog2(IW + PW + 1)-1:0]    sched_in_col;
    wire [$clog2(OH * OW + 1)-1:0]    sched_outputs_emitted;
    wire [KH*KW*IC*8-1:0]             window_flat;
    wire                              mac_busy_w;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            started       <= 1'b0;
            start_pulse   <= 1'b0;
            pending_rearm <= 1'b0;
        end else begin
            start_pulse <= 1'b0;
            if (sched_out_frame_done) begin
                pending_rearm <= 1'b1;
            end
            if (!started) begin
                started     <= 1'b1;
                start_pulse <= 1'b1;
            end else if (pending_rearm && !mac_busy_w) begin
                started       <= 1'b0;
                pending_rearm <= 1'b0;
            end
        end
    end

    wire stall_in = mac_busy_w;

    coord_scheduler #(
        .IH(IH), .IW(IW), .OH(OH), .OW(OW),
        .KH(KH), .KW(KW), .SH(SH), .SW(SW),
        .PH(PH), .PW(PW)
    ) scheduler (
        .clk(clk), .rst_n(rst_n),
        .start(start_pulse),
        .stall_in(stall_in),
        .valid_in(valid_in),
        .ready_in(sched_ready_in),
        .needs_real_input(sched_needs_real_input),
        .in_row(sched_in_row),
        .in_col(sched_in_col),
        .output_fires(sched_output_fires),
        .advance(sched_advance),
        .in_frame_done(),
        .out_frame_done(sched_out_frame_done),
        .outputs_emitted(sched_outputs_emitted)
    );

    line_buf_window #(
        .IC(IC), .IW(IW), .IH(IH),
        .KH(KH), .KW(KW), .PW(PW), .PH(PH)
    ) lbw (
        .clk(clk), .rst_n(rst_n),
        .frame_start(start_pulse),
        .sched_in_row(sched_in_row),
        .sched_in_col(sched_in_col),
        .sched_needs_real_input(sched_needs_real_input),
        .sched_advance(sched_advance),
        .sched_output_fires(sched_output_fires),
        .valid_in(valid_in),
        .data_in(data_in),
        .window_flat(window_flat)
    );

    localparam ST_IDLE   = 3'd0;
    localparam ST_MAC    = 3'd1;
    localparam ST_BIAS   = 3'd2;
    localparam ST_SCALE  = 3'd3;
    localparam ST_OUTPUT = 3'd4;

    reg [2:0]   state;
    reg         valid_out_r;
    reg [{OUT_W-1}:0] data_out_r;

    // ---- Wide weight ROM: MP*K_PAR bytes/word, [oc_group*K_GROUPS + k_group] ----
    (* rom_style = "block", ram_style = "block" *) reg [WIDE_W-1:0]  weights_wide [0:NUM_WIDE-1];
    (* rom_style = "block", ram_style = "block" *) reg signed [31:0] biases_mem   [0:OC-1];
    initial begin
        $readmemh("{wide_hex}", weights_wide);
        $readmemh("{b_hex}",       biases_mem);
    end

    reg signed [ACC_W-1:0]    acc    [0:MP-1];
    reg signed [BIASED_W-1:0] biased [0:MP-1];
    reg signed [SCALED_W-1:0] scaled [0:MP-1];
    reg [5:0]                 shift_lane [0:MP-1];
    reg signed [SCALED_W-1:0] v_tmp;

    reg [KGROUP_W-1:0]   k_group;
    reg [OC_GROUP_W-1:0] oc_group;

    integer i, lane_i;
    integer bias_oc;
    integer out_oc;

    assign mac_busy_w = (state != ST_IDLE);
    assign valid_out  = valid_out_r;     // [INVARIANT:VALID_OUT_LATENCY]
    assign data_out   = data_out_r;
    assign ready_in   = sched_ready_in;  // [INVARIANT:READY_IN_GATING]

    wire [$clog2(NUM_WIDE+1)-1:0] weight_read_addr = oc_group * K_GROUPS + k_group;

    // Window-tap indexer. Linear k index -> (kh,kw,ic) -> flat window slice.
    function [7:0] tap_at;
        input integer k_lin;
        integer kh_idx, kw_idx, ic_idx, flat_idx;
        begin
            kh_idx   = (k_lin % (KH * KW)) / KW;
            kw_idx   = k_lin % KW;
            ic_idx   = k_lin / (KH * KW);
            flat_idx = kh_idx * KW * IC + kw_idx * IC + ic_idx;
            tap_at   = window_flat[flat_idx*8 +: 8];
        end
    endfunction

    // ---- Stage 1: register wide weight word + K_PAR taps for current k_group ----
    reg [WIDE_W-1:0] weight_word_q;
    reg signed [7:0] tap_q [0:K_PAR-1];
    integer ld_i;
    always @(posedge clk) begin
        weight_word_q <= weights_wide[weight_read_addr];
        for (ld_i = 0; ld_i < K_PAR; ld_i = ld_i + 1)
            tap_q[ld_i] <= $signed(tap_at(k_group * K_PAR + ld_i));
    end

    // ---- Stage 2: MP*K_PAR multipliers + per-lane reduction (legacy linear OR
    //               pipelined balanced adder tree, selected by TREE_STAGES). ----
__STAGE2_DECLS__
__STAGE2_COMB__

    reg                       mac_valid_q1;
    reg [OC_GROUP_W-1:0]      mac_oc_group_q1;
__MAC_VALID_DECLS__
    reg                       mac_done_issuing;
    integer p_i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= ST_IDLE;
            valid_out_r      <= 1'b0;
            data_out_r       <= {OUT_W}'d0;
            k_group          <= 0;
            oc_group         <= 0;
            mac_valid_q1     <= 1'b0;
            mac_oc_group_q1  <= 0;
__MAC_VALID_RESET__
            mac_done_issuing <= 1'b0;
            for (i = 0; i < MP; i = i + 1) begin
                acc[i]        <= 0;
                biased[i]     <= 0;
                scaled[i]     <= 0;
                shift_lane[i] <= 0;
            end
        end else begin
            valid_out_r <= 1'b0;

            // Stage 3: valid-chain propagation + gated accumulate into MP lanes.
            // The valid chain is deepened to match the data path's reduction
            // latency so each k_group's partial is accumulated EXACTLY once.
__STAGE3_ACCUM__

            case (state)
                ST_IDLE: begin
                    if (sched_output_fires) begin
                        state            <= ST_MAC;
                        k_group          <= 0;
                        oc_group         <= 0;
                        mac_valid_q1     <= 1'b0;
__IDLE_VALID_RESET__
                        mac_done_issuing <= 1'b0;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                            acc[lane_i] <= 0;
                    end
                end

                ST_MAC: begin
                    if (mac_done_issuing) begin
                        mac_valid_q1 <= 1'b0;
                        if (__DRAIN_COND__) begin
                            mac_done_issuing <= 1'b0;
                            state            <= ST_BIAS;
                        end
                    end else begin
                        mac_oc_group_q1 <= oc_group;
                        mac_valid_q1    <= 1'b1;
                        if (k_group == K_GROUPS - 1) begin
                            mac_done_issuing <= 1'b1;
                        end else begin
                            k_group <= k_group + 1'b1;
                        end
                    end
                end

                ST_BIAS: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        bias_oc = oc_group * MP + lane_i;
                        biased[lane_i] <= $signed(acc[lane_i]) + $signed(biases_mem[bias_oc]);
                    end
                    state <= ST_SCALE;
                end

                ST_SCALE: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        scaled[lane_i]     <= $signed(biased[lane_i]) *
                                              $signed(scale_mult_rom[oc_group * MP + lane_i]);
                        shift_lane[lane_i] <= scale_shift_rom[oc_group * MP + lane_i];
                    end
                    state <= ST_OUTPUT;
                end

                ST_OUTPUT: begin
                    for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1) begin
                        out_oc = oc_group * MP + lane_i;
                        // [INVARIANT:ROUNDING] single positive bias + arith >>> = golden floor.
                        v_tmp = (scaled[lane_i] +
                                 ($signed(__ROUND_BIAS__) <<< (shift_lane[lane_i] - 1))
                                ) >>> shift_lane[lane_i];
                        data_out_r[out_oc*8 +: 8] <=
                            (v_tmp >  127) ?  8'sd127 :
                            (v_tmp < -128) ? -8'sd128 : v_tmp[7:0];
                    end

                    if (oc_group == OC_PASSES - 1) begin
                        valid_out_r <= 1'b1;
                        state       <= ST_IDLE;
                    end else begin
                        oc_group         <= oc_group + 1'b1;
                        k_group          <= 0;
                        mac_valid_q1     <= 1'b0;
__OCPASS_VALID_RESET__
                        mac_done_issuing <= 1'b0;
                        for (lane_i = 0; lane_i < MP; lane_i = lane_i + 1)
                            acc[lane_i] <= 0;
                        state <= ST_MAC;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
"""
    round_bias = "{{(SCALED_W-1){1'b0}}, 1'b1}"

    # ---- Build the reduction (DSP_PACK packed, OR legacy linear / balanced tree) ----
    if dsp_pack:
        decls, comb, final_partial, data_latency = emit_packed_reduction(
            mp, k_par, use_dsp, DSP_PACK_OFFSET, prim=dsp_prim, keep=dsp_keep)
    else:
        decls, comb, final_partial, data_latency = emit_tree_reduction(
            mp, k_par, tree_stages, use_dsp)
    # Total valid-chain depth from k_group issue = 1 (mac_valid_q1, aligned with
    # weight_word_q) + data_latency (reduction register stages). Accumulate gates
    # on the LAST stage; legacy data_latency=1 -> q1,q2 (unchanged behavior).
    n_valid = 1 + data_latency
    # valid stage names: mac_valid_q1 .. mac_valid_q{n_valid}; oc_group tag tracks
    # in parallel (mac_oc_group_q1 .. q{n_valid}). q1 declared in the body.
    qnames = [f"mac_valid_q{i}" for i in range(1, n_valid + 1)]
    ocnames = [f"mac_oc_group_q{i}" for i in range(1, n_valid + 1)]
    last_v, last_oc = qnames[-1], ocnames[-1]

    # __MAC_VALID_DECLS__: declare q2..q{n_valid} (+ their oc tags). q1 already in body.
    mvd = []
    for i in range(2, n_valid + 1):
        mvd.append(f"    reg                       mac_valid_q{i};")
        mvd.append(f"    reg [OC_GROUP_W-1:0]      mac_oc_group_q{i};")
    mac_valid_decls = "\n".join(mvd) if mvd else "    // (no extra valid stages)"

    # __MAC_VALID_RESET__ (async reset block): all q2..q{n_valid}=0 + their oc tags.
    mvr = []
    for i in range(2, n_valid + 1):
        mvr.append(f"            mac_valid_q{i}     <= 1'b0;")
        mvr.append(f"            mac_oc_group_q{i}  <= 0;")
    mac_valid_reset = "\n".join(mvr) if mvr else "            // (no extra valid stages)"

    # __STAGE3_ACCUM__: shift the valid/oc chain by one each cycle, accumulate on
    # the LAST stage with the final reduced partial.
    s3 = []
    for i in range(2, n_valid + 1):
        s3.append(f"            mac_valid_q{i}    <= mac_valid_q{i-1};")
        s3.append(f"            mac_oc_group_q{i} <= mac_oc_group_q{i-1};")
    s3.append(f"            if ({last_v}) begin")
    s3.append(f"                for (p_i = 0; p_i < MP; p_i = p_i + 1)")
    s3.append(f"                    acc[p_i] <= acc[p_i] + $signed({final_partial}[p_i]);")
    s3.append(f"            end")
    stage3_accum = "\n".join(s3)

    # __DRAIN_COND__: ST_BIAS only after EVERY valid stage has cleared.
    drain_cond = " && ".join(f"!{q}" for q in qnames)

    # __IDLE_VALID_RESET__ / __OCPASS_VALID_RESET__: clear q2..q{n_valid}.
    ivr = []
    for i in range(2, n_valid + 1):
        ivr.append(f"                        mac_valid_q{i}     <= 1'b0;")
    idle_valid_reset = "\n".join(ivr) if ivr else "                        // (no extra valid stages)"
    ocr = []
    for i in range(2, n_valid + 1):
        ocr.append(f"                        mac_valid_q{i}     <= 1'b0;")
    ocpass_valid_reset = "\n".join(ocr) if ocr else "                        // (no extra valid stages)"

    return (body
            .replace("__ROUND_BIAS__", round_bias)
            .replace("__STAGE2_DECLS__", decls)
            .replace("__STAGE2_COMB__", comb)
            .replace("__MAC_VALID_DECLS__", mac_valid_decls)
            .replace("__MAC_VALID_RESET__", mac_valid_reset)
            .replace("__STAGE3_ACCUM__", stage3_accum)
            .replace("__DRAIN_COND__", drain_cond)
            .replace("__IDLE_VALID_RESET__", idle_valid_reset)
            .replace("__OCPASS_VALID_RESET__", ocpass_valid_reset))


def _pack_tag(dsp_pack: bool, dsp_prim: bool, dsp_keep: bool) -> str:
    if dsp_pack and dsp_prim:
        return " PACK2PRIM"
    if dsp_pack and dsp_keep:
        return " PACK2KEEP"
    if dsp_pack:
        return " PACK2"
    return ""


def apply_one(mid: str) -> bool:
    mp, k_par = CONFIG[mid]
    use_dsp = USE_DSP_PER_CONV.get(mid, USE_DSP_DEFAULT)
    tree_stages = TREE_STAGES.get(mid, 0)
    dsp_pack = DSP_PACK.get(mid, False)
    dsp_prim = DSP_PACK_PRIM.get(mid, False)
    dsp_keep = DSP_PACK_KEEP.get(mid, False)
    if tree_stages != 0 and tree_stages != _ceil_log2(k_par):
        raise SystemExit(f"{mid}: TREE_STAGES={tree_stages} must equal "
                         f"ceil(log2(K_PAR={k_par}))={_ceil_log2(k_par)} or 0")
    if (dsp_prim or dsp_keep) and not dsp_pack:
        raise SystemExit(f"{mid}: DSP_PACK_PRIM/KEEP require DSP_PACK[{mid}]=True")
    if dsp_prim and dsp_keep:
        raise SystemExit(f"{mid}: DSP_PACK_PRIM and DSP_PACK_KEEP are mutually exclusive")
    if dsp_pack:
        if mp % 2 != 0:
            raise SystemExit(f"{mid}: DSP_PACK needs even MP, got {mp}")
        if k_par % 4 != 0:
            raise SystemExit(f"{mid}: DSP_PACK needs K_PAR % 4 == 0, got {k_par}")
    path = RTL_DIR / f"{mid}.v"
    txt = path.read_text()
    dsp_tag = "DSP" if use_dsp else "LUT"
    tree_tag = f" TREE{tree_stages}" if tree_stages else ""
    pack_tag = _pack_tag(dsp_pack, dsp_prim, dsp_keep)
    marker = f"RE-PARALLELIZED: MP={mp} lanes x K_PAR={k_par} [{dsp_tag}]{tree_tag}{pack_tag}"
    if marker in txt:
        print(f"{mid}: already MP={mp} K_PAR={k_par} {dsp_tag}{tree_tag}{pack_tag}; skip")
        return False
    g = read_geom(mid)
    pairs, oc = per_oc_pairs(mid)
    if oc != g["OC"]:
        raise SystemExit(f"{mid}: IR OC {oc} != .v OC {g['OC']}")
    backup(path)  # .prekpar = pristine pre-kpar serial form (kept)
    # Also snapshot the CURRENT certified PACK2 floor before any PRIM/KEEP rewrite,
    # so the 24,581-cyc / 8-of-8 floor is one `cp` away if PRIM regresses.
    if dsp_prim or dsp_keep:
        backup(path, ".prepack2prim")
    wide_hex = build_wide_hex(mid, g, mp, k_par)
    new = emit_kpar_fsm(mid, g, pairs, mp, k_par, wide_hex, use_dsp, tree_stages,
                        dsp_pack, dsp_prim, dsp_keep)
    path.write_text(new)
    K_TOTAL = g["IC"] * g["KH"] * g["KW"]
    print(f"{mid}: re-parallelized -> MP={mp} K_PAR={k_par} map={dsp_tag}{tree_tag}{pack_tag} "
          f"(IC={g['IC']} OC={g['OC']} K_TOTAL={K_TOTAL} "
          f"OC_PASSES={oc // mp} K_GROUPS={K_TOTAL // k_par} mult={mp * k_par} "
          f"dsp_mults={(mp//2)*k_par if dsp_pack else mp*k_par})")
    return True


def main():
    print("Re-parallelizing ResNet-8 convs with MP + K_PAR:")
    changed = 0
    for mid in CONFIG:
        if apply_one(mid):
            changed += 1
    print(f"Done. ({changed} convs re-parallelized)")


if __name__ == "__main__":
    main()
