#!/usr/bin/env python3
"""apply_resnet_kpar4.py — ENGINE K-PARALLEL P=4 for the ResNet-50 top,
EXTENDING the MBV2 KPAR4 fast walk to dense 3x3 dispatches. Anchor-asserted
+ idempotent; writes .prekp4r backups before first mutation of each file.

DESIGN (see docs/agent_tasks/RESNET_KPAR4_ANALYSIS.md)
------------------------------------------------------
* SHARED file output/rtl/engine/address_generator.v (K_PAR>1 generate
  branch ONLY — the K_PAR==1 verbatim-legacy branch is untouched):
  1. kpar_fast eligibility DROPS the (KH==KW==1) requirement: any dense
     layer with IC%4==0 and base%4==0 is fast. MBV2-INERT: every MBV2
     dense dispatch is 1x1 (its 12 non-1x1 dispatches are all depthwise,
     verified against depthwise_rom), so no MBV2 layer changes eligibility
     and the MBV2 fast walk is cycle-identical (re-gated, see analysis doc).
  2. FAST weight address uses the POS-MAJOR transposed layout
     (pass_offset + (kh*KW+kw)*IC + ic) which the ResNet _kp4 bank repack
     provides (scripts/repack_resnet_kpar4_banks.py, proofs P0-P4). For
     1x1 the formula equals the legacy ic-major one (pos==0) — MBV2's
     untransposed 1x1 banks keep working unchanged.
  3. FAST ic-wrap branch: at ic == IC-4 within a (kh,kw) position, advance
     kw/kh and step k_cnt by 4. UNREACHABLE for 1x1 (k_at_last fires the
     same cycle and has priority) -> MBV2 cycle-identical.
* output/rtl/nn2rtl_top.v (ResNet-only): K_PAR=4 on the shared_engine,
  banks 96b x 67072 -> 384b x 16768 (_kp4 files, tap-major), GROUP
  addressing (engine exports old>>2 -> [14:0]), tap-major 3072b bus.

A 4-aligned k-group never crosses a (kh,kw) boundary (IC%4==0 => k%4==ic%4),
so all 4 taps share one act word, one ic chunk, and one in_bounds decision —
padding zeroes a whole group exactly as it zeroed 4 serial steps.

Usage: python scripts/apply_resnet_kpar4.py [--check]
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AG = REPO / "output" / "rtl" / "engine" / "address_generator.v"
TOP = REPO / "output" / "rtl" / "nn2rtl_top.v"

_backed_up: set[Path] = set()


def patch(path: Path, old: str, new: str, tag: str, count: int = 1,
          after: str | None = None) -> None:
    """Anchor-asserted replace. Idempotent: presence of `new` == applied.
    `after`: only consider text AFTER the (unique) split anchor — used to
    disambiguate hunks duplicated across the AG's two generate branches."""
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"  [skip] {path.name}: {tag} already applied")
        return
    head = ""
    body = text
    if after is not None:
        n_split = text.count(after)
        if n_split != 1:
            raise SystemExit(f"SPLIT-ANCHOR FAIL {path.name} / {tag}: "
                             f"found {n_split}, want 1")
        i = text.index(after) + len(after)
        head, body = text[:i], text[i:]
    n = body.count(old)
    if n != count:
        raise SystemExit(f"ANCHOR FAIL {path.name} / {tag}: found {n}, want {count}")
    if path not in _backed_up:
        bak = path.with_name(path.name + ".prekp4r")
        if not bak.exists():
            bak.write_text(text, encoding="utf-8", newline="\n")
        _backed_up.add(path)
    path.write_text(head + body.replace(old, new), encoding="utf-8", newline="\n")
    print(f"  [ok]   {path.name}: {tag}")


KPAR_BRANCH = "end else begin : g_walk_kpar"   # unique split anchor in the AG


# ============================================================================
# 1. address_generator.v — K_PAR>1 branch: dense-3x3 fast-walk extension
# ============================================================================
def patch_ag() -> None:
    # 1a. eligibility: drop the 1x1 requirement (dense KxK now fast).
    patch(AG, """    wire        kpar_fast = (!cfg_depthwise)
                          && (cfg_kh == 3'd1) && (cfg_kw == 3'd1)
                          && (cfg_ic[1:0] == 2'b00) && (cfg_ic[11:2] != 10'd0)
                          && (cfg_weight_uram_base[1:0] == 2'b00);
""", """    // [KPAR4-RN 2026-06-10] eligibility EXTENDED to dense KxK (was 1x1-
    // only). For KH*KW>1 the fast walk REQUIRES that layer's weight region
    // be POS-MAJOR transposed in the repacked banks (word at (kh*KW+kw)*IC
    // + ic; scripts/repack_resnet_kpar4_banks.py) because the walk issues 4
    // consecutive ic of ONE (kh,kw) per cycle. For 1x1 the transposed and
    // legacy layouts are IDENTICAL (pos==0), so MBV2 (whose dense dispatches
    // are ALL 1x1 — its 12 non-1x1 dispatches are depthwise and stay
    // excluded by cfg_depthwise) is bit- and cycle-identical.
    wire        kpar_fast = (!cfg_depthwise)
                          && (cfg_ic[1:0] == 2'b00) && (cfg_ic[11:2] != 10'd0)
                          && (cfg_weight_uram_base[1:0] == 2'b00);
""", "eligibility: dense KxK fast")

    # 1b. fast pos-major weight address (after the fast_mask wires).
    patch(AG, """    reg  [3:0]  k_tap_mask_r;
    assign k_tap_mask = k_tap_mask_r;
""", """    reg  [3:0]  k_tap_mask_r;
    assign k_tap_mask = k_tap_mask_r;

    // [KPAR4-RN] FAST weight address: POS-MAJOR transposed layout, offset =
    // pass_offset + (kh*KW + kw)*IC + ic. Equals the legacy ic-major offset
    // when KH==KW==1 (kpos9==0). The fast 4-group's old address is 4-aligned
    // (base, pass_offset, pos*IC and ic are all %4==0), so the skeleton's
    // group export (addr>>2) lands on one repacked line, subword select 0.
    wire [8:0]  kpos9            = kh_offset + {6'b0, kw_cnt};        // kh*KW + kw <= 48
    wire [20:0] kpos_ic          = {12'b0, kpos9} * {9'b0, cfg_ic};   // <= 48*2048
    wire [21:0] weight_offset_fast =
        {2'b0, pass_offset} + {1'b0, kpos_ic} + {10'b0, ic_cnt};
    wire [21:0] weight_addr_next_fast = cfg_weight_uram_base + weight_offset_fast;
""", "fast pos-major weight address wires", after=KPAR_BRANCH)

    # 1c. weight_rd_addr fast mux (kpar branch occurrence only).
    patch(AG, """                // Emit URAM weight read every active cycle.
                weight_rd_addr      <= weight_addr_next;
""", """                // Emit URAM weight read every active cycle.
                // [KPAR4-RN] fast layers use the pos-major transposed layout.
                weight_rd_addr      <= kpar_fast ? weight_addr_next_fast
                                                 : weight_addr_next;
""", "weight_rd_addr fast mux", after=KPAR_BRANCH)

    # 1d. fast ic-wrap advance (dense KxK position step).
    patch(AG, """                    end else if (!kpar_fast && (ic_cnt == (loop_ic - 12'd1))) begin   // [DW-ENGINE P1] loop_ic==1 in DW mode; [KPAR4] unreachable in fast walk (1x1: k_at_last fires first)
""", """                    end else if (kpar_fast && (ic_cnt == (loop_ic - 12'd4))) begin
                        // [KPAR4-RN] FAST ic-wrap: last 4-group of this
                        // (kh,kw) position -> advance kw/kh, k_cnt += 4.
                        // UNREACHABLE for 1x1 (there k_at_last fires on the
                        // same cycle and wins above) -> MBV2 fast walks are
                        // cycle-identical; only dense KxK (ResNet) takes it.
                        ic_cnt <= 12'd0;
                        if (kw_cnt == (cfg_kw - 3'd1)) begin
                            kw_cnt <= 3'd0;
                            kh_cnt <= kh_cnt + 3'd1;
                        end else begin
                            kw_cnt <= kw_cnt + 3'd1;
                        end
                        k_cnt <= k_cnt + 16'd4;
                    end else if (!kpar_fast && (ic_cnt == (loop_ic - 12'd1))) begin   // [DW-ENGINE P1] loop_ic==1 in DW mode
""", "fast ic-wrap advance branch")


# ============================================================================
# 2. nn2rtl_top.v — ResNet engine: K_PAR=4 + repacked banks + tap-major bus
# ============================================================================
def patch_top() -> None:
    # 2a. ENGINE_K_PAR + widened weight bus.
    patch(TOP, """    localparam integer ENGINE_LANE_B = 32 * ENGINE_WGT_W;    // real bits/bank (128|96)
    localparam integer ENGINE_WBUS_W = 8 * ENGINE_LANE_B;    // weight bus width (1024|768)
""", """    localparam integer ENGINE_LANE_B = 32 * ENGINE_WGT_W;    // real bits/bank (128|96)
    // [KPAR4-RN 2026-06-10] ENGINE K-PARALLEL P=4: the weight bus carries
    // ENGINE_K_PAR tap-major 768b words per (group-addressed) repacked bank
    // line (scripts/repack_resnet_kpar4_banks.py: 96b->384b lines, depth /4
    // = 16768, dense-3x3 regions transposed pos-major). All 17 dispatches
    // are fast-eligible (every base and IC %4==0 — proof P0 in the repack).
    localparam integer ENGINE_K_PAR  = 4;
    localparam integer ENGINE_WBUS_W = ENGINE_K_PAR * 8 * ENGINE_LANE_B;  // weight bus width (4 taps x 768)
""", "ENGINE_K_PAR localparam + bus width")

    # 2b. group addressing (engine exports old>>2; 16768 lines need 15 bits).
    patch(TOP, """    wire [16:0] weight_bank_rd_addr = engine_weight_rd_addr[16:0];
""", """    wire [14:0] weight_bank_rd_addr = engine_weight_rd_addr[14:0];  // [KPAR4-RN] GROUP address (engine exports old>>2; 16768 wide lines)
""", "group-addressed bank read")

    # 2c. widened per-bank data wires.
    old_wires = "".join(
        f"    wire [ENGINE_BANK_W-1:0] uram_bank{b}_rd_data;\n" for b in range(8))
    new_wires = "".join(
        f"    wire [ENGINE_K_PAR*ENGINE_BANK_W-1:0] uram_bank{b}_rd_data;  // [KPAR4-RN] 4 x 96b tap-major\n"
        for b in range(8))
    patch(TOP, old_wires, new_wires, "widen bank rd_data wires")

    # 2d. tap-major bus reconstruction.
    patch(TOP, """    // MAC bus = concat of the low ENGINE_LANE_B real bits of each bank (bank 0
    // lowest). INT4: low 128 of the 144-bit word; INT3: all 96.
    assign engine_weight_rd_data = {uram_bank7_rd_data[ENGINE_LANE_B-1:0],
        uram_bank6_rd_data[ENGINE_LANE_B-1:0],
        uram_bank5_rd_data[ENGINE_LANE_B-1:0],
        uram_bank4_rd_data[ENGINE_LANE_B-1:0],
        uram_bank3_rd_data[ENGINE_LANE_B-1:0],
        uram_bank2_rd_data[ENGINE_LANE_B-1:0],
        uram_bank1_rd_data[ENGINE_LANE_B-1:0],
        uram_bank0_rd_data[ENGINE_LANE_B-1:0]};
""", """    // [KPAR4-RN] MAC bus = ENGINE_K_PAR tap-major words. Each repacked bank
    // line carries 4 old words (tap j at [j*ENGINE_BANK_W +: ENGINE_BANK_W]);
    // engine tap-j word = concat of the low ENGINE_LANE_B real bits of each
    // bank's tap-j slice (bank 0 lowest — IDENTICAL lane order per tap).
    genvar kp_tap;
    generate for (kp_tap = 0; kp_tap < ENGINE_K_PAR; kp_tap = kp_tap + 1) begin : g_kpar_wbus
        assign engine_weight_rd_data[kp_tap*(8*ENGINE_LANE_B) +: 8*ENGINE_LANE_B] = {
            uram_bank7_rd_data[kp_tap*ENGINE_BANK_W +: ENGINE_LANE_B],
            uram_bank6_rd_data[kp_tap*ENGINE_BANK_W +: ENGINE_LANE_B],
            uram_bank5_rd_data[kp_tap*ENGINE_BANK_W +: ENGINE_LANE_B],
            uram_bank4_rd_data[kp_tap*ENGINE_BANK_W +: ENGINE_LANE_B],
            uram_bank3_rd_data[kp_tap*ENGINE_BANK_W +: ENGINE_LANE_B],
            uram_bank2_rd_data[kp_tap*ENGINE_BANK_W +: ENGINE_LANE_B],
            uram_bank1_rd_data[kp_tap*ENGINE_BANK_W +: ENGINE_LANE_B],
            uram_bank0_rd_data[kp_tap*ENGINE_BANK_W +: ENGINE_LANE_B]};
    end endgenerate
""", "tap-major weight bus")

    # 2e. the 8 bank instantiations -> repacked geometry + _kp4 files.
    for b in range(8):
        patch(TOP, f"""    uram_weight_bank #(
        .DEPTH(67072),
        .ADDR_W(17),
        .WORD_W(ENGINE_BANK_W),
        .MEM_INIT_FILE("output/weights/uram_weights_bank{b}.mem")
""", f"""    uram_weight_bank #(
        .DEPTH(16768),          // [KPAR4-RN] 67072/4 wide lines
        .ADDR_W(15),
        .WORD_W(ENGINE_K_PAR*ENGINE_BANK_W),   // [KPAR4-RN] 4 x 96b tap-major
        .MEM_INIT_FILE("output/weights/uram_weights_bank{b}_kp4.mem")
""", f"bank{b} repacked params")

    # 2f. K_PAR on the engine instantiation.
    patch(TOP, """        .WGT_W(ENGINE_WGT_W),
        .URAM_DATA_W(ENGINE_WBUS_W)
    ) u_shared_engine (
""", """        .WGT_W(ENGINE_WGT_W),
        .URAM_DATA_W(ENGINE_WBUS_W),
        // [KPAR4-RN] 4 taps/cycle/lane on ALL 17 dispatches (8 dense 1x1 +
        // 9 dense 3x3 via the pos-major transposed _kp4 banks).
        .K_PAR(ENGINE_K_PAR)
    ) u_shared_engine (
""", "engine K_PAR param")


def main() -> int:
    check = "--check" in sys.argv
    if check:
        for f in (AG, TOP):
            t = f.read_text(encoding="utf-8")
            print(f"{f.name}: KPAR4-RN markers = {t.count('[KPAR4-RN')}")
        return 0
    print("[kpar4-rn] patching address_generator.v (K_PAR>1 branch only) ...")
    patch_ag()
    print("[kpar4-rn] patching nn2rtl_top.v (ResNet engine) ...")
    patch_top()
    print("[kpar4-rn] done. Backups: *.prekp4r. Re-run is a no-op.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
