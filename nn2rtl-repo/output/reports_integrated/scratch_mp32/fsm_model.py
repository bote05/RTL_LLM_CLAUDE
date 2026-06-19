#!/usr/bin/env python3
"""Cycle-accurate Python model of rtl_library/conv_datapath_mp_k.v.

Goal: verify the datapath LOGIC at MP=32 vs MP=16:
  (a) byte-exact final per-OC data_out for a fixed window+weights, and
  (b) mac_busy deasserts and valid_out fires EXACTLY ONCE per frame (terminates).

This models the exact register pipeline of the RTL:
  - assign mac_busy = (state != ST_IDLE)                          (line 130)
  - Stage 1 (line 152-156): weight_word_q <= weights_wide[addr];
        tap_q[i] <= tap_at(k_group*MP_K + i)
  - Stage 2 combinational sum_lane_w (line 181-190): per-lane tree-sum of
        signed4 * signed8 products, 16-bit signed product context.
  - FSM block (line 192-323), all on posedge:
       partial_q <= sum_lane_w; mac_valid_q2 <= mac_valid_q1;
       mac_oc_group_q2 <= mac_oc_group_q1;
       if (mac_valid_q2) acc[lane] += partial_q[lane] (guarded oc<OC)
       case(state): IDLE/MAC/BIAS/SCALE/OUTPUT ...

Everything updates on the same posedge with NONBLOCKING semantics: we read the
CURRENT register values, compute next values, then commit all at once.
"""
from __future__ import annotations
import random

def sext(val, bits):
    """interpret low `bits` of val as 2's complement signed."""
    val &= (1 << bits) - 1
    if val & (1 << (bits - 1)):
        val -= (1 << bits)
    return val

def clog2(n):
    # Verilog $clog2: smallest k with 2^k >= n
    k = 0
    while (1 << k) < n:
        k += 1
    return k

class ConvDatapath:
    def __init__(self, IC, OC, KH, KW, MP, MP_K, weights_wide, biases, scale_rom):
        self.IC, self.OC, self.KH, self.KW = IC, OC, KH, KW
        self.MP, self.MP_K = MP, MP_K
        self.K_TOTAL = IC * KH * KW
        assert self.K_TOTAL % MP_K == 0, "K_TOTAL must be divisible by MP_K"
        self.K_GROUPS = self.K_TOTAL // MP_K
        self.OC_PASSES = (OC + MP - 1) // MP
        self.NUM_WIDE_WORDS = self.OC_PASSES * self.K_GROUPS
        # widths
        PROD_W = 16
        self.TREE_W = PROD_W + (0 if MP_K <= 1 else clog2(MP_K))
        self.ACC_W = self.TREE_W + (0 if self.K_GROUPS <= 1 else clog2(self.K_GROUPS))
        BIAS_W = 32
        self.BIASED_W = (max(self.ACC_W, BIAS_W)) + 1
        self.SCALE_CONST_W = 16
        self.SCALED_W = self.BIASED_W + self.SCALE_CONST_W
        # ROM: weights_wide is a list of length NUM_WIDE_WORDS, each an int holding
        # MP*MP_K nibbles (4-bit), nibble at (lane*MP_K+kpos)*4 +: 4 is weight[oc,k].
        self.weights_wide = weights_wide
        self.biases = biases      # list len OC, signed 32-bit
        self.scale_rom = scale_rom  # list len OC, 32-bit {shift[21:16], mult[15:0]}

        # ST codes
        self.ST_IDLE, self.ST_MAC, self.ST_BIAS, self.ST_SCALE, self.ST_OUTPUT = range(5)

    def tap_at(self, k_lin, window_flat):
        KH, KW, IC = self.KH, self.KW, self.IC
        kh_idx = (k_lin % (KH * KW)) // KW
        kw_idx = k_lin % KW
        ic_idx = k_lin // (KH * KW)
        flat_idx = kh_idx * KW * IC + kw_idx * IC + ic_idx
        return sext((window_flat >> (flat_idx * 8)) & 0xFF, 8)

    def run(self, window_flat, max_cycles=200000):
        MP, MP_K = self.MP, self.MP_K
        OC, MP_ = self.OC, MP
        # registers (post-reset state)
        state = self.ST_IDLE
        valid_out = 0
        data_out = [0] * OC          # per-OC byte (signed -128..127)
        k_group = 0
        oc_group = 0
        mac_valid_q1 = 0
        mac_valid_q2 = 0
        mac_oc_group_q1 = 0
        mac_oc_group_q2 = 0
        mac_done_issuing = 0
        acc = [0] * MP
        biased = [0] * MP
        scaled = [0] * MP
        partial_q = [0] * MP
        weight_word_q = 0
        tap_q = [0] * MP_K

        valid_out_count = 0
        valid_out_cycles = []
        busy_history = []

        # Drive start_mac for exactly one cycle (like the wrapper's start pulse
        # via sched_output_fires; here we model a single-frame start).
        # The wrapper asserts start_mac (sched_output_fires) once; ST_IDLE picks it up.
        started = False
        for cyc in range(max_cycles):
            start_mac = (cyc == 0)  # one-cycle start pulse at cycle 0

            mac_busy = 1 if state != self.ST_IDLE else 0
            busy_history.append(mac_busy)

            # ---- combinational Stage-2 sum_lane_w from CURRENT weight_word_q/tap_q ----
            sum_lane_w = [0] * MP
            for lane in range(MP):
                s = 0
                for kpos in range(MP_K):
                    w_nib = sext((weight_word_q >> ((lane * MP_K + kpos) * 4)) & 0xF, 4)
                    prod = sext((w_nib * tap_q[kpos]) & ((1 << 16) - 1), 16)  # PROD_W=16 signed
                    s += prod
                sum_lane_w[lane] = sext(s & ((1 << self.TREE_W) - 1), self.TREE_W)

            # ---- compute Stage-1 next (weight_word_q, tap_q) from CURRENT k_group/oc_group ----
            weight_read_addr = oc_group * self.K_GROUPS + k_group
            n_weight_word_q = self.weights_wide[weight_read_addr]
            n_tap_q = [self.tap_at(k_group * MP_K + i, window_flat) for i in range(MP_K)]

            # ---- next-state defaults (nonblocking: based on CURRENT regs) ----
            n_state = state
            n_valid_out = 0
            n_data_out = list(data_out)
            n_k_group = k_group
            n_oc_group = oc_group
            n_mac_valid_q1 = mac_valid_q1
            n_mac_valid_q2 = mac_valid_q1          # mac_valid_q2 <= mac_valid_q1
            n_mac_oc_group_q1 = mac_oc_group_q1
            n_mac_oc_group_q2 = mac_oc_group_q1    # <= mac_oc_group_q1
            n_mac_done_issuing = mac_done_issuing
            n_acc = list(acc)
            n_biased = list(biased)
            n_scaled = list(scaled)

            # Stage 2: partial_q <= sum_lane_w
            n_partial_q = list(sum_lane_w)

            # Stage 3: accumulate (uses CURRENT mac_valid_q2, partial_q, acc)
            if mac_valid_q2:
                for lane in range(MP):
                    if mac_oc_group_q2 * MP + lane < OC:
                        n_acc[lane] = sext((acc[lane] + partial_q[lane]) & ((1 << self.ACC_W) - 1), self.ACC_W)

            # FSM
            if state == self.ST_IDLE:
                if start_mac:
                    n_state = self.ST_MAC
                    n_k_group = 0
                    n_oc_group = 0
                    n_mac_valid_q1 = 0
                    n_mac_valid_q2 = 0   # NOTE: ST_IDLE explicitly clears q2 (overrides the <= mac_valid_q1 above)
                    n_mac_done_issuing = 0
                    for lane in range(MP):
                        n_acc[lane] = 0

            elif state == self.ST_MAC:
                if mac_done_issuing:
                    n_mac_valid_q1 = 0
                    # NB: this if reads the CURRENT mac_valid_q1/q2
                    if (not mac_valid_q1) and (not mac_valid_q2):
                        n_mac_done_issuing = 0
                        n_state = self.ST_BIAS
                else:
                    n_mac_oc_group_q1 = oc_group
                    n_mac_valid_q1 = 1
                    if k_group == self.K_GROUPS - 1:
                        n_mac_done_issuing = 1
                    else:
                        n_k_group = k_group + 1

            elif state == self.ST_BIAS:
                for lane in range(MP):
                    bias_oc = oc_group * MP + lane
                    if bias_oc < OC:
                        v = sext(acc[lane], self.ACC_W) + sext(self.biases[bias_oc] & 0xFFFFFFFF, 32)
                        n_biased[lane] = sext(v & ((1 << self.BIASED_W) - 1), self.BIASED_W)
                    else:
                        n_biased[lane] = 0
                n_state = self.ST_SCALE

            elif state == self.ST_SCALE:
                for lane in range(MP):
                    sc_oc = oc_group * MP + lane
                    if sc_oc < OC:
                        mult = self.scale_rom[sc_oc] & 0xFFFF  # [15:0], positive
                        v = sext(biased[lane], self.BIASED_W) * mult
                        n_scaled[lane] = sext(v & ((1 << self.SCALED_W) - 1), self.SCALED_W)
                    else:
                        n_scaled[lane] = 0
                n_state = self.ST_OUTPUT

            elif state == self.ST_OUTPUT:
                for lane in range(MP):
                    out_oc = oc_group * MP + lane
                    if out_oc < OC:
                        out_shift = (self.scale_rom[out_oc] >> 16) & 0x3F  # [21:16]
                        if out_shift == 0:
                            out_round = 0
                        else:
                            out_round = 1 << (out_shift - 1)
                        # arithmetic >>> on signed
                        v_tmp = (sext(scaled[lane], self.SCALED_W) + out_round) >> out_shift
                        if v_tmp > 127:
                            n_data_out[out_oc] = 127
                        elif v_tmp < -128:
                            n_data_out[out_oc] = -128
                        else:
                            n_data_out[out_oc] = sext(v_tmp & 0xFF, 8)
                if oc_group == self.OC_PASSES - 1:
                    n_valid_out = 1
                    n_state = self.ST_IDLE
                else:
                    n_oc_group = oc_group + 1
                    n_k_group = 0
                    for lane in range(MP):
                        n_acc[lane] = 0
                    n_state = self.ST_MAC
            else:
                n_state = self.ST_IDLE

            # ---- commit ----
            state = n_state
            valid_out = n_valid_out
            data_out = n_data_out
            k_group = n_k_group
            oc_group = n_oc_group
            mac_valid_q1 = n_mac_valid_q1
            mac_valid_q2 = n_mac_valid_q2
            mac_oc_group_q1 = n_mac_oc_group_q1
            mac_oc_group_q2 = n_mac_oc_group_q2
            mac_done_issuing = n_mac_done_issuing
            acc = n_acc
            biased = n_biased
            scaled = n_scaled
            partial_q = n_partial_q
            weight_word_q = n_weight_word_q
            tap_q = n_tap_q

            if valid_out:
                valid_out_count += 1
                valid_out_cycles.append(cyc)
                # snapshot data_out at the cycle valid_out is asserted
                final_out = list(data_out)
                # frame done: ensure we observe a return to IDLE (mac deassert).
                # Keep running a few cycles to confirm mac_busy drops & stays.

            # Terminate detection: once IDLE again after a frame, stop.
            if started and state == self.ST_IDLE and valid_out_count >= 1:
                # run 5 extra settle cycles then break
                pass
            if start_mac:
                started = True
            # break out shortly after first valid_out + return to idle
            if valid_out_count >= 1 and state == self.ST_IDLE and cyc > valid_out_cycles[-1] + 3:
                break
        else:
            return {"terminated": False, "valid_out_count": valid_out_count,
                    "valid_out_cycles": valid_out_cycles, "final_out": None,
                    "busy_history": busy_history}

        return {"terminated": True, "valid_out_count": valid_out_count,
                "valid_out_cycles": valid_out_cycles, "final_out": final_out,
                "busy_history": busy_history,
                "last_busy_after_done": busy_history[valid_out_cycles[-1]:]}


def pack_weights_wide(w_oc_k, OC, K_TOTAL, MP, MP_K):
    """w_oc_k[oc][k] in [-8,7]. Produce weights_wide[oc_group*K_GROUPS + k_group]
    as int, nibble at (lane*MP_K+kpos)*4. Mirrors repack_weights_wide layout."""
    K_GROUPS = K_TOTAL // MP_K
    OC_PASSES = (OC + MP - 1) // MP
    rom = []
    for oc_group in range(OC_PASSES):
        for k_group in range(K_GROUPS):
            word = 0
            for lane in range(MP):
                oc = oc_group * MP + lane
                for kpos in range(MP_K):
                    k = k_group * MP_K + kpos
                    if oc < OC:
                        nib = w_oc_k[oc][k] & 0xF
                    else:
                        nib = 0
                    word |= nib << ((lane * MP_K + kpos) * 4)
            rom.append(word)
    return rom


def golden_ref(window, w_oc_k, biases, scale_rom, OC, K_TOTAL):
    """Pure per-OC reference: acc = sum_k w[oc,k]*window[k]; +bias; *mult; round>>shift; sat."""
    out = []
    for oc in range(OC):
        acc = 0
        for k in range(K_TOTAL):
            acc += w_oc_k[oc][k] * window[k]
        biased = acc + biases[oc]
        mult = scale_rom[oc] & 0xFFFF
        shift = (scale_rom[oc] >> 16) & 0x3F
        scaled = biased * mult
        out_round = 0 if shift == 0 else (1 << (shift - 1))
        v = (scaled + out_round) >> shift
        out.append(127 if v > 127 else (-128 if v < -128 else sext(v & 0xFF, 8)))
    return out


def build_window_flat(window, IC, KH, KW):
    """window indexed by k_lin (ic-major: k = ic*(KH*KW)+kh*KW+kw). Build window_flat
    as the RTL expects: flat_idx = kh*KW*IC + kw*IC + ic, byte at flat_idx*8."""
    wf = 0
    K_TOTAL = IC * KH * KW
    for k in range(K_TOTAL):
        kh = (k % (KH * KW)) // KW
        kw = k % KW
        ic = k // (KH * KW)
        flat_idx = kh * KW * IC + kw * IC + ic
        wf |= (window[k] & 0xFF) << (flat_idx * 8)
    return wf


def run_case(IC, OC, KH, KW, MP_K, seed=1234, n_windows=8):
    random.seed(seed)
    K_TOTAL = IC * KH * KW
    # random INT4 weights, random scale rom (positive 15-bit mult, shift in valid range)
    w_oc_k = [[random.randint(-8, 7) for _ in range(K_TOTAL)] for _ in range(OC)]
    biases = [random.randint(-(1 << 20), (1 << 20) - 1) for _ in range(OC)]
    scale_rom = []
    for oc in range(OC):
        mult = random.randint(1, 32767)   # 15-bit positive
        shift = random.randint(0, 23)
        scale_rom.append(((shift & 0x3F) << 16) | (mult & 0xFFFF))

    # Generate the SAME window sequence used for BOTH MP values (critical: the
    # equivalence test requires identical inputs to MP=16 and MP=32).
    windows = [[random.randint(-128, 127) for _ in range(K_TOTAL)] for _ in range(n_windows)]

    results = {}
    for MP in (16, 32):
        rom = pack_weights_wide(w_oc_k, OC, K_TOTAL, MP, MP_K)
        dp = ConvDatapath(IC, OC, KH, KW, MP, MP_K, rom, biases, scale_rom)
        per_window = []
        for wi in range(n_windows):
            window = windows[wi]
            wf = build_window_flat(window, IC, KH, KW)
            r = dp.run(wf)
            ref = golden_ref(window, w_oc_k, biases, scale_rom, OC, K_TOTAL)
            per_window.append((r, ref, window))
        results[MP] = (dp, per_window)
    return results, w_oc_k, biases, scale_rom


def main():
    print("=== conv_datapath_mp_k FSM model: MP=16 vs MP=32 equivalence + termination ===\n")
    cases = [
        # (IC, OC, KH, KW, MP_K, label)
        (64, 64, 3, 3, 9, "conv_200-like 3x3 OC=64"),
        (64, 256, 1, 1, 8, "conv_202-like 1x1 OC=256 (the overproducer in the log)"),
        (64, 256, 1, 1, 8, "conv_204-like 1x1 OC=256"),
        (64, 64, 1, 1, 8, "conv_198-like 1x1 OC=64"),
    ]
    all_ok = True
    for (IC, OC, KH, KW, MP_K, label) in cases:
        K_TOTAL = IC * KH * KW
        results, _, _, _ = run_case(IC, OC, KH, KW, MP_K, seed=hash(label) & 0xFFFFFFFF, n_windows=6)
        dp16, w16 = results[16]
        dp32, w32 = results[32]
        print(f"--- {label}: IC={IC} OC={OC} K_TOTAL={K_TOTAL} MP_K={MP_K} "
              f"| OC_PASSES(MP16)={dp16.OC_PASSES} OC_PASSES(MP32)={dp32.OC_PASSES} ---")
        case_ok = True
        for wi in range(len(w16)):
            r16, ref16, win16 = w16[wi]
            r32, ref32, win32 = w32[wi]
            # termination checks
            t16, t32 = r16["terminated"], r32["terminated"]
            vc16, vc32 = r16["valid_out_count"], r32["valid_out_count"]
            # byte-exact: MP16 vs MP32 AND both vs pure reference
            eq_16_ref = (r16["final_out"] == ref16) if r16["final_out"] is not None else False
            eq_32_ref = (r32["final_out"] == ref32) if r32["final_out"] is not None else False
            eq_16_32 = (r16["final_out"] == r32["final_out"]) if (r16["final_out"] is not None and r32["final_out"] is not None) else False
            ok = t16 and t32 and vc16 == 1 and vc32 == 1 and eq_16_ref and eq_32_ref and eq_16_32
            case_ok = case_ok and ok
            if wi == 0 or not ok:
                print(f"   win{wi}: MP16[term={t16} vc={vc16} valid@={r16['valid_out_cycles']}] "
                      f"MP32[term={t32} vc={vc32} valid@={r32['valid_out_cycles']}] "
                      f"| MP16==ref:{eq_16_ref} MP32==ref:{eq_32_ref} MP16==MP32:{eq_16_32} -> {'OK' if ok else 'FAIL'}")
            if not ok and r32["final_out"] is not None:
                # show first mismatch
                for oc in range(OC):
                    if r32["final_out"][oc] != ref32[oc]:
                        print(f"      first OC mismatch @oc={oc}: MP32={r32['final_out'][oc]} ref={ref32[oc]} MP16={r16['final_out'][oc] if r16['final_out'] else None}")
                        break
        # mac_busy deassert check at MP32 (must drop to 0 and stay 0 after valid_out)
        r32_w0 = w32[0][0]
        if r32_w0["terminated"]:
            tail = r32_w0["last_busy_after_done"]
            busy_after = sum(tail[1:])  # cycles busy strictly after the valid_out cycle
            print(f"   MP32 mac_busy after valid_out (should -> 0): tail_busy_sum={busy_after} "
                  f"(0 = clean deassert)")
        print(f"   => {label}: {'ALL OK' if case_ok else 'FAILED'}\n")
        all_ok = all_ok and case_ok

    print("================================================================")
    print(f"OVERALL: {'DATAPATH LOGIC byte-exact AND terminating at MP=32' if all_ok else 'DATAPATH HAS A LOGIC/TERMINATION ISSUE AT MP=32'}")
    print("================================================================")


if __name__ == "__main__":
    main()
