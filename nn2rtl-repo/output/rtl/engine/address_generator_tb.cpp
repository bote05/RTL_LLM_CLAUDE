// address_generator_tb.cpp
// --------------------------------------------------------------------------
// Verilator unit testbench for output/rtl/engine/address_generator.v.
//
// Spec: docs/agent_tasks/09_engine_address_generator.md "How to verify".
// Locked port list: docs/agent_tasks/00_engine_skeleton_spec_PORTS.md
//                   `## SUBBLOCK: address_generator`.
//
// Verification protocol (one binary, four scenarios):
//
//   1. node_conv_298 full-walk: IC=512 OC=512 KH=KW=3 SH=SW=1 PH=PW=1
//      IH=IW=OH=OW=7. Walk every (oc_pass, oh, ow) tuple and confirm the
//      emitted (weight_rd_addr, weight_rd_en, act_in_rd_addr, act_in_rd_en,
//      act_in_ic_byte_idx, bias_rd_addr, bias_rd_en, act_out_wr_addr,
//      k_index) sequence is bit-exact against the Python-style golden walk
//      computed in C++. mac_done must pulse OH*OW*OC_PASSES = 98 times;
//      pixel_done must pulse exactly once (latched high after the final
//      mac_done).
//
//   2. Padding bounds: dispatch node_conv_196 (KH=KW=7, PH=PW=3) for a
//      corner output pixel and confirm act_in_rd_en goes low at every
//      receptive-field position outside [0,IH) x [0,IW).
//
//   3. Multi-layer dispatch: run three different configs back-to-back
//      (conv_298 -> conv_220 -> conv_196), one inner-loop each, and check
//      that addresses for each layer match its own golden — counters reset
//      cleanly across dispatches.
//
//   4. layer_done count: in scenario 1 confirm exactly one pixel_done
//      rising edge across the full walk.
//
// Build (from oss-cad-suite env or any verilator install):
//   verilator -Wall -cc --build --exe \
//             output/rtl/engine/address_generator.v \
//             output/rtl/engine/address_generator_tb.cpp \
//             -Mdir build_address_generator_tb \
//             --top-module address_generator
//   ./build_address_generator_tb/Vaddress_generator
//
// Exits 0 on success; non-zero (and prints the first mismatch) on failure.
// --------------------------------------------------------------------------

#include <verilated.h>
#include "Vaddress_generator.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Verilator's linker looks for this stub; the testbench does not emit
// waves, so a constant timestamp is fine.
double sc_time_stamp() { return 0.0; }

namespace {

struct LayerCfg {
    const char* name;
    int ic;
    int oc;
    int kh;
    int kw;
    int ih;
    int iw;
    int oh;
    int ow;
    int sh;
    int sw;
    int ph;
    int pw;
    uint32_t weight_base;
    uint32_t bias_base;
    uint16_t act_in_base;
    uint16_t act_out_base;
};

const LayerCfg kConv298 = {
    "node_conv_298", 512, 512, 3, 3, 7, 7, 7, 7, 1, 1, 1, 1,
    556881u, 600000u, 0u, 32768u,
};
const LayerCfg kConv220 = {
    // 128x128 3x3 stride 2 pad 1, IH=IW=56, OH=OW=28.
    "node_conv_220", 128, 128, 3, 3, 56, 56, 28, 28, 2, 2, 1, 1,
    7095u, 8000u, 0u, 16384u,
};
const LayerCfg kConv196 = {
    // 3x64 7x7 stride 2 pad 3, IH=IW=224, OH=OW=112. Used for padding test.
    "node_conv_196", 3, 64, 7, 7, 224, 224, 112, 112, 2, 2, 3, 3,
    0u, 1000u, 0u, 0u,
};

vluint64_t g_main_time = 0;

void tick(Vaddress_generator* dut) {
    dut->clk = 0;
    dut->eval();
    g_main_time++;
    dut->clk = 1;
    dut->eval();
    g_main_time++;
}

void apply_cfg(Vaddress_generator* dut, const LayerCfg& c) {
    dut->cfg_ic = c.ic;
    dut->cfg_oc = c.oc;
    dut->cfg_kh = c.kh;
    dut->cfg_kw = c.kw;
    dut->cfg_ih = c.ih;
    dut->cfg_iw = c.iw;
    dut->cfg_oh = c.oh;
    dut->cfg_ow = c.ow;
    dut->cfg_stride_h = c.sh;
    dut->cfg_stride_w = c.sw;
    dut->cfg_pad_h    = c.ph;
    dut->cfg_pad_w    = c.pw;
    dut->cfg_weight_uram_base  = c.weight_base;
    dut->cfg_bias_uram_base    = c.bias_base;
    dut->cfg_act_in_bram_base  = c.act_in_base;
    dut->cfg_act_out_bram_base = c.act_out_base;
}

void reset(Vaddress_generator* dut) {
    dut->clk = 0;
    dut->rst_n = 0;
    dut->run_active = 0;
    dut->oc_pass_idx = 0;
    dut->pixel_h = 0;
    dut->pixel_w = 0;
    apply_cfg(dut, kConv298);
    dut->eval();
    for (int i = 0; i < 4; ++i) tick(dut);
    dut->rst_n = 1;
    tick(dut);
}

int oc_passes(const LayerCfg& c) {
    return (c.oc + 255) / 256;
}

// Python-style golden: returns the (weight_addr, act_in_addr, act_in_en,
// ic_byte_idx, k_index, act_out_addr) tuple for one (oc_pass, oh, ow, kh,
// kw, ic) step. The same arithmetic as the locked algorithm in the spec.
struct GoldenStep {
    uint32_t weight_addr;
    uint16_t act_in_addr;
    bool     act_in_en;
    uint8_t  ic_byte_idx;
    uint16_t k_index;
    uint16_t act_out_addr;
};

GoldenStep golden_step(const LayerCfg& c, int oc_pass, int oh, int ow,
                       int kh, int kw, int ic) {
    GoldenStep g;
    int in_r = oh * c.sh + kh - c.ph;
    int in_c = ow * c.sw + kw - c.pw;
    bool ib_r = (in_r >= 0) && (in_r < c.ih);
    bool ib_c = (in_c >= 0) && (in_c < c.iw);
    g.act_in_en   = ib_r && ib_c;
    // 13a audit fix (Fix B): the engine address_generator now strides
    // act_in_addr by ic_chunks for IC > 256, mirroring the wrapper's
    // unified BRAM layout (one BRAM word = 256 channels = one MAC pass).
    // The previous TB golden used the legacy formula
    //   base + (in_r*IW + in_c)
    // which is correct only for IC <= 256 (ic_chunks == 1). The new
    // formula
    //   base + (in_r*IW + in_c)*ic_chunks + ic_chunk_idx
    // matches address_generator.v line 175. For ic_chunks==1 the new
    // expression collapses to the old one.
    int ic_chunks = (c.ic + 255) / 256;
    if (ic_chunks < 1) ic_chunks = 1;
    int ic_chunk_idx = ic / 256;
    int pixel_word_idx = in_r * c.iw + in_c;
    g.act_in_addr = static_cast<uint16_t>(
        c.act_in_base + (ib_r && ib_c
                         ? (pixel_word_idx * ic_chunks + ic_chunk_idx)
                         : 0));
    g.weight_addr = c.weight_base
                    + static_cast<uint32_t>(oc_pass) * c.ic * c.kh * c.kw
                    + static_cast<uint32_t>(ic) * c.kh * c.kw
                    + static_cast<uint32_t>(kh) * c.kw
                    + static_cast<uint32_t>(kw);
    g.ic_byte_idx = static_cast<uint8_t>(ic & 0xFF);
    int k = kh * c.kw * c.ic + kw * c.ic + ic;
    g.k_index = static_cast<uint16_t>(k);
    int passes = oc_passes(c);
    g.act_out_addr = static_cast<uint16_t>(
        c.act_out_base + (oh * c.ow + ow) * passes + oc_pass);
    return g;
}

// Run the address generator for one (oc_pass, oh, ow). Sample the emitted
// outputs every cycle. Returns: vector of length K_TOTAL of (weight_addr,
// act_in_addr, act_in_en, ic_byte_idx, k_index, act_out_addr) plus whether
// mac_done pulsed at the end. The bias_rd_en cycle is checked separately.
struct InnerWalk {
    std::vector<GoldenStep> emitted;
    bool   mac_done_pulsed;
    bool   bias_en_pulsed_once;
    uint32_t bias_addr_seen;
};

InnerWalk run_inner_walk(Vaddress_generator* dut, const LayerCfg& c,
                         int oc_pass, int oh, int ow) {
    InnerWalk r;
    r.mac_done_pulsed     = false;
    r.bias_en_pulsed_once = false;
    r.bias_addr_seen      = 0;

    const int k_total = c.ic * c.kh * c.kw;

    dut->oc_pass_idx = oc_pass;
    dut->pixel_h     = oh;
    dut->pixel_w     = ow;
    dut->run_active  = 1;

    int bias_en_count = 0;

    // Cycle 0 .. K_TOTAL-1: inner-loop emit cycles.
    for (int k = 0; k < k_total; ++k) {
        tick(dut);
        GoldenStep g;
        g.weight_addr  = dut->weight_rd_addr;
        g.act_in_addr  = dut->act_in_rd_addr;
        g.act_in_en    = (dut->act_in_rd_en != 0);
        g.ic_byte_idx  = dut->act_in_ic_byte_idx;
        g.k_index      = dut->k_index;
        g.act_out_addr = dut->act_out_wr_addr;
        r.emitted.push_back(g);

        if (dut->bias_rd_en) {
            bias_en_count++;
            r.bias_addr_seen = dut->bias_rd_addr;
        }
    }
    // One extra cycle to capture mac_done (mac_done is registered, pulses on
    // the cycle AFTER the last emit cycle's k_at_last detection).
    if (dut->mac_done) r.mac_done_pulsed = true;
    r.bias_en_pulsed_once = (bias_en_count == 1);

    // Deassert run_active for one cycle to mimic the FSM's REQUANT/DRAIN
    // gap before the next OC pass.
    dut->run_active = 0;
    tick(dut);

    return r;
}

bool diff_step(const GoldenStep& g, const GoldenStep& a,
               const char* tag, int step_idx, const LayerCfg& c,
               int oc_pass, int oh, int ow, int kh, int kw, int ic) {
    bool ok = true;
    auto fail = [&](const char* field, uint64_t exp, uint64_t got) {
        ok = false;
        printf("FAIL[%s] %s pixel=(%d,%d) oc_pass=%d k=%d kh=%d kw=%d ic=%d "
               "field=%s exp=0x%llx got=0x%llx\n",
               tag, c.name, oh, ow, oc_pass, step_idx, kh, kw, ic, field,
               static_cast<unsigned long long>(exp),
               static_cast<unsigned long long>(got));
    };
    if (a.weight_addr  != g.weight_addr)  fail("weight_addr",  g.weight_addr,  a.weight_addr);
    if (a.act_in_en    != g.act_in_en)    fail("act_in_en",    g.act_in_en,    a.act_in_en);
    if (a.ic_byte_idx  != g.ic_byte_idx)  fail("ic_byte_idx",  g.ic_byte_idx,  a.ic_byte_idx);
    if (a.k_index      != g.k_index)      fail("k_index",      g.k_index,      a.k_index);
    if (a.act_out_addr != g.act_out_addr) fail("act_out_addr", g.act_out_addr, a.act_out_addr);
    // act_in_addr is only meaningful when act_in_en is true.
    if (g.act_in_en && a.act_in_addr != g.act_in_addr) {
        fail("act_in_addr", g.act_in_addr, a.act_in_addr);
    }
    return ok;
}

int scenario_full_walk(Vaddress_generator* dut) {
    printf("[scenario 1] conv_298 full-walk (OH*OW*OC_PASSES = 7*7*2 = 98 inner loops)\n");
    apply_cfg(dut, kConv298);
    // Sit in load_config for a cycle so cfg_* are sampled by the
    // combinational logic before we assert run_active.
    dut->run_active = 0;
    tick(dut);

    int mac_done_count = 0;
    int pixel_done_rises = 0;
    int prev_pixel_done = 0;
    bool any_mismatch = false;

    const int passes = oc_passes(kConv298);

    for (int oc_pass = 0; oc_pass < passes; ++oc_pass) {
        for (int oh = 0; oh < kConv298.oh; ++oh) {
            for (int ow = 0; ow < kConv298.ow; ++ow) {
                InnerWalk w = run_inner_walk(dut, kConv298, oc_pass, oh, ow);

                // Compare every emit against the golden.
                int idx = 0;
                for (int kh = 0; kh < kConv298.kh && !any_mismatch; ++kh)
                for (int kw = 0; kw < kConv298.kw && !any_mismatch; ++kw)
                for (int ic = 0; ic < kConv298.ic && !any_mismatch; ++ic, ++idx) {
                    GoldenStep g = golden_step(kConv298, oc_pass, oh, ow, kh, kw, ic);
                    if (!diff_step(g, w.emitted[idx], "full_walk", idx,
                                   kConv298, oc_pass, oh, ow, kh, kw, ic)) {
                        any_mismatch = true;
                    }
                }
                if (any_mismatch) return 1;

                if (w.mac_done_pulsed) mac_done_count++;
                else {
                    printf("FAIL[full_walk] mac_done did not pulse at pixel=(%d,%d) oc_pass=%d\n",
                           oh, ow, oc_pass);
                    return 1;
                }

                // Bias must pulse exactly once per OC pass start (which is the
                // start of every inner walk in this scenario because we drop
                // run_active between).
                if (!w.bias_en_pulsed_once) {
                    printf("FAIL[full_walk] bias_rd_en pulse count != 1 at pixel=(%d,%d) oc_pass=%d\n",
                           oh, ow, oc_pass);
                    return 1;
                }
                uint32_t exp_bias = kConv298.bias_base + oc_pass;
                if (w.bias_addr_seen != exp_bias) {
                    printf("FAIL[full_walk] bias_rd_addr exp=0x%x got=0x%x\n",
                           exp_bias, w.bias_addr_seen);
                    return 1;
                }

                // Sample pixel_done after the gap cycle.
                if (dut->pixel_done && !prev_pixel_done) pixel_done_rises++;
                prev_pixel_done = dut->pixel_done;
            }
        }
    }

    if (mac_done_count != passes * kConv298.oh * kConv298.ow) {
        printf("FAIL[full_walk] mac_done pulses exp=%d got=%d\n",
               passes * kConv298.oh * kConv298.ow, mac_done_count);
        return 1;
    }
    if (pixel_done_rises != 1) {
        printf("FAIL[full_walk] pixel_done rising edges exp=1 got=%d\n",
               pixel_done_rises);
        return 1;
    }
    printf("  OK: %d mac_done pulses, 1 pixel_done rising edge\n", mac_done_count);
    return 0;
}

int scenario_padding(Vaddress_generator* dut) {
    printf("[scenario 2] conv_196 padding bounds at output pixel (0,0)\n");
    // Reset state so pixel_done_latch from scenario 1 is cleared.
    reset(dut);
    apply_cfg(dut, kConv196);
    dut->run_active = 0;
    tick(dut);

    InnerWalk w = run_inner_walk(dut, kConv196, /*oc_pass=*/0, /*oh=*/0, /*ow=*/0);
    int padded_observed = 0;
    int padded_expected = 0;
    int idx = 0;
    for (int kh = 0; kh < kConv196.kh; ++kh)
    for (int kw = 0; kw < kConv196.kw; ++kw)
    for (int ic = 0; ic < kConv196.ic; ++ic, ++idx) {
        GoldenStep g = golden_step(kConv196, 0, 0, 0, kh, kw, ic);
        if (!g.act_in_en) padded_expected++;
        if (!w.emitted[idx].act_in_en) padded_observed++;
        if (!diff_step(g, w.emitted[idx], "padding", idx, kConv196,
                       0, 0, 0, kh, kw, ic)) {
            return 1;
        }
    }
    if (padded_observed != padded_expected) {
        printf("FAIL[padding] padded-position count exp=%d got=%d\n",
               padded_expected, padded_observed);
        return 1;
    }
    if (padded_expected == 0) {
        printf("FAIL[padding] expected nonzero padded count for corner (0,0)\n");
        return 1;
    }
    printf("  OK: %d padded positions correctly flagged with act_in_rd_en=0\n",
           padded_observed);
    return 0;
}

int scenario_multi_layer(Vaddress_generator* dut) {
    printf("[scenario 3] multi-layer dispatch: conv_298 -> conv_220 -> conv_196\n");
    reset(dut);

    const LayerCfg* layers[] = { &kConv298, &kConv220, &kConv196 };
    for (int li = 0; li < 3; ++li) {
        const LayerCfg& c = *layers[li];
        apply_cfg(dut, c);
        dut->run_active = 0;
        tick(dut);

        // Run one inner loop at pixel (1,1) so receptive field straddles a
        // mix of in-bounds and (for conv_196) some borderline positions.
        InnerWalk w = run_inner_walk(dut, c, /*oc_pass=*/0, /*oh=*/1, /*ow=*/1);
        int idx = 0;
        bool any_mismatch = false;
        for (int kh = 0; kh < c.kh && !any_mismatch; ++kh)
        for (int kw = 0; kw < c.kw && !any_mismatch; ++kw)
        for (int ic = 0; ic < c.ic && !any_mismatch; ++ic, ++idx) {
            GoldenStep g = golden_step(c, 0, 1, 1, kh, kw, ic);
            if (!diff_step(g, w.emitted[idx], "multi_layer", idx, c,
                           0, 1, 1, kh, kw, ic)) {
                any_mismatch = true;
            }
        }
        if (any_mismatch) return 1;
        if (!w.mac_done_pulsed) {
            printf("FAIL[multi_layer] mac_done did not pulse for %s\n", c.name);
            return 1;
        }
        printf("  OK: %s emit sequence (K_TOTAL=%d) bit-exact\n",
               c.name, c.ic * c.kh * c.kw);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto* dut = new Vaddress_generator;

    reset(dut);

    int rc = 0;
    if (rc == 0) rc = scenario_full_walk(dut);
    if (rc == 0) rc = scenario_padding(dut);
    if (rc == 0) rc = scenario_multi_layer(dut);

    delete dut;
    if (rc != 0) {
        printf("address_generator_tb: FAIL\n");
        return rc;
    }
    printf("address_generator_tb: PASS\n");
    return 0;
}
