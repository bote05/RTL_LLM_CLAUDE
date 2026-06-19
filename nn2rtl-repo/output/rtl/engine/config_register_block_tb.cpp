// config_register_block_tb.cpp
// --------------------------------------------------------------------------
// Verilator scoreboard unit testbench for output/rtl/engine/config_register_block.v
//
// Verification protocol (docs/agent_tasks/10_engine_config_register_block.md
// §"How to verify"):
//
//   1. AXI4-Lite scoreboard: for every writable register, drive a known
//      32-bit pattern over the AXI write channels, then read it back over
//      the AXI read channels. Expected == written.
//   2. Field decode: confirm the narrower cfg_* outputs match the
//      bit-slices specified in the register-map comment in the RTL.
//   3. engine_start_pulse with engine_busy=0 fires for exactly 1 cycle on
//      the cycle the AXI write to CONTROL.bit[0] commits.
//   4. engine_start_pulse with engine_busy=1 does NOT fire (busy gate).
//
// Build (from oss-cad-suite env or any verilator install):
//   verilator -Wall -cc --build --exe \
//             output/rtl/engine/config_register_block.v \
//             output/rtl/engine/config_register_block_tb.cpp \
//             -Mdir build_config_register_block_tb \
//             --top-module config_register_block
//   ./build_config_register_block_tb/Vconfig_register_block
// --------------------------------------------------------------------------

#include <verilated.h>
#include "Vconfig_register_block.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

vluint64_t main_time = 0;

double sc_time_stamp() {
    return static_cast<double>(main_time);
}

namespace {

void tick(Vconfig_register_block* dut) {
    dut->clk = 0;
    dut->eval();
    main_time++;
    dut->clk = 1;
    dut->eval();
    main_time++;
}

int fail_count = 0;
int check_count = 0;

#define CHECK_EQ(label, got, exp)                                           \
    do {                                                                    \
        ++check_count;                                                      \
        uint64_t _g = static_cast<uint64_t>(got);                           \
        uint64_t _e = static_cast<uint64_t>(exp);                           \
        if (_g != _e) {                                                     \
            ++fail_count;                                                   \
            printf("FAIL %s: got 0x%llx, expected 0x%llx\n", label,         \
                   (unsigned long long)_g, (unsigned long long)_e);         \
        }                                                                   \
    } while (0)

// Drive a complete AXI4-Lite write transaction. Leaves bus signals
// deasserted on return. Always holds bready high so the response handshake
// closes in the second cycle of the transaction.
void axi_write(Vconfig_register_block* dut, uint8_t addr, uint32_t data) {
    dut->s_axil_bready  = 1;
    dut->s_axil_awvalid = 1;
    dut->s_axil_wvalid  = 1;
    dut->s_axil_awaddr  = addr;
    dut->s_axil_wdata   = data;
    dut->s_axil_wstrb   = 0xF;
    dut->eval();

    // Sanity: awready and wready must be combinationally high since no
    // prior bvalid is outstanding (helper is only called from idle).
    if (!dut->s_axil_awready || !dut->s_axil_wready) {
        ++fail_count;
        printf("FAIL pre-write handshake: awready=%d wready=%d at addr 0x%02X\n",
               dut->s_axil_awready, dut->s_axil_wready, addr);
    }

    // Cycle the clock: handshake commits, register write happens, bvalid_r
    // becomes 1.
    tick(dut);

    // After commit: deassert the address/data valids before the next tick
    // so we do not start a second transaction.
    dut->s_axil_awvalid = 0;
    dut->s_axil_wvalid  = 0;
    dut->eval();

    if (!dut->s_axil_bvalid) {
        ++fail_count;
        printf("FAIL post-write bvalid not asserted at addr 0x%02X\n", addr);
    }

    // bready was held high, so bvalid_r clears on this tick.
    tick(dut);
    dut->s_axil_bready = 0;
    dut->eval();
}

uint32_t axi_read(Vconfig_register_block* dut, uint8_t addr) {
    dut->s_axil_rready  = 1;
    dut->s_axil_arvalid = 1;
    dut->s_axil_araddr  = addr;
    dut->eval();

    if (!dut->s_axil_arready) {
        ++fail_count;
        printf("FAIL pre-read arready not asserted at addr 0x%02X\n", addr);
    }

    tick(dut);  // capture, set rvalid_r

    dut->s_axil_arvalid = 0;
    dut->eval();

    if (!dut->s_axil_rvalid) {
        ++fail_count;
        printf("FAIL post-read rvalid not asserted at addr 0x%02X\n", addr);
    }
    uint32_t value = dut->s_axil_rdata;

    tick(dut);  // rvalid clears
    dut->s_axil_rready = 0;
    dut->eval();
    return value;
}

void reset(Vconfig_register_block* dut) {
    dut->rst_n             = 0;
    dut->s_axil_awvalid    = 0;
    dut->s_axil_wvalid     = 0;
    dut->s_axil_bready     = 0;
    dut->s_axil_arvalid    = 0;
    dut->s_axil_rready     = 0;
    dut->s_axil_awaddr     = 0;
    dut->s_axil_wdata      = 0;
    dut->s_axil_wstrb      = 0;
    dut->s_axil_araddr     = 0;
    dut->engine_start_ext  = 0;
    dut->engine_busy_in    = 0;
    dut->engine_done_in    = 0;
    for (int i = 0; i < 4; ++i) tick(dut);
    dut->rst_n = 1;
    tick(dut);
}

struct RegCase {
    const char* name;
    uint8_t     addr;
    uint32_t    pattern;
};

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vconfig_register_block* dut = new Vconfig_register_block;

    reset(dut);

    // -----------------------------------------------------------------
    // 1. Scoreboard: write 13 writable registers with distinct patterns,
    //    read each back, compare.
    // -----------------------------------------------------------------
    const RegCase cases[] = {
        {"INPUT_CHANNELS",     0x00, 0xDEADBEEFu},
        {"OUTPUT_CHANNELS",    0x04, 0x12345678u},
        {"KERNEL_H_W",         0x08, 0x00000033u},  // kh=3, kw=3 in low byte
        {"STRIDE_H_W",         0x0C, 0x00000012u},  // sh=2, sw=2
        {"PADDING_H_W",        0x10, 0x00000009u},  // ph=1, pw=1
        {"INPUT_H_W",          0x14, 0x00E000E0u},  // ih=0xE0 (224), iw=0xE0
        {"OUTPUT_H_W",         0x18, 0x00700070u},  // oh=0x70 (112), ow=0x70
        {"WEIGHT_BASE_WORD",   0x1C, 0x000F1234u},
        {"BIAS_BASE_WORD",     0x20, 0x00012345u},
        {"SCALE_MULT",         0x24, 0xABCDEF01u},
        {"SCALE_SHIFT_AND_ZP", 0x28, 0x000003C5u},  // zp=0x0F, shift=0x05
        {"ACT_IN_BASE",        0x34, 0x00001234u},
        {"ACT_OUT_BASE",       0x38, 0x00005678u},
    };
    constexpr int kNumRegs = sizeof(cases) / sizeof(cases[0]);

    // Phase 1a: write all
    for (int i = 0; i < kNumRegs; ++i) {
        axi_write(dut, cases[i].addr, cases[i].pattern);
    }
    // Phase 1b: read all back
    for (int i = 0; i < kNumRegs; ++i) {
        uint32_t got = axi_read(dut, cases[i].addr);
        char lbl[64];
        std::snprintf(lbl, sizeof(lbl), "round-trip %s @0x%02X",
                      cases[i].name, cases[i].addr);
        CHECK_EQ(lbl, got, cases[i].pattern);
    }

    // -----------------------------------------------------------------
    // 2. Field-decode check on cfg_* outputs.
    // -----------------------------------------------------------------
    CHECK_EQ("cfg_ic",                dut->cfg_ic,                0xEEFu);
    CHECK_EQ("cfg_oc",                dut->cfg_oc,                0x678u);
    CHECK_EQ("cfg_kh",                dut->cfg_kh,                3u);
    CHECK_EQ("cfg_kw",                dut->cfg_kw,                3u);
    CHECK_EQ("cfg_stride_h",          dut->cfg_stride_h,          2u);
    CHECK_EQ("cfg_stride_w",          dut->cfg_stride_w,          2u);
    CHECK_EQ("cfg_pad_h",             dut->cfg_pad_h,             1u);
    CHECK_EQ("cfg_pad_w",             dut->cfg_pad_w,             1u);
    CHECK_EQ("cfg_ih",                dut->cfg_ih,                0xE0u);
    CHECK_EQ("cfg_iw",                dut->cfg_iw,                0xE0u);
    CHECK_EQ("cfg_oh",                dut->cfg_oh,                0x70u);
    CHECK_EQ("cfg_ow",                dut->cfg_ow,                0x70u);
    CHECK_EQ("cfg_weight_uram_base",  dut->cfg_weight_uram_base,  0x0F1234u);
    CHECK_EQ("cfg_bias_uram_base",    dut->cfg_bias_uram_base,    0x012345u);
    // 13a audit fix: cfg_scale_mult was widened from 16 to 32 bits in
    // config_register_block.v (line ~96 / 177) so the test expectation
    // must mirror the full 32-bit value written at offset 0x24, not just
    // the low 16 bits.
    CHECK_EQ("cfg_scale_mult",        dut->cfg_scale_mult,        0xABCDEF01u);
    CHECK_EQ("cfg_scale_shift",       dut->cfg_scale_shift,       0x05u);
    CHECK_EQ("cfg_act_in_bram_base",  dut->cfg_act_in_bram_base,  0x1234u);
    CHECK_EQ("cfg_act_out_bram_base", dut->cfg_act_out_bram_base, 0x5678u);

    // -----------------------------------------------------------------
    // 3. STATUS / CONTROL read decode.
    //    STATUS @0x30: bit[0]=done, bit[1]=busy.
    //    CONTROL @0x2C read returns busy mirror in bit[1].
    // -----------------------------------------------------------------
    dut->engine_busy_in = 1;
    dut->engine_done_in = 1;
    dut->eval();
    {
        uint32_t status = axi_read(dut, 0x30);
        CHECK_EQ("STATUS read busy", (status >> 1) & 1u, 1u);
        CHECK_EQ("STATUS read done", status & 1u,        1u);
    }
    dut->engine_busy_in = 1;
    dut->engine_done_in = 0;
    dut->eval();
    {
        uint32_t ctrl = axi_read(dut, 0x2C);
        CHECK_EQ("CONTROL read busy bit",   (ctrl >> 1) & 1u, 1u);
        CHECK_EQ("CONTROL read start bit",  ctrl & 1u,        0u);
    }
    dut->engine_busy_in = 0;
    dut->engine_done_in = 0;
    dut->eval();

    // -----------------------------------------------------------------
    // 4. engine_start_pulse fires for exactly 1 cycle on AXI START write
    //    while engine_busy_in is low.
    // -----------------------------------------------------------------
    {
        // Pre-condition: pulse must be low before the write.
        CHECK_EQ("pulse low pre-write", dut->engine_start_pulse, 0u);

        // Drive the AXI write to 0x2C with START=1, engine_busy=0.
        dut->engine_busy_in = 0;
        dut->s_axil_bready  = 1;
        dut->s_axil_awvalid = 1;
        dut->s_axil_wvalid  = 1;
        dut->s_axil_awaddr  = 0x2C;
        dut->s_axil_wdata   = 0x00000001;
        dut->s_axil_wstrb   = 0xF;
        dut->eval();
        if (!dut->s_axil_awready || !dut->s_axil_wready) {
            ++fail_count;
            printf("FAIL START write: handshake not ready\n");
        }

        tick(dut);  // commit: start_trigger latched, pulse rises
        dut->s_axil_awvalid = 0;
        dut->s_axil_wvalid  = 0;
        dut->eval();
        CHECK_EQ("pulse high after commit", dut->engine_start_pulse, 1u);

        tick(dut);  // pulse default-low fires; bvalid clears
        dut->eval();
        CHECK_EQ("pulse low one cycle later", dut->engine_start_pulse, 0u);

        tick(dut);
        dut->eval();
        CHECK_EQ("pulse stays low", dut->engine_start_pulse, 0u);

        dut->s_axil_bready = 0;
    }

    // -----------------------------------------------------------------
    // 5. engine_busy=1 inhibits engine_start_pulse (busy gate).
    // -----------------------------------------------------------------
    {
        dut->engine_busy_in = 1;
        dut->s_axil_bready  = 1;
        dut->s_axil_awvalid = 1;
        dut->s_axil_wvalid  = 1;
        dut->s_axil_awaddr  = 0x2C;
        dut->s_axil_wdata   = 0x00000001;
        dut->s_axil_wstrb   = 0xF;
        dut->eval();

        tick(dut);  // commit, but busy gate inhibits pulse
        dut->s_axil_awvalid = 0;
        dut->s_axil_wvalid  = 0;
        dut->eval();
        CHECK_EQ("busy gate: no pulse after commit", dut->engine_start_pulse, 0u);

        tick(dut);
        dut->eval();
        CHECK_EQ("busy gate: pulse still low", dut->engine_start_pulse, 0u);

        // Even with several follow-up cycles, no pulse should appear.
        for (int k = 0; k < 4; ++k) {
            tick(dut);
            if (dut->engine_start_pulse) {
                ++fail_count;
                printf("FAIL busy gate: spurious pulse at extra cycle %d\n", k);
            }
        }
        dut->s_axil_bready  = 0;
        dut->engine_busy_in = 0;
    }

    // -----------------------------------------------------------------
    // 6. engine_start_ext pin path also pulses (busy=0) and is gated
    //    (busy=1).
    // -----------------------------------------------------------------
    {
        // Pin path, busy=0 -> pulse expected one cycle later.
        dut->engine_busy_in   = 0;
        dut->engine_start_ext = 1;
        dut->eval();
        tick(dut);
        dut->engine_start_ext = 0;
        dut->eval();
        CHECK_EQ("ext pin: pulse high", dut->engine_start_pulse, 1u);

        tick(dut);
        dut->eval();
        CHECK_EQ("ext pin: pulse low after one cycle",
                 dut->engine_start_pulse, 0u);

        // Pin path, busy=1 -> no pulse.
        dut->engine_busy_in   = 1;
        dut->engine_start_ext = 1;
        dut->eval();
        tick(dut);
        dut->engine_start_ext = 0;
        dut->eval();
        CHECK_EQ("ext pin busy-gated: no pulse", dut->engine_start_pulse, 0u);
        dut->engine_busy_in = 0;
    }

    // -----------------------------------------------------------------
    // 7. engine_busy_ext / engine_done_ext are combinational pass-through
    //    of their _in counterparts.
    // -----------------------------------------------------------------
    dut->engine_busy_in = 1;
    dut->engine_done_in = 0;
    dut->eval();
    CHECK_EQ("busy_ext pass-through (1)", dut->engine_busy_ext, 1u);
    CHECK_EQ("done_ext pass-through (0)", dut->engine_done_ext, 0u);
    dut->engine_busy_in = 0;
    dut->engine_done_in = 1;
    dut->eval();
    CHECK_EQ("busy_ext pass-through (0)", dut->engine_busy_ext, 0u);
    CHECK_EQ("done_ext pass-through (1)", dut->engine_done_ext, 1u);

    // -----------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------
    printf("=== config_register_block unit test ===\n");
    printf("checks          : %d\n", check_count);
    printf("failures        : %d\n", fail_count);
    printf("STATUS          : %s\n", fail_count == 0 ? "PASS" : "FAIL");

    delete dut;
    return fail_count == 0 ? 0 : 1;
}
