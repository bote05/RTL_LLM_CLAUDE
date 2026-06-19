// bram_to_stream_bridge_tb.cpp
// --------------------------------------------------------------------------
// Verilator unit testbench for output/rtl/engine/bram_to_stream_bridge.v.
//
// The bridge's port list is locked by docs/agent_tasks/00_engine_skeleton_spec_PORTS.md
// `## SUBBLOCK: bram_to_stream_bridge`. The DUT has no `start` /
// `total_words` / `ready_out` / `done` ports — those described in the
// task-11 prose belong to a different (legacy) interface concept. This TB
// exercises the actual locked interface:
//
//   Phase 1 — read half, contiguous valid:
//     Drive 64 successive `act_in_rd_data` beats with `act_in_rd_data_valid=1`
//     every cycle and a varying `act_in_ic_byte_idx`. After each tick the
//     registered outputs (`mac_act_byte`, `mac_act_byte_valid`) must
//     reflect the byte selected from that cycle's input.
//
//   Phase 2 — read half, bursty valid:
//     Same 64 beats but with `act_in_rd_data_valid` pseudo-randomly low /
//     high. `mac_act_byte_valid` must mirror the sampled input each cycle —
//     no byte lost, no byte duplicated, order preserved.
//
//   Phase 3 — write half, bursty valid:
//     Drive 64 known `requant_data` beats with `requant_valid` cycling.
//     After each tick, `act_out_wr_en` mirrors the sampled `requant_valid`
//     and `act_out_wr_data` mirrors the sampled `requant_data`.
//
//   Phase 4 — bridge_busy span:
//     For every cycle k where `requant_valid[k]=1`, a downstream FSM must
//     see `bridge_busy` high both in cycle k (when the input is being
//     latched) and in cycle k+1 (when the registered `act_out_wr_en` is the
//     output pulse). bridge_busy is checked combinationally just before
//     the rising edge of each cycle, which is the moment a downstream FSM
//     would sample it.
//
// Build (Linux/WSL recommended; Windows MinGW also works with the
// w64devkit toolchain ahead in PATH):
//   verilator -Wall -cc --build --exe \
//             output/rtl/engine/bram_to_stream_bridge.v \
//             output/rtl/engine/bram_to_stream_bridge_tb.cpp \
//             -Mdir build_bram_to_stream_bridge_tb \
//             --top-module bram_to_stream_bridge
//   ./build_bram_to_stream_bridge_tb/Vbram_to_stream_bridge
// --------------------------------------------------------------------------

#include <verilated.h>
#include "Vbram_to_stream_bridge.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kBeats    = 64;
constexpr int kBusBits  = 2048;
constexpr int kBusBytes = kBusBits / 8;          // 256
constexpr int kBusWords = kBusBits / 32;         // 64

vluint64_t main_time = 0;

}  // namespace

// Verilator's runtime references sc_time_stamp() unconditionally on some
// builds; define it here so the harness links on any toolchain.
double sc_time_stamp() {
    return static_cast<double>(main_time);
}

namespace {

void tick(Vbram_to_stream_bridge* dut) {
    dut->clk = 0;
    dut->eval();
    main_time++;
    dut->clk = 1;
    dut->eval();
    main_time++;
}

uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

uint32_t* word_ptr(void* port) {
    return reinterpret_cast<uint32_t*>(port);
}

void set_beat_byte(uint32_t* words, int byte_idx, uint8_t val) {
    int word_idx = byte_idx / 4;
    int bit_off  = (byte_idx % 4) * 8;
    words[word_idx] =
        (words[word_idx] & ~(uint32_t(0xFF) << bit_off)) |
        ((uint32_t(val) & 0xFF) << bit_off);
}

uint8_t get_beat_byte(const uint32_t* words, int byte_idx) {
    int word_idx = byte_idx / 4;
    int bit_off  = (byte_idx % 4) * 8;
    return uint8_t((words[word_idx] >> bit_off) & 0xFF);
}

void zero_beat(uint32_t* words) {
    std::memset(words, 0, kBusBytes);
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vbram_to_stream_bridge* dut = new Vbram_to_stream_bridge;

    // ---- Initialise ----
    dut->clk                  = 0;
    dut->rst_n                = 0;
    dut->act_in_rd_data_valid = 0;
    dut->act_in_ic_byte_idx   = 0;
    dut->requant_valid        = 0;
    zero_beat(word_ptr(&dut->act_in_rd_data));
    zero_beat(word_ptr(&dut->requant_data));
    dut->eval();

    for (int i = 0; i < 4; ++i) tick(dut);
    dut->rst_n = 1;
    tick(dut);

    int reset_state_errors = 0;
    if (dut->mac_act_byte_valid != 0) reset_state_errors++;
    if (dut->mac_act_byte       != 0) reset_state_errors++;
    if (dut->act_out_wr_en      != 0) reset_state_errors++;
    if (dut->bridge_busy        != 0) reset_state_errors++;

    uint32_t rng = 0xC0FFEEu;

    // ------------------------------------------------------------------
    // Phase 1 — read half, contiguous valid.
    // ------------------------------------------------------------------
    std::vector<std::vector<uint8_t>> p1_beats(kBeats, std::vector<uint8_t>(kBusBytes, 0));
    std::vector<uint8_t> p1_idx(kBeats, 0);
    for (int b = 0; b < kBeats; ++b) {
        p1_idx[b] = uint8_t(xorshift32(rng) & 0xFF);
        for (int by = 0; by < kBusBytes; ++by) {
            p1_beats[b][by] = uint8_t(xorshift32(rng) & 0xFF);
        }
    }

    int p1_errors = 0;
    int p1_valid_pulses = 0;
    for (int b = 0; b < kBeats; ++b) {
        dut->act_in_rd_data_valid = 1;
        dut->act_in_ic_byte_idx   = p1_idx[b];
        uint32_t* beat_w = word_ptr(&dut->act_in_rd_data);
        zero_beat(beat_w);
        for (int by = 0; by < kBusBytes; ++by) {
            set_beat_byte(beat_w, by, p1_beats[b][by]);
        }
        tick(dut);  // rising edge samples beat[b] into registers

        uint8_t expected = p1_beats[b][p1_idx[b]];
        if (dut->mac_act_byte_valid != 1) {
            if (p1_errors < 4) {
                printf("[phase1] beat %d: mac_act_byte_valid=0 expected 1\n", b);
            }
            p1_errors++;
        } else {
            p1_valid_pulses++;
        }
        if (dut->mac_act_byte != expected) {
            if (p1_errors < 4) {
                printf("[phase1] beat %d: mac_act_byte=0x%02X expected 0x%02X (idx=%d)\n",
                       b, dut->mac_act_byte, expected, p1_idx[b]);
            }
            p1_errors++;
        }
    }
    // Deassert valid and confirm next tick drops mac_act_byte_valid.
    dut->act_in_rd_data_valid = 0;
    dut->act_in_ic_byte_idx   = 0;
    zero_beat(word_ptr(&dut->act_in_rd_data));
    tick(dut);
    if (dut->mac_act_byte_valid != 0) {
        printf("[phase1] tail: mac_act_byte_valid stuck high after drain\n");
        p1_errors++;
    }

    // ------------------------------------------------------------------
    // Phase 2 — read half, bursty valid.
    // ------------------------------------------------------------------
    constexpr int kP2Cycles = 128;
    std::vector<int>     p2_valid(kP2Cycles, 0);
    std::vector<uint8_t> p2_idx(kP2Cycles, 0);
    std::vector<std::vector<uint8_t>> p2_beats(kP2Cycles, std::vector<uint8_t>(kBusBytes, 0));
    int p2_beat_count = 0;
    for (int c = 0; c < kP2Cycles && p2_beat_count < kBeats; ++c) {
        p2_valid[c] = (xorshift32(rng) & 0x3) != 0;  // ~75% high
        if (p2_valid[c]) {
            p2_beat_count++;
            p2_idx[c] = uint8_t(xorshift32(rng) & 0xFF);
            for (int by = 0; by < kBusBytes; ++by) {
                p2_beats[c][by] = uint8_t(xorshift32(rng) & 0xFF);
            }
        }
    }

    int p2_errors = 0;
    int p2_valid_pulses = 0;
    for (int c = 0; c < kP2Cycles; ++c) {
        dut->act_in_rd_data_valid = p2_valid[c] ? 1 : 0;
        dut->act_in_ic_byte_idx   = p2_idx[c];
        uint32_t* beat_w = word_ptr(&dut->act_in_rd_data);
        zero_beat(beat_w);
        if (p2_valid[c]) {
            for (int by = 0; by < kBusBytes; ++by) {
                set_beat_byte(beat_w, by, p2_beats[c][by]);
            }
        }
        tick(dut);

        int exp_valid = p2_valid[c] ? 1 : 0;
        if (dut->mac_act_byte_valid != exp_valid) {
            if (p2_errors < 4) {
                printf("[phase2] cycle %d: mac_act_byte_valid=%d expected %d\n",
                       c, dut->mac_act_byte_valid, exp_valid);
            }
            p2_errors++;
        }
        if (exp_valid) {
            p2_valid_pulses++;
            uint8_t exp_byte = p2_beats[c][p2_idx[c]];
            if (dut->mac_act_byte != exp_byte) {
                if (p2_errors < 4) {
                    printf("[phase2] cycle %d: mac_act_byte=0x%02X expected 0x%02X (idx=%d)\n",
                           c, dut->mac_act_byte, exp_byte, p2_idx[c]);
                }
                p2_errors++;
            }
        }
    }

    // ------------------------------------------------------------------
    // Phase 3 + Phase 4 — write half + bridge_busy span.
    // ------------------------------------------------------------------
    constexpr int kP3Cycles = 128;
    std::vector<int> p3_valid(kP3Cycles, 0);
    std::vector<std::vector<uint8_t>> p3_beats(kP3Cycles, std::vector<uint8_t>(kBusBytes, 0));
    int p3_beat_count = 0;
    for (int c = 0; c < kP3Cycles && p3_beat_count < kBeats; ++c) {
        p3_valid[c] = (xorshift32(rng) & 0x3) != 0;
        if (p3_valid[c]) {
            p3_beat_count++;
            for (int by = 0; by < kBusBytes; ++by) {
                p3_beats[c][by] = uint8_t(xorshift32(rng) & 0xFF);
            }
        }
    }

    // First force inputs and DUT state to a clean idle so the first
    // bridge_busy sample is meaningful.
    dut->requant_valid = 0;
    zero_beat(word_ptr(&dut->requant_data));
    tick(dut);
    tick(dut);

    int p3_errors      = 0;
    int p3_wr_pulses   = 0;
    int p4_busy_errors = 0;

    // Drive the schedule. For each cycle k:
    //   1. Set inputs for cycle k. Call eval() — bridge_busy now reflects
    //      (current requant_valid) | (act_out_wr_en_register from cycle k-1).
    //      This is what a downstream FSM samples at the rising edge of k.
    //      Expected: requant_valid[k] | requant_valid[k-1].
    //   2. tick() — registers latch cycle k's inputs.
    //   3. Check registered outputs against cycle k's inputs.
    int prev_valid = 0;
    for (int c = 0; c < kP3Cycles + 1; ++c) {
        int cur_valid = (c < kP3Cycles) ? p3_valid[c] : 0;
        if (c < kP3Cycles) {
            dut->requant_valid = cur_valid ? 1 : 0;
            uint32_t* beat_w = word_ptr(&dut->requant_data);
            zero_beat(beat_w);
            if (cur_valid) {
                for (int by = 0; by < kBusBytes; ++by) {
                    set_beat_byte(beat_w, by, p3_beats[c][by]);
                }
            }
        } else {
            dut->requant_valid = 0;
            zero_beat(word_ptr(&dut->requant_data));
        }
        dut->eval();  // settle combinational logic with new inputs

        // Phase 4 — bridge_busy as seen by a downstream FSM at the rising
        // edge of cycle c.
        int expected_busy = (cur_valid | prev_valid) ? 1 : 0;
        if (dut->bridge_busy != expected_busy) {
            if (p4_busy_errors < 4) {
                printf("[phase4] cycle %d: bridge_busy=%d expected %d "
                       "(cur=%d prev=%d)\n",
                       c, dut->bridge_busy, expected_busy, cur_valid, prev_valid);
            }
            p4_busy_errors++;
        }

        tick(dut);

        // Phase 3 — registered write-half outputs reflect cycle c's inputs.
        int exp_en = cur_valid;
        if (dut->act_out_wr_en != exp_en) {
            if (p3_errors < 4) {
                printf("[phase3] cycle %d: act_out_wr_en=%d expected %d\n",
                       c, dut->act_out_wr_en, exp_en);
            }
            p3_errors++;
        }
        if (exp_en) {
            p3_wr_pulses++;
            uint32_t* got = word_ptr(&dut->act_out_wr_data);
            bool match = true;
            for (int by = 0; by < kBusBytes; ++by) {
                if (get_beat_byte(got, by) != p3_beats[c][by]) {
                    match = false;
                    break;
                }
            }
            if (!match) {
                if (p3_errors < 4) {
                    printf("[phase3] cycle %d: act_out_wr_data mismatch\n", c);
                }
                p3_errors++;
            }
        }

        prev_valid = cur_valid;
    }

    // Drain — after enough zero-input cycles bridge_busy must be 0.
    for (int i = 0; i < 2; ++i) tick(dut);
    dut->eval();
    if (dut->bridge_busy != 0) {
        printf("[phase4] tail: bridge_busy stuck high after drain\n");
        p4_busy_errors++;
    }

    // ---- Summary ----
    int total_errors = reset_state_errors + p1_errors + p2_errors + p3_errors + p4_busy_errors;
    printf("=== bram_to_stream_bridge unit test ===\n");
    printf("reset_state_errors : %d\n", reset_state_errors);
    printf("phase1 errors      : %d  (valid_pulses=%d/%d)\n",
           p1_errors, p1_valid_pulses, kBeats);
    printf("phase2 errors      : %d  (valid_pulses=%d/%d)\n",
           p2_errors, p2_valid_pulses, p2_beat_count);
    printf("phase3 errors      : %d  (wr_pulses=%d/%d)\n",
           p3_errors, p3_wr_pulses, p3_beat_count);
    printf("phase4 busy errors : %d\n", p4_busy_errors);
    printf("STATUS             : %s\n", total_errors == 0 ? "PASS" : "FAIL");

    delete dut;
    return total_errors == 0 ? 0 : 1;
}
