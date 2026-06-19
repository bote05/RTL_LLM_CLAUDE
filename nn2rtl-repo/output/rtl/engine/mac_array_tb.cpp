// mac_array_tb.cpp
// --------------------------------------------------------------------------
// Verilator unit testbench for output/rtl/engine/mac_array.v
//
// Verification protocol (matches docs/agent_tasks/07_engine_mac_array.md
// "How to verify" §2):
//
//   1. Pulse mac_clear for one cycle so all 256 lanes start at 0.
//   2. Drive N >= 100 random (act_byte, weight_bus) pairs with mac_valid_in
//      high. Each cycle:
//         act_byte         = uniform random signed INT8
//         weight_bus[lane] = uniform random signed INT8 for lane in [0..255]
//      and update a parallel C++ golden:
//         golden[lane] += int32(act_byte) * int32(weight_bus[lane]).
//   3. Drop mac_valid_in and tick the clock 3 more cycles so the stage-1 and
//      stage-2 pipeline registers drain into the accumulators.
//   4. Read acc_out[lane] for every lane and compare to golden[lane].
//      max_error MUST be 0.
//
// Build (from oss-cad-suite env or any verilator install):
//   verilator -Wall -cc --build --exe \
//             output/rtl/engine/mac_array.v \
//             output/rtl/engine/mac_array_tb.cpp \
//             -Mdir build_mac_array_tb \
//             --top-module mac_array
//   ./build_mac_array_tb/Vmac_array
// --------------------------------------------------------------------------

#include <verilated.h>
#include "Vmac_array.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kLanes    = 256;
constexpr int kSamples  = 128;       // >= 100 per spec; 128 is a clean power of two
constexpr uint32_t kSeed = 0xC0FFEEu;

vluint64_t main_time = 0;

void tick(Vmac_array* dut) {
    dut->clk = 0;
    dut->eval();
    main_time++;
    dut->clk = 1;
    dut->eval();
    main_time++;
}

void set_weight_lane(Vmac_array* dut, int lane, int8_t w) {
    // weight_bus is [2047:0]; Verilator surfaces it as an array of uint32_t
    // words. Lane `lane` occupies bits [lane*8 +: 8] = bits
    // [lane*8 .. lane*8+7]. Word index = (lane*8)/32 = lane/4. Bit within
    // word = (lane*8)%32 = (lane%4)*8.
    uint32_t* words = reinterpret_cast<uint32_t*>(&dut->weight_bus);
    int word_idx = lane / 4;
    int bit_off  = (lane % 4) * 8;
    words[word_idx] = (words[word_idx] & ~(uint32_t(0xFF) << bit_off)) |
                      ((uint32_t(uint8_t(w)) & 0xFF) << bit_off);
}

int32_t get_acc_lane(Vmac_array* dut, int lane) {
    // acc_out is [8191:0]. Each lane is a signed 32-bit value at bits
    // [lane*32 +: 32]. Verilator-flattened: word index = lane.
    uint32_t* words = reinterpret_cast<uint32_t*>(&dut->acc_out);
    return static_cast<int32_t>(words[lane]);
}

int8_t rand_int8(uint32_t& state) {
    // xorshift32 -> uniform 0..255 -> signed cast
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return static_cast<int8_t>(state & 0xFF);
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vmac_array* dut = new Vmac_array;

    // ---- Initialise inputs ----
    dut->clk          = 0;
    dut->rst_n        = 0;
    dut->mac_clear    = 0;
    dut->mac_valid_in = 0;
    dut->act_byte     = 0;
    std::memset(&dut->weight_bus, 0, sizeof(dut->weight_bus));
    dut->eval();

    // ---- Hold reset for a few cycles ----
    for (int i = 0; i < 4; ++i) tick(dut);
    dut->rst_n = 1;
    tick(dut);

    // ---- Single-cycle mac_clear pulse to be explicit about the start state ----
    dut->mac_clear = 1;
    tick(dut);
    dut->mac_clear = 0;
    tick(dut);

    // ---- Golden accumulators (computed in software as we drive the DUT) ----
    int64_t golden[kLanes];
    int8_t  weight_log[kSamples][kLanes];
    int8_t  act_log[kSamples];
    for (int lane = 0; lane < kLanes; ++lane) golden[lane] = 0;

    uint32_t rng_state = kSeed;

    // ---- Drive kSamples consecutive MAC steps ----
    for (int s = 0; s < kSamples; ++s) {
        int8_t a = rand_int8(rng_state);
        act_log[s] = a;
        dut->act_byte = static_cast<uint8_t>(a);

        for (int lane = 0; lane < kLanes; ++lane) {
            int8_t w = rand_int8(rng_state);
            weight_log[s][lane] = w;
            set_weight_lane(dut, lane, w);
            // Golden updates in lockstep with what the DUT will register on
            // the NEXT clock edge.
            golden[lane] += static_cast<int32_t>(a) * static_cast<int32_t>(w);
        }
        dut->mac_valid_in = 1;
        tick(dut);
    }

    // ---- Drop mac_valid_in and flush the 2-stage pipeline ----
    dut->mac_valid_in = 0;
    // The last sample's product lands in mul_q1 after 1 tick post-drop, then
    // into acc after 1 more tick. Two ticks is the minimum; three gives a
    // comfortable margin for the eval/commit ordering inside Verilator.
    for (int i = 0; i < 3; ++i) tick(dut);

    // ---- Check ----
    int    mismatches  = 0;
    int    max_error   = 0;
    int    first_bad   = -1;
    int32_t first_got  = 0;
    int32_t first_exp  = 0;
    for (int lane = 0; lane < kLanes; ++lane) {
        int32_t got = get_acc_lane(dut, lane);
        int32_t exp = static_cast<int32_t>(golden[lane]);
        int     err = std::abs(got - exp);
        if (err > max_error) max_error = err;
        if (err != 0) {
            if (first_bad < 0) {
                first_bad = lane;
                first_got = got;
                first_exp = exp;
            }
            mismatches++;
        }
    }

    printf("=== mac_array unit test ===\n");
    printf("samples         : %d\n", kSamples);
    printf("lanes           : %d\n", kLanes);
    printf("mismatches      : %d / %d\n", mismatches, kLanes);
    printf("max_error       : %d\n", max_error);
    if (mismatches > 0) {
        printf("first_bad lane  : %d  (got=%d, exp=%d)\n",
               first_bad, first_got, first_exp);
    }
    printf("STATUS          : %s\n", max_error == 0 ? "PASS" : "FAIL");

    delete dut;
    return max_error == 0 ? 0 : 1;
}
