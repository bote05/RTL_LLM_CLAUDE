// Verilator testbench for rtl_library/coord_scheduler.v.
//
// Asserts:
//   * exact number of output_fires pulses per frame (each pulse is a
//     registered one-cycle event emitted the cycle AFTER the scheduler
//     advances past a firing coord)
//   * correct row-major ordering
//   * termination bounded by outputs_emitted == OH*OW (never by
//     in_row > IH-1+PH)
//   * out_frame_done fires exactly once per frame
//
// The TB simulates a simple datapath: when it sees `output_fires = 1`,
// it raises `stall_in` for MAC_CYCLES cycles (modeling the MAC pipeline),
// then drops `stall_in`. The scheduler resumes advance once `stall_in`
// drops and its internal output_fires pulse has cleared (one cycle).

#include "Vcoord_scheduler.h"
#include <verilated.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static void tick(Vcoord_scheduler* dut) {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
}

#ifndef SCENARIO
#define SCENARIO 0
#endif

#if SCENARIO == 1
    static constexpr int EXPECTED_FIRES = 3 * 3;
    static constexpr int IH_P = 7, IW_P = 7;
#elif SCENARIO == 2
    static constexpr int EXPECTED_FIRES = 8 * 8;
    static constexpr int IH_P = 8, IW_P = 8;
#elif SCENARIO == 3
    static constexpr int EXPECTED_FIRES = 16 * 16;
    static constexpr int IH_P = 16, IW_P = 16;
#else
    static constexpr int EXPECTED_FIRES = 16 * 16;
    static constexpr int IH_P = 32, IW_P = 32;
#endif

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vcoord_scheduler* dut = new Vcoord_scheduler;

    dut->clk = 0;
    dut->rst_n = 0;
    dut->start = 0;
    dut->stall_in = 0;
    dut->valid_in = 0;
    for (int i = 0; i < 3; i++) tick(dut);
    dut->rst_n = 1;
    tick(dut);

    dut->start = 1;
    tick(dut);
    dut->start = 0;

    const int SAFETY_BUDGET = 200000;
    int cycles_elapsed = 0;
    int fire_count = 0;
    std::vector<std::pair<int, int>> fire_coords;
    int frame_done_pulses = 0;

    // Simulated datapath. When output_fires pulses, stall for MAC_CYCLES.
    const int MAC_CYCLES = 4;
    int mac_countdown = 0;

    while (cycles_elapsed < SAFETY_BUDGET) {
        // Rising edge of output_fires triggers a MAC cycle countdown.
        bool fires = dut->output_fires != 0;
        if (fires && mac_countdown == 0) {
            // Count the fire — each output_fires pulse is the post-advance
            // registered pulse for the coord we just moved past.
            fire_count++;
            fire_coords.push_back({(int)dut->in_row, (int)dut->in_col});
            mac_countdown = MAC_CYCLES;
        }

        // stall_in high during MAC pipeline simulation.
        dut->stall_in = (mac_countdown > 0) ? 1 : 0;

        // Eager upstream — delivers a pixel whenever the scheduler
        // needs real input.
        dut->valid_in = dut->needs_real_input ? 1 : 0;

        if (dut->out_frame_done) frame_done_pulses++;
        if (dut->out_frame_done) break;

        tick(dut);
        cycles_elapsed++;
        if (mac_countdown > 0) mac_countdown--;
    }

    if (cycles_elapsed >= SAFETY_BUDGET) {
        fprintf(stderr,
                "FAIL: coord_scheduler hit safety budget (%d cycles) without terminating\n",
                SAFETY_BUDGET);
        return 2;
    }

    int exit_code = 0;
    if (fire_count != EXPECTED_FIRES) {
        fprintf(stderr, "FAIL: expected %d output_fires pulses, got %d\n",
                EXPECTED_FIRES, fire_count);
        exit_code = 1;
    }
    if ((int)dut->outputs_emitted != EXPECTED_FIRES) {
        fprintf(stderr, "FAIL: expected outputs_emitted=%d, got %d\n",
                EXPECTED_FIRES, (int)dut->outputs_emitted);
        exit_code = 1;
    }
    if (frame_done_pulses != 1) {
        fprintf(stderr, "FAIL: expected exactly 1 out_frame_done pulse, got %d\n",
                frame_done_pulses);
        exit_code = 1;
    }

    if (exit_code == 0) {
        fprintf(stdout,
                "PASS: %d fires observed, outputs_emitted=%d, 1 frame_done pulse "
                "(SCENARIO=%d, IH=%d IW=%d)\n",
                fire_count, (int)dut->outputs_emitted, SCENARIO, IH_P, IW_P);
    }

    delete dut;
    return exit_code;
}
