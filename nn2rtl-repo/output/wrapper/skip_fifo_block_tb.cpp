// skip_fifo_block_tb.cpp
// --------------------------------------------------------------------------
// Verilator C++ driver for skip_fifo_block_dut.v (task 04 Phase B,
// revised for 04c — throttled producer).
//
// Usage (via plusargs):
//   +block=<add_id>             — label for the verdict line
//   +main=<cycles>              — main-path spatial latency (NO engine
//                                  term; 04c drops it from the analytical
//                                  formula). The engine occupancy is now
//                                  modelled by the throttle pulses below.
//   +skip=<cycles>              — skip-path latency
//   +depth=<words>              — FIFO depth under test
//   +nin=<count>                — number of input samples to drive
//   +budget=<cycles>            — hard cycle ceiling
//   +throttle_events=<k>        — number of engine_busy pulses to emit
//                                  during the run (one per engine-dispatched
//                                  layer in the main path)
//   +throttle_duration=<cycles> — duration of each pulse (engine
//                                  worst-case occupancy)
//   +throttle_period=<cycles>   — distance between successive pulse
//                                  starts (>= duration)
//   +goldin=<path>              — optional, real activation file (audit-
//                                  only; bytes are not consumed)
//
// Throttle schedule
// -----------------
// Pulse i starts at cycle `i * throttle_period` and lasts
// `throttle_duration` cycles. We emit exactly `throttle_events` pulses.
// This matches the deployment-level pattern: every engine-dispatched
// layer in the residual block produces one engine_busy pulse, and the
// spatial chain freezes for that duration.
//
// On exit prints exactly one line:
//   VERDICT block=<id> result=<status> peak=<n> cycles=<n>
//           outputs=<n> expected=<n> eff_cycles=<n> throttled=<n>
// --------------------------------------------------------------------------

#include <verilated.h>
#include "Vskip_fifo_block_dut.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

double sc_time_stamp() { return 0.0; }

namespace {

vluint64_t g_main_time = 0;

void tick(Vskip_fifo_block_dut* dut) {
    dut->clk = 0;
    dut->eval();
    g_main_time++;
    dut->clk = 1;
    dut->eval();
    g_main_time++;
}

uint32_t plus_u32(const char* key, uint32_t fallback) {
    std::string prefix = std::string(key) + "=";
    const char* raw = Verilated::commandArgsPlusMatch(prefix.c_str());
    if (!raw || !*raw) return fallback;
    const char* eq = std::strchr(raw, '=');
    if (!eq) return fallback;
    return static_cast<uint32_t>(std::strtoull(eq + 1, nullptr, 10));
}

const char* plus_str(const char* key, const char* fallback) {
    static thread_local std::string buffer;
    std::string prefix = std::string(key) + "=";
    const char* raw = Verilated::commandArgsPlusMatch(prefix.c_str());
    if (!raw || !*raw) return fallback;
    const char* eq = std::strchr(raw, '=');
    if (!eq) return fallback;
    buffer = std::string(eq + 1);
    return buffer.c_str();
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto* dut = new Vskip_fifo_block_dut;

    const char* block_id   = plus_str("block", "unknown_block");
    const uint32_t main_lat = plus_u32("main",   1000);
    const uint32_t skip_lat = plus_u32("skip",   0);
    const uint32_t depth    = plus_u32("depth",  1024);
    const uint32_t nin      = plus_u32("nin",    1024);
    const uint32_t thr_evt  = plus_u32("throttle_events",   0);
    const uint32_t thr_dur  = plus_u32("throttle_duration", 0);
    const uint32_t thr_per_ = plus_u32("throttle_period",
                                       thr_dur + 1024u);
    const uint32_t thr_per  = (thr_per_ < thr_dur + 1u) ? thr_dur + 1u : thr_per_;
    // Default budget must cover effective cycles (main + nin) PLUS the
    // total throttled time (events * duration) PLUS slack — otherwise
    // the run terminates as `cycle_budget_exhausted` even when the
    // FIFO behaviour is fine.
    const uint64_t throttle_total =
        static_cast<uint64_t>(thr_evt) * static_cast<uint64_t>(thr_dur);
    const uint64_t default_budget =
        static_cast<uint64_t>(main_lat) +
        static_cast<uint64_t>(nin) +
        throttle_total +
        4096ull;
    const uint32_t budget = plus_u32("budget",
        default_budget > 0xFFFFFFFFull ? 0xFFFFFFFFu
                                       : static_cast<uint32_t>(default_budget));
    const char* goldin_path = plus_str("goldin", "");

    dut->clk           = 0;
    dut->rst_n         = 0;
    dut->start         = 0;
    dut->throttle      = 0;
    dut->main_latency  = main_lat;
    dut->skip_latency  = skip_lat;
    dut->fifo_depth    = depth;
    dut->num_inputs    = nin;
    dut->cycle_budget  = budget;
    dut->eval();

    for (int i = 0; i < 4; ++i) tick(dut);
    dut->rst_n = 1;
    tick(dut);

    dut->start = 1;
    tick(dut);
    dut->start = 0;

    if (goldin_path && *goldin_path) {
        FILE* fp = std::fopen(goldin_path, "rb");
        if (fp) {
            std::fseek(fp, 0, SEEK_END);
            long sz = std::ftell(fp);
            std::fclose(fp);
            std::fprintf(stderr,
                         "[skip_fifo_block_tb] block=%s goldin=%s bytes=%ld "
                         "(timing-only, bytes not consumed)\n",
                         block_id, goldin_path, sz);
        }
    }

    // Drive throttle from the C++ side. We use the DUT's wall-clock
    // `cycles_run` register to decide which pulse window we are in.
    const uint64_t hard_cap = static_cast<uint64_t>(budget) * 10ull + 1024ull;
    uint64_t ticks = 0;
    while (!dut->done && ticks < hard_cap) {
        uint32_t now = dut->cycles_run;
        // Compute throttle activity for the current cycle.
        uint8_t throttle = 0;
        if (thr_evt > 0 && thr_dur > 0) {
            uint32_t period = thr_per;
            uint32_t pulse_idx = now / period;
            if (pulse_idx < thr_evt) {
                uint32_t phase = now - pulse_idx * period;
                if (phase < thr_dur) throttle = 1;
            }
        }
        dut->throttle = throttle;
        tick(dut);
        ticks++;
    }

    const uint32_t peak     = dut->peak_fifo_occupancy;
    const uint32_t outs     = dut->outputs_produced;
    const uint32_t cycles   = dut->cycles_run;
    const uint32_t effcyc   = dut->eff_cycles_run;
    const uint32_t thrcyc   = dut->throttled_cycles;
    const uint32_t bpcyc    = dut->backpressure_cycles;
    const bool overflow     = dut->overflow_detected != 0;
    const bool deadlock     = dut->deadlock_detected != 0;
    const uint32_t dl_cycle = dut->deadlock_cycle;

    char status[64];
    if (deadlock) {
        std::snprintf(status, sizeof(status),
                      "deadlock_at_cycle_%u", dl_cycle);
    } else if (overflow) {
        std::snprintf(status, sizeof(status), "overflow");
    } else if (outs < nin) {
        std::snprintf(status, sizeof(status), "cycle_budget_exhausted");
    } else {
        std::snprintf(status, sizeof(status), "no_deadlock_no_overflow");
    }

    std::printf("VERDICT block=%s result=%s peak=%u cycles=%u "
                "outputs=%u expected=%u eff_cycles=%u throttled=%u "
                "backpressure=%u\n",
                block_id, status, peak, cycles, outs, nin, effcyc, thrcyc,
                bpcyc);

    delete dut;
    return 0;
}
