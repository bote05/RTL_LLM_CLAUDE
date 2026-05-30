// Cycle-accurate Verilator testbench for nn2rtl_top.v
//
// Goal: measure end-to-end cycles for one ResNet-50 inference through the
// integrated top. We don't check output correctness here (per-module
// Verilator + engine TB already cover that); the cycle count is the
// throughput number for the thesis PPA.
//
// Strategy:
//   - Drive clk
//   - Hold reset for N cycles, deassert
//   - Tie m_axis_tready = 1 (always accept output)
//   - Tie all AXI-Lite signals idle (the scheduler kicks off on the first
//     accepted s_axis beat per [nn2rtl_top.v Fix 6])
//   - Stream 50,176 beats of 256-bit zeros into s_axis (= 224x224 ImageNet
//     input frame at one pixel per beat, channels packed in low 24 bits,
//     rest don't-care). Assert s_axis_tlast on the final beat.
//   - Count from first accepted input beat to last accepted output beat
//     (m_axis_tlast & m_axis_tvalid & m_axis_tready)
//   - Print cycles
//
// Build:
//   verilator_bin.exe --cc --exe -O3 --threads 4 \
//     -CFLAGS "-O2 -std=c++17" \
//     --top-module nn2rtl_top \
//     --Mdir build_verilator_nn2rtl_top \
//     output/rtl/nn2rtl_top.v output/rtl/engine/*.v \
//     output/rtl/shared_engine_skeleton.v output/rtl/nn2rtl_scheduler.v \
//     rtl_library/*.v output/rtl/node_*.v \
//     tb/nn2rtl_top_cycle_count_tb.cpp
//   cd build_verilator_nn2rtl_top && make -j -f Vnn2rtl_top.mk Vnn2rtl_top
//   ./Vnn2rtl_top

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "Vnn2rtl_top.h"
#include "verilated.h"
#ifdef NN2RTL_TRACE_FST
#  include "verilated_vcd_c.h"
#endif

double sc_time_stamp() { return 0; }

namespace {

#ifdef NN2RTL_TRACE_FST
constexpr uint64_t kMaxCycles      = 5'000'000;    // shorter run for trace
#else
constexpr uint64_t kMaxCycles      = 50'000'000;   // debug iteration: working frame ~30M, fails fast on deadlock
#endif
constexpr int      kResetCycles    = 16;
constexpr uint64_t kInputBeats     = 224 * 224;    // 50,176 beats of 256-bit
constexpr uint64_t kOutputBeats    = 3136;         // m_axis_beat_count goes 0..3135

#ifdef NN2RTL_TRACE_FST
VerilatedVcdC* g_tfp = nullptr;
uint64_t       g_sim_time = 0;
void tick(Vnn2rtl_top* dut, uint64_t& cycle) {
  dut->clk = 1;
  dut->eval();
  if (g_tfp) g_tfp->dump(g_sim_time++);
  dut->clk = 0;
  dut->eval();
  if (g_tfp) g_tfp->dump(g_sim_time++);
  cycle++;
}
#else
void tick(Vnn2rtl_top* dut, uint64_t& cycle) {
  // Rising edge
  dut->clk = 1;
  dut->eval();
  // Falling edge
  dut->clk = 0;
  dut->eval();
  cycle++;
}
#endif

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
#ifdef NN2RTL_TRACE_FST
  Verilated::traceEverOn(true);
#else
  Verilated::traceEverOn(false);
#endif
  Vnn2rtl_top* dut = new Vnn2rtl_top;

#ifdef NN2RTL_TRACE_FST
  g_tfp = new VerilatedVcdC;
  dut->trace(g_tfp, 99);  // 99 = max depth
  const char* trace_path = std::getenv("NN2RTL_TRACE_PATH");
  if (!trace_path) trace_path = "nn2rtl_top_trace.vcd";
  g_tfp->open(trace_path);
  std::printf("[tb] FST trace -> %s (max %llu cycles)\n", trace_path,
              (unsigned long long)kMaxCycles);
#endif

  // Initial idle values
  dut->clk           = 0;
  dut->rst_n         = 0;
  dut->s_axis_tvalid = 0;
  dut->s_axis_tlast  = 0;
  // s_axis_tdata is a [255:0] which Verilator may pack as VlWide.
  // Zero-initialize whatever shape Verilator chose.
  memset(&dut->s_axis_tdata, 0, sizeof(dut->s_axis_tdata));
  dut->m_axis_tready    = 1;  // always accept output
  dut->s_axil_awvalid   = 0;
  dut->s_axil_awaddr    = 0;
  dut->s_axil_wvalid    = 0;
  dut->s_axil_wdata     = 0;
  dut->s_axil_wstrb     = 0;
  dut->s_axil_bready    = 0;
  dut->s_axil_arvalid   = 0;
  dut->s_axil_araddr    = 0;
  dut->s_axil_rready    = 0;
  dut->eval();

  uint64_t cycle = 0;

  // ---- reset ----
  for (int i = 0; i < kResetCycles; i++) tick(dut, cycle);
  dut->rst_n = 1;
  tick(dut, cycle);

  std::printf("[tb] reset deasserted at cycle %llu\n",
              (unsigned long long)cycle);

  // ---- stream input frame ----
  uint64_t input_beats_sent  = 0;
  uint64_t first_input_cycle = 0;
  bool     first_input_seen  = false;

  // Output bookkeeping
  uint64_t output_beats_seen = 0;
  uint64_t last_output_cycle = 0;
  bool     done              = false;

  while (cycle < kMaxCycles && !done) {
    // ---- drive input side ----
    if (input_beats_sent < kInputBeats) {
      dut->s_axis_tvalid = 1;
      dut->s_axis_tlast  = (input_beats_sent + 1 == kInputBeats) ? 1 : 0;
    } else {
      dut->s_axis_tvalid = 0;
      dut->s_axis_tlast  = 0;
    }
    dut->eval();

    // Capture handshake BEFORE the tick (combinational ready/valid)
    bool input_accept  = dut->s_axis_tvalid && dut->s_axis_tready;
    bool output_accept = dut->m_axis_tvalid && dut->m_axis_tready;
    bool output_last   = output_accept && dut->m_axis_tlast;

    if (input_accept) {
      if (!first_input_seen) {
        first_input_seen  = true;
        first_input_cycle = cycle;
        std::printf("[tb] first input beat accepted at cycle %llu\n",
                    (unsigned long long)cycle);
        std::fflush(stdout);
      }
      input_beats_sent++;
    }
    if (output_accept) {
      if (output_beats_seen == 0) {
        std::printf("[tb] FIRST OUTPUT BEAT at cycle %llu\n", (unsigned long long)cycle);
        std::fflush(stdout);
      }
      output_beats_seen++;
      // Periodically log output progress so we see how far the chain got
      if ((output_beats_seen % 100) == 0) {
        std::printf("[tb] output_beats=%llu/%llu at cycle %llu\n",
                    (unsigned long long)output_beats_seen,
                    (unsigned long long)kOutputBeats,
                    (unsigned long long)cycle);
        std::fflush(stdout);
      }
      if (output_last) {
        last_output_cycle = cycle;
        done              = true;
      }
    }

    // Periodic status to confirm liveness, flushed so we see progress live.
    if ((cycle & 0xFFFFF) == 0 && cycle > 0) {
      std::printf("[tb] cycle=%llu in=%llu/%llu out=%llu/%llu "
                  "s_axis_tready=%d m_axis_tvalid=%d\n",
                  (unsigned long long)cycle,
                  (unsigned long long)input_beats_sent,
                  (unsigned long long)kInputBeats,
                  (unsigned long long)output_beats_seen,
                  (unsigned long long)kOutputBeats,
                  (int)dut->s_axis_tready,
                  (int)dut->m_axis_tvalid);
      std::fflush(stdout);
    }

    tick(dut, cycle);
  }

  // ---- report ----
  // For trace mode, "done" just means trace duration reached without output.
  // For non-trace mode, "done" means full frame emitted.
  if (done) {
    std::printf("[tb] DONE: first_input_cycle=%llu last_output_cycle=%llu\n",
                (unsigned long long)first_input_cycle,
                (unsigned long long)last_output_cycle);
    std::printf("[tb] e2e cycles (first input accept -> last output beat) = %llu\n",
                (unsigned long long)(last_output_cycle - first_input_cycle));
    std::printf("[tb] total input beats   = %llu (expected %llu)\n",
                (unsigned long long)input_beats_sent,
                (unsigned long long)kInputBeats);
    std::printf("[tb] total output beats  = %llu (expected %llu)\n",
                (unsigned long long)output_beats_seen,
                (unsigned long long)kOutputBeats);
    std::printf("[tb][summary] e2e_cycles=%llu input_beats=%llu output_beats=%llu\n",
                (unsigned long long)(last_output_cycle - first_input_cycle),
                (unsigned long long)input_beats_sent,
                (unsigned long long)output_beats_seen);
  } else {
    std::printf("[tb] TIMEOUT at cycle %llu — only %llu input + %llu output beats seen\n",
                (unsigned long long)cycle,
                (unsigned long long)input_beats_sent,
                (unsigned long long)output_beats_seen);
    delete dut;
    return 1;
  }

#ifdef NN2RTL_TRACE_FST
  if (g_tfp) { g_tfp->close(); delete g_tfp; }
#endif
  dut->final();   // run Verilog `final` blocks (FIFO peak-occupancy audit prints)
  delete dut;
  return 0;
}
