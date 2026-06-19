// Verilator harness: drive skip_fifo + bram_fifo in lockstep with randomized
// backpressure; assert the popped data SEQUENCE is identical and matches the
// increasing producer payload. Proves bram_fifo is FWFT-equivalent to skip_fifo
// (only fill latency differs).
#include "Vfifo_equiv_top.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cstdlib>

static Vfifo_equiv_top* dut;
static vluint64_t main_time = 0;
double sc_time_stamp() { return (double)main_time; }

static uint32_t lfsr_a = 0x1234'5678u, lfsr_b = 0x9abc'def0u;
static uint32_t nxt(uint32_t s){ return (s>>1) ^ (-(int32_t)(s&1) & 0xD0000001u); }

static void tick() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vfifo_equiv_top;

    const int NBEATS = 5000;
    dut->rst_n = 0; dut->feed_valid_s=0; dut->feed_valid_b=0; dut->feed_data=0;
    dut->s_out_ready=0; dut->b_out_ready=0;
    for (int i=0;i<6;i++) tick();
    dut->rst_n = 1;

    std::vector<uint32_t> sq, bq;
    int fed=0; bool feeding=false; bool s_acc=false, b_acc=false; uint32_t cur=0;

    for (int cyc=0; cyc<400000; cyc++) {
        // present a new beat until both have accepted
        if (!feeding && fed < NBEATS) { cur = (uint32_t)fed; feeding=true; s_acc=false; b_acc=false; }
        dut->feed_data = cur;
        dut->feed_valid_s = (feeding && !s_acc) ? 1 : 0;
        dut->feed_valid_b = (feeding && !b_acc) ? 1 : 0;
        // randomized output backpressure (vary phases between the two)
        dut->s_out_ready = (lfsr_b & 1) ? 1 : 0;
        dut->b_out_ready = ((lfsr_b>>3) & 1) ? 1 : 0;

        // sample combinational readies/valids BEFORE the edge commits
        bool s_push = dut->feed_valid_s && dut->s_in_ready;
        bool b_push = dut->feed_valid_b && dut->b_in_ready;
        bool s_pop  = dut->s_out_valid && dut->s_out_ready;
        bool b_pop  = dut->b_out_valid && dut->b_out_ready;
        uint32_t s_d = dut->s_out_data, b_d = dut->b_out_data;

        tick();

        if (s_push) s_acc=true;
        if (b_push) b_acc=true;
        if (feeding && s_acc && b_acc) { feeding=false; fed++; }
        if (s_pop) sq.push_back(s_d);
        if (b_pop) bq.push_back(b_d);

        lfsr_a = nxt(lfsr_a); lfsr_b = nxt(lfsr_b);
        if ((int)sq.size()>=NBEATS && (int)bq.size()>=NBEATS) break;
    }

    // drain both
    dut->feed_valid_s=0; dut->feed_valid_b=0; dut->s_out_ready=1; dut->b_out_ready=1;
    for (int i=0;i<20000;i++){
        bool s_pop = dut->s_out_valid && dut->s_out_ready;
        bool b_pop = dut->b_out_valid && dut->b_out_ready;
        uint32_t s_d=dut->s_out_data, b_d=dut->b_out_data;
        tick();
        if (s_pop && (int)sq.size()<NBEATS) sq.push_back(s_d);
        if (b_pop && (int)bq.size()<NBEATS) bq.push_back(b_d);
    }

    printf("[fifo_equiv] fed=%d popped skip=%zu bram=%zu\n", fed, sq.size(), bq.size());
    if (sq.size()!=bq.size()) { printf("[fifo_equiv] FAIL count mismatch\n"); return 1; }
    for (size_t i=0;i<sq.size();i++){
        if (sq[i]!=bq[i]) { printf("[fifo_equiv] FAIL beat %zu skip=%u bram=%u\n", i, sq[i], bq[i]); return 1; }
        if (sq[i]!=(uint32_t)i) { printf("[fifo_equiv] FAIL skip stream not increasing at %zu: %u\n", i, sq[i]); return 1; }
    }
    printf("[fifo_equiv] PASS %zu beats identical ordering\n", sq.size());
    return 0;
}
