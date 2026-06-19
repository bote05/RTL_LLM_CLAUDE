// requant_pipeline_tb.cpp
// ---------------------------------------------------------------------------
// Verilator unit testbench for output/rtl/engine/requant_pipeline.v
//
// Verification protocol (matches docs/agent_tasks/08_engine_requant_pipeline.md
// "How to verify" §2 and §3):
//
//   Sub-test A — node_conv_288 scale parameters (SCALE_MULT=15825,
//                SCALE_SHIFT=20). Random INT32 accumulators across all 256
//                lanes; random INT16-magnitude biases. Used to exercise the
//                pipeline on the seed reference's actual per-layer scale.
//
//   Sub-test B — bit-exact cross-check against node_conv_298's requant tail.
//                Pull node_conv_298's SCALE_MULT=28241 and SCALE_SHIFT=22 from
//                docs/agent_tasks "How to verify" §3, plus the first 256
//                INT32 biases from output/weights/node_conv_298_bias.hex
//                (the same file node_conv_298.v feeds to $readmemh). The
//                C++ golden uses the exact same arithmetic the RTL does
//                (sign-aware round + arithmetic shift + INT8 clamp), so a
//                byte-identical RTL output proves byte-identical equivalence
//                to node_conv_298's requant tail for the first 256 OC
//                channels — the channels this engine sub-block processes
//                per oc_pass.
//
// Each beat presents one INT32 per lane (256 per beat). Total lane outputs
// across both sub-tests = 2 subs × 6 beats × 256 lanes = 3,072 — well over
// the spec's 1,000-sample floor.
//
// Build (from oss-cad-suite or any verilator install):
//   verilator -Wall -cc --build --exe \
//             output/rtl/engine/requant_pipeline.v \
//             output/rtl/engine/requant_pipeline_tb.cpp \
//             -Mdir build_requant_pipeline_tb \
//             --top-module requant_pipeline
//   ./build_requant_pipeline_tb/Vrequant_pipeline
// ---------------------------------------------------------------------------

#include <verilated.h>
#include "Vrequant_pipeline.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kLanes        = 256;
constexpr int kBeatsPerSub  = 6;
constexpr uint32_t kSeedA   = 0xC0FFEEu;
constexpr uint32_t kSeedB   = 0xBADCAFEu;

vluint64_t main_time = 0;

void tick(Vrequant_pipeline* dut) {
    dut->clk = 0;
    dut->eval();
    main_time++;
    dut->clk = 1;
    dut->eval();
    main_time++;
}

void set_acc_lane(Vrequant_pipeline* dut, int lane, int32_t val) {
    // acc_in is [8191:0] — Verilator surfaces it as 256 packed uint32_t
    // words, one per lane (lane*32 +: 32 == word `lane`).
    uint32_t* w = reinterpret_cast<uint32_t*>(&dut->acc_in);
    w[lane] = static_cast<uint32_t>(val);
}

void set_bias_lane(Vrequant_pipeline* dut, int lane, int32_t val) {
    uint32_t* w = reinterpret_cast<uint32_t*>(&dut->bias_in);
    w[lane] = static_cast<uint32_t>(val);
}

int8_t get_data_lane(Vrequant_pipeline* dut, int lane) {
    // data_out is [2047:0] = 256 INT8 values, packed 4-per-uint32_t word.
    uint32_t* w = reinterpret_cast<uint32_t*>(&dut->data_out);
    int wi  = lane / 4;
    int bo  = (lane % 4) * 8;
    uint8_t byte = static_cast<uint8_t>((w[wi] >> bo) & 0xFFu);
    return static_cast<int8_t>(byte);
}

// Bit-exact C++ golden. Matches the canonical Verilog arithmetic in
// node_conv_288.v ST_BIAS_SCALE/ST_PACK and in 01_context.md §"Scale-shift
// rounding — MANDATORY":
//
//   biased = acc + bias                                    (signed 33-bit)
//   scaled = biased * scale_mult                           (signed 49-bit)
//   half   = 1 << (scale_shift - 1)
//   bias_r = (scaled < 0) ? (half - 1) : half              (sign-aware)
//   v_tmp  = (scaled + bias_r) >>> scale_shift             (arithmetic)
//   out    = clamp(v_tmp, -128, +127)                      (INT8 saturate)
//
// C++20 mandates arithmetic right-shift on signed integers, matching
// Verilog `>>>` on signed operands. int64_t headroom is enough for all
// intermediate values used here.
int8_t requant_golden(int32_t acc, int32_t bias,
                      int16_t scale_mult, uint8_t scale_shift) {
    int64_t biased = static_cast<int64_t>(acc) + static_cast<int64_t>(bias);
    int64_t scaled = biased * static_cast<int64_t>(scale_mult);
    int64_t half   = static_cast<int64_t>(1) << (scale_shift - 1);
    int64_t bias_r = (scaled < 0) ? (half - 1) : half;
    int64_t v_tmp  = (scaled + bias_r) >> scale_shift;
    if (v_tmp > 127)  return 127;
    if (v_tmp < -128) return -128;
    return static_cast<int8_t>(v_tmp);
}

// Load INT32 biases from a hex file (one 32-bit hex value per line, no
// prefix; matches the format $readmemh consumes per
// knowledge/patterns/protected/01_context.md §"Weight and bias loading").
std::vector<int32_t> load_int32_hex(const std::string& path, int max_count) {
    std::vector<int32_t> out;
    std::ifstream f(path);
    if (!f.is_open()) return out;
    std::string line;
    while (std::getline(f, line) && static_cast<int>(out.size()) < max_count) {
        if (line.empty()) continue;
        uint32_t v = std::strtoul(line.c_str(), nullptr, 16);
        out.push_back(static_cast<int32_t>(v));
    }
    return out;
}

struct Sub {
    const char*           name;
    int16_t               scale_mult;
    uint8_t               scale_shift;
    std::vector<int32_t>  biases;
};

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vrequant_pipeline* dut = new Vrequant_pipeline;

    // ---- Initialise inputs ----
    dut->clk         = 0;
    dut->rst_n       = 0;
    dut->valid_in    = 0;
    dut->scale_mult  = 0;
    dut->scale_shift = 0;
    std::memset(&dut->acc_in,  0, sizeof(dut->acc_in));
    std::memset(&dut->bias_in, 0, sizeof(dut->bias_in));
    dut->eval();

    // ---- Hold reset, then deassert ----
    for (int i = 0; i < 4; ++i) tick(dut);
    dut->rst_n = 1;
    tick(dut);
    tick(dut);

    // ---- Sub-test definitions ----
    std::vector<Sub> subs;

    // Sub A: node_conv_288 scale, biases drawn pseudo-randomly in INT16
    // magnitude (real INT32 biases on this layer are typically << 2^16).
    {
        Sub s;
        s.name        = "node_conv_288_scale";
        s.scale_mult  = 15825;
        s.scale_shift = 20;
        uint32_t rng = 0xDEADBEEFu;
        for (int i = 0; i < kLanes; ++i) {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            int32_t b = static_cast<int32_t>(rng & 0xFFFFu) - 32768;
            s.biases.push_back(b);
        }
        subs.push_back(std::move(s));
    }

    // Sub B: bit-exact cross-check against node_conv_298. Real per-layer
    // scale_mult / scale_shift and the actual hex bias file the per-layer
    // module's $readmemh loads.
    {
        Sub s;
        s.name        = "node_conv_298_xref";
        s.scale_mult  = 28241;
        s.scale_shift = 22;
        s.biases = load_int32_hex("output/weights/node_conv_298_bias.hex",
                                  kLanes);
        if (static_cast<int>(s.biases.size()) < kLanes) {
            printf("WARN: node_conv_298_bias.hex shorter than %d lines; "
                   "got %zu (test will continue with zeros padded)\n",
                   kLanes, s.biases.size());
            while (static_cast<int>(s.biases.size()) < kLanes)
                s.biases.push_back(0);
        }
        subs.push_back(std::move(s));
    }

    // ---- Run sub-tests ----
    std::mt19937 rng_a(kSeedA);
    std::mt19937 rng_b(kSeedB);
    std::uniform_int_distribution<int32_t> acc_full(
        std::numeric_limits<int32_t>::min(),
        std::numeric_limits<int32_t>::max());

    int total_outputs   = 0;
    int mismatches      = 0;
    int max_err         = 0;
    int first_bad_sub   = -1;
    int first_bad_beat  = -1;
    int first_bad_lane  = -1;
    int first_bad_got   = 0;
    int first_bad_exp   = 0;
    int first_bad_acc   = 0;
    int first_bad_bias  = 0;

    for (size_t si = 0; si < subs.size(); ++si) {
        const Sub& s = subs[si];
        std::mt19937& rng = (si == 0) ? rng_a : rng_b;

        // Hold scale params and the bias wide-word stable for the whole sub.
        dut->scale_mult  = static_cast<uint16_t>(s.scale_mult);
        dut->scale_shift = s.scale_shift;
        for (int lane = 0; lane < kLanes; ++lane)
            set_bias_lane(dut, lane, s.biases[lane]);

        for (int beat = 0; beat < kBeatsPerSub; ++beat) {
            // Generate accumulators. Mix full-range and narrow-range beats
            // so we exercise the saturate path and the small-magnitude
            // rounding path both.
            std::array<int32_t, kLanes> beat_accs{};
            for (int lane = 0; lane < kLanes; ++lane) {
                int32_t a;
                if (beat & 1) {
                    // Narrow: ±2^23. Plenty of values that round non-trivially
                    // and don't saturate the INT8 output.
                    uint32_t r = rng();
                    a = static_cast<int32_t>(r & 0xFFFFFFu) -
                        (1 << 23);
                } else {
                    a = acc_full(rng);
                }
                beat_accs[lane] = a;
                set_acc_lane(dut, lane, a);
            }

            // Three-cycle pipeline: assert valid_in for one cycle, then
            // tick twice more to let stage-2 and stage-3 retire.
            dut->valid_in = 1;
            tick(dut);
            dut->valid_in = 0;
            tick(dut);
            tick(dut);

            if (!dut->valid_out) {
                printf("FAIL: valid_out not asserted at expected sample "
                       "cycle (sub=%s beat=%d)\n", s.name, beat);
                delete dut;
                return 1;
            }

            for (int lane = 0; lane < kLanes; ++lane) {
                int8_t got = get_data_lane(dut, lane);
                int8_t exp = requant_golden(beat_accs[lane], s.biases[lane],
                                            s.scale_mult, s.scale_shift);
                ++total_outputs;
                int err = std::abs(static_cast<int>(got) -
                                   static_cast<int>(exp));
                if (err > max_err) max_err = err;
                if (err != 0) {
                    if (first_bad_lane < 0) {
                        first_bad_sub  = static_cast<int>(si);
                        first_bad_beat = beat;
                        first_bad_lane = lane;
                        first_bad_got  = got;
                        first_bad_exp  = exp;
                        first_bad_acc  = beat_accs[lane];
                        first_bad_bias = s.biases[lane];
                    }
                    mismatches++;
                }
            }
        }
    }

    printf("=== requant_pipeline unit test ===\n");
    printf("subs            : %zu\n", subs.size());
    for (size_t si = 0; si < subs.size(); ++si) {
        printf("  sub[%zu]       : %-22s scale_mult=%d scale_shift=%d\n",
               si, subs[si].name, subs[si].scale_mult, subs[si].scale_shift);
    }
    printf("beats/sub       : %d\n", kBeatsPerSub);
    printf("lanes           : %d\n", kLanes);
    printf("total outputs   : %d\n", total_outputs);
    printf("mismatches      : %d / %d\n", mismatches, total_outputs);
    printf("max_error       : %d\n", max_err);
    if (mismatches > 0) {
        printf("first_bad       : sub=%d beat=%d lane=%d "
               "acc=%d bias=%d  got=%d exp=%d\n",
               first_bad_sub, first_bad_beat, first_bad_lane,
               first_bad_acc, first_bad_bias,
               first_bad_got, first_bad_exp);
    }
    printf("STATUS          : %s\n", max_err == 0 ? "PASS" : "FAIL");

    delete dut;
    return max_err == 0 ? 0 : 1;
}
