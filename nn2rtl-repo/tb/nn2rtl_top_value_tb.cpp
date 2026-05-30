// End-to-end VALUE verification testbench for nn2rtl_top.v
//
// Goal: prove the *assembled* network produces byte-exact correct values, not
// just correct beat counts. The cycle-count TB (nn2rtl_top_cycle_count_tb.cpp)
// feeds zeros and only measures throughput; per-module Verilator runs each node
// in isolation. Neither catches an *integration* bug (e.g. an inter-module
// tiling/ABI mismatch). This TB closes that gap: it drives the real network
// input through the whole top and compares the real network output to golden.
//
// ABI (verified against the NN2V golden headers):
//   input : contract node_conv_196.goldin — 50,176 samples x 32 bytes (256-bit).
//           This is exactly the s_axis_tdata beat layout conv_196's data_in
//           consumed during its per-module verification, so feeding it straight
//           into s_axis replicates that ABI bit-for-bit (no repacking).
//   output: contract node_relu_48.goldout — 3,136 samples x 32 bytes (256-bit).
//           m_axis_tdata = node_relu_48_data_out, so capturing m_axis and
//           comparing to this golden replicates relu_48's per-module output check.
//   => if every module's goldout == next module's goldin (consistent ABIs), the
//      e2e output MUST equal relu_48's goldout. A mismatch localizes an
//      integration bug per-module testing could not see.
//
// Handshake logic is copied verbatim from the cycle-count TB (proven to drive
// 50,176 input beats and collect 3,136 output beats, terminating cleanly).
//
// Run-time contract:
//   argv[1] = path to conv_196 contract .goldin (256-bit / 32-byte samples)
//   argv[2] = path to relu_48 contract .goldout (256-bit / 32-byte samples)
//   argv[3] = vector index to run (default 0)
//
// Build: see scripts/run_nn2rtl_top_value.ts (clone of the cycle-count runner).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "Vnn2rtl_top.h"
#include "verilated.h"

double sc_time_stamp() { return 0; }

namespace {

constexpr uint64_t kMaxCycles   = 50'000'000;  // working frame ~13.3M; fail-fast ceiling
constexpr int      kResetCycles = 16;
constexpr uint64_t kInputBeats  = 224 * 224;   // 50,176 — one 256-bit pixel beat each
// kOutputBeats is derived at runtime from goldout.samples_per_vector (see main)
// so the TB is layer-agnostic for the truncated-output bisect. Default 3136 = relu_48.

// ---- NN2V binary vector file (.goldin / .goldout) ----
//   [ 0..4)  magic "NN2V"
//   [ 4..8)  version (=2)  LE u32
//   [ 8..12) num_vectors   LE u32
//   [12..16) samples_per_vector LE u32
//   [16..20) bytes_per_sample   LE u32
//   [20..)   num_vectors * samples_per_vector * ceil(bytes_per_sample/4) LE i32 words
struct VectorFile {
  uint32_t num_vectors = 0;
  uint32_t samples_per_vector = 0;
  uint32_t bytes_per_sample = 0;
  uint32_t words_per_sample = 0;
  // vectors[v][sample] -> words_per_sample uint32 words
  std::vector<std::vector<std::vector<uint32_t>>> vectors;
};

VectorFile loadVectorFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) throw std::runtime_error("cannot open vector file '" + path + "'");

  char magic[4];
  uint32_t version = 0, num_vectors = 0, spv = 0, bps = 0;
  in.read(magic, 4);
  in.read(reinterpret_cast<char*>(&version), 4);
  in.read(reinterpret_cast<char*>(&num_vectors), 4);
  in.read(reinterpret_cast<char*>(&spv), 4);
  in.read(reinterpret_cast<char*>(&bps), 4);
  if (!in) throw std::runtime_error("vector file '" + path + "' header truncated");
  if (magic[0] != 'N' || magic[1] != 'N' || magic[2] != '2' || magic[3] != 'V')
    throw std::runtime_error("vector file '" + path + "' bad magic (expected NN2V)");
  if (version != 2) throw std::runtime_error("vector file '" + path + "' unsupported version");
  if (bps == 0) throw std::runtime_error("vector file '" + path + "' bytes_per_sample == 0");

  VectorFile f;
  f.num_vectors = num_vectors;
  f.samples_per_vector = spv;
  f.bytes_per_sample = bps;
  f.words_per_sample = (bps + 3U) / 4U;
  f.vectors.reserve(num_vectors);

  std::vector<int32_t> row(static_cast<size_t>(spv) * f.words_per_sample);
  for (uint32_t v = 0; v < num_vectors; ++v) {
    in.read(reinterpret_cast<char*>(row.data()),
            static_cast<std::streamsize>(row.size() * sizeof(int32_t)));
    if (!in) throw std::runtime_error("vector file '" + path + "' data truncated");
    std::vector<std::vector<uint32_t>> samples;
    samples.reserve(spv);
    for (uint32_t s = 0; s < spv; ++s) {
      std::vector<uint32_t> w(f.words_per_sample, 0U);
      for (uint32_t k = 0; k < f.words_per_sample; ++k)
        w[k] = static_cast<uint32_t>(row[static_cast<size_t>(s) * f.words_per_sample + k]);
      samples.push_back(std::move(w));
    }
    f.vectors.push_back(std::move(samples));
  }
  return f;
}

// ---- packed-signal helpers (handle Verilator's VlWide / WData[] / integral) ----
template <typename SignalT>
void assignWords(SignalT& sig, const std::vector<uint32_t>& w) {
  if constexpr (std::is_integral_v<SignalT>) {
    uint64_t packed = 0;
    for (size_t i = 0; i < w.size() && i < (sizeof(SignalT) + 3) / 4; ++i)
      packed |= static_cast<uint64_t>(w[i]) << (32U * i);
    sig = static_cast<SignalT>(packed);
  }
}
template <std::size_t N>
void assignWords(VlWide<N>& sig, const std::vector<uint32_t>& w) {
  for (size_t i = 0; i < N; ++i) sig.at(i) = (i < w.size()) ? w[i] : 0U;
}
template <std::size_t N>
void assignWords(WData (&sig)[N], const std::vector<uint32_t>& w) {
  for (size_t i = 0; i < N; ++i) sig[i] = (i < w.size()) ? w[i] : 0U;
}

template <typename SignalT>
std::vector<uint32_t> readWords(const SignalT& sig, size_t n) {
  std::vector<uint32_t> w(n, 0U);
  if constexpr (std::is_integral_v<SignalT>) {
    using U = std::make_unsigned_t<SignalT>;
    const uint64_t packed = static_cast<uint64_t>(static_cast<U>(sig));
    for (size_t i = 0; i < n; ++i) w[i] = static_cast<uint32_t>(packed >> (32U * i));
  }
  return w;
}
template <std::size_t N>
std::vector<uint32_t> readWords(const VlWide<N>& sig, size_t n) {
  std::vector<uint32_t> w(n, 0U);
  for (size_t i = 0; i < n && i < N; ++i) w[i] = sig.at(i);
  return w;
}
template <std::size_t N>
std::vector<uint32_t> readWords(const WData (&sig)[N], size_t n) {
  std::vector<uint32_t> w(n, 0U);
  for (size_t i = 0; i < n && i < N; ++i) w[i] = sig[i];
  return w;
}

void tick(Vnn2rtl_top* dut, uint64_t& cycle) {
  dut->clk = 1; dut->eval();
  dut->clk = 0; dut->eval();
  cycle++;
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <conv_196.goldin> <relu_48.goldout> [vector_idx]\n", argv[0]);
    return 2;
  }
  const std::string goldin_path  = argv[1];
  const std::string goldout_path = argv[2];
  const uint32_t    vec_idx      = (argc >= 4) ? static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 0U;
  const std::string dump_path    = (argc >= 5) ? argv[4] : std::string();  // raw capture dump (3136*32 bytes)

  VectorFile goldin, goldout;
  try {
    goldin  = loadVectorFile(goldin_path);
    goldout = loadVectorFile(goldout_path);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[tb][value] golden load error: %s\n", e.what());
    return 2;
  }

  std::printf("[tb][value] goldin  : vectors=%u samples=%u bytes/sample=%u words/sample=%u\n",
              goldin.num_vectors, goldin.samples_per_vector, goldin.bytes_per_sample, goldin.words_per_sample);
  std::printf("[tb][value] goldout : vectors=%u samples=%u bytes/sample=%u words/sample=%u\n",
              goldout.num_vectors, goldout.samples_per_vector, goldout.bytes_per_sample, goldout.words_per_sample);
  // Layer-agnostic output beat count (truncated bisect retargets m_axis to an
  // intermediate layer whose frame size differs from relu_48's 3136).
  const uint64_t kOutputBeats = goldout.samples_per_vector;

  // ABI sanity — the whole test rests on these matching the top's 256-bit ports.
  if (vec_idx >= goldin.num_vectors || vec_idx >= goldout.num_vectors) {
    std::fprintf(stderr, "[tb][value] vector_idx %u out of range\n", vec_idx); return 2;
  }
  if (goldin.samples_per_vector != kInputBeats) {
    std::fprintf(stderr, "[tb][value] goldin samples %u != expected input beats %llu\n",
                 goldin.samples_per_vector, (unsigned long long)kInputBeats); return 2;
  }
  // (goldout beat count is taken from the file itself — no fixed expectation.)
  if (goldin.bytes_per_sample != 32 || goldout.bytes_per_sample != 32) {
    std::fprintf(stderr, "[tb][value] expected 32-byte (256-bit) samples on both ends "
                 "(use the *contract* goldens, not the logical ones)\n"); return 2;
  }

  const auto& in_samples  = goldin.vectors[vec_idx];
  const auto& out_samples = goldout.vectors[vec_idx];

  Vnn2rtl_top* dut = new Vnn2rtl_top;

  // idle init (matches cycle-count TB)
  dut->clk = 0; dut->rst_n = 0;
  dut->s_axis_tvalid = 0; dut->s_axis_tlast = 0;
  memset(&dut->s_axis_tdata, 0, sizeof(dut->s_axis_tdata));
  dut->m_axis_tready = 1;
  dut->s_axil_awvalid = 0; dut->s_axil_awaddr = 0;
  dut->s_axil_wvalid = 0; dut->s_axil_wdata = 0; dut->s_axil_wstrb = 0;
  dut->s_axil_bready = 0;
  dut->s_axil_arvalid = 0; dut->s_axil_araddr = 0; dut->s_axil_rready = 0;
  dut->eval();

  uint64_t cycle = 0;
  for (int i = 0; i < kResetCycles; i++) tick(dut, cycle);
  dut->rst_n = 1;
  tick(dut, cycle);
  std::printf("[tb][value] reset deasserted at cycle %llu\n", (unsigned long long)cycle);

  // captured output beats: each is words_per_sample uint32 words
  std::vector<std::vector<uint32_t>> captured;
  captured.reserve(kOutputBeats);

  uint64_t input_beats_sent  = 0;
  uint64_t output_beats_seen = 0;
  uint64_t first_input_cycle = 0, last_output_cycle = 0;
  bool first_input_seen = false, done = false;

  while (cycle < kMaxCycles && !done) {
    if (input_beats_sent < kInputBeats) {
      dut->s_axis_tvalid = 1;
      dut->s_axis_tlast  = (input_beats_sent + 1 == kInputBeats) ? 1 : 0;
      assignWords(dut->s_axis_tdata, in_samples[input_beats_sent]);
    } else {
      dut->s_axis_tvalid = 0;
      dut->s_axis_tlast  = 0;
    }
    dut->eval();

    const bool input_accept  = dut->s_axis_tvalid && dut->s_axis_tready;
    const bool output_accept = dut->m_axis_tvalid && dut->m_axis_tready;
    const bool output_last   = output_accept && dut->m_axis_tlast;

    if (input_accept) {
      if (input_beats_sent < 3) {
        auto iw = readWords(dut->s_axis_tdata, 8);
        std::printf("[dbg-in] beat %llu words[0..2]=%u %u %u (driven s_axis_tdata)\n",
                    (unsigned long long)input_beats_sent, iw[0], iw[1], iw[2]);
      }
      if (!first_input_seen) { first_input_seen = true; first_input_cycle = cycle; }
      input_beats_sent++;
    }
    if (output_accept) {
      auto ow = readWords(dut->m_axis_tdata, goldout.words_per_sample);
      if (output_beats_seen < 6)
        std::printf("[dbg-out] beat %llu words=%u %u %u %u %u %u %u %u\n",
                    (unsigned long long)output_beats_seen, ow[0], ow[1], ow[2], ow[3], ow[4], ow[5], ow[6], ow[7]);
      captured.push_back(ow);
      output_beats_seen++;
      if ((output_beats_seen % 500) == 0)
        std::printf("[tb][value] output_beats=%llu/%llu at cycle %llu\n",
                    (unsigned long long)output_beats_seen, (unsigned long long)goldout.samples_per_vector,
                    (unsigned long long)cycle), std::fflush(stdout);
      // Stop on tlast OR once we've captured a full frame of goldout beats. The
      // count-based stop makes the TB layer-agnostic (truncated bisect: m_axis
      // can be retargeted to an intermediate layer whose tlast won't fire here).
      if (output_last || output_beats_seen >= goldout.samples_per_vector) {
        last_output_cycle = cycle; done = true;
      }
    }

    if ((cycle & 0xFFFFF) == 0 && cycle > 0) {
      std::printf("[tb][value] cycle=%llu in=%llu/%llu out=%llu/%llu tready=%d tvalid=%d\n",
                  (unsigned long long)cycle, (unsigned long long)input_beats_sent,
                  (unsigned long long)kInputBeats, (unsigned long long)output_beats_seen,
                  (unsigned long long)kOutputBeats, (int)dut->s_axis_tready, (int)dut->m_axis_tvalid);
      std::fflush(stdout);
    }
    tick(dut, cycle);
  }

  dut->final();

  if (!done) {
    std::printf("[tb][value] TIMEOUT at cycle %llu — in=%llu out=%llu (no frame end)\n",
                (unsigned long long)cycle, (unsigned long long)input_beats_sent,
                (unsigned long long)output_beats_seen);
    std::printf("[tb][value][summary] result=TIMEOUT beats=%llu/%llu mismatch_bytes=-1 first_mismatch_beat=-1\n",
                (unsigned long long)output_beats_seen, (unsigned long long)kOutputBeats);
    delete dut;
    return 1;
  }

  std::printf("[tb][value] frame done: first_input_cycle=%llu last_output_cycle=%llu e2e_cycles=%llu beats=%llu\n",
              (unsigned long long)first_input_cycle, (unsigned long long)last_output_cycle,
              (unsigned long long)(last_output_cycle - first_input_cycle),
              (unsigned long long)output_beats_seen);

  // ---- dump raw captured output for offline (Python) structural analysis ----
  if (!dump_path.empty()) {
    std::ofstream df(dump_path, std::ios::binary);
    if (df.is_open()) {
      for (const auto& beat : captured)
        for (uint32_t w = 0; w < goldout.words_per_sample; ++w) {
          const uint32_t word = (w < beat.size()) ? beat[w] : 0U;
          df.write(reinterpret_cast<const char*>(&word), 4);
        }
      std::printf("[tb][value] dumped %zu beats (%zu bytes) to %s\n",
                  captured.size(), captured.size() * goldout.bytes_per_sample, dump_path.c_str());
    } else {
      std::fprintf(stderr, "[tb][value] WARN could not open dump path %s\n", dump_path.c_str());
    }
  }

  // ---- byte-exact compare: captured vs goldout, 32 bytes/beat ----
  const uint32_t bps = goldout.bytes_per_sample;       // 32
  const uint32_t wps = goldout.words_per_sample;        // 8
  uint64_t mismatch_bytes = 0;
  int64_t  first_mm_beat = -1, first_mm_byte = -1;
  int      first_mm_exp = 0, first_mm_got = 0;

  const uint64_t beats_to_cmp = (output_beats_seen < kOutputBeats) ? output_beats_seen : kOutputBeats;
  for (uint64_t b = 0; b < beats_to_cmp; ++b) {
    const auto& got = captured[b];
    const auto& exp = out_samples[b];
    for (uint32_t byte = 0; byte < bps; ++byte) {
      const uint8_t gb = static_cast<uint8_t>((got[byte / 4] >> (8U * (byte % 4))) & 0xFF);
      const uint8_t eb = static_cast<uint8_t>((exp[byte / 4] >> (8U * (byte % 4))) & 0xFF);
      if (gb != eb) {
        if (first_mm_beat < 0) {
          first_mm_beat = static_cast<int64_t>(b);
          first_mm_byte = static_cast<int64_t>(byte);
          first_mm_exp = static_cast<int8_t>(eb);
          first_mm_got = static_cast<int8_t>(gb);
        }
        mismatch_bytes++;
      }
    }
    (void)wps;
  }

  const bool pass = (mismatch_bytes == 0) && (output_beats_seen == kOutputBeats);
  if (first_mm_beat >= 0) {
    std::printf("[tb][value] FIRST MISMATCH: beat=%lld byte=%lld (tile_channel=%lld) expected=%d got=%d\n",
                (long long)first_mm_beat, (long long)first_mm_byte, (long long)first_mm_byte,
                first_mm_exp, first_mm_got);
    // beat -> (pixel, tile): 64 tiles/pixel, 32 ch/tile
    const long long px = first_mm_beat / 64, tile = first_mm_beat % 64;
    const long long ch = tile * 32 + first_mm_byte;
    std::printf("[tb][value] mismatch maps to pixel=%lld (of 49), channel=%lld (of 2048)\n", px, ch);
  }
  std::printf("[tb][value] total mismatching bytes = %llu / %llu\n",
              (unsigned long long)mismatch_bytes, (unsigned long long)(beats_to_cmp * bps));
  std::printf("[tb][value][summary] result=%s beats=%llu/%llu mismatch_bytes=%llu first_mismatch_beat=%lld\n",
              pass ? "PASS" : "FAIL",
              (unsigned long long)output_beats_seen, (unsigned long long)kOutputBeats,
              (unsigned long long)mismatch_bytes, (long long)first_mm_beat);

  delete dut;
  return pass ? 0 : 1;
}
