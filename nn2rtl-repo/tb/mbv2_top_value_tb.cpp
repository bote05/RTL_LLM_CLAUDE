// End-to-end VALUE verification testbench for the MobileNetV2 all-spatial
// nn2rtl_top.v (output/mobilenet-v2/rtl/nn2rtl_top.v).
//
// Goal: prove the *assembled* MobileNetV2 produces byte-exact correct logits,
// not just correct beat counts. Per-module Verilator runs each node in
// isolation and the cycle-count TB only measures throughput; neither catches
// an integration (inter-module ABI / handshake) bug. This TB closes that gap:
// it drives the real network input through the whole top and compares the real
// network output to golden.
//
// ABI (verified against the NN2V golden headers):
//   input : node_conv_810.goldin — nvec=8, samples_per_vec=50176, bps=3.
//           Each 3-byte sample is one 24-bit s_axis beat = one RGB pixel.
//           50176 = 224*224 pixels (the network input image), so feeding it
//           straight into s_axis replicates conv_810's per-module input ABI
//           bit-for-bit (no repacking).
//   output: node_linear.goldout — nvec=8, samples_per_vec=1, bps=1000.
//           Each vector's output is ONE 8000-bit m_axis beat = 1000 INT8
//           logits. m_axis_tdata = node_linear_data_out, so capturing m_axis
//           and comparing to this golden replicates linear's per-module output
//           check end-to-end.
//   => if every module's goldout == next module's goldin (consistent ABIs),
//      the e2e output MUST equal node_linear's goldout. A mismatch localizes an
//      integration bug per-module testing could not see.
//
// NN2V vector file layout (both .goldin and .goldout):
//   [ 0.. 4)  magic "NN2V"
//   [ 4.. 8)  version (=2)             LE u32
//   [ 8..12)  num_vectors             LE u32
//   [12..16)  samples_per_vector      LE u32
//   [16..20)  bytes_per_sample        LE u32
//   [20..  )  num_vectors * samples_per_vector * ceil(bps/4) LE i32 words
//   Each sample occupies words_per_sample = ceil(bps/4) little-endian u32
//   words; for the goldin (bps=3) the RGB pixel sits in the low 24 bits of one
//   word, for the goldout (bps=1000) it is 250 words = 1000 bytes.
//
// Run-time contract:
//   argv[1] = path to node_conv_810.goldin (24-bit / 3-byte samples)
//   argv[2] = path to node_linear.goldout  (1000-byte / 8000-bit samples)
//   argv[3] = vector index to run (default 0 / VEC 0 — one image)
//   argv[4] = (optional) raw m_axis capture dump path
//   env MBV2_VEC          : overrides the vector index (argv[3] wins if given)
//   env MBV2_ALL_VECS=1   : run every vector in the file (slow: 8x 50176 beats)
//   env MBV2_MAX_CYCLES   : per-frame cycle cap (default 50,000,000 — the
//                           all-spatial frame is gated by node_conv_912 at
//                           ~20.2M cycles, so the e2e latency is ~20-30M).
//                           Hitting the cap prints a TIMEOUT/DEADLOCK summary
//                           with the last input/output beat counts so a
//                           handshake stall is obvious in the log.
//
// Build: see scripts/run_mbv2_top_value.ts.

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

constexpr int      kResetCycles = 16;
constexpr uint64_t kInputBeats  = 224ULL * 224ULL;   // 50,176 — one 24-bit RGB pixel beat each
// Per-frame fail-fast ceiling. The mbv2 all-spatial frame is gated by its
// slowest stage (node_conv_912 ~20.2M cycles/frame per the throughput report),
// so the real e2e first-in-to-last-out latency is on the order of 20-30M
// cycles. 5M would falsely trip a TIMEOUT on a correctly-progressing run.
// 50M matches the ResNet TB's margin (~3.75x over its ~13.3M-cycle frame).
// Override with env MBV2_MAX_CYCLES if a build runs longer/shorter.
constexpr uint64_t kDefaultMaxCycles = 50'000'000;

// ---- NN2V binary vector file (.goldin / .goldout) ----
struct VectorFile {
  uint32_t num_vectors = 0;
  uint32_t samples_per_vector = 0;
  uint32_t bytes_per_sample = 0;
  uint32_t words_per_sample = 0;
  // vectors[v][sample] -> words_per_sample uint32 words (little-endian byte order)
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

// ---- one full frame for vector `vec_idx`. Returns true on PASS. ----
struct FrameStats {
  bool     timed_out = false;
  uint64_t input_beats_sent = 0;
  uint64_t output_beats_seen = 0;
  uint64_t e2e_cycles = 0;
  uint64_t mismatch_bytes = 0;
  int64_t  first_mm_beat = -1;
  int64_t  first_mm_byte = -1;
  int      first_mm_exp = 0;
  int      first_mm_got = 0;
};

FrameStats runFrame(Vnn2rtl_top* dut, const VectorFile& goldin, const VectorFile& goldout,
                    uint32_t vec_idx, uint64_t max_cycles, const std::string& dump_path) {
  FrameStats st;
  const uint64_t kOutputBeats = goldout.samples_per_vector;   // 1 for node_linear
  const auto& in_samples  = goldin.vectors[vec_idx];
  const auto& out_samples = goldout.vectors[vec_idx];

  // idle init each frame
  dut->rst_n = 0;
  dut->s_axis_tvalid = 0; dut->s_axis_tlast = 0;
  memset(&dut->s_axis_tdata, 0, sizeof(dut->s_axis_tdata));
  dut->m_axis_tready = 1;
  dut->s_axil_awvalid = 0; dut->s_axil_awaddr = 0;
  dut->s_axil_wvalid = 0; dut->s_axil_wdata = 0; dut->s_axil_wstrb = 0;
  dut->s_axil_bready = 0;
  dut->s_axil_arvalid = 0; dut->s_axil_araddr = 0; dut->s_axil_rready = 0;
  dut->clk = 0; dut->eval();

  uint64_t cycle = 0;
  for (int i = 0; i < kResetCycles; i++) tick(dut, cycle);
  dut->rst_n = 1;
  tick(dut, cycle);
  std::printf("[tb][mbv2][vec%u] reset deasserted at cycle %llu\n",
              vec_idx, (unsigned long long)cycle);

  std::vector<std::vector<uint32_t>> captured;
  captured.reserve(static_cast<size_t>(kOutputBeats));

  uint64_t first_input_cycle = 0, last_output_cycle = 0;
  bool first_input_seen = false, done = false;

  while (cycle < max_cycles && !done) {
    // ----- drive s_axis (one 24-bit RGB pixel per beat; tlast on the last) -----
    if (st.input_beats_sent < kInputBeats) {
      dut->s_axis_tvalid = 1;
      dut->s_axis_tlast  = (st.input_beats_sent + 1 == kInputBeats) ? 1 : 0;
      assignWords(dut->s_axis_tdata, in_samples[st.input_beats_sent]);
    } else {
      dut->s_axis_tvalid = 0;
      dut->s_axis_tlast  = 0;
    }
    dut->m_axis_tready = 1;
    dut->eval();

    const bool input_accept  = dut->s_axis_tvalid && dut->s_axis_tready;
    const bool output_accept = dut->m_axis_tvalid && dut->m_axis_tready;
    const bool output_last   = output_accept && dut->m_axis_tlast;

    if (input_accept) {
      if (st.input_beats_sent < 3) {
        auto iw = readWords(dut->s_axis_tdata, 1);
        std::printf("[dbg-in] beat %llu rgb=0x%06X\n",
                    (unsigned long long)st.input_beats_sent, iw[0] & 0xFFFFFFu);
      }
      if (!first_input_seen) { first_input_seen = true; first_input_cycle = cycle; }
      st.input_beats_sent++;
    }
    if (output_accept) {
      // [SERIALIZED OUTPUT 2026-06-08] m_axis is now a 256b stream (8 words/beat); the 1000
      // logits arrive over ceil(1000/32)=32 beats with tlast on the last. Capture every beat
      // (8 words), then reassemble into the 1000-byte logit vector in the compare below.
      auto ow = readWords(dut->m_axis_tdata, 8);
      if (st.output_beats_seen < 2) {
        // first few logit bytes for a quick sanity glance
        std::printf("[dbg-out] beat %llu logit[0..7]=%d %d %d %d %d %d %d %d\n",
                    (unsigned long long)st.output_beats_seen,
                    (int8_t)(ow[0] & 0xFF), (int8_t)((ow[0] >> 8) & 0xFF),
                    (int8_t)((ow[0] >> 16) & 0xFF), (int8_t)((ow[0] >> 24) & 0xFF),
                    (int8_t)(ow[1] & 0xFF), (int8_t)((ow[1] >> 8) & 0xFF),
                    (int8_t)((ow[1] >> 16) & 0xFF), (int8_t)((ow[1] >> 24) & 0xFF));
      }
      captured.push_back(std::move(ow));
      st.output_beats_seen++;
      // Stop on tlast (serialized: tlast on the final 256b beat). 256 = safety cap >> 32.
      if (output_last || st.output_beats_seen >= 256) {
        last_output_cycle = cycle; done = true;
      }
    }

    if ((cycle & 0xFFFFF) == 0 && cycle > 0) {
      std::printf("[tb][mbv2][vec%u] cycle=%llu in=%llu/%llu out=%llu/%llu tready=%d tvalid=%d\n",
                  vec_idx, (unsigned long long)cycle,
                  (unsigned long long)st.input_beats_sent, (unsigned long long)kInputBeats,
                  (unsigned long long)st.output_beats_seen, (unsigned long long)kOutputBeats,
                  (int)dut->s_axis_tready, (int)dut->m_axis_tvalid);
      std::fflush(stdout);
    }
    tick(dut, cycle);
  }

  if (!done) {
    st.timed_out = true;
    return st;
  }

  st.e2e_cycles = last_output_cycle - first_input_cycle;
  std::printf("[tb][mbv2][vec%u] frame done: first_input_cycle=%llu last_output_cycle=%llu "
              "e2e_cycles=%llu out_beats=%llu\n",
              vec_idx, (unsigned long long)first_input_cycle, (unsigned long long)last_output_cycle,
              (unsigned long long)st.e2e_cycles, (unsigned long long)st.output_beats_seen);

  // ---- optional raw dump of captured output (for offline analysis) ----
  if (!dump_path.empty()) {
    std::ofstream df(dump_path, std::ios::binary);
    if (df.is_open()) {
      for (const auto& beat : captured)
        for (uint32_t w = 0; w < goldout.words_per_sample; ++w) {
          const uint32_t word = (w < beat.size()) ? beat[w] : 0U;
          df.write(reinterpret_cast<const char*>(&word), 4);
        }
      std::printf("[tb][mbv2][vec%u] dumped %zu beats (%zu bytes) to %s\n",
                  vec_idx, captured.size(),
                  captured.size() * goldout.bytes_per_sample, dump_path.c_str());
    } else {
      std::fprintf(stderr, "[tb][mbv2][vec%u] WARN could not open dump path %s\n",
                   vec_idx, dump_path.c_str());
    }
  }

  // ---- byte-exact compare: reassemble the serialized 256b beats into one byte stream
  //      (beat0 bytes 0..31, beat1 32..63, ...) then compare the first bps(=1000) bytes to
  //      the single 1000-byte goldout sample. Byte order = same low-to-high logit order the
  //      output_serializer emits (beat b lane j = logit b*32+j), so the concatenation is the
  //      original dp_data_out[7999:0] byte-for-byte. ----
  const uint32_t bps = goldout.bytes_per_sample;   // 1000
  std::vector<uint8_t> got_bytes;
  got_bytes.reserve(captured.size() * 32);
  for (const auto& beat : captured)
    for (uint32_t w = 0; w < beat.size(); ++w)
      for (int k = 0; k < 4; ++k)
        got_bytes.push_back(static_cast<uint8_t>((beat[w] >> (8U * k)) & 0xFF));
  const auto& exp = out_samples[0];
  for (uint32_t byte = 0; byte < bps; ++byte) {
    const uint8_t gb = (byte < got_bytes.size()) ? got_bytes[byte] : 0xFFu;
    const uint8_t eb = static_cast<uint8_t>((exp[byte / 4] >> (8U * (byte % 4))) & 0xFF);
    if (gb != eb) {
      if (st.first_mm_beat < 0) {
        st.first_mm_beat = 0;
        st.first_mm_byte = static_cast<int64_t>(byte);
        st.first_mm_exp  = static_cast<int8_t>(eb);
        st.first_mm_got  = static_cast<int8_t>(gb);
      }
      st.mismatch_bytes++;
    }
  }
  return st;
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <node_conv_810.goldin> <node_linear.goldout> [vector_idx] [dump_path]\n",
                 argv[0]);
    return 2;
  }
  const std::string goldin_path  = argv[1];
  const std::string goldout_path = argv[2];

  // Vector selection: argv[3] wins, else env MBV2_VEC, else 0.
  uint32_t vec_idx = 0U;
  if (const char* e = std::getenv("MBV2_VEC")) vec_idx = static_cast<uint32_t>(std::strtoul(e, nullptr, 10));
  if (argc >= 4) vec_idx = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
  const std::string dump_path = (argc >= 5) ? argv[4] : std::string();

  const bool all_vecs = []() { const char* e = std::getenv("MBV2_ALL_VECS"); return e && e[0] == '1'; }();

  uint64_t max_cycles = kDefaultMaxCycles;
  if (const char* e = std::getenv("MBV2_MAX_CYCLES")) {
    const uint64_t v = std::strtoull(e, nullptr, 10);
    if (v > 0) max_cycles = v;
  }

  VectorFile goldin, goldout;
  try {
    goldin  = loadVectorFile(goldin_path);
    goldout = loadVectorFile(goldout_path);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[tb][mbv2] golden load error: %s\n", e.what());
    return 2;
  }

  std::printf("[tb][mbv2] goldin  : vectors=%u samples=%u bytes/sample=%u words/sample=%u\n",
              goldin.num_vectors, goldin.samples_per_vector, goldin.bytes_per_sample, goldin.words_per_sample);
  std::printf("[tb][mbv2] goldout : vectors=%u samples=%u bytes/sample=%u words/sample=%u\n",
              goldout.num_vectors, goldout.samples_per_vector, goldout.bytes_per_sample, goldout.words_per_sample);
  std::printf("[tb][mbv2] max_cycles=%llu input_beats=%llu all_vecs=%d\n",
              (unsigned long long)max_cycles, (unsigned long long)kInputBeats, (int)all_vecs);

  // ABI sanity — the whole test rests on these matching the top's ports.
  if (goldin.samples_per_vector != kInputBeats) {
    std::fprintf(stderr, "[tb][mbv2] goldin samples %u != expected input beats %llu\n",
                 goldin.samples_per_vector, (unsigned long long)kInputBeats);
    return 2;
  }
  if (goldin.bytes_per_sample != 3) {
    std::fprintf(stderr, "[tb][mbv2] goldin bytes/sample %u != 3 (expected 24-bit RGB pixels)\n",
                 goldin.bytes_per_sample);
    return 2;
  }
  if (goldout.bytes_per_sample != 1000) {
    std::fprintf(stderr, "[tb][mbv2] goldout bytes/sample %u != 1000 (expected 1000 INT8 logits)\n",
                 goldout.bytes_per_sample);
    return 2;
  }
  if (goldout.samples_per_vector != 1) {
    std::fprintf(stderr, "[tb][mbv2] goldout samples/vector %u != 1 (expected one 8000-bit beat)\n",
                 goldout.samples_per_vector);
    return 2;
  }

  const uint32_t first_vec = all_vecs ? 0U : vec_idx;
  const uint32_t last_vec  = all_vecs ? goldin.num_vectors : (vec_idx + 1U);
  if (first_vec >= goldin.num_vectors || last_vec - 1U >= goldin.num_vectors ||
      vec_idx >= goldout.num_vectors) {
    std::fprintf(stderr, "[tb][mbv2] vector index out of range (vec=%u nvec_in=%u nvec_out=%u)\n",
                 vec_idx, goldin.num_vectors, goldout.num_vectors);
    return 2;
  }

  Vnn2rtl_top* dut = new Vnn2rtl_top;

  // Aggregate results across the run.
  uint64_t total_mismatch = 0;
  int64_t  overall_first_mm_beat = -1;
  uint64_t total_in_seen = 0, total_out_seen = 0, total_out_expected = 0;
  bool any_timeout = false;
  bool all_pass = true;

  for (uint32_t v = first_vec; v < last_vec; ++v) {
    FrameStats st = runFrame(dut, goldin, goldout, v, max_cycles, (v == vec_idx) ? dump_path : std::string());
    total_in_seen      += st.input_beats_sent;
    total_out_seen     += st.output_beats_seen;
    total_out_expected += goldout.samples_per_vector;

    if (st.timed_out) {
      any_timeout = true; all_pass = false;
      std::printf("[tb][mbv2][vec%u] TIMEOUT/DEADLOCK at cycle cap %llu — "
                  "last_input_beat=%llu/%llu last_output_beat=%llu/%llu "
                  "s_axis_tready=%d m_axis_tvalid=%d\n",
                  v, (unsigned long long)max_cycles,
                  (unsigned long long)st.input_beats_sent, (unsigned long long)kInputBeats,
                  (unsigned long long)st.output_beats_seen, (unsigned long long)goldout.samples_per_vector,
                  (int)dut->s_axis_tready, (int)dut->m_axis_tvalid);
      std::fflush(stdout);
      // A deadlock on one vector poisons DUT state; stop the run.
      break;
    }

    total_mismatch += st.mismatch_bytes;
    // [SERIALIZED OUTPUT 2026-06-08] the DUT now emits the 1000-byte logit vector as
    // ceil(1000/32)=32 beats of 256b (was 1 beat of 8000b). The reassembled byte content
    // (st.mismatch_bytes) is the correctness gate; expect the serialized beat count, not 1.
    const uint64_t kExpBeats = (static_cast<uint64_t>(goldout.bytes_per_sample) * goldout.samples_per_vector + 31) / 32;
    const bool vec_pass = (st.mismatch_bytes == 0) && (st.output_beats_seen == kExpBeats);
    if (!vec_pass) all_pass = false;
    if (st.first_mm_beat >= 0) {
      if (overall_first_mm_beat < 0) overall_first_mm_beat = st.first_mm_beat;
      std::printf("[tb][mbv2][vec%u] FIRST MISMATCH: beat=%lld logit_byte=%lld expected=%d got=%d\n",
                  v, (long long)st.first_mm_beat, (long long)st.first_mm_byte,
                  st.first_mm_exp, st.first_mm_got);
    }
    std::printf("[tb][mbv2][vec%u] result=%s mismatch_bytes=%llu out_beats=%llu/%u\n",
                v, vec_pass ? "PASS" : "FAIL",
                (unsigned long long)st.mismatch_bytes,
                (unsigned long long)st.output_beats_seen, goldout.samples_per_vector);
    std::fflush(stdout);
  }

  dut->final();
  delete dut;

  // ---- single machine-parseable summary line (mirrors the ResNet TB) ----
  if (any_timeout) {
    std::printf("[tb][mbv2][summary] result=TIMEOUT mismatch_bytes=-1 first_mismatch_beat=-1 "
                "beats_seen=%llu/%llu in_beats=%llu/%llu\n",
                (unsigned long long)total_out_seen, (unsigned long long)total_out_expected,
                (unsigned long long)total_in_seen, (unsigned long long)kInputBeats);
    return 1;
  }

  std::printf("[tb][mbv2] total mismatching bytes = %llu / %llu\n",
              (unsigned long long)total_mismatch,
              (unsigned long long)(total_out_seen * goldout.bytes_per_sample));
  std::printf("[tb][mbv2][summary] result=%s mismatch_bytes=%llu first_mismatch_beat=%lld "
              "beats_seen=%llu/%llu in_beats=%llu/%llu\n",
              all_pass ? "PASS" : "FAIL",
              (unsigned long long)total_mismatch, (long long)overall_first_mm_beat,
              (unsigned long long)total_out_seen, (unsigned long long)total_out_expected,
              (unsigned long long)total_in_seen, (unsigned long long)kInputBeats);

  return all_pass ? 0 : 1;
}
