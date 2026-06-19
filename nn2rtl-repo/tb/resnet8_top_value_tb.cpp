// End-to-end VALUE verification testbench for the ResNet-8 all-spatial
// nn2rtl_top.v (output/resnet8/rtl/nn2rtl_top.v).
//
// Cloned from tb/mbv2_top_value_tb.cpp, adapted to ResNet-8's ABI:
//   - input  : node_conv2d.goldin — nvec=8, samples_per_vec=1024, bps=3.
//              Each 3-byte sample = one 24-bit s_axis beat (one RGB pixel).
//              1024 = 32*32 = the CIFAR image, fed straight into s_axis.
//   - output : node_linear.goldout — nvec=8, samples_per_vec=1, bps=10.
//              The ResNet-8 head emits all 10 INT8 logits in ONE 80-bit
//              m_axis beat (m_axis_tdata[79:0], m_axis_tlast=1 immediately),
//              so there is NO 256b serialization (unlike MBV2's 1000-logit
//              head). We capture the single beat (3 words = 80 bits rounded up
//              to 96) and compare the low 10 bytes to the goldout.
//
// If every module's goldout == the next module's goldin (consistent ABIs), the
// e2e output MUST equal node_linear's goldout. A mismatch localizes an
// integration (inter-module wiring / handshake) bug per-module testing cannot
// see.
//
// NN2V vector file layout — identical to the MBV2 TB (see that file's header).
//
// Run-time contract:
//   argv[1] = node_conv2d.goldin (24-bit / 3-byte samples, 1024/vector)
//   argv[2] = node_linear.goldout (10-byte / 80-bit samples, 1/vector)
//   argv[3] = vector index (default 0)
//   argv[4] = (optional) raw m_axis capture dump path
//   env RESNET8_VEC        : overrides vector index (argv[3] wins)
//   env RESNET8_ALL_VECS=1 : run every vector
//   env RESNET8_MAX_CYCLES : per-frame fail-fast cap (default 10,000,000;
//                            the ResNet-8 all-spatial frame is well under 1M
//                            cycles, so this is a generous deadlock guard).
//
// Build: see scripts/run_resnet8_top_value.ts.

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
#ifdef RESNET8_PROBE
#include "Vnn2rtl_top___024root.h"
#endif
#include "verilated.h"

double sc_time_stamp() { return 0; }

namespace {

constexpr int      kResetCycles = 16;
constexpr uint64_t kInputBeats  = 32ULL * 32ULL;   // 1024 — one 24-bit RGB pixel beat each
// ResNet-8 all-spatial frame is small (<1M cycles). 10M is a generous guard
// that still trips fast on a handshake deadlock.
constexpr uint64_t kDefaultMaxCycles = 10'000'000;

// ---- NN2V binary vector file (.goldin / .goldout) ----
struct VectorFile {
  uint32_t num_vectors = 0;
  uint32_t samples_per_vector = 0;
  uint32_t bytes_per_sample = 0;
  uint32_t words_per_sample = 0;
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

// node_linear data_out is 80 bits => 3 u32 words hold all 10 logit bytes.
constexpr uint32_t kOutWords = 3;

FrameStats runFrame(Vnn2rtl_top* dut, const VectorFile& goldin, const VectorFile& goldout,
                    uint32_t vec_idx, uint64_t max_cycles, const std::string& dump_path) {
  FrameStats st;
  const auto& in_samples  = goldin.vectors[vec_idx];
  const auto& out_samples = goldout.vectors[vec_idx];

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
  std::printf("[tb][resnet8][vec%u] reset deasserted at cycle %llu\n",
              vec_idx, (unsigned long long)cycle);

  std::vector<std::vector<uint32_t>> captured;

  uint64_t first_input_cycle = 0, last_output_cycle = 0;
  bool first_input_seen = false, done = false;

#ifdef RESNET8_PROBE
  // cumulative valid_out counts for each conv/add (public_flat_rd wires)
  uint64_t p_stem=0,p1=0,p2=0,p25=0,p4=0,p5=0,p56=0,p7=0,p8=0,p87=0;
  // per-layer first/last active cycle (to reveal the schedule timeline + overlap)
  static const char* kLayerNames[10] =
    {"stem","conv1","conv2","add25","conv4","conv5","add56","conv7","conv8","add87"};
  uint64_t pcnt[10]={0}, pfirst[10]={0}, plast[10]={0};
#endif

  // [FRAME-PIPE TEST] RESNET8_FRAMES=N feeds N identical frames back-to-back to
  // measure SUSTAINED throughput (inter-frame interval) vs single-frame latency.
  const int frames_n = []{ const char* e = std::getenv("RESNET8_FRAMES");
                           int n = e ? std::atoi(e) : 1; return n < 1 ? 1 : n; }();
  const uint64_t total_in_beats = (uint64_t)frames_n * kInputBeats;
  std::vector<uint64_t> out_cycles;
  while (cycle < max_cycles && !done) {
    if (st.input_beats_sent < total_in_beats) {
      dut->s_axis_tvalid = 1;
      dut->s_axis_tlast  = (((st.input_beats_sent + 1) % kInputBeats) == 0) ? 1 : 0;
      assignWords(dut->s_axis_tdata, in_samples[st.input_beats_sent % kInputBeats]);
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
      // ResNet-8 head: all 10 logits arrive in ONE 80-bit beat (no serialization).
      auto ow = readWords(dut->m_axis_tdata, kOutWords);
      if (st.output_beats_seen < 1) {
        std::printf("[dbg-out] beat %llu logit[0..9]=%d %d %d %d %d %d %d %d %d %d\n",
                    (unsigned long long)st.output_beats_seen,
                    (int8_t)(ow[0] & 0xFF), (int8_t)((ow[0] >> 8) & 0xFF),
                    (int8_t)((ow[0] >> 16) & 0xFF), (int8_t)((ow[0] >> 24) & 0xFF),
                    (int8_t)(ow[1] & 0xFF), (int8_t)((ow[1] >> 8) & 0xFF),
                    (int8_t)((ow[1] >> 16) & 0xFF), (int8_t)((ow[1] >> 24) & 0xFF),
                    (int8_t)(ow[2] & 0xFF), (int8_t)((ow[2] >> 8) & 0xFF));
      }
      captured.push_back(std::move(ow));
      out_cycles.push_back(cycle);
      st.output_beats_seen++;
      last_output_cycle = cycle;
      if (st.output_beats_seen >= (uint64_t)frames_n && (output_last || frames_n > 1)) {
        done = true;
      }
    }

#ifdef RESNET8_PROBE
    {
      const bool vo[10] = {
        (bool)dut->rootp->nn2rtl_top__DOT__node_conv2d_valid_out,
        (bool)dut->rootp->nn2rtl_top__DOT__node_conv2d_1_valid_out,
        (bool)dut->rootp->nn2rtl_top__DOT__node_conv2d_2_valid_out,
        (bool)dut->rootp->nn2rtl_top__DOT__node_add_25_valid_out,
        (bool)dut->rootp->nn2rtl_top__DOT__node_conv2d_4_valid_out,
        (bool)dut->rootp->nn2rtl_top__DOT__node_conv2d_5_valid_out,
        (bool)dut->rootp->nn2rtl_top__DOT__node_add_56_valid_out,
        (bool)dut->rootp->nn2rtl_top__DOT__node_conv2d_7_valid_out,
        (bool)dut->rootp->nn2rtl_top__DOT__node_conv2d_8_valid_out,
        (bool)dut->rootp->nn2rtl_top__DOT__node_add_87_valid_out};
      for (int li = 0; li < 10; ++li) {
        if (vo[li]) {
          if (pcnt[li] == 0) pfirst[li] = cycle;
          plast[li] = cycle;
          pcnt[li]++;
        }
      }
    }
    p_stem=pcnt[0];p1=pcnt[1];p2=pcnt[2];p25=pcnt[3];p4=pcnt[4];
    p5=pcnt[5];p56=pcnt[6];p7=pcnt[7];p8=pcnt[8];p87=pcnt[9];
#endif

    if ((cycle & 0x3FFFF) == 0 && cycle > 0) {
#ifdef RESNET8_PROBE
      std::printf("[probe] cyc=%llu stem=%llu c1=%llu c2=%llu a25=%llu c4=%llu c5=%llu a56=%llu c7=%llu c8=%llu a87=%llu\n",
                  (unsigned long long)cycle,(unsigned long long)p_stem,(unsigned long long)p1,(unsigned long long)p2,
                  (unsigned long long)p25,(unsigned long long)p4,(unsigned long long)p5,(unsigned long long)p56,
                  (unsigned long long)p7,(unsigned long long)p8,(unsigned long long)p87);
#endif
      std::printf("[tb][resnet8][vec%u] cycle=%llu in=%llu/%llu out=%llu tready=%d tvalid=%d\n",
                  vec_idx, (unsigned long long)cycle,
                  (unsigned long long)st.input_beats_sent, (unsigned long long)kInputBeats,
                  (unsigned long long)st.output_beats_seen,
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
  std::printf("[tb][resnet8][vec%u] frame done: first_input_cycle=%llu last_output_cycle=%llu "
              "e2e_cycles=%llu out_beats=%llu\n",
              vec_idx, (unsigned long long)first_input_cycle, (unsigned long long)last_output_cycle,
              (unsigned long long)st.e2e_cycles, (unsigned long long)st.output_beats_seen);
  if (frames_n > 1 && out_cycles.size() >= 2) {
    std::printf("[frame-pipe] %d frames, output cycles:", frames_n);
    for (size_t fi = 0; fi < out_cycles.size(); ++fi)
      std::printf(" f%zu=%llu", fi, (unsigned long long)out_cycles[fi]);
    std::printf("\n[frame-pipe] inter-frame II (steady-state throughput cycles):");
    for (size_t fi = 1; fi < out_cycles.size(); ++fi)
      std::printf(" %llu", (unsigned long long)(out_cycles[fi] - out_cycles[fi-1]));
    std::printf("\n[frame-pipe] => single-frame LATENCY=%llu ; sustained II=%llu (last gap)\n",
                (unsigned long long)st.e2e_cycles,
                (unsigned long long)(out_cycles.back() - out_cycles[out_cycles.size()-2]));
  }

#ifdef RESNET8_PROBE
  std::printf("[probe] per-layer timeline (active=valid_out cycles; span=last-first):\n");
  std::printf("[probe] %-7s %10s %10s %10s %10s\n","layer","active","first","last","span");
  for (int li = 0; li < 10; ++li)
    std::printf("[probe] %-7s %10llu %10llu %10llu %10llu\n",
                kLayerNames[li],(unsigned long long)pcnt[li],
                (unsigned long long)pfirst[li],(unsigned long long)plast[li],
                (unsigned long long)(plast[li] >= pfirst[li] ? plast[li]-pfirst[li] : 0));
  std::fflush(stdout);
#endif

  if (!dump_path.empty()) {
    std::ofstream df(dump_path, std::ios::binary);
    if (df.is_open()) {
      for (const auto& beat : captured)
        for (uint32_t w = 0; w < kOutWords; ++w) {
          const uint32_t word = (w < beat.size()) ? beat[w] : 0U;
          df.write(reinterpret_cast<const char*>(&word), 4);
        }
      std::printf("[tb][resnet8][vec%u] dumped %zu beats to %s\n",
                  vec_idx, captured.size(), dump_path.c_str());
    }
  }

  // ---- byte-exact compare: the single captured beat's low `bps`(=10) bytes vs goldout ----
  const uint32_t bps = goldout.bytes_per_sample;   // 10
  std::vector<uint8_t> got_bytes;
  if (!captured.empty()) {
    const auto& beat = captured[0];
    for (uint32_t w = 0; w < beat.size(); ++w)
      for (int k = 0; k < 4; ++k)
        got_bytes.push_back(static_cast<uint8_t>((beat[w] >> (8U * k)) & 0xFF));
  }
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
    std::fprintf(stderr, "usage: %s <node_conv2d.goldin> <node_linear.goldout> [vector_idx] [dump_path]\n",
                 argv[0]);
    return 2;
  }
  const std::string goldin_path  = argv[1];
  const std::string goldout_path = argv[2];

  uint32_t vec_idx = 0U;
  if (const char* e = std::getenv("RESNET8_VEC")) vec_idx = static_cast<uint32_t>(std::strtoul(e, nullptr, 10));
  if (argc >= 4) vec_idx = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
  const std::string dump_path = (argc >= 5) ? argv[4] : std::string();

  const bool all_vecs = []() { const char* e = std::getenv("RESNET8_ALL_VECS"); return e && e[0] == '1'; }();

  uint64_t max_cycles = kDefaultMaxCycles;
  if (const char* e = std::getenv("RESNET8_MAX_CYCLES")) {
    const uint64_t v = std::strtoull(e, nullptr, 10);
    if (v > 0) max_cycles = v;
  }

  VectorFile goldin, goldout;
  try {
    goldin  = loadVectorFile(goldin_path);
    goldout = loadVectorFile(goldout_path);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[tb][resnet8] golden load error: %s\n", e.what());
    return 2;
  }

  std::printf("[tb][resnet8] goldin  : vectors=%u samples=%u bytes/sample=%u words/sample=%u\n",
              goldin.num_vectors, goldin.samples_per_vector, goldin.bytes_per_sample, goldin.words_per_sample);
  std::printf("[tb][resnet8] goldout : vectors=%u samples=%u bytes/sample=%u words/sample=%u\n",
              goldout.num_vectors, goldout.samples_per_vector, goldout.bytes_per_sample, goldout.words_per_sample);
  std::printf("[tb][resnet8] max_cycles=%llu input_beats=%llu all_vecs=%d\n",
              (unsigned long long)max_cycles, (unsigned long long)kInputBeats, (int)all_vecs);

  // ABI sanity — the whole test rests on these matching the top's ports.
  if (goldin.samples_per_vector != kInputBeats) {
    std::fprintf(stderr, "[tb][resnet8] goldin samples %u != expected input beats %llu\n",
                 goldin.samples_per_vector, (unsigned long long)kInputBeats);
    return 2;
  }
  if (goldin.bytes_per_sample != 3) {
    std::fprintf(stderr, "[tb][resnet8] goldin bytes/sample %u != 3 (expected 24-bit RGB pixels)\n",
                 goldin.bytes_per_sample);
    return 2;
  }
  if (goldout.bytes_per_sample != 10) {
    std::fprintf(stderr, "[tb][resnet8] goldout bytes/sample %u != 10 (expected 10 INT8 logits)\n",
                 goldout.bytes_per_sample);
    return 2;
  }
  if (goldout.samples_per_vector != 1) {
    std::fprintf(stderr, "[tb][resnet8] goldout samples/vector %u != 1 (expected one 80-bit beat)\n",
                 goldout.samples_per_vector);
    return 2;
  }

  const uint32_t first_vec = all_vecs ? 0U : vec_idx;
  const uint32_t last_vec  = all_vecs ? goldin.num_vectors : (vec_idx + 1U);
  if (first_vec >= goldin.num_vectors || last_vec - 1U >= goldin.num_vectors ||
      vec_idx >= goldout.num_vectors) {
    std::fprintf(stderr, "[tb][resnet8] vector index out of range (vec=%u nvec_in=%u nvec_out=%u)\n",
                 vec_idx, goldin.num_vectors, goldout.num_vectors);
    return 2;
  }

  Vnn2rtl_top* dut = new Vnn2rtl_top;

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
      std::printf("[tb][resnet8][vec%u] TIMEOUT/DEADLOCK at cycle cap %llu — "
                  "last_input_beat=%llu/%llu last_output_beat=%llu "
                  "s_axis_tready=%d m_axis_tvalid=%d\n",
                  v, (unsigned long long)max_cycles,
                  (unsigned long long)st.input_beats_sent, (unsigned long long)kInputBeats,
                  (unsigned long long)st.output_beats_seen,
                  (int)dut->s_axis_tready, (int)dut->m_axis_tvalid);
      std::fflush(stdout);
      break;
    }

    total_mismatch += st.mismatch_bytes;
    const bool vec_pass = (st.mismatch_bytes == 0) && (st.output_beats_seen == 1);
    if (!vec_pass) all_pass = false;
    if (st.first_mm_beat >= 0) {
      if (overall_first_mm_beat < 0) overall_first_mm_beat = st.first_mm_beat;
      std::printf("[tb][resnet8][vec%u] FIRST MISMATCH: beat=%lld logit_byte=%lld expected=%d got=%d\n",
                  v, (long long)st.first_mm_beat, (long long)st.first_mm_byte,
                  st.first_mm_exp, st.first_mm_got);
    }
    std::printf("[tb][resnet8][vec%u] result=%s mismatch_bytes=%llu out_beats=%llu e2e_cycles=%llu\n",
                v, vec_pass ? "PASS" : "FAIL",
                (unsigned long long)st.mismatch_bytes,
                (unsigned long long)st.output_beats_seen,
                (unsigned long long)st.e2e_cycles);
    std::fflush(stdout);
  }

  dut->final();
  delete dut;

  if (any_timeout) {
    std::printf("[tb][resnet8][summary] result=TIMEOUT mismatch_bytes=-1 first_mismatch_beat=-1 "
                "beats_seen=%llu/%llu in_beats=%llu/%llu\n",
                (unsigned long long)total_out_seen, (unsigned long long)total_out_expected,
                (unsigned long long)total_in_seen, (unsigned long long)kInputBeats);
    return 1;
  }

  std::printf("[tb][resnet8] total mismatching bytes = %llu / %llu\n",
              (unsigned long long)total_mismatch,
              (unsigned long long)(total_out_seen * goldout.bytes_per_sample));
  std::printf("[tb][resnet8][summary] result=%s mismatch_bytes=%llu first_mismatch_beat=%lld "
              "beats_seen=%llu/%llu in_beats=%llu/%llu\n",
              all_pass ? "PASS" : "FAIL",
              (unsigned long long)total_mismatch, (long long)overall_first_mm_beat,
              (unsigned long long)total_out_seen, (unsigned long long)total_out_expected,
              (unsigned long long)total_in_seen, (unsigned long long)kInputBeats);

  return all_pass ? 0 : 1;
}
