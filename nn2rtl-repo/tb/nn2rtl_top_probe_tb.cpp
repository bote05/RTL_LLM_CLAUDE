// Chain-probe testbench: drives conv_196 goldin vec0 through nn2rtl_top and
// captures the residual-add + stem output streams (via Verilator public vars,
// see tb/nn2rtl_top_probe.vlt) to localize the e2e all-zero bug. Compare each
// dumped probe_<id>.bin to that module's contract goldout in Python; the first
// checkpoint that diverges (or goes all-zero) localizes the integration bug.
//
// argv[1] = conv_196 contract .goldin   argv[2] = dump dir   argv[3] = vector (def 0)

#include <array>
#include <map>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Vnn2rtl_top.h"
#include "Vnn2rtl_top___024root.h"   // rootp internal-signal access
#include "verilated.h"
#include "probe_capture.inc"

double sc_time_stamp() { return 0; }

namespace {
constexpr uint64_t kMaxCycles  = 50'000'000;
constexpr int      kResetCycles = 16;
constexpr uint64_t kInputBeats = 224 * 224;
constexpr uint64_t kOutputBeats = 3136;

struct VF { uint32_t nv=0, spv=0, bps=0, wps=0; std::vector<std::vector<std::vector<uint32_t>>> v; };
VF loadVF(const std::string& p){
  std::ifstream in(p, std::ios::binary);
  if(!in) throw std::runtime_error("open "+p);
  char mg[4]; uint32_t ver=0,nv=0,spv=0,bps=0;
  in.read(mg,4); in.read((char*)&ver,4); in.read((char*)&nv,4); in.read((char*)&spv,4); in.read((char*)&bps,4);
  if(mg[0]!='N'||mg[1]!='N'||mg[2]!='2'||mg[3]!='V') throw std::runtime_error("magic");
  VF f; f.nv=nv; f.spv=spv; f.bps=bps; f.wps=(bps+3)/4; f.v.reserve(nv);
  std::vector<int32_t> row((size_t)spv*f.wps);
  for(uint32_t i=0;i<nv;i++){ in.read((char*)row.data(), (std::streamsize)row.size()*4);
    std::vector<std::vector<uint32_t>> s; s.reserve(spv);
    for(uint32_t k=0;k<spv;k++){ std::vector<uint32_t> w(f.wps);
      for(uint32_t q=0;q<f.wps;q++) w[q]=(uint32_t)row[(size_t)k*f.wps+q]; s.push_back(std::move(w)); }
    f.v.push_back(std::move(s)); }
  return f;
}
void tick(Vnn2rtl_top* d, uint64_t& c){ d->clk=1; d->eval(); d->clk=0; d->eval(); c++; }
}  // namespace

int main(int argc, char** argv){
  Verilated::commandArgs(argc, argv);
  if(argc<3){ std::fprintf(stderr,"usage: %s <goldin> <dumpdir> [vec]\n",argv[0]); return 2; }
  const std::string goldin=argv[1], dumpdir=argv[2];
  const uint32_t vec=(argc>=4)?(uint32_t)std::strtoul(argv[3],nullptr,10):0u;
  // argv[4] = dispatch index to capture (default 0). Lets us sweep dispatches
  // without a rebuild to localize which engine dispatch is fed/computes garbage.
  const uint32_t DISP=(argc>=5)?(uint32_t)std::strtoul(argv[4],nullptr,10):0u;
  std::fprintf(stderr,"[probe] capturing dispatch %u\n", DISP);
  VF gi=loadVF(goldin);
  if(gi.spv!=kInputBeats||gi.bps!=32){ std::fprintf(stderr,"bad goldin spv=%u bps=%u\n",gi.spv,gi.bps); return 2; }
  const auto& in_s=gi.v[vec];

  auto* dut=new Vnn2rtl_top;
  dut->clk=0; dut->rst_n=0; dut->s_axis_tvalid=0; dut->s_axis_tlast=0;
  memset(&dut->s_axis_tdata,0,sizeof(dut->s_axis_tdata));
  dut->m_axis_tready=1;
  dut->s_axil_awvalid=0; dut->s_axil_awaddr=0; dut->s_axil_wvalid=0; dut->s_axil_wdata=0;
  dut->s_axil_wstrb=0; dut->s_axil_bready=0; dut->s_axil_arvalid=0; dut->s_axil_araddr=0; dut->s_axil_rready=0;
  dut->eval();

  uint64_t cyc=0;
  for(int i=0;i<kResetCycles;i++) tick(dut,cyc);
  dut->rst_n=1; tick(dut,cyc);

  PROBE_DECLS
  std::vector<std::array<uint32_t,8>> cap_m_axis;
  // add_9 input taps: capture lhs(main=conv_262) + rhs(skip) at every add_9 transfer.
  // If each is individually clean (no 127) but add_9 OUT saturates -> main/skip MISALIGNMENT.
  std::vector<std::array<uint32_t,8>> cap_add9_lhs, cap_add9_rhs;
  // Bisection conv taps (single-stream, reliable like conv_196): multiset-compare to golden.
  std::vector<std::array<uint32_t,8>> cap_c198, cap_c212, cap_c244, cap_c284;
  std::vector<std::array<uint32_t,8>> cap_c200, cap_r3, cap_c206;  // stage-1 bracket of first residual block
  uint64_t in_sent=0, out_seen=0; bool done=false;

  // ---- DYNAMIC engine-read capture (dispatch 0 = conv_246) ----
  // Records the 2048-bit word the engine reads at each act_in address while
  // sched_dispatch_idx==0. Captured both SAME-cycle and 1-cycle-DELAYED to be
  // robust to BRAM read latency. Compare to conv_246.goldin offline: if neither
  // matches -> the engine reads wrong activations (loader/contention/timing);
  // if one matches -> engine reads correct data but mis-computes.
  #define R(v) (dut->rootp->nn2rtl_top__DOT__##v)
  std::map<uint32_t,std::array<uint32_t,64>> eng_same, eng_delayed;
  uint32_t prev_addr=0; bool prev_valid=false;
  // dispatch-0 config-write sequence (scheduler->engine AXI-Lite) + weight reads
  std::vector<std::pair<uint32_t,uint32_t>> cfg_writes;     // (awaddr, wdata)
  std::map<uint32_t,std::array<uint32_t,64>> wt_same, wt_d1, wt_d2;  // 3 read latencies
  uint32_t wa1=0,wa2=0; bool wv1=false,wv2=false;
  std::map<uint32_t,std::array<uint32_t,64>> eng_out;   // engine RAW output (act_out_wr) dispatch 0

  while(cyc<kMaxCycles && !done){
    if(in_sent<kInputBeats){ dut->s_axis_tvalid=1; dut->s_axis_tlast=(in_sent+1==kInputBeats)?1:0;
      for(int i=0;i<8;i++) dut->s_axis_tdata[i]=(i<(int)in_s[in_sent].size())?in_s[in_sent][i]:0u; }
    else { dut->s_axis_tvalid=0; dut->s_axis_tlast=0; }
    dut->eval();

    const bool in_acc = dut->s_axis_tvalid && dut->s_axis_tready;
    const bool out_acc = dut->m_axis_tvalid && dut->m_axis_tready;
    PROBE_CAPTURE(dut);

    // CONFIG-WRITE capture: gate on dispatch_idx==0 ONLY (config is written
    // BEFORE engine_busy goes high — the prior gate wrongly required engine_busy).
    if (R(sched_dispatch_idx)==DISP && R(sched_axil_wvalid) && R(sched_axil_wready))
      cfg_writes.push_back({(uint32_t)R(sched_axil_awaddr), (uint32_t)R(sched_axil_wdata)});

    // engine act_in + weight read capture during dispatch DISP compute
    if (R(sched_dispatch_idx)==DISP && R(engine_busy)) {
      std::array<uint32_t,64> w; for(int i=0;i<64;i++) w[i]=R(engine_act_in_rd_data)[i];
      if (R(engine_act_in_rd_en)) eng_same[R(engine_act_in_rd_addr)] = w;
      if (prev_valid) eng_delayed[prev_addr] = w;
      prev_valid = R(engine_act_in_rd_en); prev_addr = R(engine_act_in_rd_addr);
      // weight reads at 3 latencies (URAM read latency unknown; find the match)
      std::array<uint32_t,64> ww; for(int i=0;i<64;i++) ww[i]=R(engine_weight_rd_data)[i];
      if (R(engine_weight_rd_en)) wt_same[R(engine_weight_rd_addr)] = ww;
      if (wv1) wt_d1[wa1] = ww;
      if (wv2) wt_d2[wa2] = ww;
      wv2=wv1; wa2=wa1; wv1=R(engine_weight_rd_en); wa1=R(engine_weight_rd_addr);
      // engine RAW output write (pre-bridge): act_out_wr_data per pixel
      if (R(engine_act_out_wr_en)) {
        std::array<uint32_t,64> wo; for(int i=0;i<64;i++) wo[i]=R(engine_act_out_wr_data)[i];
        eng_out[R(engine_act_out_wr_addr)] = wo;
      }
    } else { prev_valid=false; wv1=false; wv2=false; }
    // add_9 transfer = its valid_in && ready_in (matches the wrapper's accept condition).
    if (R(node_conv_262_valid_out) && R(node_add_9_skip_valid) && R(spatial_run) && R(node_add_9_ready_in)) {
      std::array<uint32_t,8> wl, wr;
      for(int i=0;i<8;i++){ wl[i]=R(node_conv_262_data_out)[i]; wr[i]=R(node_add_9_skip_data)[i]; }
      cap_add9_lhs.push_back(wl); cap_add9_rhs.push_back(wr);
    }
    // Bisection conv captures (transfer = valid_out & spatial_run & downstream_skid_ready), same
    // reliable gate as conv_196. Single-stream -> clean multiset.
    #define CAPCONV(vec, vsig, rdy, dsig) \
      if (R(vsig) && R(spatial_run) && R(rdy)) { std::array<uint32_t,8> w; for(int _i=0;_i<8;_i++) w[_i]=R(dsig)[_i]; vec.push_back(w); }
    CAPCONV(cap_c198, node_conv_198_valid_out, skid_node_relu_1_ready,  node_conv_198_data_out)
    CAPCONV(cap_c212, node_conv_212_valid_out, skid_node_relu_7_ready,  node_conv_212_data_out)
    CAPCONV(cap_c244, node_conv_244_valid_out, skid_node_relu_22_ready, node_conv_244_data_out)
    CAPCONV(cap_c284, node_conv_284_valid_out, skid_node_relu_41_ready, node_conv_284_data_out)
    CAPCONV(cap_c200, node_conv_200_valid_out, skid_node_relu_2_ready,  node_conv_200_data_out)
    CAPCONV(cap_r3,   node_relu_3_valid_out,   node_relu_3_ready_out_combined, node_relu_3_data_out)
    CAPCONV(cap_c206, node_conv_206_valid_out, skid_node_relu_4_ready,  node_conv_206_data_out)
    if(out_acc){ std::array<uint32_t,8> w; for(int i=0;i<8;i++) w[i]=dut->m_axis_tdata[i]; cap_m_axis.push_back(w);
      out_seen++; if(out_acc && dut->m_axis_tlast) done=true; }
    if(in_acc) in_sent++;
    if((cyc&0xFFFFF)==0 && cyc>0) std::printf("[probe] cyc=%llu in=%llu out=%llu\n",
        (unsigned long long)cyc,(unsigned long long)in_sent,(unsigned long long)out_seen), std::fflush(stdout);
    tick(dut,cyc);
  }
  dut->final();
  std::printf("[probe] frame done=%d cyc=%llu in=%llu out=%llu\n",(int)done,
      (unsigned long long)cyc,(unsigned long long)in_sent,(unsigned long long)out_seen);
  PROBE_DUMP(dumpdir.c_str());
  { std::string p=dumpdir+"/probe_node_relu_48.bin"; std::ofstream o(p,std::ios::binary);
    for(auto&w:cap_m_axis) o.write((const char*)w.data(),32);
    std::printf("[probe] node_relu_48 (m_axis): %zu beats -> %s\n", cap_m_axis.size(), p.c_str()); }
  { std::string p=dumpdir+"/probe_add9_lhs.bin"; std::ofstream o(p,std::ios::binary);
    for(auto&w:cap_add9_lhs) o.write((const char*)w.data(),32);
    std::printf("[probe] add9_lhs (conv_262 main): %zu beats -> %s\n", cap_add9_lhs.size(), p.c_str()); }
  { std::string p=dumpdir+"/probe_add9_rhs.bin"; std::ofstream o(p,std::ios::binary);
    for(auto&w:cap_add9_rhs) o.write((const char*)w.data(),32);
    std::printf("[probe] add9_rhs (skip): %zu beats -> %s\n", cap_add9_rhs.size(), p.c_str()); }
  { std::string p=dumpdir+"/probe_node_conv_198.bin"; std::ofstream o(p,std::ios::binary); for(auto&w:cap_c198) o.write((const char*)w.data(),32); std::printf("[probe] conv_198: %zu beats -> %s\n", cap_c198.size(), p.c_str()); }
  { std::string p=dumpdir+"/probe_node_conv_212.bin"; std::ofstream o(p,std::ios::binary); for(auto&w:cap_c212) o.write((const char*)w.data(),32); std::printf("[probe] conv_212: %zu beats -> %s\n", cap_c212.size(), p.c_str()); }
  { std::string p=dumpdir+"/probe_node_conv_244.bin"; std::ofstream o(p,std::ios::binary); for(auto&w:cap_c244) o.write((const char*)w.data(),32); std::printf("[probe] conv_244: %zu beats -> %s\n", cap_c244.size(), p.c_str()); }
  { std::string p=dumpdir+"/probe_node_conv_284.bin"; std::ofstream o(p,std::ios::binary); for(auto&w:cap_c284) o.write((const char*)w.data(),32); std::printf("[probe] conv_284: %zu beats -> %s\n", cap_c284.size(), p.c_str()); }
  { std::string p=dumpdir+"/probe_node_conv_200.bin"; std::ofstream o(p,std::ios::binary); for(auto&w:cap_c200) o.write((const char*)w.data(),32); std::printf("[probe] conv_200: %zu beats -> %s\n", cap_c200.size(), p.c_str()); }
  { std::string p=dumpdir+"/probe_node_relu_3.bin"; std::ofstream o(p,std::ios::binary); for(auto&w:cap_r3) o.write((const char*)w.data(),32); std::printf("[probe] relu_3: %zu beats -> %s\n", cap_r3.size(), p.c_str()); }
  { std::string p=dumpdir+"/probe_node_conv_206.bin"; std::ofstream o(p,std::ios::binary); for(auto&w:cap_c206) o.write((const char*)w.data(),32); std::printf("[probe] conv_206: %zu beats -> %s\n", cap_c206.size(), p.c_str()); }
  // dump engine act_in reads (addr:uint32 + 64 words each) for both latencies
  auto dump_reads=[&](const std::map<uint32_t,std::array<uint32_t,64>>& m, const char* nm){
    std::string p=dumpdir+"/engreads_"+nm+".bin"; std::ofstream o(p,std::ios::binary);
    for(auto&kv:m){ uint32_t a=kv.first; o.write((const char*)&a,4); o.write((const char*)kv.second.data(),256); }
    std::printf("[probe] engreads_%s: %zu unique addrs -> %s\n", nm, m.size(), p.c_str());
  };
  dump_reads(eng_same,"same"); dump_reads(eng_delayed,"delayed");
  dump_reads(wt_same,"weights_same"); dump_reads(wt_d1,"weights_d1"); dump_reads(wt_d2,"weights_d2");
  dump_reads(eng_out,"engout");   // engine RAW output (pre-bridge), addr=act_out_base+pixel
  { std::string p=dumpdir+"/cfg_writes_d0.txt"; std::ofstream o(p);
    for(auto&kv:cfg_writes) o<<"0x"<<std::hex<<kv.first<<" 0x"<<kv.second<<"\n";
    std::printf("[probe] cfg_writes_d0: %zu writes -> %s\n", cfg_writes.size(), p.c_str()); }
  delete dut;
  return done?0:1;
}
