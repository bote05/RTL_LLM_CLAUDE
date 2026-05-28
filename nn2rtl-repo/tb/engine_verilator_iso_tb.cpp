// Cycle-driven Verilator harness for shared_engine ISOLATION (conv_246).
// De-confounds the engine in-chain ±1: the 3 byte-exact sims (iverilog, XSim
// behavioral, XSim gates) were all ISOLATION; the only FAIL (Verilator e2e) was
// IN-CHAIN. The prior Verilator-isolation attempt used the iverilog .v TB and
// STALLED under --timing. This pure cycle-driven C++ harness (no Verilog TB
// timing constructs) drives shared_engine directly and models the act/weight/
// bias memories in C++ with the SAME 1-cycle registered-read latency as
// engine_one_layer_tb's behavioral modules (which the engine is byte-exact
// against under iverilog/XSim). Result:
//   byte-exact  -> Verilator-isolation matches the others => in-chain ±1 is
//                  in-chain-specific (real hazard / latency / shared-mem), NOT
//                  a generic Verilator artifact.
//   ±1          -> genuinely Verilator-specific in isolation.
//
// argv[1]=conv_246 INT8 .goldin  argv[2]=weights dir  argv[3]=bias.mem
// argv[4]=conv_246 INT8 .goldout
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "Vshared_engine.h"
#include "verilated.h"

double sc_time_stamp() { return 0; }

namespace {
// conv_246 dispatch config (from build_engine_one_layer_tb/dispatch_cfg.vh)
constexpr uint32_t IC=256, OC=256, IH=28, IW=28, OH=14, OW=14;
constexpr uint32_t WEIGHT_BASE=11155, BIAS_BASE=31;
constexpr uint32_t SCALE_MULT=1284434803u, SCALE_SHIFT=39, ZP=0;
constexpr uint32_t ACT_IN_BASE=8192, ACT_OUT_BASE=4096;
constexpr uint32_t N_IN_PIXELS=784, N_OUT_PIXELS=196;
constexpr uint64_t kMaxCycles=60'000'000;

// parse a hex line into LE uint32 words (word0 = least-significant 8 hex chars)
void hex_to_words(const std::string& h, uint32_t* out, int nwords) {
  for (int i=0;i<nwords;i++) out[i]=0;
  int len=(int)h.size();
  for (int w=0; w<nwords; w++) {
    int hi=len-w*8;                 // exclusive end for this word's 8 hex chars
    int lo=hi-8;
    if (hi<=0) break;
    if (lo<0) lo=0;
    out[w]=(uint32_t)strtoul(h.substr(lo,hi-lo).c_str(),nullptr,16);
  }
}
}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  if (argc<5){ std::fprintf(stderr,"usage: %s goldin wdir bias goldout\n",argv[0]); return 2; }
  const std::string goldin=argv[1], wdir=argv[2], biasf=argv[3], goldout=argv[4];

  // ---- memories ----
  std::vector<std::array<uint32_t,64>> act(24576);   // 2048b words, 1-cyc RW
  for(auto&w:act) w.fill(0);
  std::vector<std::array<uint32_t,64>> wmem;          // combined 8-bank low256 = 2048b
  std::vector<std::array<uint32_t,256>> bmem;         // 8192b bias words

  // load 8 weight banks -> combined wmem[addr] = {bank7..bank0 low256}
  {
    std::vector<std::vector<std::array<uint32_t,8>>> banks(8);
    size_t depth=0;
    for(int b=0;b<8;b++){
      std::ifstream f(wdir+"/uram_weights_bank"+std::to_string(b)+".mem");
      if(!f){ std::fprintf(stderr,"missing bank %d\n",b); return 2; }
      std::string ln; std::vector<std::array<uint32_t,8>> rows;
      while(std::getline(f,ln)){ if(ln.empty()) continue;
        std::array<uint32_t,8> wd; hex_to_words(ln,wd.data(),8); rows.push_back(wd); }
      banks[b]=std::move(rows); depth=std::max(depth,banks[b].size());
    }
    wmem.resize(depth); for(auto&w:wmem) w.fill(0);
    for(size_t a=0;a<depth;a++) for(int b=0;b<8;b++) if(a<banks[b].size())
      for(int k=0;k<8;k++) wmem[a][b*8+k]=banks[b][a][k];
    std::printf("[iso] loaded weights depth=%zu\n",depth);
  }
  // load bias
  {
    std::ifstream f(biasf); if(!f){ std::fprintf(stderr,"missing bias\n"); return 2; }
    std::string ln; while(std::getline(f,ln)){ if(ln.empty()) continue;
      std::array<uint32_t,256> wd; hex_to_words(ln,wd.data(),256); bmem.push_back(wd); }
    std::printf("[iso] loaded bias entries=%zu\n",bmem.size());
  }
  // preload activations from goldin vec0 (256 bytes/pixel -> 2048b word, ch0=byte0)
  {
    std::ifstream f(goldin,std::ios::binary); if(!f){ std::fprintf(stderr,"missing goldin\n"); return 2; }
    char hdr[20]; f.read(hdr,20);
    for(uint32_t p=0;p<N_IN_PIXELS;p++){
      unsigned char buf[256]; f.read((char*)buf,256);
      std::array<uint32_t,64> word; word.fill(0);
      for(int by=0;by<256;by++) word[by/4] |= ((uint32_t)buf[by])<<((by%4)*8);
      act[ACT_IN_BASE+p]=word;
    }
    std::printf("[iso] preloaded %u activation pixels\n",N_IN_PIXELS);
  }

  // Weight read latency: 1 (isolation TB / engine-sweep) or 2 (in-chain xpm URAM).
  // De-confound: if WLAT=2 reproduces the ±1, the in-chain bug is a weight-latency
  // mismatch (engine pipelined for 1-cyc, deployed URAM is 2-cyc).
  int WLAT = 1;
  if (const char* e=getenv("WLAT")) WLAT=atoi(e);
  std::printf("[iso] weight read latency = %d cycle(s)\n", WLAT);

  auto* dut=new Vshared_engine;
  // registered-read holding regs (value latched at previous posedge)
  std::array<uint32_t,64> act_reg{}, w_reg{}, w_reg2{}; std::array<uint32_t,256> b_reg{};
  act_reg.fill(0); w_reg.fill(0); w_reg2.fill(0); b_reg.fill(0);
  uint64_t cyc=0;

  auto tick=[&](){
    // 1) present registered read results (prev-edge values); weights at WLAT latency
    const auto& wpresent = (WLAT>=2) ? w_reg2 : w_reg;
    for(int i=0;i<64;i++) dut->act_in_rd_data[i]=act_reg[i];
    for(int i=0;i<64;i++) dut->weight_rd_data[i]=wpresent[i];
    for(int i=0;i<256;i++) dut->bias_rd_data[i]=b_reg[i];
    // 2) clk low: settle combinational, engine presents addr/en + writes
    dut->clk=0; dut->eval();
    bool are=dut->act_in_rd_en; uint32_t ara=dut->act_in_rd_addr & 0x7FFF;
    bool we=dut->act_out_wr_en; uint32_t wa=dut->act_out_wr_addr & 0x7FFF;
    bool wre=dut->weight_rd_en; uint32_t wra=dut->weight_rd_addr & 0x1FFFF; // [16:0]
    bool bre=dut->bias_rd_en;   uint32_t bra=dut->bias_rd_addr & 0xFF;
    std::array<uint32_t,64> wrd; for(int i=0;i<64;i++) wrd[i]=dut->act_out_wr_data[i];
    // 3) compute upcoming registered values
    std::array<uint32_t,64> n_act=act_reg, n_w=w_reg; std::array<uint32_t,256> n_b=b_reg;
    if(are && ara<act.size()) n_act=act[ara];
    if(wre && wra<wmem.size()) n_w=wmem[wra];
    if(bre && bra<bmem.size()) n_b=bmem[bra];
    // 4) clk high: posedge (engine consumes rd_data = *_reg)
    dut->clk=1; dut->eval();
    // 5) commit posedge effects (weights pipelined: w_reg2 <= w_reg <= mem)
    act_reg=n_act; w_reg2=w_reg; w_reg=n_w; b_reg=n_b;
    if(we && wa<act.size()) act[wa]=wrd;
    cyc++;
  };

  // ---- reset ----
  dut->clk=0; dut->rst_n=0; dut->engine_start=0;
  dut->s_axil_awvalid=0; dut->s_axil_awaddr=0; dut->s_axil_wvalid=0; dut->s_axil_wdata=0;
  dut->s_axil_wstrb=0; dut->s_axil_bready=0; dut->s_axil_arvalid=0; dut->s_axil_araddr=0; dut->s_axil_rready=0;
  dut->eval();
  for(int i=0;i<20;i++) tick();
  dut->rst_n=1; for(int i=0;i<5;i++) tick();

  // ---- AXI-Lite config writes ----
  auto axi_write=[&](uint32_t addr,uint32_t data){
    dut->s_axil_awvalid=1; dut->s_axil_awaddr=addr;
    dut->s_axil_wvalid=1;  dut->s_axil_wdata=data; dut->s_axil_wstrb=0xF; dut->s_axil_bready=1;
    // wait until both awready & wready observed (sample at clk-low inside tick via re-eval)
    for(int g=0; g<1000; g++){
      tick();
      if(dut->s_axil_awready && dut->s_axil_wready) break;
    }
    dut->s_axil_awvalid=0; dut->s_axil_wvalid=0;
    for(int g=0; g<1000; g++){ tick(); if(dut->s_axil_bvalid) break; }
    dut->s_axil_bready=0; tick();
  };
  axi_write(0x00, IC);
  axi_write(0x04, OC);
  axi_write(0x08, (0u<<7)|(3u<<4)|(0u<<3)|3u);      // {KH=3,KW=3}
  axi_write(0x0C, (2u<<3)|2u);                       // {SH,SW}
  axi_write(0x10, (1u<<3)|1u);                       // {PH,PW}
  axi_write(0x14, (IH<<16)|IW);
  axi_write(0x18, (OH<<16)|OW);
  axi_write(0x1C, WEIGHT_BASE);
  axi_write(0x20, BIAS_BASE);
  axi_write(0x24, SCALE_MULT);
  axi_write(0x28, (ZP<<6)|SCALE_SHIFT);
  axi_write(0x34, ACT_IN_BASE);
  axi_write(0x38, ACT_OUT_BASE);
  std::printf("[iso] config written at cyc=%llu\n",(unsigned long long)cyc);

  for(int i=0;i<4;i++) tick();
  dut->engine_start=1; tick(); dut->engine_start=0;
  uint64_t start=cyc;
  bool done=false;
  while(cyc<kMaxCycles){
    tick();
    if(dut->engine_done){ done=true; break; }
    if(((cyc-start)&0x3FFFF)==0) { std::printf("[iso] cyc=%llu busy=%d\n",
        (unsigned long long)(cyc-start),(int)dut->engine_busy); std::fflush(stdout); }
  }
  std::printf("[iso] done=%d at cyc=%llu (took %llu)\n",(int)done,
      (unsigned long long)cyc,(unsigned long long)(cyc-start));
  for(int i=0;i<8;i++) tick();  // drain final writes
  dut->final();
  if(!done){ std::printf("[iso] FAIL: engine never completed\n"); return 1; }

  // ---- compare act_out region to goldout vec0 ----
  std::ifstream gf(goldout,std::ios::binary);
  char hdr[20]; gf.read(hdr,20);
  int total=0, mism=0, maxe=0; int firstbad=-1;
  for(uint32_t p=0;p<N_OUT_PIXELS;p++){
    unsigned char g[256]; gf.read((char*)g,256);
    const auto& word=act[ACT_OUT_BASE+p];
    for(int by=0;by<256;by++){
      uint8_t got=(word[by/4]>>((by%4)*8))&0xFF;
      total++;
      if(got!=g[by]){ mism++;
        int gs=(g[by]>=128)?g[by]-256:g[by], os=(got>=128)?got-256:got;
        if(abs(gs-os)>maxe) maxe=abs(gs-os);
        if(firstbad<0) firstbad=(int)p;
      }
    }
  }
  std::printf("\n=== VERILATOR engine-ISOLATION (conv_246) vs INT8 goldout ===\n");
  std::printf("bytes=%d mismatch=%d max|err|=%d firstbad_pixel=%d\n",total,mism,maxe,firstbad);
  std::printf("%s\n", mism==0 ? "PASS: byte-exact -> in-chain +/-1 is IN-CHAIN-SPECIFIC (not a generic Verilator artifact)"
                              : "MISMATCH: Verilator-isolation also wrong -> simulator-specific");
  delete dut;
  return mism==0?0:1;
}
