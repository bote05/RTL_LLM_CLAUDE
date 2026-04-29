// Weight-tiling contract testbench template.
// Outputs are sampled only after the final tile has accumulated. The common
// stream runtime validates final results; tile control ports are declared by
// metadata and checked by SDK preflight.
#define NN2RTL_CONTRACT_ID "weight-tiling"
#define NN2RTL_CONTRACT_DESCRIPTION "DRAM-backed partial weight tiles with partial-sum accumulation"
#include "contract_tb_runtime.cpp"
