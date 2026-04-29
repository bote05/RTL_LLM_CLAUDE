// DRAM-backed-weights contract testbench template.
// The common runtime validates the activation stream. RTL using the AXI weight
// interface must expose the metadata-declared ports; deterministic AXI memory
// response hooks live in this contract family and are driven by Foundry docs.
#define NN2RTL_CONTRACT_ID "dram-backed-weights"
#define NN2RTL_CONTRACT_DESCRIPTION "activation stream plus AXI read channel for weights"
#include "contract_tb_runtime.cpp"
