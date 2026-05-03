// DRAM-backed-weights contract testbench template.
// The common runtime validates the activation stream. RTL using the AXI weight
// interface must expose the metadata-declared ports; the common runtime drives
// a deterministic AXI memory responder from the sidecar's weights_path.
#define NN2RTL_CONTRACT_ID "dram-backed-weights"
#define NN2RTL_CONTRACT_DESCRIPTION "activation stream plus AXI read channel for weights"
#include "contract_tb_runtime.cpp"
