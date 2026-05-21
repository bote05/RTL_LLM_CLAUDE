// On-chip-weights contract testbench template.
// The common runtime validates the activation stream. RTL using the URAM
// weight read port must expose the metadata-declared ports; the common
// runtime drives a deterministic behavioural URAM model that responds to
// weight_rd_addr / weight_rd_en with weight_rd_data sourced from the
// sidecar's weights_path (.mem image), with a fixed two-cycle read latency
// matching UltraScale+ URAM288.
#define NN2RTL_CONTRACT_ID "on-chip-weights"
#define NN2RTL_CONTRACT_DESCRIPTION "activation stream plus on-chip URAM read port for weights"
#include "contract_tb_runtime.cpp"
