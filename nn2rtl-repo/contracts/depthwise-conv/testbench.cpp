// Depthwise-conv contract testbench template.
// The MCP runner copies this file to static_verilator_tb.cpp and also copies
// tb/static_verilator_tb.cpp as contract_tb_runtime.cpp. The public interface
// is identical to flat-bus (one packed activation pixel per beat); only the
// internal compute pattern differs (per-channel 2D conv, no cross-channel
// reduction).
#define NN2RTL_CONTRACT_ID "depthwise-conv"
#define NN2RTL_CONTRACT_DESCRIPTION "one packed activation pixel per valid_in beat; per-channel 2D conv"
#include "contract_tb_runtime.cpp"
