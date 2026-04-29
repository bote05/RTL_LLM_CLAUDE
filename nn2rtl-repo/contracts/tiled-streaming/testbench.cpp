// Tiled-streaming contract testbench template.
// Golden files are beat-oriented: one logical pixel appears as N consecutive
// channel-tile beats on valid_in/data_in and N output beats on valid_out/data_out.
#define NN2RTL_CONTRACT_ID "tiled-streaming"
#define NN2RTL_CONTRACT_DESCRIPTION "fixed-width channel tile beats"
#include "contract_tb_runtime.cpp"
