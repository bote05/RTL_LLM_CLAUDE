// Activation-double-buffering contract testbench template.
// The stream semantics stay valid/ready compatible while the RTL exposes
// buffer-select observability for ping-pong load/compute scheduling.
#define NN2RTL_CONTRACT_ID "activation-double-buffering"
#define NN2RTL_CONTRACT_DESCRIPTION "ping-pong activation buffers hiding load latency"
#include "contract_tb_runtime.cpp"
