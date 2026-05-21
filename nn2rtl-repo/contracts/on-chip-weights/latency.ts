// URAM read latency on UltraScale+ UltraRAM (URAM288) is 2 cycles end-to-end
// (address-register stage + memory output register). The on-chip-weights
// contract pre-issues weight reads inside the MAC pipeline so this fixed
// pipeline cost is added once, on top of the flat-bus convolution latency
// computed by the Python frontend's `compute_conv2d_latency_cycles` helper.
const URAM_READ_LATENCY_CYCLES = 2;

export function expectedLatencyCycles(layer: {
  pipeline_latency_cycles: number;
  contract_params?: Record<string, unknown>;
}): number {
  const uramReadLatency =
    Number(layer.contract_params?.uram_read_latency_cycles) || URAM_READ_LATENCY_CYCLES;
  return layer.pipeline_latency_cycles + uramReadLatency;
}

export function checkLatency(
  report: { latency_cycles?: number },
  layer: Parameters<typeof expectedLatencyCycles>[0],
): boolean {
  return report.latency_cycles === undefined || report.latency_cycles === expectedLatencyCycles(layer);
}
