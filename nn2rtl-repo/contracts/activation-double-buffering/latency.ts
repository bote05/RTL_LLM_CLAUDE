export function expectedLatencyCycles(layer: {
  pipeline_latency_cycles: number;
  contract_params?: Record<string, unknown>;
}): number {
  const fillLatency = Number(layer.contract_params?.activation_buffer_fill_cycles) || 0;
  return layer.pipeline_latency_cycles + fillLatency;
}

export function checkLatency(report: { latency_cycles?: number }, layer: Parameters<typeof expectedLatencyCycles>[0]): boolean {
  return report.latency_cycles === undefined || report.latency_cycles === expectedLatencyCycles(layer);
}
