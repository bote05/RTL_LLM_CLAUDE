export function expectedLatencyCycles(layer: {
  pipeline_latency_cycles: number;
  input_width_bits: number;
  contract_params?: Record<string, unknown>;
}): number {
  const beatWidth = Number(layer.contract_params?.beat_width_bits) || 256;
  const beats = Math.max(1, Math.ceil(layer.input_width_bits / beatWidth));
  return layer.pipeline_latency_cycles + beats - 1;
}

export function checkLatency(report: { latency_cycles?: number }, layer: Parameters<typeof expectedLatencyCycles>[0]): boolean {
  return report.latency_cycles === undefined || report.latency_cycles === expectedLatencyCycles(layer);
}
