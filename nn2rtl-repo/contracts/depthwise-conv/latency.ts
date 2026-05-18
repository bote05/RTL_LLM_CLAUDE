export function expectedLatencyCycles(layer: { pipeline_latency_cycles: number }): number {
  return layer.pipeline_latency_cycles;
}

export function checkLatency(report: { latency_cycles?: number }, layer: { pipeline_latency_cycles: number }): boolean {
  return report.latency_cycles === undefined || report.latency_cycles === expectedLatencyCycles(layer);
}
