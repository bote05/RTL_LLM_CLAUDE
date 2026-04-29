export function expectedLatencyCycles(layer: {
  pipeline_latency_cycles: number;
  contract_params?: Record<string, unknown>;
}): number {
  const prefetchLatency = Number(layer.contract_params?.weight_prefetch_latency_cycles) || 0;
  const underrunSlack = Number(layer.contract_params?.prefetch_underrun_slack_cycles) || 0;
  return layer.pipeline_latency_cycles + prefetchLatency + underrunSlack;
}

export function checkLatency(report: { latency_cycles?: number }, layer: Parameters<typeof expectedLatencyCycles>[0]): boolean {
  return report.latency_cycles === undefined || report.latency_cycles === expectedLatencyCycles(layer);
}
