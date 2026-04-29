export function expectedLatencyCycles(layer: {
  pipeline_latency_cycles: number;
  contract_params?: Record<string, unknown>;
}): number {
  const tileCount = Math.max(1, Number(layer.contract_params?.weight_tile_count) || 1);
  const tileLoadLatency = Number(layer.contract_params?.weight_tile_load_cycles) || 0;
  return layer.pipeline_latency_cycles * tileCount + tileLoadLatency * tileCount;
}

export function checkLatency(report: { latency_cycles?: number }, layer: Parameters<typeof expectedLatencyCycles>[0]): boolean {
  return report.latency_cycles === undefined || report.latency_cycles === expectedLatencyCycles(layer);
}
