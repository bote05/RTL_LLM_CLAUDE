// Neural-network registry.
//
// The dashboard is being prepared for multi-network support. Today there is
// exactly one fully wired network (ResNet-50), but the data layer and the UI
// both flow through this registry so adding a second network — for example
// MobileNet-V2 — is a single config entry: append to `NETWORKS`, point its
// `outputDir` at the per-network artifact root, and it shows up in the
// selector. No other code change is required.
//
// IMPORTANT: For backwards-compatibility with the existing single-network
// layout, ResNet-50's `outputDir` is "output" (the repo-root flat folder the
// pipeline currently writes to). New networks SHOULD use `output/<network-id>`
// so the flat layout stops conflating runs.

export type NetworkId = "resnet-50";

export type NetworkConfig = {
  /** Stable identifier; used in URLs and JobAction.networkId. */
  readonly id: NetworkId;
  /** Human-friendly label shown in the selector. */
  readonly label: string;
  /** Model name the pipeline writes into output/layer_ir.json (for cross-checks). */
  readonly modelName: string;
  /** Repo-relative root where the pipeline writes output/, e.g. "output" or "output/mobilenet-v2". */
  readonly outputDir: string;
  /**
   * Default checkpoint path used to pre-populate the "Generate RTL" form.
   * Empty string is acceptable for networks that don't have a canonical
   * checkpoint yet — the form will require the user to type one.
   */
  readonly defaultCheckpointPath: string;
  /** Is the pipeline expected to actually work for this network today? */
  readonly available: boolean;
  /** Short one-line description shown next to the label. */
  readonly description: string;
};

// Add new networks here. Each entry is fully declarative — no other code change.
export const NETWORKS: readonly NetworkConfig[] = [
  {
    id: "resnet-50",
    label: "ResNet-50",
    modelName: "resnet50",
    outputDir: "output",
    defaultCheckpointPath: "checkpoints/resnet50_int8.pth",
    available: true,
    description: "ImageNet classifier, INT8 PTQ, ~17 distinct modules",
  },
] as const;

export const DEFAULT_NETWORK_ID: NetworkId = "resnet-50";

export function isKnownNetworkId(value: unknown): value is NetworkId {
  return typeof value === "string" && NETWORKS.some((network) => network.id === value);
}

export function getNetwork(id: NetworkId): NetworkConfig {
  const found = NETWORKS.find((network) => network.id === id);
  if (!found) {
    throw new Error(`Unknown network id '${id}'. Known: ${NETWORKS.map((n) => n.id).join(", ")}`);
  }
  return found;
}

export function resolveNetworkId(value: unknown): NetworkId {
  return isKnownNetworkId(value) ? value : DEFAULT_NETWORK_ID;
}
