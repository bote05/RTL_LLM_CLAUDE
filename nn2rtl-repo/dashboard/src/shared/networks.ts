import registry from "../../../networks.json";

export type NetworkId = string;

export type NetworkConfig = {
  readonly id: NetworkId;
  readonly label: string;
  readonly modelName: string;
  readonly outputDir: string;
  readonly defaultCheckpointPath: string;
  readonly frontend: string;
  readonly available: boolean;
  readonly description: string;
};

export const NETWORKS: readonly NetworkConfig[] = registry.networks;
export const DEFAULT_NETWORK_ID: NetworkId = registry.defaultNetworkId;

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
