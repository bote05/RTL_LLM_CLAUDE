import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const sdkRoot = path.resolve(
  __dirname,
  path.basename(__dirname) === "dist" ? ".." : ".",
);
const repoRoot = path.resolve(sdkRoot, "..");
const registryPath = path.join(repoRoot, "networks.json");

export type NetworkId = string;

export type NetworkConfig = {
  id: string;
  label: string;
  modelName: string;
  outputDir: string;
  defaultCheckpointPath: string;
  frontend: "checkpoint" | "onnx" | string;
  available: boolean;
  description: string;
};

export type NetworkRegistry = {
  version: number;
  defaultNetworkId: string;
  networks: NetworkConfig[];
};

let cachedRegistry: NetworkRegistry | null = null;

export function readNetworkRegistry(): NetworkRegistry {
  if (cachedRegistry !== null) return cachedRegistry;
  const parsed = JSON.parse(readFileSync(registryPath, "utf8")) as NetworkRegistry;
  if (!Array.isArray(parsed.networks) || parsed.networks.length === 0) {
    throw new Error("networks.json must contain a non-empty networks array.");
  }
  cachedRegistry = parsed;
  return parsed;
}

export function listNetworks(): NetworkConfig[] {
  return readNetworkRegistry().networks;
}

export function defaultNetworkId(): string {
  return readNetworkRegistry().defaultNetworkId;
}

export function isKnownNetworkId(value: unknown): value is string {
  return typeof value === "string" && listNetworks().some((network) => network.id === value);
}

export function getNetwork(id: string = defaultNetworkId()): NetworkConfig {
  const found = listNetworks().find((network) => network.id === id);
  if (!found) {
    throw new Error(`Unknown network id '${id}'. Known: ${listNetworks().map((n) => n.id).join(", ")}`);
  }
  return found;
}

export function outputDirForNetwork(id: string = defaultNetworkId(), root = repoRoot): string {
  return path.resolve(root, getNetwork(id).outputDir);
}
