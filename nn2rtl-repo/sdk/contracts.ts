import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import type { LayerIR } from "./types.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

export const CONTRACT_IDS = [
  "flat-bus",
  "tiled-streaming",
  "dram-backed-weights",
  "activation-double-buffering",
  "weight-tiling",
] as const;

export type ContractId = typeof CONTRACT_IDS[number];

export type ContractSignal = {
  name: string;
  direction: "input" | "output" | "inout";
  width_bits?: number;
  width_expr?: string;
  role?: string;
};

export type ContractMetadata = {
  name: ContractId;
  display_name: string;
  complexity_rank: number;
  interface_signals: ContractSignal[];
  fit_constraints: {
    max_bus_width_bits: number;
    // Maximum bytes of weights this contract is willing to keep on-chip. Set
    // on contracts whose weights live in BRAM (flat-bus, tiled-streaming,
    // activation-double-buffering); omitted on contracts that stream weights
    // from DRAM (dram-backed-weights, weight-tiling) where the layer's total
    // weight size doesn't bound on-chip storage. Layers exceeding this cap
    // fail `contractFitFailure`, which routes the failure_classifier through
    // a contract-walk to a heavier contract.
    max_on_chip_weight_bytes?: number;
    default_beat_width_bits?: number;
    default_channel_tile?: number;
    bram_formula: string;
    dram_bandwidth_assumption?: string;
    buffer_sizing_rules?: string[];
    weight_tiling_rules?: string[];
  };
  supported_ops: LayerIR["op_type"][];
  dependencies: string[];
  docs: string[];
  protocol_rules?: string[];
};

export type ContractSelectionPayload = {
  selected: ContractMetadata;
  available: ContractMetadata[];
  metadata_path: string;
};

const CONTRACT_ID_SET = new Set<string>(CONTRACT_IDS);

export function isContractId(value: unknown): value is ContractId {
  return typeof value === "string" && CONTRACT_ID_SET.has(value);
}

export function contractRoot(): string {
  return path.join(repoRoot, "contracts");
}

export function contractMetadataPath(contractId: ContractId): string {
  return path.join(contractRoot(), contractId, "metadata.json");
}

export function contractTestbenchTemplatePath(contractId: ContractId): string {
  return path.join(contractRoot(), contractId, "testbench.cpp");
}

export function loadContractMetadata(contractId: ContractId): ContractMetadata {
  const metadataPath = contractMetadataPath(contractId);
  if (!existsSync(metadataPath)) {
    throw new Error(`Contract metadata not found for '${contractId}' at '${metadataPath}'.`);
  }
  const parsed = JSON.parse(readFileSync(metadataPath, "utf8")) as ContractMetadata;
  if (parsed.name !== contractId) {
    throw new Error(
      `Contract metadata '${metadataPath}' declares name='${parsed.name}', expected '${contractId}'.`,
    );
  }
  return parsed;
}

export function allContractMetadata(): ContractMetadata[] {
  return CONTRACT_IDS
    .map(loadContractMetadata)
    .sort((a, b) => a.complexity_rank - b.complexity_rank);
}

export function resolveLayerContractId(layer: LayerIR): ContractId {
  if (isContractId(layer.contract_id)) {
    return layer.contract_id;
  }
  switch (layer.io_mode) {
    case "channel_tiled":
      return "tiled-streaming";
    case "dram_backed_weights":
      return "dram-backed-weights";
    case "activation_double_buffered":
      return "activation-double-buffering";
    case "weight_tiled":
      return "weight-tiling";
    case "packed_full":
    case undefined:
      return "flat-bus";
    default:
      return "flat-bus";
  }
}

export function contractSelectionForLayer(layer: LayerIR): ContractSelectionPayload {
  const selected = loadContractMetadata(resolveLayerContractId(layer));
  return {
    selected,
    available: allContractMetadata(),
    metadata_path: contractMetadataPath(selected.name),
  };
}

export function contractSupportsLayer(layer: LayerIR): boolean {
  const metadata = loadContractMetadata(resolveLayerContractId(layer));
  return metadata.supported_ops.includes(layer.op_type);
}

export function effectiveInputStreamWidthBits(layer: LayerIR): number {
  return layer.op_type === "add" ? layer.input_width_bits / 2 : layer.input_width_bits;
}

export function contractFitFailure(layer: LayerIR): string | null {
  const metadata = loadContractMetadata(resolveLayerContractId(layer));
  if (!metadata.supported_ops.includes(layer.op_type)) {
    return (
      `Contract '${metadata.name}' does not support op_type='${layer.op_type}'. ` +
      `Supported ops: ${metadata.supported_ops.join(", ")}.`
    );
  }

  const inputWidth = effectiveInputStreamWidthBits(layer);
  const outputWidth = layer.output_width_bits;
  const cap = metadata.fit_constraints.max_bus_width_bits;
  const over: string[] = [];
  if (inputWidth > cap) over.push(`effective_input_width_bits=${inputWidth}>max_bus_width_bits=${cap}`);
  if (outputWidth > cap) over.push(`output_width_bits=${outputWidth}>max_bus_width_bits=${cap}`);

  // Weight-memory budget gate. Contracts whose weights live on-chip declare
  // `max_on_chip_weight_bytes`; layers whose INT8 weight tensor exceeds that
  // budget can synth-pass but won't actually fit alongside the rest of the
  // network on the target device (BRAM18 budget on ZCU102 is ~32.8 Mbit
  // shared across 119 layers — single layers spilling to LUTRAM blow the
  // LUT count, and timing_met from synth-only Vivado hides the issue).
  const weightCap = metadata.fit_constraints.max_on_chip_weight_bytes;
  const weightBytes = layer.num_weights ?? 0;
  if (weightCap !== undefined && weightBytes > weightCap) {
    over.push(`weight_bytes=${weightBytes}>max_on_chip_weight_bytes=${weightCap}`);
  }

  if (over.length === 0) return null;

  const nudge =
    metadata.name === "flat-bus"
      ? "Tag the LayerIR with contract_id='tiled-streaming' or a heavier contract."
      : "Reduce channel_tile / beat width or select a heavier contract.";
  return (
    `Layer ${layer.module_id} does not fit contract '${metadata.name}': ` +
    `${over.join(" and ")} exceeds capability. ${nudge}`
  );
}

export function contractSidecarFields(layer: LayerIR): Record<string, unknown> {
  const contractId = resolveLayerContractId(layer);
  const metadata = loadContractMetadata(contractId);
  const beatWidth =
    Number(layer.contract_params?.beat_width_bits) ||
    metadata.fit_constraints.default_beat_width_bits ||
    layer.input_width_bits;
  const inputChannels = layer.input_shape.length >= 2 ? layer.input_shape[1] : 1;
  const outputChannels = layer.output_shape.length >= 2 ? layer.output_shape[1] : 1;
  const logicalInputWidthBits = layer.op_type === "add" ? inputChannels * 16 : inputChannels * 8;
  const logicalOutputWidthBits = outputChannels * 8;
  const inputBeats = Math.max(1, Math.ceil(logicalInputWidthBits / beatWidth));
  const outputBeats = Math.max(1, Math.ceil(logicalOutputWidthBits / beatWidth));

  const fields: Record<string, unknown> = {
    contract_id: contractId,
    contract_name: metadata.display_name,
    contract_metadata_path: contractMetadataPath(contractId),
    beat_width_bits: beatWidth,
    beats_per_input_sample: inputBeats,
    beats_per_output_sample: outputBeats,
    contract_params: layer.contract_params ?? {},
  };
  if (contractId === "dram-backed-weights") {
    fields.weights_path = layer.weights_path;
    if (layer.weight_bank_paths) {
      fields.weight_bank_paths = layer.weight_bank_paths;
    }
    fields.axi_weight_data_width_bits = 64;
  }
  return fields;
}
