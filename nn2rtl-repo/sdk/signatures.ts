import { createHash } from "node:crypto";

import type { LayerIR } from "./types.js";

export type QuantizationFamily =
  | "int8_symmetric_per_tensor"
  | "int8_symmetric_per_channel_weight"
  | "int8_symmetric_per_channel_activation"
  | "int8_asymmetric_per_tensor"
  | "mixed_or_unknown";

export type SpatialFamily = "tiny" | "small" | "medium" | "large" | "xlarge" | "unknown";
export type SignaturePair = [number | null, number | null];
export type SignatureMatchLevel =
  | "exact_signature"
  | "exact_reference_key"
  | "op_contract_kernel_stride_groups"
  | "op_contract_kernel"
  | "op_contract"
  | "op";

export type LayerSignature = {
  version: 1;
  op_type: LayerIR["op_type"];
  contract_id: string;
  kernel: SignaturePair;
  stride: SignaturePair;
  padding: SignaturePair;
  dilation: SignaturePair;
  groups: number;
  input_channels: number | null;
  output_channels: number | null;
  input_width_bits: number;
  output_width_bits: number;
  bus_width_bits: number;
  channel_tile: number | null;
  spatial_shape: [number | null, number | null];
  spatial_family: SpatialFamily;
  quantization_family: QuantizationFamily;
};

export type SignatureBundle = {
  base_layer_signature: LayerSignature;
  runtime_layer_signature: LayerSignature;
  signature_hash: string;
  exact_reference_key: string | null;
};

export type SignatureTarget = SignatureBundle & {
  network_id?: string | null;
};

export type SignatureCandidate = {
  op_type?: unknown;
  contract_id?: unknown;
  signature_hash?: unknown;
  signature_hashes?: unknown;
  exact_reference_key?: unknown;
  exact_reference_keys?: unknown;
  runtime_layer_signature?: unknown;
  applicability?: unknown;
  shape?: unknown;
};

export const SIGNATURE_MATCH_LEVELS: readonly SignatureMatchLevel[] = [
  "exact_signature",
  "exact_reference_key",
  "op_contract_kernel_stride_groups",
  "op_contract_kernel",
  "op_contract",
  "op",
];

export function signatureMatchRank(level: SignatureMatchLevel): number {
  return SIGNATURE_MATCH_LEVELS.indexOf(level);
}

function pair(value: unknown): [number | null, number | null] {
  if (!Array.isArray(value)) return [null, null];
  const a = typeof value[0] === "number" && Number.isFinite(value[0]) ? value[0] : null;
  const b = typeof value[1] === "number" && Number.isFinite(value[1]) ? value[1] : a;
  return [a, b];
}

function channels(shape: unknown): number | null {
  return Array.isArray(shape) && typeof shape[1] === "number" ? shape[1] : null;
}

function spatial(shape: unknown): [number | null, number | null] {
  return Array.isArray(shape)
    ? [
        typeof shape[2] === "number" ? shape[2] : null,
        typeof shape[3] === "number" ? shape[3] : null,
      ]
    : [null, null];
}

export function spatialFamily(shape: unknown): SpatialFamily {
  const [h, w] = spatial(shape);
  if (h === null || w === null) return "unknown";
  const area = h * w;
  if (area <= 14 * 14) return "tiny";
  if (area <= 28 * 28) return "small";
  if (area <= 56 * 56) return "medium";
  if (area <= 112 * 112) return "large";
  return "xlarge";
}

export function quantizationFamily(layer: LayerIR, modelQuantization?: string): QuantizationFamily {
  const candidates = [
    (layer as unknown as { quantization_family?: unknown }).quantization_family,
    (layer as unknown as { quantization?: unknown }).quantization,
    modelQuantization,
  ].filter((v): v is string => typeof v === "string");
  const text = candidates.join(" ").toLowerCase();
  if (text.includes("per_channel") && text.includes("activation")) return "int8_symmetric_per_channel_activation";
  if (text.includes("per_channel")) return "int8_symmetric_per_channel_weight";
  if (text.includes("asymmetric")) return "int8_asymmetric_per_tensor";
  if (text.includes("int8") && text.includes("symmetric") && text.includes("per_tensor")) return "int8_symmetric_per_tensor";
  return "mixed_or_unknown";
}

function layerKernel(layer: LayerIR): SignaturePair {
  if (layer.op_type === "conv2d" && Array.isArray(layer.weight_shape)) {
    return [layer.weight_shape[2] ?? null, layer.weight_shape[3] ?? null];
  }
  if (layer.op_type === "maxpool") {
    return pair(layer.kernel_size);
  }
  return [null, null];
}

function layerStride(layer: LayerIR): SignaturePair {
  if (layer.op_type === "maxpool") {
    return pair(layer.pool_stride ?? [1, 1]);
  }
  return pair(layer.stride ?? [1, 1]);
}

function layerPadding(layer: LayerIR): SignaturePair {
  if (layer.op_type === "maxpool") {
    return pair(layer.pool_padding ?? [0, 0]);
  }
  return pair(layer.padding ?? [0, 0]);
}

export function layerSignature(
  layer: LayerIR,
  contractId: string,
  modelQuantization?: string,
): LayerSignature {
  const inputChannels = channels(layer.input_shape);
  const outputChannels = channels(layer.output_shape);
  return {
    version: 1,
    op_type: layer.op_type,
    contract_id: contractId,
    kernel: layerKernel(layer),
    stride: layerStride(layer),
    padding: layerPadding(layer),
    dilation: pair(layer.dilation ?? [1, 1]),
    groups: typeof layer.groups === "number" ? layer.groups : 1,
    input_channels: inputChannels,
    output_channels: outputChannels,
    input_width_bits: layer.input_width_bits,
    output_width_bits: layer.output_width_bits,
    bus_width_bits: Math.max(layer.input_width_bits, layer.output_width_bits),
    channel_tile: typeof layer.channel_tile === "number" ? layer.channel_tile : null,
    spatial_shape: spatial(layer.output_shape),
    spatial_family: spatialFamily(layer.output_shape),
    quantization_family: quantizationFamily(layer, modelQuantization),
  };
}

export function signatureHash(signature: LayerSignature): string {
  const canonical = stableJson(signature);
  return createHash("sha256").update(canonical).digest("hex").slice(0, 32);
}

function stableJson(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(stableJson).join(",")}]`;
  }
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${stableJson(record[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

export function exactReferenceKey(signature: LayerSignature): string | null {
  if (signature.quantization_family === "mixed_or_unknown") return null;
  return [
    signature.op_type,
    signature.contract_id,
    `k${signature.kernel.join("x")}`,
    `s${signature.stride.join("x")}`,
    `d${signature.dilation.join("x")}`,
    `g${signature.groups}`,
    `bus${signature.bus_width_bits}`,
    `tile${signature.channel_tile ?? "none"}`,
    `c${signature.input_channels ?? "?"}-${signature.output_channels ?? "?"}`,
    signature.quantization_family,
  ].join("|");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringValues(value: unknown): string[] {
  if (typeof value === "string" && value.length > 0) return [value];
  if (Array.isArray(value)) {
    return value.filter((item): item is string => typeof item === "string" && item.length > 0);
  }
  return [];
}

function firstString(...values: unknown[]): string | undefined {
  for (const value of values) {
    const [found] = stringValues(value);
    if (found) return found;
  }
  return undefined;
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function pairValue(value: unknown): SignaturePair | undefined {
  if (!Array.isArray(value)) return undefined;
  const [a, b] = pair(value);
  if (a === null && b === null) return undefined;
  return [a, b];
}

function samePair(a: SignaturePair | undefined, b: SignaturePair): boolean {
  return a !== undefined && a[0] === b[0] && a[1] === b[1];
}

function kernelFromShape(shape: Record<string, unknown> | undefined): SignaturePair | undefined {
  const weightShape = shape?.weight_shape;
  if (!Array.isArray(weightShape)) return undefined;
  const kh = numberValue(weightShape[2]);
  const kw = numberValue(weightShape[3]);
  return kh === undefined || kw === undefined ? undefined : [kh, kw];
}

function candidateStructuralFields(candidate: SignatureCandidate): {
  op_type?: string;
  contract_id?: string;
  kernel?: SignaturePair;
  stride?: SignaturePair;
  padding?: SignaturePair;
  groups?: number;
} {
  const applicability = isRecord(candidate.applicability) ? candidate.applicability : {};
  const runtimeSignature = isRecord(candidate.runtime_layer_signature) ? candidate.runtime_layer_signature : {};
  const shape = isRecord(candidate.shape) ? candidate.shape : undefined;
  return {
    op_type: firstString(applicability.op_type, runtimeSignature.op_type, candidate.op_type),
    contract_id: firstString(applicability.contract_id, runtimeSignature.contract_id, candidate.contract_id),
    kernel:
      pairValue(applicability.kernel) ??
      pairValue(runtimeSignature.kernel) ??
      kernelFromShape(shape),
    stride:
      pairValue(applicability.stride) ??
      pairValue(runtimeSignature.stride) ??
      pairValue(shape?.stride),
    padding:
      pairValue(applicability.padding) ??
      pairValue(runtimeSignature.padding) ??
      pairValue(shape?.padding),
    groups:
      numberValue(applicability.groups) ??
      numberValue(runtimeSignature.groups) ??
      numberValue(shape?.groups),
  };
}

function candidateSignatureHashes(candidate: SignatureCandidate): string[] {
  const applicability = isRecord(candidate.applicability) ? candidate.applicability : {};
  return [
    ...stringValues(candidate.signature_hash),
    ...stringValues(candidate.signature_hashes),
    ...stringValues(applicability.signature_hash),
    ...stringValues(applicability.signature_hashes),
  ];
}

function candidateExactReferenceKeys(candidate: SignatureCandidate): string[] {
  const applicability = isRecord(candidate.applicability) ? candidate.applicability : {};
  return [
    ...stringValues(candidate.exact_reference_key),
    ...stringValues(candidate.exact_reference_keys),
    ...stringValues(applicability.exact_reference_key),
    ...stringValues(applicability.exact_reference_keys),
  ];
}

export function signatureCandidateMatchLevel(
  candidate: SignatureCandidate,
  target: SignatureTarget,
): SignatureMatchLevel | null {
  if (candidateSignatureHashes(candidate).includes(target.signature_hash)) {
    return "exact_signature";
  }

  if (
    target.exact_reference_key !== null &&
    candidateExactReferenceKeys(candidate).includes(target.exact_reference_key)
  ) {
    return "exact_reference_key";
  }

  const runtime = target.runtime_layer_signature;
  const fields = candidateStructuralFields(candidate);
  if (fields.op_type !== runtime.op_type) return null;

  const contractMatches = fields.contract_id === runtime.contract_id;
  const kernelMatches = samePair(fields.kernel, runtime.kernel);
  const strideMatches = samePair(fields.stride, runtime.stride);
  const groupsMatches = fields.groups === runtime.groups;

  if (contractMatches && kernelMatches && strideMatches && groupsMatches) {
    return "op_contract_kernel_stride_groups";
  }
  if (contractMatches && kernelMatches) {
    return "op_contract_kernel";
  }
  // Backward-compat / generic-doc tier: if the candidate matches op_type AND
  // contract_id BUT didn't specify kernel/stride/groups (legacy lifecycle
  // docs, or intentionally generic contract-level docs), treat it as an
  // op_contract-level cover. This lets `auto_tiled_existing`-style seeded
  // docs cover a whole contract's worth of conv2d traffic without listing
  // every kernel shape explicitly. Specified-but-mismatched kernels still
  // fall through to "op".
  if (contractMatches && fields.kernel === undefined) {
    return "op_contract";
  }
  return "op";
}

export function signaturePaddingMatches(candidate: SignatureCandidate, target: SignatureTarget): boolean {
  const fields = candidateStructuralFields(candidate);
  return samePair(fields.padding, target.runtime_layer_signature.padding);
}

export function applicabilityForSignature(input: {
  networkId: string;
  signatures: SignatureBundle;
}): Record<string, unknown> {
  const signature = input.signatures.runtime_layer_signature;
  return {
    networks: [input.networkId],
    signature_hashes: [input.signatures.signature_hash],
    exact_reference_keys: input.signatures.exact_reference_key ? [input.signatures.exact_reference_key] : [],
    exact_reference_key: input.signatures.exact_reference_key,
    op_type: signature.op_type,
    contract_id: signature.contract_id,
    kernel: signature.kernel,
    stride: signature.stride,
    padding: signature.padding,
    dilation: signature.dilation,
    groups: signature.groups,
    bus_width_bits: signature.bus_width_bits,
    channel_tile: signature.channel_tile,
    input_channels: signature.input_channels,
    output_channels: signature.output_channels,
    quantization_family: signature.quantization_family,
    spatial_family: signature.spatial_family,
  };
}

export function signatureBundle(input: {
  baseLayer: LayerIR;
  runtimeLayer: LayerIR;
  baseContractId: string;
  runtimeContractId: string;
  modelQuantization?: string;
}): SignatureBundle {
  const base = layerSignature(input.baseLayer, input.baseContractId, input.modelQuantization);
  const runtime = layerSignature(input.runtimeLayer, input.runtimeContractId, input.modelQuantization);
  return {
    base_layer_signature: base,
    runtime_layer_signature: runtime,
    signature_hash: signatureHash(runtime),
    exact_reference_key: exactReferenceKey(runtime),
  };
}
