import { describe, expect, it } from "vitest";

import {
  applicabilityForSignature,
  exactReferenceKey,
  layerSignature,
  signatureBundle,
  signatureCandidateMatchLevel,
  signatureHash,
} from "../signatures.js";
import type { LayerIR } from "../types.js";

function layer(overrides: Partial<LayerIR> = {}): LayerIR {
  return {
    module_id: "m0",
    op_type: "conv2d",
    input_shape: [1, 16, 28, 28],
    output_shape: [1, 32, 28, 28],
    weights_path: "weights.hex",
    bias_path: null,
    weight_shape: [32, 16, 3, 3],
    num_weights: 32 * 16 * 3 * 3,
    scale_factor: 1,
    zero_point: 0,
    pipeline_latency_cycles: 123,
    clock_period_ns: 5,
    input_width_bits: 256,
    output_width_bits: 256,
    clock_signal: "clk",
    reset_signal: "rst_n",
    valid_in_signal: "valid_in",
    valid_out_signal: "valid_out",
    ready_in_signal: "ready_in",
    data_in_signal: "data_in",
    data_out_signal: "data_out",
    golden_inputs_path: "in.gold",
    golden_outputs_path: "out.gold",
    stride: [1, 1],
    padding: [1, 1],
    dilation: [1, 1],
    groups: 1,
    channel_tile: 32,
    quantization_family: "int8_symmetric_per_tensor",
    ...overrides,
  } as LayerIR;
}

describe("layer signatures", () => {
  it("hashes deterministically after contract planning fields are present", () => {
    const sig = layerSignature(layer({ contract_id: "tiled-streaming" }), "tiled-streaming");
    expect(signatureHash(sig)).toEqual(signatureHash(JSON.parse(JSON.stringify(sig))));
    expect(sig.channel_tile).toBe(32);
  });

  it("separates per-tensor and per-channel quantization families", () => {
    const perTensor = layerSignature(layer({ quantization_family: "int8_symmetric_per_tensor" }), "flat-bus");
    const perChannel = layerSignature(layer({ quantization_family: "int8_symmetric_per_channel_weight" }), "flat-bus");
    expect(signatureHash(perTensor)).not.toEqual(signatureHash(perChannel));
    expect(perTensor.quantization_family).toBe("int8_symmetric_per_tensor");
    expect(perChannel.quantization_family).toBe("int8_symmetric_per_channel_weight");
  });

  it("does not allow exact reference matching for mixed or unknown quantization", () => {
    const sig = layerSignature(layer({ quantization_family: "mixed_or_unknown" }), "flat-bus");
    expect(sig.quantization_family).toBe("mixed_or_unknown");
    expect(exactReferenceKey(sig)).toBeNull();
  });

  it("uses maxpool geometry in signatures", () => {
    const base = layer({
      op_type: "maxpool",
      weight_shape: [1, 1, 1, 1],
      kernel_size: [3, 3],
      pool_stride: [2, 2],
      pool_padding: [1, 1],
    });
    const changed = layerSignature(base, "flat-bus");
    expect(changed.kernel).toEqual([3, 3]);
    expect(changed.stride).toEqual([2, 2]);
    expect(changed.padding).toEqual([1, 1]);

    const other = layerSignature({ ...base, kernel_size: [2, 2] }, "flat-bus");
    expect(signatureHash(changed)).not.toEqual(signatureHash(other));
  });

  it("matches candidates by exact signature before relaxed applicability fields", () => {
    const signatures = signatureBundle({
      baseLayer: layer(),
      runtimeLayer: layer({ contract_id: "tiled-streaming" }),
      baseContractId: "flat-bus",
      runtimeContractId: "tiled-streaming",
    });
    const applicability = applicabilityForSignature({ networkId: "toy-net", signatures });
    expect(signatureCandidateMatchLevel({ applicability }, { ...signatures, network_id: "toy-net" }))
      .toBe("exact_signature");

    expect(signatureCandidateMatchLevel({
      applicability: {
        ...applicability,
        signature_hashes: ["different"],
        exact_reference_key: "different-key",
        exact_reference_keys: ["different-key"],
      },
    }, { ...signatures, network_id: "toy-net" })).toBe("op_contract_kernel_stride_groups");
  });

  it("does not treat signature_hashes as structural applicability", () => {
    const signatures = signatureBundle({
      baseLayer: layer(),
      runtimeLayer: layer({ contract_id: "tiled-streaming" }),
      baseContractId: "flat-bus",
      runtimeContractId: "tiled-streaming",
    });
    const target = { ...signatures, network_id: "toy-net" };
    expect(signatureCandidateMatchLevel({
      op_type: "relu",
      contract_id: "flat-bus",
      signature_hashes: ["not-the-target"],
    }, target)).toBeNull();
  });
});
