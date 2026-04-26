import { z } from "zod";

export const failureClassSchema = z.enum([
  "integer_overflow",
  "sign_extension_error",
  "bit_shift_wrong",
  "rounding_mode_wrong",
  "saturation_missing",
  "loop_bounds_incorrect",
  "array_indexing_error",
  "port_width_mismatch",
  "residual_addition_overflow",
  "missing_pipeline_register",
  "pipeline_latency_wrong",
  "reset_logic_broken",
  "enable_signal_ignored",
  "scale_factor_misapplied",
  "bias_term_missing",
  "batch_norm_not_folded",
  "synthesis_failed",
  "verilator_timeout",
  "architectural_unsupported",
  "structural_preflight_failed",
]);

// See sdk/schemas.ts for the rationale; the two schemas must stay in
// lockstep (check:twins).
export const layerIrBaseSchema = z
  .object({
    module_id: z.string(),
    op_type: z.enum(["conv2d", "relu", "add", "maxpool"]),
    input_shape: z.array(z.number().int().positive()),
    output_shape: z.array(z.number().int().positive()),
    weights_path: z.string(),
    bias_path: z.string().nullable(),
    weight_shape: z.array(z.number().int().positive()),
    num_weights: z.number().int().nonnegative(),
    scale_factor: z.number(),
    lhs_scale_factor: z.number().optional(),
    rhs_scale_factor: z.number().optional(),
    zero_point: z.number().int(),
    pipeline_latency_cycles: z.number().int().positive(),
    clock_period_ns: z.number().nonnegative(),
    input_width_bits: z.number().int().positive(),
    output_width_bits: z.number().int().positive(),
    clock_signal: z.literal("clk"),
    reset_signal: z.literal("rst_n"),
    valid_in_signal: z.literal("valid_in"),
    valid_out_signal: z.literal("valid_out"),
    ready_in_signal: z.literal("ready_in"),
    data_in_signal: z.literal("data_in"),
    data_out_signal: z.literal("data_out"),
    golden_inputs_path: z.string(),
    golden_outputs_path: z.string(),
    // Conv2d geometry — emitted by the frontends for op-aware generation/repair.
    stride: z.array(z.number().int().positive()).optional(),
    padding: z.array(z.number().int().nonnegative()).optional(),
    mac_parallelism: z.number().int().positive().optional(),
    weight_bank_paths: z.array(z.string()).optional(),
    io_mode: z.enum(["packed_full", "channel_tiled"]).optional(),
    channel_tile: z.number().int().positive().optional(),
    kernel_size: z.array(z.number().int().positive()).optional(),
    pool_stride: z.array(z.number().int().positive()).optional(),
    pool_padding: z.array(z.number().int().nonnegative()).optional(),
  })
  .strict();

export const layerIrSchema = layerIrBaseSchema.superRefine((layer, ctx) => {
  if (layer.op_type !== "maxpool") return;
  const missing: string[] = [];
  if (!layer.kernel_size || layer.kernel_size.length < 2) missing.push("kernel_size");
  if (!layer.pool_stride || layer.pool_stride.length < 2) missing.push("pool_stride");
  if (!layer.pool_padding || layer.pool_padding.length < 2) missing.push("pool_padding");
  if (missing.length > 0) {
    ctx.addIssue({
      code: "custom",
      path: ["op_type"],
      message:
        `maxpool LayerIR '${layer.module_id}' is missing required geometry fields: ${missing.join(", ")}. ` +
        `Each of kernel_size / pool_stride / pool_padding must be a 2-element array [H, W].`,
    });
  }
});

export const pipelineIrSchema = z
  .object({
    model_name: z.string(),
    quantization: z.literal("int8_symmetric_per_tensor"),
    generated_at: z.string(),
    layers: z.array(layerIrSchema),
  })
  .strict();

export const verilogModuleSchema = z
  .object({
    module_id: z.string(),
    spec_hash: z.string(),
    verilog_source: z.string(),
    generated_by: z.enum(["Foundry", "Surgeon"]),
    attempt: z.number().int().positive(),
  })
  .strict();

// Agents (Assayer in particular) sometimes emit `null` for optional string
// fields instead of omitting them. Coerce null → undefined at parse time so
// downstream TS code stays on the original `T | undefined` contract.
const nullToUndef = <S extends z.ZodTypeAny>(schema: S) =>
  z.preprocess((v) => (v === null ? undefined : v), schema.optional());

export const verifResultSchema = z
  .object({
    module_id: z.string(),
    status: z.enum(["pass", "fail", "syntax_error"]),
    timing_pass: nullToUndef(z.boolean()),
    timing_actual_cycles: nullToUndef(z.number()),
    timing_expected_cycles: nullToUndef(z.number()),
    mismatch_layer: nullToUndef(z.string()),
    expected: nullToUndef(z.array(z.number())),
    got: nullToUndef(z.array(z.number())),
    max_error: nullToUndef(z.number()),
    mean_error: nullToUndef(z.number()),
    sample_count: nullToUndef(z.number()),
    // Agents sometimes emit non-enum "unknown"-style strings for failure_class
    // on pass (e.g. "none", "N/A", ""). Coerce anything not in the taxonomy
    // to undefined — status=="pass"|"fail" remains the authoritative gate,
    // failure_class is a classification aid for Surgeon.
    failure_class: z.preprocess(
      (v) => {
        if (v === null || v === undefined) return undefined;
        if (typeof v !== "string") return undefined;
        return failureClassSchema.safeParse(v).success ? v : undefined;
      },
      failureClassSchema.optional(),
    ),
    fix_hint: nullToUndef(z.string()),
    iverilog_stderr: nullToUndef(z.string()),
    verilator_stderr: nullToUndef(z.string()),
    // Raw simulation evidence emitted by the testbench — Surgeon reasons
    // from these facts rather than from a pre-written "likely X" hint.
    // See tb/static_verilator_tb.cpp for the emission logic and
    // nn2rtl-plugin/agents/surgeon.md for how to interpret them.
    status_class: nullToUndef(
      z.enum([
        "sim_passed",
        "sim_stalled",
        "sim_completed_mismatch",
        "tb_setup_error",
      ]),
    ),
    outputs_expected: nullToUndef(z.number()),
    outputs_received: nullToUndef(z.number()),
    missing_index_start: nullToUndef(z.number()),
    missing_index_end: nullToUndef(z.number()),
    last_valid_out_cycle: nullToUndef(z.number()),
    simulation_end_cycle: nullToUndef(z.number()),
    output_gap_histogram: nullToUndef(z.array(z.number())),
    first_mismatch_index: nullToUndef(z.number()),
    first_mismatch_expected: nullToUndef(z.number()),
    first_mismatch_got: nullToUndef(z.number()),
  })
  .strict();

export const verificationSidecarSchema = z
  .object({
    module_name: z.string(),
    module_id: z.string(),
    clock_signal: z.literal("clk"),
    reset_signal: z.literal("rst_n"),
    valid_in_signal: z.literal("valid_in"),
    valid_out_signal: z.literal("valid_out"),
    ready_in_signal: z.literal("ready_in"),
    data_in_signal: z.literal("data_in"),
    data_out_signal: z.literal("data_out"),
    bus_bytes_per_sample: z.number().int().positive(),
    input_width_bits: z.number().int().positive(),
    output_width_bits: z.number().int().positive(),
    pipeline_latency_cycles: z.number().int().positive(),
    clock_period_ns: z.number().nonnegative(),
    golden_inputs_path: z.string(),
    golden_outputs_path: z.string(),
    results_path: z.string(),
    testbench_template_path: z.string(),
  })
  .strict();

export const runIverilogInput = z
  .object({
    verilog_source: z.string(),
    module_name: z.string(),
  })
  .strict();

export const runVerilatorInput = z
  .object({
    verilog_source: z.string(),
    module_name: z.string(),
    sidecar_path: z.string(),
  })
  .strict();

export const runVivadoInput = z
  .object({
    verilog_source: z.string(),
    module_name: z.string(),
    clock_period_ns: z.number().nonnegative().default(0),
    part: z.string().optional(),
    threads: z.number().int().positive().optional(),
  })
  .strict();

export const readWeightsInput = z
  .object({
    checkpoint_path: z.string(),
    quantization_config: z.record(z.string(), z.unknown()),
  })
  .strict();

export const writeVerilogInput = z
  .object({
    module: verilogModuleSchema,
    output_dir: z.string(),
  })
  .strict();

export const runIverilogOutput = z
  .object({
    success: z.boolean(),
    stderr: z.string(),
  })
  .strict();

export const runVivadoOutput = z
  .object({
    success: z.boolean(),
    tool: z.literal("vivado").default("vivado"),
    part: z.string().default("xc7a100tcsg324-1"),
    stage: z.literal("synth").default("synth"),
    lut_count: z.number(),
    ff_count: z.number().default(0),
    dsp_count: z.number().default(0),
    bram18_count: z.number().default(0),
    bram36_count: z.number().default(0),
    bram18_equiv: z.number().default(0),
    wns_ns: z.number().nullable().default(null),
    timing_met: z.boolean().default(false),
    fmax_mhz: z.number(),
    report: z.string(),
  })
  .strict();

export const writeVerilogOutput = z
  .object({
    path: z.string(),
  })
  .strict();

export const getRtlPatternsInput = z
  .object({
    op_type: z.enum(["conv2d", "relu", "add", "maxpool"]),
    kernel_h: z.number().int().positive().optional(),
    kernel_w: z.number().int().positive().optional(),
  })
  .strict();

export const getRtlPatternsOutput = z
  .object({
    pattern_markdown: z.string(),
    reference_verilog: z.string().nullable(),
    license_notice: z.string().nullable(),
  })
  .strict();
