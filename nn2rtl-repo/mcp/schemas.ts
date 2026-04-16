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
]);

export const layerIrSchema = z
  .object({
    module_id: z.string(),
    op_type: z.enum(["conv2d", "relu", "add"]),
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
  })
  .strict();

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

export const runYosysInput = z
  .object({
    verilog_source: z.string(),
    module_name: z.string(),
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

export const runYosysOutput = z
  .object({
    success: z.boolean(),
    lut_count: z.number(),
    fmax_mhz: z.number(),
    area_um2: z.number().default(0),
    report: z.string(),
  })
  .strict();

export const writeVerilogOutput = z
  .object({
    path: z.string(),
  })
  .strict();
