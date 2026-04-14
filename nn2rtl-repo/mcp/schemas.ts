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
    golden_inputs: z.array(z.array(z.number())),
    golden_outputs: z.array(z.array(z.number())),
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

export const verifResultSchema = z
  .object({
    module_id: z.string(),
    status: z.enum(["pass", "fail", "syntax_error"]),
    timing_pass: z.boolean().optional(),
    timing_actual_cycles: z.number().optional(),
    timing_expected_cycles: z.number().optional(),
    mismatch_layer: z.string().optional(),
    expected: z.array(z.number()).optional(),
    got: z.array(z.number()).optional(),
    max_error: z.number().optional(),
    mean_error: z.number().optional(),
    failure_class: failureClassSchema.nullable().optional(),
    fix_hint: z.string().optional(),
    iverilog_stderr: z.string().optional(),
    verilator_stderr: z.string().optional(),
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
    report: z.string(),
  })
  .strict();

export const writeVerilogOutput = z
  .object({
    path: z.string(),
  })
  .strict();
