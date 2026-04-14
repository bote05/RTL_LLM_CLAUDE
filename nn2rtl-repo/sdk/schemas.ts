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
]);

export const moduleStatusSchema = z.enum([
  "pending",
  "generating",
  "verifying",
  "pass",
  "fail_retry",
  "fail_abort",
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

export const synthesisReportSchema = z
  .object({
    success: z.boolean(),
    lut_count: z.number(),
    fmax_mhz: z.number(),
    report: z.string(),
  })
  .strict();

export const modelUsageEntrySchema = z.record(z.string(), z.unknown());

export const pipelineStateSchema = z
  .object({
    run_id: z.string(),
    started_at: z.string(),
    modules: z.record(z.string(), moduleStatusSchema),
    attempts: z.record(z.string(), z.number().int().nonnegative()),
    results: z.record(z.string(), verifResultSchema),
    max_retries: z.number().int().nonnegative(),
    total_cost_usd: z.number().nonnegative(),
    model_usage: z.record(z.string(), modelUsageEntrySchema),
  })
  .strict()
  .superRefine((state, ctx) => {
    const moduleIds = new Set(Object.keys(state.modules));
    const attemptIds = new Set(Object.keys(state.attempts));

    for (const id of moduleIds) {
      if (!attemptIds.has(id)) {
        ctx.addIssue({
          code: "custom",
          path: ["attempts", id],
          message: `attempts is missing an entry for module '${id}' present in modules`,
        });
      }
    }
    for (const id of attemptIds) {
      if (!moduleIds.has(id)) {
        ctx.addIssue({
          code: "custom",
          path: ["attempts", id],
          message: `attempts has entry for unknown module '${id}' not present in modules`,
        });
      }
    }
    for (const id of Object.keys(state.results)) {
      if (!moduleIds.has(id)) {
        ctx.addIssue({
          code: "custom",
          path: ["results", id],
          message: `results has entry for unknown module '${id}' not present in modules`,
        });
      }
    }
    for (const [id, status] of Object.entries(state.modules)) {
      const attempts = state.attempts[id];
      const result = state.results[id];

      if (status === "fail_retry") {
        if (!result) {
          ctx.addIssue({
            code: "custom",
            path: ["results", id],
            message: `module '${id}' is in 'fail_retry' but has no prior VerifResult in results`,
          });
        }
        if (attempts !== undefined && attempts >= state.max_retries) {
          ctx.addIssue({
            code: "custom",
            path: ["attempts", id],
            message: `module '${id}' is in 'fail_retry' but attempts (${attempts}) has already reached max_retries (${state.max_retries}); expected 'fail_abort'`,
          });
        }
      }

      if (status === "fail_abort") {
        if (!result) {
          ctx.addIssue({
            code: "custom",
            path: ["results", id],
            message: `module '${id}' is in 'fail_abort' but has no prior VerifResult in results`,
          });
        }
        if (attempts !== undefined && attempts < state.max_retries) {
          ctx.addIssue({
            code: "custom",
            path: ["attempts", id],
            message: `module '${id}' is in 'fail_abort' but attempts (${attempts}) is below max_retries (${state.max_retries}); expected 'fail_retry'`,
          });
        }
      }

      if (status === "pass") {
        if (!result) {
          ctx.addIssue({
            code: "custom",
            path: ["results", id],
            message: `module '${id}' is in 'pass' but has no VerifResult in results`,
          });
        } else if (result.status !== "pass") {
          ctx.addIssue({
            code: "custom",
            path: ["results", id, "status"],
            message: `module '${id}' is in 'pass' but its VerifResult.status is '${result.status}'`,
          });
        }
      }

      if (result && result.module_id !== id) {
        ctx.addIssue({
          code: "custom",
          path: ["results", id, "module_id"],
          message: `results['${id}'].module_id is '${result.module_id}'; expected '${id}'`,
        });
      }
    }
  });
