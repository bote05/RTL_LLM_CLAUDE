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
  "manual_correction_needed",
  "spec_hash_mismatch",
  "agent_max_turns_exhausted",
]);

export const failureCategorySchema = z.enum([
  "code_bug",
  "architectural_fit",
  "toolchain_infra",
  "verification_env",
  "unknown",
]);

export const failureClassificationSchema = z
  .object({
    category: failureCategorySchema,
    violated_resource: z.string().nullable().optional(),
    violated_constraint: z.string().nullable().optional(),
    rationale: z.string().min(1),
  })
  .strict();

// Retrospector chooses the post-analysis actor: by default a same-contract
// final repair is handed to Surgeon (preserves a near-passing artifact);
// Retrospector picks Foundry only when the architecture or contract itself
// must change. Older Retrospector outputs that pre-date these fields are
// still accepted — the orchestrator defaults them safely.
export const retrospectorNextActorSchema = z.enum(["surgeon", "foundry"]);
export const retrospectorBaseArtifactSchema = z.enum([
  "latest",      // resume from whichever attempt failed most recently
  "best_known",  // resume from the highest-scoring attempt across history
  "fresh",       // discard prior attempts (only sensible with next_actor=foundry)
]);
export const retrospectorRepairScopeSchema = z.enum([
  "targeted_fsm_or_datapath_fix",
  "numerical_pipeline_fix",
  "interface_or_contract_fix",
  "architecture_replacement",
]);

export const retrospectorAdviceSchema = z
  .object({
    analysis: z.string().min(1),
    suggestion: z.string().min(1),
    doc_fault: z.boolean().optional(),
    faulty_doc_paths: z.array(z.string()).optional(),
    next_actor: retrospectorNextActorSchema.optional(),
    base_artifact: retrospectorBaseArtifactSchema.optional(),
    repair_scope: retrospectorRepairScopeSchema.optional(),
  })
  .strict();

export const moduleStatusSchema = z.enum([
  "pending",
  "generating",
  "verifying",
  "pass",
  "fail_retry",
  "fail_abort",
]);

const contractIdSchema = z.enum([
  "flat-bus",
  "tiled-streaming",
  "dram-backed-weights",
  "activation-double-buffering",
  "weight-tiling",
  "depthwise-conv",
]);

const contractParamSchema = z.record(
  z.string(),
  z.union([z.string(), z.number(), z.boolean(), z.null()]),
);

// Base ZodObject — exposes .pick / .omit / .partial for callers that need
// structural slicing (tests, delegation output-format derivation).
export const layerIrBaseSchema = z
  .object({
    module_id: z.string(),
    op_type: z.enum(["conv2d", "relu", "add", "maxpool", "global_avg_pool", "gemm"]),
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
    quantization_family: z.string().optional(),
    // Optional upper-bound clip applied AFTER op (e.g. ReLU6 sets clip_max=6).
    // For relu, when set, output = clamp(x, 0, clip_max). When absent the
    // activation is the standard unbounded ReLU. Other op_types currently
    // ignore the field; carrying it as optional avoids a schema migration.
    clip_max: z.number().optional(),
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
    dilation: z.array(z.number().int().positive()).optional(),
    groups: z.number().int().positive().optional(),
    // Number of accumulator lanes in each conv output-channel group. In the
    // current verified conv contract, a lane_counter serializes those lanes:
    // one lane issues one weight read / MAC per cycle. The FSM runs
    // ceil(OC / mac_parallelism) passes per output pixel. Only emitted for
    // op_type == "conv2d"; ignored otherwise. Frontend computes it as
    // min(OC, PIPELINE_CONFIG.MAX_PARALLEL_MACS).
    mac_parallelism: z.number().int().positive().optional(),
    weight_bank_paths: z.array(z.string()).optional(),
    contract_id: contractIdSchema.optional(),
    contract_params: contractParamSchema.optional(),
    io_mode: z
      .enum([
        "packed_full",
        "channel_tiled",
        "dram_backed_weights",
        "activation_double_buffered",
        "weight_tiled",
      ])
      .optional(),
    channel_tile: z.number().int().positive().optional(),
    // MaxPool2d geometry — optional at the type level because only present
    // when op_type === "maxpool". The *refined* schema below requires them
    // for maxpool layers; this base schema does not so that .pick()/.omit()
    // keep working for callers that don't care about the refinement.
    kernel_size: z.array(z.number().int().positive()).optional(),
    pool_stride: z.array(z.number().int().positive()).optional(),
    pool_padding: z.array(z.number().int().nonnegative()).optional(),
    // GlobalAveragePool spatial dims [H, W] — folded into SCALE_MULT/SHIFT.
    gap_spatial: z.array(z.number().int().positive()).optional(),
    // Gemm (FC) layer geometry, mirroring weight_shape [M, K].
    gemm_in_features: z.number().int().positive().optional(),
    gemm_out_features: z.number().int().positive().optional(),
    base_layer_signature: z.record(z.string(), z.unknown()).optional(),
    runtime_layer_signature: z.record(z.string(), z.unknown()).optional(),
    signature_hash: z.string().optional(),
    exact_reference_key: z.string().nullable().optional(),
  })
  .strict();

// Runtime-validation schema. Every .parse / .safeParse path on a LayerIR
// must go through this one — it enforces maxpool geometry presence.
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
    initiation_interval_cycles: nullToUndef(z.number()),
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
    failure_category: z.preprocess(
      (v) => {
        if (v === null || v === undefined) return undefined;
        if (typeof v !== "string") return undefined;
        return failureCategorySchema.safeParse(v).success ? v : undefined;
      },
      failureCategorySchema.optional(),
    ),
    violated_resource: nullToUndef(z.string()),
    violated_constraint: nullToUndef(z.string()),
    classifier_reason: nullToUndef(z.string()),
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
    first_mismatch_vector_index: nullToUndef(z.number()),
    first_mismatch_output_index: nullToUndef(z.number()),
    first_mismatch_channel_index: nullToUndef(z.number()),
    exact_match_count: nullToUndef(z.number()),
    mismatch_count: nullToUndef(z.number()),
    signed_error_sum: nullToUndef(z.number()),
    positive_error_count: nullToUndef(z.number()),
    negative_error_count: nullToUndef(z.number()),
    first_valid_in_cycle: nullToUndef(z.number()),
    first_valid_out_cycle: nullToUndef(z.number()),
    axi_weight_memory_model_enabled: nullToUndef(z.boolean()),
    axi_weight_memory_model_status: nullToUndef(z.string()),
    axi_weight_bytes_loaded: nullToUndef(z.number()),
    axi_weight_bytes_per_beat: nullToUndef(z.number()),
    axi_weight_arvalid_cycles: nullToUndef(z.number()),
    axi_weight_arready_cycles: nullToUndef(z.number()),
    axi_weight_ar_handshakes: nullToUndef(z.number()),
    axi_weight_rvalid_cycles: nullToUndef(z.number()),
    axi_weight_rready_cycles: nullToUndef(z.number()),
    axi_weight_r_beats: nullToUndef(z.number()),
    axi_weight_completed_bursts: nullToUndef(z.number()),
    axi_weight_first_arvalid_cycle: nullToUndef(z.number()),
    axi_weight_first_ar_handshake_cycle: nullToUndef(z.number()),
    axi_weight_first_r_beat_cycle: nullToUndef(z.number()),
    axi_weight_out_of_range_reads: nullToUndef(z.number()),
    // Verbatim simulator stdout (truncated). Captured for Surgeon /
    // Assayer when they need to see `$display` / `$write` traces from
    // probe code embedded in the DUT or testbench. Foundry should not
    // probe-debug from this — it is an evidence channel, not a workflow.
    verilator_stdout: nullToUndef(z.string()),
    // Per-vector breakdown of simulation outcomes. The TB drives N
    // distinct goldin vectors per run; aggregate metrics
    // (max_error / first_mismatch_*) collapse them into one number.
    // This array preserves which specific vector(s) failed, so a
    // module that passes vector 0 + 1 but mis-pipelines vector 2
    // (e.g. stride-2 active-pixel-counter bug) can be diagnosed
    // without re-running the simulation. See tb/static_verilator_tb.cpp
    // for emission logic.
    per_vector: nullToUndef(
      z.array(
        z
          .object({
            vector_idx: z.number(),
            outputs_received: z.number(),
            exact_match_count: z.number(),
            mismatch_count: z.number(),
            max_error: z.number(),
            mean_error: z.number(),
            actual_cycles: z.number(),
            first_mismatch_output_index: z.number().nullable(),
          })
          .strict(),
      ),
    ),
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
    contract_id: contractIdSchema.optional(),
    contract_name: z.string().optional(),
    contract_metadata_path: z.string().optional(),
    beat_width_bits: z.number().int().positive().optional(),
    beats_per_input_sample: z.number().int().positive().optional(),
    beats_per_output_sample: z.number().int().positive().optional(),
    weights_path: z.string().optional(),
    weight_bank_paths: z.array(z.string()).optional(),
    axi_weight_data_width_bits: z.number().int().positive().optional(),
    contract_params: contractParamSchema.optional(),
  })
  .strict();

export const synthesisReportSchema = z
  .object({
    success: z.boolean(),
    tool: z.literal("vivado").default("vivado"),
    part: z.string().default("xczu9eg-ffvb1156-2-e"),
    stage: z.literal("synth").default("synth"),
    lut_count: z.number(),
    ff_count: z.number().default(0),
    dsp_count: z.number().default(0),
    bram18_count: z.number().default(0),
    bram36_count: z.number().default(0),
    bram18_equiv: z.number().default(0),
    // Setup-path Worst Negative Slack. `wns_ns` is the historical name and
    // is always Setup WNS. `setup_wns_ns` is the explicit alias.
    wns_ns: z.number().nullable().default(null),
    setup_wns_ns: z.number().nullable().default(null),
    // Hold-path Worst Hold Slack ("WHS(ns)" in Vivado). Reported for
    // visibility but not gated on at synth-only stage — small hold
    // violations on a pre-placement netlist are routinely fixed during
    // place_design / opt_design.
    hold_wns_ns: z.number().nullable().default(null),
    timing_met: z.boolean().default(false),
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
    retrospector_calls: z.record(z.string(), z.number().int().nonnegative()).default({}),
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
        } else if (result.status === "pass") {
          // A passing assayer result must promote the module to 'pass';
          // a manual override that flips the module to 'fail_abort' while
          // leaving result.status='pass' produces a contradictory state
          // that survives across resumes (and confuses the dashboard,
          // failure corpus filters, and Vivado dispatch).
          ctx.addIssue({
            code: "custom",
            path: ["results", id, "status"],
            message: `module '${id}' is in 'fail_abort' but its VerifResult.status is 'pass'; either promote the module to 'pass' or set the result to a failing status with an appropriate failure_class.`,
          });
        }
        const earlyAbort =
          result?.status_class === "tb_setup_error" ||
          result?.failure_class === "architectural_unsupported" ||
          result?.failure_class === "manual_correction_needed" ||
          result?.failure_category === "toolchain_infra" ||
          result?.failure_category === "verification_env" ||
          result?.failure_category === "architectural_fit" ||
          result?.failure_category === "unknown";
        if (attempts !== undefined && attempts < state.max_retries && !earlyAbort) {
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
