export type LayerOpType = "conv2d" | "relu" | "add";

export interface LayerIR {
  module_id: string;
  op_type: LayerOpType;
  input_shape: number[];
  output_shape: number[];
  weights_path: string;
  bias_path: string | null;
  weight_shape: number[];
  num_weights: number;
  scale_factor: number;
  lhs_scale_factor?: number;
  rhs_scale_factor?: number;
  zero_point: number;
  pipeline_latency_cycles: number;
  clock_period_ns: number;
  input_width_bits: number;
  output_width_bits: number;
  clock_signal: "clk";
  reset_signal: "rst_n";
  valid_in_signal: "valid_in";
  valid_out_signal: "valid_out";
  ready_in_signal: "ready_in";
  data_in_signal: "data_in";
  data_out_signal: "data_out";
  golden_inputs_path: string;
  golden_outputs_path: string;
}

export interface PipelineIR {
  model_name: string;
  quantization: "int8_symmetric_per_tensor";
  generated_at: string;
  layers: LayerIR[];
}

export interface VerilogModule {
  module_id: string;
  spec_hash: string;
  verilog_source: string;
  generated_by: "Foundry" | "Surgeon";
  attempt: number;
}

export type FailureClass =
  | "integer_overflow"
  | "sign_extension_error"
  | "bit_shift_wrong"
  | "rounding_mode_wrong"
  | "saturation_missing"
  | "loop_bounds_incorrect"
  | "array_indexing_error"
  | "port_width_mismatch"
  | "residual_addition_overflow"
  | "missing_pipeline_register"
  | "pipeline_latency_wrong"
  | "reset_logic_broken"
  | "enable_signal_ignored"
  | "scale_factor_misapplied"
  | "bias_term_missing"
  | "batch_norm_not_folded"
  | "synthesis_failed";

export interface VerifResult {
  module_id: string;
  status: "pass" | "fail" | "syntax_error";
  timing_pass?: boolean;
  timing_actual_cycles?: number;
  timing_expected_cycles?: number;
  mismatch_layer?: string;
  expected?: number[];
  got?: number[];
  max_error?: number;
  mean_error?: number;
  failure_class?: FailureClass | null;
  fix_hint?: string;
  iverilog_stderr?: string;
  verilator_stderr?: string;
}

export interface ModelUsageEntry {
  input_tokens?: number;
  output_tokens?: number;
  cache_creation_input_tokens?: number | null;
  cache_read_input_tokens?: number | null;
  server_tool_use?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface VerificationSidecar {
  module_name: string;
  module_id: string;
  clock_signal: string;
  reset_signal: string;
  valid_in_signal: string;
  valid_out_signal: string;
  ready_in_signal: string;
  data_in_signal: string;
  data_out_signal: string;
  bus_bytes_per_sample: number;
  input_width_bits: number;
  output_width_bits: number;
  pipeline_latency_cycles: number;
  clock_period_ns: number;
  golden_inputs_path: string;
  golden_outputs_path: string;
  results_path: string;
  testbench_template_path: string;
}

export type ModuleStatus =
  | "pending"
  | "generating"
  | "verifying"
  | "pass"
  | "fail_retry"
  | "fail_abort";

export interface PipelineState {
  run_id: string;
  started_at: string;
  modules: Record<string, ModuleStatus>;
  attempts: Record<string, number>;
  results: Record<string, VerifResult>;
  max_retries: number;
  total_cost_usd: number;
  model_usage: Record<string, ModelUsageEntry>;
}

export type NextAction =
  | { action: "invoke_cartographer" }
  | { action: "invoke_foundry"; module_id: string }
  | { action: "invoke_assayer"; module_id: string }
  | { action: "invoke_surgeon"; module_id: string }
  | { action: "done" };
