export type LayerOpType = "conv2d" | "relu" | "add" | "maxpool";
export type ContractId =
  | "flat-bus"
  | "tiled-streaming"
  | "dram-backed-weights"
  | "activation-double-buffering"
  | "weight-tiling";
export type IoMode =
  | "packed_full"
  | "channel_tiled"
  | "dram_backed_weights"
  | "activation_double_buffered"
  | "weight_tiled";

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
  // Conv2d geometry — populated by the modern frontends when op_type == "conv2d"
  stride?: number[];
  padding?: number[];
  dilation?: number[];
  groups?: number;
  // Number of accumulator lanes in each output-channel group. In the current
  // serialized-read conv contract, only one lane issues a weight read / MAC
  // per cycle, selected by lane_counter; MP still controls OC_PASSES and the
  // number of acc/biased/scaled registers. Only set for op_type == "conv2d".
  mac_parallelism?: number;
  // Optional BRAM-bank artifact paths for Vivado-oriented conv generation.
  // Layout is one file per lane; current verified RTL may continue to use the
  // flat weights_path until the banked datapath contract is enabled.
  weight_bank_paths?: string[];
  // Contract selection hook. When omitted, flat-bus is the default. io_mode is
  // kept for backward compatibility with older manually-tagged LayerIR files.
  contract_id?: ContractId;
  contract_params?: Record<string, string | number | boolean | null>;
  io_mode?: IoMode;
  channel_tile?: number;
  // MaxPool2d geometry — only present when op_type == "maxpool"
  kernel_size?: number[];
  pool_stride?: number[];
  pool_padding?: number[];
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
  | "synthesis_failed"
  | "verilator_timeout"
  | "architectural_unsupported"
  | "structural_preflight_failed"
  | "manual_correction_needed"
  | "spec_hash_mismatch";

export type FailureCategory =
  | "code_bug"
  | "architectural_fit"
  | "toolchain_infra"
  | "verification_env"
  | "unknown";

export interface FailureClassification {
  category: FailureCategory;
  violated_resource?: string | null;
  violated_constraint?: string | null;
  rationale: string;
}

export interface RetrospectorAdvice {
  analysis: string;
  suggestion: string;
  doc_fault?: boolean;
  faulty_doc_paths?: string[];
}

export interface VerifResult {
  module_id: string;
  status: "pass" | "fail" | "syntax_error";
  timing_pass?: boolean;
  timing_actual_cycles?: number;
  timing_expected_cycles?: number;
  initiation_interval_cycles?: number;
  mismatch_layer?: string;
  expected?: number[];
  got?: number[];
  max_error?: number;
  mean_error?: number;
  sample_count?: number;
  failure_class?: FailureClass | null;
  failure_category?: FailureCategory | null;
  violated_resource?: string | null;
  violated_constraint?: string | null;
  classifier_reason?: string;
  fix_hint?: string;
  iverilog_stderr?: string;
  verilator_stderr?: string;
  // Raw simulation evidence the testbench emits. Surgeon reads these
  // directly; no pre-written diagnosis is supplied or trusted.
  status_class?: "sim_passed" | "sim_stalled" | "sim_completed_mismatch" | "tb_setup_error";
  outputs_expected?: number;
  outputs_received?: number;
  missing_index_start?: number;
  missing_index_end?: number;
  last_valid_out_cycle?: number;
  simulation_end_cycle?: number;
  output_gap_histogram?: number[];
  first_mismatch_index?: number;
  first_mismatch_expected?: number;
  first_mismatch_got?: number;
  axi_weight_memory_model_enabled?: boolean;
  axi_weight_memory_model_status?: string;
  axi_weight_bytes_loaded?: number;
  axi_weight_bytes_per_beat?: number;
  axi_weight_arvalid_cycles?: number;
  axi_weight_arready_cycles?: number;
  axi_weight_ar_handshakes?: number;
  axi_weight_rvalid_cycles?: number;
  axi_weight_rready_cycles?: number;
  axi_weight_r_beats?: number;
  axi_weight_completed_bursts?: number;
  axi_weight_first_arvalid_cycle?: number;
  axi_weight_first_ar_handshake_cycle?: number;
  axi_weight_first_r_beat_cycle?: number;
  axi_weight_out_of_range_reads?: number;
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
  contract_id?: ContractId;
  contract_name?: string;
  contract_metadata_path?: string;
  beat_width_bits?: number;
  beats_per_input_sample?: number;
  beats_per_output_sample?: number;
  weights_path?: string;
  weight_bank_paths?: string[];
  axi_weight_data_width_bits?: number;
  contract_params?: Record<string, string | number | boolean | null>;
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
  retrospector_calls: Record<string, number>;
}

export type NextAction =
  | { action: "invoke_cartographer" }
  | { action: "invoke_foundry"; module_id: string }
  | { action: "invoke_assayer"; module_id: string }
  | { action: "invoke_surgeon"; module_id: string }
  | { action: "done" };
