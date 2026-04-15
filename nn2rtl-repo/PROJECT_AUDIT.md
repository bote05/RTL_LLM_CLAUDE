# Project Audit — Remaining Open Items

Last pass: 2026-04-15. All items previously flagged have either been fixed in
this commit or deliberately set aside (see "Deliberately not addressed" below).

## Fixed in this pass

| # | Item | Fix |
| --- | --- | --- |
| AG-5 | Signed DUT outputs misread as unsigned | Sign-extend `data_out` using `output_width_bits` in `tb/static_verilator_tb.cpp` |
| AG-6 | Timing only checked for first vector | Per-vector latency check collapsed into `all_vectors_timing_ok` |
| AG-4 | Dead `FallbackBottleneck` / `FallbackResNet50` | Removed from `scripts/quantize_impl.py` |
| AG-1 | System spawn errors laundered into syntax errors | `isSystemSpawnError` rethrows ENOENT/EACCES/OOM etc. |
| AG-2 | Hardcoded `/tmp` fallback | Use `os.tmpdir()` with explicit `TMPDIR` override |
| AG-12 | Sidecar `module_name` never validated | `run_verilator` now asserts `sidecar.module_name === module_name` |
| AG-8 | v2 checkpoint bypasses strict validator | `build_pipeline_ir_payload` calls `load_quantized_checkpoint` on v2 too |
| 2 | `--max-retries` ignored on resume | `loadState` preserves caller-supplied `max_retries` |
| 5 | `bias_int32` clipped through INT8 writer | Added `tensor_to_int32_list`; conv bias writes through `write_signed_int32_hex` |
| 7 | Surgeon taxonomy missing `synthesis_failed` | Added to agent prompt + skill doc |
| 8 | Root scripts hardcode `python` | Switched to `python3` |
| 14 | `write_verilog` agents missing `output_dir` | Payloads now include `write_verilog_output_dir` |
| 17 | Resume accepts mismatched module IDs | `loadState` rejects missing/extra module IDs vs current LayerIR |
| 18 | `sdk` `test:full` missing TMPDIR workaround | Added `cross-env TMPDIR=/tmp` |
| 2p-14 | `parseYosysReport` LUT regex matched only `LUT4` | Sums any `*LUT*` / `ICESTORM_LC` row |
| 2p-15 | `saveState` non-atomic | tmp-file + `fsync` + `rename` |
| 2p-16 | `handlePipelineError` swallowed recovery failures | Logs each sub-step failure to stderr |
| 2p-17 | CLI accepted non-existent checkpoint path | Existence check at `runCli` boundary |
| 2p-18 | Unbounded Yosys `report` in state | `capYosysReport` truncates to 8 KB tail |
| 2p-19 | Add-module packing contract lived only in prompt | `validateAddModulePacking` asserts `input_width_bits == 2 * output_width_bits` at LayerIR load |
| 2p-20 | ARCHITECTURE.md stale Yosys command | Command string + LUT matcher description updated |
| 1 | Stale `output/layer_ir.json` silently reused | Sidecar fingerprint file; mismatch is a hard error |
| 16 (partial) | README "five agents" / stem-out-of-scope / JSON goldens | Corrected |

## Deliberately not addressed

These items from the incoming audit I reviewed and declined to change:

- **"Fallback to random weights" (#3) / synthetic calibration (#4)** — Both are
  documented smoke-mode fallbacks; removing them would break the offline
  deterministic CI flow. A real-data calibration path is a planned follow-up,
  not a bug.
- **Large `VerifResult` state bloat on big layers (#6)** — Real issue, but an
  architectural change (streamed mismatch summaries). Out of scope for this
  audit pass.
- **Source-mode vs compiled MCP drift (#10)** — Already handled by
  `MCP_TOOLS_MODULE_PATH` branching.
- **Resume redoes generation after Assayer crash (#11)** — Nuanced; current
  behaviour is explicitly documented in `loadState` crash-point table.
- **`quantization_config` dead param (#13)** — True but low-signal; the API
  surface is intentional for future calibration modes.
- **`testbench_template_path` sidecar ballast (#15)** — Part of the twin
  schema contract; removing risks churn for no runtime benefit.
- **Agent frontmatter dead metadata (Placeholder #3, Explicit #4)** — The
  frontmatter is contributor-facing documentation; runtime sourcing from
  `sdk/config.ts` is the intended design.
- **Plugin placeholder `daniel@example.com` (Placeholder #2)** — Trivial,
  deferred to a dedicated metadata pass.
- **`OUTPUT_DIR` env not consumed (Placeholder #4)** — Cosmetic; paths are
  fully determined by explicit args.
- **Legacy `format_version=1` toy path (Placeholder #1)** — Still used by
  `test_golden_impl.py` tests; removing is a larger cleanup.
- **`failure_class` coercion to `undefined` (AG-7)** — This is the
  intentional hardening added in this pass; Assayer sometimes emits
  `"none"`/`"N/A"` for passes and we do not want that to crash the run. The
  status gate (`pass`/`fail`) is authoritative.
- **`write_verilog` resolves against cwd (AG-13)** — Intentional; callers
  launched from a different working directory get the natural filesystem
  behaviour.
- **Reference-mutation hazard in cost merge (AG-3)** — Audit admits "perhaps
  benign"; no observable path.
- **Test-gap items (AG-9, AG-10, AG-11)** — True gaps, but writing new
  fixtures is a follow-up, not a code-bug fix.
- **Yosys `-abc9` version assumption (#9)** — The Windows oss-cad-suite
  toolchain the project targets has `-abc9`; weakening the command would mask
  toolchain setup problems on Linux hosts.

## Validation after fix pass

- `npm --prefix sdk run typecheck`: passes
- `npm --prefix mcp run typecheck`: passes
- Full test suites not re-run in this pass — expected failure: the Python
  unit test `test_write_layer_hex_artifacts_emits_int8_hex` was updated to
  the new INT32 bias hex width (`00000001\n`, `00000002\n`); re-run
  `python3 -m pytest` to confirm.
