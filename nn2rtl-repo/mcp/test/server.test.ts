import { describe, expect, it, vi } from "vitest";

import { createServer, handleToolCall, toolDefinitions, type ToolImplementations } from "../server.js";

function createToolImpls(): ToolImplementations {
  return {
    get_failure_corpus: vi.fn(async () => ({ visible_tier: "output/failure_corpus/visible", entries: [] })),
    get_rtl_patterns: vi.fn(async () => ({ pattern_markdown: "fixture", reference_verilog: null, license_notice: null })),
    read_weights: vi.fn(async () => ({ model_name: "fixture", quantization: "int8_symmetric_per_tensor", generated_at: "now", layers: [] })),
    run_iverilog: vi.fn(async () => ({ success: true, stderr: "" })),
    run_verilator: vi.fn(async () => ({ module_id: "m1", status: "pass" })),
    run_vivado: vi.fn(async () => ({
      success: true,
      tool: "vivado",
      part: "xczu9eg-ffvb1156-2-e",
      stage: "synth",
      lut_count: 1,
      ff_count: 1,
      dsp_count: 0,
      bram18_count: 0,
      bram36_count: 0,
      bram18_equiv: 0,
      wns_ns: 1,
      timing_met: true,
      fmax_mhz: 10,
      report: "fixture",
    })),
    write_verilog: vi.fn(async () => "/tmp/module.v"),
  };
}

describe("mcp server", () => {
  it("declares the expected tools", () => {
    expect(toolDefinitions.map((tool) => tool.name)).toEqual([
      "run_iverilog",
      "run_verilator",
      "run_vivado",
      "read_weights",
      "write_verilog",
      "get_rtl_patterns",
      "get_failure_corpus",
    ]);
  });

  it("routes each tool call to the correct implementation", async () => {
    const impls = createToolImpls();

    await handleToolCall("run_iverilog", { verilog_source: "module m; endmodule", module_name: "m" }, impls);
    await handleToolCall("run_verilator", { verilog_source: "module m; endmodule", module_name: "m", sidecar_path: "/tmp/sidecar.json" }, impls);
    await handleToolCall("run_vivado", { verilog_source: "module m; endmodule", module_name: "m", clock_period_ns: 20, part: "xczu9eg-ffvb1156-2-e", threads: 8 }, impls);
    await handleToolCall("read_weights", { checkpoint_path: "checkpoint.pth", quantization_config: {} }, impls);
    await handleToolCall("write_verilog", { module: { module_id: "m", spec_hash: "h", verilog_source: "module m; endmodule", generated_by: "Foundry", attempt: 1 }, output_dir: "/tmp" }, impls);
    await handleToolCall("get_rtl_patterns", { op_type: "conv2d", kernel_h: 1, kernel_w: 1 }, impls);
    await handleToolCall("get_failure_corpus", { module_id: "m", max_entries: 2 }, impls);

    expect(impls.run_iverilog).toHaveBeenCalledOnce();
    expect(impls.run_verilator).toHaveBeenCalledOnce();
    expect(impls.run_vivado).toHaveBeenCalledWith("module m; endmodule", "m", 20, "xczu9eg-ffvb1156-2-e", 8);
    expect(impls.read_weights).toHaveBeenCalledOnce();
    expect(impls.write_verilog).toHaveBeenCalledOnce();
    expect(impls.get_rtl_patterns).toHaveBeenCalledWith("conv2d", 1, 1, undefined);
    expect(impls.get_failure_corpus).toHaveBeenCalledWith({
      module_id: "m",
      max_entries: 2,
      include_verilog: false,
    });
  });

  it("fails malformed input before calling a handler", async () => {
    const impls = createToolImpls();
    await expect(handleToolCall("run_iverilog", { module_name: "m" }, impls)).rejects.toThrow();
    expect(impls.run_iverilog).not.toHaveBeenCalled();
  });

  it("returns an MCP error for unknown tools", async () => {
    const impls = createToolImpls();
    const result = await handleToolCall("missing_tool", {}, impls);
    expect(result.isError).toBe(true);
    expect(result.content[0]).toMatchObject({ type: "text" });
  });

  it("creates a server instance without connecting transport", () => {
    expect(createServer(createToolImpls())).toBeTruthy();
  });
});
