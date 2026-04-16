import { describe, expect, it, vi } from "vitest";

import { createServer, handleToolCall, toolDefinitions, type ToolImplementations } from "../server.js";

function createToolImpls(): ToolImplementations {
  return {
    read_weights: vi.fn(async () => ({ model_name: "fixture", quantization: "int8_symmetric_per_tensor", generated_at: "now", layers: [] })),
    run_iverilog: vi.fn(async () => ({ success: true, stderr: "" })),
    run_verilator: vi.fn(async () => ({ module_id: "m1", status: "pass" })),
    run_yosys: vi.fn(async () => ({ success: true, lut_count: 1, fmax_mhz: 10, report: "fixture" })),
    write_verilog: vi.fn(async () => "/tmp/module.v"),
  };
}

describe("mcp server", () => {
  it("declares the expected five tools", () => {
    expect(toolDefinitions.map((tool) => tool.name)).toEqual([
      "run_iverilog",
      "run_verilator",
      "run_yosys",
      "read_weights",
      "write_verilog",
    ]);
  });

  it("routes each tool call to the correct implementation", async () => {
    const impls = createToolImpls();

    await handleToolCall("run_iverilog", { verilog_source: "module m; endmodule", module_name: "m" }, impls);
    await handleToolCall("run_verilator", { verilog_source: "module m; endmodule", module_name: "m", sidecar_path: "/tmp/sidecar.json" }, impls);
    await handleToolCall("run_yosys", { verilog_source: "module m; endmodule", module_name: "m", clock_period_ns: 20 }, impls);
    await handleToolCall("read_weights", { checkpoint_path: "checkpoint.pth", quantization_config: {} }, impls);
    await handleToolCall("write_verilog", { module: { module_id: "m", spec_hash: "h", verilog_source: "module m; endmodule", generated_by: "Foundry", attempt: 1 }, output_dir: "/tmp" }, impls);

    expect(impls.run_iverilog).toHaveBeenCalledOnce();
    expect(impls.run_verilator).toHaveBeenCalledOnce();
    expect(impls.run_yosys).toHaveBeenCalledWith("module m; endmodule", "m", 20);
    expect(impls.read_weights).toHaveBeenCalledOnce();
    expect(impls.write_verilog).toHaveBeenCalledOnce();
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
