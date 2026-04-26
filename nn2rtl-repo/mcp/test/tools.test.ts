import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  VERILATOR_COMMAND,
  get_rtl_patterns,
  parseVivadoReport,
  readSidecarIfPresent,
  read_weights,
  resolveOutputRoot,
  resolveRepoRootFromEnv,
  run_iverilog,
  run_verilator,
  run_vivado,
  stderrFromUnknown,
  toVivadoPath,
  withTempDir,
  write_verilog,
} from "../tools.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../..");
const fixtureRoot = path.join(repoRoot, "test", "fixtures");
const tempDirs: string[] = [];

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
});

async function makeTempDir(prefix: string): Promise<string> {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), prefix));
  tempDirs.push(tempDir);
  return tempDir;
}

async function writeSidecar(
  tempDir: string,
  overrides: Record<string, unknown> = {},
): Promise<string> {
  const sidecarPath = path.join(tempDir, "sidecar.json");
  const sidecar = {
    module_name: "unit_module",
    module_id: "unit_module",
    clock_signal: "clk",
    reset_signal: "rst_n",
    valid_in_signal: "valid_in",
    valid_out_signal: "valid_out",
    ready_in_signal: "ready_in",
    data_in_signal: "data_in",
    data_out_signal: "data_out",
    bus_bytes_per_sample: 1,
    input_width_bits: 8,
    output_width_bits: 8,
    pipeline_latency_cycles: 1,
    clock_period_ns: 20,
    golden_inputs_path: path.join(tempDir, "inputs.json"),
    golden_outputs_path: path.join(tempDir, "outputs.json"),
    results_path: path.join(tempDir, "results.json"),
    testbench_template_path: path.join(repoRoot, "tb", "static_verilator_tb.cpp"),
    ...overrides,
  };

  // Only materialize the vector files when the sidecar points at absolute
  // paths — the "rejects relative path" test deliberately supplies a bare
  // filename like `relative-inputs.json`, and writing to it would leak into
  // the test's cwd (which is the mcp package root on Windows runs).
  if (path.isAbsolute(sidecar.golden_inputs_path)) {
    await writeFile(sidecar.golden_inputs_path, JSON.stringify([[0, 1, 2]]), "utf8");
  }
  if (path.isAbsolute(sidecar.golden_outputs_path)) {
    await writeFile(sidecar.golden_outputs_path, JSON.stringify([[0, 1, 2]]), "utf8");
  }
  await writeFile(sidecarPath, JSON.stringify(sidecar), "utf8");
  return sidecarPath;
}

describe("mcp tools", () => {
  it("get_rtl_patterns returns a reference Verilog for conv2d 1x1 when the file exists", async () => {
    const result = await get_rtl_patterns("conv2d", 1, 1);
    expect(typeof result.pattern_markdown).toBe("string");
    // The 1x1 branch returns the checked-in reference module.
    expect(result.reference_verilog).toContain("module layer1_0_conv1");
    expect(result.reference_verilog).toContain("localparam IC        = 64;");
  });

  it("get_rtl_patterns returns null reference_verilog for op_types without a proven reference", async () => {
    const relu = await get_rtl_patterns("relu");
    expect(relu.reference_verilog).toBeNull();
    const pool = await get_rtl_patterns("maxpool");
    expect(pool.reference_verilog).toBeNull();
  });

  it("extracts stderr text from unknown errors", () => {
    expect(stderrFromUnknown({ stderr: Buffer.from("boom") })).toBe("boom");
    expect(stderrFromUnknown(new Error("plain-error"))).toBe("plain-error");
  });

  it("creates and cleans temporary directories", async () => {
    let createdDir = "";
    await withTempDir("nn2rtl-tools-test-", async (tempDir) => {
      createdDir = tempDir;
      await writeFile(path.join(tempDir, "fixture.txt"), "ok", "utf8");
      expect(await readFile(path.join(tempDir, "fixture.txt"), "utf8")).toBe("ok");
    });

    await expect(readFile(path.join(createdDir, "fixture.txt"), "utf8")).rejects.toThrow();
  });

  it("resolves repo root overrides from the environment", () => {
    expect(resolveRepoRootFromEnv({ NN2RTL_REPO_ROOT: "/tmp/override" })).toBe(
      path.resolve("/tmp/override"),
    );
    expect(resolveRepoRootFromEnv({})).toBe(repoRoot);
  });

  it("runs iverilog successfully through the command abstraction", async () => {
    const commandRunner = vi.fn(async (_file, args: string[]) => {
      expect(args[3]).toContain("unit_module.v");
      expect(await readFile(args[3], "utf8")).toContain("module unit_module");
      return { stdout: "", stderr: "" };
    });

    await expect(
      run_iverilog("module unit_module; endmodule", "unit_module", { commandRunner }),
    ).resolves.toEqual({
      success: true,
      stderr: "",
    });
    expect(commandRunner).toHaveBeenCalledOnce();
  });

  it("returns syntax errors from iverilog failures", async () => {
    const result = await run_iverilog("module broken", "broken_module", {
      commandRunner: async () => {
        throw { stderr: "syntax error" };
      },
    });
    expect(result).toEqual({
      success: false,
      stderr: "syntax error",
    });
  });

  it("converts WSL and Windows paths into Vivado-friendly paths", () => {
    expect(toVivadoPath("/mnt/c/Users/User/project/file with spaces.v")).toBe(
      "C:/Users/User/project/file with spaces.v",
    );
    expect(toVivadoPath("C:\\Users\\User\\project\\weights.hex")).toBe(
      "C:/Users/User/project/weights.hex",
    );
    expect(toVivadoPath("D:/fpga/out/report.rpt")).toBe("D:/fpga/out/report.rpt");
    expect(toVivadoPath("/home/user/project/module.v")).toBe("/home/user/project/module.v");
  });

  it("parses Vivado utilization and timing reports", () => {
    const report = [
      "| Slice LUTs*        | 1,234 |",
      "| Slice Registers    | 567   |",
      "| DSPs               | 8     |",
      "| RAMB36/FIFO*       | 2     |",
      "| RAMB18             | 1     |",
      "| WNS(ns) | TNS(ns) |",
      "| 2.500   | 0.000   |",
    ].join("\n");
    const parsed = parseVivadoReport(report, 20, "xc7a100tcsg324-1");
    expect(parsed).toMatchObject({
      success: true,
      tool: "vivado",
      part: "xc7a100tcsg324-1",
      stage: "synth",
      lut_count: 1234,
      ff_count: 567,
      dsp_count: 8,
      bram36_count: 2,
      bram18_count: 1,
      bram18_equiv: 5,
      wns_ns: 2.5,
      timing_met: true,
    });
    expect(parsed.fmax_mhz).toBeCloseTo(57.1428, 3);
  });

  it("marks Vivado timing as failed when WNS is negative", () => {
    const parsed = parseVivadoReport("WNS(ns): -1.250\n| Slice LUTs* | 4 |", 20);
    expect(parsed.timing_met).toBe(false);
    expect(parsed.fmax_mhz).toBeCloseTo(47.0588, 3);
  });

  it("returns fmax_mhz=0 when Vivado timing data is absent", () => {
    const parsed = parseVivadoReport("| Slice LUTs* | 3 |", 20);
    expect(parsed.wns_ns).toBeNull();
    expect(parsed.fmax_mhz).toBe(0);
    expect(parsed.timing_met).toBe(false);
  });

  it("runs vivado successfully when the command layer returns valid reports", async () => {
    const commandRunner = vi.fn(async (_file, args: string[], options) => {
      expect(args).toContain("-mode");
      expect(args).toContain("batch");
      expect(args).toContain("-notrace");
      expect(args.some((arg) => arg.includes("synth.tcl"))).toBe(true);
      expect(options?.cwd).toBeTruthy();
      const cwd = options?.cwd as string;
      const tcl = await readFile(path.join(cwd, "synth.tcl"), "utf8");
      const rtl = await readFile(path.join(cwd, "passthrough.v"), "utf8");
      expect(tcl).toContain("set_param general.maxThreads");
      expect(tcl).toContain("synth_design -top passthrough -part xc7a100tcsg324-1");
      expect(rtl).toContain('$readmemh("C:/Users/User/weights.hex", weights)');
      await writeFile(path.join(cwd, "post_synth_utilization.rpt"), "| Slice LUTs* | 4 |\n| Slice Registers | 2 |\n| DSPs | 1 |", "utf8");
      await writeFile(path.join(cwd, "post_synth_ram_utilization.rpt"), "| RAMB36/FIFO* | 1 |\n| RAMB18 | 0 |", "utf8");
      await writeFile(path.join(cwd, "post_synth_timing_summary.rpt"), "WNS(ns): 5.000", "utf8");
      return { stdout: "vivado ok", stderr: "" };
    });
    const result = await run_vivado(
      'module passthrough; reg [7:0] weights [0:0]; initial $readmemh("/mnt/c/Users/User/weights.hex", weights); endmodule',
      "passthrough",
      20,
      { commandRunner },
    );
    expect(commandRunner).toHaveBeenCalledOnce();
    expect(result.success).toBe(true);
    expect(result.lut_count).toBe(4);
    expect(result.ff_count).toBe(2);
    expect(result.dsp_count).toBe(1);
    expect(result.bram18_equiv).toBe(2);
    expect(result.fmax_mhz).toBeCloseTo(66.6667, 4);
  });

  it("returns a failure report when vivado execution fails", async () => {
    const result = await run_vivado("module bad; endmodule", "bad", 20, {
      commandRunner: async () => {
        throw { stderr: "vivado failed" };
      },
    });
    expect(result).toMatchObject({
      success: false,
      tool: "vivado",
      part: "xc7a100tcsg324-1",
      stage: "synth",
      lut_count: 0,
      ff_count: 0,
      dsp_count: 0,
      bram18_count: 0,
      bram36_count: 0,
      bram18_equiv: 0,
      wns_ns: null,
      timing_met: false,
      fmax_mhz: 0,
      report: "vivado failed",
    });
  });

  it("throws missing-binary errors from vivado as infrastructure failures", async () => {
    const missingBinary = Object.assign(new Error("spawn vivado ENOENT"), { code: "ENOENT" });
    await expect(
      run_vivado("module bad; endmodule", "bad", 20, {
        commandRunner: async () => {
          throw missingBinary;
        },
      }),
    ).rejects.toThrow("spawn vivado ENOENT");
  });

  it("throws vivado timeouts as infrastructure failures", async () => {
    const timeout = Object.assign(new Error("vivado timed out"), {
      killed: true,
      signal: "SIGTERM",
    });
    await expect(
      run_vivado("module bad; endmodule", "bad", 20, {
        commandRunner: async () => {
          throw timeout;
        },
      }),
    ).rejects.toThrow("vivado timed out");
  });

  it("uses explicit Vivado part and thread settings", async () => {
    const result = await run_vivado("module passthrough; endmodule", "passthrough", 10, "xc7a35tcpg236-1", 12, {
      commandRunner: async (_file, _args, options) => {
        const cwd = options?.cwd as string;
        const tcl = await readFile(path.join(cwd, "synth.tcl"), "utf8");
        expect(tcl).toContain("set_param general.maxThreads 12");
        expect(tcl).toContain("synth_design -top passthrough -part xc7a35tcpg236-1");
        await writeFile(path.join(cwd, "post_synth_utilization.rpt"), "| Slice LUTs* | 4 |", "utf8");
        await writeFile(path.join(cwd, "post_synth_timing_summary.rpt"), "WNS(ns): 1.000", "utf8");
        return { stdout: "", stderr: "" };
      },
    });
    expect(result).toMatchObject({
      success: true,
      part: "xc7a35tcpg236-1",
      lut_count: 4,
    });
  });

  it("writes Verilog source and metadata sidecars", async () => {
    const tempDir = await makeTempDir("nn2rtl-write-verilog-");
    const module = JSON.parse(
      await readFile(path.join(fixtureRoot, "verilog_module.json"), "utf8"),
    );

    const writtenPath = await write_verilog(module, tempDir);

    expect(resolveOutputRoot(tempDir)).toBe(path.resolve(process.cwd(), tempDir));
    expect(await readFile(writtenPath, "utf8")).toContain("module unit_module");
    expect(
      JSON.parse(await readFile(path.join(tempDir, "rtl", "unit_module.meta.json"), "utf8")),
    ).toEqual(module);
  });

  it("loads sidecars when present and returns null when missing", async () => {
    const tempDir = await makeTempDir("nn2rtl-sidecar-");
    const sidecarPath = await writeSidecar(tempDir);

    await expect(readSidecarIfPresent(sidecarPath)).resolves.toMatchObject({
      module_id: "unit_module",
      ready_in_signal: "ready_in",
    });
    await expect(readSidecarIfPresent(path.join(tempDir, "missing.json"))).resolves.toBeNull();
  });

  it("rejects invalid sidecars", async () => {
    const tempDir = await makeTempDir("nn2rtl-sidecar-invalid-");
    const sidecarPath = path.join(tempDir, "sidecar.json");
    await writeFile(sidecarPath, JSON.stringify({ module_name: "bad" }), "utf8");
    await expect(readSidecarIfPresent(sidecarPath)).rejects.toThrow("VerificationSidecar");
  });

  it("returns syntax_error when verilator compilation fails", async () => {
    const tempDir = await makeTempDir("nn2rtl-verilator-build-");
    const sidecarPath = await writeSidecar(tempDir);

    const result = await run_verilator("module unit_module; endmodule", "unit_module", sidecarPath, {
      commandRunner: async (file) => {
        if (file === VERILATOR_COMMAND) {
          throw { stderr: "compile boom" };
        }
        return { stdout: "", stderr: "" };
      },
    });

    expect(result.status).toBe("syntax_error");
    expect(result.verilator_stderr).toBe("compile boom");
  });

  it("classifies static testbench build failures as tb_setup_error", async () => {
    const tempDir = await makeTempDir("nn2rtl-verilator-tb-build-");
    const sidecarPath = await writeSidecar(tempDir);

    const result = await run_verilator("module unit_module; endmodule", "unit_module", sidecarPath, {
      commandRunner: async (file) => {
        if (file === VERILATOR_COMMAND) {
          throw {
            stderr:
              "static_verilator_tb.cpp:91: error: invalid operands to binary expression\n" +
              "note: candidate template ignored: substitution failure in 'VlWide<8>'",
          };
        }
        return { stdout: "", stderr: "" };
      },
    });

    expect(result).toMatchObject({
      module_id: "unit_module",
      status: "fail",
      status_class: "tb_setup_error",
      timing_actual_cycles: -1,
      timing_expected_cycles: 1,
    });
    expect(result.fix_hint).toContain("external C++ / bus-width diagnostics");
    expect(result.verilator_stderr).toContain("static_verilator_tb.cpp");
  });

  it("rejects non-absolute sidecar vector paths before invoking verilator", async () => {
    const tempDir = await makeTempDir("nn2rtl-verilator-sidecar-");
    const sidecarPath = await writeSidecar(tempDir, {
      golden_inputs_path: "relative-inputs.json",
    });

    await expect(
      run_verilator("module unit_module; endmodule", "unit_module", sidecarPath, {
        commandRunner: async () => ({ stdout: "", stderr: "" }),
      }),
    ).rejects.toThrow("must be an absolute path");
  });

  it("returns a fallback verification result when simulation finishes without results", async () => {
    const tempDir = await makeTempDir("nn2rtl-verilator-results-");
    const sidecarPath = await writeSidecar(tempDir);
    const commandRunner = vi.fn(async () => ({ stdout: "", stderr: "" }));

    const result = await run_verilator("module unit_module; endmodule", "unit_module", sidecarPath, {
      commandRunner,
    });

    expect(result).toMatchObject({
      module_id: "unit_module",
      status: "fail",
      status_class: "tb_setup_error",
      timing_actual_cycles: -1,
      timing_expected_cycles: 1,
    });
    expect(result.fix_hint).toContain("did not produce results JSON");
    expect(commandRunner).toHaveBeenCalledTimes(2);
  });

  it("validates emitted PipelineIR in read_weights when the script layer succeeds", async () => {
    const tempRepoRoot = await makeTempDir("nn2rtl-read-weights-");
    const pipelineIr = JSON.parse(
      await readFile(path.join(fixtureRoot, "pipeline_ir.json"), "utf8"),
    );
    const outputPath = path.join(tempRepoRoot, "output", "golden_vectors.json");
    await mkdir(path.dirname(outputPath), { recursive: true });
    await writeFile(outputPath, JSON.stringify(pipelineIr, null, 2), "utf8");

    const commandRunner = vi.fn(async () => ({ stdout: "", stderr: "" }));

    await expect(
      read_weights(
        "checkpoint.pth",
        { quantization: "int8" },
        {
          commandRunner,
          env: {
            ...process.env,
            NN2RTL_REPO_ROOT: tempRepoRoot,
          },
        },
      ),
    ).resolves.toEqual(pipelineIr);

    expect(commandRunner).toHaveBeenCalledOnce();
  });

  it("rejects invalid PipelineIR emitted by the Python layer", async () => {
    const tempRepoRoot = await makeTempDir("nn2rtl-read-weights-invalid-");
    const outputPath = path.join(tempRepoRoot, "output", "golden_vectors.json");
    await mkdir(path.dirname(outputPath), { recursive: true });
    await writeFile(outputPath, JSON.stringify({ nope: true }), "utf8");

    await expect(
      read_weights(
        "checkpoint.pth",
        { quantization: "int8" },
        {
          commandRunner: async () => ({ stdout: "", stderr: "" }),
          env: {
            ...process.env,
            NN2RTL_REPO_ROOT: tempRepoRoot,
          },
        },
      ),
    ).rejects.toThrow("is not a valid PipelineIR");
  });
});
