// Diagnostic: synth a single uram_weight_bank module standalone with
// MEMORY_PRIMITIVE="ultra". Faster way to iterate on XPM black-box errors than
// re-running the 4.5h full integrated synth.
//
// Tests:
//   - auto_detect_xpm vs no auto_detect
//   - READ_LATENCY values
//   - CASCADE_HEIGHT explicit vs 0
//
// Usage:
//   set NN2RTL_VIVADO_BIN=D:/vivado/2025.2/Vivado/bin/vivado.bat
//   npx tsx scripts/diag_uram_bank_only.ts

import { writeFile, mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

import {
  resolveVivadoCommand,
  toVivadoPath,
  withTempDir,
  VIVADO_MAX_BUFFER_BYTES,
} from "../mcp/tools.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

const part = "xcu250-figd2104-2L-e";
const tlogPath = path.join(repoRoot, "output", "reports_integrated", "diag_uram_bank.log");

function tclQuote(value: string): string {
  return `"${toVivadoPath(value).replace(/(["$[\]])/g, "\\$1")}"`;
}

// Minimal top: one uram_weight_bank with the same params as the production banks.
// We provide our own uram_weight_bank.v inline so we don't need to extract from nn2rtl_top.v.
function buildTestVerilog(): string {
  return `
\`timescale 1ns / 1ps
\`define NN2RTL_SYNTHESIS

module uram_weight_bank #(
    parameter integer DEPTH         = 96659,
    parameter integer ADDR_W        = 17,
    parameter         MEM_INIT_FILE = ""
) (
    input  wire                    clk,
    input  wire [ADDR_W-1:0]       rd_addr,
    output wire [287:0]            rd_data,
    input  wire                    rd_en
);
    xpm_memory_sprom #(
        .ADDR_WIDTH_A(ADDR_W),
        .AUTO_SLEEP_TIME(0),
        .CASCADE_HEIGHT(0),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE(MEM_INIT_FILE),
        .MEMORY_INIT_PARAM(""),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("ultra"),
        .MEMORY_SIZE(DEPTH * 288),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_A(288),
        .READ_LATENCY_A(2),
        .READ_RESET_VALUE_A("0"),
        .RST_MODE_A("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_MEM_INIT(1),
        .USE_MEM_INIT_MMI(0),
        .WAKEUP_TIME("disable_sleep")
    ) u_xpm (
        .douta(rd_data), .addra(rd_addr), .clka(clk), .ena(rd_en),
        .rsta(1'b0), .regcea(1'b1), .sleep(1'b0),
        .injectdbiterra(1'b0), .injectsbiterra(1'b0),
        .dbiterra(), .sbiterra()
    );
endmodule

module diag_top (
    input  wire        clk,
    input  wire [16:0] rd_addr,
    output wire [287:0] rd_data,
    input  wire        rd_en
);
    uram_weight_bank #(
        .DEPTH(96659), .ADDR_W(17),
        .MEM_INIT_FILE("${toVivadoPath(path.join(repoRoot, "output", "weights", "uram_weights_bank0.mem"))}")
    ) u_bank (
        .clk(clk), .rd_addr(rd_addr), .rd_data(rd_data), .rd_en(rd_en)
    );
endmodule
`;
}

function buildTcl(input: { verilogPath: string; utilRpt: string }): string {
  return [
    `set_param general.maxThreads 8`,
    `read_verilog -sv ${tclQuote(input.verilogPath)}`,
    `puts "NN2RTL_INFO: auto_detect_xpm before synth"`,
    `auto_detect_xpm`,
    `puts "NN2RTL_INFO: starting synth_design"`,
    `synth_design -top diag_top -part ${part} -flatten_hierarchy rebuilt`,
    `report_utilization -file ${tclQuote(input.utilRpt)}`,
    `puts "NN2RTL_INFO: diagnostic complete"`,
  ].join("\n") + "\n";
}

const execFileP = promisify(execFile);

async function main(): Promise<void> {
  await mkdir(path.dirname(tlogPath), { recursive: true });
  await withTempDir("nn2rtl-diag-uram-", async (tempDir) => {
    const vPath = path.join(tempDir, "diag.v");
    const utilRpt = path.join(tempDir, "diag_util.rpt");
    const tclPath = path.join(tempDir, "diag.tcl");
    await writeFile(vPath, buildTestVerilog(), "utf8");
    await writeFile(tclPath, buildTcl({ verilogPath: vPath, utilRpt }), "utf8");

    const vivadoBin = resolveVivadoCommand(process.env);
    const vivadoArgs = ["-mode", "batch", "-source", toVivadoPath(tclPath), "-notrace"];
    const isWindowsBatch = process.platform === "win32" && /\.(bat|cmd)$/i.test(vivadoBin);
    const spawnFile = isWindowsBatch ? "cmd.exe" : vivadoBin;
    const spawnArgs = isWindowsBatch ? ["/c", vivadoBin, ...vivadoArgs] : vivadoArgs;

    console.log(`[diag] launching vivado in ${tempDir}`);
    const t0 = Date.now();
    let stdout = "", stderr = "", exitOk = true;
    try {
      const res = await execFileP(spawnFile, spawnArgs, {
        cwd: tempDir, env: process.env,
        timeout: 1800 * 1000,  // 30 min hard cap
        maxBuffer: VIVADO_MAX_BUFFER_BYTES,
      });
      stdout = res.stdout; stderr = res.stderr;
    } catch (err: unknown) {
      exitOk = false;
      const e = err as { stdout?: string | Buffer; stderr?: string | Buffer; message?: string };
      stdout = typeof e.stdout === "string" ? e.stdout : (e.stdout?.toString() ?? "");
      stderr = typeof e.stderr === "string" ? e.stderr : (e.stderr?.toString() ?? e.message ?? "");
    }
    const elapsed = (Date.now() - t0) / 1000;
    const utilTxt = existsSync(utilRpt) ? await readFile(utilRpt, "utf8") : "";
    const vivadoLog = path.join(tempDir, "vivado.log");
    const logTxt = existsSync(vivadoLog) ? await readFile(vivadoLog, "utf8") : "";
    const combined = [`elapsed_s=${elapsed.toFixed(1)} exit_ok=${exitOk}`, "--- vivado.log ---", logTxt, "--- util.rpt ---", utilTxt, "--- stdout ---", stdout, "--- stderr ---", stderr].join("\n");
    await writeFile(tlogPath, combined, "utf8");
    console.log(`[diag] elapsed=${elapsed.toFixed(1)}s ok=${exitOk}, log -> ${path.relative(repoRoot, tlogPath)}`);
    if (exitOk) {
      const hasBlackBox = /black box|undefined contents|INBB-3/i.test(logTxt);
      const uramLine = utilTxt.split(/\r?\n/).find((l) => /URAM|UltraRAM/i.test(l));
      console.log(`[diag] verdict: ${hasBlackBox ? "FAIL (still black box)" : "PASS (no black box detected)"}`);
      if (uramLine) console.log(`[diag] URAM line: ${uramLine.trim()}`);
    }
  });
}

main().catch((err: unknown) => { console.error(err instanceof Error ? err.stack ?? err.message : String(err)); process.exit(1); });
