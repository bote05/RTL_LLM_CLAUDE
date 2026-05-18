import { execFile } from "node:child_process";
import { existsSync, readFileSync as readFileSyncNative } from "node:fs";
import { appendFile, copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";

import { pipelineIrSchema, verifResultSchema, verificationSidecarSchema } from "./schemas.js";
import type { PipelineIR, VerificationSidecar, VerifResult, VerilogModule } from "./types.js";

const execFileAsync = promisify(execFile);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(
  __dirname,
  path.basename(__dirname) === "dist" ? ".." : ".",
  "..",
);
export const TB_SOURCE_PATH = path.resolve(repoRoot, "tb", "static_verilator_tb.cpp");
export const TB_JSON_HPP_PATH = path.resolve(repoRoot, "tb", "third_party", "json.hpp");

// Handwritten library modules that generated RTL may instantiate. Each of
// run_iverilog / run_verilator / run_vivado copies these into its temp build
// dir and passes them as additional source files so "unknown module" errors
// don't appear at elaboration time. Add to this list when a new handwritten
// library module lands in rtl_library/.
export const RTL_LIBRARY_SOURCES: readonly string[] = [
  path.resolve(repoRoot, "rtl_library", "coord_scheduler.v"),
  path.resolve(repoRoot, "rtl_library", "line_buf_window.v"),
  path.resolve(repoRoot, "rtl_library", "conv_datapath.v"),
];
export const VIVADO_DEFAULT_PART = "xczu9eg-ffvb1156-2-e";
export const VIVADO_TIMEOUT_MS = 90 * 60 * 1000;
export const VIVADO_MAX_BUFFER_BYTES = 64 * 1024 * 1024;
// Sim-threading default is 0 (= no `--threads` flag, single-threaded
// model). Empirically, on the layers we run (stem 7x7 119M cycles, conv2
// 3x3 463M cycles), threaded Verilator was *slower* than single-threaded:
// stem with --threads 8 hit the 600s wall-clock cap; the same DUT with
// --threads 0 finished in 24s (~5 MHz). Threading helps only on much
// larger / asymmetric models where the cycle-update graph splits cleanly
// across cores; for our tightly-coupled split-architecture pipeline the
// inter-thread coordination overhead dominates. Override with
// `NN2RTL_VERILATOR_THREADS=N` if you want to test an alternative.
// Build parallelism is independent and stays high (16 by default) so the
// C++ compile finishes in seconds on a multi-core host.
const VERILATOR_DEFAULT_THREAD_CAP = 0;
const VERILATOR_DEFAULT_BUILD_JOB_CAP = 16;

// Hard wall-clock cap for the Verilator simulation binary (after build).
// Motivation: a partially-functioning FSM that fires valid_out intermittently
// never triggers the TB's hang_budget (which only catches total silence), so
// the binary can run forever. Capping at 10 min lets us fail fast and route
// the module to Surgeon with failure_class=verilator_timeout instead of
// burning wall-clock. Covers every layer; not ResNet-specific.
export const VERILATOR_SIM_TIMEOUT_MS = Number(process.env.NN2RTL_VERILATOR_SIM_TIMEOUT_MS ?? "") || 10 * 60 * 1000;

// Cap on how much simulator stdout is forwarded into the VerifResult. The
// captured text is meant for $display/$write probes Surgeon embeds during
// repair; multi-MHz simulations can otherwise produce hundreds of MB of
// output and blow up pipeline_state.json plus the Surgeon delegation prompt.
// 32 KiB head + 32 KiB tail keeps boundary diagnostics intact.
const VERILATOR_STDOUT_HEAD_BYTES = 32 * 1024;
const VERILATOR_STDOUT_TAIL_BYTES = 32 * 1024;

function truncateSimulationStdout(stdout: string): string | undefined {
  if (!stdout) return undefined;
  const total = stdout.length;
  if (total <= VERILATOR_STDOUT_HEAD_BYTES + VERILATOR_STDOUT_TAIL_BYTES) {
    return stdout;
  }
  const head = stdout.slice(0, VERILATOR_STDOUT_HEAD_BYTES);
  const tail = stdout.slice(total - VERILATOR_STDOUT_TAIL_BYTES);
  const elided = total - VERILATOR_STDOUT_HEAD_BYTES - VERILATOR_STDOUT_TAIL_BYTES;
  return `${head}\n…[${elided} bytes elided]…\n${tail}`;
}
export function resolveVivadoCommand(env: NodeJS.ProcessEnv = process.env): string {
  return env.NN2RTL_VIVADO_BIN ?? "vivado";
}

// On Windows, the `verilator` entry point is a Perl script that OSS CAD Suite
// does not ship Perl for; `verilator_bin.exe` is the native Windows binary and
// works directly. Linux/macOS installs ship the Perl wrapper with a working
// system Perl, so `verilator` is the right command there. An env override lets
// callers force a specific binary (e.g. a distro-managed `verilator`).
export const VERILATOR_COMMAND =
  process.env.NN2RTL_VERILATOR_BIN ??
  (process.platform === "win32" ? "verilator_bin" : "verilator");

// On Windows, the `python3` command often resolves to a broken Microsoft Store
// alias; `python` typically points at a real interpreter. On Linux/macOS
// `python3` is the conventional entry point. An env override lets callers
// force a specific interpreter (e.g. a virtualenv binary).
export const PYTHON_COMMAND =
  process.env.NN2RTL_PYTHON_BIN ??
  (process.platform === "win32" ? "python" : "python3");

function resolveHostParallelism(): number {
  try {
    const available = os.availableParallelism();
    if (Number.isInteger(available) && available > 0) {
      return available;
    }
  } catch {
    // Fall back below. availableParallelism can throw on unusual platforms.
  }
  return Math.max(1, os.cpus().length);
}

function parseIntegerEnv(
  env: NodeJS.ProcessEnv,
  name: string,
  minValue: number,
): number | undefined {
  const raw = env[name];
  if (raw === undefined || raw.trim() === "") {
    return undefined;
  }
  const parsed = Number(raw);
  return Number.isInteger(parsed) && parsed >= minValue ? parsed : undefined;
}

export function resolveVerilatorThreads(env: NodeJS.ProcessEnv = process.env): number {
  const envThreads = parseIntegerEnv(env, "NN2RTL_VERILATOR_THREADS", 0);
  if (envThreads !== undefined) {
    return envThreads;
  }
  const available = resolveHostParallelism();
  return available >= 2 ? Math.min(VERILATOR_DEFAULT_THREAD_CAP, available) : 0;
}

export function resolveVerilatorBuildJobs(env: NodeJS.ProcessEnv = process.env): number {
  const envJobs = parseIntegerEnv(env, "NN2RTL_VERILATOR_BUILD_JOBS", 0);
  if (envJobs !== undefined) {
    return envJobs;
  }
  return Math.min(VERILATOR_DEFAULT_BUILD_JOB_CAP, resolveHostParallelism());
}

function resolveTmpDirRoot(): string {
  // os.tmpdir() handles TMPDIR / TMP / TEMP / USERPROFILE / platform defaults
  // correctly across Windows, macOS, and Linux. Preserve an explicit absolute
  // TMPDIR override for callers that deliberately redirect tmp (e.g. the
  // Windows cross-env TMPDIR=/tmp workaround in the vitest scripts), but fall
  // through to os.tmpdir() for anything else rather than hardcoding "/tmp".
  const override = process.env.TMPDIR;
  if (override && (path.isAbsolute(override) || isWindowsAbsolutePath(override))) {
    return normalizePathForCurrentHost(override);
  }
  return os.tmpdir();
}

// On Windows, OSS CAD Suite binaries (Verilator, iverilog, and legacy yosys)
// need YOSYSHQ_ROOT set + both `bin/` and `lib/` prepended to PATH or they
// spawn silently and exit non-zero with no stderr (DLLs not found). The
// shipped `environment.bat` does this setup; Node's execFile inherits the
// parent process env, which usually does *not* have these set unless the
// user launched their shell from that batch file.
//
// Detect the suite root by walking up from the first known OSS CAD binary on
// PATH. If found, return an env object with YOSYSHQ_ROOT populated and
// bin/lib prepended to PATH. The env override NN2RTL_YOSYSHQ_ROOT lets
// callers force a specific location.
function resolveOssCadSuiteRoot(env: NodeJS.ProcessEnv): string | null {
  const override = env.NN2RTL_YOSYSHQ_ROOT;
  if (override) {
    return normalizePathForCurrentHost(override);
  }
  if (env.YOSYSHQ_ROOT) {
    return normalizePathForCurrentHost(env.YOSYSHQ_ROOT);
  }
  const pathVar = env.PATH ?? env.Path ?? "";
  const sep = process.platform === "win32" ? ";" : ":";
  const candidates = pathVar.split(sep).filter(Boolean);
  const binaries = process.platform === "win32"
    ? ["verilator.exe", "iverilog.exe", "yosys.exe", "yosys"]
    : ["verilator", "iverilog", "yosys"];
  for (const dir of candidates) {
    for (const bin of binaries) {
      if (existsSync(path.join(dir, bin))) {
        // bin dir -> suite root is one level up, IFF sibling `lib/` exists.
        const root = path.resolve(dir, "..");
        if (existsSync(path.join(root, "lib"))) {
          return root;
        }
      }
    }
  }
  return null;
}

// Verilator on Windows needs three things on PATH, in this order:
//  1. A modern g++ (w64devkit / GCC 13+). Without it the C++ compile step
//     fails on `-faligned-new`, `-fcf-protection=none`, etc.
//  2. A real python3 (oss-cad-suite ships one at `lib/python3.exe`). Without
//     it, Verilator's makefile hits the Microsoft Store `python3.exe` shim
//     which prints "Python was not found" and exits with error 0x2331.
//  3. oss-cad-suite's own DLLs (lib/) for anything it shells out to.
// We layer both augmentations: Verilator first (so its g++ wins), then
// oss-cad-suite lib (without touching g++). Override the C++ toolchain
// with NN2RTL_WIN_CXX_TOOLCHAIN_BIN.
export function augmentEnvForVerilatorCxx(env: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  if (process.platform !== "win32") {
    return env;
  }
  const candidates = [
    env.NN2RTL_WIN_CXX_TOOLCHAIN_BIN,
    env.USERPROFILE ? path.join(env.USERPROFILE, "w64devkit", "bin") : undefined,
    "C:\\w64devkit\\bin",
  ].filter((c): c is string => typeof c === "string" && c.length > 0);
  for (const dir of candidates) {
    if (existsSync(path.join(dir, "g++.exe"))) {
      const augmented: NodeJS.ProcessEnv = { ...env };
      const currentPath = env.PATH ?? env.Path ?? "";
      const sep = ";";
      const norm = (d: string) => path.resolve(d).toLowerCase();
      const target = norm(dir);
      const remaining = currentPath
        .split(sep)
        .filter((entry) => entry && norm(entry) !== target);
      const newPath = [dir, ...remaining].join(sep);
      augmented.PATH = newPath;
      augmented.Path = newPath;
      return augmented;
    }
  }
  return env;
}

// Like augmentEnvForOssCadSuite but only prepends `lib/` (DLLs + python3.exe).
// Used for Verilator, which needs oss-cad-suite's python3 but must NOT see
// oss-cad-suite's older g++ (that lives elsewhere on PATH; we want the
// modern w64devkit g++ from augmentEnvForVerilatorCxx to win).
export function augmentEnvForOssCadSuiteLibOnly(env: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const root = resolveOssCadSuiteRoot(env);
  if (!root) {
    return env;
  }
  const augmented: NodeJS.ProcessEnv = { ...env, YOSYSHQ_ROOT: root };
  const sep = process.platform === "win32" ? ";" : ":";
  const libDir = path.join(root, "lib");
  const currentPath = env.PATH ?? env.Path ?? "";
  const norm = (dir: string) => path.resolve(dir).toLowerCase();
  const target = norm(libDir);
  const remaining = currentPath
    .split(sep)
    .filter((entry) => entry && norm(entry) !== target);
  const newPath = [libDir, ...remaining].join(sep);
  augmented.PATH = newPath;
  if (process.platform === "win32") {
    augmented.Path = newPath;
  }
  return augmented;
}

export function augmentEnvForOssCadSuite(env: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const root = resolveOssCadSuiteRoot(env);
  if (!root) {
    return env;
  }
  const augmented: NodeJS.ProcessEnv = { ...env, YOSYSHQ_ROOT: root };
  const sep = process.platform === "win32" ? ";" : ":";
  const binDir = path.join(root, "bin");
  const libDir = path.join(root, "lib");
  // On Windows, Node exposes process.env with BOTH `PATH` and `Path`
  // populated (duplicated). If we update only one, the child process may
  // read the other and see the un-augmented value. So read from whichever
  // is set and write to BOTH keys.
  const currentPath = env.PATH ?? env.Path ?? "";
  // Strip any existing copies of binDir/libDir and prepend them — Windows
  // DLL resolution walks PATH in order, and if another toolchain (e.g.
  // git's mingw64) appears earlier it can load an incompatible
  // libstdc++/libgcc before oss-cad-suite's own DLLs are seen. Force the suite
  // to win by putting it at position 0.
  const norm = (dir: string) => path.resolve(dir).toLowerCase();
  const targets = new Set([norm(binDir), norm(libDir)]);
  const remaining = currentPath
    .split(sep)
    .filter((entry) => entry && !targets.has(norm(entry)));
  const newPath = [binDir, libDir, ...remaining].join(sep);
  augmented.PATH = newPath;
  if (process.platform === "win32") {
    augmented.Path = newPath;
  }
  const certFile = path.join(root, "etc", "cacert.pem");
  if (!augmented.SSL_CERT_FILE && existsSync(certFile)) {
    augmented.SSL_CERT_FILE = certFile;
  }
  return augmented;
}

type CommandOptions = {
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  timeout?: number;
  maxBuffer?: number;
};

type CommandResult = {
  stderr: string;
  stdout: string;
};

export type CommandRunner = (
  file: string,
  args: string[],
  options?: CommandOptions,
) => Promise<CommandResult>;

export type ToolsRuntime = {
  commandRunner: CommandRunner;
  cwd: string;
  env: NodeJS.ProcessEnv;
  tmpDirRoot: string;
};

const DEFAULT_TOOLS_RUNTIME: ToolsRuntime = {
  async commandRunner(file, args, options) {
    const result = await execFileAsync(file, args, options);
    const stdoutRaw: unknown = result.stdout;
    const stderrRaw: unknown = result.stderr;
    return {
      stdout: typeof stdoutRaw === "string" ? stdoutRaw : Buffer.isBuffer(stdoutRaw) ? stdoutRaw.toString("utf8") : String(stdoutRaw ?? ""),
      stderr: typeof stderrRaw === "string" ? stderrRaw : Buffer.isBuffer(stderrRaw) ? stderrRaw.toString("utf8") : String(stderrRaw ?? ""),
    };
  },
  cwd: repoRoot,
  env: process.env,
  tmpDirRoot: resolveTmpDirRoot(),
};

export function createToolsRuntime(
  overrides: Partial<ToolsRuntime> = {},
): ToolsRuntime {
  return {
    ...DEFAULT_TOOLS_RUNTIME,
    ...overrides,
  };
}

/**
 * Copy every handwritten `rtl_library/*.v` source into `tempDir` so that
 * iverilog / Verilator / Vivado see them alongside the candidate module
 * file at elaboration time. Returns the list of copied paths (absolute,
 * inside tempDir) so callers can include them as additional source args.
 * Silently skips any library file that does not exist on disk — the
 * caller's elaboration still fails loudly on any `unknown module`
 * reference so the gap is visible.
 */
export async function copyRtlLibrarySources(tempDir: string): Promise<string[]> {
  const copied: string[] = [];
  for (const srcPath of RTL_LIBRARY_SOURCES) {
    if (!existsSync(srcPath)) continue;
    const destPath = path.join(tempDir, path.basename(srcPath));
    await copyFile(srcPath, destPath);
    copied.push(destPath);
  }
  return copied;
}

export async function withTempDir<T>(
  prefix: string,
  fn: (tempDir: string) => Promise<T>,
  runtime: ToolsRuntime = createToolsRuntime(),
): Promise<T> {
  const tempDir = await mkdtemp(path.join(runtime.tmpDirRoot, prefix));
  let result: T;
  try {
    result = await fn(tempDir);
  } finally {
    // Windows + Vivado: child processes can hold file handles inside the
    // temp dir for tens of seconds after vivado.exe returns, so an immediate
    // rm fails with EBUSY/EPERM/ENOTEMPTY and torpedoes the otherwise-valid
    // synthesis result. Retry a few times, then leak the dir rather than
    // discard the report.
    for (let attempt = 0; attempt < 6; attempt += 1) {
      try {
        await rm(tempDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 500 });
        break;
      } catch (cleanupErr) {
        const code = (cleanupErr as { code?: string } | null)?.code;
        const transient = code === "EBUSY" || code === "EPERM" || code === "ENOTEMPTY";
        if (!transient || attempt === 5) {
          // Final attempt failed: leak the temp dir, don't mask the fn's result.
          break;
        }
        await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
      }
    }
  }
  return result;
}

// System-level spawn errors (ENOENT, EACCES, timeout, OOM) must not be
// laundered into Verilog syntax/synthesis failures: Surgeon would then try to
// "fix" an out-of-memory error by rewriting correct code. A genuine tool exit
// from an external tool has a numeric exit code on the Error object; a Node
// spawn failure surfaces `code` as a string like "ENOENT" and typically has no
// `signal` or `stdout`.
export function isSystemSpawnError(error: unknown): boolean {
  if (typeof error !== "object" || error === null) return false;
  const err = error as { code?: unknown; killed?: boolean; signal?: unknown };
  if (typeof err.code === "string") {
    const c = err.code;
    if (
      c === "ENOENT" ||
      c === "EACCES" ||
      c === "EPERM" ||
      c === "ENOMEM" ||
      c === "ETIMEDOUT" ||
      c === "EMFILE" ||
      c === "ENFILE" ||
      // Node's execFile kills the child when stdout/stderr exceeds
      // `maxBuffer`. That is infra (our buffer config is too small), not an
      // RTL bug — must not be routed to Surgeon as a synthesis failure.
      c === "ERR_CHILD_PROCESS_STDIO_MAXBUFFER"
    ) {
      return true;
    }
  }
  if (err.killed === true && err.signal) return true;
  return false;
}

export function stderrFromUnknown(error: unknown): string {
  const stderr = outputFieldFromUnknown(error, "stderr");
  if (stderr.length > 0) return stderr;

  const stdout = outputFieldFromUnknown(error, "stdout");
  if (stdout.length > 0) return stdout;

  if (error instanceof Error) {
    return error.message;
  }

  const summary = processErrorSummary(error);
  return summary || String(error);
}

function outputFieldFromUnknown(error: unknown, field: "stdout" | "stderr"): string {
  if (typeof error === "object" && error !== null && field in error) {
    const value = (error as { stdout?: string | Buffer; stderr?: string | Buffer })[field];
    if (typeof value === "string") return value;
    if (value instanceof Buffer) return value.toString("utf8");
  }
  return "";
}

function processErrorSummary(error: unknown): string {
  if (typeof error !== "object" || error === null) return "";

  const err = error as {
    code?: unknown;
    signal?: unknown;
    killed?: unknown;
    syscall?: unknown;
    path?: unknown;
    spawnargs?: unknown;
  };
  const fields: string[] = [];
  if (err.code !== undefined) fields.push(`exit_code=${String(err.code)}`);
  if (err.signal !== undefined && err.signal !== null) fields.push(`signal=${String(err.signal)}`);
  if (err.killed !== undefined) fields.push(`killed=${String(err.killed)}`);
  if (err.syscall !== undefined) fields.push(`syscall=${String(err.syscall)}`);
  if (err.path !== undefined) fields.push(`path=${String(err.path)}`);
  if (Array.isArray(err.spawnargs) && err.spawnargs.length > 0) {
    fields.push(`spawnargs=${err.spawnargs.map(String).join(" ")}`);
  }
  return fields.join(" ");
}

function toolFailureDiagnostic(
  toolName: string,
  args: readonly string[],
  cwd: string,
  error: unknown,
): string {
  const stderr = outputFieldFromUnknown(error, "stderr").trim();
  if (stderr.length > 0) return stderr;

  const stdout = outputFieldFromUnknown(error, "stdout").trim();
  if (stdout.length > 0) {
    return [
      `${toolName} exited non-zero and wrote diagnostics to stdout instead of stderr.`,
      stdout,
    ].join("\n\n");
  }

  const message = error instanceof Error ? error.message.trim() : "";
  const summary = processErrorSummary(error);

  const command = [toolName, ...args].join(" ");
  const lines = [
    `${toolName} exited non-zero without diagnostic output.`,
    `command: ${command}`,
    `cwd: ${cwd}`,
  ];
  if (message.length > 0 && message !== summary) {
    lines.push(`node_error: ${message}`);
  }
  lines.push(
    summary ? `process: ${summary}` : "process: no exit metadata was provided by Node.",
    "Treat this as a toolchain/runtime setup failure unless the same source produces a real compiler diagnostic when replayed.",
  );
  return lines.join("\n");
}

export type VivadoSynthesisReport = {
  success: boolean;
  tool: "vivado";
  part: string;
  stage: "synth";
  lut_count: number;
  ff_count: number;
  dsp_count: number;
  bram18_count: number;
  bram36_count: number;
  bram18_equiv: number;
  // Setup-path Worst Negative Slack from `report_timing_summary`. This is
  // what determines whether the design can run at the configured clock
  // frequency. `wns_ns` is the historical name kept for backward
  // compatibility; it always meant Setup WNS. `setup_wns_ns` is provided as
  // the explicit name so downstream code does not have to rely on the
  // historical convention.
  wns_ns: number | null;
  setup_wns_ns: number | null;
  // Hold-path Worst Hold Slack (Vivado's "WHS(ns)" column). On synth-only
  // flows this is reported against a pre-placement netlist where most small
  // hold violations get fixed automatically by `place_design` /
  // `opt_design`. We surface the number for visibility but do NOT gate
  // pass/fail on it — that is a P&R-stage concern, not synthesis.
  hold_wns_ns: number | null;
  timing_met: boolean;
  fmax_mhz: number;
  report: string;
};

function parseNumber(value: string | undefined): number {
  if (!value) return 0;
  const cleaned = value.replace(/,/g, "").trim();
  const parsed = Number(cleaned);
  return Number.isFinite(parsed) ? parsed : 0;
}

function firstVivadoTableValue(report: string, labels: RegExp[]): number {
  for (const label of labels) {
    const row = report.match(new RegExp(`\\|\\s*${label.source}\\s*\\|\\s*([0-9,.]+)`, "i"));
    if (row) return parseNumber(row[1]);
  }
  return 0;
}

function parseVivadoWns(report: string): number | null {
  const inline = report.match(/\bWNS(?:\(ns\))?\s*[:=]\s*(-?[0-9]+(?:\.[0-9]+)?)/i);
  if (inline) return Number(inline[1]);

  // Vivado's report_timing_summary "Design Timing Summary" looks like:
  //
  //     WNS(ns)      TNS(ns)  TNS Failing Endpoints  ...
  //     -------      -------  ---------------------  ...
  //      13.679        0.000                      0  ...
  //
  // i.e. column-oriented, separated by whitespace (no `|`). Match the
  // WNS(ns) header, skip the dashed separator row, then read the first
  // numeric token on the next non-empty line.
  const headerWithDashes = report.match(
    /WNS\(ns\)[^\n]*\n[^\n]*-{2,}[^\n]*\n\s+(-?[0-9]+(?:\.[0-9]+)?)/i,
  );
  if (headerWithDashes) return Number(headerWithDashes[1]);

  // Older / project-mode tables sometimes use a `|`-bordered grid.
  const headerPiped = report.match(/WNS\(ns\)[\s\S]{0,400}?\n\s*\|\s*(-?[0-9]+(?:\.[0-9]+)?)/i);
  if (headerPiped) return Number(headerPiped[1]);

  return null;
}

/**
 * Extract Vivado's "Worst Hold Slack" (WHS) from `report_timing_summary`.
 *
 * The Design Timing Summary table puts the values for all timing checks on
 * a single whitespace-separated row whose columns are, in order:
 *   WNS(ns)  TNS(ns)  TNS Failing  TNS Total  WHS(ns)  THS(ns)  THS Failing  THS Total  WPWS(ns)  ...
 *
 * Setup WNS is column 1; Hold WHS is column 5. We anchor on the dashed
 * separator below the header line and read the 5th whitespace-separated
 * numeric token. Returns null when no hold information is present (e.g. a
 * trivial pass-through with no inter-FF paths).
 */
function parseVivadoHoldWns(report: string): number | null {
  const inline = report.match(/\bWHS(?:\(ns\))?\s*[:=]\s*(-?[0-9]+(?:\.[0-9]+)?)/i);
  if (inline) return Number(inline[1]);

  // Anchor on the dashed separator under the table header. The values row
  // immediately follows. We grab the first 5 whitespace-separated tokens
  // (col 1 = setup WNS, cols 2-4 = setup TNS / failing / total, col 5 = WHS).
  const tableValues = report.match(
    /WHS\(ns\)[^\n]*\n[^\n]*-{2,}[^\n]*\n\s+(-?[0-9]+(?:\.[0-9]+)?)\s+(-?[0-9]+(?:\.[0-9]+)?)\s+(-?[0-9]+)\s+(-?[0-9]+)\s+(-?[0-9]+(?:\.[0-9]+)?)/i,
  );
  if (tableValues) return Number(tableValues[5]);

  return null;
}

export function parseVivadoReport(
  report: string,
  clock_period_ns: number,
  part: string = VIVADO_DEFAULT_PART,
): VivadoSynthesisReport {
  // Vivado labels these resources differently per device family:
  //   - Artix-7 / Kintex-7 7-series: `Slice LUTs*` and `Slice Registers`
  //   - UltraScale / UltraScale+: `CLB LUTs*` and `CLB Registers`
  // The trailing `*` is a literal asterisk Vivado prints (with a footnote
  // about post-implementation count); make it optional in the regex so we
  // match both column variants. `Register as Flip Flop` is the safe
  // fallback when the high-level rollup row is absent.
  const lut_count = firstVivadoTableValue(report, [/Slice LUTs\*?/, /CLB LUTs\*?/]);
  const ff_count = firstVivadoTableValue(report, [/Slice Registers/, /CLB Registers/, /Register as Flip Flop/]);
  const dsp_count = firstVivadoTableValue(report, [/DSPs/, /DSP48E1/]);
  const bram36_count = firstVivadoTableValue(report, [/RAMB36\/FIFO\*?/, /RAMB36/]);
  const bram18_count = firstVivadoTableValue(report, [/RAMB18/, /RAMB18E1/]);
  const block_ram_tiles = firstVivadoTableValue(report, [/Block RAM Tile/]);
  const bram18_equiv =
    bram18_count > 0 || bram36_count > 0
      ? bram18_count + bram36_count * 2
      : block_ram_tiles * 2;
  const setup_wns_ns = parseVivadoWns(report);
  const hold_wns_ns = parseVivadoHoldWns(report);
  // Vivado's Design Timing Summary prints `WNS = NA` for designs that have
  // no inter-FF setup paths -- typically a 1-cycle pass-through where every
  // register is driven only from primary inputs. The report still asserts
  // "All user specified timing constraints are met." in that case, so treat
  // it as timing_met = true even though there's no numeric WNS to extract.
  // Without this branch, trivially-meeting designs (e.g. a stream-through
  // ReLU) get classified as synth failures.
  const timingExplicitlyMet =
    /All user specified timing constraints are met/i.test(report);
  // Setup-only `timing_met` for synth-only flows. The "Timing constraints
  // are not met." string and per-path "Slack (VIOLATED)" labels Vivado
  // prints when HOLD is failing are not pass/fail signals at this stage —
  // synth runs against a pre-placement netlist where small hold violations
  // (typically tens of picoseconds) are routinely absorbed by
  // `place_design` and `opt_design`. We extract hold_wns_ns and surface it
  // for visibility, but the gate is setup. Real silicon hold validation
  // requires the implementation flow.
  //
  // The setup gate is the sign of setup_wns_ns alone: positive → passes
  // setup at the configured frequency. A NEGATIVE setup WNS would mean the
  // critical path can't make timing at the target clock — that is a real
  // problem P&R can't fix without RTL changes (more pipeline registers).
  //
  // We deliberately do NOT search the report text for "VIOLATED": Vivado
  // stamps `Slack (VIOLATED) : -0.033ns` against every individual hold
  // path that fails, which would re-introduce the synth-only hold-gate bug.
  const timing_met =
    setup_wns_ns !== null
      ? setup_wns_ns >= 0
      : timingExplicitlyMet;
  const critical_path_ns =
    setup_wns_ns !== null && clock_period_ns > 0
      ? clock_period_ns - setup_wns_ns
      : 0;
  const fmax_mhz =
    critical_path_ns > 0 ? 1_000 / critical_path_ns : 0;

  return {
    success: true,
    tool: "vivado",
    part,
    stage: "synth",
    lut_count,
    ff_count,
    dsp_count,
    bram18_count,
    bram36_count,
    bram18_equiv,
    wns_ns: setup_wns_ns,
    setup_wns_ns,
    hold_wns_ns,
    timing_met,
    fmax_mhz,
    report,
  };
}

export function toVivadoPath(inputPath: string): string {
  const normalized = inputPath.replace(/\\/g, "/");
  const wslDrive = normalized.match(/^\/mnt\/([a-zA-Z])(?:\/(.*))?$/);
  if (wslDrive) {
    const drive = wslDrive[1].toUpperCase();
    const rest = wslDrive[2] ?? "";
    return rest ? `${drive}:/${rest}` : `${drive}:/`;
  }
  return normalized;
}

export function resolveOutputRoot(outputDir: string): string {
  const hostPath = normalizePathForCurrentHost(outputDir);
  return path.isAbsolute(hostPath) || isWindowsAbsolutePath(hostPath)
    ? hostPath
    : path.resolve(process.cwd(), hostPath);
}

export function resolveRepoRootFromEnv(env: NodeJS.ProcessEnv = process.env): string {
  const override = env.NN2RTL_REPO_ROOT;
  return override ? normalizePathForCurrentHost(override) : repoRoot;
}

function outputDirForNetworkEnv(env: NodeJS.ProcessEnv): string | null {
  const networkId = env.NN2RTL_NETWORK_ID;
  if (!networkId) return null;
  try {
    const registry = JSON.parse(
      readFileSyncNative(path.join(resolveRepoRootFromEnv(env), "networks.json"), "utf8"),
    ) as { networks?: Array<{ id?: string; outputDir?: string }> };
    const network = registry.networks?.find((entry) => entry.id === networkId);
    return typeof network?.outputDir === "string" && network.outputDir.length > 0
      ? network.outputDir
      : null;
  } catch {
    return null;
  }
}

export function resolveActiveOutputRoot(env: NodeJS.ProcessEnv = process.env): string {
  const root = resolveRepoRootFromEnv(env);
  const raw =
    env.NN2RTL_OUTPUT_DIR ??
    outputDirForNetworkEnv(env) ??
    env.OUTPUT_DIR ??
    "output";
  const hostPath = normalizePathForCurrentHost(raw);
  return path.isAbsolute(hostPath) || isWindowsAbsolutePath(hostPath)
    ? hostPath
    : path.resolve(root, hostPath);
}

function requireAbsoluteSidecarPaths(sidecar: VerificationSidecar): void {
  const pathFields = [
    "golden_inputs_path",
    "golden_outputs_path",
    "results_path",
    "testbench_template_path",
  ] as const;

  for (const field of pathFields) {
    if (!isAbsoluteHostPath(sidecar[field])) {
      throw new Error(
        `run_verilator: sidecar field '${field}' must be an absolute path; got '${sidecar[field]}'.`,
      );
    }
  }
  if (sidecar.contract_id === "dram-backed-weights") {
    if (!sidecar.weights_path || !isAbsoluteHostPath(sidecar.weights_path)) {
      throw new Error(
        `run_verilator: dram-backed-weights sidecar field 'weights_path' must be an absolute path; got '${sidecar.weights_path ?? ""}'.`,
      );
    }
    for (const [index, bankPath] of (sidecar.weight_bank_paths ?? []).entries()) {
      if (!isAbsoluteHostPath(bankPath)) {
        throw new Error(
          `run_verilator: sidecar field 'weight_bank_paths[${index}]' must be an absolute path; got '${bankPath}'.`,
        );
      }
    }
  }
}

function isWindowsAbsolutePath(inputPath: string): boolean {
  return /^[a-zA-Z]:[\\/]/.test(inputPath);
}

function isAbsoluteHostPath(inputPath: string): boolean {
  return path.isAbsolute(inputPath) || isWindowsAbsolutePath(inputPath);
}

function normalizePathForCurrentHost(inputPath: string): string {
  const normalized = inputPath.replace(/\\/g, "/");
  if (process.platform !== "win32") {
    const drivePath = normalized.match(/^([a-zA-Z]):\/(.*)$/);
    if (drivePath) {
      return `/mnt/${drivePath[1].toLowerCase()}/${drivePath[2]}`;
    }
  }
  if (process.platform === "win32") {
    const wslPath = normalized.match(/^\/mnt\/([a-zA-Z])(?:\/(.*))?$/);
    if (wslPath) {
      const rest = wslPath[2] ?? "";
      return rest ? `${wslPath[1].toUpperCase()}:/${rest}` : `${wslPath[1].toUpperCase()}:/`;
    }
  }
  return normalized;
}

function normalizeSidecarPathsForCurrentHost(sidecar: VerificationSidecar): VerificationSidecar {
  return {
    ...sidecar,
    golden_inputs_path: normalizePathForCurrentHost(sidecar.golden_inputs_path),
    golden_outputs_path: normalizePathForCurrentHost(sidecar.golden_outputs_path),
    results_path: normalizePathForCurrentHost(sidecar.results_path),
    testbench_template_path: normalizePathForCurrentHost(sidecar.testbench_template_path),
    weights_path: sidecar.weights_path ? normalizePathForCurrentHost(sidecar.weights_path) : undefined,
    weight_bank_paths: sidecar.weight_bank_paths?.map(normalizePathForCurrentHost),
  };
}

async function readVerilatorResults(resultsPath: string): Promise<VerifResult> {
  const raw = await readFile(resultsPath, "utf8");
  const parsed: unknown = JSON.parse(raw);
  const validated = verifResultSchema.safeParse(parsed);
  if (!validated.success) {
    throw new Error(
      `run_verilator: results JSON at '${resultsPath}' failed schema validation:\n${JSON.stringify(validated.error.issues, null, 2)}`,
    );
  }
  return validated.data;
}

async function readVerilatorResultsIfPresent(resultsPath: string): Promise<VerifResult | null> {
  try {
    return await readVerilatorResults(resultsPath);
  } catch (error: unknown) {
    if (
      typeof error === "object" &&
      error !== null &&
      "code" in error &&
      (error as { code?: string }).code === "ENOENT"
    ) {
      return null;
    }
    throw error;
  }
}

export async function run_iverilog(
  verilog_source: string,
  module_name: string,
  runtimeOverrides: Partial<ToolsRuntime> = {},
): Promise<{ success: boolean; stderr: string }> {
  const runtime = createToolsRuntime(runtimeOverrides);
  return withTempDir("nn2rtl-iverilog-", async (tempDir) => {
    const verilogPath = path.join(tempDir, `${module_name}.v`);
    await writeFile(verilogPath, verilog_source, "utf8");
    const libraryPaths = await copyRtlLibrarySources(tempDir);
    const outputPath = path.join(tempDir, `${module_name}.ivvp`);
    const args = ["-o", outputPath, "-g2012", verilogPath, ...libraryPaths];
    const iverilogCommand = runtime.env.NN2RTL_IVERILOG_BIN || "iverilog";

    try {
      await runtime.commandRunner(
        iverilogCommand,
        args,
        {
          cwd: tempDir,
          env: augmentEnvForOssCadSuite(runtime.env),
        },
      );
      return { success: true, stderr: "" };
    } catch (error: unknown) {
      if (isSystemSpawnError(error)) {
        throw error;
      }
      return {
        success: false,
        stderr: toolFailureDiagnostic(iverilogCommand, args, tempDir, error),
      };
    }
  }, runtime);
}

export async function run_verilator(
  verilog_source: string,
  module_name: string,
  sidecar_path: string,
  runtimeOverrides: Partial<ToolsRuntime> = {},
): Promise<VerifResult> {
  const runtime = createToolsRuntime(runtimeOverrides);
  return withTempDir("nn2rtl-verilator-", async (tempDir) => {
    const rawSidecar = await readSidecarIfPresent(sidecar_path);
    if (!rawSidecar) {
      throw new Error(`run_verilator: sidecar '${sidecar_path}' was not found.`);
    }
    const sidecar = normalizeSidecarPathsForCurrentHost(rawSidecar);
    requireAbsoluteSidecarPaths(sidecar);

    // The sidecar carries `module_name` and `module_id` fields that the bench
    // never rechecks against the DUT it was given. If a caller passes a
    // mismatched pair we want to fail loudly here rather than silently build
    // the wrong module.
    if (sidecar.module_name !== module_name) {
      throw new Error(
        `run_verilator: sidecar.module_name='${sidecar.module_name}' does not match the module_name argument '${module_name}'.`,
      );
    }

    const verilogPath = path.join(tempDir, `${module_name}.v`);
    const tempTbPath = path.join(tempDir, "static_verilator_tb.cpp");
    const tempSidecarPath = path.join(tempDir, "sidecar.host.json");
    const tempJsonDir = path.join(tempDir, "third_party");
    const tempJsonPath = path.join(tempJsonDir, "json.hpp");

    await writeFile(verilogPath, verilog_source, "utf8");
    await writeFile(tempSidecarPath, `${JSON.stringify(sidecar, null, 2)}\n`, "utf8");
    await mkdir(tempJsonDir, { recursive: true });
    await copyFile(sidecar.testbench_template_path || TB_SOURCE_PATH, tempTbPath);
    await copyFile(TB_SOURCE_PATH, path.join(tempDir, "contract_tb_runtime.cpp"));
    await copyFile(TB_JSON_HPP_PATH, tempJsonPath);
    const libraryPaths = await copyRtlLibrarySources(tempDir);
    const libraryBasenames = libraryPaths.map((p) => path.basename(p));

    try {
      // Simulation-speed flags. Measured on layer1_0_conv2 (3×3 spatial conv,
      // IC=OC=64, 112×112, MP=4): default build crawls at ~0.7 MHz and can't
      // clear one 463M-cycle frame inside VERILATOR_SIM_TIMEOUT_MS. With these
      // flags the same DUT runs ~1.4 MHz and completes frame 1 + hits the TB's
      // hang_budget cleanly in ~5.5 min, letting Surgeon see real evidence
      // instead of a synthesised timeout result.
      //   -O3                   Verilator-level opts (also forces OPT_FAST=-O3)
      //   --x-assign/x-initial  skip X-propagation bookkeeping
      //   -march=native         let g++ vectorize with the host's AVX2 (+AVX-512
      //                         on SKUs that have it) — safe because the binary
      //                         is only ever executed on the machine that built
      //                         it (temp dir, single shot).
      //   -DNDEBUG              kill Verilator runtime asserts.
      //   --threads N           generate a threaded simulation model. Default
      //                         is capped at 8 workers; set
      //                         NN2RTL_VERILATOR_THREADS=0 to disable.
      //   -j N                  parallelize the C++ build. Default is capped
      //                         at 16 jobs; override with
      //                         NN2RTL_VERILATOR_BUILD_JOBS.
      //   -MAKEFLAGS ...        Verilator 4.x appends its threaded-runtime
      //                         C++ standard flag after -CFLAGS; force that
      //                         generated make variable to C++17 too.
      const verilatorThreads = resolveVerilatorThreads(runtime.env);
      const verilatorBuildJobs = resolveVerilatorBuildJobs(runtime.env);
      const verilatorThreadArgs =
        verilatorThreads > 0 ? ["--threads", String(verilatorThreads)] : [];
      const verilatorBuildArgs =
        verilatorBuildJobs > 0 ? ["-j", String(verilatorBuildJobs)] : [];
      await runtime.commandRunner(
        VERILATOR_COMMAND,
        [
          "--cc",
          "--exe",
          "--build",
          ...verilatorBuildArgs,
          "--Mdir",
          "obj_dir",
          ...verilatorThreadArgs,
          "-O3",
          "-MAKEFLAGS",
          "CFG_CXXFLAGS_STD_NEWEST=-std=c++17",
          "--x-assign",
          "fast",
          "--x-initial",
          "fast",
          "-Wall",
          "-Wno-fatal",
          "--top-module",
          module_name,
          "-CFLAGS",
          `-std=c++17 -O3 -march=native -DNDEBUG -DVMODEL_HEADER="\\\"V${module_name}.h\\\"" -DVMODEL_CLASS=V${module_name}`,
          "static_verilator_tb.cpp",
          `${module_name}.v`,
          ...libraryBasenames,
        ],
        { cwd: tempDir, env: augmentEnvForVerilatorCxx(augmentEnvForOssCadSuiteLibOnly(runtime.env)) },
      );
    } catch (error: unknown) {
      if (isSystemSpawnError(error)) {
        throw error;
      }
      const stderr = stderrFromUnknown(error);
      const tbSetupFailure =
        /static_verilator_tb\.cpp|VlWide<|verilated_types\.h|VMODEL_HEADER|VMODEL_CLASS/.test(stderr);
      return {
        module_id: sidecar.module_id,
        status: tbSetupFailure ? "fail" : "syntax_error",
        status_class: tbSetupFailure ? "tb_setup_error" : undefined,
        timing_pass: false,
        timing_actual_cycles: tbSetupFailure ? -1 : 0,
        timing_expected_cycles: sidecar.pipeline_latency_cycles,
        verilator_stderr: stderr,
        fix_hint: tbSetupFailure
          ? `Static Verilator testbench build failed while compiling '${module_name}'. The RTL may be fine; inspect the external C++ / bus-width diagnostics before attempting module-local repair.`
          : `Verilator build failed while compiling '${module_name}' with the static testbench.`,
      };
    }

    const binaryName = `V${module_name}${process.platform === "win32" ? ".exe" : ""}`;
    const binaryPath = path.join(tempDir, "obj_dir", binaryName);
    let simulationError: unknown = null;
    let simulationTimedOut = false;
    let simulationStdout = "";

    try {
      const simResult = await runtime.commandRunner(binaryPath, [tempSidecarPath], {
        cwd: tempDir,
        env: augmentEnvForVerilatorCxx(augmentEnvForOssCadSuiteLibOnly(runtime.env)),
        timeout: VERILATOR_SIM_TIMEOUT_MS,
      });
      simulationStdout = simResult.stdout ?? "";
    } catch (error: unknown) {
      simulationStdout = outputFieldFromUnknown(error, "stdout");
      simulationError = error;
      // Node's execFile marks `killed=true` + `signal` set when it reaps a
      // child whose wall-clock exceeded `timeout`. Distinguish that from a
      // genuine non-zero exit so Surgeon gets the right failure class.
      if (
        typeof error === "object" &&
        error !== null &&
        (error as { killed?: boolean }).killed === true &&
        (error as { signal?: unknown }).signal
      ) {
        simulationTimedOut = true;
      }
    }

    const truncatedStdout = truncateSimulationStdout(simulationStdout);

    if (simulationTimedOut) {
      return {
        module_id: sidecar.module_id,
        status: "fail",
        status_class: "sim_stalled",
        timing_pass: false,
        timing_actual_cycles: -1,
        timing_expected_cycles: sidecar.pipeline_latency_cycles,
        expected: [],
        got: [],
        failure_class: "verilator_timeout",
        verilator_stderr: stderrFromUnknown(simulationError),
        verilator_stdout: truncatedStdout,
        fix_hint: [
          `Verilator simulation exceeded the ${VERILATOR_SIM_TIMEOUT_MS / 1000}s wall-clock cap.`,
          "The TB's hang_budget only fires on total valid_out silence, so a timeout means the FSM is",
          "structurally wrong enough that it either never completes the output stream or fires valid_out",
          "intermittently forever. Look at FSM exit conditions, output-counter bounds, and any state that",
          "re-enters a wait on a signal that can never arrive. Do not assume the RTL is partially correct.",
        ].join(" "),
      };
    }

    const parsedResults = await readVerilatorResultsIfPresent(sidecar.results_path);
    if (parsedResults) {
      if (truncatedStdout !== undefined) {
        parsedResults.verilator_stdout = truncatedStdout;
      }
      return parsedResults;
    }

    return {
      module_id: sidecar.module_id,
      status: "fail",
      status_class: "tb_setup_error",
      timing_pass: false,
      timing_actual_cycles: -1,
      timing_expected_cycles: sidecar.pipeline_latency_cycles,
      expected: [],
      got: [],
      failure_class: null,
      verilator_stderr: simulationError ? stderrFromUnknown(simulationError) : "",
      verilator_stdout: truncatedStdout,
      fix_hint: `Static testbench did not produce results JSON at '${sidecar.results_path}'.`,
    };
  }, runtime);
}

function tclQuote(value: string): string {
  return `"${toVivadoPath(value).replace(/(["$[\]])/g, "\\$1")}"`;
}

function resolveVivadoThreads(
  explicitThreads: number | undefined,
  env: NodeJS.ProcessEnv,
): number {
  if (explicitThreads && Number.isInteger(explicitThreads) && explicitThreads > 0) {
    return explicitThreads;
  }
  const envThreads = Number(env.NN2RTL_VIVADO_THREADS ?? "");
  if (Number.isInteger(envThreads) && envThreads > 0) {
    return envThreads;
  }
  return resolveHostParallelism();
}

function convertReadmemhPathsForVivado(verilogSource: string): string {
  return verilogSource.replace(
    /(\$readmemh\s*\(\s*)"([^"]+)"/g,
    (_match, prefix: string, filePath: string) => `${prefix}"${toVivadoPath(filePath)}"`,
  );
}

function buildVivadoTcl(input: {
  module_name: string;
  part: string;
  clock_period_ns: number;
  threads: number;
  verilog_paths: string[];
  util_report_path: string;
  ram_report_path: string;
  timing_report_path: string;
  checkpoint_path: string;
}): string {
  const clockPeriod = input.clock_period_ns > 0 ? input.clock_period_ns : 20;
  // Order matters in Vivado batch / non-project mode:
  //   1. read_verilog              -- load source(s)
  //   2. synth_design              -- creates the in-memory design (without
  //                                   this, every command that touches the
  //                                   design — create_clock, report_timing —
  //                                   fails with "No open design").
  //   3. create_clock              -- now the `clk` port exists in the design
  //                                   and can be constrained.
  //   4. report_* / write_checkpoint
  return [
    `set_param general.maxThreads ${input.threads}`,
    // Echo the value Vivado actually accepted so the report carries proof
    // of the parallelism setting. Vivado clamps `general.maxThreads` to a
    // version-specific cap (historically 8 on Windows; 32 on Linux and on
    // Vivado 2024+ Windows) — log both the requested and effective value.
    `puts "NN2RTL_INFO: requested general.maxThreads=${input.threads}, effective=[get_param general.maxThreads]"`,
    `read_verilog -sv ${input.verilog_paths.map(tclQuote).join(" ")}`,
    `synth_design -top ${input.module_name} -part ${input.part} -flatten_hierarchy rebuilt`,
    `create_clock -name clk -period ${clockPeriod} [get_ports clk]`,
    // Plain `report_utilization` (no `-hierarchical`) emits the row-oriented
    // summary table that `parseVivadoReport` consumes:
    //   | Site Type            | Used | ... |
    //   | Slice LUTs*          |   12 | ... |
    //   | Slice Registers      |   32 | ... |
    // The `-hierarchical` form is column-oriented per-instance and breaks the
    // parser. Use the summary form here; per-module designs don't need
    // hierarchy info anyway.
    `report_utilization -file ${tclQuote(input.util_report_path)}`,
    `if {[catch {report_ram_utilization -file ${tclQuote(input.ram_report_path)}} ram_err]} { puts "NN2RTL_WARN: report_ram_utilization failed: $ram_err" }`,
    `report_timing_summary -check_timing_verbose -max_paths 20 -file ${tclQuote(input.timing_report_path)}`,
    `write_checkpoint -force ${tclQuote(input.checkpoint_path)}`,
  ].join("\n") + "\n";
}

async function readTextIfPresent(filePath: string): Promise<string> {
  try {
    return await readFile(filePath, "utf8");
  } catch (error: unknown) {
    if (
      typeof error === "object" &&
      error !== null &&
      "code" in error &&
      (error as { code?: string }).code === "ENOENT"
    ) {
      return "";
    }
    throw error;
  }
}

export async function run_vivado(
  verilog_source: string,
  module_name: string,
  clockPeriodNsOrRuntimeOverrides: number | Partial<ToolsRuntime> = 0,
  partOrRuntimeOverrides: string | Partial<ToolsRuntime> | undefined = VIVADO_DEFAULT_PART,
  threadsOrRuntimeOverrides: number | Partial<ToolsRuntime> | undefined = undefined,
  runtimeOverrides: Partial<ToolsRuntime> = {},
): Promise<VivadoSynthesisReport> {
  const clock_period_ns =
    typeof clockPeriodNsOrRuntimeOverrides === "number" ? clockPeriodNsOrRuntimeOverrides : 0;
  const part =
    typeof partOrRuntimeOverrides === "string" ? partOrRuntimeOverrides : VIVADO_DEFAULT_PART;
  const explicitThreads =
    typeof threadsOrRuntimeOverrides === "number" ? threadsOrRuntimeOverrides : undefined;
  const overrides =
    typeof clockPeriodNsOrRuntimeOverrides !== "number"
      ? clockPeriodNsOrRuntimeOverrides
      : typeof partOrRuntimeOverrides !== "string"
        ? partOrRuntimeOverrides
        : typeof threadsOrRuntimeOverrides === "object" && threadsOrRuntimeOverrides !== null
          ? threadsOrRuntimeOverrides
          : runtimeOverrides;
  const runtime = createToolsRuntime(overrides);
  const vivadoTmpRoot = path.join(resolveRepoRootFromEnv(runtime.env), "output", "tmp");
  await mkdir(vivadoTmpRoot, { recursive: true });
  return withTempDir("nn2rtl-vivado-", async (tempDir) => {
    const threads = resolveVivadoThreads(explicitThreads, runtime.env);
    const verilogPath = path.join(tempDir, `${module_name}.v`);
    await writeFile(verilogPath, convertReadmemhPathsForVivado(verilog_source), "utf8");
    const libraryPaths = await copyRtlLibrarySources(tempDir);
    const utilReportPath = path.join(tempDir, "post_synth_utilization.rpt");
    const ramReportPath = path.join(tempDir, "post_synth_ram_utilization.rpt");
    const timingReportPath = path.join(tempDir, "post_synth_timing_summary.rpt");
    const checkpointPath = path.join(tempDir, "post_synth.dcp");
    const tclPath = path.join(tempDir, "synth.tcl");
    await writeFile(
      tclPath,
      buildVivadoTcl({
        module_name,
        part,
        clock_period_ns,
        threads,
        verilog_paths: [verilogPath, ...libraryPaths],
        util_report_path: utilReportPath,
        ram_report_path: ramReportPath,
        timing_report_path: timingReportPath,
        checkpoint_path: checkpointPath,
      }),
      "utf8",
    );

    let commandResult: { stdout: string; stderr: string };
    try {
      // Vivado on Windows ships as `vivado.bat`. Modern Node refuses to
      // `execFile` a `.bat` / `.cmd` without going through a shell (EINVAL
      // for security reasons), so route those through `cmd.exe /c` here.
      // POSIX builds of Vivado are real ELF binaries and execute directly.
      const vivadoBin = resolveVivadoCommand(runtime.env);
      const vivadoArgs = ["-mode", "batch", "-source", toVivadoPath(tclPath), "-notrace"];
      const isWindowsBatch =
        process.platform === "win32" && /\.(bat|cmd)$/i.test(vivadoBin);
      const spawnFile = isWindowsBatch ? "cmd.exe" : vivadoBin;
      const spawnArgs = isWindowsBatch ? ["/c", vivadoBin, ...vivadoArgs] : vivadoArgs;
      commandResult = await runtime.commandRunner(
        spawnFile,
        spawnArgs,
        {
          cwd: tempDir,
          env: runtime.env,
          timeout: VIVADO_TIMEOUT_MS,
          maxBuffer: VIVADO_MAX_BUFFER_BYTES,
        },
      );
    } catch (error: unknown) {
      if (isSystemSpawnError(error)) {
        throw error;
      }
      const stdout = outputFieldFromUnknown(error, "stdout");
      const stderr = stderrFromUnknown(error);
      return {
        success: false,
        tool: "vivado",
        part,
        stage: "synth",
        lut_count: 0,
        ff_count: 0,
        dsp_count: 0,
        bram18_count: 0,
        bram36_count: 0,
        bram18_equiv: 0,
        wns_ns: null,
        setup_wns_ns: null,
        hold_wns_ns: null,
        timing_met: false,
        fmax_mhz: 0,
        report: [stdout, stderr].filter(Boolean).join("\n"),
      };
    }

    const utilReport = await readTextIfPresent(utilReportPath);
    const ramReport = await readTextIfPresent(ramReportPath);
    const timingReport = await readTextIfPresent(timingReportPath);
    const combinedReport = [
      commandResult.stdout,
      commandResult.stderr,
      "--- post_synth_utilization.rpt ---",
      utilReport,
      "--- post_synth_ram_utilization.rpt ---",
      ramReport,
      "--- post_synth_timing_summary.rpt ---",
      timingReport,
    ].filter(Boolean).join("\n");

    return parseVivadoReport(combinedReport, clock_period_ns, part);
  }, { ...runtime, tmpDirRoot: vivadoTmpRoot });
}

export async function read_weights(
  checkpoint_path: string,
  quantization_config: object,
  runtimeOverrides: Partial<ToolsRuntime> = {},
): Promise<PipelineIR> {
  const runtime = createToolsRuntime(runtimeOverrides);
  const scriptPath = path.join(repoRoot, "scripts", "generate_golden.py");
  const outputRootRaw = runtime.env.NN2RTL_OUTPUT_DIR ?? "output";
  const hostOutputRoot = normalizePathForCurrentHost(outputRootRaw);
  const outputRoot = path.isAbsolute(hostOutputRoot) || isWindowsAbsolutePath(hostOutputRoot)
    ? hostOutputRoot
    : path.join(resolveRepoRootFromEnv(runtime.env), hostOutputRoot);
  const outputPath = path.join(outputRoot, "golden_vectors.json");
  const generateArgs = [scriptPath, checkpoint_path, "--output-dir", outputRoot];
  if (runtime.env.NN2RTL_NETWORK_ID) {
    generateArgs.push("--network", runtime.env.NN2RTL_NETWORK_ID);
  }

  await runtime.commandRunner(PYTHON_COMMAND, generateArgs, {
    cwd: runtime.cwd,
    env: {
      ...runtime.env,
      NN2RTL_NETWORK_ID: runtime.env.NN2RTL_NETWORK_ID,
      NN2RTL_OUTPUT_DIR: outputRoot,
      NN2RTL_QUANTIZATION_CONFIG: JSON.stringify(quantization_config),
    },
  });

  const raw = await readFile(outputPath, "utf8");
  const parsed: unknown = JSON.parse(raw);
  const validated = pipelineIrSchema.safeParse(parsed);
  if (!validated.success) {
    throw new Error(
      `read_weights: '${outputPath}' is not a valid PipelineIR:\n${JSON.stringify(validated.error.issues, null, 2)}`,
    );
  }
  return validated.data as PipelineIR;
}

export async function write_verilog(
  module: VerilogModule,
  output_dir: string,
): Promise<string> {
  const outputRoot = resolveOutputRoot(output_dir);
  const rtlDir = path.join(outputRoot, "rtl");
  const verilogPath = path.join(rtlDir, `${module.module_id}.v`);
  const metadataPath = path.join(rtlDir, `${module.module_id}.meta.json`);

  await mkdir(rtlDir, { recursive: true });
  await writeFile(verilogPath, module.verilog_source, "utf8");
  await writeFile(metadataPath, `${JSON.stringify(module, null, 2)}\n`, "utf8");

  return verilogPath;
}

// ---------------------------------------------------------------------------
// Pattern library lookup — Tier 0 + 1 of the pattern-library plan.
//
// Returns architectural guidance (`pattern_markdown`) plus an optional proven
// reference Verilog (`reference_verilog`) for an op_type + kernel combination.
// Foundry calls this before emitting RTL; Surgeon calls it when diagnosing
// synth / sim failures. All content lives under `knowledge/` — no external
// network calls.
// ---------------------------------------------------------------------------

export type GetRtlPatternsResult = {
  pattern_markdown: string;
  reference_verilog: string | null;
  license_notice: string | null;
};

export type GetFailureCorpusResult = {
  visible_tier: string;
  entries: Array<Record<string, unknown>>;
};

const PATTERN_LIBRARY_ROOT = path.resolve(repoRoot, "knowledge");
const DOC_LIFECYCLE_STATE_PATH = path.join(PATTERN_LIBRARY_ROOT, "doc_lifecycle.json");
export const KNOWLEDGE_READ_TIERS = ["protected", "active", "probationary"] as const;
export const KNOWLEDGE_ARCHIVE_TIER = "archive" as const;

type GeneratedDocTier = "active" | "probationary";
type GeneratedDocEntry = {
  id: string;
  op_type: string;
  contract_id?: string;
  contract_key?: string;
  signature_hashes?: string[];
  exact_reference_keys?: string[];
  applicability?: Record<string, unknown>;
  contraindications?: Array<string | Record<string, unknown>>;
  status: GeneratedDocTier | "archived";
  pattern_path?: string;
  reference_path?: string;
};
type GeneratedDocState = {
  docs?: Record<string, GeneratedDocEntry>;
};

async function tryReadText(absPath: string): Promise<string | null> {
  try {
    return await readFile(absPath, "utf8");
  } catch {
    return null;
  }
}

function resolvePatternPaths(
  op_type: string,
  kh?: number,
  kw?: number,
  contract_id = "flat-bus",
): string[] {
  const tieredPatternPaths = (fileName: string): string[] =>
    KNOWLEDGE_READ_TIERS.map((tier) =>
      path.join(PATTERN_LIBRARY_ROOT, "patterns", tier, fileName),
    );

  // Always include shared context + common-bugs files (when present).
  const paths: string[] = [
    ...tieredPatternPaths("01_context.md"),
    ...tieredPatternPaths("08_common_bugs.md"),
  ];
  if (contract_id === "dram-backed-weights") {
    paths.push(...tieredPatternPaths("09_dram_backed_weights.md"));
    return paths;
  }
  if (contract_id === "depthwise-conv") {
    paths.push(...tieredPatternPaths("12_depthwise_conv.md"));
    return paths;
  }
  if (contract_id !== "flat-bus") {
    return paths;
  }
  if (op_type === "conv2d") {
    if (kh === 1 && kw === 1) {
      paths.push(...tieredPatternPaths("02_conv1x1.md"));
    } else if (kh === 3 && kw === 3) {
      paths.push(...tieredPatternPaths("03_conv3x3_pad1.md"));
    } else if (kh === 7 && kw === 7) {
      paths.push(...tieredPatternPaths("04_conv7x7_pad3.md"));
    }
  } else if (op_type === "add") {
    paths.push(...tieredPatternPaths("05_add_quantized.md"));
  } else if (op_type === "relu") {
    paths.push(...tieredPatternPaths("06_relu.md"));
  } else if (op_type === "maxpool") {
    paths.push(...tieredPatternPaths("07_maxpool.md"));
  } else if (op_type === "global_avg_pool") {
    paths.push(...tieredPatternPaths("10_global_avg_pool.md"));
  } else if (op_type === "gemm") {
    paths.push(...tieredPatternPaths("11_gemm.md"));
  }
  return paths;
}

async function readDocLifecycleState(): Promise<GeneratedDocState> {
  try {
    const raw = await readFile(DOC_LIFECYCLE_STATE_PATH, "utf8");
    const parsed = JSON.parse(raw) as unknown;
    return typeof parsed === "object" && parsed !== null ? (parsed as GeneratedDocState) : {};
  } catch {
    return {};
  }
}

function lifecycleDocsFor(
  state: GeneratedDocState,
  op_type: string,
  kind: "pattern" | "reference",
  contract_id?: string,
  lookup?: CorpusLookupInput,
): Array<{ path: string; contract_id?: string; contract_key?: string }> {
  const docs = Object.values(state.docs ?? {});
  const tierOrder: Record<GeneratedDocTier, number> = { active: 0, probationary: 1 };
  const ranked = docs
    .map((doc) => ({
      doc,
      matchLevel: corpusMatchLevel({
        op_type: doc.op_type,
        contract_id: doc.contract_id ?? "flat-bus",
        signature_hashes: doc.signature_hashes,
        exact_reference_keys: doc.exact_reference_keys,
        applicability: doc.applicability,
        contraindications: doc.contraindications,
      }, {
        op_type,
        contract_id,
        runtime_layer_signature: lookup?.runtime_layer_signature,
        signature_hash: lookup?.signature_hash,
        exact_reference_key: lookup?.exact_reference_key,
      }),
    }))
    .filter(
      (entry): entry is { doc: GeneratedDocEntry & { status: GeneratedDocTier }; matchLevel: CorpusMatchLevel } =>
        entry.matchLevel !== null &&
        (entry.doc.status === "active" || entry.doc.status === "probationary"),
    )
    .sort((a, b) => {
      const matchDelta = CORPUS_MATCH_RANK[a.matchLevel] - CORPUS_MATCH_RANK[b.matchLevel];
      if (matchDelta !== 0) return matchDelta;
      const tierDelta = tierOrder[a.doc.status] - tierOrder[b.doc.status];
      return tierDelta !== 0 ? tierDelta : a.doc.id.localeCompare(b.doc.id);
    })
    .map(({ doc }) => ({
      doc,
      path: kind === "pattern" ? doc.pattern_path : doc.reference_path,
      contract_id: doc.contract_id,
      contract_key: doc.contract_key,
    }));

  type InjectedEntry = {
    doc: GeneratedDocEntry;
    path: string;
    contract_id?: string;
    contract_key?: string;
  };
  const injected: InjectedEntry[] = [];
  for (const entry of ranked) {
    if (typeof entry.path !== "string" || entry.path.length === 0) continue;
    injected.push({
      doc: entry.doc,
      path: path.resolve(repoRoot, entry.path),
      contract_id: entry.contract_id,
      contract_key: entry.contract_key,
    });
  }

  // Architectural dedup. Two references with the same structural shape
  // (contract + op + kernel + stride + groups + channel_tile + quant_family)
  // teach the same template; injecting both wastes Foundry's context budget
  // for zero added signal. The signature ladder already sorted by match
  // quality + tier, so keeping the FIRST entry per architectural fingerprint
  // preserves the strongest representative of each distinct architecture.
  // Different channel counts / bus widths within the same fingerprint do
  // NOT count as different architecture — the design pattern is identical,
  // only the parameters scale.
  const seenFingerprints = new Set<string>();
  const seenPaths = new Set<string>();
  const deduped: InjectedEntry[] = [];
  for (const entry of injected) {
    if (seenPaths.has(entry.path)) continue;
    seenPaths.add(entry.path);
    const fp = architecturalFingerprint(entry.doc);
    if (seenFingerprints.has(fp)) continue;
    seenFingerprints.add(fp);
    deduped.push(entry);
  }
  return deduped.map(({ path: p, contract_id: cid, contract_key: ck }) => ({
    path: p,
    contract_id: cid,
    contract_key: ck,
  }));
}

// Build a stable "architecture" key from a lifecycle doc. Two docs with the
// same key describe the same RTL template shape and should not both be
// injected — they teach the same lesson. Intentionally excludes channel
// counts and bus widths (those scale the template, they don't change it).
function architecturalFingerprint(doc: GeneratedDocEntry): string {
  const app = (doc.applicability ?? {}) as Record<string, unknown>;
  const pairLike = (v: unknown): string => {
    if (Array.isArray(v) && v.length >= 2) return `${v[0]}x${v[1]}`;
    if (typeof v === "number") return `${v}`;
    if (typeof v === "string") return v;
    return "?";
  };
  const numLike = (v: unknown): string => {
    if (typeof v === "number") return `${v}`;
    if (typeof v === "string") return v;
    return "?";
  };
  return [
    doc.op_type ?? "?",
    doc.contract_id ?? "?",
    pairLike(app.kernel),
    pairLike(app.stride),
    pairLike(app.dilation),
    numLike(app.groups),
    numLike(app.channel_tile),
    numLike(app.quantization_family),
  ].join("|");
}

function resolveReferencePaths(fileName: string): string[] {
  return KNOWLEDGE_READ_TIERS.map((tier) =>
    path.join(PATTERN_LIBRARY_ROOT, "references", tier, fileName),
  );
}

async function readReferenceVariants(fileName: string, extraPaths: string[] = []): Promise<string | null> {
  const variants: Array<{ tier: string; text: string }> = [];
  for (const p of [...resolveReferencePaths(fileName), ...extraPaths]) {
    const text = await tryReadText(p);
    if (text !== null) {
      const rel = path.relative(PATTERN_LIBRARY_ROOT, p);
      variants.push({ tier: rel || path.basename(path.dirname(p)), text });
    }
  }
  if (variants.length === 0) {
    return null;
  }
  if (variants.length === 1) {
    return variants[0].text;
  }
  return variants
    .map(({ tier, text }) => `// ---- knowledge/references/${tier}/${fileName} ----\n${text.trim()}\n`)
    .join("\n");
}

export async function get_rtl_patterns(
  op_type: string,
  kernel_h?: number,
  kernel_w?: number,
  contract_id?: string,
  signature_hash?: string,
  exact_reference_key?: string | null,
  runtime_layer_signature?: Record<string, unknown>,
): Promise<GetRtlPatternsResult> {
  const lifecycleState = await readDocLifecycleState();
  const lookup: CorpusLookupInput = {
    op_type,
    contract_id,
    signature_hash,
    exact_reference_key,
    runtime_layer_signature: runtime_layer_signature ?? {
      op_type,
      contract_id: contract_id ?? "flat-bus",
      ...(kernel_h !== undefined || kernel_w !== undefined
        ? { kernel: [kernel_h ?? null, kernel_w ?? kernel_h ?? null] }
        : {}),
    },
  };
  const lifecyclePatternDocs = lifecycleDocsFor(lifecycleState, op_type, "pattern", contract_id, lookup);
  const patternPaths = [
    ...resolvePatternPaths(op_type, kernel_h, kernel_w, contract_id ?? "flat-bus"),
    ...lifecyclePatternDocs.map((doc) => doc.path),
  ];
  const sections: string[] = [];
  for (const p of patternPaths) {
    const text = await tryReadText(p);
    if (text !== null) {
      const rel = path.relative(PATTERN_LIBRARY_ROOT, p);
      const tier = path.basename(path.dirname(p));
      const title =
        tier === "protected"
          ? path.basename(p)
          : rel;
      sections.push(`# ${title}\n\n${text.trim()}\n`);
    }
  }

  let reference_verilog: string | null = null;
  let license_notice: string | null = null;
  const lifecycleReferenceDocs = lifecycleDocsFor(lifecycleState, op_type, "reference", contract_id, lookup);
  const lifecycleReferencePaths = lifecycleReferenceDocs.map((doc) => doc.path);

  // Concrete worked-example wrappers — one per kernel shape we have a
  // proven reference for. Foundry adapts the localparams + $readmemh
  // paths from the LayerIR; the architecture (FSM / library
  // instantiation / start_pulse) stays identical.
  const includeFlatBusProtectedReferences = (contract_id ?? "flat-bus") === "flat-bus";
  const isDramBacked = contract_id === "dram-backed-weights";
  if (isDramBacked && op_type === "conv2d" && kernel_h === 3 && kernel_w === 3) {
    // dram-backed conv has a fundamentally different MAC pipeline than the
    // flat-bus on-chip-weights references (AXI prefetch + ping-pong cache,
    // no shared conv_datapath). Use the contract-specific reference so
    // agents see the correct prefetch-guard / cache-loaded gating shape.
    const ref = await readReferenceVariants("conv3x3_drambacked_passing_reference.v", lifecycleReferencePaths);
    if (ref !== null) {
      reference_verilog = ref;
    }
  } else if (includeFlatBusProtectedReferences && op_type === "conv2d" && kernel_h === 1 && kernel_w === 1) {
    const ref = await readReferenceVariants("conv1x1_passing_reference.v", lifecycleReferencePaths);
    if (ref !== null) {
      reference_verilog = ref;
    }
  } else if (includeFlatBusProtectedReferences && op_type === "conv2d" && kernel_h === 3 && kernel_w === 3) {
    const ref = await readReferenceVariants("conv3x3_passing_reference.v", lifecycleReferencePaths);
    if (ref !== null) {
      reference_verilog = ref;
    }
  } else if (includeFlatBusProtectedReferences && op_type === "conv2d" && kernel_h === 7 && kernel_w === 7) {
    const ref = await readReferenceVariants("conv7x7_passing_reference.v", lifecycleReferencePaths);
    if (ref !== null) {
      reference_verilog = ref;
    }
  } else if (lifecycleReferencePaths.length > 0) {
    const refs: string[] = [];
    for (const p of lifecycleReferencePaths) {
      const text = await tryReadText(p);
      if (text !== null) {
        refs.push(`// ---- ${path.relative(PATTERN_LIBRARY_ROOT, p)} ----\n${text.trim()}\n`);
      }
    }
    reference_verilog = refs.length > 0 ? refs.join("\n") : null;
  }

  const pattern_markdown =
    sections.length > 0
      ? sections.join("\n---\n\n")
      : "No pattern available for this op_type yet. Proceed with foundry.md rules.";

  // Side-log every invocation so post-hoc analysis can confirm whether
  // Foundry / Surgeon actually called this tool. The log is append-only at
  // <repoRoot>/output/reports/tool_calls.jsonl. Never blocks the tool's
  // success path; a diagnostic goes to stderr (visible in pipeline_run.log)
  // if the write fails so a broken side-log doesn't go unnoticed.
  //
  // Path resolution: repoRoot is the anchor because the MCP server is
  // always launched with its dist/ sibling to mcp/, and repoRoot is
  // `mcp/..` under both `tsx` and compiled `dist` execution. `.mcp.json`
  // sets OUTPUT_DIR=../output (relative to the plugin dir), which is
  // the same absolute path; we use the env var when present to keep
  // this robust against plugin-dir layout changes.
  const candidateOutputRoots = [
    process.env.NN2RTL_OUTPUT_DIR,
    process.env.OUTPUT_DIR,
  ].filter((v): v is string => typeof v === "string" && v.length > 0);
  let outputRoot = path.resolve(repoRoot, "output");
  for (const candidate of candidateOutputRoots) {
    const hostCandidate = normalizePathForCurrentHost(candidate);
    const resolved = path.isAbsolute(hostCandidate) || isWindowsAbsolutePath(hostCandidate)
      ? hostCandidate
      : path.resolve(repoRoot, candidate);
    if (existsSync(resolved)) {
      outputRoot = resolved;
      break;
    }
  }
  const logPath = path.join(outputRoot, "reports", "tool_calls.jsonl");
  const entry = {
    timestamp: new Date().toISOString(),
    tool: "get_rtl_patterns",
    op_type,
    kernel_h: kernel_h ?? null,
    kernel_w: kernel_w ?? null,
    contract_id: contract_id ?? null,
    pattern_markdown_chars: pattern_markdown.length,
    reference_verilog_chars: reference_verilog ? reference_verilog.length : 0,
    lifecycle_docs: lifecyclePatternDocs
      .concat(lifecycleReferenceDocs)
      .map((doc) => ({
        path: path.relative(repoRoot, doc.path),
        contract_id: doc.contract_id ?? null,
        contract_key: doc.contract_key ?? null,
      })),
  };
  try {
    await mkdir(path.dirname(logPath), { recursive: true });
    await appendFile(logPath, `${JSON.stringify(entry)}\n`, "utf8");
  } catch (err) {
    // Observability is nice-to-have; never block the tool on it, but
    // surface a diagnostic so a broken log path doesn't go unnoticed.
    const msg = err instanceof Error ? err.message : String(err);
    process.stderr.write(
      `[get_rtl_patterns] side-log to '${logPath}' failed: ${msg}\n`,
    );
  }

  return { pattern_markdown, reference_verilog, license_notice };
}

type CorpusLookupInput = {
  module_id?: string;
  op_type?: string;
  contract_id?: string;
  spec_hash?: string;
  signature_hash?: string;
  exact_reference_key?: string | null;
  runtime_layer_signature?: Record<string, unknown>;
};

type CorpusMatchLevel =
  | "exact_signature"
  | "exact_reference_key"
  | "op_contract_kernel_stride_groups"
  | "op_contract_kernel"
  | "op"
  | "legacy_filter";

const CORPUS_MATCH_RANK: Record<CorpusMatchLevel, number> = {
  exact_signature: 0,
  exact_reference_key: 1,
  op_contract_kernel_stride_groups: 2,
  op_contract_kernel: 3,
  op: 4,
  legacy_filter: 5,
};

function recordValue(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function stringValues(value: unknown): string[] {
  if (typeof value === "string" && value.length > 0) return [value];
  if (Array.isArray(value)) {
    return value.filter((item): item is string => typeof item === "string" && item.length > 0);
  }
  return [];
}

function firstString(...values: unknown[]): string | undefined {
  for (const value of values) {
    const [found] = stringValues(value);
    if (found) return found;
  }
  return undefined;
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function pairValue(value: unknown): [number | null, number | null] | undefined {
  if (!Array.isArray(value)) return undefined;
  const a = numberValue(value[0]) ?? null;
  const b = numberValue(value[1]) ?? a;
  return a === null && b === null ? undefined : [a, b];
}

function samePair(
  a: [number | null, number | null] | undefined,
  b: [number | null, number | null] | undefined,
): boolean {
  return a !== undefined && b !== undefined && a[0] === b[0] && a[1] === b[1];
}

function kernelFromShape(shape: Record<string, unknown>): [number | null, number | null] | undefined {
  const weightShape = shape.weight_shape;
  if (!Array.isArray(weightShape)) return undefined;
  const kh = numberValue(weightShape[2]);
  const kw = numberValue(weightShape[3]);
  return kh === undefined || kw === undefined ? undefined : [kh, kw];
}

function corpusFields(record: Record<string, unknown>): {
  op_type?: string;
  contract_id?: string;
  kernel?: [number | null, number | null];
  stride?: [number | null, number | null];
  padding?: [number | null, number | null];
  groups?: number;
} {
  const applicability = recordValue(record.applicability);
  const signature = recordValue(record.runtime_layer_signature);
  const shape = recordValue(record.shape);
  return {
    op_type: firstString(applicability.op_type, signature.op_type, record.op_type),
    contract_id: firstString(applicability.contract_id, signature.contract_id, record.contract_id),
    kernel: pairValue(applicability.kernel) ?? pairValue(signature.kernel) ?? kernelFromShape(shape),
    stride: pairValue(applicability.stride) ?? pairValue(signature.stride) ?? pairValue(shape.stride),
    padding: pairValue(applicability.padding) ?? pairValue(signature.padding) ?? pairValue(shape.padding),
    groups:
      numberValue(applicability.groups) ??
      numberValue(signature.groups) ??
      numberValue(shape.groups),
  };
}

function entrySignatureHashes(entry: Record<string, unknown>): string[] {
  const applicability = recordValue(entry.applicability);
  return [
    ...stringValues(entry.signature_hash),
    ...stringValues(entry.signature_hashes),
    ...stringValues(applicability.signature_hash),
    ...stringValues(applicability.signature_hashes),
  ];
}

function entryExactReferenceKeys(entry: Record<string, unknown>): string[] {
  const applicability = recordValue(entry.applicability);
  return [
    ...stringValues(entry.exact_reference_key),
    ...stringValues(entry.exact_reference_keys),
    ...stringValues(applicability.exact_reference_key),
    ...stringValues(applicability.exact_reference_keys),
  ];
}

function valueMatchesRule(value: unknown, actual: string | null | undefined): boolean {
  if (actual === null || actual === undefined) return false;
  return stringValues(value).includes(actual);
}

function corpusContraindicationVetoes(
  entry: Record<string, unknown>,
  input: CorpusLookupInput,
): boolean {
  const rules = Array.isArray(entry.contraindications) ? entry.contraindications : [];
  const networkId = process.env.NN2RTL_NETWORK_ID ?? null;
  for (const rule of rules) {
    if (typeof rule === "string") {
      if (
        rule === input.signature_hash ||
        rule === input.exact_reference_key ||
        rule === input.op_type ||
        rule === input.contract_id ||
        rule === networkId ||
        rule === `op_type:${input.op_type}` ||
        rule === `contract_id:${input.contract_id}` ||
        rule === `network_id:${networkId}` ||
        rule === `signature_hash:${input.signature_hash}`
      ) {
        return true;
      }
      continue;
    }
    const record = recordValue(rule);
    if (
      valueMatchesRule(record.op_type, input.op_type) ||
      valueMatchesRule(record.contract_id, input.contract_id) ||
      valueMatchesRule(record.network_id, networkId) ||
      valueMatchesRule(record.signature_hash, input.signature_hash) ||
      valueMatchesRule(record.signature_hashes, input.signature_hash) ||
      valueMatchesRule(record.exact_reference_key, input.exact_reference_key) ||
      valueMatchesRule(record.exact_reference_keys, input.exact_reference_key)
    ) {
      return true;
    }
  }
  return false;
}

function corpusMatchLevel(
  entry: Record<string, unknown>,
  input: CorpusLookupInput,
): CorpusMatchLevel | null {
  if (corpusContraindicationVetoes(entry, input)) return null;
  if (input.module_id && entry.module_id !== input.module_id) return null;
  if (input.spec_hash && entry.spec_hash !== input.spec_hash) return null;

  if (input.signature_hash && entrySignatureHashes(entry).includes(input.signature_hash)) {
    return "exact_signature";
  }
  if (
    input.exact_reference_key &&
    entryExactReferenceKeys(entry).includes(input.exact_reference_key)
  ) {
    return "exact_reference_key";
  }

  const target = corpusFields({
    op_type: input.op_type,
    contract_id: input.contract_id,
    runtime_layer_signature: input.runtime_layer_signature,
  });
  const fields = corpusFields(entry);
  if (input.op_type && fields.op_type !== input.op_type) return null;
  if (!target.op_type || fields.op_type !== target.op_type) {
    return input.op_type || input.contract_id ? "legacy_filter" : null;
  }

  const contractMatches = fields.contract_id === target.contract_id;
  const kernelMatches = samePair(fields.kernel, target.kernel);
  const strideMatches = samePair(fields.stride, target.stride);
  const groupsMatches = fields.groups !== undefined && fields.groups === target.groups;
  if (contractMatches && kernelMatches && strideMatches && groupsMatches) {
    return "op_contract_kernel_stride_groups";
  }
  if (contractMatches && kernelMatches) {
    return "op_contract_kernel";
  }
  if (fields.op_type === target.op_type) {
    if (input.contract_id && fields.contract_id !== input.contract_id) return null;
    return "op";
  }
  return input.op_type || input.contract_id ? "legacy_filter" : null;
}

export async function get_failure_corpus(input: {
  module_id?: string;
  op_type?: string;
  contract_id?: string;
  spec_hash?: string;
  signature_hash?: string;
  exact_reference_key?: string | null;
  runtime_layer_signature?: Record<string, unknown>;
  max_entries?: number;
  include_verilog?: boolean;
}): Promise<GetFailureCorpusResult> {
  const outputRootRaw = process.env.NN2RTL_OUTPUT_DIR ?? "output";
  const hostOutputRoot = normalizePathForCurrentHost(outputRootRaw);
  const outputRoot = path.isAbsolute(hostOutputRoot) || isWindowsAbsolutePath(hostOutputRoot)
    ? hostOutputRoot
    : path.resolve(repoRoot, hostOutputRoot);
  const visibleRoots: Array<{ network_id: string | null; model_name: string | null; root: string }> = [];
  try {
    const registry = JSON.parse(await readFile(path.join(repoRoot, "networks.json"), "utf8")) as {
      networks?: Array<{ id?: string; modelName?: string; outputDir?: string }>;
    };
    for (const network of registry.networks ?? []) {
      if (!network.outputDir) continue;
      visibleRoots.push({
        network_id: network.id ?? null,
        model_name: network.modelName ?? null,
        root: path.resolve(repoRoot, network.outputDir, "failure_corpus", "visible"),
      });
    }
  } catch {
    // Fall back to the current env-selected output root.
  }
  visibleRoots.push({
    network_id: process.env.NN2RTL_NETWORK_ID ?? null,
    model_name: null,
    root: path.resolve(outputRoot, "failure_corpus", "visible"),
  });
  const dedupedRoots = [...new Map(visibleRoots.map((entry) => [path.resolve(entry.root), entry])).values()];
  const entries: Array<Record<string, unknown>> = [];
  for (const visible of dedupedRoots) {
    const indexPath = path.join(visible.root, "index.jsonl");
    let raw: string;
    try {
      raw = await readFile(indexPath, "utf8");
    } catch {
      continue;
    }
    for (const line of raw.split(/\r?\n/)) {
      if (!line.trim()) continue;
      try {
        const entry = JSON.parse(line) as Record<string, unknown>;
        entry.network_id ??= visible.network_id;
        entry.model_name ??= visible.model_name;
        const matchLevel = corpusMatchLevel(entry, input);
        if (matchLevel === null) continue;
        entry.match_level = matchLevel;
        entries.push(entry);
      } catch {
        // Ignore malformed historical corpus lines.
      }
    }
  }
  entries.sort((a, b) => {
    const sameModuleA = input.module_id && a.module_id === input.module_id ? 0 : 1;
    const sameModuleB = input.module_id && b.module_id === input.module_id ? 0 : 1;
    if (sameModuleA !== sameModuleB) return sameModuleA - sameModuleB;
    const rankA = CORPUS_MATCH_RANK[String(a.match_level ?? "legacy_filter") as CorpusMatchLevel] ?? 999;
    const rankB = CORPUS_MATCH_RANK[String(b.match_level ?? "legacy_filter") as CorpusMatchLevel] ?? 999;
    if (rankA !== rankB) return rankA - rankB;
    return String(b.created_at ?? "").localeCompare(String(a.created_at ?? ""));
  });
  const limited = entries.slice(0, input.max_entries ?? 5);
  if (input.include_verilog) {
    for (const entry of limited) {
      const rel = typeof entry.rtl_path === "string" ? entry.rtl_path : null;
      if (!rel) continue;
      try {
        entry.verilog_source = await readFile(path.resolve(repoRoot, rel), "utf8");
      } catch {
        entry.verilog_source = "";
      }
    }
  }
  return { visible_tier: "registered_network_outputs/failure_corpus/visible", entries: limited };
}

// ---------------------------------------------------------------------------
// compute_layer_reference: bit-exact ground-truth oracle for a single output
// pixel. Reads the same weights.hex / bias.hex / goldin files the testbench
// reads, runs the conv math in pure int64, applies the same sign-aware
// rounding the canonical RTL uses, and returns the expected INT8 outputs (and
// optional integer-domain intermediates).
//
// Access policy (enforced via agent .md docs, audit-tracked here via
// caller_role + tool_use_summary; not a hard runtime cap):
//   - assayer:  free use for deterministic failure enrichment.
//   - surgeon:  free use during repair / debugging.
//   - foundry:  read-only sanity checks only, cap 3 calls per attempt.
// ---------------------------------------------------------------------------

interface ComputeLayerReferenceInput {
  module_id: string;
  vector_idx: number;
  output_pixel_oy: number;
  output_pixel_ox: number;
  oc_start?: number;
  oc_end?: number;
  include_intermediates?: boolean;
  caller_role?: "foundry" | "surgeon" | "assayer";
}

interface ComputeLayerReferenceOutput {
  module_id: string;
  vector_idx: number;
  output_pixel_oy: number;
  output_pixel_ox: number;
  oc_range: [number, number];
  scale_constants: { mult: number; shift: number };
  output: number[];
  intermediates?: {
    acc: number[];
    biased: number[];
    scaled: number[];
    v_tmp: number[];
  };
  // Fingerprint for quick-equality checks across calls.
  output_fingerprint: string;
}

/** Mirror of sdk/orchestrate.ts:computeScaleApprox. Inlined to keep mcp/
 *  decoupled from sdk/. Search range shift 0..23, mult 1..32767 — the
 *  0-end is needed for deep-network layers whose scale_factor exceeds
 *  ~128 (where shift>=8 would overflow the INT16 mult cap). */
function mcpComputeScaleApprox(scaleFactor: number): { mult: number; shift: number } {
  if (!Number.isFinite(scaleFactor) || scaleFactor <= 0) {
    throw new Error(`compute_layer_reference: scale_factor must be > 0, got ${scaleFactor}.`);
  }
  let best = { mult: 1, shift: 0, err: Infinity };
  for (let shift = 0; shift <= 23; shift += 1) {
    const mult = Math.round(scaleFactor * Math.pow(2, shift));
    if (mult >= 1 && mult < 32768) {
      const err = Math.abs(mult / Math.pow(2, shift) - scaleFactor) / scaleFactor;
      if (err < best.err) {
        best = { mult, shift, err };
      }
    }
  }
  if (!Number.isFinite(best.err)) {
    throw new Error(
      `compute_layer_reference: scale_factor ${scaleFactor} outside representable mult/shift range.`,
    );
  }
  return { mult: best.mult, shift: best.shift };
}

function loadSignedHex(path: string, expectedCount: number, byteWidth: 1 | 4): number[] {
  const text = readFileSync(path, "utf8");
  const out: number[] = [];
  const limit = byteWidth === 1 ? 0xff : 0xffffffff;
  const signBit = byteWidth === 1 ? 0x80 : 0x80000000;
  const wrap = byteWidth === 1 ? 0x100 : 0x1_0000_0000;
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.split("//")[0].trim();
    if (!line) continue;
    const v = Number.parseInt(line, 16);
    if (Number.isNaN(v) || v < 0 || v > limit) {
      throw new Error(
        `compute_layer_reference: hex file '${path}' has invalid line '${rawLine.trim()}'.`,
      );
    }
    out.push(v >= signBit ? v - wrap : v);
  }
  if (out.length !== expectedCount) {
    throw new Error(
      `compute_layer_reference: hex file '${path}' has ${out.length} entries, expected ${expectedCount}.`,
    );
  }
  return out;
}

interface GoldenInputVector {
  bytes: Uint8Array;
}

function loadGoldinVectors(path: string): {
  num_vectors: number;
  samples_per_vector: number;
  bytes_per_sample: number;
  vectors: GoldenInputVector[];
} {
  const buf = readFileSyncBytes(path);
  if (buf.length < 20) {
    throw new Error(`compute_layer_reference: '${path}' is shorter than the 20-byte header.`);
  }
  const magic = String.fromCharCode(buf[0], buf[1], buf[2], buf[3]);
  if (magic !== "NN2V") {
    throw new Error(`compute_layer_reference: '${path}' has wrong magic '${magic}', expected 'NN2V'.`);
  }
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const version = view.getUint32(4, true);
  const numVectors = view.getUint32(8, true);
  const samplesPerVector = view.getUint32(12, true);
  const bytesPerSample = view.getUint32(16, true);
  if (version !== 2) {
    throw new Error(`compute_layer_reference: '${path}' version ${version} unsupported (expected 2).`);
  }
  const wordsPerSample = Math.ceil(bytesPerSample / 4);
  const perVecBytes = samplesPerVector * wordsPerSample * 4;
  const vectors: GoldenInputVector[] = [];
  for (let v = 0; v < numVectors; v += 1) {
    const start = 20 + v * perVecBytes;
    if (start + perVecBytes > buf.length) {
      throw new Error(`compute_layer_reference: '${path}' is truncated at vector ${v}.`);
    }
    vectors.push({ bytes: buf.subarray(start, start + perVecBytes) });
  }
  return { num_vectors: numVectors, samples_per_vector: samplesPerVector, bytes_per_sample: bytesPerSample, vectors };
}

function readFileSyncBytes(p: string): Uint8Array {
  // Local helper to keep this self-contained without pulling fs/promises sync apis.
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const fs = require("node:fs") as typeof import("node:fs");
  return fs.readFileSync(p);
}

function readFileSync(p: string, enc: "utf8"): string {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const fs = require("node:fs") as typeof import("node:fs");
  return fs.readFileSync(p, enc);
}

interface LayerIRForCompute {
  module_id: string;
  op_type: string;
  input_shape: number[];
  output_shape: number[];
  weights_path: string;
  bias_path: string | null;
  weight_shape: number[];
  scale_factor: number;
  zero_point: number;
  golden_inputs_path: string;
  stride?: number[];
  padding?: number[];
  groups?: number;
}

function findLayer(layerIrPath: string, moduleId: string): LayerIRForCompute {
  const text = readFileSync(layerIrPath, "utf8");
  const ir = JSON.parse(text) as { layers: LayerIRForCompute[] };
  const layer = ir.layers.find((l) => l.module_id === moduleId);
  if (!layer) {
    throw new Error(
      `compute_layer_reference: module_id '${moduleId}' not found in '${layerIrPath}'.`,
    );
  }
  return layer;
}

function clampInt8(v: number): number {
  if (v > 127) return 127;
  if (v < -128) return -128;
  return v;
}

function fingerprintInt8(values: number[]): string {
  // Deterministic short hash for cross-call equality checks. Not crypto.
  let h = 0;
  for (const v of values) {
    h = ((h * 31) + v + 256) | 0;
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}

export async function compute_layer_reference(
  input: ComputeLayerReferenceInput,
): Promise<ComputeLayerReferenceOutput> {
  const layerIrPath = path.join(resolveActiveOutputRoot(process.env), "layer_ir.json");
  const layer = findLayer(layerIrPath, input.module_id);
  if (layer.op_type !== "conv2d") {
    throw new Error(
      `compute_layer_reference: only op_type='conv2d' is supported, got '${layer.op_type}'.`,
    );
  }

  const [, IC, IH, IW] = layer.input_shape;
  const [, OC, OH, OW] = layer.output_shape;
  const [W_OC, W_IC, KH, KW] = layer.weight_shape;
  if (W_OC !== OC || W_IC !== IC) {
    throw new Error(
      `compute_layer_reference: weight_shape [${layer.weight_shape}] disagrees with OC=${OC}/IC=${IC}.`,
    );
  }
  const [SH, SW] = layer.stride ?? [1, 1];
  const [PH, PW] = layer.padding ?? [0, 0];
  const groups = layer.groups ?? 1;
  if (groups !== 1) {
    throw new Error(`compute_layer_reference: groups=${groups} not yet supported (groups=1 only).`);
  }

  const { vector_idx, output_pixel_oy, output_pixel_ox } = input;
  if (output_pixel_oy < 0 || output_pixel_oy >= OH || output_pixel_ox < 0 || output_pixel_ox >= OW) {
    throw new Error(
      `compute_layer_reference: output pixel (${output_pixel_oy},${output_pixel_ox}) out of bounds (OH=${OH}, OW=${OW}).`,
    );
  }
  const ocStart = Math.max(0, input.oc_start ?? 0);
  const ocEnd = Math.min(OC, input.oc_end ?? OC);
  if (ocEnd <= ocStart) {
    throw new Error(`compute_layer_reference: empty oc range [${ocStart}, ${ocEnd}).`);
  }

  const weightsPath = normalizePathForCurrentHost(layer.weights_path);
  const biasPath = layer.bias_path ? normalizePathForCurrentHost(layer.bias_path) : null;
  const goldinPath = normalizePathForCurrentHost(layer.golden_inputs_path);

  const weights = loadSignedHex(weightsPath, OC * IC * KH * KW, 1);
  const biases = biasPath ? loadSignedHex(biasPath, OC, 4) : new Array(OC).fill(0);
  const goldin = loadGoldinVectors(goldinPath);
  if (vector_idx < 0 || vector_idx >= goldin.num_vectors) {
    throw new Error(
      `compute_layer_reference: vector_idx=${vector_idx} out of range (num_vectors=${goldin.num_vectors}).`,
    );
  }
  const vecBytes = goldin.vectors[vector_idx].bytes;

  const { mult, shift } = mcpComputeScaleApprox(layer.scale_factor);
  const HALF = 1 << (shift - 1);
  const HALF_M1 = HALF - 1;

  // Input layout in the goldin: linear `pixel*IC + channel` over IH*IW pixels.
  function pixelChannel(iy: number, ix: number, ic: number): number {
    if (iy < 0 || iy >= IH || ix < 0 || ix >= IW) return 0; // zero-padding
    const idx = (iy * IW + ix) * IC + ic;
    const b = vecBytes[idx];
    return b >= 128 ? b - 256 : b;
  }

  const accs: number[] = [];
  const biasedVals: number[] = [];
  const scaledVals: number[] = [];
  const vTmpVals: number[] = [];
  const outputs: number[] = [];

  for (let oc = ocStart; oc < ocEnd; oc += 1) {
    let acc = 0;
    const wBase = oc * IC * KH * KW;
    for (let kh = 0; kh < KH; kh += 1) {
      const iy = output_pixel_oy * SH - PH + kh;
      for (let kw = 0; kw < KW; kw += 1) {
        const ix = output_pixel_ox * SW - PW + kw;
        const wRow = wBase + (kh * KW + kw) * IC;
        for (let ic = 0; ic < IC; ic += 1) {
          acc += weights[wRow + ic] * pixelChannel(iy, ix, ic);
        }
      }
    }
    const biased = acc + biases[oc];
    const scaled = biased * mult;
    const signBias = scaled < 0 ? HALF_M1 : HALF;
    // Math.floor((scaled + signBias) / 2^shift) — JS // for negatives uses
    // truncation; emulate floor via Math.floor on the float division which
    // is exact for values < 2^53.
    const vTmp = Math.floor((scaled + signBias) / Math.pow(2, shift));
    const out = clampInt8(vTmp);

    accs.push(acc);
    biasedVals.push(biased);
    scaledVals.push(scaled);
    vTmpVals.push(vTmp);
    outputs.push(out);
  }

  const result: ComputeLayerReferenceOutput = {
    module_id: layer.module_id,
    vector_idx,
    output_pixel_oy,
    output_pixel_ox,
    oc_range: [ocStart, ocEnd],
    scale_constants: { mult, shift },
    output: outputs,
    output_fingerprint: fingerprintInt8(outputs),
  };
  if (input.include_intermediates) {
    result.intermediates = { acc: accs, biased: biasedVals, scaled: scaledVals, v_tmp: vTmpVals };
  }
  return result;
}

export async function readSidecarIfPresent(
  filePath: string,
): Promise<VerificationSidecar | null> {
  let raw: string;
  const hostPath = normalizePathForCurrentHost(filePath);
  try {
    raw = await readFile(hostPath, "utf8");
  } catch (error: unknown) {
    if (
      typeof error === "object" &&
      error !== null &&
      "code" in error &&
      (error as { code?: string }).code === "ENOENT"
    ) {
      return null;
    }
    throw error;
  }

  const parsed: unknown = JSON.parse(raw);
  const validated = verificationSidecarSchema.safeParse(parsed);
  if (!validated.success) {
    throw new Error(
      `run_verilator: '${filePath}' is not a valid VerificationSidecar:\n${JSON.stringify(validated.error.issues, null, 2)}`,
    );
  }
  return validated.data;
}
