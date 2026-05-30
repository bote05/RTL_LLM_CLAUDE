#!/usr/bin/env python3
"""Engine sweep driver: run engine_one_layer_tb across all heavy dispatches.

For each dispatch index in output/rtl/nn2rtl_scheduler_schedule.json:
  1. Generate build_engine_one_layer_tb/dispatch_cfg.vh from the JSON.
  2. Compile (or reuse) the iverilog .vvp.
  3. Run the sim with +TIMEOUT_CYCLES.
  4. Parse engine cycles from the sim log.
  5. Run compare_engine_output.py and capture its RESULT_JSON line.
  6. Aggregate per-dispatch records into output/engine_sweep_results.json
     and a human-readable docs/agent_tasks/13_engine_sweep_REPORT.md.

Exit 0 iff every dispatch PASSes byte-exact. Otherwise exit 1 and the
report calls out the failures.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "build_engine_one_layer_tb"
SCHEDULE_PATH = REPO_ROOT / "output" / "rtl" / "nn2rtl_scheduler_schedule.json"
LAYER_IR_PATH = REPO_ROOT / "output" / "layer_ir.json"
TB_FILE = REPO_ROOT / "tb" / "engine_one_layer_tb.v"
RESULTS_JSON = REPO_ROOT / "output" / "engine_sweep_results.json"
REPORT_MD = REPO_ROOT / "docs" / "agent_tasks" / "13_engine_sweep_REPORT.md"


# Each bank is 4096 BRAM words (24576 / 6 banks).
BANK_WORDS = 4096


def _normalize_tool_path(p: str) -> str:
    """Convert /c/... style Git-Bash paths to C:\\... for Windows subprocess.

    Leaves paths unchanged on non-Windows or when they are already absolute
    Windows paths."""
    if os.name != "nt":
        return p
    # /c/Users/.../iverilog -> C:/Users/.../iverilog (works in Win32 APIs).
    m = re.match(r"^/([a-zA-Z])/(.*)$", p)
    if m:
        drive = m.group(1).upper()
        rest = m.group(2)
        candidates = [f"{drive}:/{rest}", f"{drive}:/{rest}.exe"]
        for c in candidates:
            if Path(c).exists():
                return c
        return f"{drive}:/{rest}"
    return p


def _to_repo_path(p: str) -> Path:
    """Translate the WSL-style paths in layer_ir.json to local Windows
    relative paths under the repo root. Return as repo-relative POSIX
    path string suitable for embedding in Verilog $fopen literals."""
    if p.startswith("/mnt/c/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/"):
        rel = p[len("/mnt/c/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/"):]
        return REPO_ROOT / rel
    return Path(p)


def _posix_rel_to_repo(path: Path) -> str:
    """Return a POSIX path relative to REPO_ROOT for use in Verilog
    $fopen / $readmemh string literals. iverilog/vvp runs with cwd =
    REPO_ROOT (the sweep script cds there)."""
    abs_p = path.resolve()
    try:
        rel = abs_p.relative_to(REPO_ROOT.resolve())
    except ValueError:
        # Outside repo: fall back to absolute POSIX path.
        return abs_p.as_posix()
    return rel.as_posix()


def load_dispatches() -> list[dict]:
    schedule = json.loads(SCHEDULE_PATH.read_text())
    layer_ir = json.loads(LAYER_IR_PATH.read_text())
    layers_by_id = {l["module_id"]: l for l in layer_ir["layers"]
                    if "module_id" in l}

    enriched = []
    for d in schedule["dispatches"]:
        mid = d["module_id"]
        if mid not in layers_by_id:
            raise SystemExit(f"layer_ir.json missing entry for {mid}")
        layer = layers_by_id[mid]
        enriched.append({
            "dispatch": d,
            "layer": layer,
            "module_id": mid,
            "goldin_path": _to_repo_path(layer["golden_inputs_path"]),
            "goldout_path": _to_repo_path(layer["golden_outputs_path"]),
        })
    return enriched


def write_cfg_vh(info: dict, observed_hex_path: Path,
                 cfg_dir: Path | None = None) -> Path:
    """Generate dispatch_cfg.vh for this dispatch.

    `cfg_dir` is the include directory the iverilog -I will point at; the
    file is always named `dispatch_cfg.vh` inside that directory. Default
    is the canonical BUILD_DIR; parallel workers pass their own per-worker
    dir to avoid races.
    """
    d = info["dispatch"]
    ic = d["channel_in"]
    oc = d["channel_out"]
    kh, kw = d["kernel"]
    sh, sw = d["stride"]
    ph, pw = d["padding"]
    ih, iw = d["input_hw"]
    oh, ow = d["output_hw"]
    # The engine expects activations to be channel-tiled. For each input
    # pixel it reads ceil(IC/256) consecutive BRAM words (channels 0..255 at
    # chunk 0, 256..511 at chunk 1, ...). See
    # output/rtl/engine/address_generator.v lines 192-197.
    ic_chunks = (ic + 255) // 256
    n_in_pixels = ih * iw
    n_in_bram_words = n_in_pixels * ic_chunks
    # When OC > 256, the engine writes ceil(OC/256) BRAM words per pixel
    # (one per oc_pass), laid out as [pixel0_pass0, pixel0_pass1, ...,
    # pixel1_pass0, ...]. See output/rtl/engine/address_generator.v line 204.
    oc_passes = (oc + 255) // 256
    n_out_pixels = oh * ow
    n_out_bram_words = n_out_pixels * oc_passes
    act_in_base = d["input_bank"] * BANK_WORDS
    act_out_base = d["output_bank"] * BANK_WORDS

    goldin_posix = _posix_rel_to_repo(info["goldin_path"])
    observed_posix = _posix_rel_to_repo(observed_hex_path)

    lines = [
        "// AUTO-GENERATED by scripts/engine_sweep_driver.py -- do not edit.",
        f"// dispatch_index={d['dispatch_index']} module_id={info['module_id']}",
        "",
        f"`define CFG_IC          12'd{ic}",
        f"`define CFG_OC          12'd{oc}",
        f"`define CFG_KH          3'd{kh}",
        f"`define CFG_KW          3'd{kw}",
        f"`define CFG_SH          3'd{sh}",
        f"`define CFG_SW          3'd{sw}",
        f"`define CFG_PH          3'd{ph}",
        f"`define CFG_PW          3'd{pw}",
        f"`define CFG_IH          8'd{ih}",
        f"`define CFG_IW          8'd{iw}",
        f"`define CFG_OH          8'd{oh}",
        f"`define CFG_OW          8'd{ow}",
        f"`define CFG_WEIGHT_BASE 20'd{d['weight_base_word']}",
        f"`define CFG_BIAS_BASE   16'd{d['bias_base_word']}",
        f"`define CFG_SCALE_MULT  32'd{d['scale_mult']}",
        f"`define CFG_SCALE_SHIFT 6'd{d['scale_shift']}",
        f"`define CFG_ZP          8'd{d['zero_point']}",
        f"`define CFG_ACT_IN_BASE  16'd{act_in_base}",
        f"`define CFG_ACT_OUT_BASE 16'd{act_out_base}",
        f"`define N_IN_PIXELS  {n_in_pixels}",
        f"`define N_IN_WORDS   {n_in_bram_words}",
        f"`define IC_CHUNKS    {ic_chunks}",
        f"`define N_OUT_WORDS  {n_out_bram_words}",
        f"`define N_OUT_PIXELS {n_out_pixels}",
        f"`define OC_PASSES    {oc_passes}",
        f"`define IC_BYTES     {ic}",
        f"`define OC_BYTES     {oc}",
        f'`define GOLDIN_PATH "{goldin_posix}"',
        f'`define OBSERVED_HEX_PATH "{observed_posix}"',
        "",
    ]
    out_dir = cfg_dir if cfg_dir is not None else BUILD_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "dispatch_cfg.vh"
    cfg_path.write_text("\n".join(lines))
    return cfg_path


def _oss_cad_env() -> dict:
    """Build an environment dict that lets iverilog/vvp find their helpers.

    OSS-CAD-Suite requires YOSYSHQ_ROOT pointing at the install dir AND the
    bin/lib dirs in PATH; otherwise iverilog silently exits 127 because it
    can't spawn ivlpp."""
    env = os.environ.copy()
    oss_root = env.get("OSS_CAD_ROOT")
    if not oss_root:
        # Default install location.
        # Under Git Bash the shell script uses "/c/Users/User/oss-cad-suite"
        # for YOSYSHQ_ROOT, but Python's subprocess on Windows talks
        # directly to Win32 API without Git Bash's path translation.
        # Use C:/... directly so iverilog.exe can resolve DLLs from
        # bin/lib (otherwise dies with STATUS_INVALID_IMAGE_PROTECT 0xC000007D).
        oss_root = "C:/Users/User/oss-cad-suite" if os.name == "nt" else "/c/Users/User/oss-cad-suite"
    # YOSYSHQ_ROOT must end with a separator and use forward slashes per
    # the upstream OSS-CAD-Suite scripts.
    if not env.get("YOSYSHQ_ROOT"):
        env["YOSYSHQ_ROOT"] = oss_root.rstrip("/") + "/"
    bin_dir = oss_root.rstrip("/") + "/bin"
    lib_dir = oss_root.rstrip("/") + "/lib"
    path_sep = ";" if os.name == "nt" else ":"
    env["PATH"] = bin_dir + path_sep + lib_dir + path_sep + env.get("PATH", "")
    return env


def compile_tb(iverilog: str, vvp_out: Path, build_dir: Path | None = None) -> None:
    """Recompile the TB. Must be re-run for every dispatch because the
    include file changes.

    `build_dir` is the include search path for the per-worker
    dispatch_cfg.vh. Defaults to the canonical BUILD_DIR.
    """
    if build_dir is None:
        build_dir = BUILD_DIR
    build_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        iverilog,
        "-g2012", "-gno-strict-declaration",
        "-DNN2RTL_ENGINE_SUBBLOCKS_PROVIDED",
        "-I", str(build_dir),
        "-o", str(vvp_out),
        str(TB_FILE),
        str(REPO_ROOT / "output" / "rtl" / "shared_engine_skeleton.v"),
        str(REPO_ROOT / "output" / "rtl" / "engine" / "address_generator.v"),
        str(REPO_ROOT / "output" / "rtl" / "engine" / "bram_to_stream_bridge.v"),
        str(REPO_ROOT / "output" / "rtl" / "engine" / "config_register_block.v"),
        str(REPO_ROOT / "output" / "rtl" / "engine" / "mac_array.v"),
        str(REPO_ROOT / "output" / "rtl" / "engine" / "requant_pipeline.v"),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                          env=_oss_cad_env())
    if proc.returncode != 0:
        sys.stderr.write("[sweep] iverilog compile FAILED\n")
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"iverilog returned {proc.returncode}")


def run_sim(vvp: str, vvp_file: Path, timeout_cycles: int,
            log_path: Path) -> tuple[bool, int | None, str]:
    """Run vvp with +TIMEOUT_CYCLES. Returns (ok, engine_cycles, full_log).
    `ok` is True iff "CYCLES_FROM_START" appears in the log AND "FATAL"
    does not."""
    cmd = [vvp, str(vvp_file), f"+TIMEOUT_CYCLES={timeout_cycles}"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as fh:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=fh,
                              stderr=subprocess.STDOUT, text=True,
                              env=_oss_cad_env())
    log = log_path.read_text(errors="replace")
    fatal = ("FATAL" in log) or (proc.returncode != 0)
    cycles = None
    m = re.search(r"CYCLES_FROM_START=(\d+)", log)
    if m:
        cycles = int(m.group(1))
    ok = (not fatal) and (cycles is not None)
    return ok, cycles, log


def run_comparator(python_exe: str, dispatch_idx: int,
                   observed_hex_path: Path,
                   json_out_path: Path) -> dict:
    cmd = [
        python_exe, str(REPO_ROOT / "scripts" / "compare_engine_output.py"),
        "--dispatch-idx", str(dispatch_idx),
        "--observed", str(observed_hex_path),
        "--json-out", str(json_out_path),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    # The script also writes the JSON to disk; load it.
    if json_out_path.exists():
        try:
            return json.loads(json_out_path.read_text())
        except Exception:
            pass
    # Fallback: parse RESULT_JSON line.
    m = re.search(r"^RESULT_JSON:\s*(.+)$", proc.stdout, re.MULTILINE)
    if m:
        return json.loads(m.group(1))
    return {
        "status": "ERROR",
        "error": (f"comparator returned {proc.returncode} with no RESULT_JSON; "
                  f"stdout={proc.stdout[-500:]}; stderr={proc.stderr[-500:]}")
    }


def write_report(results: list[dict], wall_clock_s: float,
                 total_cycles: int) -> None:
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    n_pass = sum(1 for r in results if r.get("comparator", {}).get("status") == "PASS")
    n_fail = len(results) - n_pass
    lines = [
        "# Engine Sweep Report",
        "",
        f"- Generator: `scripts/run_engine_sweep_all.sh` -> "
        f"`scripts/engine_sweep_driver.py`",
        f"- Total dispatches run: {len(results)}",
        f"- PASS: {n_pass}",
        f"- FAIL: {n_fail}",
        f"- Wall clock: {wall_clock_s:.1f}s ({wall_clock_s/60.0:.2f} min)",
        f"- Total engine cycles across all dispatches: {total_cycles}",
        "",
        "## Per-dispatch results",
        "",
        "| idx | module_id | IC | OC | KxK | S | PxP | IHxIW | OHxOW | cycles | status | mismatches | max_err |",
        "|----:|-----------|---:|---:|:---:|:-:|:---:|:-----:|:-----:|------:|:------:|----------:|--------:|",
    ]
    for r in results:
        d = r["dispatch"]
        comp = r.get("comparator", {})
        status = comp.get("status", "ERROR")
        mism = comp.get("n_mismatches", "-")
        maxe = comp.get("max_error", "-")
        cycles = r.get("engine_cycles", "-")
        lines.append(
            f"| {d['dispatch_index']} | {d['module_id']} | "
            f"{d['channel_in']} | {d['channel_out']} | "
            f"{d['kernel'][0]}x{d['kernel'][1]} | {d['stride'][0]} | "
            f"{d['padding'][0]}x{d['padding'][1]} | "
            f"{d['input_hw'][0]}x{d['input_hw'][1]} | "
            f"{d['output_hw'][0]}x{d['output_hw'][1]} | "
            f"{cycles} | {status} | {mism} | {maxe} |"
        )
    # Failure details
    fails = [r for r in results if r.get("comparator", {}).get("status") != "PASS"]
    if fails:
        lines.append("")
        lines.append("## Failures (first-mismatch details)")
        for r in fails:
            d = r["dispatch"]
            comp = r.get("comparator", {})
            lines.append("")
            lines.append(f"### dispatch {d['dispatch_index']} — {d['module_id']}")
            lines.append("")
            lines.append(f"- engine_cycles: {r.get('engine_cycles')}")
            lines.append(f"- status: {comp.get('status')}")
            if comp.get("error"):
                lines.append(f"- error: {comp['error']}")
            lines.append(f"- n_mismatches: {comp.get('n_mismatches')}")
            lines.append(f"- max_error: {comp.get('max_error')}")
            fms = comp.get("first_mismatches", [])
            if fms:
                lines.append("- first mismatches:")
                for m in fms:
                    loc = (f"pixel[{m.get('pixel_row','?')},"
                           f"{m.get('pixel_col','?')}] ch{m['channel']}")
                    lines.append(
                        f"  - byte[{m['byte']}] ({loc}): "
                        f"expected 0x{m['expected']:02x} ({m['expected_s']}), "
                        f"got 0x{m['got']:02x} ({m['got_s']})"
                    )
    lines.append("")
    REPORT_MD.write_text("\n".join(lines))


def run_one_dispatch(info: dict, args, build_dir: Path,
                     print_lock: threading.Lock) -> dict:
    """Compile + sim + compare one dispatch into a per-worker build_dir.

    Returns the result dict (same shape as the serial loop produced).
    Designed to be called from a worker thread under ThreadPoolExecutor.
    """
    d = info["dispatch"]
    di = d["dispatch_index"]
    mid = info["module_id"]
    with print_lock:
        print(f"=== [sweep] dispatch {di} ({mid}) build_dir={build_dir.name}")
    observed_hex_path = REPO_ROOT / "output" / "engine_sweep" / (
        f"observed_dispatch{di:02d}_{mid}.hex"
    )
    observed_hex_path.parent.mkdir(parents=True, exist_ok=True)
    json_out_path = REPO_ROOT / "output" / "engine_sweep" / (
        f"result_dispatch{di:02d}_{mid}.json"
    )
    sim_log_path = REPO_ROOT / "output" / "engine_sweep" / (
        f"sim_dispatch{di:02d}_{mid}.log"
    )
    vvp_out = build_dir / "engine_tb.vvp"

    # write directly to the per-worker dir — avoids racing on the
    # canonical BUILD_DIR/dispatch_cfg.vh across workers.
    build_dir.mkdir(parents=True, exist_ok=True)
    write_cfg_vh(info, observed_hex_path, cfg_dir=build_dir)
    try:
        compile_tb(args.iverilog, vvp_out, build_dir=build_dir)
    except SystemExit as e:
        return {
            "dispatch": d, "module_id": mid,
            "engine_cycles": None,
            "comparator": {"status": "ERROR",
                           "error": f"compile failed: {e}"},
        }

    t0 = time.time()
    ok, cycles, _log = run_sim(args.vvp, vvp_out,
                                args.timeout_cycles, sim_log_path)
    sim_elapsed = time.time() - t0
    with print_lock:
        print(f"[sweep][{di:02d} {mid}] sim {sim_elapsed:.1f}s ok={ok} cycles={cycles}")

    if not ok:
        tail = sim_log_path.read_text(errors="replace").splitlines()[-30:]
        return {
            "dispatch": d, "module_id": mid,
            "engine_cycles": cycles,
            "sim_elapsed_s": sim_elapsed,
            "comparator": {
                "status": "ERROR",
                "error": f"sim FAILED (cycles={cycles}); last log lines: "
                         + " | ".join(tail[-10:])
            },
        }

    comp = run_comparator(args.python, di, observed_hex_path, json_out_path)
    with print_lock:
        print(f"[sweep][{di:02d} {mid}] {comp.get('status')} "
              f"mismatches={comp.get('n_mismatches')} "
              f"max_error={comp.get('max_error')}")
    return {
        "dispatch": d,
        "module_id": mid,
        "engine_cycles": cycles,
        "sim_elapsed_s": sim_elapsed,
        "comparator": comp,
    }


def _write_cfg_vh_to(info: dict, observed_hex_path: Path, cfg_path: Path) -> Path:
    """Like write_cfg_vh but writes to an explicit path (per-worker)."""
    import io
    buf = io.StringIO()
    # Capture write_cfg_vh's output into our buf by temporarily overriding
    # BUILD_DIR. Simpler: re-implement the inner write using the existing
    # function and then move the file.
    write_cfg_vh(info, observed_hex_path)
    # The original write_cfg_vh wrote to BUILD_DIR / "dispatch_cfg.vh".
    canonical = BUILD_DIR / "dispatch_cfg.vh"
    if cfg_path != canonical:
        cfg_path.write_text(canonical.read_text())
    return cfg_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iverilog", default=os.environ.get(
        "IVERILOG", "/c/Users/User/oss-cad-suite/bin/iverilog"))
    parser.add_argument("--vvp", default=os.environ.get(
        "VVP", "/c/Users/User/oss-cad-suite/bin/vvp"))
    parser.add_argument("--python", default=os.environ.get("PYTHON", "py"))
    parser.add_argument("--timeout-cycles", type=int,
                        default=int(os.environ.get("TIMEOUT_CYCLES", "50000000")))
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated dispatch indices to run "
                             "(default: all)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default 1 = serial). "
                             "Each worker uses its own build_engine_one_layer_tb_workerN/ dir. "
                             "Threading-bound (subprocess waits), so no GIL pressure.")
    args = parser.parse_args()
    # Bash wrapper may pass /c/... Git-Bash paths; Windows subprocess needs
    # C:/... — normalize.
    args.iverilog = _normalize_tool_path(args.iverilog)
    args.vvp = _normalize_tool_path(args.vvp)

    dispatches = load_dispatches()
    if args.only:
        keep = set(int(x) for x in args.only.split(","))
        dispatches = [info for info in dispatches
                      if info["dispatch"]["dispatch_index"] in keep]

    workers = max(1, args.workers)
    print(f"[sweep] running {len(dispatches)} dispatches with {workers} worker(s)")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    total_cycles = 0
    t_start = time.time()
    print_lock = threading.Lock()

    if workers == 1:
        # Serial path (canonical, also used as a fallback if anyone passes --workers=1).
        for info in dispatches:
            r = run_one_dispatch(info, args, BUILD_DIR, print_lock)
            results.append(r)
            if r.get("engine_cycles"):
                total_cycles += r["engine_cycles"]
    else:
        # Parallel path: each worker has its own build dir to avoid
        # dispatch_cfg.vh / engine_tb.vvp collisions.
        worker_dirs = [REPO_ROOT / f"build_engine_one_layer_tb_worker{i}"
                       for i in range(workers)]
        for wd in worker_dirs:
            wd.mkdir(parents=True, exist_ok=True)
        # Distribute dispatches across workers round-robin via a queue.
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            fut_to_info = {}
            for idx, info in enumerate(dispatches):
                wd = worker_dirs[idx % workers]
                fut_to_info[pool.submit(run_one_dispatch, info, args, wd, print_lock)] = info
            for fut in futures.as_completed(fut_to_info):
                try:
                    r = fut.result()
                except Exception as e:
                    info = fut_to_info[fut]
                    r = {
                        "dispatch": info["dispatch"],
                        "module_id": info["module_id"],
                        "engine_cycles": None,
                        "comparator": {"status": "ERROR",
                                       "error": f"worker exception: {e}"},
                    }
                results.append(r)
                if r.get("engine_cycles"):
                    total_cycles += r["engine_cycles"]
        # Sort results by dispatch_index for stable reporting.
        results.sort(key=lambda r: r["dispatch"]["dispatch_index"])

    wall_clock_s = time.time() - t_start

    # Persist machine-readable + human-readable summaries.
    summary = {
        "n_dispatches": len(results),
        "n_pass": sum(1 for r in results
                      if r.get("comparator", {}).get("status") == "PASS"),
        "n_fail": sum(1 for r in results
                      if r.get("comparator", {}).get("status") != "PASS"),
        "wall_clock_s": wall_clock_s,
        "total_engine_cycles": total_cycles,
        "results": results,
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(summary, indent=2))
    write_report(results, wall_clock_s, total_cycles)

    print()
    print(f"=== [sweep] DONE: {summary['n_pass']}/{summary['n_dispatches']} PASS, "
          f"wall_clock={wall_clock_s:.1f}s, "
          f"total_engine_cycles={total_cycles} ===")
    print(f"Results JSON: {RESULTS_JSON}")
    print(f"Report MD   : {REPORT_MD}")

    return 0 if summary["n_fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
