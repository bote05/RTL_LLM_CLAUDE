#!/usr/bin/env python3
"""Skip-FIFO sizing tool for ResNet-50 residual adds.

Implements task 04 from docs/agent_tasks/04_skip_fifo_sizing_tool.md.

Phase A (analytical) walks the LayerIR, identifies the main-path and skip-path
for each residual add, sums `pipeline_latency_cycles` along each path, optionally
adds engine sequentialisation overhead, applies a 1.5x backpressure margin and
rounds up to the next power of two.

Phase B (Verilator) builds a small cycle-accurate harness around
`output/wrapper/skip_fifo_block_dut.v` that models the residual block as
two fixed-latency pipelines feeding the skip FIFO + add. For each entry it
runs a sim at the current `verified_depth`; if the model reports overflow
it doubles the depth and retries up to OVERFLOW_RETRY_LIMIT times. Use
`--skip-verilator` to suppress Phase B (Wave 1 default).

Output schema: docs/agent_tasks/04_skip_fifo_sizing_tool.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


BACKPRESSURE_MARGIN_FACTOR = 1.5
OVERFLOW_RETRY_LIMIT = 6  # max doublings before we give up on a block

# 04c: the wrapper now drives `spatial_throttle = engine_busy` AND every
# residual fork point has implicit BRAM-side backpressure (the spatial
# chain stalls when the skip FIFO is full). The FIFO depth therefore
# becomes a deployment choice — a small constant that fits the U250's
# BRAM budget and bounded by backpressure rather than by pipeline-fill
# latency.
#
# Per the §"Verification gate" in 04c, the sum of (verified_depth *
# 512 bytes) across all 16 FIFOs must fit in 12 MB. 16 × 1024 × 512 B
# = 8 MB hits the soft target (≤6 MB lower bound, ≤12 MB hard ceiling).
# This cap is enforced in Phase A; the bounded-FIFO Verilator model in
# Phase B then verifies no deadlock at the chosen depth.
FIFO_DEPTH_CAP_THROTTLED = 1024
FIFO_DEPTH_MIN           = 64
# An analytical-only (un-capped) bound is also recorded for audit, but
# the cap is what gets handed to the wrapper / synthesised hardware.
VERIFIED_DEPTH_CAP = FIFO_DEPTH_CAP_THROTTLED


def detect_repo_root(script_path: Path) -> Path:
    override = os.environ.get("NN2RTL_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return script_path.resolve().parent.parent


def next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def load_layer_ir(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    layers = data.get("layers")
    if not isinstance(layers, list):
        raise ValueError(f"{path}: missing 'layers' array")
    return layers


def load_engine_modules(
    heavy_list_path: Optional[Path],
    schedule_path: Optional[Path],
) -> set[str]:
    """Return the set of module IDs the engine will dispatch.

    Prefer task 06's heavy-list text file when it exists. Otherwise fall
    back to the dispatch list in task 03's `schedule.json` — that is the
    authoritative artefact once the engine and scheduler have landed.
    """
    if heavy_list_path is not None and heavy_list_path.exists():
        modules: set[str] = set()
        for raw in heavy_list_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            modules.add(line)
        return modules
    if schedule_path is not None and schedule_path.exists():
        with schedule_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        dispatches = data.get("dispatches") or data.get("engine_dispatches") or []
        return {
            entry["module_id"]
            for entry in dispatches
            if isinstance(entry, dict) and "module_id" in entry
        }
    return set()


def load_engine_worst_case(
    schedule_path: Optional[Path],
    engine_modules: set[str],
    layers: list[dict],
) -> int:
    """Engine worst-case occupancy = max layer latency the engine ever runs.

    Order of preference:
      1. explicit `engine_worst_case_occupancy_cycles` in the schedule JSON,
      2. max `pipeline_latency_cycles` among the engine_modules looked up
         in the LayerIR (this is the authoritative source once both
         schedule and LayerIR are present),
      3. any per-dispatch `occupancy_cycles` / `pipeline_latency_cycles`
         hint inside the schedule JSON (legacy field),
      4. 0 (Phase A pre-engine case).
    """
    if schedule_path is not None and schedule_path.exists():
        with schedule_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        explicit = data.get("engine_worst_case_occupancy_cycles")
        if isinstance(explicit, int):
            return explicit
    if engine_modules and layers:
        latencies = [
            int(layer.get("pipeline_latency_cycles", 0))
            for layer in layers
            if layer.get("module_id") in engine_modules
        ]
        if latencies:
            return max(latencies)
    if schedule_path is not None and schedule_path.exists():
        with schedule_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        candidates = []
        dispatches = data.get("dispatches") or data.get("engine_dispatches") or []
        for entry in dispatches:
            cycles = entry.get("occupancy_cycles") or entry.get(
                "pipeline_latency_cycles"
            )
            if cycles is not None:
                candidates.append(int(cycles))
        if candidates:
            return max(candidates)
    return 0


def split_residual_block(
    convs_between: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Return (main_path_conv_layers, skip_path_conv_layers).

    ResNet-50 residual blocks come in two shapes:
      - Projection (conv) block: 4 conv layers between adds — the first 3 form
        the bottleneck on the main path, the 4th is the 1x1 projection on the
        skip path.
      - Identity block: 3 conv layers between adds — all on the main path; the
        skip is wires only.
    """
    n = len(convs_between)
    if n == 3:
        return convs_between, []
    if n == 4:
        return convs_between[:3], [convs_between[3]]
    raise ValueError(
        f"unexpected residual-block conv count {n} between adds; "
        "ResNet-50 should always have 3 (identity) or 4 (projection)"
    )


def build_block_groups(layers: list[dict]) -> list[dict]:
    """Walk LayerIR and produce one record per residual add.

    The block boundary is the previous add — or, for the first add, the
    maxpool that terminates the network stem. Anything that is not a
    conv2d/relu/add (e.g. the stem maxpool, an early conv before the residual
    stack starts, the global pool / FC tail after the last add) is treated as
    a boundary marker: it resets the in-flight `pending` list and is not
    attributed to any residual block.
    """
    groups: list[dict] = []
    pending: list[dict] = []
    add_count = 0
    for layer in layers:
        op = layer.get("op_type")
        if op == "add":
            convs = [l for l in pending if l.get("op_type") == "conv2d"]
            main_convs, skip_convs = split_residual_block(convs)
            skip_ids = {id(l) for l in skip_convs}
            main_layers = [l for l in pending if id(l) not in skip_ids]
            skip_layers = list(skip_convs)
            groups.append(
                {
                    "add_layer": layer,
                    "add_index": add_count,
                    "main_layers": main_layers,
                    "skip_layers": skip_layers,
                }
            )
            add_count += 1
            pending = []
        elif op in ("conv2d", "relu"):
            pending.append(layer)
        else:
            pending = []
    return groups


def compute_entry(
    group: dict,
    engine_modules: set[str],
    engine_worst_case_cycles: int,
    margin_factor: float,
) -> dict:
    """Phase-A analytical sizing under the THROTTLED-producer assumption
    with BRAM-bounded FIFOs (task 04c).

    Three rules from the 04c rewrite of §6.5:

    1. Engine-dispatched layers in the main path contribute 0 to the
       analytical fill — the wrapper's `engine_busy → spatial_throttle`
       gate stalls the producer for the entire engine_occupancy window,
       so their cycles never push into the FIFO.

       This is stronger than the 04a "drop k × engine_worst_case from
       raw_difference" rule, because here we also drop the engine
       layer's own `pipeline_latency_cycles` from the spatial sum
       (since that latency is no longer realised — the engine takes
       over).

    2. After step 1 the raw delta becomes
              main_spatial = sum(layer.pipeline_latency_cycles
                                 for layer in main if layer ∉ engine)
              skip_spatial = sum(layer.pipeline_latency_cycles
                                 for layer in skip if layer ∉ engine)
              raw_delta    = max(0, main_spatial - skip_spatial)
       Clamp to 0 when skip arrives first — the add module's internal
       buffering handles that case.

    3. The wrapper now also implements backpressure: at each residual
       fork the producer stalls when the skip FIFO is full. The FIFO
       depth is therefore a DEPLOYMENT CHOICE bounded by U250's on-chip
       memory budget, not by `pipeline_latency_cycles`. Cap each FIFO at
       FIFO_DEPTH_CAP_THROTTLED and let backpressure regulate the rest.

    The `engine_dispatches_in_main_path` / `engine_worst_case_occupancy`
    fields are still recorded — Phase B's cycle-accurate model uses them
    to schedule the throttle pulses.
    """
    add_layer = group["add_layer"]
    main_layers = group["main_layers"]
    skip_layers = group["skip_layers"]

    def spatial_latency(layers: list[dict]) -> int:
        return sum(
            int(l.get("pipeline_latency_cycles", 0))
            for l in layers
            if l.get("module_id") not in engine_modules
        )

    # Full pipeline_latency_cycles summed over all main/skip layers — kept
    # for audit and used by Phase B's throttle scheduling.
    main_latency_total = sum(
        int(l.get("pipeline_latency_cycles", 0)) for l in main_layers
    )
    skip_latency_total = sum(
        int(l.get("pipeline_latency_cycles", 0)) for l in skip_layers
    )

    # Spatial-only latencies (engine-dispatched layers excluded).
    main_spatial = spatial_latency(main_layers)
    skip_spatial = spatial_latency(skip_layers)

    dispatched_in_main = [
        l for l in main_layers if l.get("module_id") in engine_modules
    ]
    k = len(dispatched_in_main)

    raw_difference = main_spatial - skip_spatial
    if raw_difference < 0:
        # Skip path is longer than the spatial portion of main — the add
        # waits on the main side instead. A small FIFO is still useful
        # to smooth handshake jitter.
        raw_difference = 0

    with_margin = math.ceil(raw_difference * margin_factor)
    raw_pow2 = next_power_of_two(max(with_margin, FIFO_DEPTH_MIN))
    analytical_depth_uncapped = raw_pow2
    analytical_depth = min(raw_pow2, FIFO_DEPTH_CAP_THROTTLED)

    return {
        "add_module_id": add_layer.get("module_id"),
        "main_path_modules": [l.get("module_id") for l in main_layers],
        "skip_path_modules": [l.get("module_id") for l in skip_layers],
        # 04b kept the unfiltered "all-layer" latency under
        # `main_path_latency_cycles`; keep that name for downstream
        # compatibility but mark the spatial-only number explicitly.
        "main_path_latency_cycles": main_latency_total,
        "skip_path_latency_cycles": skip_latency_total,
        "main_path_spatial_latency_cycles": main_spatial,
        "skip_path_spatial_latency_cycles": skip_spatial,
        "engine_dispatches_in_main_path": k,
        "engine_worst_case_occupancy_cycles": engine_worst_case_cycles if k else 0,
        "analytical_depth_uncapped": analytical_depth_uncapped,
        "analytical_depth": analytical_depth,
        "verified_depth": analytical_depth,
        "verilator_status": "not_yet_verified",
        "sizing_model": "throttled + backpressure-bounded (task 04c)",
    }


VERDICT_RE = re.compile(
    r"^VERDICT\s+block=(?P<block>\S+)\s+result=(?P<result>\S+)\s+"
    r"peak=(?P<peak>\d+)\s+cycles=(?P<cycles>\d+)\s+"
    r"outputs=(?P<outs>\d+)\s+expected=(?P<exp>\d+)"
)


def find_w64devkit_bin(env: dict[str, str]) -> Optional[str]:
    """Locate w64devkit's g++ on Windows; matches mcp/tools.ts's logic."""
    if sys.platform != "win32":
        return None
    candidates = [
        env.get("NN2RTL_WIN_CXX_TOOLCHAIN_BIN"),
        (str(Path(env["USERPROFILE"]) / "w64devkit" / "bin")
         if env.get("USERPROFILE") else None),
        "C:\\w64devkit\\bin",
    ]
    for cand in candidates:
        if cand and Path(cand, "g++.exe").exists():
            return cand
    return None


def find_oss_cad_suite(env: dict[str, str]) -> Optional[str]:
    if sys.platform != "win32":
        return None
    cand_roots = [
        env.get("OSS_CAD_SUITE_ROOT"),
        env.get("YOSYSHQ_ROOT"),
        str(Path(env["USERPROFILE"]) / "oss-cad-suite")
        if env.get("USERPROFILE") else None,
        "C:\\oss-cad-suite",
    ]
    for cand in cand_roots:
        if cand and Path(cand, "bin", "verilator_bin.exe").exists():
            return cand
    return None


def build_verilator_env(repo_env: dict[str, str]) -> dict[str, str]:
    env = dict(repo_env)
    oss = find_oss_cad_suite(env)
    w64 = find_w64devkit_bin(env)
    if oss is None:
        return env
    sep = ";" if sys.platform == "win32" else ":"
    lib = str(Path(oss, "lib"))
    bin_ = str(Path(oss, "bin"))
    libexec = str(Path(oss, "libexec"))
    path_parts = [p for p in (w64, lib, bin_, libexec) if p]
    existing = env.get("PATH", "")
    env["PATH"] = sep.join([*path_parts, existing])
    if sys.platform == "win32":
        env["Path"] = env["PATH"]
        env["YOSYSHQ_ROOT"] = oss
    return env


def build_verilator_binary(
    repo_root: Path,
    dut: Path,
    tb: Path,
    build_dir: Path,
) -> Path:
    """Compile the timing model under Verilator. Returns the binary path."""
    binary_name = "Vskip_fifo_block_dut"
    if sys.platform == "win32":
        binary_name += ".exe"
    bin_path = build_dir / binary_name
    if bin_path.exists():
        # Rebuild only if sources are newer than the binary.
        src_mtime = max(dut.stat().st_mtime, tb.stat().st_mtime)
        if src_mtime <= bin_path.stat().st_mtime:
            return bin_path
        # Stale build — wipe and rebuild from scratch.
        shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.parent.mkdir(parents=True, exist_ok=True)

    verilator_cmd = (
        "verilator_bin" if sys.platform == "win32" else "verilator"
    )
    # Pass paths relative to repo_root with forward slashes — make on
    # Windows mangles backslashes in cc1plus invocations.
    def rel(p: Path) -> str:
        try:
            return p.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            return p.as_posix()

    args = [
        verilator_cmd,
        "--cc",
        "--exe",
        "--build",
        "--Mdir",
        rel(build_dir),
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
        "skip_fifo_block_dut",
        "-CFLAGS",
        "-std=c++17 -O2",
        rel(dut),
        rel(tb),
    ]
    env = build_verilator_env(dict(os.environ))
    proc = subprocess.run(
        args, capture_output=True, text=True, env=env, cwd=str(repo_root)
    )
    if proc.returncode != 0 or not bin_path.exists():
        raise RuntimeError(
            "Verilator build failed for skip_fifo_block_dut:\n"
            f"args: {' '.join(args)}\n"
            f"stdout: {proc.stdout[-2000:]}\n"
            f"stderr: {proc.stderr[-2000:]}"
        )
    return bin_path


def run_one_block(
    binary: Path,
    block_id: str,
    main_latency: int,
    skip_latency: int,
    depth: int,
    num_inputs: int,
    cycle_budget: int,
    goldin_path: Optional[Path],
    throttle_events: int = 0,
    throttle_duration: int = 0,
    throttle_period: int = 0,
) -> dict:
    """Invoke the Verilator binary once and parse the VERDICT line."""
    args = [
        str(binary),
        f"+block={block_id}",
        f"+main={main_latency}",
        f"+skip={skip_latency}",
        f"+depth={depth}",
        f"+nin={num_inputs}",
        f"+budget={cycle_budget}",
        f"+throttle_events={throttle_events}",
        f"+throttle_duration={throttle_duration}",
        f"+throttle_period={throttle_period}",
    ]
    if goldin_path is not None and goldin_path.exists():
        args.append(f"+goldin={goldin_path}")
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Verilator sim returned {proc.returncode} for {block_id}:\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    verdict_line = None
    for line in proc.stdout.splitlines():
        if line.startswith("VERDICT"):
            verdict_line = line
            break
    if verdict_line is None:
        raise RuntimeError(
            f"No VERDICT line in Verilator stdout for {block_id}:\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    m = VERDICT_RE.match(verdict_line)
    if not m:
        raise RuntimeError(
            f"Unparseable VERDICT line for {block_id}: {verdict_line!r}"
        )
    return {
        "result": m.group("result"),
        "peak": int(m.group("peak")),
        "cycles": int(m.group("cycles")),
        "outputs": int(m.group("outs")),
        "expected": int(m.group("exp")),
    }


def goldin_for_entry(repo_root: Path, entry: dict, layers: list[dict]) -> Optional[Path]:
    """Pick the goldin file for the residual block's first main-path layer.

    Used only for audit logging per the §6.5 "representative input stream"
    guidance — the bytes are not consumed by the timing model.
    """
    main_modules = entry.get("main_path_modules") or []
    if not main_modules:
        return None
    first = main_modules[0]
    for layer in layers:
        if layer.get("module_id") == first:
            path = layer.get("golden_inputs_path")
            if path:
                # Translate WSL-style mounts (/mnt/c/...) to Windows paths
                # when running natively on Windows.
                if sys.platform == "win32" and path.startswith("/mnt/c/"):
                    return Path("c:/" + path[len("/mnt/c/"):])
                return Path(path)
            return None
    return None


def run_verilator_phase(
    entries: list[dict],
    repo_root: Path,
    layers: list[dict],
    engine_modules: set[str],
    engine_worst_case_cycles: int,
) -> None:
    """Run cycle-accurate verification on each residual block.

    For each entry:
      1. Recompute `engine_dispatches_in_main_path` and
         `engine_worst_case_occupancy_cycles` from the current schedule +
         LayerIR (Phase A may have been run pre-engine with k=0). The
         analytical_depth field stays at the Phase A value — that is the
         historical snapshot.
      2. Compute `effective_main_latency` = main_path_latency_cycles +
         k * engine_worst_case_cycles.
      3. Run the Verilator harness at the current `verified_depth`.
      4. If overflow: double the depth and rerun, up to
         OVERFLOW_RETRY_LIMIT iterations.
      5. If deadlock or budget exhaustion: flag the entry; do NOT silently
         grow depth — that masks real bugs (per task spec §"Methodology
         per residual block" step 5).
      6. Clean: record verified_depth + verilator_status.
    """
    dut = repo_root / "output" / "wrapper" / "skip_fifo_block_dut.v"
    tb = repo_root / "output" / "wrapper" / "skip_fifo_block_tb.cpp"
    build_dir = repo_root / "build_skip_fifo_block"
    binary = build_verilator_binary(repo_root, dut, tb, build_dir)
    print(f"[phase B] using verilator binary at {binary}")

    for entry in entries:
        block_id = entry["add_module_id"]
        # Refresh engine fields against the up-to-date heavy list.
        main_modules = entry.get("main_path_modules", [])
        k = sum(1 for m in main_modules if m in engine_modules)
        entry["engine_dispatches_in_main_path"] = k
        entry["engine_worst_case_occupancy_cycles"] = (
            engine_worst_case_cycles if k else 0
        )

        skip_latency = int(entry.get("skip_path_spatial_latency_cycles",
                                      entry["skip_path_latency_cycles"]))
        spatial_main_latency = int(entry.get("main_path_spatial_latency_cycles",
                                              entry["main_path_latency_cycles"]))
        # 04c: simulate the SPATIAL-only main path; engine layers are
        # modelled as throttle pulses. This matches the wrapper's
        # `engine_busy → spatial_throttle` gating that freezes the
        # producer while a heavy layer runs.
        sim_main_latency = spatial_main_latency

        # With the BRAM-bounded FIFO + producer backpressure model, the
        # FIFO never grows beyond `verified_depth`. Drive enough samples
        # for the FIFO to fill, drain, refill, drain through a couple of
        # cycles — that exposes any sequencing bug in either the throttle
        # or the backpressure path. 4 × depth + a few hundred is a tight
        # but representative test.
        depth0 = int(entry.get("verified_depth",
                              entry.get("analytical_depth", FIFO_DEPTH_MIN)))
        num_inputs = max(depth0 * 4, 1024)

        # Throttle schedule: k pulses of `engine_worst_case_cycles` each.
        # Space them across the producer's steady-state window so the
        # throttle hits an already-filled FIFO.
        if k > 0 and engine_worst_case_cycles > 0:
            throttle_events = k
            throttle_duration = engine_worst_case_cycles
            spacing = max(num_inputs // max(k, 1), throttle_duration + 64)
            throttle_period = spacing
        else:
            throttle_events = 0
            throttle_duration = 0
            throttle_period = 0

        throttle_total = throttle_events * throttle_duration
        # Cycle budget = main warmup + steady-state cycles + throttle
        # overhead + slack. The backpressure-stall window inside that
        # warmup is already part of sim_main_latency.
        cycle_budget = sim_main_latency + num_inputs + throttle_total + 4096

        goldin = goldin_for_entry(repo_root, entry, layers)
        attempts: list[dict] = []
        depth = int(entry.get("verified_depth", entry.get("analytical_depth", 1)))
        terminal_status: Optional[str] = None
        for attempt in range(OVERFLOW_RETRY_LIMIT):
            result = run_one_block(
                binary,
                block_id=block_id,
                main_latency=sim_main_latency,
                skip_latency=skip_latency,
                depth=depth,
                num_inputs=num_inputs,
                cycle_budget=cycle_budget,
                goldin_path=goldin,
                throttle_events=throttle_events,
                throttle_duration=throttle_duration,
                throttle_period=throttle_period,
            )
            attempts.append({"depth": depth, **result})
            status = result["result"]
            if status == "no_deadlock_no_overflow":
                terminal_status = status
                break
            if status.startswith("deadlock_at_cycle_"):
                terminal_status = status
                break
            if status == "cycle_budget_exhausted":
                terminal_status = status
                break
            if status == "overflow":
                if depth >= VERIFIED_DEPTH_CAP:
                    terminal_status = "overflow_at_cap"
                    break
                depth = depth * 2
                continue
            terminal_status = f"unrecognised_{status}"
            break
        else:
            terminal_status = "overflow_retry_exhausted"

        entry["verified_depth"] = depth
        entry["verilator_status"] = terminal_status
        last = attempts[-1]
        entry["verilator_peak_occupancy"] = last["peak"]
        entry["verilator_cycles_run"] = last["cycles"]
        entry["verilator_attempts"] = attempts
        entry["sim_main_latency_cycles"] = sim_main_latency
        entry["throttle_events"] = throttle_events
        entry["throttle_duration_cycles"] = throttle_duration
        entry["throttle_period_cycles"] = throttle_period
        print(
            f"  {block_id:14s} spatial_main={sim_main_latency:>9d} "
            f"spatial_skip={skip_latency:>8d} k={k} "
            f"depth={depth} nin={num_inputs} peak={last['peak']} "
            f"status={terminal_status}"
        )


def main(argv: Optional[list[str]] = None) -> int:
    repo_root = detect_repo_root(Path(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default="resnet-50")
    parser.add_argument(
        "--layer-ir",
        default=str(repo_root / "output" / "layer_ir.json"),
    )
    parser.add_argument(
        "--schedule",
        default=str(repo_root / "output" / "rtl" / "nn2rtl_scheduler_schedule.json"),
    )
    parser.add_argument(
        "--engine-modules",
        default=str(
            repo_root
            / "docs"
            / "agent_tasks"
            / "06_phase1_compression_candidates_HEAVY.txt"
        ),
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=BACKPRESSURE_MARGIN_FACTOR,
    )
    parser.add_argument(
        "--out",
        default=str(repo_root / "output" / "wrapper" / "skip_fifo_sizes.json"),
    )
    parser.add_argument(
        "--skip-verilator",
        action="store_true",
        help="Wave 1: write analytical depths only; do not invoke Verilator",
    )
    args = parser.parse_args(argv)

    if args.network != "resnet-50":
        print(
            f"error: --network={args.network!r} is not supported; "
            "this tool is ResNet-50-specific",
            file=sys.stderr,
        )
        return 2

    layer_ir_path = Path(args.layer_ir)
    if not layer_ir_path.exists():
        print(f"error: LayerIR not found at {layer_ir_path}", file=sys.stderr)
        return 2

    layers = load_layer_ir(layer_ir_path)

    engine_modules_path = Path(args.engine_modules) if args.engine_modules else None
    schedule_path = Path(args.schedule) if args.schedule else None
    engine_modules = load_engine_modules(engine_modules_path, schedule_path)
    engine_worst_case_cycles = load_engine_worst_case(
        schedule_path, engine_modules, layers
    )

    groups = build_block_groups(layers)
    if len(groups) != 16:
        print(
            f"error: expected 16 residual adds in ResNet-50, got {len(groups)}",
            file=sys.stderr,
        )
        return 2

    entries = [
        compute_entry(
            group=g,
            engine_modules=engine_modules,
            engine_worst_case_cycles=engine_worst_case_cycles,
            margin_factor=args.margin,
        )
        for g in groups
    ]

    for entry in entries:
        if entry["main_path_latency_cycles"] < entry["skip_path_latency_cycles"]:
            print(
                f"error: {entry['add_module_id']} has main "
                f"({entry['main_path_latency_cycles']}) < skip "
                f"({entry['skip_path_latency_cycles']}) — refusing to write",
                file=sys.stderr,
            )
            return 2
        depth = entry["analytical_depth"]
        if depth & (depth - 1) != 0 or depth < 1:
            print(
                f"error: {entry['add_module_id']} analytical_depth={depth} "
                "is not a power of two",
                file=sys.stderr,
            )
            return 2

    if args.skip_verilator:
        method = "analytical"
    else:
        run_verilator_phase(
            entries,
            repo_root=repo_root,
            layers=layers,
            engine_modules=engine_modules,
            engine_worst_case_cycles=engine_worst_case_cycles,
        )
        method = "analytical + verilator-verified"

    output = {
        "method": method,
        "backpressure_margin_factor": args.margin,
        "fifos": entries,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, sort_keys=False)
        fh.write("\n")

    print(
        f"wrote {len(entries)} entries to {out_path} "
        f"(method={method!r}, engine_modules={len(engine_modules)}, "
        f"engine_worst_case_cycles={engine_worst_case_cycles})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
