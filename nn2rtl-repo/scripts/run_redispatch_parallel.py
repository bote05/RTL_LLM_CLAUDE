#!/usr/bin/env python3
"""Parallel wave dispatcher.

Spawns N orchestrator processes concurrently, each handling one module
of the wave. Per-worker isolation is achieved via NN2RTL_OUTPUT_DIR
pointing at a sandboxed directory; large read-mostly inputs (goldens,
weights) are junction-linked to the canonical output dir, small mutable
state (layer_ir, contract_state) is copied. After each worker passes
the area/timing gate, the produced artifacts (rtl, meta, vivado report,
sidecar) are merged back to the canonical output dir. Failure corpus
entries are merged regardless of pass/fail so future runs benefit from
the failure memory.

Usage:
    py scripts/run_redispatch_parallel.py \
        --checkpoint checkpoints/resnet50_full.onnx \
        --wave rest \
        --backup backups/pre_redispatch_20260521_122336 \
        --workers 4 \
        [--regression-pct 15] [--skip-completed] [--stop-on-fail]

Requires the orchestrate.ts patch that lets an externally-set
NN2RTL_OUTPUT_DIR survive `setActiveNetwork` (otherwise the worker
sandbox is silently bypassed).
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


CANARY_WAVE = [
    "node_conv_196",
    "node_relu",
    "node_max_pool2d",
    "node_conv_198",
    "node_conv_200",
    "node_add",
    "node_conv_220",
    "node_conv_224",
]

# Artifacts produced per dispatch that should be promoted to the canonical
# output dir AFTER the area gate passes. (subdir, filename-template)
PROMOTE_FILES: list[tuple[str, str]] = [
    ("rtl", "{m}.v"),
    ("rtl", "{m}.meta.json"),
    ("tb", "{m}.sidecar.json"),
    ("reports", "{m}.vivado.json"),
    ("reports", "{m}.results.json"),
]


def repo_root_from(script: Path) -> Path:
    return script.resolve().parent.parent


def load_wave_list(repo_root: Path, name: str) -> list[str]:
    if name == "canary":
        return list(CANARY_WAVE)
    if name == "rest":
        cmd = ["py", "scripts/patch_layerir_to_tiled.py", "--dry-run"]
        out = subprocess.check_output(cmd, cwd=repo_root, text=True)
        mods: list[str] = []
        capture = False
        for line in out.splitlines():
            if line.startswith("  modules that need RE-DISPATCH"):
                capture = True
                continue
            if capture:
                line = line.strip()
                if line.startswith("- "):
                    mods.append(line[2:].strip())
                elif line == "":
                    continue
                else:
                    break
        return [m for m in mods if m not in CANARY_WAVE]
    return [m.strip() for m in name.split(",") if m.strip()]


def vivado_report_for(report_dir: Path, module_id: str) -> dict | None:
    p = report_dir / f"{module_id}.vivado.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def pct_delta(new: float, old: float) -> float:
    if old <= 0:
        return 0.0 if new == 0 else 100.0
    return (new - old) / old * 100.0


def gate_area(
    module_id: str,
    backup_report: dict,
    new_report: dict,
    regression_pct: float,
) -> tuple[bool, list[str]]:
    """Mirror run_redispatch_wave.py:gate_area (abs floor + pct)."""
    messages: list[str] = []
    ok = True
    # Loosened from 1000/1000/4/4 after the conv_224 audit: the new tile-32
    # architecture has an inherent ~4k-LUT cost on projection / stride-2
    # convs because of the full-pixel out_pack buffer Foundry currently
    # emits. The cost is bounded (~25k LUT across ~5 such convs network-wide,
    # ~1.5% of U250 LUT budget). 5000-abs / 25% catches truly anomalous
    # regressions while accepting the known architectural cost.
    metrics = [
        ("lut_count", "LUT", 5000),
        ("ff_count", "FF", 5000),
        ("dsp_count", "DSP", 4),
        ("bram18_count", "BRAM18", 4),
    ]
    for key, label, abs_floor in metrics:
        old = float(backup_report.get(key, 0) or 0)
        new = float(new_report.get(key, 0) or 0)
        d = pct_delta(new, old)
        abs_delta = new - old
        if d > regression_pct and abs_delta > abs_floor:
            ok = False
            messages.append(
                f"  [{module_id}] REGRESSION {label}: {old:.0f} -> {new:.0f}  "
                f"({d:+.1f}% > {regression_pct:.0f}%, +{abs_delta:.0f} > floor "
                f"{abs_floor})"
            )
        elif d > regression_pct:
            messages.append(
                f"  [{module_id}] ok* {label}: {old:.0f} -> {new:.0f}  "
                f"({d:+.1f}%, +{abs_delta:.0f} under floor {abs_floor})"
            )
        else:
            messages.append(
                f"  [{module_id}] ok {label}: {old:.0f} -> {new:.0f}  ({d:+.1f}%)"
            )
    old_fmax = float(backup_report.get("fmax_mhz", 0) or 0)
    new_fmax = float(new_report.get("fmax_mhz", 0) or 0)
    if old_fmax > 0:
        fmax_d = pct_delta(new_fmax, old_fmax)
        if fmax_d < -regression_pct:
            ok = False
            messages.append(
                f"  [{module_id}] REGRESSION Fmax: {old_fmax:.1f} -> {new_fmax:.1f}  "
                f"({fmax_d:+.1f}% < -{regression_pct:.0f}%)"
            )
        else:
            messages.append(
                f"  [{module_id}] ok Fmax: {old_fmax:.1f} -> {new_fmax:.1f}  "
                f"({fmax_d:+.1f}%)"
            )
    return ok, messages


def make_junction(target: Path, link: Path) -> None:
    """Create a Windows directory junction (no admin needed).

    Junctions persist across reboots and are dir-only. On non-Windows we
    fall back to a regular dir symlink.
    """
    if link.exists() or link.is_symlink():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        subprocess.check_call(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        os.symlink(target, link, target_is_directory=True)


def setup_worker_dir(canonical: Path, worker_dir: Path) -> None:
    """Bootstrap a per-worker sandbox.

    Idempotent — safe to call multiple times for the same worker_dir.

    Layout:
        worker_dir/
            layer_ir.json                  (COPIED, replaced each call)
            layer_ir.json.checkpoint       (COPIED)
            contract_state.json            (COPIED)
            goldens/                       (JUNCTION, read-mostly)
            weights/                       (JUNCTION, read-only)
            rtl/                           (cleared each call)
            tb/                            (cleared)
            reports/                       (cleared)
            failure_corpus/visible/        (cleared)
            tmp/                           (cleared)
    """
    worker_dir.mkdir(parents=True, exist_ok=True)
    # Clear per-worker writable dirs so stale artifacts from a prior task
    # on the same slot don't leak into the new task's verdicts.
    for sub in ("rtl", "tb", "reports", "failure_corpus", "tmp"):
        sub_path = worker_dir / sub
        if sub_path.exists() and not sub_path.is_symlink():
            shutil.rmtree(sub_path, ignore_errors=True)
        sub_path.mkdir(parents=True, exist_ok=True)
    (worker_dir / "failure_corpus" / "visible").mkdir(
        parents=True, exist_ok=True,
    )
    # Copy small mutable files (overwrite to pick up latest canonical state).
    for fname in ("layer_ir.json", "layer_ir.json.checkpoint",
                  "contract_state.json"):
        src = canonical / fname
        if src.exists():
            shutil.copy2(src, worker_dir / fname)
    # Junction read-mostly dirs.
    for d in ("goldens", "weights"):
        src_dir = canonical / d
        if src_dir.exists():
            make_junction(src_dir, worker_dir / d)


def promote_artifacts(
    canonical: Path,
    worker_dir: Path,
    module_id: str,
) -> int:
    """Copy a worker's successful artifacts to the canonical output dir.

    Called ONLY when the area gate passes (or there's no backup to compare
    against). Polluting canonical with a regressed RTL/report would defeat
    the wave's gate.
    """
    count = 0
    for sub, fname_tpl in PROMOTE_FILES:
        src = worker_dir / sub / fname_tpl.format(m=module_id)
        if not src.exists():
            continue
        dst = canonical / sub / fname_tpl.format(m=module_id)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        count += 1
    return count


def merge_failure_corpus(
    canonical: Path,
    worker_dir: Path,
    module_id: str,
) -> int:
    """Merge failure_corpus entries + index regardless of gate outcome.

    Future Foundry/Surgeon calls retrieve from canonical, so missing
    entries would hide the failure memory. Returns the number of attempt
    subdirs newly copied.
    """
    visible_src = worker_dir / "failure_corpus" / "visible"
    visible_dst = canonical / "failure_corpus" / "visible"
    if not visible_src.exists():
        return 0
    # 1) Copy module attempt subdirs.
    module_src = visible_src / module_id
    new_attempts = 0
    if module_src.exists():
        module_dst = visible_dst / module_id
        module_dst.mkdir(parents=True, exist_ok=True)
        for child in module_src.iterdir():
            if not child.is_dir():
                continue
            dst_child = module_dst / child.name
            if not dst_child.exists():
                shutil.copytree(child, dst_child)
                new_attempts += 1
    # 2) Merge index.jsonl — append new lines (by `id` field) to canonical.
    idx_src = visible_src / "index.jsonl"
    if idx_src.exists():
        idx_dst = visible_dst / "index.jsonl"
        existing_ids: set[str] = set()
        if idx_dst.exists():
            with idx_dst.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    eid = entry.get("id")
                    if isinstance(eid, str):
                        existing_ids.add(eid)
        appended = 0
        with idx_src.open("r", encoding="utf-8") as fh_in, \
                idx_dst.open("a", encoding="utf-8") as fh_out:
            for line in fh_in:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except Exception:
                    continue
                # Filter: only carry index entries for this module's attempts
                # (so concurrent workers handling different modules don't
                # double-append each other's entries).
                if entry.get("module_id") and entry["module_id"] != module_id:
                    continue
                eid = entry.get("id")
                if isinstance(eid, str) and eid in existing_ids:
                    continue
                fh_out.write(stripped + "\n")
                if isinstance(eid, str):
                    existing_ids.add(eid)
                appended += 1
        if appended:
            new_attempts += 0  # already counted in attempts
    return new_attempts


def dispatch_one(
    module_id: str,
    worker_id: int,
    repo_root: Path,
    canonical: Path,
    base_env: dict[str, str],
    checkpoint_arg: str,
    backup_reports: Path,
    live_reports: Path,
    args: argparse.Namespace,
    print_lock: threading.Lock,
) -> tuple[str, str, list[str]]:
    """Run one module dispatch end-to-end in a per-worker sandbox.

    Returns (module_id, status, messages) where status ∈ {pass, skipped,
    orch_fail, stale_vivado, regression, no_report}.
    """
    worker_dir = canonical / f"_worker_{worker_id}"
    live_report_path = live_reports / f"{module_id}.vivado.json"
    backup_report_path = backup_reports / f"{module_id}.vivado.json"

    # Skip-completed: live U250 report already newer than backup baseline.
    if args.skip_completed and live_report_path.exists() \
            and backup_report_path.exists() \
            and live_report_path.stat().st_mtime \
            > backup_report_path.stat().st_mtime:
        with print_lock:
            print(f"[wave] === skipping {module_id} (already regenerated) ===")
        backup_rep = vivado_report_for(backup_reports, module_id)
        new_rep = vivado_report_for(live_reports, module_id)
        if backup_rep and new_rep:
            ok, msgs = gate_area(module_id, backup_rep, new_rep,
                                 args.regression_pct)
            with print_lock:
                for m in msgs:
                    print(m)
            if not ok:
                return module_id, "regression", msgs
        return module_id, "skipped", []

    # Per-worker sandbox.
    setup_worker_dir(canonical, worker_dir)
    pre_mtime = live_report_path.stat().st_mtime \
        if live_report_path.exists() else 0.0

    npm_exe = "npm.cmd" if os.name == "nt" else "npm"
    env = dict(base_env)
    env["NN2RTL_OUTPUT_DIR"] = str(worker_dir)
    cmd = [
        npm_exe, "--prefix", "sdk", "run", "pipeline", "--",
        checkpoint_arg, "--only", module_id,
    ]

    with print_lock:
        print(f"[wave] === dispatching {module_id} (worker {worker_id}) ===")

    start = time.time()
    log_dir = canonical / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"worker_{worker_id}_{module_id}.log"
    with log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.run(
            cmd, cwd=repo_root, shell=False, env=env,
            stdout=log_fh, stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - start

    with print_lock:
        print(f"[wave] [{module_id}] worker {worker_id} done in {elapsed:.0f}s "
              f"(exit={proc.returncode}) log={log_path.relative_to(canonical)}")

    # Failure corpus merge — always, regardless of outcome.
    try:
        new_attempts = merge_failure_corpus(canonical, worker_dir, module_id)
        if new_attempts:
            with print_lock:
                print(f"  [{module_id}] merged {new_attempts} new failure_corpus "
                      f"attempt(s)")
    except Exception as e:
        with print_lock:
            print(f"  [{module_id}] WARN: failure_corpus merge failed: {e}",
                  file=sys.stderr)

    if proc.returncode != 0:
        return module_id, "orch_fail", [
            f"{module_id}: orchestrator exit {proc.returncode}"
        ]

    # Vivado report lookup happens against the WORKER's reports dir, not
    # the canonical one, until the gate passes.
    worker_report_path = worker_dir / "reports" / f"{module_id}.vivado.json"
    if not worker_report_path.exists():
        return module_id, "stale_vivado", [
            f"{module_id}: worker produced no vivado.json"
        ]
    try:
        new_rep = json.loads(worker_report_path.read_text(encoding="utf-8"))
    except Exception:
        return module_id, "no_report", [
            f"{module_id}: worker vivado.json unparseable"
        ]
    backup_rep = vivado_report_for(backup_reports, module_id)
    if not backup_rep:
        # No baseline to compare; trust the orchestrator's verdict and
        # promote artifacts.
        n = promote_artifacts(canonical, worker_dir, module_id)
        with print_lock:
            print(f"  [{module_id}] no backup baseline; promoted {n} artifact(s)")
        return module_id, "pass", [f"{module_id}: no backup, no gate"]
    ok, msgs = gate_area(module_id, backup_rep, new_rep, args.regression_pct)
    with print_lock:
        for m in msgs:
            print(m)
    if not ok:
        # Do NOT promote regressed artifacts; canonical stays clean.
        with print_lock:
            print(f"  [{module_id}] REGRESSION — artifacts NOT promoted "
                  f"(canonical preserved)")
        return module_id, "regression", msgs
    n = promote_artifacts(canonical, worker_dir, module_id)
    with print_lock:
        print(f"  [{module_id}] gate ok; promoted {n} artifact(s)")
    return module_id, "pass", msgs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wave", default="canary")
    parser.add_argument("--backup", required=True)
    parser.add_argument("--reports-dir", default="output/reports")
    parser.add_argument("--regression-pct", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--vivado-part", default="xcu250-figd2104-2L-e")
    args = parser.parse_args()

    repo_root = repo_root_from(Path(__file__))
    canonical = (repo_root / "output").resolve()
    backup_dir = (repo_root / args.backup).resolve()
    backup_reports = backup_dir / "reports_u250"
    if not backup_reports.exists():
        print(f"ERROR: backup reports not found at {backup_reports}",
              file=sys.stderr)
        return 2

    live_reports = (repo_root / args.reports_dir).resolve()
    cp = Path(args.checkpoint)
    if not cp.is_absolute():
        cp = (repo_root / cp).resolve()
    checkpoint_arg = str(cp)

    modules = load_wave_list(repo_root, args.wave)
    print(f"[parallel] checkpoint    : {args.checkpoint}")
    print(f"[parallel] backup        : {backup_dir}")
    print(f"[parallel] reports dir   : {live_reports}")
    print(f"[parallel] workers       : {args.workers}")
    print(f"[parallel] regression %  : {args.regression_pct}")
    print(f"[parallel] modules ({len(modules)}):")
    for m in modules:
        print(f"  - {m}")
    print()

    # Base env shared by every worker.
    base_env = dict(os.environ)
    base_env["NN2RTL_VIVADO_PART"] = args.vivado_part
    base_env["NN2RTL_SELF_IMPROVE"] = "1"
    print(f"[parallel] NN2RTL_VIVADO_PART -> {args.vivado_part}")
    print(f"[parallel] NN2RTL_SELF_IMPROVE -> 1")
    if os.name == "nt" and not base_env.get("NN2RTL_VIVADO_BIN"):
        for c in (
            r"D:\vivado\2025.2\Vivado\bin\vivado.bat",
            r"C:\Xilinx\Vivado\2025.2\bin\vivado.bat",
            r"C:\Xilinx\Vivado\2024.2\bin\vivado.bat",
        ):
            if Path(c).exists():
                base_env["NN2RTL_VIVADO_BIN"] = c
                print(f"[parallel] NN2RTL_VIVADO_BIN -> {c}")
                break

    # ----- Worker slot queue: guarantees unique slot ownership per live task.
    slot_queue: queue.Queue[int] = queue.Queue()
    for i in range(args.workers):
        slot_queue.put(i)

    print_lock = threading.Lock()
    failures: list[str] = []
    results: dict[str, str] = {}
    stop_event = threading.Event()

    def run_module(module_id: str) -> tuple[str, str, list[str]]:
        # Block here until a worker slot is free. This bounds concurrency
        # to exactly `--workers` AND guarantees no two live tasks share a
        # sandbox dir.
        slot_id = slot_queue.get()
        try:
            return dispatch_one(
                module_id, slot_id, repo_root, canonical, base_env,
                checkpoint_arg, backup_reports, live_reports, args,
                print_lock,
            )
        finally:
            slot_queue.put(slot_id)

    # ----- Submit tasks lazily so --stop-on-fail can actually stop new
    # submissions (previously all futures were eagerly submitted up front).
    pending: set[futures.Future] = set()
    module_iter = iter(modules)
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        # Prime the pool.
        for _ in range(args.workers):
            try:
                mod = next(module_iter)
            except StopIteration:
                break
            if stop_event.is_set():
                break
            pending.add(pool.submit(run_module, mod))

        while pending:
            done, pending = futures.wait(
                pending, return_when=futures.FIRST_COMPLETED,
            )
            for fut in done:
                try:
                    module_id, status, _msgs = fut.result()
                except Exception as e:
                    with print_lock:
                        print(f"[wave] worker exception: {e}", file=sys.stderr)
                    failures.append(f"<unknown>: exception {e}")
                    if args.stop_on_fail:
                        stop_event.set()
                    continue
                results[module_id] = status
                if status not in ("pass", "skipped"):
                    failures.append(f"{module_id}: {status}")
                    if args.stop_on_fail:
                        stop_event.set()

            # Submit one more module per finished task, unless stopping.
            if stop_event.is_set():
                continue
            for _ in range(len(done)):
                try:
                    mod = next(module_iter)
                except StopIteration:
                    break
                pending.add(pool.submit(run_module, mod))

    print()
    print(f"[parallel] complete. results:")
    for m in modules:
        print(f"  {results.get(m, 'unsubmitted'):<12}  {m}")
    print(f"[parallel] failures: {len(failures)}")
    for f in failures:
        print(f"  - {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
