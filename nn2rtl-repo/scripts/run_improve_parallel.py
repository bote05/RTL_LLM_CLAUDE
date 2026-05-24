#!/usr/bin/env python3
"""Parallel improve sweep.

Spawns N orchestrator processes concurrently, each running
`npx tsx sdk/main.ts improve <module> --targets=...` for one SPATIAL
candidate from the Task-12 list. Per-worker isolation is achieved via
NN2RTL_OUTPUT_DIR (output sandbox) plus NN2RTL_KNOWLEDGE_DIR (knowledge
sandbox with its own doc_lifecycle.json and improved/ tiers).

Per-module targets are computed from current Vivado baseline using the
Task-12 rules:
    R1: lut > 30000           -> reduce-lut
    R2: bram18==0 and ff>5000 -> use-bram
    R3: ff > 50000            -> reduce-ff
After all workers complete:
    - improved/ files (patterns + references) are merged to canonical
    - doc_lifecycle.json entries are merged into canonical serially

Requires the sdk/improve.ts patch that honors NN2RTL_KNOWLEDGE_DIR; without
it, workers race on the canonical knowledge/doc_lifecycle.json.

Usage:
    py scripts/run_improve_parallel.py \\
        --modules conv_252,conv_258,... \\
        --workers 4 \\
        [--network resnet-50] \\
        [--keep-reference]
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


SPATIAL_DEFAULT = [
    "node_conv_252", "node_conv_258", "node_conv_262",
    "node_conv_270", "node_conv_276", "node_conv_284",
    "node_conv_288",
    "node_conv_220", "node_conv_228", "node_conv_234", "node_conv_240",
    "node_conv_224", "node_conv_244",
    # Already lean (skip): conv_248, conv_268, conv_274
]


def repo_root_from(script: Path) -> Path:
    return script.resolve().parent.parent


def vivado_metrics(report_dir: Path, module_id: str) -> dict | None:
    p = report_dir / f"{module_id}.vivado.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def compute_targets(metrics: dict) -> list[str]:
    """Apply Task-12 rules R1/R2/R3 in order; emit a non-empty target list."""
    lut = float(metrics.get("lut_count", 0) or 0)
    ff = float(metrics.get("ff_count", 0) or 0)
    bram = float(metrics.get("bram18_equiv", metrics.get("bram18_count", 0)) or 0)
    targets: list[str] = []
    if lut > 30000:
        targets.append("reduce-lut")
    if bram == 0 and ff > 5000:
        targets.append("use-bram")
    if ff > 50000:
        targets.append("reduce-ff")
    return targets


def make_junction(target: Path, link: Path) -> None:
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


def setup_worker_dirs(
    canonical_output: Path,
    canonical_knowledge: Path,
    worker_output: Path,
    worker_knowledge: Path,
) -> None:
    """Bootstrap per-worker output + knowledge sandboxes.

    Output sandbox: copy small mutable state (layer_ir, contract_state),
    junction read-mostly dirs (goldens, weights), clear writable subdirs.
    Knowledge sandbox: deep-copy the entire knowledge tree (~600KB) so
    the worker's doc_lifecycle.json writes do not race with peers.
    """
    # Output sandbox
    worker_output.mkdir(parents=True, exist_ok=True)
    for sub in ("rtl", "tb", "reports", "failure_corpus", "tmp", "improve"):
        sub_path = worker_output / sub
        if sub_path.exists() and not sub_path.is_symlink():
            shutil.rmtree(sub_path, ignore_errors=True)
        sub_path.mkdir(parents=True, exist_ok=True)
    for fname in ("layer_ir.json", "layer_ir.json.checkpoint",
                  "contract_state.json", "pipeline_state.json"):
        src = canonical_output / fname
        if src.exists():
            shutil.copy2(src, worker_output / fname)
    for d in ("goldens", "weights"):
        src_dir = canonical_output / d
        if src_dir.exists():
            make_junction(src_dir, worker_output / d)
    # Also seed the worker's rtl/ + reports/ with the canonical RTL + Vivado
    # report for THIS module so the improve flow has the "original passing"
    # artifacts to read.
    src_rtl_dir = canonical_output / "rtl"
    if src_rtl_dir.exists():
        for child in src_rtl_dir.iterdir():
            if child.is_file():
                shutil.copy2(child, worker_output / "rtl" / child.name)
    src_reports_dir = canonical_output / "reports"
    if src_reports_dir.exists():
        for child in src_reports_dir.iterdir():
            if child.is_file() and (
                child.name.endswith(".vivado.json")
                or child.name.endswith(".results.json")
                or child.name.endswith(".metrics.json")
            ):
                shutil.copy2(child, worker_output / "reports" / child.name)

    # Knowledge sandbox — full copy so workers' lifecycle.json + improved/
    # writes do not race.
    if worker_knowledge.exists():
        shutil.rmtree(worker_knowledge, ignore_errors=True)
    shutil.copytree(canonical_knowledge, worker_knowledge)


def merge_improved_back(
    canonical_knowledge: Path,
    worker_knowledge: Path,
    module_id: str,
    merge_lock: threading.Lock,
) -> dict:
    """Merge a worker's improved/ files + lifecycle entries into canonical.

    Holds merge_lock so concurrent workers serialize their lifecycle writes.
    Returns a summary dict.
    """
    summary = {"improved_files": 0, "lifecycle_entries": 0, "lifecycle_existed": False}
    with merge_lock:
        # 1) Copy any improved/ files that mention this module_id
        for tier in ("patterns", "references"):
            src_dir = worker_knowledge / tier / "improved"
            dst_dir = canonical_knowledge / tier / "improved"
            dst_dir.mkdir(parents=True, exist_ok=True)
            if not src_dir.exists():
                continue
            for child in src_dir.iterdir():
                if not child.is_file():
                    continue
                if module_id not in child.name:
                    continue
                dst = dst_dir / child.name
                # Overwrite — workers operate on disjoint module slugs but
                # within one module a re-run is allowed to refresh content.
                shutil.copy2(child, dst)
                summary["improved_files"] += 1
        # 2) Merge doc_lifecycle.json entries for this module
        src_lc = worker_knowledge / "doc_lifecycle.json"
        dst_lc = canonical_knowledge / "doc_lifecycle.json"
        if not src_lc.exists():
            return summary
        worker_lc = json.loads(src_lc.read_text(encoding="utf-8"))
        if dst_lc.exists():
            canonical_lc = json.loads(dst_lc.read_text(encoding="utf-8"))
            summary["lifecycle_existed"] = True
        else:
            canonical_lc = {"version": 1, "docs": {}}
        worker_docs = worker_lc.get("docs", {})
        canonical_docs = canonical_lc.setdefault("docs", {})
        for doc_id, entry in worker_docs.items():
            # Only adopt entries created/touched by this module — the worker's
            # knowledge was a snapshot of canonical, so leaving other entries
            # alone keeps the canonical's view of OTHER modules authoritative.
            created_by = entry.get("created_by_module")
            derived = entry.get("derived_from_modules", [])
            relevant = (created_by == module_id) or (module_id in derived)
            if not relevant:
                continue
            canonical_docs[doc_id] = entry
            summary["lifecycle_entries"] += 1
        dst_lc.write_text(
            json.dumps(canonical_lc, indent=2) + "\n", encoding="utf-8",
        )
    return summary


def promote_improved_rtl(
    canonical_output: Path,
    worker_output: Path,
    module_id: str,
) -> int:
    """Copy the improved RTL + reports for this module to canonical.

    `commitImprovedReference` already wrote the improved RTL to
    worker_output/rtl/<module>.v (via NN2RTL_OUTPUT_DIR). Promote those
    artifacts so downstream linters and the top-wrapper builder see them.
    """
    count = 0
    files = [
        ("rtl", f"{module_id}.v"),
        ("rtl", f"{module_id}.meta.json"),
        ("reports", f"{module_id}.vivado.json"),
        ("reports", f"{module_id}.results.json"),
        ("reports", f"{module_id}.metrics.json"),
    ]
    for sub, fname in files:
        src = worker_output / sub / fname
        if not src.exists():
            continue
        dst = canonical_output / sub / fname
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        count += 1
    return count


def run_one_worker(
    module_id: str,
    targets: list[str],
    worker_id: int,
    canonical_output: Path,
    canonical_knowledge: Path,
    repo_root: Path,
    network_id: str,
    keep_reference: bool,
    base_env: dict,
    print_lock: threading.Lock,
    merge_lock: threading.Lock,
) -> tuple[str, str, list[str]]:
    """Dispatch one improve <module> in a sandboxed subprocess.

    Returns (module_id, status, msgs).
    status in {"pass", "fail_no_improvement", "orch_fail", "skipped"}.
    """
    worker_output = canonical_output / f"_worker_{worker_id}"
    worker_knowledge = canonical_output / f"_worker_{worker_id}_knowledge"

    if not targets:
        with print_lock:
            print(f"  [{module_id}] skipped — no rule fires (already lean)")
        return module_id, "skipped", []

    setup_worker_dirs(
        canonical_output, canonical_knowledge,
        worker_output, worker_knowledge,
    )

    env = dict(base_env)
    env["NN2RTL_OUTPUT_DIR"] = str(worker_output)
    env["NN2RTL_KNOWLEDGE_DIR"] = str(worker_knowledge)

    npx_exe = "npx.cmd" if os.name == "nt" else "npx"
    cmd = [
        npx_exe, "tsx", "sdk/main.ts", "improve",
        module_id,
        f"--targets={','.join(targets)}",
        f"--network={network_id}",
    ]
    if keep_reference:
        cmd.append("--keep-reference")

    with print_lock:
        print(f"[improve] === dispatching {module_id} "
              f"targets=[{','.join(targets)}] (worker {worker_id}) ===")

    start = time.time()
    log_dir = canonical_output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"improve_worker_{worker_id}_{module_id}.log"
    with log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.run(
            cmd, cwd=repo_root, shell=False, env=env,
            stdout=log_fh, stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - start

    with print_lock:
        print(f"[improve] [{module_id}] worker {worker_id} done in "
              f"{elapsed:.0f}s (exit={proc.returncode}) "
              f"log={log_path.relative_to(canonical_output)}")

    if proc.returncode != 0:
        return module_id, "orch_fail", [
            f"{module_id}: improve subprocess exited {proc.returncode}"
        ]

    # Check whether commitImprovedReference fired — if it did, the worker
    # wrote `<module>.v` over the seed copy in worker_output/rtl/.
    worker_rtl = worker_output / "rtl" / f"{module_id}.v"
    canonical_rtl = canonical_output / "rtl" / f"{module_id}.v"
    if not worker_rtl.exists():
        return module_id, "orch_fail", [
            f"{module_id}: worker rtl missing after improve"
        ]
    # Heuristic: worker_rtl's mtime > start_time means improve replaced it.
    if worker_rtl.stat().st_mtime < start:
        with print_lock:
            print(f"  [{module_id}] no improvement promoted "
                  f"(canonical RTL kept)")
        return module_id, "fail_no_improvement", []

    promoted = promote_improved_rtl(
        canonical_output, worker_output, module_id,
    )
    merge_summary = merge_improved_back(
        canonical_knowledge, worker_knowledge, module_id, merge_lock,
    )

    # Read worker's Vivado report for delta summary
    worker_vivado = vivado_metrics(worker_output / "reports", module_id)
    canonical_vivado = vivado_metrics(canonical_output / "reports", module_id)
    msgs: list[str] = []
    if worker_vivado is not None:
        new_lut = worker_vivado.get("lut_count", "?")
        new_ff = worker_vivado.get("ff_count", "?")
        new_bram = worker_vivado.get("bram18_count", "?")
        msgs.append(
            f"  [{module_id}] improved -> LUT={new_lut} FF={new_ff} "
            f"BRAM18={new_bram} "
            f"(promoted {promoted} artifacts, "
            f"{merge_summary['improved_files']} improved files, "
            f"{merge_summary['lifecycle_entries']} lifecycle entries)"
        )
    return module_id, "pass", msgs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modules", default=",".join(SPATIAL_DEFAULT),
                    help="comma-separated module ids (defaults to SPATIAL list)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--network", default="resnet-50")
    ap.add_argument("--keep-reference", action="store_true",
                    help="pass --keep-reference to improve flow")
    args = ap.parse_args()

    repo_root = repo_root_from(Path(__file__))
    canonical_output = repo_root / "output"
    canonical_knowledge = repo_root / "knowledge"

    modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    if not modules:
        print("[improve] no modules specified", file=sys.stderr)
        return 1

    # Compute per-module targets from canonical Vivado reports.
    module_targets: list[tuple[str, list[str]]] = []
    for m in modules:
        metrics = vivado_metrics(canonical_output / "reports", m)
        if metrics is None:
            print(f"[improve] WARN: no Vivado report for {m}; skipping",
                  file=sys.stderr)
            continue
        targets = compute_targets(metrics)
        module_targets.append((m, targets))

    print(f"[improve] checkpoint     : {canonical_output}")
    print(f"[improve] knowledge      : {canonical_knowledge}")
    print(f"[improve] network        : {args.network}")
    print(f"[improve] workers        : {args.workers}")
    print(f"[improve] modules ({len(module_targets)}):")
    for m, t in module_targets:
        print(f"  - {m}  targets=[{','.join(t) if t else 'NONE'}]")
    print()

    base_env = dict(os.environ)
    base_env.setdefault("NN2RTL_VIVADO_PART", "xcu250-figd2104-2L-e")
    base_env.setdefault("NN2RTL_SELF_IMPROVE", "1")
    if os.name == "nt" and not base_env.get("NN2RTL_VIVADO_BIN"):
        for c in (
            r"D:\vivado\2025.2\Vivado\bin\vivado.bat",
            r"C:\Xilinx\Vivado\2025.2\bin\vivado.bat",
            r"C:\Xilinx\Vivado\2024.2\bin\vivado.bat",
        ):
            if Path(c).exists():
                base_env["NN2RTL_VIVADO_BIN"] = c
                break
    print(f"[improve] NN2RTL_VIVADO_BIN  : {base_env.get('NN2RTL_VIVADO_BIN', 'NOT SET')}")
    print(f"[improve] NN2RTL_VIVADO_PART : {base_env['NN2RTL_VIVADO_PART']}")
    print(f"[improve] NN2RTL_SELF_IMPROVE: {base_env['NN2RTL_SELF_IMPROVE']}")
    print()

    print_lock = threading.Lock()
    merge_lock = threading.Lock()

    results: dict[str, tuple[str, list[str]]] = {}
    slot_q: queue.Queue[int] = queue.Queue()
    for i in range(args.workers):
        slot_q.put(i)

    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_module = {}
        for module_id, targets in module_targets:
            slot = slot_q.get()
            fut = pool.submit(
                run_one_worker,
                module_id, targets, slot,
                canonical_output, canonical_knowledge, repo_root,
                args.network, args.keep_reference,
                base_env, print_lock, merge_lock,
            )
            future_to_module[fut] = (module_id, slot)

            def _release_slot(f, _slot=slot):
                slot_q.put(_slot)
            fut.add_done_callback(_release_slot)

        for fut in futures.as_completed(future_to_module):
            module_id, _ = future_to_module[fut]
            try:
                m_id, status, msgs = fut.result()
                results[m_id] = (status, msgs)
                for line in msgs:
                    print(line)
            except Exception as e:
                results[module_id] = ("orch_fail",
                                       [f"{module_id}: exception {e}"])
                print(f"[improve] [{module_id}] EXCEPTION: {e}",
                      file=sys.stderr)

    print()
    print("[improve] complete. results:")
    failures = []
    passes = 0
    skipped = 0
    for m, _ in module_targets:
        status = results.get(m, ("unknown", []))[0]
        print(f"  {status:<20} {m}")
        if status == "pass":
            passes += 1
        elif status == "skipped":
            skipped += 1
        else:
            failures.append((m, status))
    print(f"[improve] passes={passes} skipped={skipped} failures={len(failures)}")
    if failures:
        for m, s in failures:
            print(f"  - {m}: {s}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
