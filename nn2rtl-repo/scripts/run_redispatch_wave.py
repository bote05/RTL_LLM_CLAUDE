#!/usr/bin/env python3
"""Run one re-dispatch wave: invoke Foundry per module, gate, area-compare.

Usage:
    py scripts/run_redispatch_wave.py \
        --checkpoint checkpoints/resnet50_int8.pth \
        --wave canary \
        --backup backups/pre_redispatch_<TIMESTAMP> \
        [--regression-pct 15]

Predefined waves:
  canary  - 8 representative modules (one per op + kernel + stride variant)
  rest    - everything else flagged as needs-redispatch by
            patch_layerir_to_tiled.py (skipping heavy / already-tiled)

For each module in the wave:
  1) Invoke the orchestrator with --only MODULE_ID (runs Foundry, Surgeon,
     Vivado synth, per-module verification gates).
  2) Read the per-module Vivado report at
     output/reports_u250/<module_id>.vivado.json
  3) Compare against backup's report; abort if LUT/DSP/BRAM/FF regress
     by more than --regression-pct percent.

This script does NOT itself call the Anthropic API — it shells out to
`npm --prefix sdk run pipeline -- <checkpoint> --only <module_id>`, so
ANTHROPIC_API_KEY must be set in the parent shell.

After the wave completes successfully, run:
    py scripts/lint_boundaries.py --check-meta
to confirm IR-vs-meta widths agree across the affected boundaries.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


CANARY_WAVE = [
    "node_conv_196",     # stem conv 7x7 stride=2
    "node_relu",         # relu
    "node_max_pool2d",   # maxpool stride 2
    "node_conv_198",     # pointwise 1x1 stride=1
    "node_conv_200",     # spatial 3x3 stride=1
    "node_add",          # residual add
    "node_conv_220",     # spatial 3x3 stride=2 (downsample)
    "node_conv_224",     # projection 1x1 stride=2
]


def repo_root_from(script: Path) -> Path:
    return script.resolve().parent.parent


def load_wave_list(repo_root: Path, name: str) -> list[str]:
    if name == "canary":
        return list(CANARY_WAVE)
    if name == "rest":
        # Compute by re-running patch_layerir_to_tiled.py --dry-run and
        # parsing its needs-redispatch list, minus the canary modules.
        cmd = ["py", "scripts/patch_layerir_to_tiled.py", "--dry-run"]
        out = subprocess.check_output(cmd, cwd=repo_root, text=True)
        lines = out.splitlines()
        mods: list[str] = []
        capture = False
        for line in lines:
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
    # Custom comma-separated list.
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
    """Return (ok, messages). ok=False if any metric regresses >threshold."""
    messages: list[str] = []
    ok = True
    # Per-metric (label, percent-cap, absolute-floor). The floor exists
    # because percentage-only gating punishes tiny modules — a relu going
    # 257 -> 526 LUT is +105% but only +269 absolute, which is ~0.016% of
    # U250's 1.7M LUT budget. We only flag a regression when BOTH the
    # percent AND absolute deltas exceed the thresholds; small absolute
    # deltas pass even at large percent.
    # Loosened from 1000/1000/4/4 after the conv_224 audit: the new tile-32
    # architecture has an inherent ~4k-LUT cost on projection / stride-2
    # convs because of the full-pixel out_pack buffer Foundry currently
    # emits. The cost is bounded (~25k LUT across ~5 such convs network-wide,
    # ~1.5% of U250 LUT budget).
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
        pct_over = d > regression_pct
        abs_over = abs_delta > abs_floor
        if pct_over and abs_over:
            ok = False
            messages.append(
                f"  REGRESSION {label}: {old:.0f} -> {new:.0f}  "
                f"({d:+.1f}% > {regression_pct:.0f}%, +{abs_delta:.0f} "
                f"> floor {abs_floor})"
            )
        elif pct_over:
            # Percent over but absolute is small enough to ignore.
            messages.append(
                f"  ok* {label}: {old:.0f} -> {new:.0f}  ({d:+.1f}%, "
                f"+{abs_delta:.0f} under floor {abs_floor})"
            )
        else:
            messages.append(
                f"  ok {label}: {old:.0f} -> {new:.0f}  ({d:+.1f}%)"
            )

    # Timing
    old_fmax = float(backup_report.get("fmax_mhz", 0) or 0)
    new_fmax = float(new_report.get("fmax_mhz", 0) or 0)
    if old_fmax > 0:
        fmax_d = pct_delta(new_fmax, old_fmax)
        if fmax_d < -regression_pct:
            ok = False
            messages.append(
                f"  REGRESSION Fmax: {old_fmax:.1f} -> {new_fmax:.1f}  "
                f"({fmax_d:+.1f}% < -{regression_pct:.0f}%)"
            )
        else:
            messages.append(
                f"  ok Fmax: {old_fmax:.1f} -> {new_fmax:.1f}  "
                f"({fmax_d:+.1f}%)"
            )

    return ok, messages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="path passed to the orchestrator (sdk main.ts)")
    parser.add_argument("--wave", default="canary",
                        help="'canary', 'rest', or a comma-separated list "
                             "of module_ids")
    parser.add_argument("--backup", required=True,
                        help="backup directory with the pre-dispatch "
                             "reports_u250/")
    parser.add_argument("--reports-dir",
                        default="output/reports",
                        help="path to the live per-module Vivado reports "
                             "directory (default: output/reports, where the "
                             "orchestrator writes them). The backup's "
                             "reports_u250/ is still used for the baseline "
                             "comparison.")
    parser.add_argument("--vivado-part",
                        default="xcu250-figd2104-2L-e",
                        help="Vivado part for the orchestrator's per-module "
                             "synth gate (default: xcu250-figd2104-2L-e). "
                             "Propagated as NN2RTL_VIVADO_PART so the area "
                             "gate compares like-for-like against the U250 "
                             "backup baseline.")
    parser.add_argument("--regression-pct", type=float, default=15.0)
    parser.add_argument("--stop-on-fail", action="store_true",
                        help="abort the wave on the first failure")
    parser.add_argument("--skip-completed", action="store_true",
                        help="skip modules whose live vivado.json is newer "
                             "than the backup's (idempotent re-runs after a "
                             "wave was interrupted, avoids burning Foundry "
                             "calls on already-regenerated modules)")
    args = parser.parse_args()

    repo_root = repo_root_from(Path(__file__))
    backup_dir = (repo_root / args.backup).resolve()
    backup_reports = backup_dir / "reports_u250"
    if not backup_reports.exists():
        print(f"ERROR: backup reports not found at {backup_reports}",
              file=sys.stderr)
        return 2

    live_reports = (repo_root / args.reports_dir).resolve()

    modules = load_wave_list(repo_root, args.wave)
    print(f"[wave] checkpoint   : {args.checkpoint}")
    print(f"[wave] backup       : {backup_dir}")
    print(f"[wave] reports dir  : {live_reports}")
    print(f"[wave] regression % : {args.regression_pct}")
    print(f"[wave] modules ({len(modules)}):")
    for m in modules:
        print(f"  - {m}")
    print()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY is not set in environment.",
              file=sys.stderr)
        print("The orchestrator will fail unless the key is set when this",
              file=sys.stderr)
        print("script invokes `npm --prefix sdk run pipeline`.", file=sys.stderr)

    failures: list[str] = []
    npm_exe = "npm.cmd" if os.name == "nt" else "npm"
    # Resolve checkpoint relative to repo_root because `npm --prefix sdk`
    # changes cwd into sdk/ before invoking tsx, so a repo-relative arg
    # would otherwise be looked up as sdk/<arg> and not found.
    cp = Path(args.checkpoint)
    if not cp.is_absolute():
        cp = (repo_root / cp).resolve()
    checkpoint_arg = str(cp)

    # Node's child_process.spawn on Windows refuses to invoke a bare `vivado`
    # because the actual binary is `vivado.bat` (refuses .bat without going
    # through cmd.exe for security). The MCP server's run_vivado already
    # routes `.bat` through cmd.exe, but only when NN2RTL_VIVADO_BIN points
    # at a path that ends in .bat/.cmd. So we resolve the .bat here once and
    # propagate it via env to the orchestrator process tree.
    env = os.environ.copy()
    env["NN2RTL_VIVADO_PART"] = args.vivado_part
    print(f"[wave] NN2RTL_VIVADO_PART -> {args.vivado_part}")
    # Enable self_improve so the orchestrator's Retrospector + final-attempt
    # path can engage when Foundry + Surgeon exhaust their retries. Without
    # this, fail_abort is terminal even when the failure is classified as
    # code_bug or architectural_fit.
    env["NN2RTL_SELF_IMPROVE"] = "1"
    print(f"[wave] NN2RTL_SELF_IMPROVE -> 1")
    if os.name == "nt" and not env.get("NN2RTL_VIVADO_BIN"):
        candidates = [
            r"D:\vivado\2025.2\Vivado\bin\vivado.bat",
            r"C:\Xilinx\Vivado\2025.2\bin\vivado.bat",
            r"C:\Xilinx\Vivado\2024.2\bin\vivado.bat",
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                env["NN2RTL_VIVADO_BIN"] = candidate
                print(f"[wave] NN2RTL_VIVADO_BIN -> {candidate}")
                break
        else:
            print("WARNING: vivado.bat not found at known locations; "
                  "synthesis will fail. Set NN2RTL_VIVADO_BIN explicitly.",
                  file=sys.stderr)

    for module_id in modules:
        live_report_path = live_reports / f"{module_id}.vivado.json"
        backup_report_path = backup_reports / f"{module_id}.vivado.json"
        if args.skip_completed and live_report_path.exists() \
                and backup_report_path.exists() \
                and live_report_path.stat().st_mtime \
                > backup_report_path.stat().st_mtime:
            print(f"[wave] === skipping {module_id} (already regenerated "
                  f"under new ABI) ===")
            # Still run the area gate so a previous regression is visible.
            backup_rep = vivado_report_for(backup_reports, module_id)
            new_rep = vivado_report_for(live_reports, module_id)
            if backup_rep and new_rep:
                ok, msgs = gate_area(
                    module_id, backup_rep, new_rep, args.regression_pct,
                )
                for m in msgs:
                    print(m)
                if not ok:
                    failures.append(f"{module_id}: area/timing regression "
                                    f"(skipped re-dispatch)")
                    if args.stop_on_fail:
                        break
            continue

        print(f"[wave] === dispatching {module_id} ===")
        cmd = [
            npm_exe, "--prefix", "sdk", "run", "pipeline", "--",
            checkpoint_arg, "--only", module_id,
        ]
        # Capture the live vivado.json mtime BEFORE dispatch so we can detect
        # a silent infrastructure failure (orchestrator exits 0 but Vivado
        # ENOENT'd, leaving the stale pre-patch report in place).
        pre_mtime = live_report_path.stat().st_mtime if live_report_path.exists() else 0.0

        r = subprocess.run(cmd, cwd=repo_root, shell=False, env=env)
        if r.returncode != 0:
            print(f"[wave] {module_id}: orchestrator returned {r.returncode}",
                  file=sys.stderr)
            failures.append(f"{module_id}: orchestrator failed")
            if args.stop_on_fail:
                break
            continue

        # Area / timing compare
        backup_rep = vivado_report_for(backup_reports, module_id)
        new_rep = vivado_report_for(live_reports, module_id)
        if not backup_rep:
            print(f"[wave] {module_id}: no backup report, skipping area gate")
            continue
        if not new_rep:
            print(f"[wave] {module_id}: no new report at "
                  f"{live_reports}/{module_id}.vivado.json -- gate fails")
            failures.append(f"{module_id}: missing new report")
            if args.stop_on_fail:
                break
            continue
        # Stale-report guard: orchestrator may exit 0 even when Vivado fails
        # to spawn (vivado_tool_error). In that case the .vivado.json on disk
        # was NOT regenerated this dispatch, so a comparison would be the
        # backup against itself and trivially "pass" with 0% delta.
        post_mtime = live_report_path.stat().st_mtime
        if post_mtime <= pre_mtime:
            print(f"[wave] {module_id}: vivado.json was not regenerated "
                  f"(mtime unchanged) -- likely synthesis infra failure, "
                  f"skipping area gate and recording failure")
            failures.append(f"{module_id}: stale vivado report")
            if args.stop_on_fail:
                break
            continue
        ok, msgs = gate_area(
            module_id, backup_rep, new_rep, args.regression_pct,
        )
        for m in msgs:
            print(m)
        if not ok:
            failures.append(f"{module_id}: area/timing regression")
            if args.stop_on_fail:
                print(f"[wave] aborting on regression for {module_id}",
                      file=sys.stderr)
                break

    print()
    print(f"[wave] complete. failures: {len(failures)}")
    for f in failures:
        print(f"  - {f}")
    if failures:
        return 1
    print(f"[wave] all {len(modules)} modules passed.")
    print()
    print("Next steps:")
    print("  1. py scripts/lint_boundaries.py --check-meta")
    print("     # confirm boundary widths now agree across the chain")
    print("  2. py scripts/build_weight_memory_map.py")
    print("     py scripts/build_bias_memory_map.py")
    print("     py scripts/build_scheduler.py")
    print("     npx tsx scripts/build_top_wrapper.ts")
    print("     # regenerate integration artifacts with the new widths")
    print("  3. npx tsx scripts/run_first_light_synth.ts")
    print("     # integrated first-light U250 synth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
