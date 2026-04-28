"""Run the full demo asset pipeline in one command.

Produces every demo artifact judges and blog readers will see:
    1. Eval bar chart with Δ-improvement annotations
    2. Training curve with smoothing + peak marker
    3. Track grid montage (trained policy across 8 tracks)
    4. Before/after GIFs for all four scenario families
    5. Per-track demo GIFs (trained policy on Monza, Spa, Suzuka, ...)

Usage:
    python scripts/run_full_demo.py
    python scripts/run_full_demo.py --skip-track-gifs    # faster, only static images
    python scripts/run_full_demo.py --grpo-run-dir ./grpo_v1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grpo-run-dir", default="./grpo_v1")
    parser.add_argument("--n-seeds", type=int, default=2,
                        help="seeds-per-mode for the eval sweep")
    parser.add_argument("--skip-eval", action="store_true",
                        help="reuse results/eval_summary.json instead of re-running")
    parser.add_argument("--skip-track-gifs", action="store_true",
                        help="only emit static grid PNG (no per-track GIFs)")
    parser.add_argument("--skip-comparisons", action="store_true",
                        help="don't run before/after GIFs for each family")
    parser.add_argument("--families", nargs="+",
                        default=["dry_strategy_sprint", "weather_roulette",
                                 "late_safety_car", "championship_decider"])
    parser.add_argument("--tracks", nargs="+",
                        default=["Monza", "Spa", "Silverstone", "Suzuka",
                                 "Spielberg", "Catalunya", "Zandvoort", "Sakhir"])
    args = parser.parse_args()

    started = time.time()
    completed: list[tuple[str, str, float]] = []  # (step, status, seconds)
    print("=" * 72)
    print(" F1 Strategist — full demo asset pipeline")
    print("=" * 72)

    # 1. Evaluation sweep
    if not args.skip_eval:
        completed.append(_run_step(
            "[1/5] held-out evaluation",
            [
                sys.executable, "evaluate.py",
                "--tasks", *args.families,
                "--n-seeds", str(args.n_seeds),
                "--modes", "random", "untrained", "trained", "expert",
                "--output-json", "results/eval_summary.json",
                "--output-png", "results/eval_curve.png",
            ],
        ))
    else:
        completed.append(("[1/5] held-out evaluation",
                         "skipped (using existing results/eval_summary.json)", 0.0))

    # 2. Training curve
    completed.append(_run_step(
        "[2/5] training curve",
        [
            sys.executable, "scripts/plot_training_curve.py",
            "--run-dir", args.grpo_run_dir,
            "--output", "results/training_loss_curve.png",
        ],
    ))

    # 3. Track grid — expert mode with track-appropriate scenarios
    grid_args = [
        sys.executable, "scripts/render_track_grid.py",
        "--mode", "expert",    # expert gives highest + most varied scores
        "--output", "results/track_grid.png",
    ]
    if args.skip_track_gifs:
        grid_args.append("--no-individual-gifs")
    completed.append(_run_step("[3/5] track grid + per-track GIFs", grid_args))

    # 3b. Race story chart (before/after in a single static PNG)
    completed.append(_run_step(
        "[3b/5] race story chart",
        [
            sys.executable, "scripts/plot_race_story.py",
            "--task", "weather_roulette",
            "--seed", "7",
            "--output", "results/race_story.png",
        ],
    ))

    # 4. Before/after comparisons
    if not args.skip_comparisons:
        for i, family in enumerate(args.families):
            seed = {"weather_roulette": 7, "dry_strategy_sprint": 0,
                    "late_safety_car": 3, "championship_decider": 5}.get(family, i)
            completed.append(_run_step(
                f"[4/5] before/after — {family}",
                [
                    sys.executable, "scripts/render_compare.py",
                    "--task", family,
                    "--seed", str(seed),
                    "--output", f"captures/before_after_{family}.gif",
                ],
            ))

    # 5. Final summary write-out
    summary_path = REPO_ROOT / "results" / "DEMO_PIPELINE_SUMMARY.md"
    _write_summary(summary_path, completed, time.time() - started)

    elapsed = time.time() - started
    print()
    print("=" * 72)
    print(f" Done in {elapsed:.1f}s")
    print("=" * 72)
    for step, status, t in completed:
        print(f"  {step:<46}  {status:<10}  ({t:.1f}s)")
    print()
    print("Assets ready:")
    for p in [
        "results/eval_curve.png",
        "results/training_loss_curve.png",
        "results/track_grid.png",
        "results/eval_summary.json",
        "results/DEMO_PIPELINE_SUMMARY.md",
        *[f"captures/before_after_{f}.gif" for f in args.families
          if not args.skip_comparisons],
    ]:
        full = REPO_ROOT / p
        if full.exists():
            sz = full.stat().st_size
            print(f"  {p}  ({_human_bytes(sz)})")


def _run_step(label: str, cmd: list[str]) -> tuple[str, str, float]:
    print(f"\n>>> {label}")
    print("    " + " ".join(cmd))
    start = time.time()
    try:
        res = subprocess.run(
            cmd, cwd=REPO_ROOT, check=False, text=True,
            capture_output=False,
        )
        elapsed = time.time() - start
        if res.returncode == 0:
            return (label, "ok", elapsed)
        else:
            return (label, f"FAIL (rc={res.returncode})", elapsed)
    except Exception as exc:
        return (label, f"ERROR: {exc}", time.time() - start)


def _write_summary(output: Path, completed, total_s: float) -> None:
    eval_json = REPO_ROOT / "results" / "eval_summary.json"
    if not eval_json.exists():
        return
    try:
        data = json.loads(eval_json.read_text(encoding="utf-8"))
    except Exception:
        return

    lines = [
        "# Demo pipeline summary",
        "",
        f"Total wall-clock: {total_s:.1f}s",
        "",
        "## Held-out evaluation",
        "",
        "| family | random | untrained | **trained** | expert |",
        "|---|---:|---:|---:|---:|",
    ]
    families = list(data.get("trained", {}).keys())
    for family in families:
        row = [
            family.replace("_", " "),
            f"{data.get('random', {}).get(family, {}).get('mean', 0):.3f}",
            f"{data.get('untrained', {}).get(family, {}).get('mean', 0):.3f}",
            f"**{data.get('trained', {}).get(family, {}).get('mean', 0):.3f}**",
            f"{data.get('expert', {}).get(family, {}).get('mean', 0):.3f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.extend([
        "",
        "## Pipeline steps",
        "",
        "| step | status | time |",
        "|---|---|---:|",
    ])
    for step, status, t in completed:
        lines.append(f"| {step} | {status} | {t:.1f}s |")
    lines.extend([
        "",
        "## Artifacts produced",
        "",
        "- `results/eval_curve.png` — annotated bar chart with Δ-improvement arrows",
        "- `results/training_loss_curve.png` — smoothed reward curve + log-scale loss",
        "- `results/track_grid.png` — trained policy across 8 tracks",
        "- `captures/before_after_*.gif` — paired untrained-vs-trained for each family",
        "- `captures/<task>__<track>_*_seed*.gif` — individual per-track demos",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n //= 1024
    return f"{n:.1f}GB"


if __name__ == "__main__":
    main()
