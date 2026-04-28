"""Capture pre-training performance baseline across all scenarios and modes.

Run this BEFORE training to establish the comparison baseline required for
the progress report and reward-improvement evidence (20% of judging weight).

The output JSON is the authoritative "before training" artefact.  After
training, run evaluate.py and compare results/eval_summary.json against
results/pretrain_baseline.json to show measurable reward improvement.

Usage:
    # From repo root:
    python scripts/capture_pretrain_baseline.py --n-seeds 10
    python scripts/capture_pretrain_baseline.py --n-seeds 3   # quick smoke
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# Ensure repo root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import run_one  # noqa: E402
from server.scenarios import SCENARIOS  # noqa: E402

CANONICAL_TASKS = [
    "dry_strategy_sprint",
    "weather_roulette",
    "late_safety_car",
    "championship_decider",
    "virtual_safety_car_window",
    "tyre_cliff_management",
]

MODES = ["random", "untrained", "expert"]


def capture_baseline(n_seeds: int = 10, output: str = "results/pretrain_baseline.json") -> dict:
    """Run all scenarios × seeds × modes and persist the pre-training baseline."""
    results: dict[str, dict[str, dict]] = {}

    for task in CANONICAL_TASKS:
        print(f"\n[{task}]")
        results[task] = {}
        for mode in MODES:
            scores: list[float] = []
            for seed in range(n_seeds):
                score = run_one(task, mode, seed)
                scores.append(score)
            mean = statistics.mean(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            results[task][mode] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
                "scores": [round(s, 4) for s in scores],
                "n_seeds": n_seeds,
            }
            print(f"  {mode:<12} mean={mean:.3f}  std={std:.3f}  [{min(scores):.3f}, {max(scores):.3f}]")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nBaseline saved: {output}")

    _print_table(results)
    _write_markdown(results, Path(output).with_suffix(".md"))
    return results


def _print_table(results: dict) -> None:
    header = f"{'Scenario':<32} {'Random':>8} {'Untrained':>10} {'Expert':>8}  Gap(exp-rand)"
    print("\n" + "=" * 70)
    print("Pre-Training Baseline Summary")
    print("=" * 70)
    print(header)
    print("-" * 70)
    for task, mode_data in results.items():
        r = mode_data.get("random", {}).get("mean", 0.0)
        u = mode_data.get("untrained", {}).get("mean", 0.0)
        e = mode_data.get("expert", {}).get("mean", 0.0)
        gap = e - r
        print(f"{task:<32} {r:>8.3f} {u:>10.3f} {e:>8.3f}  {gap:>+.3f}")
    print("=" * 70)


def _write_markdown(results: dict, output: Path) -> None:
    lines = [
        "# Pre-Training Baseline",
        "",
        "Scores are `weighted_final` averaged over held-out seeds.",
        "This is the **before-training** reference for the progress report.",
        "",
        "| Scenario | Random | Untrained | Expert | Gap (exp-rand) |",
        "|---|---:|---:|---:|---:|",
    ]
    for task, mode_data in results.items():
        r = mode_data.get("random", {}).get("mean", 0.0)
        u = mode_data.get("untrained", {}).get("mean", 0.0)
        e = mode_data.get("expert", {}).get("mean", 0.0)
        lines.append(f"| {task} | {r:.3f} | {u:.3f} | {e:.3f} | {e - r:+.3f} |")
    lines.extend(
        [
            "",
            "## What to do after training",
            "",
            "Run `python evaluate.py --modes random untrained trained expert` to produce",
            "`results/eval_summary.json` and `results/eval_curve.png`.",
            "The **trained** row should sit between *untrained* and *expert* on every",
            "scenario — that is the reward-improvement evidence.",
            "",
            "Compare `trained.mean` vs `untrained.mean` in this table.",
            "An improvement of +0.05 or more per scenario is a strong signal.",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown report: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture pre-training performance baseline.")
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="Number of random seeds per (scenario, mode) cell. Use 3 for a quick smoke check.",
    )
    parser.add_argument(
        "--output",
        default="results/pretrain_baseline.json",
        help="Path to write the baseline JSON.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=CANONICAL_TASKS,
        help="Subset of tasks to evaluate (default: all).",
    )
    args = parser.parse_args()
    capture_baseline(args.n_seeds, args.output)


if __name__ == "__main__":
    main()
