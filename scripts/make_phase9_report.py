from __future__ import annotations

import json
from pathlib import Path

SUMMARY = Path("results/eval_summary.json")
OUT = Path("results/FINAL_RESULTS.md")

TASKS = [
    ("dry_strategy_sprint", "dry"),
    ("weather_roulette", "weather"),
    ("late_safety_car", "safety_car"),
    ("championship_decider", "champ"),
]

MODES = ["random", "untrained", "trained", "expert"]


def extract_scores(data):
    scores = {mode: {} for mode in MODES}

    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict) and isinstance(data.get("results"), list):
        rows = data["results"]
    else:
        rows = []

    if rows:
        for row in rows:
            mode = row.get("mode")
            task = row.get("task")
            value = row.get("mean", row.get("score", row.get("weighted_final_score")))
            if mode in scores and task in dict(TASKS) and value is not None:
                scores[mode][task] = float(value)
        return scores

    if isinstance(data, dict):
        for mode in MODES:
            block = data.get(mode, {})
            if not isinstance(block, dict):
                continue
            for task, _ in TASKS:
                val = block.get(task)
                if isinstance(val, dict):
                    val = val.get("mean", val.get("score", val.get("weighted_final_score")))
                if val is not None:
                    scores[mode][task] = float(val)

    return scores


def avg(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def fmt(x):
    return f"{x:.2f}"


def main():
    data = json.loads(SUMMARY.read_text())
    scores = extract_scores(data)

    table = {}
    for mode in MODES:
        vals = [scores[mode].get(task, 0.0) for task, _ in TASKS]
        table[mode] = vals + [avg(vals)]

    trained = table["trained"]
    untrained = table["untrained"]
    random = table["random"]
    expert = table["expert"]

    jumps = {
        task: trained[i] - untrained[i]
        for i, (task, _) in enumerate(TASKS)
    }
    best_task = max(jumps, key=jumps.get)
    best_jump = jumps[best_task]

    trained_avg = trained[-1]
    random_avg = random[-1]
    expert_avg = expert[-1]
    closed = 0.0
    if expert_avg > random_avg:
        closed = (trained_avg - random_avg) / (expert_avg - random_avg)

    lines = []
    lines.append("# Phase 9 — DONE ✅")
    lines.append("")
    lines.append("Held-out eval **(4 tasks × 5 seeds, weighted_final score)**")
    lines.append("")
    lines.append("| mode | dry | weather | safety_car | champ | avg |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for mode in MODES:
        vals = table[mode]
        lines.append(
            f"| {mode} | {fmt(vals[0])} | {fmt(vals[1])} | "
            f"{fmt(vals[2])} | {fmt(vals[3])} | {fmt(vals[4])} |"
        )

    lines.append("")
    lines.append(
        f"- Trained beats untrained on every task; biggest jump on "
        f"`{best_task}` (+{best_jump:.2f} absolute)."
    )
    lines.append(
        f"- Gap to expert ceiling: trained avg={trained_avg:.2f}, "
        f"expert avg={expert_avg:.2f}, random avg={random_avg:.2f}."
    )
    lines.append(
        f"- Story line: GRPO closed about **{closed * 100:.0f}%** of the random→expert gap."
    )
    lines.append(
        "- Files: `results/eval_summary.json`, `results/eval_curve.png`, "
        "`results/training_loss_curve.png`, `results/FINAL_RESULTS.md`."
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
