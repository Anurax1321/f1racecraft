"""Create a markdown diff for memory ablation eval runs."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("variant")
    parser.add_argument("--output", default="results/ablation.md")
    args = parser.parse_args()
    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    variant = json.loads(Path(args.variant).read_text(encoding="utf-8"))
    lines = [
        "# Postmortem Memory Ablation",
        "",
        "| mode | task | base | with memory | delta |",
        "|---|---|---:|---:|---:|",
    ]
    deltas = []
    for mode in sorted(set(base) & set(variant)):
        for task in sorted(set(base[mode]) & set(variant[mode])):
            b = float(base[mode][task]["mean"])
            v = float(variant[mode][task]["mean"])
            delta = v - b
            deltas.append(delta)
            lines.append(f"| {mode} | {task} | {b:.3f} | {v:.3f} | {delta:+.3f} |")
    avg = sum(deltas) / len(deltas) if deltas else 0.0
    lines.extend(["", f"Average delta: **{avg:+.3f}**"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
