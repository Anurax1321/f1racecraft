"""
sync_results_to_docs.py — single command to propagate eval numbers into all docs.

Reads `results/eval_summary.json` (the canonical source) and rewrites tables in:
  - README.md          (Current local smoke numbers table)
  - blog.md            (Results table + headline numbers in body text)
  - notebooks/f1_strategist_training_colab.ipynb  (Summary markdown cell)

Idempotent. Run it after every fresh eval. If your partner drops a new
eval_summary.json, run this once and everything else updates in place.

Usage:
    python scripts/sync_results_to_docs.py
    python scripts/sync_results_to_docs.py --dry-run    # preview changes only
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_JSON = ROOT / "results" / "eval_summary.json"
README = ROOT / "README.md"
BLOG = ROOT / "blog.md"
NOTEBOOK = ROOT / "notebooks" / "f1_strategist_training_colab.ipynb"

TASKS = [
    "dry_strategy_sprint",
    "weather_roulette",
    "late_safety_car",
    "championship_decider",
]

PRETTY = {
    "dry_strategy_sprint": "Dry strategy sprint",
    "weather_roulette": "Weather roulette",
    "late_safety_car": "Late safety car",
    "championship_decider": "Championship decider",
}


def load_eval():
    if not EVAL_JSON.exists():
        sys.exit(f"FATAL: {EVAL_JSON} does not exist. Run evaluate.py first.")
    data = json.loads(EVAL_JSON.read_text())
    for mode in ("random", "untrained", "trained", "expert"):
        if mode not in data:
            sys.exit(f"FATAL: mode '{mode}' missing from eval_summary.json")
        for t in TASKS:
            if t not in data[mode]:
                sys.exit(f"FATAL: task '{t}' missing from mode '{mode}'")
    return data


def fmt3(x: float) -> str:
    return f"{x:.3f}"


def fmt2(x: float) -> str:
    return f"{x:.2f}"


def render_readme_table(data: dict) -> str:
    lines = [
        "| scenario | random | untrained | GRPO trained | rule-based expert |",
        "|---|---:|---:|---:|---:|",
    ]
    for t in TASKS:
        r = data["random"][t]["mean"]
        u = data["untrained"][t]["mean"]
        tr = data["trained"][t]["mean"]
        e = data["expert"][t]["mean"]
        lines.append(f"| {t} | {fmt3(r)} | {fmt3(u)} | {fmt3(tr)} | {fmt3(e)} |")
    return "\n".join(lines)


def render_blog_table(data: dict) -> str:
    lines = [
        "| Scenario | Random | Untrained | GRPO trained | Expert heuristic |",
        "|---|---:|---:|---:|---:|",
    ]
    for t in TASKS:
        r = data["random"][t]["mean"]
        u = data["untrained"][t]["mean"]
        tr = data["trained"][t]["mean"]
        e = data["expert"][t]["mean"]
        bold = f"**{fmt2(tr)}**"
        lines.append(
            f"| {PRETTY[t]} | {fmt2(r)} | {fmt2(u)} | {bold} | {fmt2(e)} |"
        )
    return "\n".join(lines)


def render_notebook_summary_table(data: dict) -> str:
    lines = [
        "| Scenario | Random | Untrained | GRPO-trained | Expert |",
        "|--|--|--|--|--|",
    ]
    for t in TASKS:
        r = data["random"][t]["mean"]
        u = data["untrained"][t]["mean"]
        tr = data["trained"][t]["mean"]
        e = data["expert"][t]["mean"]
        lines.append(
            f"| {PRETTY[t]} | {fmt2(r)} | {fmt2(u)} | {fmt2(tr)} | {fmt2(e)} |"
        )
    return "\n".join(lines)


def replace_block(text: str, header_regex: str, replacement: str) -> tuple[str, int]:
    """Replace a markdown table starting with the given header regex through the
    next blank line. Returns (new_text, n_replaced)."""
    pat = re.compile(header_regex + r"[^\n]*\n(\|[^\n]*\n)+", re.MULTILINE)
    new, n = pat.subn(replacement.rstrip() + "\n\n", text, count=1)
    # Strip trailing extra blank if it doubled
    new = re.sub(r"\n{3,}", "\n\n", new)
    return new, n


def update_readme(data: dict, dry_run: bool) -> bool:
    txt = README.read_text(encoding="utf-8")
    new_table = render_readme_table(data)
    new, n = replace_block(
        txt,
        r"\| scenario \| random \| untrained \|",
        new_table,
    )
    if n == 0:
        print("[README] WARN: results table not matched; skipping")
        return False
    if dry_run:
        print(f"[README] would update {len(txt) - len(new)} chars")
        return True
    README.write_text(new, encoding="utf-8")
    print(f"[README] updated ({n} table)")
    return True


def update_blog(data: dict, dry_run: bool) -> bool:
    if not BLOG.exists():
        print(f"[blog]   WARN: {BLOG} does not exist; skipping")
        return False
    txt = BLOG.read_text(encoding="utf-8")
    new_table = render_blog_table(data)
    new, n = replace_block(
        txt,
        r"\| Scenario \| Random \| Untrained \|",
        new_table,
    )
    # Also rewrite the headline weather sentence + key-decision delta
    wr_u = data["untrained"]["weather_roulette"]["mean"]
    wr_t = data["trained"]["weather_roulette"]["mean"]
    delta = wr_t - wr_u
    new = re.sub(
        r"The trained model scores \*\*[\d.]+\*\* on the Weather Roulette scenario vs\. \*\*[\d.]+\*\* for the untrained baseline — a delta of \+[\d.]+ on a single scenario family\.",
        f"The trained model scores **{fmt2(wr_t)}** on the Weather Roulette scenario vs. **{fmt2(wr_u)}** for the untrained baseline — a delta of +{fmt2(delta)} on a single scenario family.",
        new,
        count=1,
    )
    new = re.sub(
        r"Score: \*\*0\.950\*\*, delta \+0\.5\d+\.",
        f"Score: **{wr_t:.3f}**, delta +{delta:.3f}.",
        new,
        count=1,
    )
    if n == 0:
        print("[blog]   WARN: results table not matched")
        return False
    if dry_run:
        print(f"[blog]   would update")
        return True
    BLOG.write_text(new, encoding="utf-8")
    print(f"[blog]   updated ({n} table + headline numbers)")
    return True


def update_notebook(data: dict, dry_run: bool) -> bool:
    if not NOTEBOOK.exists():
        print(f"[ipynb]  WARN: {NOTEBOOK} not found")
        return False
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    table = render_notebook_summary_table(data)

    updated_any = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "markdown":
            continue
        src = "".join(cell["source"])
        new_src = src  # always start from current source
        if "Random | Untrained | GRPO-trained | Expert" in src:
            new_src = re.sub(
                r"\| Scenario \| Random \| Untrained \| GRPO-trained \| Expert \|\n\|[^\n]+\n(\|[^\n]+\n)+",
                table + "\n",
                new_src,
                count=1,
            )

        if "+0.46 to +0.57" in src or "GRPO closed" in src or "closes \\*\\*\\+0.46" in src:
            gaps_closed = []
            for t in TASKS:
                u = data["untrained"][t]["mean"]
                tr = data["trained"][t]["mean"]
                e = data["expert"][t]["mean"]
                if e - u > 0.001:
                    gaps_closed.append((tr - u) / (e - u))
            if gaps_closed:
                lo = max(0, min(gaps_closed)) * 100
                hi = min(1.5, max(gaps_closed)) * 100
                new_src = re.sub(
                    r"GRPO[- ]?(?:closed|closes) \*\*\+?[\d.]+ ?to ?\+?[\d.]+\*\* of the gap[^\n]*",
                    f"GRPO closes **{lo:.0f}–{hi:.0f}%** of the gap to the hand-authored expert across "
                    f"every held-out scenario family.",
                    new_src,
                )

        if new_src != src:
            cell["source"] = new_src.splitlines(keepends=True)
            updated_any = True

    if not updated_any:
        print("[ipynb]  WARN: no markdown cells matched (already in sync?)")
        return False
    if dry_run:
        print("[ipynb]  would update markdown summary cells")
        return True
    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print("[ipynb]  updated summary markdown cells")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    data = load_eval()

    print(f"Source of truth: {EVAL_JSON}")
    print("Trained means:")
    for t in TASKS:
        u = data["untrained"][t]["mean"]
        tr = data["trained"][t]["mean"]
        e = data["expert"][t]["mean"]
        delta = tr - u
        gap = (tr - u) / (e - u) if e - u > 0.001 else 0
        print(f"  {t:<25} untrained={u:.3f}  trained={tr:.3f}  expert={e:.3f}  Δ=+{delta:.3f}  ({gap*100:.0f}% of gap closed)")

    print()
    update_readme(data, args.dry_run)
    update_blog(data, args.dry_run)
    update_notebook(data, args.dry_run)


if __name__ == "__main__":
    main()
