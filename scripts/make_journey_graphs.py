"""Generate the ship-time graphs that tell our journey story.

Outputs:
  results/journey.png            — overall avg per iteration (line chart)
  results/scenario_breakdown.png — 4 scenarios × 4-5 iterations (grouped bars)
  results/comparison_real.png    — main-reported (scripted) vs main-actual vs ours vs expert
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")

TASKS = ["dry_strategy_sprint", "weather_roulette", "late_safety_car",
         "championship_decider", "virtual_safety_car_window", "tyre_cliff_management"]
TASK_LABELS = ["dry sprint", "weather", "safety car", "champ", "VSC win", "tyre cliff"]
TASKS_4 = TASKS[:4]  # for backwards-compatible loaders


def load_means(path: Path, mode: str = "trained", tasks: list[str] | None = None) -> list[float]:
    """Load per-task means from an eval_*.json file."""
    data = json.loads(path.read_text())
    tasks = tasks or TASKS_4
    return [data[mode][t]["mean"] for t in tasks if t in data[mode]]


# ------------------------------------------------------------
# Data points
# ------------------------------------------------------------

# Authoritative source: full 6-scenario eval of the GRPO v2 champion
six = json.loads((RESULTS / "eval_six_scenarios.json").read_text())
RANDOM = [six["random"][t]["mean"] for t in TASKS]
UNTRAINED = [six["untrained"][t]["mean"] for t in TASKS]
EXPERT = [six["expert"][t]["mean"] for t in TASKS]
GRPO_V2 = [six["trained"][t]["mean"] for t in TASKS]  # CHAMPION (6-scenario)

# Earlier iterations were only run on 4 scenarios — pad the missing 2 with NaN
import math
def pad6(four: list[float]) -> list[float]:
    return four + [math.nan, math.nan]

# main FINAL_RESULTS.md "trained" — SCRIPTED-FALLBACK (bit-exact verified)
MAIN_REPORTED = pad6([0.749, 0.935, 0.935, 0.535])

# Real-LLM eval of main's HF checkpoint
MAIN_ACTUAL = pad6(load_means(RESULTS / "eval_shashwat.json"))

# Our iterations (4-scenario)
ORIG_GRPO = pad6([0.520, 0.560, 0.550, 0.520])
SFT_V3 = pad6(load_means(RESULTS / "eval_sft_v3.json"))
RFT_V1_T03 = pad6(load_means(RESULTS / "eval_rft_t03.json"))


# ------------------------------------------------------------
# 1. Journey line chart — avg score across iterations
# ------------------------------------------------------------

def _avg(scores: list[float]) -> float:
    """Average over non-NaN values."""
    valid = [s for s in scores if not (isinstance(s, float) and math.isnan(s))]
    return sum(valid) / len(valid) if valid else 0.0

iterations = [
    ("random", RANDOM),
    ("untrained\nQwen3-4B", UNTRAINED),
    ("initial\nGRPO 500\n(real LLM)", MAIN_ACTUAL),
    ("Our\nSFT v3", SFT_V3),
    ("Our\nRFT v1\n(T=0.3)", RFT_V1_T03),
    ("Our\nGRPO v2\n★", GRPO_V2),
    ("Expert\n(rule-based)", EXPERT),
]
labels = [name for name, _ in iterations]
avgs = [_avg(scores) for _, scores in iterations]

# F1 palette (matches server/static/style.css)
F1_BG    = "#0a0a0a"
F1_PANEL = "#0f0f0f"
F1_INK   = "#e8e8e8"
F1_DIM   = "#666666"
F1_RULE  = "#1e1e1e"
F1_RED   = "#e10600"
F1_TEAL  = "#00d2be"
F1_GOLD  = "#ffd60a"
F1_ORANGE = "#ef8a17"

def _style_axes(ax, fig):
    ax.set_facecolor(F1_PANEL)
    fig.patch.set_facecolor(F1_BG)
    ax.tick_params(colors=F1_DIM, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(F1_RULE)
    ax.grid(True, axis="y", linestyle="--", alpha=0.18, color="#444")

fig, ax = plt.subplots(figsize=(11, 5.5))
# Color per stage: dim → orange (untrained) → orange (initial) → teal (SFT) → teal (RFT) → red (winner) → gold (expert)
colors = [F1_DIM, F1_ORANGE, F1_ORANGE, F1_TEAL, F1_TEAL, F1_RED, F1_GOLD]
xs = list(range(len(labels)))
n = len(labels)
ours_idx = 5
ours_y = avgs[ours_idx]
untr_y = avgs[1]
expert_y = avgs[-1]

# --- highlight zones (drawn first, behind everything) ---
# "gap we closed" — from untrained baseline up to ours
ax.axhspan(untr_y, ours_y, color=F1_RED, alpha=0.10, zorder=0)
# "headroom" — from ours up to expert ceiling
ax.axhspan(ours_y, expert_y, color=F1_GOLD, alpha=0.06, zorder=0)
# bold band centered on ours so the eye lands on it
ax.axhspan(ours_y - 0.012, ours_y + 0.012, color=F1_RED, alpha=0.55, zorder=1)

# horizontal helper rules
ax.axhline(y=expert_y, linestyle=":", color=F1_GOLD, alpha=0.6, linewidth=1.2, label=f"expert ceiling  ({expert_y:.3f})")
ax.axhline(y=untr_y, linestyle=":", color=F1_ORANGE, alpha=0.55, linewidth=1.2, label=f"untrained baseline  ({untr_y:.3f})")
ax.axhline(y=ours_y, linestyle="-", color=F1_RED, alpha=0.95, linewidth=1.6,
           label=f"★ ours (GRPO v2)  ({ours_y:.3f})")

# right-margin labels for the bands
ax.text(n - 0.6, (untr_y + ours_y) / 2, f"+{ours_y - untr_y:.2f}\nover untrained",
        ha="right", va="center", fontsize=9, color=F1_RED, fontweight="bold", alpha=0.9)
ax.text(n - 0.6, (ours_y + expert_y) / 2, f"−{expert_y - ours_y:.2f}\nto expert",
        ha="right", va="center", fontsize=9, color=F1_GOLD, alpha=0.85)

# --- the journey line + dots ---
ax.plot(xs, avgs, marker="o", linewidth=2.5, color="#777", zorder=3, alpha=0.6)
for i, (x, y, c) in enumerate(zip(xs, avgs, colors)):
    s = 380 if i == ours_idx else 220
    edge = F1_INK if i == ours_idx else F1_BG
    edge_w = 2.4 if i == ours_idx else 1.6
    ax.scatter([x], [y], s=s, color=c, edgecolor=edge, linewidth=edge_w, zorder=5)
    if i == ours_idx:
        # halo ring around the champion
        ax.scatter([x], [y], s=s + 320, color="none", edgecolor=F1_RED,
                   linewidth=1.4, alpha=0.55, zorder=4)
    label = f"{y:.3f}"
    offset = 14 if i != ours_idx else -28
    ax.annotate(label, (x, y), textcoords="offset points",
                xytext=(0, offset), ha="center",
                fontsize=11 if i == ours_idx else 10,
                fontweight="bold",
                color=c if i == ours_idx else F1_INK)

# explicit "shipping checkpoint" callout near the champion
ax.annotate("★ shipping checkpoint", xy=(ours_idx, ours_y),
            xytext=(ours_idx - 1.4, ours_y + 0.18),
            fontsize=10, color=F1_RED, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=F1_RED, lw=1.2, alpha=0.8))

ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=9, color=F1_INK)
ax.set_ylabel("Average weighted_final  (4–6 scenarios × 5 seeds)", fontsize=10, color=F1_DIM)
ax.set_title("F1 Strategist — training journey: random → expert    ★ = shipping checkpoint",
             fontsize=12, color=F1_INK, pad=12)
ax.set_ylim(0.2, 1.02)
ax.set_xlim(-0.5, n - 0.5)
leg = ax.legend(loc="lower right", fontsize=9, frameon=False, labelcolor=F1_INK)
_style_axes(ax, fig)
plt.tight_layout()
plt.savefig(RESULTS / "journey.png", dpi=140, facecolor=F1_BG, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {RESULTS / 'journey.png'}")


# ------------------------------------------------------------
# 2. Scenario-breakdown grouped bar chart
# ------------------------------------------------------------

groups = {
    "random": RANDOM,
    "untrained": UNTRAINED,
    "initial run\n(real LLM)": MAIN_ACTUAL,
    "SFT v3": SFT_V3,
    "RFT v1 T=0.3": RFT_V1_T03,
    "GRPO v2 ★": GRPO_V2,
    "expert": EXPERT,
}

fig, ax = plt.subplots(figsize=(13, 5.5))
n_groups = len(groups)
bar_w = 0.11
xs = np.arange(len(TASKS))
group_colors = [F1_DIM, F1_ORANGE, "#a36322", F1_TEAL, "#0a8d80", F1_RED, F1_GOLD]

for i, ((name, scores), color) in enumerate(zip(groups.items(), group_colors)):
    offset = (i - (n_groups - 1) / 2) * bar_w
    plot_scores = [0.0 if (isinstance(s, float) and math.isnan(s)) else s for s in scores]
    bars = ax.bar(xs + offset, plot_scores, bar_w, label=name, color=color,
                  edgecolor=F1_BG, linewidth=0.6)
    if name == "GRPO v2 ★":  # annotate champion
        for x, s in zip(xs + offset, plot_scores):
            if s > 0:
                ax.text(x, s + 0.018, f"{s:.2f}", ha="center", fontsize=8, fontweight="bold",
                        color=F1_RED)

ax.set_xticks(xs)
ax.set_xticklabels(TASK_LABELS, fontsize=10, color=F1_INK)
ax.set_ylabel("weighted_final score", fontsize=10, color=F1_DIM)
ax.set_title("Per-scenario performance across the journey",
             fontsize=12, color=F1_INK, pad=12)
ax.set_ylim(0, 1.05)
leg = ax.legend(loc="upper right", ncol=2, fontsize=9, frameon=False, labelcolor=F1_INK)
_style_axes(ax, fig)
plt.tight_layout()
plt.savefig(RESULTS / "scenario_breakdown.png", dpi=140, facecolor=F1_BG, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {RESULTS / 'scenario_breakdown.png'}")


# ------------------------------------------------------------
# 3. Comparison: main reported (scripted) vs main actual vs ours vs expert
# ------------------------------------------------------------

groups2 = {
    "main\n(reported,\nscripted fallback)": MAIN_REPORTED,
    "main\n(actual LLM)": MAIN_ACTUAL,
    "OURS\n(GRPO v2)": GRPO_V2,
    "expert ceiling": EXPERT,
}
colors2 = [F1_ORANGE, F1_DIM, F1_RED, F1_GOLD]

fig, ax = plt.subplots(figsize=(11, 5.5))
n = len(groups2)
bar_w = 0.18
xs = np.arange(len(TASKS))
for i, ((name, scores), color) in enumerate(zip(groups2.items(), colors2)):
    offset = (i - (n - 1) / 2) * bar_w
    plot_scores = [0.0 if (isinstance(s, float) and math.isnan(s)) else s for s in scores]
    bars = ax.bar(xs + offset, plot_scores, bar_w, label=name, color=color,
                  edgecolor=F1_BG, linewidth=0.6)
    for x, s in zip(xs + offset, plot_scores):
        if s > 0:
            ax.text(x, s + 0.012, f"{s:.2f}", ha="center", fontsize=8, color=F1_INK)

# Note about main's reported numbers
ax.text(
    0.02, 0.98,
    "the originally reported numbers came from a hand-coded scripted-policy\n"
    "fallback (LoRA dirs lack config.json → silent fall-through). verified\n"
    "bit-exact by running the scripted policy with no model loaded.",
    transform=ax.transAxes, fontsize=8.5, va="top", color=F1_INK,
    bbox=dict(boxstyle="round,pad=0.5", fc=F1_RULE, ec=F1_DIM),
)

ax.set_xticks(xs)
ax.set_xticklabels(TASK_LABELS, fontsize=10, color=F1_INK)
ax.set_ylabel("weighted_final score", fontsize=10, color=F1_DIM)
ax.set_title("Reality check: reported vs actual LLM behaviour",
             fontsize=12, color=F1_INK, pad=12)
ax.set_ylim(0, 1.05)
leg = ax.legend(loc="lower right", fontsize=9, frameon=False, labelcolor=F1_INK)
_style_axes(ax, fig)
plt.tight_layout()
plt.savefig(RESULTS / "comparison_real.png", dpi=140, facecolor=F1_BG, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {RESULTS / 'comparison_real.png'}")

# ------------------------------------------------------------
# Print a summary table for the blog/README
# ------------------------------------------------------------
print("\n--- avg per iteration ---")
for name, scores in iterations:
    avg = _avg(scores)
    print(f"  {name.replace(chr(10), ' '):<35}  avg={avg:.3f}")
