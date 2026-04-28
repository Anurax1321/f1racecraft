"""Plot the GRPO training reward curve with context.

Key design: the chart shows not just the reward trajectory, but WHERE it sits
relative to meaningful baselines — random, untrained LLM, and expert heuristic.
Without these reference lines, a reward of 0.875 → 0.907 looks like a tiny
improvement; with them, you can see the model trained from SFT warm-start is
already at expert level and GRPO is refining the final few percent.

The chart tells the full story in one image:
  - SFT warm-start → model starts GRPO at 0.875 (just below expert ceiling)
  - GRPO 500 steps → peaks at 0.907 @ step 480
  - Expert ceiling: ~0.90 (average across 4 held-out families)
  - Untrained LLM: ~0.40 (average across 4 held-out families)
  - Random agent: ~0.30 (average across 4 held-out families)
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── F1-paddock colour palette ────────────────────────────────────────────────
BG             = "#0f0f0f"
REWARD_RAW     = "#5a9fd4"    # pale blue raw trace
REWARD_SMOOTH  = "#00d2be"    # Mercedes teal — smoothed
REWARD_BAND    = "#003e3a"    # dark teal band
LOSS_COLOR     = "#e10600"    # Ferrari red
EXPERT_LINE    = "#ffd60a"    # gold — expert ceiling
TRAINED_LINE   = "#1a8754"    # green — "trained" level from eval
UNTRAINED_LINE = "#ef8a17"    # orange — untrained level from eval
RANDOM_LINE    = "#9aa0a6"    # grey — random level from eval
WHITE          = "#ffffff"
GRID_COLOR     = "#2a2a2a"
# ────────────────────────────────────────────────────────────────────────────

# Reference baselines derived from results/eval_summary.json
# (averaged across all 4 held-out families)
# random: 0.30  |  untrained: 0.39  |  expert: 0.90
BASELINES = {
    "Random agent":            0.296,
    "Untrained Qwen3":         0.395,
    "Expert heuristic (ceiling)": 0.905,
}
BASELINE_COLORS = {
    "Random agent":            RANDOM_LINE,
    "Untrained Qwen3":         UNTRAINED_LINE,
    "Expert heuristic (ceiling)": EXPERT_LINE,
}
BASELINE_STYLES = {
    "Random agent":               (1.2, "--"),
    "Untrained Qwen3":            (1.4, "-."),
    "Expert heuristic (ceiling)": (1.6, ":"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="./grpo_v1")
    parser.add_argument("--output", default="results/training_loss_curve.png")
    parser.add_argument("--smooth-window", type=int, default=7)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    state_path = _find_state(Path(args.run_dir))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    rows = [r for r in state.get("log_history", []) if "step" in r]
    if not rows:
        raise SystemExit(f"No plottable log_history rows in {state_path}")

    steps   = [r["step"] for r in rows]
    losses  = [r.get("loss") for r in rows]
    rewards = [r.get("reward", r.get("rewards/mean")) for r in rows]
    rewards_clean    = [r for r in rewards if r is not None]
    rewards_smoothed = _rolling_mean(rewards, args.smooth_window)

    # ── Layout: dark theme ───────────────────────────────────────────────────
    fig, ax_r = plt.subplots(figsize=(12, 6), dpi=160)
    fig.patch.set_facecolor(BG)
    ax_r.set_facecolor(BG)
    for spine in ax_r.spines.values():
        spine.set_edgecolor("#333333")

    # ── Reference baseline band annotations ─────────────────────────────────
    # Shade region between random and expert to give a sense of possible range
    ax_r.axhspan(BASELINES["Random agent"] - 0.01,
                 BASELINES["Expert heuristic (ceiling)"] + 0.01,
                 alpha=0.07, color="#ffffff", zorder=0)

    for label, val in BASELINES.items():
        lw, style = BASELINE_STYLES[label]
        color = BASELINE_COLORS[label]
        ax_r.axhline(val, linestyle=style, color=color, linewidth=lw,
                     alpha=0.9, zorder=2)
        # Right-side label
        ax_r.text(
            steps[-1] + steps[-1] * 0.01, val,
            f"  {label}",
            color=color, fontsize=8.0, va="center",
            fontweight="bold",
        )

    # ── Reward ±σ band ───────────────────────────────────────────────────────
    if len(rewards_clean) >= 3:
        sd = statistics.pstdev(rewards_clean)
        upper = [v + sd if v is not None else None for v in rewards_smoothed]
        lower = [v - sd if v is not None else None for v in rewards_smoothed]
        valid = [(s, lo, hi) for s, lo, hi in zip(steps, lower, upper)
                 if lo is not None and hi is not None]
        if valid:
            xs, ls, us = zip(*valid)
            ax_r.fill_between(xs, ls, us, color=REWARD_BAND, alpha=0.8, zorder=3)

    # ── Raw reward trace (faint) ─────────────────────────────────────────────
    if any(v is not None for v in rewards):
        ax_r.plot(
            steps,
            [v if v is not None else float("nan") for v in rewards],
            color=REWARD_RAW, linewidth=1.0, alpha=0.5,
            zorder=4, label="Reward (raw)",
        )

    # ── Smoothed reward (bold) ───────────────────────────────────────────────
    if any(v is not None for v in rewards_smoothed):
        ax_r.plot(
            steps,
            [v if v is not None else float("nan") for v in rewards_smoothed],
            color=REWARD_SMOOTH, linewidth=2.8, zorder=5,
            label=f"Reward (smoothed, window={args.smooth_window})",
        )

    # ── Peak marker ─────────────────────────────────────────────────────────
    if rewards_clean:
        peak_reward = max(rewards_clean)
        peak_idx    = rewards.index(peak_reward) if peak_reward in rewards else 0
        peak_step   = steps[peak_idx]
        ax_r.scatter(
            [peak_step], [peak_reward],
            color=EXPERT_LINE, edgecolor=WHITE, s=200, zorder=8,
            marker="*", label=f"Peak {peak_reward:.3f} @ step {peak_step}",
        )
        ax_r.annotate(
            f" peak {peak_reward:.3f}",
            xy=(peak_step, peak_reward),
            xytext=(peak_step - len(steps) * 4, peak_reward + 0.006),
            fontsize=9, color=EXPERT_LINE, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=EXPERT_LINE,
                            linewidth=0.8, alpha=0.7),
        )

    # ── Axes + labels ────────────────────────────────────────────────────────
    ax_r.set_xlabel("Training step", fontsize=11, color=WHITE)
    ax_r.set_ylabel("Mean episode reward", fontsize=11,
                    color=REWARD_SMOOTH, fontweight="bold")
    ax_r.tick_params(colors=WHITE, which="both")
    ax_r.xaxis.label.set_color(WHITE)
    ax_r.grid(True, alpha=0.25, color=GRID_COLOR, linestyle="--")
    ax_r.set_axisbelow(True)

    if rewards_clean:
        rmin, rmax = min(rewards_clean), max(rewards_clean)
        pad = max(0.03, (rmax - rmin) * 0.6)
        # Extend ylim to show all baselines
        ylo = min(BASELINES["Random agent"] - 0.04, rmin - pad)
        yhi = max(BASELINES["Expert heuristic (ceiling)"] + 0.04, rmax + pad)
        ax_r.set_ylim(ylo, yhi)

    # ── Loss on right axis (log scale) ──────────────────────────────────────
    ax_l = ax_r.twinx()
    ax_l.set_facecolor(BG)
    if any(v is not None and v > 0 for v in losses):
        pos_losses = [v if (v is not None and v > 1e-9) else float("nan")
                      for v in losses]
        ax_l.plot(steps, pos_losses, color=LOSS_COLOR, linewidth=1.4,
                  alpha=0.75, zorder=3, label="Loss (log scale)")
        ax_l.set_yscale("log")
        ax_l.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0e"))
    ax_l.set_ylabel("Loss (log scale)", color=LOSS_COLOR, fontsize=10)
    ax_l.tick_params(axis="y", labelcolor=LOSS_COLOR, colors=LOSS_COLOR)
    for spine in ax_l.spines.values():
        spine.set_edgecolor("#333333")

    # ── Headline annotation box ──────────────────────────────────────────────
    if rewards_clean:
        early_n    = max(1, len(rewards_clean) // 10)
        late_n     = max(1, len(rewards_clean) // 10)
        early_r    = statistics.mean(rewards_clean[:early_n])
        late_r     = statistics.mean(rewards_clean[-late_n:])
        delta      = late_r - early_r
        headline = (
            f"500-step GRPO  ·  Qwen3-4B + LoRA  ·  RTX 5090\n"
            f"SFT warm-start → GRPO start {early_r:.3f} → late {late_r:.3f}  "
            f"(Δ {delta:+.3f})  ·  peak {peak_reward:.3f} @ step {peak_step}"
        )
        ax_r.text(
            0.02, 0.04, headline,
            transform=ax_r.transAxes,
            fontsize=9.5, color=WHITE,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.55",
                      facecolor="#1a1a1a", edgecolor=REWARD_SMOOTH,
                      linewidth=1.5, alpha=0.96),
            zorder=10,
        )

    # ── Title ────────────────────────────────────────────────────────────────
    title = args.title or f"GRPO training — {Path(args.run_dir).name}"
    ax_r.set_title(title, fontsize=14, fontweight="bold",
                   color=WHITE, pad=14)

    # ── Legend ───────────────────────────────────────────────────────────────
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    lines_l, labels_l = ax_l.get_legend_handles_labels()
    # Add manual baseline legend entries
    baseline_handles = [
        Line2D([0], [0], color=BASELINE_COLORS[k], linewidth=1.4,
               linestyle=BASELINE_STYLES[k][1], label=k)
        for k in BASELINES
    ]
    all_handles = lines_r + baseline_handles + lines_l
    all_labels  = labels_r + list(BASELINES.keys()) + labels_l
    leg = fig.legend(
        all_handles, all_labels,
        loc="lower right", bbox_to_anchor=(0.97, 0.07),
        fontsize=8.5, framealpha=0.92,
        facecolor="#1a1a1a", labelcolor=WHITE, edgecolor="#444",
    )

    fig.tight_layout(rect=[0, 0, 0.86, 1])   # leave room for right-side labels
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"wrote {output} from {state_path}")


def _rolling_mean(values: list, window: int) -> list:
    n = len(values)
    out = [None] * n
    half = window // 2
    for i in range(n):
        start = max(0, i - half)
        stop  = min(n, i + half + 1)
        chunk = [v for v in values[start:stop] if v is not None]
        if chunk:
            out[i] = sum(chunk) / len(chunk)
    return out


def _find_state(run_dir: Path) -> Path:
    direct = run_dir / "trainer_state.json"
    if direct.exists():
        return direct
    candidates = sorted(run_dir.glob("checkpoint-*/trainer_state.json"))
    if candidates:
        return candidates[-1]
    raise SystemExit(f"No trainer_state.json found under {run_dir}")


if __name__ == "__main__":
    main()
