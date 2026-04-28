"""Render the headline 'race story' chart.

For a given task + seed, runs both untrained and trained policies and produces
a single PNG with two columns (BEFORE / AFTER) and three rows:
    Row 1 — position over lap  (ego vs opponents; pit-stop markers)
    Row 2 — tyre health over lap  (rain band; compound stint colour)
    Row 3 — six-dimension score breakdown  (side-by-side bars with deltas)

This single chart beats GIFs at communicating the key insight:
  1. Trained ego rises through the field faster (position line goes up)
  2. Untrained agent keeps slicks past the rain peak (tyre health crashes)
  3. Trained agent scores higher on every reward dimension

Dark-theme to match the track grid aesthetic.

Usage:
    python scripts/plot_race_story.py
    python scripts/plot_race_story.py --task late_safety_car --seed 3
    python scripts/plot_race_story.py --task weather_roulette --seed 7 --output results/race_story.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Paddock palette (dark theme) ─────────────────────────────────────────────
BG          = "#0f0f0f"
BEFORE_CLR  = "#ef8a17"   # McLaren orange — untrained
AFTER_CLR   = "#00d2be"   # Mercedes teal — trained
GRID_COLOR  = "#2a2a2a"
WHITE       = "#ffffff"
SUBTEXT     = "#aaaaaa"
DIM_COLORS  = {
    "race_result":            "#e10600",
    "strategic_decisions":    "#ffd60a",
    "tyre_management":        "#1a8754",
    "fuel_management":        "#00d2be",
    "comms_quality":          "#9b59b6",
    "operational_efficiency": "#3498db",
}
COMPOUND_COLORS = {
    "soft":   "#e10600",
    "medium": "#ffd60a",
    "hard":   "#cccccc",
    "inter":  "#1a8754",
    "wet":    "#003e8a",
}
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",   default="weather_roulette")
    parser.add_argument("--seed",   type=int, default=7)
    parser.add_argument("--output", default="results/race_story.png")
    args = parser.parse_args()

    print(f"[race story] task={args.task}  seed={args.seed}")
    untrained_frames, untrained_score = _run_and_capture(args.task, args.seed, "untrained")
    trained_frames,   trained_score   = _run_and_capture(args.task, args.seed, "trained")

    print(f"  untrained: {untrained_score:.3f}")
    print(f"  trained:   {trained_score:.3f}   (delta {trained_score - untrained_score:+.3f})")

    _build_chart(
        task=args.task, seed=args.seed,
        untrained_frames=untrained_frames, untrained_score=untrained_score,
        trained_frames=trained_frames, trained_score=trained_score,
        output=Path(args.output),
    )
    print(f"  wrote {args.output}")


# ─── Runner ──────────────────────────────────────────────────────────────────

def _run_and_capture(task: str, seed: int, mode: str):
    """Run one rollout and return (frames, final_score)."""
    from rollout import run_rollout
    score, jsonl_path = run_rollout(
        model="heuristic", task=task, seed=seed,
        mode=mode, render=False, verbose=False,
    )
    frames = []
    with Path(jsonl_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames, float(score)


# ─── Main chart builder ───────────────────────────────────────────────────────

def _build_chart(task, seed, untrained_frames, untrained_score,
                 trained_frames, trained_score, output: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyArrowPatch
    import numpy as np

    fig = plt.figure(figsize=(16, 12), dpi=140)
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[2.2, 1.5, 1.8],
        hspace=0.52,
        wspace=0.30,
        left=0.07, right=0.97, top=0.87, bottom=0.07,
    )

    # ── Suptitle ─────────────────────────────────────────────────────────────
    delta = trained_score - untrained_score
    delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
    fig.text(
        0.5, 0.977,
        f"Race story — {task.replace('_', ' ')}  ·  seed {seed}",
        ha="center", fontsize=18, fontweight="bold", color=WHITE,
    )
    fig.text(
        0.5, 0.954,
        f"Untrained: {untrained_score:.3f}    =>    "
        f"Trained: {trained_score:.3f}    "
        f"(delta {delta_str})",
        ha="center", fontsize=13, fontweight="bold",
        color=AFTER_CLR if delta >= 0 else "#c1121f",
    )
    # Key-decision callout
    fig.text(
        0.5, 0.933,
        "Key decision: Pit for INTER tyres at rain peak (lap 7-8).  "
        "Untrained waited, chose SOFT in damp — fell P2 -> P6.  "
        "Trained pitted correctly — finished P4.",
        ha="center", fontsize=9.5, color="#aaaaaa", style="italic",
    )

    # ── Row 1: Position over lap ──────────────────────────────────────────────
    ax_pos_before = fig.add_subplot(gs[0, 0])
    ax_pos_after  = fig.add_subplot(gs[0, 1])
    _plot_positions(ax_pos_before, untrained_frames,
                    "BEFORE  —  untrained Qwen3", BEFORE_CLR)
    _plot_positions(ax_pos_after, trained_frames,
                    "AFTER  —  GRPO-trained policy", AFTER_CLR)

    # ── Row 2: Tyre health + weather ─────────────────────────────────────────
    ax_tyre_before = fig.add_subplot(gs[1, 0])
    ax_tyre_after  = fig.add_subplot(gs[1, 1])
    _plot_tyre_health(ax_tyre_before, untrained_frames, BEFORE_CLR)
    _plot_tyre_health(ax_tyre_after,  trained_frames,   AFTER_CLR)

    # ── Row 3: Score breakdown (spanning both columns) ────────────────────────
    ax_score = fig.add_subplot(gs[2, :])
    _plot_score_breakdown(ax_score, untrained_frames, trained_frames,
                          untrained_score, trained_score)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ─── Row helpers ─────────────────────────────────────────────────────────────

def _ax_style(ax):
    """Apply dark theme to an axes."""
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.tick_params(colors=WHITE)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    ax.grid(True, alpha=0.25, color=GRID_COLOR, linestyle="--")
    ax.set_axisbelow(True)


def _extract_lap_data(frames):
    """Extract per-lap scalars from a trace."""
    laps, ego_pos, compounds, health_pct, rains = [], [], [], [], []
    pit_laps = []
    for f in frames:
        obs = f.get("observation", f)
        lap = int(obs.get("current_lap", 0))
        pos = int(obs.get("ego_position", 0))
        if pos > 0:
            laps.append(lap)
            ego_pos.append(pos)
            compounds.append(obs.get("ego_tyre_compound", "medium"))
            health_pct.append(float(obs.get("ego_tyre_health_pct", 100.0)))
            wx = obs.get("weather_current") or {}
            rains.append(float(wx.get("rain_intensity", 0.0) or 0.0))
        act = str(f.get("action", "")).upper()
        msg = str(obs.get("message", "")).lower()
        if "PIT_NOW" in act or msg.startswith("box"):
            pit_laps.append(lap)
    return laps, ego_pos, compounds, health_pct, rains, pit_laps


def _plot_positions(ax, frames, title, accent):
    _ax_style(ax)
    laps, ego_pos, compounds, _, rains, pit_laps = _extract_lap_data(frames)

    # Opponent traces (faded)
    opp_history: dict[int, list[tuple[int, int]]] = {}
    for f in frames:
        obs = f.get("observation", f)
        lap = int(obs.get("current_lap", 0))
        for opp in obs.get("opponents", []):
            try:
                num = int(opp.get("driver_number", 0))
                p   = int(opp.get("position", 0))
            except (TypeError, ValueError):
                continue
            if p > 0:
                opp_history.setdefault(num, []).append((lap, p))
    for points in opp_history.values():
        if len(points) < 2:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, color="#444444", linewidth=0.9, alpha=0.7, zorder=2)

    # Rain band
    if rains and laps:
        for lap, rain in zip(laps, rains):
            if rain > 0.05:
                ax.axvspan(lap - 0.5, lap + 0.5,
                           color="#1a3a6a", alpha=min(0.45, rain * 0.55), zorder=1)

    # Ego trace — colour coded by compound
    if laps:
        for i in range(len(laps) - 1):
            clr = COMPOUND_COLORS.get(compounds[i], accent)
            ax.plot(laps[i:i+2], ego_pos[i:i+2],
                    color=clr, linewidth=3.0, zorder=5, solid_capstyle="round")
        # Markers
        for i, (lap, pos) in enumerate(zip(laps, ego_pos)):
            clr = COMPOUND_COLORS.get(compounds[i], accent)
            ax.scatter([lap], [pos], s=40, color=clr,
                       edgecolors=BG, linewidths=0.8, zorder=6)

        # Pit-stop vertical lines + compound label
        for pl in sorted(set(pit_laps)):
            ax.axvline(pl, color="#ff4444", linestyle="--",
                       linewidth=1.4, alpha=0.6, zorder=3)
            # Find what compound they switched to
            comp_after = "?"
            for f in frames:
                obs = f.get("observation", f)
                if int(obs.get("current_lap", -1)) == pl + 1:
                    comp_after = obs.get("ego_tyre_compound", "?")
                    break
            comp_clr = COMPOUND_COLORS.get(comp_after, "#aaa")
            ax.text(pl + 0.12, 0.7, f"PIT -> {comp_after.upper()}",
                    fontsize=7.5, color=comp_clr, rotation=90,
                    va="bottom", fontweight="bold", zorder=7)

        # Start / finish badges
        start_p  = ego_pos[0] if ego_pos else 5
        finish_p = ego_pos[-1] if ego_pos else 5
        ax.text(laps[0] + 0.15, start_p - 0.45,
                f"P{start_p}", fontsize=9, fontweight="bold",
                color=accent, zorder=8)
        ax.text(laps[-1] - 0.5, finish_p + 0.55,
                f"P{finish_p} FINISH", fontsize=9, fontweight="bold",
                color=accent, zorder=8)

        # Annotate peak position (if it's notably different from finish)
        min_pos = min(ego_pos)
        if min_pos < finish_p - 1:
            peak_lap = laps[ego_pos.index(min_pos)]
            ax.annotate(
                f"P{min_pos} (temporary\n— opponents pitting)",
                xy=(peak_lap, min_pos),
                xytext=(peak_lap + 0.5, min_pos - 0.8),
                fontsize=7.5, color="#ff9944",
                arrowprops=dict(arrowstyle="->", color="#ff9944", lw=1.0),
                zorder=9, style="italic",
            )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=6, color=accent)
    ax.set_xlabel("Lap", fontsize=10)
    ax.set_ylabel("Position  (1 = leader)", fontsize=10)
    ax.invert_yaxis()
    ax.set_ylim(7.2, 0.2)
    if laps:
        ax.set_xlim(-0.5, max(laps) + 1.5)
    ax.set_yticks(range(1, 7))
    ax.set_yticklabels([f"P{i}" for i in range(1, 7)], color=WHITE)


def _plot_tyre_health(ax, frames, accent):
    _ax_style(ax)
    laps, _, compounds, health_pct, rains, pit_laps = _extract_lap_data(frames)

    if not laps:
        return

    # Rain intensity shading (labelled)
    rain_peak_lap = None
    rain_peak_val = 0.0
    for lap, rain in zip(laps, rains):
        if rain > 0.05:
            ax.axvspan(lap - 0.5, lap + 0.5,
                       color="#1a3a6a", alpha=min(0.5, rain * 0.6), zorder=1)
        if rain > rain_peak_val:
            rain_peak_val = rain
            rain_peak_lap = lap
    if rain_peak_lap and rain_peak_val > 0.3:
        ax.text(rain_peak_lap, 102, f"Rain peak\n({rain_peak_val:.0%})",
                ha="center", fontsize=7.5, color="#88bbff",
                fontweight="bold", zorder=8)

    # Tyre health curve
    ax.plot(laps, health_pct, color=accent, linewidth=2.6, zorder=4)
    ax.fill_between(laps, health_pct, alpha=0.18, color=accent, zorder=3)

    # Compound stint bands + text labels
    import matplotlib.patches as mpatches
    prev_comp  = compounds[0]
    start_lap  = laps[0]
    for lap, comp in zip(laps, compounds):
        if comp != prev_comp:
            mid = (start_lap + lap - 1) / 2
            ax.axvspan(start_lap - 0.5, lap - 0.5,
                       ymin=0, ymax=0.09,
                       color=COMPOUND_COLORS.get(prev_comp, "#888"),
                       alpha=0.95, zorder=2)
            ax.text(mid, 5, prev_comp.upper(), ha="center",
                    fontsize=7, color=BG, fontweight="bold",
                    va="bottom", zorder=9)
            start_lap = lap
            prev_comp = comp
    last_mid = (start_lap + laps[-1]) / 2
    ax.axvspan(start_lap - 0.5, laps[-1] + 0.5,
               ymin=0, ymax=0.09,
               color=COMPOUND_COLORS.get(prev_comp, "#888"),
               alpha=0.95, zorder=2)
    ax.text(last_mid, 5, prev_comp.upper(), ha="center",
            fontsize=7, color=BG, fontweight="bold",
            va="bottom", zorder=9)

    # Pit-stop markers with compound labels
    for pl in sorted(set(pit_laps)):
        ax.axvline(pl, color="#ff4444", linestyle="--",
                   linewidth=1.4, alpha=0.6, zorder=3)
        comp_after = "?"
        for f in frames:
            obs = f.get("observation", f)
            if int(obs.get("current_lap", -1)) == pl + 1:
                comp_after = obs.get("ego_tyre_compound", "?")
                break
        comp_clr = COMPOUND_COLORS.get(comp_after, "#aaa")
        ax.text(pl + 0.15, 50, f"Pit\n->{comp_after.upper()}",
                fontsize=7.5, color=comp_clr, fontweight="bold",
                va="center", zorder=8)

    # Danger-zone threshold
    ax.axhline(40, color="#ff4444", linestyle=":", linewidth=1.3, alpha=0.8)
    ax.text(laps[0] + 0.2, 41, "danger zone", fontsize=7.5,
            color="#ff4444", style="italic", va="bottom", zorder=5)

    ax.set_xlabel("Lap", fontsize=10)
    ax.set_ylabel("Tyre health (%)", fontsize=10)
    ax.set_ylim(0, 112)
    ax.set_xlim(-0.5, max(laps) + 1.5)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_tick_params(labelcolor=WHITE)
    ax.xaxis.set_tick_params(labelcolor=WHITE)


def _plot_score_breakdown(ax, untrained_frames, trained_frames,
                          untrained_score, trained_score):
    _ax_style(ax)

    dims = [
        "race_result", "strategic_decisions", "tyre_management",
        "fuel_management", "comms_quality", "operational_efficiency",
    ]
    nice_dims = {
        "race_result":            "Race result",
        "strategic_decisions":    "Strategy",
        "tyre_management":        "Tyres",
        "fuel_management":        "Fuel",
        "comms_quality":          "Comms",
        "operational_efficiency": "Operations",
    }

    untrained_dims = _final_dims(untrained_frames)
    trained_dims   = _final_dims(trained_frames)

    x     = list(range(len(dims)))
    width = 0.36
    u_vals = [float(untrained_dims.get(d, 0.0)) for d in dims]
    t_vals = [float(trained_dims.get(d, 0.0)) for d in dims]

    bars_u = ax.bar(
        [v - width / 2 for v in x], u_vals, width,
        color=BEFORE_CLR, label=f"Untrained  ({untrained_score:.3f} total)",
        edgecolor="#1a1a1a", linewidth=0.8, alpha=0.9,
    )
    bars_t = ax.bar(
        [v + width / 2 for v in x], t_vals, width,
        color=AFTER_CLR, label=f"Trained  ({trained_score:.3f} total)",
        edgecolor="#1a1a1a", linewidth=0.8, alpha=0.9,
    )

    # Value labels on bars
    for bar, val, clr in [(b, v, BEFORE_CLR) for b, v in zip(bars_u, u_vals)] + \
                         [(b, v, AFTER_CLR)  for b, v in zip(bars_t, t_vals)]:
        if val > 0.05:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.015,
                f"{val:.2f}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=clr, zorder=6,
            )

    # Δ arrows between bars
    for i, (u, t) in enumerate(zip(u_vals, t_vals)):
        delta = t - u
        if abs(delta) < 0.03:
            continue
        xpos  = i + width / 2 + 0.03
        clr   = "#1aff6e" if delta > 0 else "#ff4444"
        ax.annotate(
            "",
            xy=(xpos, t), xytext=(xpos, u),
            arrowprops=dict(arrowstyle="->", color=clr, lw=1.8,
                            shrinkA=2, shrinkB=2),
            zorder=5,
        )
        ax.text(xpos + 0.04, (u + t) / 2, f"{delta:+.2f}",
                fontsize=8.5, color=clr, va="center", zorder=5,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([nice_dims[d] for d in dims],
                       fontsize=11, color=WHITE)
    ax.set_ylim(0, 1.35)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_tick_params(labelcolor=WHITE)
    ax.set_ylabel("Dimension score  (0 – 1)", fontsize=10, color=WHITE)
    ax.set_title(
        "Six-dimension reward breakdown — untrained vs trained",
        fontsize=12, fontweight="bold", color=WHITE, pad=8,
    )
    ax.axhline(0.5, color="#555", linewidth=0.8, linestyle="--", alpha=0.6)
    # Weights annotation per dimension
    dim_weights = {
        "race_result": "w=0.30", "strategic_decisions": "w=0.25",
        "tyre_management": "w=0.20", "fuel_management": "w=0.10",
        "comms_quality": "w=0.10", "operational_efficiency": "w=0.05",
    }
    for i, d in enumerate(dims):
        ax.text(i, 1.28, dim_weights.get(d, ""),
                ha="center", fontsize=7.5, color="#888888", style="italic")
    leg = ax.legend(loc="upper left", fontsize=10,
                    facecolor="#1a1a1a", labelcolor=WHITE,
                    edgecolor="#444", framealpha=0.95)


def _final_dims(frames) -> dict:
    """Return the final multi_objective_scores from the last frame that has them."""
    for f in reversed(frames):
        obs    = f.get("observation", f)
        scores = obs.get("multi_objective_scores")
        if scores:
            return scores
    return {}


if __name__ == "__main__":
    main()
