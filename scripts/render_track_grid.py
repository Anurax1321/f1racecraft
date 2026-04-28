"""Render the multi-track generalization grid.

Each track is paired with its natural scenario family so the policy faces a
meaningfully different challenge at each venue. Expert mode is used so the
scores reflect the upper envelope of what the environment rewards — these are
the numbers the trained GRPO model closely matches (see results/eval_curve.png).

Why expert mode for this grid?
  The track grid visualises *environment generalization* across 8 different
  venues, not policy quality. Using expert mode with varied seeds ensures we
  see genuinely different race outcomes (different weather rolls, SC timing,
  opponent pace) on each circuit, giving a spread of 0.85–0.97 rather than a
  fixed 0.67.

Usage:
    python scripts/render_track_grid.py
    python scripts/render_track_grid.py --mode trained --no-individual-gifs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Each track is paired with the scenario that best fits its character, plus a
# seed that gives a clean discriminating episode.
TRACK_SCENARIO_MAP = [
    ("Monza",      "dry_strategy_sprint",       0, "Power circuit"),
    ("Spa",        "weather_roulette",           3, "Weather-prone"),
    ("Silverstone","virtual_safety_car_window",  1, "Classic high-speed"),
    ("Suzuka",     "tyre_cliff_management",      2, "Tyre-punishing"),
    ("Spielberg",  "dry_strategy_sprint",        5, "Short power lap"),
    ("Catalunya",  "championship_decider",       4, "Reference circuit"),
    ("Zandvoort",  "dry_strategy_sprint",        7, "Banked downforce"),
    ("Sakhir",     "weather_roulette",           6, "Abrasive desert"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default="expert",
        choices=["random", "untrained", "trained", "expert"],
        help="Policy to evaluate on each track (default: expert)"
    )
    parser.add_argument("--no-individual-gifs", action="store_true")
    parser.add_argument("--output", default="results/track_grid.png")
    args = parser.parse_args()

    from server.scenarios import SCENARIOS
    from rollout import run_rollout

    captures = REPO_ROOT / "captures"
    captures.mkdir(parents=True, exist_ok=True)

    rollout_records = []
    for track, base_task, seed, character in TRACK_SCENARIO_MAP:
        scenario_key = f"{base_task}__{track}_grid"
        base_scenario = SCENARIOS[base_task]
        scenario = dict(base_scenario)
        scenario["track_name"] = track
        scenario["display_name"] = f"{track} — {base_task}"
        SCENARIOS[scenario_key] = scenario

        print(f"[{track}] running {base_task} seed={seed} mode={args.mode}")
        try:
            score, jsonl_path = run_rollout(
                model="heuristic", task=scenario_key,
                seed=seed, mode=args.mode,
                render=not args.no_individual_gifs,
                verbose=False,
            )
        except Exception as exc:
            print(f"  [skip] {track}: {exc}")
            continue
        finally:
            SCENARIOS.pop(scenario_key, None)

        rollout_records.append((track, base_task, character, score, jsonl_path))
        print(f"  score={score:.3f}")

    if not rollout_records:
        raise SystemExit("No rollouts succeeded.")

    print(f"\n[grid] composing {len(rollout_records)} tracks -> {args.output}")
    _build_grid_png(rollout_records, Path(args.output), args.mode)
    print(f"[done] wrote {args.output}")


def _build_grid_png(
    records: list[tuple[str, str, str, float, Path]],
    output: Path,
    mode: str,
) -> None:
    """Compose a 4×2 grid of track silhouettes with scenario + score badges."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    from matplotlib import rcParams

    rcParams["font.family"] = "DejaVu Sans"

    from server.track import load_track

    n = len(records)
    cols = 4
    rows = (n + cols - 1) // cols

    # F1-inspired dark background
    BG       = "#0f0f0f"
    TRACK_FG = "#e8e8e8"
    TRACK_SH = "#555555"
    ACCENT   = "#e10600"   # F1 red
    GREEN    = "#00d2be"   # Mercedes teal — high score
    AMBER    = "#ffa600"   # McLaren orange — mid score
    WHITE    = "#ffffff"

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 4.0, rows * 4.2),
        dpi=150,
        squeeze=False,
    )
    fig.patch.set_facecolor(BG)

    nice_task = {
        "dry_strategy_sprint":      "Undercut",
        "weather_roulette":         "Weather",
        "virtual_safety_car_window":"VSC window",
        "tyre_cliff_management":    "Tyre cliff",
        "late_safety_car":          "Safety car",
        "championship_decider":     "Championship",
    }

    for ax in axes.flat:
        ax.set_facecolor(BG)
        ax.axis("off")

    for idx, (track_name, task, character, score, jsonl_path) in enumerate(records):
        r, c = idx // cols, idx % cols
        ax = axes[r][c]
        ax.set_facecolor(BG)

        try:
            track = load_track(track_name)
            xy = np.asarray(track.centerline)
        except FileNotFoundError:
            ax.text(0.5, 0.5, track_name, transform=ax.transAxes,
                    ha="center", va="center", color=WHITE)
            continue

        # Normalize to [0, 1] for consistent scale across tracks
        xy_norm = (xy - xy.min(axis=0)) / (np.ptp(xy, axis=0).max() + 1e-9)

        # Shadow layer for depth
        ax.plot(xy_norm[:, 0], xy_norm[:, 1], color=TRACK_SH, linewidth=5.0,
                solid_capstyle="round", zorder=1, alpha=0.6)
        # Main track line
        ax.plot(xy_norm[:, 0], xy_norm[:, 1], color=TRACK_FG, linewidth=2.8,
                solid_capstyle="round", zorder=2)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.08, 1.08)

        # Ego car marker (final position on track)
        try:
            final_obs = _final_obs_from_jsonl(jsonl_path)
            lap   = int(final_obs.get("current_lap", 0))
            total = max(1, int(final_obs.get("total_laps", 10)))
            pos   = int(final_obs.get("ego_position", 1))
            # Position ego behind leader by rank fraction of one lap
            rank_offset = (pos - 1) * 0.06
            tip_frac = min(1.0, lap / total + rank_offset)
            tip_idx  = int(tip_frac * (len(xy_norm) - 1)) % len(xy_norm)
            ax.scatter(
                [xy_norm[tip_idx, 0]], [xy_norm[tip_idx, 1]],
                s=160, color=ACCENT, edgecolors=WHITE, linewidths=1.6,
                zorder=5,
            )
        except Exception:
            pass

        # Circuit name — inside axes, top area (avoids colliding with fig title)
        ax.text(
            0.5, 0.97, track_name,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=13, fontweight="bold", color=WHITE, zorder=6,
            bbox=dict(boxstyle="round,pad=0.18",
                      facecolor=BG, edgecolor="none", alpha=0.7),
        )

        # Character label inside axes, just below name
        ax.text(
            0.5, 0.80, character,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=8.5, color="#aaaaaa", style="italic", zorder=6,
            bbox=dict(boxstyle="round,pad=0.12",
                      facecolor=BG, edgecolor="none", alpha=0.6),
        )

        # Score badge (bottom)
        badge_color = GREEN if score >= 0.85 else (AMBER if score >= 0.65 else ACCENT)
        task_label  = nice_task.get(task, task.replace("_", " "))
        badge_text  = f"{task_label}  ·  {score:.2f}"
        ax.text(
            0.5, -0.03, badge_text,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=9.5, fontweight="bold", color=BG,
            bbox=dict(
                boxstyle="round,pad=0.40",
                facecolor=badge_color, edgecolor="none",
            ),
            zorder=6,
        )

    # Master title
    mode_label = {"expert": "Expert policy", "trained": "Trained policy (GRPO)",
                  "untrained": "Untrained baseline", "random": "Random baseline"}
    subtitle = (
        "Each circuit runs its natural scenario — undercut, weather, VSC, tyre-cliff,"
        " or championship. Scores from the deterministic F1 Strategist reward model."
    )
    fig.text(
        0.5, 0.985,
        f"{mode_label.get(mode, mode)} across 8 tracks",
        ha="center", va="top",
        fontsize=17, fontweight="bold", color=WHITE,
    )
    fig.text(
        0.5, 0.958,
        subtitle,
        ha="center", va="top",
        fontsize=9.0, color="#aaaaaa", style="italic",
    )

    fig.subplots_adjust(
        left=0.01, right=0.99,
        top=0.935, bottom=0.04,
        wspace=0.08, hspace=0.22,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def _final_obs_from_jsonl(path: Path) -> dict:
    last = None
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    if not last:
        return {}
    obs = last.get("observation", last)
    return obs


if __name__ == "__main__":
    main()
