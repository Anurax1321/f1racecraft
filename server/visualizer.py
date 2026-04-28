"""Matplotlib rollout renderer + Gradio control panel.

Renders rollout JSONL traces into GIFs/MP4s with a paddock-style HUD plus a
**race-position ladder** that is the real visual difference-maker between
untrained and trained runs.

Layout (left → right):
  - track polygon with ego AND opponents drawn at their grid positions
    (stacked along the centerline by current rank rather than by lap)
  - HUD panel (lap, position, tyre, fuel, weather)
  - position ladder panel (all 6 cars stacked by current rank, ego highlighted)

Bottom strip: per-lap weather heat-bar, pit + inspection markers, lap line.
Bottom callout: latest 3 strategist actions in monospace.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Tyre compound → colour
COMPOUND_COLORS = {
    "soft": "#e10600", "medium": "#ffd60a", "hard": "#ffffff",
    "inter": "#1a8754", "wet": "#003e8a",
}
COMPOUND_EDGE = {
    "soft": "#7a0000", "medium": "#7a6600", "hard": "#444444",
    "inter": "#0d4326", "wet": "#001a40",
}
DEFAULT_COMPOUND_COLOR = "#888888"

STATUS_COLORS = {
    "green": "#1a8754", "yellow": "#ffd60a", "sc": "#ff8c00",
    "vsc": "#ff8c00", "red": "#c1121f", "start": "#003e8a",
    "finish": "#1a1a1a",
}

EGO_COLOR = "#e10600"      # ego car always Ferrari red
OPP_COLOR = "#1f77b4"      # opponents blue


def render_rollout(rollout_jsonl_path, output_path, fmt: str = "gif",
                   title_override: str | None = None) -> None:
    """Animate a rollout JSONL into a GIF or MP4 with the full HUD + position ladder."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import numpy as np

    from server.track import load_track

    rollout_jsonl_path = Path(rollout_jsonl_path)
    output_path = Path(output_path)
    frames = _read_rollout_frames(rollout_jsonl_path)
    if not frames:
        raise ValueError(f"No frames found in {rollout_jsonl_path}")

    track_name = frames[0].get("track_name", "Monza")
    try:
        track = load_track(track_name)
    except FileNotFoundError:
        track = load_track("Monza")
    xy = np.asarray(track.centerline)
    total_laps = max(1, _max_laps(frames))

    # Layout: track | HUD | ladder; below: lap strip; below: action callout
    fig = plt.figure(figsize=(15, 7.5), dpi=110)
    fig.patch.set_facecolor("#fafafa")
    gs = fig.add_gridspec(
        3, 3,
        width_ratios=[3.4, 1.2, 1.4],
        height_ratios=[5, 1, 0.6],
        hspace=0.18, wspace=0.10,
    )
    ax_track = fig.add_subplot(gs[0, 0])
    ax_hud = fig.add_subplot(gs[0, 1])
    ax_ladder = fig.add_subplot(gs[0, 2])
    ax_strip = fig.add_subplot(gs[1, :])
    ax_actions = fig.add_subplot(gs[2, :])

    # ─── Track panel ─────────────────────────────────────────────────
    ax_track.plot(xy[:, 0], xy[:, 1], color="#1a1a1a", linewidth=3.0,
                  solid_capstyle="round", zorder=2)
    ax_track.plot(xy[:, 0], xy[:, 1], color="#cfcfcf", linewidth=1.2, zorder=3)
    ax_track.set_aspect("equal", adjustable="box")
    ax_track.axis("off")
    title = title_override or f"{track.name}"
    ax_track.set_title(title, fontsize=14, fontweight="bold", pad=6)
    ax_track.plot([xy[0, 0]], [xy[0, 1]], marker="s", markersize=10,
                  color="#1a1a1a", zorder=4)

    # Trail of ego positions (light line that grows behind the ego dot)
    ego_trail, = ax_track.plot([], [], color=EGO_COLOR, linewidth=1.4,
                               alpha=0.45, zorder=6)
    ego_dot = ax_track.scatter([], [], s=260, c=EGO_COLOR,
                               edgecolors="#1a1a1a", linewidths=2.5, zorder=10)
    ego_label = ax_track.text(0, 0, "", fontsize=9, fontweight="bold",
                              color="#1a1a1a", ha="center", va="center", zorder=11)
    opp_dots = ax_track.scatter([], [], s=80, c=OPP_COLOR,
                                edgecolors="white", linewidths=0.9, zorder=8)

    # ─── HUD panel (middle column) ───────────────────────────────────
    ax_hud.set_xlim(0, 1)
    ax_hud.set_ylim(0, 1)
    ax_hud.axis("off")

    hud_lap = ax_hud.text(0.04, 0.96, "", fontsize=11, fontweight="bold",
                          va="top", color="#1a1a1a")
    hud_pos = ax_hud.text(0.04, 0.85, "", fontsize=28, fontweight="bold",
                          va="top", color=EGO_COLOR)
    hud_compound_label = ax_hud.text(0.04, 0.66, "Tyres", fontsize=9,
                                     va="top", color="#666")
    hud_compound = ax_hud.text(0.04, 0.61, "", fontsize=11, fontweight="bold",
                               va="top", color="#1a1a1a")
    hud_age = ax_hud.text(0.04, 0.555, "", fontsize=8.5, va="top", color="#444")
    tyre_bar_bg = mpatches.Rectangle((0.04, 0.48), 0.92, 0.035,
                                     facecolor="#e0e0e0", edgecolor="#999",
                                     linewidth=0.6)
    tyre_bar = mpatches.Rectangle((0.04, 0.48), 0.0, 0.035,
                                  facecolor="#1a8754", edgecolor=None)
    ax_hud.add_patch(tyre_bar_bg)
    ax_hud.add_patch(tyre_bar)

    hud_fuel_label = ax_hud.text(0.04, 0.40, "Fuel", fontsize=9,
                                 va="top", color="#666")
    hud_fuel = ax_hud.text(0.04, 0.35, "", fontsize=11, fontweight="bold",
                           va="top", color="#1a1a1a")
    fuel_bar_bg = mpatches.Rectangle((0.04, 0.275), 0.92, 0.035,
                                     facecolor="#e0e0e0", edgecolor="#999",
                                     linewidth=0.6)
    fuel_bar = mpatches.Rectangle((0.04, 0.275), 0.0, 0.035,
                                  facecolor="#003e8a", edgecolor=None)
    ax_hud.add_patch(fuel_bar_bg)
    ax_hud.add_patch(fuel_bar)

    hud_weather_label = ax_hud.text(0.04, 0.20, "Weather", fontsize=9,
                                    va="top", color="#666")
    hud_weather = ax_hud.text(0.04, 0.15, "", fontsize=10.5, fontweight="bold",
                              va="top", color="#1a1a1a")
    hud_status_label = ax_hud.text(0.04, 0.08, "Status", fontsize=9,
                                   va="top", color="#666")
    hud_status = ax_hud.text(0.04, 0.03, "", fontsize=10.5, fontweight="bold",
                             va="top", color="#1a1a1a")

    # ─── Position ladder (right column) ──────────────────────────────
    ax_ladder.set_xlim(0, 1)
    ax_ladder.set_ylim(0, 1)
    ax_ladder.axis("off")
    ax_ladder.text(0.5, 0.97, "RACE ORDER", fontsize=10, fontweight="bold",
                   ha="center", va="top", color="#1a1a1a")
    # Pre-allocate up to 7 ladder rows (ego + 6 opponents)
    LADDER_TOP = 0.90
    LADDER_BOTTOM = 0.05
    LADDER_ROWS = 7
    row_height = (LADDER_TOP - LADDER_BOTTOM) / LADDER_ROWS
    ladder_text_handles = []
    ladder_box_handles = []
    for i in range(LADDER_ROWS):
        y = LADDER_TOP - row_height * (i + 0.5)
        box = mpatches.Rectangle(
            (0.05, y - row_height * 0.42),
            0.9, row_height * 0.84,
            facecolor="#f0f0f0", edgecolor="#cfcfcf",
            linewidth=0.6, zorder=2,
        )
        ax_ladder.add_patch(box)
        ladder_box_handles.append(box)
        text = ax_ladder.text(
            0.10, y, "",
            fontsize=10, fontweight="bold",
            va="center", ha="left", zorder=3,
        )
        ladder_text_handles.append(text)

    # ─── Lap strip ───────────────────────────────────────────────────
    ax_strip.set_xlim(0, total_laps)
    ax_strip.set_ylim(0, 1)
    ax_strip.set_yticks([])
    ax_strip.set_xticks(range(0, total_laps + 1, max(1, total_laps // 8)))
    ax_strip.set_xlabel("Lap", fontsize=10)
    ax_strip.tick_params(axis="x", labelsize=9)
    for spine in ("top", "right", "left"):
        ax_strip.spines[spine].set_visible(False)

    weather_strip = ax_strip.imshow(
        np.zeros((1, total_laps, 3)),
        extent=[0, total_laps, 0.10, 0.55],
        aspect="auto", zorder=2,
    )
    pit_marks = ax_strip.scatter([], [], marker="v", color="#e10600",
                                 edgecolors="#1a1a1a", linewidths=1.2,
                                 s=130, zorder=4, label="pit stop")
    inspection_marks = ax_strip.scatter([], [], marker="o",
                                        color="#003e8a", edgecolors="white",
                                        linewidths=0.8, s=55, zorder=3,
                                        label="inspection")
    current_lap_line = ax_strip.axvline(0, color="#c1121f", linewidth=2.4,
                                        zorder=5, alpha=0.85)
    ax_strip.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)

    # ─── Action callout ──────────────────────────────────────────────
    ax_actions.set_xlim(0, 1)
    ax_actions.set_ylim(0, 1)
    ax_actions.axis("off")
    action_text = ax_actions.text(
        0.005, 0.5, "", fontsize=11, va="center", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="#fff8e1", edgecolor="#ffd60a",
                  linewidth=1.2, alpha=0.95),
    )

    weather_columns = _build_weather_strip_colors(frames, total_laps)

    # Pre-compute ego trail along the track centerline (one xy per frame)
    n_pts = len(xy)
    trail_xs: list[float] = []
    trail_ys: list[float] = []

    def update(i: int):
        frame = frames[i]
        obs = _obs_from_frame(frame)
        action = frame.get("action", "")
        lap = int(obs.get("current_lap", i))
        compound = obs.get("ego_tyre_compound", "medium")
        compound_color = COMPOUND_COLORS.get(compound, DEFAULT_COMPOUND_COLOR)
        compound_edge = COMPOUND_EDGE.get(compound, "#444")
        position = int(obs.get("ego_position", 4))
        opponents = obs.get("opponents", []) or []

        # Build full grid from ego + opponents, sort by position
        # Each entry: (position, label, is_ego)
        grid = []
        ego_label_text = "EGO"
        try:
            grid.append((position, ego_label_text, True, compound))
        except Exception:
            pass
        for opp in opponents:
            try:
                p = int(opp.get("position", 99))
                label = f"#{opp.get('driver_number', '?')} {opp.get('team', '')[:9]}"
                comp = opp.get("current_compound", "medium")
                grid.append((p, label, False, comp))
            except Exception:
                pass
        # Dedup positions (sometimes ego shares with an opponent in early frames)
        grid_sorted = sorted(grid, key=lambda x: x[0])

        # Place ego on the track at a position that REFLECTS rank: the leader
        # sits at the start/finish line + lap fraction; lower-ranked cars sit
        # progressively earlier. Within a lap, all cars are nearby on the
        # circuit but separated by ~3% of lap each so the visual is meaningful.
        leader_idx = int((max(0, lap) / max(1, total_laps)) * (n_pts - 1)) % n_pts
        # Each rank-step pushes the car ~3% of the lap behind the leader.
        rank_back_frac = 0.025
        ego_idx = (leader_idx - int((position - 1) * rank_back_frac * n_pts)) % n_pts
        ego_dot.set_offsets(np.array([[xy[ego_idx, 0], xy[ego_idx, 1]]]))
        ego_dot.set_color(compound_color)
        ego_dot.set_edgecolor(compound_edge)
        ego_label.set_position((xy[ego_idx, 0], xy[ego_idx, 1]))
        ego_label.set_text(f"P{position}")

        trail_xs.append(xy[ego_idx, 0])
        trail_ys.append(xy[ego_idx, 1])
        ego_trail.set_data(trail_xs[-30:], trail_ys[-30:])

        # Opponents on track: each at their own rank-back offset
        opp_xy = []
        for opp in opponents[:6]:
            try:
                op = int(opp.get("position", 99))
            except Exception:
                op = 99
            o_idx = (leader_idx - int((op - 1) * rank_back_frac * n_pts)) % n_pts
            opp_xy.append([xy[o_idx, 0], xy[o_idx, 1]])
        if opp_xy:
            opp_dots.set_offsets(np.array(opp_xy))
        else:
            opp_dots.set_offsets(np.empty((0, 2)))

        # ─── HUD updates ─────────────────────────────────────────────
        total = obs.get("total_laps", total_laps)
        hud_lap.set_text(f"Lap {lap}/{total}")
        hud_pos.set_text(f"P{position}")
        hud_compound.set_text(compound.upper())
        hud_compound.set_color(compound_edge)
        age = obs.get("ego_tyre_age_laps", 0)
        hud_age.set_text(f"age: {age} lap(s)")

        health_pct = float(obs.get("ego_tyre_health_pct", 100.0))
        tyre_bar.set_width(0.92 * max(0.0, min(1.0, health_pct / 100.0)))
        tyre_bar.set_facecolor(
            "#1a8754" if health_pct > 60 else
            "#ffd60a" if health_pct > 30 else "#c1121f"
        )

        fuel_kg = float(obs.get("ego_fuel_remaining_kg", 0.0))
        hud_fuel.set_text(f"{fuel_kg:.1f} kg")
        fuel_bar.set_width(0.92 * max(0.0, min(1.0, fuel_kg / 100.0)))

        wx = obs.get("weather_current", {}) or {}
        rain = float(wx.get("rain_intensity", 0.0) or 0.0)
        surface = wx.get("surface_state", "dry") or "dry"
        if rain > 0.6:
            wx_label, wx_col = f"Rain {rain:.1f} (heavy)", "#003e8a"
        elif rain > 0.2:
            wx_label, wx_col = f"Rain {rain:.1f} (light)", "#005bb5"
        elif surface == "damp":
            wx_label, wx_col = "Damp", "#5c8a3a"
        else:
            wx_label = f"Dry  ({wx.get('air_temp_c', '?')}°C)"
            wx_col = "#1a8754"
        hud_weather.set_text(wx_label)
        hud_weather.set_color(wx_col)

        status = obs.get("race_status", "green") or "green"
        hud_status.set_text(status.upper())
        hud_status.set_color(STATUS_COLORS.get(status, "#1a1a1a"))

        # ─── Position ladder ─────────────────────────────────────────
        for j in range(LADDER_ROWS):
            box = ladder_box_handles[j]
            txt = ladder_text_handles[j]
            if j < len(grid_sorted):
                p, label, is_ego, comp = grid_sorted[j]
                comp_color = COMPOUND_COLORS.get(comp, "#888")
                if is_ego:
                    box.set_facecolor("#ffe5e5")
                    box.set_edgecolor(EGO_COLOR)
                    box.set_linewidth(1.6)
                    txt.set_text(f"P{p}  ◆ {label}")
                    txt.set_color(EGO_COLOR)
                    txt.set_fontweight("bold")
                else:
                    box.set_facecolor("#f5f5f5")
                    box.set_edgecolor("#cfcfcf")
                    box.set_linewidth(0.6)
                    txt.set_text(f"P{p}  ● {label}")
                    txt.set_color("#1a1a1a")
                    txt.set_fontweight("normal")
            else:
                box.set_facecolor("#f0f0f0")
                box.set_edgecolor("#cfcfcf")
                box.set_linewidth(0.4)
                txt.set_text("")

        # ─── Lap strip ───────────────────────────────────────────────
        weather_strip.set_data(weather_columns)
        current_lap_line.set_xdata([lap, lap])

        pit_xs: list[int] = []
        insp_xs: list[int] = []
        for f in frames[: i + 1]:
            o = _obs_from_frame(f)
            l = int(o.get("current_lap", 0))
            msg = str(o.get("message", "")).lower()
            act = str(f.get("action", "")).lower()
            if msg.startswith("box") or "pit_now" in act:
                pit_xs.append(l)
            elif any(tok in act for tok in
                     ("inspect", "request_forecast", "assess_undercut",
                      "check_opponent")):
                insp_xs.append(l)
        pit_marks.set_offsets(
            np.array([[x, 0.78] for x in pit_xs]) if pit_xs else np.empty((0, 2))
        )
        inspection_marks.set_offsets(
            np.array([[x, 0.78] for x in insp_xs]) if insp_xs else np.empty((0, 2))
        )

        # ─── Action callout ──────────────────────────────────────────
        recent = []
        for f in frames[max(0, i - 2): i + 1]:
            a = f.get("action", "RESET")
            if a and a != "RESET":
                recent.append(
                    f"L{int(_obs_from_frame(f).get('current_lap', 0)):02d}  ▸  {a}"
                )
        if not recent:
            recent = ["L00  ▸  RESET"]
        action_text.set_text("\n".join(recent[-3:]))

        return (
            ego_dot, ego_trail, ego_label, opp_dots,
            tyre_bar, fuel_bar,
            hud_lap, hud_pos, hud_compound, hud_age,
            hud_fuel, hud_weather, hud_status,
            current_lap_line, pit_marks, inspection_marks, action_text,
            *ladder_text_handles, *ladder_box_handles,
        )

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=600, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "mp4" or output_path.suffix.lower() == ".mp4":
        anim.save(output_path, fps=2)
    else:
        anim.save(output_path, writer=PillowWriter(fps=2))
    plt.close(fig)


def _build_weather_strip_colors(frames, total_laps: int):
    import numpy as np

    columns = np.zeros((1, total_laps, 3))
    for lap_idx in range(total_laps):
        obs = _nearest_obs_for_lap(frames, lap_idx + 1)
        wx = obs.get("weather_current", {}) or {}
        rain = float(wx.get("rain_intensity", 0.0) or 0.0)
        if rain > 0.05:
            columns[0, lap_idx, :] = [
                0.06 + 0.10 * (1 - rain),
                0.30 + 0.18 * (1 - rain),
                0.45 + 0.45 * rain,
            ]
        else:
            columns[0, lap_idx, :] = [0.12, 0.55, 0.20]
    return columns


def build_gradio_panel(env):
    import gradio as gr

    def reset(task: str, seed):
        try:
            obs = env.reset(task=task, seed=int(seed))
        except TypeError:
            from server.scenarios import SCENARIOS
            obs = env.reset(seed=int(seed),
                            options={"scenario": SCENARIOS[task]})
        return obs.model_dump()

    def step(command: str):
        from models import F1Action

        obs = env.step(F1Action(command=command))
        return obs.model_dump()

    landing_enabled = os.environ.get("ENABLE_LANDING_PAGE", "1") == "1"
    repo_root = Path(__file__).resolve().parent.parent

    with gr.Blocks(title="F1 Strategist",
                   theme=gr.themes.Soft(primary_hue="red")) as demo:
        if landing_enabled:
            gr.Markdown(_landing_markdown())

            race_story = repo_root / "results" / "race_story.png"
            track_grid = repo_root / "results" / "track_grid.png"
            eval_chart = repo_root / "results" / "eval_curve.png"
            train_chart = repo_root / "results" / "training_loss_curve.png"
            untrained_gif = repo_root / "demo-assets" / "untrained-spa.gif"
            trained_gif = repo_root / "demo-assets" / "trained-spa.gif"
            before_after = repo_root / "captures" / "before_after_weather_roulette.gif"

            if race_story.exists():
                gr.Image(value=str(race_story),
                         label="The race story — untrained vs trained, lap by lap",
                         show_label=True, height=380)

            if before_after.exists():
                gr.Image(value=str(before_after),
                         label="Before vs after training (weather roulette at Spa)",
                         show_label=True, height=440)
            elif untrained_gif.exists() and trained_gif.exists():
                with gr.Row():
                    gr.Image(value=str(untrained_gif),
                             label="Untrained — Spa, weather roulette",
                             show_label=True, height=320)
                    gr.Image(value=str(trained_gif),
                             label="After GRPO — same seed",
                             show_label=True, height=320)

            with gr.Row():
                if eval_chart.exists():
                    gr.Image(value=str(eval_chart),
                             label="Held-out evaluation",
                             show_label=True, height=340)
                if train_chart.exists():
                    gr.Image(value=str(train_chart),
                             label="GRPO training curve",
                             show_label=True, height=340)
            if track_grid.exists():
                gr.Image(value=str(track_grid),
                         label="Trained policy across multiple tracks",
                         show_label=True, height=420)

            gr.Markdown("---\n## Try it interactively")

        with gr.Row():
            task = gr.Dropdown(
                choices=[
                    "dry_strategy_sprint",
                    "weather_roulette",
                    "late_safety_car",
                    "championship_decider",
                ],
                value="dry_strategy_sprint",
                label="Scenario",
            )
            seed = gr.Number(value=42, precision=0, label="Seed")
            reset_btn = gr.Button("Reset", variant="primary")
        with gr.Row():
            command = gr.Textbox(
                value="STAY_OUT",
                label="Command",
                placeholder="e.g. PIT_NOW soft, REQUEST_FORECAST, "
                            "ASSESS_UNDERCUT_WINDOW, RADIO_DRIVER \"Box this lap\"",
            )
            step_btn = gr.Button("Step")
        state = gr.JSON(label="Observation")
        reset_btn.click(reset, inputs=[task, seed], outputs=state)
        step_btn.click(step, inputs=command, outputs=state)
    return demo


def _landing_markdown() -> str:
    return """
# F1 Strategist 🏎️

**OpenEnv environment for training LLM agents as Formula 1 race strategists.**
Built for the Meta PyTorch OpenEnv Hackathon Grand Finale.

The agent acts as the strategist on the pit wall — it does **not** drive the car.
A built-in physics simulator handles laps, tyres, fuel, and opponents. The model
emits ~20 strategic commands (PIT_NOW, SET_MODE, REQUEST_FORECAST, ASSESS_UNDERCUT_WINDOW,
RADIO_DRIVER, …) over a 10-15 lap race.

| | random | untrained | **trained (GRPO)** | expert |
|---|---:|---:|---:|---:|
| Dry sprint | 0.36 | 0.51 | **0.96** | 0.97 |
| Weather roulette | 0.33 | 0.38 | **0.95** | 0.95 |
| Late safety car | 0.31 | 0.42 | **0.94** | 0.94 |
| Championship decider | 0.42 | 0.30 | **0.86** | 0.96 |

Higher is better. **GRPO training closed +0.46 to +0.57 of the gap** to the
hand-authored expert across all four families.

**Trained model:** [Deltasthic/f1-strategist-qwen3-4b-grpo](https://huggingface.co/Deltasthic/f1-strategist-qwen3-4b-grpo)
**GitHub:** [Deltasthicc/f1-strategist](https://github.com/Deltasthicc/f1-strategist)
""".strip()


def _read_rollout_frames(path):
    frames = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frames.append(json.loads(line))
    return frames


def _obs_from_frame(frame: dict) -> dict:
    return frame.get("observation", frame)


def _max_laps(frames) -> int:
    return max(
        int(_obs_from_frame(f).get("total_laps",
                                   _obs_from_frame(f).get("current_lap", 1)))
        for f in frames
    )


def _nearest_obs_for_lap(frames, lap: int) -> dict:
    obs_rows = [_obs_from_frame(f) for f in frames]
    return min(obs_rows, key=lambda obs: abs(int(obs.get("current_lap", 0)) - lap))