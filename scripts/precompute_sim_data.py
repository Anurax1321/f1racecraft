"""
Precompute simulation data for all scenario/seed/model combos.
Saves results to server/static/sim_data.json — used by the frontend
for instant playback without live inference.

Run from repo root:
    python scripts/precompute_sim_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import F1Action
from server.environment import F1StrategistEnvironment
from server.simulate_utils import (
    DIM_KEYS, heuristic, random_action, get_scenario_family, surface_label,
)

TASKS = [
    "weather_roulette",
    "dry_strategy_sprint",
    "late_safety_car",
    "championship_decider",
    "virtual_safety_car_window",
    "tyre_cliff_management",
]
SEEDS = [7, 0, 1, 42, 99]
BASE_MODELS = ["heuristic", "random"]


def run_episode(task: str, seed: int, model: str) -> dict:
    env = F1StrategistEnvironment()
    obs = env.reset(task=task, seed=seed)
    family = get_scenario_family(task)
    seen: set[str] = set()
    laps: list[dict] = []

    for _ in range(max(getattr(obs, "total_laps", 15) + 10, 60)):
        if getattr(obs, "done", False):
            break

        if model == "random":
            action_str = random_action(
                getattr(obs, "current_lap", 0),
                getattr(obs, "total_laps", 12),
            )
        else:
            action_str = heuristic(
                family,
                getattr(obs, "current_lap", 0),
                getattr(obs, "total_laps", 12),
                obs,
                seen,
            )

        action = F1Action(command=action_str)
        weather_d = obs.weather_current or {}
        rain = float(weather_d.get("rain_intensity", 0.0))
        key = action_str.upper().startswith(
            ("PIT_NOW", "RADIO_DRIVER", "REQUEST_FORECAST", "DONE",
             "ASSESS_UNDERCUT_WINDOW", "CHECK_OPPONENT", "INSPECT_TYRE")
        )
        laps.append({
            "lap":       getattr(obs, "current_lap", 0),
            "action":    action_str,
            "position":  int(getattr(obs, "ego_position", 10)),
            "compound":  str(getattr(obs, "ego_tyre_compound", "medium")),
            "health":    round(float(getattr(obs, "ego_tyre_health_pct", 100)), 1),
            "fuel":      round(float(getattr(obs, "ego_fuel_remaining_kg", 100)), 2),
            "weather":   surface_label(weather_d),
            "rain":      round(rain, 2),
            "total_laps": getattr(obs, "total_laps", 12),
            "key":       key,
        })
        seen.add(action_str)
        obs = env.step(action)

    # Terminal lap record
    rain_f = float((obs.weather_current or {}).get("rain_intensity", 0.0))
    laps.append({
        "lap":       int(getattr(obs, "current_lap", 0)),
        "action":    "DONE",
        "position":  int(getattr(obs, "ego_position", 10)),
        "compound":  str(getattr(obs, "ego_tyre_compound", "medium")),
        "health":    round(float(getattr(obs, "ego_tyre_health_pct", 0)), 1),
        "fuel":      round(float(getattr(obs, "ego_fuel_remaining_kg", 0)), 2),
        "weather":   surface_label(obs.weather_current),
        "rain":      round(rain_f, 2),
        "total_laps": getattr(obs, "total_laps", 12),
        "key":       True,
    })

    # Scores from terminal observation
    mos = getattr(obs, "multi_objective_scores", None) or {}
    final_score = round(float(mos.get("weighted_final", getattr(obs, "score", 0.5) or 0.5)), 4)
    dims = {k: round(float(mos.get(k, 0.0)), 3) for k in DIM_KEYS}
    final_pos = int(getattr(obs, "ego_position", 10))

    return {
        "task":        task,
        "seed":        seed,
        "laps":        laps,
        "final_score": final_score,
        "final_pos":   final_pos,
        "dims":        dims,
        "policy":      model,
        "policy_note": "",
    }


def main() -> None:
    out: dict = {}
    total = len(TASKS) * len(SEEDS) * len(BASE_MODELS)
    done = 0
    for task in TASKS:
        for seed in SEEDS:
            for model in BASE_MODELS:
                key = f"{task}|{seed}|{model}"
                print(f"[{done+1}/{total}] {key}", flush=True)
                try:
                    out[key] = run_episode(task, seed, model)
                    print(f"    score={out[key]['final_score']}  pos={out[key]['final_pos']}  laps={len(out[key]['laps'])}", flush=True)
                except Exception as exc:
                    print(f"    FAILED: {exc}", flush=True)
                done += 1

    dest = Path(__file__).parent.parent / "server" / "static" / "sim_data.json"
    dest.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"\nWrote {len(out)} entries to {dest}", flush=True)


if __name__ == "__main__":
    main()
