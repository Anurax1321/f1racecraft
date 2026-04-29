#!/usr/bin/env python
"""Replay-mode runner — pit the env's strategist against a real F1 race.

Loads a real GP from FastF1-extracted data, runs the model (or a heuristic),
and prints a side-by-side comparison of the model's strategic calls vs the
ego driver's actual actions on the day.

Usage:
    python scripts/replay.py --year 2024 --gp monaco --driver LEC
    python scripts/replay.py --list                       # show available
    python scripts/replay.py --year 2024 --gp monaco      # auto-pick driver
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import F1Action
from server.environment import F1StrategistEnvironment
from server.replay import list_replays, load_replay_scenario


def _scripted_policy(obs, history, target_pit_lap: int):
    """Tiny placeholder policy — investigate early, pit at target_pit_lap.
    Real models would replace this with their own action selection."""
    lap = int(obs.current_lap)
    seen = {h.split()[0] for h in history if h}
    if lap <= 1 and "REQUEST_FORECAST" not in seen:
        return "REQUEST_FORECAST"
    if lap <= 2 and "ASSESS_UNDERCUT_WINDOW" not in seen:
        return "ASSESS_UNDERCUT_WINDOW"
    if lap <= 3 and "INSPECT_TYRE_DEGRADATION" not in seen:
        return "INSPECT_TYRE_DEGRADATION"
    if lap == max(1, target_pit_lap - 1) and "PIT_NOW" not in seen:
        return "RADIO_DRIVER Pit this lap for hards."
    if lap == target_pit_lap and "PIT_NOW" not in seen:
        return "PIT_NOW hard"
    return "STAY_OUT"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List available replays")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--gp", default="monaco", help="GP short key (e.g. monaco, italian)")
    parser.add_argument("--driver", default=None, help="3-letter driver code (auto if omitted)")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.list:
        replays = list_replays()
        print(f"{len(replays)} replays available:")
        by_year: dict[int, list[str]] = {}
        for y, gp in replays:
            by_year.setdefault(y, []).append(gp)
        for y in sorted(by_year):
            print(f"  {y}: {', '.join(sorted(by_year[y]))}")
        return 0

    scenario, actual_actions = load_replay_scenario(
        args.year, args.gp, ego_driver=args.driver,
    )
    meta = scenario["_replay_meta"]
    target_pit_lap = (
        actual_actions[0]["lap"] if actual_actions else scenario["total_laps"] // 2
    )

    print(f"╔══ {meta['event_name']} {meta['year']} — replay as {meta['ego_driver']} ══╗")
    print(f"  Total laps:        {scenario['total_laps']}")
    print(f"  Starting position: P{scenario['starting_position']}")
    print(f"  Starting compound: {scenario['starting_compound']}")
    print(f"  Real n_pits:       {meta['ego_n_pits']}")
    print(f"  Real compounds:    {' → '.join(meta['ego_compounds'])}")
    print()

    env = F1StrategistEnvironment()
    obs = env.reset(seed=args.seed, options={"scenario": scenario})
    print(f"  Grounded calibration active: {env._grounded_factors is not None}")
    print()

    history: list[str] = []
    model_actions: list[dict] = []
    steps = 0
    while not obs.done and steps < obs.total_laps + 8:
        cmd = _scripted_policy(obs, history, target_pit_lap)
        obs = env.step(F1Action(command=cmd))
        history.append(cmd)
        if cmd.startswith(("PIT_NOW", "REQUEST_FORECAST", "INSPECT_", "ASSESS_",
                           "CHECK_OPPONENT", "RADIO_DRIVER")):
            model_actions.append({"lap": obs.current_lap, "action": cmd})
        steps += 1

    print(f"  Race ended at lap {obs.current_lap}/{obs.total_laps}, "
          f"pos P{obs.ego_position}, score {obs.score:.3f}")
    print()
    print(f"  ┌─ Strategic-action comparison ─" + "─" * 38 + "┐")
    print(f"  │ {'lap':>3}  {'MODEL':<38}  {'REAL ' + meta['ego_driver']:<22} │")
    print("  ├" + "─" * 70 + "┤")
    all_laps = sorted({a["lap"] for a in model_actions} | {a["lap"] for a in actual_actions})
    for lap in all_laps:
        m = next((a for a in model_actions if a["lap"] == lap), None)
        r = next((a for a in actual_actions if a["lap"] == lap), None)
        m_str = (m["action"][:38]) if m else ""
        r_str = (f"{r['action']} ({r['context']})") if r else ""
        print(f"  │ {lap:>3}  {m_str:<38}  {r_str:<22} │")
    print(f"  └" + "─" * 70 + "┘")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
