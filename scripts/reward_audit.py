"""Diagnostic — reward signal characterization across families and lengths.

Runs scripted policies (heuristic + a deliberately-bad "panic" policy) on
every scenario family at short and long lengths, then prints the dim-by-dim
breakdown plus the per-step reward distribution.

Why this exists:
  Per `docs/reward-philosophy.md`, we don't add dense shaping by default —
  but we do need to verify the terminal scorer is doing its job. This script
  is the empirical check: when a "good" sequence beats a "bad" sequence by
  ≥0.20 weighted_final at every length, the scorer is sound.

Usage:
  python scripts/reward_audit.py
  python scripts/reward_audit.py --length long --family monaco_full_gp
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter

from models import F1Action
from server.environment import F1StrategistEnvironment
from server.generator import RACE_LENGTHS, scale_scenario_to_length
from server.scenarios import SCENARIOS

SHORT_FAMILIES = [
    "dry_strategy_sprint", "weather_roulette", "late_safety_car",
    "championship_decider", "virtual_safety_car_window", "tyre_cliff_management",
]
LONG_FAMILIES = ["monaco_full_gp", "silverstone_full_gp", "spa_full_wet"]


def _heuristic_policy(obs, history):
    """A reasonable scripted policy — investigate, then pit, then race."""
    lap = int(obs.current_lap)
    total = int(obs.total_laps)
    seen = {h.get("action", "").split()[0] for h in history if h.get("action")}
    # Investigate early
    if lap <= 2 and "REQUEST_FORECAST" not in seen:
        return "REQUEST_FORECAST"
    if lap <= 3 and "ASSESS_UNDERCUT_WINDOW" not in seen:
        return "ASSESS_UNDERCUT_WINDOW"
    if lap <= 4 and "INSPECT_TYRE_DEGRADATION" not in seen:
        return "INSPECT_TYRE_DEGRADATION"
    # Pit windows
    if total <= 15:
        if 4 <= lap <= 6 and "PIT_NOW" not in seen:
            return "PIT_NOW soft"
    else:
        # Long race: 2-stop near 1/3 and 2/3 of the way
        if abs(lap - total * 0.35) <= 1 and "PIT_NOW" not in seen:
            return "PIT_NOW medium"
        if abs(lap - total * 0.70) <= 1 and (history and history[-1].get("action", "").startswith("PIT_NOW")):
            pass  # avoid double pit on consecutive steps
        if total - lap == int(total * 0.30) and "PIT_NOW soft" not in [h.get("action") for h in history]:
            return "PIT_NOW soft"
    if lap == 1:
        return "RADIO_DRIVER Pit window opens lap 5 — push the gap."
    return "STAY_OUT"


def _panic_policy(obs, history):
    """Deliberately-bad: panic-pit lap 1, never inspect, mode-thrash."""
    lap = int(obs.current_lap)
    if lap == 1:
        return "PIT_NOW soft"
    if lap == 2:
        return "SET_MODE push"
    if lap == 3:
        return "SET_MODE conserve"
    if lap == 4:
        return "PIT_NOW hard"
    return "STAY_OUT"


def run_one(family: str, length_key: str, policy_fn) -> dict:
    env = F1StrategistEnvironment()
    options = {"task": family}
    if family not in LONG_FAMILIES:
        options["race_length"] = length_key
    obs = env.reset(seed=7, options=options)

    history: list[dict] = []
    rewards_per_step: list[float] = []
    nonzero_steps = 0
    steps = 0
    while not obs.done and steps < obs.total_laps + 8:
        cmd = policy_fn(obs, history)
        obs = env.step(F1Action(command=cmd))
        history.append({"action": cmd, "lap": obs.current_lap, "reward": obs.reward})
        rewards_per_step.append(float(obs.reward))
        if obs.reward != 0:
            nonzero_steps += 1
        steps += 1

    mos = obs.multi_objective_scores or {}
    return {
        "family": family,
        "length": length_key,
        "policy": policy_fn.__name__,
        "total_laps": int(obs.total_laps),
        "steps": steps,
        "final_pos": int(obs.ego_position),
        "weighted_final": float(mos.get("weighted_final", obs.score)),
        "race_result": float(mos.get("race_result", 0)),
        "strategic_decisions": float(mos.get("strategic_decisions", 0)),
        "tyre_management": float(mos.get("tyre_management", 0)),
        "fuel_management": float(mos.get("fuel_management", 0)),
        "comms_quality": float(mos.get("comms_quality", 0)),
        "operational_efficiency": float(mos.get("operational_efficiency", 0)),
        "shaping_total": round(sum(rewards_per_step), 3),
        "nonzero_step_frac": round(nonzero_steps / max(1, steps), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", choices=["short", "medium", "long", "all"], default="all")
    parser.add_argument("--family", default=None)
    args = parser.parse_args()

    lengths = ["short", "medium", "long"] if args.length == "all" else [args.length]
    families = [args.family] if args.family else (SHORT_FAMILIES + LONG_FAMILIES)

    rows: list[dict] = []
    for family in families:
        if family in LONG_FAMILIES:
            # Native length only — these are 55-lap by design
            for policy in (_heuristic_policy, _panic_policy):
                rows.append(run_one(family, "long", policy))
        else:
            for length in lengths:
                for policy in (_heuristic_policy, _panic_policy):
                    rows.append(run_one(family, length, policy))

    # Print
    header = f"{'family':<28} {'len':<6} {'policy':<18} {'laps':>4} {'pos':>3} " \
             f"{'final':>6} {'race':>5} {'strat':>5} {'tyre':>5} {'fuel':>5} " \
             f"{'comm':>5} {'ops':>5} {'shape':>6} {'nz%':>5}"
    print(header)
    print("─" * len(header))
    for r in rows:
        print(f"{r['family']:<28} {r['length']:<6} {r['policy']:<18} "
              f"{r['total_laps']:>4} {r['final_pos']:>3} "
              f"{r['weighted_final']:>6.3f} {r['race_result']:>5.2f} "
              f"{r['strategic_decisions']:>5.2f} {r['tyre_management']:>5.2f} "
              f"{r['fuel_management']:>5.2f} {r['comms_quality']:>5.2f} "
              f"{r['operational_efficiency']:>5.2f} {r['shaping_total']:>6.3f} "
              f"{int(r['nonzero_step_frac']*100):>4}%")

    # Summary check: heuristic should beat panic at every length
    print("\n─── Sanity: heuristic should beat panic by ≥0.20 weighted_final ───")
    by_key: dict[tuple, dict] = {}
    for r in rows:
        by_key[(r["family"], r["length"], r["policy"])] = r
    failures = []
    for family, length in {(r["family"], r["length"]) for r in rows}:
        h = by_key.get((family, length, "_heuristic_policy"))
        p = by_key.get((family, length, "_panic_policy"))
        if h and p:
            delta = h["weighted_final"] - p["weighted_final"]
            mark = "✓" if delta >= 0.20 else "✗"
            print(f"  {mark} {family:<28} {length:<6} heuristic={h['weighted_final']:.3f}  "
                  f"panic={p['weighted_final']:.3f}  delta={delta:+.3f}")
            if delta < 0.20:
                failures.append((family, length, delta))

    if failures:
        print(f"\n❌ {len(failures)} family/length pair(s) below 0.20 separation.")
        return 1
    print("\n✅ All families discriminate good vs bad play by ≥0.20.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
