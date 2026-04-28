"""Rule-based expert and panic policies for scenario smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models import F1Action
from server.environment import F1StrategistEnvironment
from server.scenarios import SCENARIOS


EXPERT_SEQUENCES = {
    "dry_strategy_sprint": [
        "INSPECT_TYRE_DEGRADATION",
        "CHECK_OPPONENT_STRATEGY 16",
        "CHECK_OPPONENT_STRATEGY 44",
        "ASSESS_UNDERCUT_WINDOW",
        "SET_MODE push",
        "STAY_OUT",
        "STAY_OUT",
        "RADIO_DRIVER Box this lap. Target P3 on softs.",
        "PIT_NOW soft",
        "SET_MODE race",
        "STAY_OUT",
        "INSPECT_FUEL_MARGIN",
        "RADIO_DRIVER All clear. Bring it home P3.",
        "DONE",
    ],
    "weather_roulette": [
        "REQUEST_FORECAST",
        "INSPECT_TYRE_DEGRADATION",
        "CHECK_OPPONENT_STRATEGY 1",
        "SET_MODE race",
        "STAY_OUT",
        "REQUEST_FORECAST",
        "RADIO_DRIVER Rain arriving. Inters incoming. Plan to box lap 7.",
        "STAY_OUT",
        "RADIO_DRIVER Box now. Inters.",
        "PIT_NOW inter",
        "SET_MODE push",
        "STAY_OUT",
        "STAY_OUT",
        "INSPECT_FUEL_MARGIN",
        "DONE",
    ],
    "late_safety_car": [
        "INSPECT_TYRE_DEGRADATION",
        "CHECK_OPPONENT_STRATEGY 4",
        "CHECK_OPPONENT_STRATEGY 11",
        "ASSESS_UNDERCUT_WINDOW",
        "SET_MODE conserve",
        "HOLD_GAP 4",
        "HOLD_GAP 4",
        "RADIO_DRIVER SC. Boxing this lap.",
        "PIT_NOW hard",
        "SET_MODE race",
        "STAY_OUT",
        "SET_MODE push",
        "ATTACK_AHEAD",
        "INSPECT_FUEL_MARGIN",
        "DONE",
    ],
    "championship_decider": [
        "INSPECT_TYRE_DEGRADATION",
        "CHECK_OPPONENT_STRATEGY 10",
        "REQUEST_FORECAST",
        "HOLD_GAP 10",
        "SET_MODE race",
        "STAY_OUT",
        "RADIO_DRIVER Cover rival #10 if he boxes.",
        "PIT_NOW medium",
        "REQUEST_FORECAST",
        "SET_MODE push",
        "DEFEND_POSITION",
        "STAY_OUT",
        "INSPECT_FUEL_MARGIN",
        "RADIO_DRIVER Rival covered. Bring home the title.",
        "DONE",
    ],
    # VSC scenario — exploit the half-cost pit window during Virtual Safety Car
    "virtual_safety_car_window": [
        "INSPECT_TYRE_DEGRADATION",
        "CHECK_OPPONENT_STRATEGY 44",
        "ASSESS_UNDERCUT_WINDOW",
        "SET_MODE race",
        "STAY_OUT",
        "STAY_OUT",
        "STAY_OUT",
        "STAY_OUT",
        "RADIO_DRIVER VSC deployed. Boxing for softs this lap. Low pit loss.",
        "PIT_NOW soft",
        "SET_MODE push",
        "STAY_OUT",
        "STAY_OUT",
        "INSPECT_FUEL_MARGIN",
        "DONE",
    ],
    # Tyre cliff — inspect degradation rate early, pit before health collapses
    "tyre_cliff_management": [
        "INSPECT_TYRE_DEGRADATION",
        "CHECK_OPPONENT_STRATEGY 16",
        "SET_MODE conserve",
        "MANAGE_TYRE_TEMP cool",
        "INSPECT_TYRE_DEGRADATION",
        "RADIO_DRIVER Tyre cliff incoming. Softs will be gone by lap 9. Boxing for medium.",
        "STAY_OUT",
        "PIT_NOW medium",
        "SET_MODE push",
        "STAY_OUT",
        "STAY_OUT",
        "STAY_OUT",
        "INSPECT_FUEL_MARGIN",
        "DONE",
    ],
}


PANIC_SEQUENCES = {
    "dry_strategy_sprint": ["PIT_NOW soft", "PIT_NOW hard", "STAY_OUT", "STAY_OUT", "DONE"],
    "weather_roulette": ["STAY_OUT", "STAY_OUT", "STAY_OUT", "PIT_NOW soft", "STAY_OUT", "DONE"],
    "late_safety_car": ["PIT_NOW hard", "STAY_OUT", "STAY_OUT", "STAY_OUT", "DONE"],
    "championship_decider": ["SET_MODE conserve", "STAY_OUT", "STAY_OUT", "STAY_OUT", "DONE"],
    # VSC panic: pit at wrong time (before VSC, then again during VSC — over-pitting)
    "virtual_safety_car_window": [
        "PIT_NOW medium", "STAY_OUT", "STAY_OUT", "STAY_OUT",
        "PIT_NOW soft", "STAY_OUT", "STAY_OUT", "DONE",
    ],
    # Cliff panic: stay out until the cliff, then pit with wrong compound
    "tyre_cliff_management": [
        "STAY_OUT", "STAY_OUT", "STAY_OUT", "STAY_OUT",
        "STAY_OUT", "STAY_OUT", "STAY_OUT", "STAY_OUT",
        "PIT_NOW hard", "STAY_OUT", "DONE",
    ],
}


class ExpertSolver:
    def solve(self, scenario: dict, env=None) -> list[F1Action]:
        family = scenario.get("scenario_family", "")
        return [F1Action(command=cmd) for cmd in EXPERT_SEQUENCES[family]]


def solve(scenario_or_family, seed: int | None = None) -> float:
    """Run the expert policy and return final weighted score."""
    if isinstance(scenario_or_family, str):
        scenario = SCENARIOS[scenario_or_family]
    else:
        scenario = scenario_or_family
    env = F1StrategistEnvironment()
    obs = env.reset(seed=seed, options={"scenario": scenario})
    for action in ExpertSolver().solve(scenario, env):
        obs = env.step(action)
        if obs.done:
            break
    return float(obs.multi_objective_scores.get("weighted_final", obs.score))


def run_sequence(
    scenario_or_family, commands: list[str], seed: int | None = None
) -> tuple[float, list[dict]]:
    scenario = (
        SCENARIOS[scenario_or_family] if isinstance(scenario_or_family, str) else scenario_or_family
    )
    env = F1StrategistEnvironment()
    obs = env.reset(seed=seed, options={"scenario": scenario})
    trace = [{"observation": obs.model_dump(), "action": "RESET"}]
    for command in commands:
        obs = env.step(F1Action(command=command))
        trace.append({"observation": obs.model_dump(), "action": command})
        if obs.done:
            break
    score = float(obs.multi_objective_scores.get("weighted_final", obs.score))
    return score, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save")
    args = parser.parse_args()
    scenario = SCENARIOS[args.task]
    score, trace = run_sequence(scenario, EXPERT_SEQUENCES[scenario["scenario_family"]], args.seed)
    print(f"{scenario['task_name']} expert score: {score:.3f}")
    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in trace:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
