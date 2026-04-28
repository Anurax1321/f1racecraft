"""Generate SFT chat data from expert trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.expert_solver import ExpertSolver
from inference import SYSTEM_PROMPT, format_obs
from server.environment import F1StrategistEnvironment
from server.generator import generate
from server.scenarios import SCENARIOS


def main(args) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True) if output.parent != Path(".") else None
    n = 0
    n_pit_dupes = 0
    with output.open("w", encoding="utf-8") as f:
        for task in args.tasks:
            for seed in range(args.n_seeds):
                scenario = generate(task, seed) if args.procedural else SCENARIOS[task]
                env = F1StrategistEnvironment()
                obs = env.reset(seed=seed, options={"scenario": scenario})
                history = [{"role": "system", "content": SYSTEM_PROMPT}]
                for action in ExpertSolver().solve(scenario, env):
                    history.append({"role": "user", "content": format_obs(obs)})
                    if args.multi_turn:
                        # Each row contains the full conversation up to this turn,
                        # so the model learns to condition on its own action history.
                        msgs = list(history) + [
                            {"role": "assistant", "content": action.command}
                        ]
                    else:
                        msgs = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": format_obs(obs)},
                            {"role": "assistant", "content": action.command},
                        ]
                    row = {
                        "messages": msgs,
                        "task": scenario["scenario_family"],
                        "seed": seed,
                        "lap": obs.current_lap,
                    }
                    f.write(json.dumps(row) + "\n")
                    n += 1
                    # PIT boost: duplicate PIT_NOW rows so the model sees them more often
                    if args.pit_boost and action.command.upper().startswith("PIT_NOW"):
                        for _ in range(args.pit_boost - 1):
                            f.write(json.dumps(row) + "\n")
                            n += 1
                            n_pit_dupes += 1
                    history.append(
                        {"role": "assistant", "content": action.command}
                    )
                    obs = env.step(action)
                    if obs.done:
                        break
    print(f"wrote {n} SFT turns to {output} (multi_turn={args.multi_turn}, pit_dupes={n_pit_dupes})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "dry_strategy_sprint",
            "weather_roulette",
            "late_safety_car",
            "championship_decider",
        ],
    )
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--output", default="sft_dataset_v1.jsonl")
    parser.add_argument(
        "--procedural",
        action="store_true",
        help="Use server.generator variants instead of templates.",
    )
    parser.add_argument(
        "--multi-turn",
        action="store_true",
        help="Each row contains conversation history up to the current turn.",
    )
    parser.add_argument(
        "--pit-boost",
        type=int,
        default=1,
        help="Duplicate PIT_NOW rows N times (1=no dup, 4=write each pit row 4x).",
    )
    main(parser.parse_args())
