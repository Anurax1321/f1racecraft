"""Run one F1 Strategist episode and optionally render it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.expert_solver import EXPERT_SEQUENCES, run_sequence
from evaluate import RANDOM_COMMANDS, _scripted_policy, _untrained_policy
from models import F1Action
from server.environment import F1StrategistEnvironment
from server.scenarios import SCENARIOS


def run_rollout(
    model: str,
    task: str,
    seed: int,
    mode: str,
    render: bool = False,
    verbose: bool = False,
) -> tuple[float, Path]:
    scenario = SCENARIOS[task]
    family = scenario["scenario_family"]
    output_dir = Path("captures")
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{task}_{mode}_seed{seed}.jsonl"

    if mode == "expert":
        score, trace = run_sequence(scenario, EXPERT_SEQUENCES[family], seed)
        # Inject track_name into first frame so the visualizer finds it
        if trace:
            trace[0]["track_name"] = scenario.get("track_name", "Monza")
    else:
        env = F1StrategistEnvironment()
        obs = env.reset(seed=seed, options={"scenario": scenario})
        # Store track_name at the top of the first frame for the visualizer
        trace = [{"action": "RESET", "observation": obs.model_dump(), "track_name": scenario.get("track_name", "Monza")}]
        history: list[dict] = []
        import random

        rng = random.Random(seed)
        while not obs.done:
            if mode == "random":
                command = rng.choice(RANDOM_COMMANDS)
            elif mode == "trained":
                command = _scripted_policy(obs, history, family)
            else:
                command = _untrained_policy(obs, rng, family)
            history.append({"role": "assistant", "content": command})
            if verbose:
                print(f"lap {obs.current_lap:02d}: {command}")
            obs = env.step(F1Action(command=command))
            trace.append({"action": command, "observation": obs.model_dump()})
        score = float(obs.multi_objective_scores.get("weighted_final", obs.score))

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in trace:
            f.write(json.dumps(row) + "\n")

    if render:
        from server.visualizer import render_rollout

        gif_path = jsonl_path.with_suffix(".gif")
        render_rollout(jsonl_path, gif_path)
        print(f"rendered {gif_path}")
    print(f"score={score:.3f} trace={jsonl_path}")
    return score, jsonl_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="heuristic")
    parser.add_argument("--task", default="dry_strategy_sprint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode", choices=["random", "untrained", "trained", "expert"], default="untrained"
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_rollout(args.model, args.task, args.seed, args.mode, args.render, args.verbose)
