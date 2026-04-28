"""Per-scenario expert-vs-panic smoke test."""

import sys
from pathlib import Path

# Ensure repo root is on sys.path regardless of where this script is invoked from.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from baselines.expert_solver import EXPERT_SEQUENCES, PANIC_SEQUENCES, run_sequence
from server.scenarios import SCENARIOS


SCENARIOS_TO_TEST = [
    "dry_strategy_sprint",
    "weather_roulette",
    "late_safety_car",
    "championship_decider",
    "virtual_safety_car_window",
    "tyre_cliff_management",
]


def main() -> int:
    failures = 0
    for family in SCENARIOS_TO_TEST:
        scenario = SCENARIOS[family]
        expert_score, _ = run_sequence(scenario, EXPERT_SEQUENCES[family], seed=42)
        panic_score, _ = run_sequence(scenario, PANIC_SEQUENCES[family], seed=42)
        print(f"{family:<24} expert={expert_score:.3f} panic={panic_score:.3f}")
        if expert_score < 0.85:
            print(f"FAIL: {family} expert score below 0.85")
            failures += 1
        if panic_score > 0.40:
            print(f"FAIL: {family} panic score above 0.40")
            failures += 1
    if failures:
        print(f"\n{failures} scenario check(s) failed.")
    else:
        print("\nAll scenario checks passed.")
    return failures


if __name__ == "__main__":
    sys.exit(main())
