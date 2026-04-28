"""
Calibrate tyre compound baselines from observed stint lengths and pace patterns.

Owner: Person 2.

Outputs:
    data/tyre_compound_baseline.json:
        {
            "soft":   {"pace_delta_s": -0.6, "wear_rate": 0.07, "wear_penalty_s_per_unit": 1.8},
            "medium": {"pace_delta_s":  0.0, "wear_rate": 0.045, "wear_penalty_s_per_unit": 1.5},
            "hard":   ...
            "inter":  ...
            "wet":    ...
        }

Used by:
    server/physics.py

The Kaggle archive does NOT include compound info per lap before 2014, so we
seed defaults from the modern-era values in docs/physics-model.md and let the
operator tune from there. This script mostly exists to make the calibration
deterministic and reproducible.

CLI:
    python scripts/calibrate_tyre_baseline.py
"""

import argparse
import json
from pathlib import Path

DEFAULTS = {
    "soft": {"pace_delta_s": -0.6, "wear_rate": 0.07, "wear_penalty_s_per_unit": 1.8},
    "medium": {"pace_delta_s": 0.0, "wear_rate": 0.045, "wear_penalty_s_per_unit": 1.5},
    "hard": {"pace_delta_s": +0.4, "wear_rate": 0.030, "wear_penalty_s_per_unit": 1.2},
    "inter": {"pace_delta_s": +1.5, "wear_rate": 0.060, "wear_penalty_s_per_unit": 2.0},
    "wet": {"pace_delta_s": +3.5, "wear_rate": 0.050, "wear_penalty_s_per_unit": 2.5},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/tyre_compound_baseline.json")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(DEFAULTS, f, indent=2)
    print(f"wrote {args.output} with default values from docs/physics-model.md")


if __name__ == "__main__":
    main()
