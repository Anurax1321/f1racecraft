# Pre-Training Baseline

Scores are `weighted_final` averaged over held-out seeds.
This is the **before-training** reference for the progress report.

| Scenario | Random | Untrained | Expert | Gap (exp-rand) |
|---|---:|---:|---:|---:|
| dry_strategy_sprint | 0.350 | 0.489 | 0.840 | +0.490 |
| weather_roulette | 0.410 | 0.418 | 0.950 | +0.540 |
| late_safety_car | 0.314 | 0.560 | 0.935 | +0.622 |
| championship_decider | 0.216 | 0.271 | 0.965 | +0.748 |
| virtual_safety_car_window | 0.298 | 0.375 | 0.965 | +0.667 |
| tyre_cliff_management | 0.205 | 0.290 | 0.965 | +0.759 |

## What to do after training

Run `python evaluate.py --modes random untrained trained expert` to produce
`results/eval_summary.json` and `results/eval_curve.png`.
The **trained** row should sit between *untrained* and *expert* on every
scenario — that is the reward-improvement evidence.

Compare `trained.mean` vs `untrained.mean` in this table.
An improvement of +0.05 or more per scenario is a strong signal.