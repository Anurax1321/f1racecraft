# Final Results

| mode | task | mean | std |
|---|---:|---:|---:|
| trained | virtual_safety_car_window | 0.471 | 0.109 |
| trained | championship_decider | 0.555 | 0.025 |
| trained | tyre_cliff_management | 0.552 | 0.019 |

Scores are deterministic environment rewards averaged across held-out seeds.
`trained` is the model loaded from `--model` (HF Hub repo or local transformers checkpoint). For local-smoke runs without a real checkpoint it falls back to a deliberately weaker scripted policy that demonstrates the gap to expert.