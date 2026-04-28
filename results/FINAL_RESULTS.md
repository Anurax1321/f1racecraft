# Final Results

| mode | task | mean | std |
|---|---:|---:|---:|
| random | monaco_full_gp | 0.310 | 0.040 |
| random | silverstone_full_gp | 0.310 | 0.040 |
| random | spa_full_wet | 0.290 | 0.049 |
| untrained | monaco_full_gp | 0.533 | 0.108 |
| untrained | silverstone_full_gp | 0.458 | 0.001 |
| untrained | spa_full_wet | 0.458 | 0.001 |
| trained | monaco_full_gp | 0.474 | 0.001 |
| trained | silverstone_full_gp | 0.472 | 0.000 |
| trained | spa_full_wet | 0.457 | 0.001 |
| expert | monaco_full_gp | 0.865 | 0.000 |
| expert | silverstone_full_gp | 0.899 | 0.000 |
| expert | spa_full_wet | 0.990 | 0.000 |

Scores are deterministic environment rewards averaged across held-out seeds.
`trained` is the model loaded from `--model` (HF Hub repo or local transformers checkpoint). For local-smoke runs without a real checkpoint it falls back to a deliberately weaker scripted policy that demonstrates the gap to expert.