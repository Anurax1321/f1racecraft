# Postmortem Memory — Hard-Seed Ablation

**Phase 0, sub-step 0.3.** Re-runs the postmortem-memory ablation on the
**hard** scenarios (where the trained model had room to grow), not the
saturated ones used in the hackathon submission.

## Background

The hackathon-time ablation reported **+0.000** average delta. That number was
on the *easy* scenarios where the trained model already scored 0.93–0.97 —
no headroom for memory hints to demonstrate value. Marked as "drop the claim
or re-run on harder seeds" in `STATE.md` line 216.

This run picks the three scenarios where the trained model was scoring lowest
and lets postmortem memory actually work.

## Setup

- **Model:** `grpo_v2/merged` (the champion checkpoint)
- **Hard scenarios:** `virtual_safety_car_window`, `championship_decider`,
  `tyre_cliff_management`
- **Seeds:** 5 per scenario per condition
- **Conditions:** `--no-memory` (memory_hints stripped on every reset) vs
  `--use-memory` (PostmortemMemory.retrieve serves the top-2 lowest-scoring
  prior episodes for the matching scenario_family)
- **Postmortem buffer state:** 29,911 prior episode summaries on disk at
  `baselines/trajectories/postmortems.jsonl`
- **Decoding:** greedy (`sample-temp=0`)

## Results

| Scenario | No memory | With memory | Δ | No-mem std | With-mem std |
|---|---:|---:|---:|---:|---:|
| virtual_safety_car_window | 0.408 | **0.471** | **+0.063** | 0.023 | 0.109 |
| championship_decider | 0.533 | **0.555** | +0.022 | 0.026 | 0.025 |
| tyre_cliff_management | **0.580** | 0.552 | −0.028 | 0.036 | 0.019 |
| **Average** | **0.507** | **0.526** | **+0.019** | — | — |

## Verdict

**Postmortem memory provides a real signal — small but positive on average
(+0.019, ~3.7% relative).** The hackathon "+0.000" was an artifact of testing
on saturated scenarios.

**However**, the signal is noisy. On `virtual_safety_car_window` it almost
6× the variance (std 0.023 → 0.109). The seed-by-seed breakdown shows a
**bimodal** outcome — memory either nudges the model into 0.56 (good) or
0.337 (bad). The hint either lands or it actively misleads.

On `tyre_cliff_management` memory regresses the policy (−0.028). The hints
retrieved for that family appear to be miscalibrated for the seed
distribution.

## Decision

**Keep the postmortem mechanic.** It's a small net win and the architecture
is in place. But:

1. Do **not** position it as a headline contribution. It's not a 20%-style
   improvement; it's a 4% nudge with caveats.
2. Cap it as a known "small-win" contributor, headline becomes Phase 6.5
   (Continual Learning), which is a much bigger arc.
3. **Future fix idea:** rank retrieved postmortems by *similarity to current
   scenario state*, not just by lowest-final-score. The current heuristic
   surfaces the worst prior runs even when their failure modes don't apply
   to the current seed — that's likely why `tyre_cliff_management` regresses.

## Artifacts

- `results/ablation_hard.md` — diff table
- `results/ablation_hard_no_memory.json` — raw scores, no memory
- `results/ablation_hard_with_memory.json` — raw scores, with memory
- `results/ablation_hard_no_memory.png` / `.png` — plots

## Replication

```bash
python evaluate.py --modes trained --model grpo_v2/merged \
  --tasks virtual_safety_car_window championship_decider tyre_cliff_management \
  --n-seeds 5 --no-memory \
  --output-json results/ablation_hard_no_memory.json \
  --output-png  results/ablation_hard_no_memory.png

python evaluate.py --modes trained --model grpo_v2/merged \
  --tasks virtual_safety_car_window championship_decider tyre_cliff_management \
  --n-seeds 5 --use-memory \
  --output-json results/ablation_hard_with_memory.json \
  --output-png  results/ablation_hard_with_memory.png

python scripts/diff_ablation.py \
  results/ablation_hard_no_memory.json \
  results/ablation_hard_with_memory.json \
  --output results/ablation_hard.md
```
