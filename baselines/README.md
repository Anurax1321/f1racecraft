# baselines/

Rule-based gold-standard strategies and the trajectories they produce.

## Contents

```
baselines/
├── __init__.py
├── expert_solver.py           — rule-based god-mode strategist (Person 2)
└── trajectories/              — recorded expert traces and postmortems
    ├── expert_dry_seed0.jsonl
    ├── expert_dry_seed1.jsonl
    ├── ... (4 families × 5 seeds = 20 expert traces)
    └── postmortems.jsonl      — append-only failure memory (Phase 4)
```

## Why the expert exists

Three reasons:

1. **Upper-bound reference** for evaluation. If a trained model beats the
   expert, something is wrong with the scenarios or the scorer. If it scores
   ~the same, training has converged. If it scores well below, more training
   is needed.

2. **SFT seed data**. `capture_everything.py` runs the expert across many
   seeds and dumps (obs, action) pairs as a chat-formatted JSONL for the
   stage-1 SFT warm-start.

3. **Scenario validation**. `tests/smoke_all_scenarios.py` runs the expert
   to confirm each scenario is actually solvable at ≥0.85. This is the canary
   that catches broken scenarios before training.

## Trajectory format

Each `.jsonl` line is one episode step:

```json
{
  "step": 1,
  "lap": 1,
  "obs": {...F1Observation as dict...},
  "action": "INSPECT_TYRE_DEGRADATION",
  "reward": 0.02,
  "done": false
}
```

Final line of each episode has `done: true` and includes a `final_scores` dict.

## Postmortems

After Phase 4 lands, `postmortems.jsonl` is the running record of every failed
episode the env has seen. Schema in `docs/architecture.md` §postmortem.
