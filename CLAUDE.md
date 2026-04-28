# F1 Strategist

<!-- Maintainer note: keep this file under 150 lines. Detailed specs live in docs/. -->

## What This Is

LLM race-strategist OpenEnv environment for the **Meta PyTorch OpenEnv Hackathon Grand
Finale (April 25–26, 2026, Bangalore)**. Domain transplant of `OpsTwin Recovery Arena` —
keep the architecture, change the nouns. Reference the OpsTwin codebase when porting logic.

**Team:** Shashwat Rajan + Tanish Shitanshu
**Repo:** `f1-strategist` under HF org `Deltasthic` (rename if you want different)
**Theme:** #2 (Super) Long-Horizon Planning + secondary #3.1 Professional Tasks

## The One Thing That Matters Most

**There is zero training evidence right now. A GRPO reward curve must exist before demo
time.** Do not add features before Phase 3 (training) is complete. See
[`docs/build-order.md`](docs/build-order.md).

## Judging Weights

| Criterion | Weight |
|-----------|--------|
| Environment Innovation | 40% |
| Storytelling | 30% |
| Reward Improvement Evidence | 20% |
| Training Pipeline | 10% |

## Repository Layout

```
f1-strategist/
├── CLAUDE.md                  ← this file (always loaded)
├── models.py                  ← F1Action, F1Observation, F1State (Pydantic)
├── client.py                  ← OpenEnv client
├── inference.py               ← baseline inference runner
├── train.py                   ← TRL/GRPO script — CRITICAL DELIVERABLE
├── evaluate.py                ← held-out seed evaluator
├── server/
│   ├── app.py                 ← FastAPI (port from OpsTwin)
│   ├── environment.py         ← reset/step/_obs loop — MAIN FILE
│   ├── physics.py             ← tyre / fuel / lap-time model
│   ├── track.py               ← Track loader + corner segmentation
│   ├── opponents.py           ← rule-based opponent strategies
│   ├── weather.py             ← pre-rolled weather + safety-car events
│   ├── scenarios.py           ← three hand-authored families + stretch
│   ├── generator.py           ← procedural generator (seed-deterministic)
│   ├── hidden_state.py        ← HiddenStateLayer (port from OpsTwin)
│   ├── scoring.py             ← 6-dim scorer (port from OpsTwin)
│   ├── postmortem.py          ← PostmortemMemory (port from OpsTwin)
│   └── visualizer.py          ← matplotlib replay → GIF/MP4
├── data/
│   ├── tracks/                ← racetrack-database CSVs
│   ├── opponent_pace_calibration.json
│   └── tyre_compound_baseline.json
├── baselines/
│   ├── expert_solver.py       ← rule-based gold strategy generator
│   └── trajectories/          ← .jsonl expert traces
├── docs/
│   ├── architecture.md        ← desks, hidden state, action ref
│   ├── build-order.md         ← PHASE-BY-PHASE BUILD PLAN
│   ├── scenarios.md           ← scenario family specs
│   ├── reward-model.md        ← 6-dim scoring spec
│   ├── physics-model.md       ← tyre/fuel/laptime equations
│   ├── person1-tasks.md       ← Shashwat queue
│   └── person2-tasks.md       ← Tanish queue
└── notebooks/
    └── f1_strategist_training_colab.ipynb
```

## Code Conventions

- Python 3.12, `uv` for deps
- Pydantic for all action / observation / state types — no raw dicts at API boundary
- `copy.deepcopy(sc)` on every scenario load — never mutate templates
- Reward values: positive for correct resolutions, `-0.02` for invalid commands,
  `-0.05` for actively harmful actions
- `_audit_trail`: append `{step, action, reward, resolved_count}` every step
- All scoring functions must be **pure** — no side effects, no LLM calls
- `SUPPORTS_CONCURRENT_SESSIONS = True` on environment class
- Keep `LAPS_PER_STEP = 1` and a per-lap clock simulation (each step = one strategic
  decision point, normally one lap, but during pit/SC events can be a partial lap)

## What NOT to Build

- No real F1 game integration (no F1 23/24, no rFactor, no Assetto Corsa)
- No live OpenF1 / FastF1 calls during rollouts — bake calibration into `data/` JSON
- No LLM-judged rewards — everything deterministic
- No driving control (throttle/steering) — physics handles it, model is strategist only
- Don't expand to >5 tracks before training works

## Bonus Theme Alignment

**Theme #2 — Long-Horizon.** 12–15 lap races, sparse final-position reward + dense
shaping reward + delayed-credit pattern (pit reward finalises only after the cascade
of opponents' subsequent stints resolves). Lap-4 mistakes manifest in lap-9 outcomes —
exactly what the theme asks for.

**Theme #3.1 — Professional Tasks.** Multi-surface coordination across tyre management,
fuel budgeting, opponent modelling, weather forecasting, and team radio comms.

## When You Need Details

- Build phases and task order → [`docs/build-order.md`](docs/build-order.md)
- Hidden state, action ref, observation schema → [`docs/architecture.md`](docs/architecture.md)
- Scenario family specs → [`docs/scenarios.md`](docs/scenarios.md)
- Scoring dimensions and weights → [`docs/reward-model.md`](docs/reward-model.md)
- Physics equations and calibration → [`docs/physics-model.md`](docs/physics-model.md)
- Per-person work queues → [`docs/person1-tasks.md`](docs/person1-tasks.md), [`docs/person2-tasks.md`](docs/person2-tasks.md)
