---
title: F1 Strategist
emoji: 🏎️
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 8000
pinned: true
license: mit
short_description: LLM race strategist trained with GRPO on F1 strategy
tags:
  - openenv
  - reinforcement-learning
  - grpo
  - long-horizon
  - world-model
  - formula-one
  - strategy
  - qwen3
models:
  - Deltasthic/f1-strategist-qwen3-4b-grpo
---

<!-- The YAML block above is HuggingFace Spaces metadata. Do not move it. -->

# F1 Strategist

**OpenEnv environment for training LLM agents as Formula 1 race strategists.**

> **Try it live:**
> [`HF Space`](https://huggingface.co/spaces/Deltasthic/f1-strategist) — landing page + interactive Gradio panel
> · [`f1.chinnaboina.com`](https://f1.chinnaboina.com/) — live deployment
> · [`GitHub`](https://github.com/Deltasthicc/F1_Simulator_OpenENV) — full source

## Judge quick-links

| Resource | Link |
|---|---|
| HF Space (live environment) | [Deltasthic/f1-strategist](https://huggingface.co/spaces/Deltasthic/f1-strategist) |
| Interactive Gradio playground | [/web](https://huggingface.co/spaces/Deltasthic/f1-strategist) → click **Interactive** |
| Blog post (writeup) | [blog.md](https://huggingface.co/spaces/Deltasthic/f1-strategist/blob/main/blog.md) |
| Training notebook (Colab) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deltasthicc/F1_Simulator_OpenENV/blob/main/notebooks/f1_strategist_training_colab.ipynb) |
| Eval results (PNG) | [eval_curve.png](https://huggingface.co/spaces/Deltasthic/f1-strategist/blob/main/results/eval_curve.png) |
| Model weights (LoRA) | [Deltasthic/f1-strategist-qwen3-4b-grpo](https://huggingface.co/Deltasthic) |
| GitHub | [Deltasthicc/F1\_Simulator\_OpenENV](https://github.com/Deltasthicc/F1_Simulator_OpenENV) |

## Training evidence

Real run results — RTX 5090, 800 steps, Qwen3-4B LoRA.

![Eval curve — before vs after GRPO across 6 scenario families](https://raw.githubusercontent.com/Deltasthicc/F1_Simulator_OpenENV/main/results/eval_curve.png)

![GRPO training reward curve — 800 steps, peak 0.907 @ step 480](https://raw.githubusercontent.com/Deltasthicc/F1_Simulator_OpenENV/main/results/training_loss_curve.png)

![Race story — Weather Roulette Spa, untrained vs trained, position / tyre / score breakdown](https://raw.githubusercontent.com/Deltasthicc/F1_Simulator_OpenENV/main/results/race_story.png)

It is lap 8 of 12 at Spa. Light rain has started. Verstappen ahead of you just stayed out on slicks. Russell behind you boxed two laps ago for inters and is setting purple sectors. Your tyres have 4 laps of grip left. Your pit window closes in 3. The board on the pit wall is asking what to call. As the strategist on the radio you have to read the weather, the field, your own car, and call it.

F1 Strategist simulates that world. It is an OpenEnv environment designed for RL post-training of LLM agents on long-horizon, partially-observable race-strategy decisions with hard rule constraints and verifiable rewards. The model does not drive the car — a built-in physics simulator handles laps, tyres, fuel, and opponents. The model is the race engineer making strategic calls.

This repository was built for the **Meta PyTorch OpenEnv Hackathon Grand Finale (Bangalore, April 25–26, 2026)**. It targets **Theme #2 — (Super) Long-Horizon Planning & Instruction Following** as the primary track, with a secondary fit for **Theme #3.1 — Professional Tasks**: a single episode spans tyre management, fuel budgeting, weather forecasting, opponent modelling, pit-window timing, and team radio comms.

## Why this environment

Race strategy is a strong fit for LLM RL specifically because:

**It is verifiable but not trivially scriptable.** Final position, tyre/fuel rule compliance, and pit-window discipline are all programmable Python checks — no LLM judge anywhere in the reward path. But the optimal sequence depends on hidden state (opponent strategies, weather evolution, true tyre degradation curve) that the agent must investigate before deciding.

**It is genuinely long-horizon.** A 12-lap sprint is 12 strategic decision points across ~25 minutes of simulated race time. Mistakes in lap 4 only show their cost in lap 9. This is the failure mode that current LLM agents handle worst, and it is exactly what Theme #2 asks for.

**It is multi-objective without being subjective.** Six independent reward dimensions (race result, tyre management, fuel discipline, strategic timing, comms quality, operational efficiency) combine into a deterministic weighted scalar. The model trains against a stable signal, judges read a coherent breakdown.

**It is dramatic.** F1 strategy is what every fan in the paddock argues about. The trained-vs-untrained demo writes itself: *"the trained model called the rain pit one lap before the field and gained three positions; the untrained model stayed out and finished last."*

## Three design choices that matter

**Hidden state must drive decisions.** Each episode has a latent opponent stint plan, a pre-rolled weather and safety-car schedule, a true tyre degradation curve, and a fuel-burn-vs-displayed delta. The agent reveals them through `INSPECT_TYRE_DEGRADATION`, `CHECK_OPPONENT_STRATEGY`, `REQUEST_FORECAST`, `ASSESS_UNDERCUT_WINDOW`, and `INSPECT_FUEL_MARGIN`. Pitting before checking the undercut window is a policy violation even if it happens to win the race. This is the same primitive that worked in our Round-1 OpsTwin submission and we are reusing it deliberately.

**Rewards are code, not an LLM judge.** Six scoring dimensions combine into a final scalar in `[0.01, 0.99]`. Per-step shaped rewards land for issue resolutions and inspection revelations; a final weighted score lands at race end. Every value comes from deterministic Python on the audit trail.

**Strategic actions only — the LLM does not drive.** Throttle and steering are run by an internal physics model. The LLM emits one of ~20 strategic commands per step (PIT_NOW, SET_MODE, DEFEND_POSITION, RADIO_DRIVER, etc.). This is the right abstraction layer for an LLM, plays to GRPO's strengths, and makes the inference loop fast enough for rollouts.

## How it's built

```
f1-strategist/
├── models.py                      F1Action / F1Observation / F1State (Pydantic)
├── client.py                      EnvClient for HTTP or Docker connections
├── inference.py                   Baseline LLM agent runner
├── train.py                       TRL GRPO training (Qwen3-4B + LoRA + Unsloth)
├── evaluate.py                    Held-out seed evaluation with bar chart
├── rollout.py                     Single-rollout helper
├── capture_everything.py          Bulk rollout capture for SFT seed data
├── server/
│   ├── app.py                     FastAPI server, shared env singleton
│   ├── environment.py             Core reset/step/_obs/_load loop
│   ├── physics.py                 Tyre wear + fuel burn + lap-time model
│   ├── track.py                   Track loader (racetrack-database CSVs)
│   ├── opponents.py               Rule-based opponent strategies
│   ├── weather.py                 Pre-rolled weather + SC events
│   ├── scenarios.py               Three hand-authored families + stretch
│   ├── generator.py               Procedural scenarios (seed-deterministic)
│   ├── hidden_state.py            Latent truth revealed by INSPECT actions
│   ├── scoring.py                 Six-dimension pure reward functions
│   ├── postmortem.py              Episode memory for self-improvement loop
│   └── visualizer.py              Matplotlib top-down replay → GIF/MP4
├── data/
│   ├── tracks/                    Selected racetrack-database CSVs
│   ├── opponent_pace_calibration.json
│   └── tyre_compound_baseline.json
├── baselines/
│   ├── expert_solver.py           Gold strategy generator
│   └── trajectories/              .jsonl expert traces
└── docs/                          Architecture, build order, reward, scenarios, person1/2
```

The skeleton is a direct architectural transplant of our Round-1 `OpsTwin Recovery Arena` codebase: same `reset / step / _obs / _exec / _load` shape, same hidden-state reveal pattern, same six-dimension scoring stack, same postmortem self-improvement loop. The domain changes; the loop does not.

## Scenarios

Three hand-authored families cover the strategy taxonomy. Each is deterministically solvable, stress-tested for reward-hackability, and has a rule-based expert solver scoring ≥ 0.92.

**Family 1 — Dry Strategy Sprint (Monza).** A 10-lap one-stop window. The obvious strategy is to extend the medium stint and pit to softs at the end. The trap is opponent undercut: a rival who pits two laps earlier on fresh softs gains 1.2 s/lap, and the agent has only 2 laps before they emerge ahead. Optimal sequence requires `ASSESS_UNDERCUT_WINDOW` plus `CHECK_OPPONENT_STRATEGY` before the pit call. Greedy "stay out longest" caps at ~0.55.

**Family 2 — Weather Roulette (Spa).** A 12-lap mixed-conditions race. Light rain begins between laps 5–8 (pre-rolled, agent sees only the forecast). The fast strategy is to pit for inters one lap before the rain peak; pitting one lap early wastes a stop, pitting one lap late loses 30 s on slicks in standing water. `REQUEST_FORECAST` reveals a 5-lap weather window. The trap is forecast over-confidence: the forecast is a probability cone, not a guarantee.

**Family 3 — Late Safety Car Lottery (Monaco).** A 12-lap race with a 35% safety-car probability at laps 7–10. A pit under SC costs ~8 s; a green-flag pit costs ~22 s. Free pit windows are brief. The trap is committing to a one-stop too early — if the SC fires, the agent who stayed out gets a free stop and the agent who pre-pitted is now on stale tyres with no leverage. Optimal play is `HOLD_GAP` until the SC window passes.

Plus a stretch scenario, **Championship Decider (Catalunya, mixed dry-then-rain, 15 laps)**, where the agent must defend a single specific rival for points while managing a tyre crossover. Aggressive pace tanks tyres before the rain change; passive pace cedes the championship position.

The procedural generator in `server/generator.py` produces seed-deterministic variants of all three families, guaranteed solvable, with issue points summing to 1.0.

## Reward structure

Every scenario has the same point budget (1.0 total) distributed across six issue categories: race result, tyre management, fuel management, strategic decisions, comms quality, operational efficiency. The per-step reward is the points value of whichever issue that action resolves, minus penalties for invalid or harmful commands.

Additional signals:
- Inspection actions (`INSPECT_TYRE_DEGRADATION`, `CHECK_OPPONENT_STRATEGY`, `REQUEST_FORECAST`, `ASSESS_UNDERCUT_WINDOW`, `INSPECT_FUEL_MARGIN`) pay out `+0.02` the first time they reveal meaningful hidden state.
- Switching drive mode pays a small shaping bonus when the new mode matches the current race phase (push at start, conserve mid-stint, push final laps).
- Finishing without running out of fuel and without DNF pays the full operational-efficiency dimension.
- Invalid commands pay `-0.02`. Harmful actions (panic-pit, unnecessary defend that costs 4+ seconds) pay `-0.05`.

The displayed final score is the weighted combination of the six dimensions, clamped to `[0.01, 0.99]`. This is what judges see. The per-step reward is the signal the RL trainer optimizes. They reinforce each other but are not identical. See [`docs/reward-model.md`](docs/reward-model.md) for the full formula stack.

## Training evidence

Run the verified local evaluation to reproduce the baseline comparison chart at
`results/eval_curve.png`:

```bash
python evaluate.py \
  --model grpo_smoke \
  --tasks dry_strategy_sprint weather_roulette late_safety_car championship_decider \
  --n-seeds 2 \
  --modes random untrained trained expert
```

Held-out eval numbers (real LLM forward pass through env, 5 seeds × 6 scenarios):

| scenario | random | untrained Qwen3-4B | **SFT+GRPO (grpo_v2)** | expert ceiling |
|---|---:|---:|---:|---:|
| dry_strategy_sprint | 0.400 | 0.509 | **0.520** | 0.840 |
| weather_roulette | 0.344 | 0.414 | **0.965** | 0.950 |
| late_safety_car | 0.332 | 0.525 | **0.645** | 0.935 |
| championship_decider | 0.209 | 0.269 | **0.555** | 0.965 |
| virtual_safety_car_window | 0.329 | 0.375 | **0.471** | 0.965 |
| tyre_cliff_management | 0.204 | 0.398 | **0.552** | 0.965 |
| **average** | **0.303** | **0.415** | **0.618** | **0.937** |

Trained model lifts average **+0.20** over untrained Qwen3-4B and closes ~33% of the random→expert gap. Biggest single win: weather scenario where the trained model (0.965) actually edges out the rule-based expert (0.950) — investigation discipline pays off. Honest weakness: dry sprint stays at 0.52 (model rarely picks `PIT_NOW` — rare-event learning gap; see [`blog.md`](blog.md) for the full journey).

`train.py` has two paths:

- `--backend local-smoke` is the default and is verified in this repo. It exercises
  the environment, writes TRL-shaped checkpoint artifacts, and produces
  `results/training_loss_curve.png`.
- `--backend trl` is the GPU training scaffold for the RTX 5090 handoff. It should
  be launched only after installing the training stack and setting `HF_TOKEN` in
  `.env`.

```bash
# Verified local smoke run
python train.py --model heuristic --task multi --max-steps 12 --output-dir grpo_smoke

# GPU handoff run, after installing TRAINING.md dependencies on the 5090 box
python train.py --backend trl --model Qwen/Qwen3-4B --task multi --max-steps 500 \
  --batch-size 1 --grad-accum 32 --output-dir grpo_v1
```

Full reproduction in [`TRAINING.md`](TRAINING.md). GPU server playbook in
[`GPU_HANDOFF.md`](GPU_HANDOFF.md).

## Visualisation

Two layers of demo:

1. **Local matplotlib replay.** `server/visualizer.py` renders any rollout as a top-down GIF/MP4: track centerline + corner labels + ego car + opponents + weather state + pit-window indicators. Generated automatically by `python rollout.py --render`.
2. **Gradio web UI in the HF Space.** Built into the OpenEnv server via the `--enable-interface` flag. Judges and visitors can hit the Space URL, click `Reset`, manually issue strategic commands, and watch the simulated race progress with a live per-lap chart. Enabled in `Dockerfile` by `ENABLE_WEB_INTERFACE=1`.

## Self-improvement loop

At the end of each episode, `server/postmortem.py` classifies the failure (missed undercut, late weather call, fuel underburn, ignored cascade alert, comms forgotten, panic pit, thrashing) and appends a structured postmortem to `baselines/trajectories/postmortems.jsonl`. On the next `reset()` the top-2 lowest-scoring past postmortems for the same scenario family are retrieved and injected into the initial observation as memory hints. This closes the loop without requiring vector databases or embedding infrastructure.

```json
{
  "scenario_family": "weather_roulette",
  "failure_category": "late_weather_call",
  "first_bad_action": "STAY_OUT lap 7",
  "missed_signal": "REQUEST_FORECAST never called; rain probability was 0.78 at lap 8",
  "preferred_intervention_order": ["REQUEST_FORECAST", "INSPECT_TYRE_DEGRADATION", "PIT_NOW inter"],
  "final_score": 0.32
}
```

This mirrors how a real race engineer hands context to the next event: write down what happened, what was tried, what actually worked.

## Running the environment

```bash
# Local server
uv run server
# or
python -m server.app

# Then in a separate terminal
python inference.py --task dry_strategy_sprint --model Qwen/Qwen3-0.6B
```

Set `HF_TOKEN` and (optionally) `IMAGE_NAME` to route inference through the docker container instead of local uvicorn. To enable the Gradio interface set `ENABLE_WEB_INTERFACE=1` before starting.

## Theme alignment

**Theme #2 — (Super) Long-Horizon Planning & Instruction Following.** A 12–15 lap race is 12–15 strategic decision points spanning ~25–35 minutes of simulated time, where lap-4 mistakes only manifest in lap-9 outcomes. Sparse outcome reward + dense shaping reward + delayed credit pattern (rollback rewards finalise only after `ASSESS_UNDERCUT_WINDOW` is called) give exactly the long-horizon structure the theme asks for.

**Theme #3.1 — Professional Tasks (secondary fit).** A single episode requires the agent to coordinate state across at least four "surfaces": tyre management, fuel budgeting, opponent strategy, weather forecasting, and team radio comms. The hidden-state layer models the kind of non-obvious cross-surface dependencies that trip up current professional-task agents.

## Demo assets

- Blog post: [`blog.md`](https://huggingface.co/spaces/Deltasthic/f1-strategist/blob/main/blog.md)
- HF Space: [`Deltasthic/f1-strategist`](https://huggingface.co/spaces/Deltasthic/f1-strategist)
- Model weights: [`Deltasthic/f1-strategist-qwen3-4b-grpo`](https://huggingface.co/Deltasthic/f1-strategist-qwen3-4b-grpo)
- Training notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deltasthicc/F1_Simulator_OpenENV/blob/main/notebooks/f1_strategist_training_colab.ipynb)

## Authors

Shashwat Rajan and Tanish Shitanshu, for the Meta PyTorch OpenEnv Hackathon Grand Finale 2026.

Work split documented in [`docs/person1-tasks.md`](docs/person1-tasks.md) and [`docs/person2-tasks.md`](docs/person2-tasks.md).

## Acknowledgments

Architectural primitives (hidden-state reveal, multi-objective scoring, postmortem memory, scenario-family scaffolding) are ported from our Round-1 submission `OpsTwin Recovery Arena`, which qualified us for the finale. Track data sourced from the open [racetrack-database](https://github.com/TUMFTM/racetrack-database) (MIT). Historical pace-and-stint calibration from the Kaggle Formula 1 World Championship dataset (1950–2024). Thanks to the Meta PyTorch, Hugging Face, Unsloth, and Scaler AI Labs teams for organising the hackathon.
