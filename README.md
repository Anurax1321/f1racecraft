# F1Racecraft

> A personal project to simulate a complete Formula 1 race where every car is a language model with its own personality, on real F1 data, watched live with commentary.

**Live preview:** [f1.chinnaboina.com/f1_LLM_racing](https://f1.chinnaboina.com/f1_LLM_racing)
**Roadmap:** [`ROADMAP.md`](ROADMAP.md) — what gets built, in what order
**Current state:** [`STATUS.md`](STATUS.md) — exact cursor at any moment

---

## What this is

It's lap 8 of 12 at Spa. Light rain just started. Verstappen ahead of you stayed out on slicks. Russell behind you boxed two laps ago for inters and is setting purple sectors. Your tyres have 4 laps of grip left. The pit window closes in 3. The pit wall is asking what to call.

**F1Racecraft simulates that.** The model is the strategist on the radio — not the driver. A built-in physics engine handles the laps; the LLM reads weather, tyres, fuel, opponents, and decides what to do. Pit now? Stay out? Push or save? It's deterministic, replayable, and trained with reinforcement learning to actually get good at it.

The vision: scale this from a single car (where it lives today) to all 20, with real F1 data, full grand prix length, and a live UI you can watch with commentary.

## Status

Hackathon shipped April 2026 — single-car strategist trained with GRPO, scoring 0.618 weighted average across six scenarios (untrained 0.415, expert 0.925). Now a personal long-running build. The plan is in [`ROADMAP.md`](ROADMAP.md).

The hackathon-era model is on HuggingFace Hub:
[`Deltasthic/f1-strategist-qwen3-4b-grpo`](https://huggingface.co/Deltasthic/f1-strategist-qwen3-4b-grpo).

## What's here

```
server/      FastAPI environment — reset/step loop, physics, scoring, scenarios
models.py    Pydantic types for actions and observations
train.py     GRPO training (Qwen3-4B + LoRA + Unsloth)
evaluate.py  Held-out seed evaluation
baselines/   Expert solver + frozen trajectories
data/        Track CSVs, calibration JSONs
docs/        Architecture, scenarios, reward model, physics
notebooks/   Colab training notebook
```

## Run it locally

```bash
uv sync
./scripts/serve.sh        # dev server, auto-reload, http://127.0.0.1:8765
./scripts/race.sh         # one race, default scenario, heuristic policy
./scripts/eval.sh         # full eval against the trained champion
./scripts/train.sh smoke  # CPU dry-run (no GPU); use `grpo` for the real thing
```

Visit [`http://127.0.0.1:8765/f1_LLM_racing`](http://127.0.0.1:8765/f1_LLM_racing) to see the project page locally.

## Eval results

Six scenario families, five seeds each, weighted-final score:

| Scenario | Random | Untrained | **Trained** | Expert |
|---|---:|---:|---:|---:|
| dry_strategy_sprint | 0.40 | 0.51 | **0.52** | 0.84 |
| weather_roulette | 0.34 | 0.41 | **0.97** | 0.95 |
| late_safety_car | 0.33 | 0.53 | **0.65** | 0.94 |
| championship_decider | 0.21 | 0.27 | **0.56** | 0.97 |
| virtual_safety_car_window | 0.33 | 0.38 | **0.47** | 0.97 |
| tyre_cliff_management | 0.20 | 0.40 | **0.55** | 0.97 |
| **Average** | 0.30 | 0.42 | **0.62** | 0.94 |

Trained beats untrained by **+0.20**, closes ~33% of the gap to a hand-coded expert. On weather, the trained model edges past the expert (0.965 vs 0.950). The full journey — including five real bugs caught and fixed — is in [`blog.md`](blog.md).

## How the model learns (short version)

Three stages, like training a new race engineer:

1. **Imitate** (SFT) — show it expert traces; it picks up the command vocabulary.
2. **Trial and reward** (GRPO) — let it race thousands of simulated races, score the outcomes, reinforce what worked.
3. **Race each other** (self-play, future) — multiple personas co-evolve.

The reward is pure deterministic Python — no LLM judge, no subjective grading. Six dimensions: race result, strategic decisions, tyre management, fuel, comms quality, operational efficiency.

## Where it's headed

- **Full grand prix length** (50–70 laps), not just 12-lap sprints
- **All 20 cars LLM-driven**, with five distinct personas (Aggressor, Cautious, Veteran, Rookie, Wildcard)
- **Real F1 data** — tyre curves, fuel burn, weather all calibrated against 2023–2025 telemetry
- **Continual learning** — every race becomes training signal
- **Live UI with commentary** — watch it happen, with a commentator LLM narrating

Full plan in [`ROADMAP.md`](ROADMAP.md). Eleven weeks part-time, ten phases.

## Authors

**Anurag Chinnaboina**, Shashwat Rajan.

Originally built for the Meta PyTorch OpenEnv Hackathon Grand Finale (April 2026). Now a personal project led by Anurag.

## License

MIT. See [`LICENSE`](LICENSE).

## References

Reward design and long-horizon RL choices in this project draw on:

- **Thomas, D. et al.** *Explainable Reinforcement Learning for Formula One Race Strategy.* ACM SIGAPP 2025. [arXiv:2501.04068](https://arxiv.org/abs/2501.04068) — directly comparable F1 race-strategy RL system. Their explicit rejection of dense reward shaping ("we are unable to determine midway through a race whether a decision was good or not") informs our [reward philosophy](docs/reward-philosophy.md).

- **HCAPO: Hindsight Credit Assignment for Long-Horizon LLM Agents.** [arXiv:2603.08754](https://arxiv.org/abs/2603.08754), 2026 — current SOTA for long-horizon LLM agents (sparse terminal reward + hindsight reweighting). Earmarked as the upgrade path for Phase 3 if 14B GRPO plateaus on long races. Beats vanilla GRPO by +13.8% on ALFWorld and +7.7% on WebShop.

- **GRPO-λ.** [arXiv:2510.00194](https://arxiv.org/abs/2510.00194), 2025 — credit-assignment improvement to GRPO via eligibility-trace λ-returns.

- **GTPO / GRPO-S: Token and Sequence-Level Reward Shaping with Policy Entropy.** [arXiv:2508.04349](https://arxiv.org/html/2508.04349v6), 2025 — entropy-weighted reward shaping for GRPO; alternative to HCAPO if hindsight reweighting underdelivers.

- **Anthropic.** *Natural Emergent Misalignment from Reward Hacking in Production RL.* 2025. [PDF](https://assets.anthropic.com/m/74342f2c96095771/original/Natural-emergent-misalignment-from-reward-hacking-paper.pdf) — informs our "immunity by construction" stance (one-shot revelation rewards, capped per-episode shaping).

- **Process Reward Models — A Survey.** [arXiv:2510.08049](https://arxiv.org/abs/2510.08049), 2025 — considered and rejected for race strategy (see [reward-philosophy.md](docs/reward-philosophy.md)) because individual race-strategy decisions can't be graded in isolation.

## Acknowledgments

Architectural primitives (hidden-state reveal, multi-objective scoring, postmortem memory) ported from `OpsTwin Recovery Arena`. Track data from the open [racetrack-database](https://github.com/TUMFTM/racetrack-database) (MIT). Historical calibration from the Kaggle F1 World Championship dataset. Thanks to the Meta PyTorch, Hugging Face, and Unsloth teams for the original hackathon environment.
