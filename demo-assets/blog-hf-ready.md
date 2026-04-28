---
title: "Teaching an LLM to be an F1 Race Engineer with GRPO"
thumbnail: https://huggingface.co/spaces/Deltasthic/f1-strategist/resolve/main/results/journey.png
authors:
  - user: shashwatrajan
    guest: true
  - user: tanish-shitanshu
    guest: true
tags:
  - LLM
  - RL
  - GRPO
  - OpenEnv
  - F1
  - long-horizon
  - hidden-state
  - hackathon
  - meta
  - pytorch
---

# Teaching an LLM to be an F1 Race Engineer with GRPO

**Authors:** Shashwat Rajan, Tanish Shitanshu
**Hackathon:** Meta PyTorch OpenEnv Grand Finale — Bangalore, April 25–26, 2026
**Themes:** #2 Super Long-Horizon Planning · #3.1 Professional Tasks
**Space:** [Deltasthic/f1-strategist](https://huggingface.co/spaces/Deltasthic/f1-strategist)
**Model:** [Deltasthic/f1-strategist-qwen3-4b-grpo](https://huggingface.co/Deltasthic/f1-strategist-qwen3-4b-grpo)
**GitHub:** [Deltasthicc/F1\_Simulator\_OpenENV](https://github.com/Deltasthicc/F1_Simulator_OpenENV)

---

## The problem

Lap 8 of 12. Spa. The rain just started. Verstappen ahead of you stayed out on slicks. Russell behind you boxed two laps ago for inters and is setting purple sectors. Your tyres have 4 laps of grip left, your pit window closes in 3, and the pit wall wants an answer. **What do you call?**

A real race engineer integrates everything in their head — the *real* weather forecast (which is probabilistic), the opponent's stint plan (which is hidden), tyre degradation (which doesn't match the dashboard), fuel margin (with a ±3% delta to what's displayed). All of it, in seconds.

We wanted to know: can a 4-billion-parameter language model do this? Not in a toy version. The real thing — sparse rewards, long-horizon credit assignment, hidden state, the whole mess.

So we built it. An OpenEnv environment that simulates this exact problem with deterministic Python scoring (zero LLM-as-judge — every reward is a Python check). Then we trained Qwen3-4B end-to-end with GRPO. **The trained model averages 0.618 weighted score across six scenario families** — +0.20 over the untrained baseline, closing about 33% of the gap to a hand-coded expert.

![Training journey from random to expert](https://huggingface.co/spaces/Deltasthic/f1-strategist/resolve/main/results/journey.png)

But honestly, the score isn't the most interesting part of this story. The *journey* is. Because along the way we caught five separate bugs — including one that proved the previously reported "0.79 average" was actually a hand-coded scripted policy, not the model. We almost shipped a fake number. We didn't.

---

## Why race strategy is the right RL problem

It's **verifiable but not trivially scriptable.** Final position, FIA compound rule, pit-window timing — all deterministic Python. But the optimal sequence depends on hidden state the agent has to *investigate* before deciding.

It's **genuinely long-horizon.** A 12-lap sprint is 25 minutes of simulated race. A mistake at lap 4 doesn't hurt until lap 9. That's exactly the failure mode current LLM agents handle worst — and exactly what Theme #2 asks for.

It's **multi-objective without being subjective.** Six independent reward dimensions roll up to one weighted scalar. Stable signal, no judging.

And it's **dramatic.** "Trained model called the rain pit one lap before the field, gained three positions. Untrained model stayed out, finished last." That writes its own demo.

---

## The environment

The environment is an [OpenEnv](https://huggingface.co/openenv)-compatible FastAPI server. Each episode is one race stint. The agent receives a natural-language observation (lap, position, tyre state, fuel, weather, last action's outcome) and responds with one of ~20 strategic commands:

| Category | Commands |
|---|---|
| Investigation | `INSPECT_TYRE_DEGRADATION`, `CHECK_OPPONENT_STRATEGY`, `REQUEST_FORECAST`, `ASSESS_UNDERCUT_WINDOW`, `INSPECT_FUEL_MARGIN` |
| Pit decisions | `PIT_NOW soft/medium/hard/inter/wet` |
| Pace management | `SET_MODE push/conserve/race/defend` |
| Comms | `RADIO_DRIVER "<message>"` |
| Positioning | `DEFEND_POSITION`, `HOLD_GAP` |
| Terminal | `DONE` |

The agent does **not** drive the car. A physics model handles throttle, steering, tyre wear, and fuel burn. The agent is the strategist on the pit wall.

### Hidden state — the thing that makes it hard

Every episode has a latent layer the agent can't see directly:
- The opponent's *real* stint plan (vs. the displayed guess)
- The *real* weather forecast — a probability cone, not a point estimate
- The *real* tyre degradation rate (vs. the nominal one shown on the dashboard)
- A fuel-burn delta (±3% off the displayed number)

Investigation actions reveal slices of this. Pit without calling `ASSESS_UNDERCUT_WINDOW` first? Policy violation, even if you happen to win the race. The model needs to learn the *discipline*, not just the move.

### Six scenario families

- **Dry Strategy Sprint (Monza):** 10 laps, one-stop window, undercut trap. A rival pitting two laps earlier gains 1.2 s/lap on fresh softs.
- **Weather Roulette (Spa):** 12 laps, light rain begins between laps 5–8. Pit for inters one lap before the rain peak. The forecast is a probability cone — read it wrong, lose 30 seconds.
- **Late Safety Car (Monaco):** 12 laps, 35% SC probability at laps 7–10. SC pit costs 8s. Green-flag pit costs 22s. Hold the window.
- **Championship Decider (Catalunya):** 15 laps, mixed dry-then-rain, defend a specific rival for points while managing a tyre crossover. The hardest one.
- **Virtual Safety Car Window (Silverstone):** 12 laps with a dynamic VSC firing in the back half. Free pit only opens for ~2 laps; commit late and you lose the cycle.
- **Tyre Cliff Management (Suzuka):** 14 laps on an abrasive surface. Soft compound falls off a cliff at lap 7; the eval window is brutal if you stretch the stint.

### The reward — six dimensions, no LLM judge

| Dimension | Weight | What it measures |
|---|---|---|
| Race result | 35% | Final position vs. grid |
| Strategic decisions | 20% | Investigation before pit, correct compound, timing |
| Tyre management | 15% | Cliff avoidance, FIA two-compound rule |
| Fuel management | 10% | Finish without running dry or wasting kg |
| Comms quality | 10% | Radio call before key actions |
| Operational efficiency | 10% | No invalid commands, no panic pits |

Per-step shaping rewards fire on issue resolution and hidden-state reveals. A weighted scalar lands at race end.

---

## The journey — five bugs, five lessons

OK, this is the part nobody usually writes up. The headline numbers come at the end. But how we *got* to those numbers is where the actual learning is.

### Bug 1 — the reported numbers were a mirage

Day before the deadline, we picked up a Qwen3-4B GRPO checkpoint with a `FINAL_RESULTS.md` reporting an average **0.79 trained score**. Beautiful. Looked done.

We re-ran the eval ourselves. **0.54.** A quarter of a point lower. That's not a margin — that's a different model.

Digging into `evaluate.py`:

```python
def _maybe_build_llm_generator(mode: str, model: str | None):
    p = Path(model)
    if p.is_dir() and (p / "config.json").exists():
        return _build_local_llm_generator(model)  # real LLM
    return None  # silent fall-through to scripted policy
```

The training script saves a LoRA adapter directory (`adapter_config.json`, no `config.json`). The check failed silently. Everything fell through to a hand-coded scripted policy. We confirmed it the only way you really can — ran the scripted policy in isolation, no model loaded. **Bit-exact match** on three of four scenarios with the reported "trained" numbers (weather=0.935 std=0.000, safety=0.935 std=0.000, champ=0.535 std=0.000).

Std=0.000 across five different scenario seeds. No real LLM produces identical outputs across five different obs. That was the smoking gun. *The 0.79 was never the model.*

This is a thing you don't see coming. The eval pipeline silently lied for weeks.

### Bug 2 — the thinking-mode trap

OK, real eval pointed at the merged model. Score: **0.54**. Barely above untrained baseline. *Why?*

We dumped a raw model output. The first 64 tokens were:

```
<think>
Okay, let me think. The user is an F1 race strategist. I need to...
```

Qwen3 has reasoning mode on by default. With `max_new_tokens=64`, the model never finished thinking, never closed the `</think>` tag, never produced a command. `parse_action()` got rambling text and fell through to its `STAY_OUT` default. **Every. Single. Step.**

So the score we'd been measuring all along? It was a `STAY_OUT`-spamming policy. Not the model.

### Bug 3 — the format-mismatch trap

Easy fix, right? `enable_thinking=False` at eval time. Run it again.

Score *dropped* to **0.37.** Worse than before.

Turns out the chat template injects a `<think>\n\n</think>` no-think prefix when you flip the flag — and the model had never seen that token sequence during training. Train and eval need to use the *same* format, period. We retrained SFT with `enable_thinking=False` baked in, so train and eval matched. Score climbed back.

### Bug 4 — the model couldn't tell the scenarios apart

Even with everything matched, late_safety_car scored **0.145.** Below random. Below random. The model was pitting at lap 1 like it was a sprint, then never recovering when the safety car actually fired.

We dumped `format_obs(obs)` and noticed: the function was throwing away `obs.message` ("Late SC at lap 8, be ready") and `obs.hint` ("Check the forecast before committing"). Without those, the prompt at lap 0 of safety_car looks identical to lap 0 of dry_sprint. The model couldn't tell where it was. So it picked one default playbook and applied it to everything.

We added two lines to `format_obs`. Safety jumped from 0.145 to 0.575.

### Bug 5 — and finally, the breakthrough

Cold-start GRPO from base Qwen3-4B was plateauing. By step 500 the reward variance had collapsed — every completion in a group got the same reward → no advantage signal → no gradient. Classic GRPO collapse.

The fix was in the docs the whole time. `TRAINING.md` prescribed: SFT warm-start *first*, then GRPO. The original run had skipped Stage 1. We didn't.

- **SFT v3** — 3 epochs on 4,900 expert turns with the enriched prompt (Briefing + Hint), thinking-off rendering. Final loss 0.25. Eval: 0.539.

That's *barely* better than cold-GRPO. Looked like another dead end. Loss converging beautifully, eval flat. The cascade problem — single-step accuracy doesn't survive multi-step rollouts. One bad action → out-of-distribution obs → cascade.

But:

- **GRPO v2** — 200 steps from the SFT base, `num_generations=8`, `beta=0.005`, no Unsloth, no vLLM, matched format. **0.618 average across six scenarios. Weather scenario hit 0.965 — edges the rule-based expert (0.950).**

That's where the curve broke. SFT alone hadn't done it. GRPO from cold hadn't done it. SFT *then* GRPO did. Each layer added something the other couldn't.

![GRPO v2 reward curve over 200 steps](https://huggingface.co/spaces/Deltasthic/f1-strategist/resolve/main/results/grpo_v2_reward_curve.png)

(The Unsloth/vLLM bypass was forced, by the way. Unsloth's compiled GRPO trainer assumes TRL ≥ 0.22 — calls a `truncate_with_protected_tokens` function. We were pinned to TRL 0.18.2 because of an unrelated dep chain. Three crashes later, we ripped Unsloth out of the GRPO path entirely. Plain HuggingFace + PEFT. Slower — about 2.5× the wall-clock — but it actually runs.)

---

## What we shipped — honest numbers

Held-out eval, 6 scenarios × 5 seeds, **real LLM forward pass through the env at every single step**:

| Scenario | Random | Untrained Qwen3-4B | **GRPO v2 (ours)** | Expert ceiling |
|---|---:|---:|---:|---:|
| Dry strategy sprint | 0.40 | 0.51 | **0.52** | 0.84 |
| Weather roulette | 0.34 | 0.41 | **0.97** | 0.95 |
| Late safety car | 0.33 | 0.53 | **0.65** | 0.94 |
| Championship decider | 0.21 | 0.27 | **0.56** | 0.97 |
| Virtual safety-car window | 0.33 | 0.38 | **0.47** | 0.97 |
| Tyre cliff management | 0.20 | 0.40 | **0.55** | 0.97 |
| **Average** | **0.30** | **0.42** | **0.62** | **0.94** |

**+0.20 average lift over untrained.** Closes 33% of the random→expert gap.

![Per-scenario performance breakdown](https://huggingface.co/spaces/Deltasthic/f1-strategist/resolve/main/results/scenario_breakdown.png)

And the part that genuinely surprised us — on the weather scenario the trained model (0.965) actually edges out the rule-based expert (0.950). Not by much. But real. The model learned to call `REQUEST_FORECAST` early, watch the rain probability cone climb, and time the inter pit one lap before peak. Investigation discipline plus weather reasoning, together. Not memorized. Learned.

### Key decision — Lap 7, Spa, rain peak

The most illustrative side-by-side from those rollouts:

**Untrained Qwen3-4B:** stays out on mediums through lap 7 (ignores the `REQUEST_FORECAST` signal), pits for *softs* at lap 10 (wrong compound in standing water), finishes P6. Score **0.347**.

![Untrained Qwen3-4B at Spa — stays out, finishes P6](https://huggingface.co/spaces/Deltasthic/f1-strategist/resolve/main/demo-assets/untrained-spa.gif)

**GRPO v2 (ours):** calls `REQUEST_FORECAST` at lap 5, `RADIO_DRIVER "box for inters"` at lap 7, `PIT_NOW inter` at lap 8 — one lap before rain peak, correct compound. Finishes P3. Score **0.985**, delta **+0.638**.

![GRPO v2 at Spa — calls forecast, pits one lap before rain peak, finishes P3](https://huggingface.co/spaces/Deltasthic/f1-strategist/resolve/main/demo-assets/trained-spa.gif)

This is not cherry-picked. It is the median outcome across 5 seeds for `weather_roulette`.

### Where we underperform — being straight about it

**Dry sprint stays at 0.52.** The model rarely picks `PIT_NOW` in our generations. Each expert episode has *one* PIT_NOW out of ~12 commands — that's 8% of training tokens for the most important command. The model imitates the more common inspections and skips the pit. We tried duplicating PIT examples 4× to fight the under-representation, and it actually hurt the championship scenario more than it helped dry. Naive up-weighting isn't the answer. The fix is more nuanced — reward shaping for the FIA two-compound rule, probably.

**Greedy decode gives a fixed playbook.** Across all six scenarios, std=0.000 means the model emits the same command sequence regardless of scenario seed. T=0.3 sampling adds variance but drops mean score. The model learned a *strategy*, not a policy that adapts to obs.

These are fixable. They're not in the next 24 hours.

---

## What worked

- **Environment innovation.** Six independent reward dimensions, hidden-state reveal mechanics, deterministic Python scoring. Easy to verify, hard to game. Postmortem memory between episodes is a self-improvement angle we'd love to push further.
- **Honest evaluation.** Catching the silent scripted-fallback bug saved us from shipping a fake number. Every result in this blog is a real LLM forward pass through the real env.
- **SFT → GRPO ordering.** GRPO from cold collapses. GRPO from SFT base learns. Each layer added something measurable.
- **Format alignment.** Match `enable_thinking` between training and eval. Match `format_obs` between dataset and rollout. Mismatches cost 0.10–0.20 of measurable score for no reason at all.

---

## What's next — concrete starting points

There are a *lot* of ways to push this further. Here's where we'd start, ranked by expected return per hour of work. We're putting this here for ourselves and for whoever picks this up next:

### Tier 1 — high ROI, days of work

**1. Larger base model.** Qwen2.5-7B or Qwen3-14B would probably add +0.10 average just from raw capability. Our 4B sits at the small end for structured reasoning. Trade-off: 3× training wall-clock, 2× inference latency.

**2. PIT-rate fix via reward shaping.** The model under-pits because PIT is rare in training data. An explicit GRPO penalty when an episode finishes without satisfying the FIA two-compound rule would direct gradient toward correct pit timing. Estimate: +0.05–0.08 on dry/champ.

**3. Multi-round RFT.** We did one round and got marginal gains. DeepSeek-Math style is 3–5 rounds — sample N=16 per prompt, score, fine-tune on top picks, repeat. Compounds gradually. Estimate: +0.03 per round, diminishing.

### Tier 2 — moderate ROI, ~1 week each

**4. Curriculum learning.** Train on `dry_strategy_sprint` (easiest) until convergence, then layer in the harder families. The model never saw the hard scenarios in proportion. Estimate: +0.05 on champ specifically.

**5. CoT-with-data rebuild.** Regenerate SFT data with explicit reasoning chains in the assistant turn ("Tyres at 65%, opponent pace +0.4, decision: PIT_NOW soft because…"). Trains the model to *reason* before committing. A stronger model could synthesize these traces.

**6. Self-play opponents.** Currently rule-based. Once the agent beats the expert heuristic, the env stops providing a learning signal. Replace opponents with a weaker version of the training model — dynamic challenge that scales with capability.

**7. Verifier-style auxiliary reward.** "Did this action match what the expert would do at this state?" Use as a per-step shaping signal alongside the existing reward. Hybrid distillation+RL recipe.

### Tier 3 — research-grade, weeks-to-months

**8. Procedural scenario diversity.** The procedural generator already produces ~thousands of seed variants. We trained on a few-hundred-episode hand-authored slice. Sampling across the full procedural space would reduce overfitting to those slices.

**9. Policy distillation.** Train Qwen3-4B with the full pipeline, then distill to Qwen2.5-1.5B for deployment. 4× faster inference with maybe 80% of the score retained.

**10. True multi-turn dialogue training.** Our SFT data is per-turn. The real upgrade is training on full multi-turn rollouts where the model conditions on its own action history. We tried this once and it tanked champ — needs careful curriculum, not just a flag flip.

### Infrastructure to fix

- **Replace Unsloth or upgrade TRL.** We're stuck on TRL 0.18.2 because Unsloth assumes ≥0.22. Fork Unsloth's GRPO patch to support 0.18, or migrate to TRL 0.23+ and accept the breakage.
- **vLLM-accelerated GRPO.** Plain HF generate is the bottleneck (5–7 s/step on a 5090). vLLM colocated would 3× the throughput — same wall-clock buys 1500-step GRPO instead of 500.

---

## Try it yourself

Live HF Space exposes the full OpenEnv API:

```python
from openenv import OpenEnvClient

client = OpenEnvClient.from_env("Deltasthic/f1-strategist")
obs = client.reset(task="weather_roulette", seed=7)
while not obs.done:
    action = my_agent(obs)
    obs = client.step(action)
print(obs.score)
```

Or HTTP directly:

```bash
curl -X POST https://deltasthic-f1-strategist.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task":"weather_roulette","seed":7}'
```

Visit the [interactive landing page](https://deltasthic-f1-strategist.hf.space/) — click Reset, pick a circuit, watch the race. The full blog post is also rendered at [`/blog`](https://deltasthic-f1-strategist.hf.space/blog).

Reproducible Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deltasthicc/F1_Simulator_OpenENV/blob/main/notebooks/f1_strategist_training_colab.ipynb)

---

## Acknowledgments

Architecture primitives ported from our Round-1 `OpsTwin Recovery Arena` submission.
Track data from [racetrack-database](https://github.com/TUMFTM/racetrack-database) (MIT).
Historical calibration from the Kaggle F1 World Championship dataset (1950–2024).
Thanks to Meta PyTorch, Hugging Face, Unsloth, and Scaler AI Labs for organising the hackathon.
