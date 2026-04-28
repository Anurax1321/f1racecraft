# Reward Philosophy

> **One-line rule:** Heavy terminal reward + sparse anti-hacking shaping.
> Don't add per-step dense shaping. The literature says don't and we won't.

This file exists so future-Anurag (or future-Claude) doesn't *quietly drift*
into adding dense per-step rewards when training feels slow. That drift is the
single highest-cost mistake we can make on this project. Read this first.

---

## The rule

The reward function has exactly two parts:

1. **Terminal score** — `compute_multi_objective_scores()` — a 6-dim weighted
   value in `[0.01, 0.99]`, computed once at episode end. This carries 80%+ of
   the learning signal.
2. **Sparse per-step shaping** — small immediate rewards for *correctness
   signals* (taking a valid investigatory action, confirming a pit, sending a
   radio call). One-shot where revelations are concerned. Never a "density
   floor."

That's it. Everything else is bookkeeping (audit trail, postmortem memory).

---

## Why this design

### The F1-RL community has explicitly studied this and rejected dense shaping

Devin Thomas et al., *Explainable Reinforcement Learning for Formula One Race
Strategy*, ACM SIGAPP 2025 (`arXiv:2501.04068`). Their reward:

| Component | Value |
|---|---|
| Terminal | `+100 × F1 points` (P1 = +2500, P11+ = 0) |
| Per-lap "alive" bonus | `+1` |
| Extraneous pit penalty | `-10` |
| Invalid action | `-1000` |

Direct quote from the paper:

> *"Reward shaping is not used in the reward function because it is inherently
> difficult in this context. This is because we are unable to determine
> midway through a race whether a decision was good or not."*

Our scorer has more sophistication (six dimensions) but the philosophy is the
same: trust the terminal signal.

### Long-horizon LLM agents agree

HCAPO (`arXiv:2603.08754`, March 2026) is current SOTA for long-horizon LLM
agents (ALFWorld, WebShop). They use **sparse terminal rewards only** and
solve credit assignment via *hindsight reweighting* in the GRPO update.
Beats vanilla GRPO by +13.8% on ALFWorld, +7.7% on WebShop.

If our 14B GRPO run plateaus in Phase 3, HCAPO is the upgrade path. Not
dense shaping.

### Process Reward Models don't fit race strategy

PRMs work where each step is independently verifiable (math, code). Race
strategy decisions can only be evaluated against future state. There is no
honest way to grade lap-22 STAY_OUT in isolation. Don't build a PRM here.

### Reward-hacking immunity should be a design principle

Anthropic 2025 work on natural emergent misalignment from reward hacking
recommends building immunity *by construction* — design the reward so it's
inherently un-hackable. Our one-shot-per-revelation pattern is exactly this.
Don't break it by adding "small bonus per inspection call always" to fight
sparsity. That's how reward hacking starts.

---

## Concrete shaping rewards in our env (reference)

| Action | Reward | One-shot? |
|---|---|---|
| `INSPECT_TYRE_DEGRADATION` | `+0.02` | Yes (per compound) |
| `CHECK_OPPONENT_STRATEGY` | `+0.02` | Yes (per opponent) |
| `REQUEST_FORECAST` | `+0.02` | Yes (entire race) |
| `ASSESS_UNDERCUT_WINDOW` | `+0.02` | Yes (entire race) |
| `INSPECT_FUEL_MARGIN` | `+0.02` | Yes (entire race) |
| `PIT_NOW` | `+0.01` | No (every pit) |
| `HOLD_GAP` | `+0.01` | No |
| `SET_MODE` / `RECOMMEND_PIT` / `RADIO_DRIVER` / etc. | `+0.005` | No |
| `STAY_OUT` | `0.0` | — |
| Invalid command | `-0.02` | — |
| Harmful action | `-0.05` | — |

Total per-episode shaping is **bounded ≈0.20 max**, regardless of race length
(short or long). That's by design. The terminal score (0.01–0.99) carries
the actual learning signal.

---

## When you'll be tempted to break this rule

You'll find yourself wanting to add dense shaping when:

1. **GRPO `frac_reward_zero_std → 1.0`** — every rollout produces the same
   reward; no signal. *This is bug #5 from the hackathon*. Fix: SFT
   warm-start, NOT dense shaping. The model needs to start in a region of
   policy space where rollouts diverge.
2. **Long-race rollouts feel too sparse** — long stretches of zero reward
   between actions. Fix: nothing. Trajectory variance still differs across
   rollouts; GRPO works at trajectory level, not step level.
3. **A specific behavior won't emerge** (e.g. always pit under VSC).
   Don't add a shaping reward for it. *Encode it in `success_criteria`*
   so the existing strategic-decisions scorer credits it terminally.

---

## What we DO add when we need richer signal

If the terminal scorer can't distinguish good play from bad play on a given
scenario family, the fix is **richer scenario authoring**, not richer shaping:

- More `issues` entries with explicit lap-window preconditions
- More precise `success_criteria` (target_n_pits, second_pit_window,
  required_compound, required_inspections)
- Scoring-pure additions to `_scenario_strategy_adjustment` for new
  family-specific patterns

These flow through the same terminal-score path and stay reward-hack-immune.

---

## Earmarked future work

If 14B GRPO in Phase 3 plateaus on long races (≤0.05 improvement over 4B),
implement HCAPO-style hindsight credit reweighting. See
[`arXiv:2603.08754`](https://arxiv.org/abs/2603.08754) and ROADMAP Phase 3.
