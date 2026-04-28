# F1 LLM Racing — Personal Project Roadmap

> **Vision:** Simulate a complete F1 race where every car is driven by an LLM
> with its own personality, on real F1 data, watchable live with commentary.

This is the **static plan**. It does not change often. For "where am I right
now", read [`STATUS.md`](STATUS.md).

---

## Hardware budget (single RTX 5090, 32 GB VRAM)

| Resource | Live race | Training |
|---|---|---|
| VRAM | 12–18 GB | 22–26 GB |
| RAM | <8 GB | <8 GB |
| Disk | ~25 GB total | same |

Hard ceilings: 70B models out, 32B borderline, 14B comfortable. Training and
serving cannot run simultaneously — swap modes.

---

## Architecture (final shape)

```
┌──────────────────────────────────────────────────────────────┐
│  FastAPI server (server/app.py)                              │
│  ├─ /                     (current submission landing)       │
│  ├─ /f1_LLM_racing        (NEW preview page — vision/plan)   │
│  ├─ /race/live/<id>       (Phase 7 — SSE stream of race)     │
│  ├─ /reset, /step         (OpenEnv API)                      │
│  └─ /blog                 (existing)                         │
│                                                              │
│  Race coordinator (Phase 5+)                                 │
│  ├─ runs lap loop, holds shared race state                   │
│  ├─ collects 20 actions/step, advances world                 │
│  └─ emits SSE ticks to /race/live                            │
│                                                              │
│  Inference router → vLLM (Phase 4+)                          │
│  └─ one Qwen3-14B base + N persona LoRAs, swap per car       │
└──────────────────────────────────────────────────────────────┘
```

No model-to-model networking. Cars share the race state as a blackboard.

---

## Phase overview

| Phase | What | Time (PT) | Tag |
|---|---|---|---|
| 0 | Foundation cleanup | 3 days | `v0.2-foundation` |
| 1 | Full race length (50–70 laps) | 1 wk | `v0.3-fullrace` |
| 2 | FastF1 grounding | 1 wk | `v0.4-grounded` |
| 3 | Bigger model (Qwen3-14B) | 1 wk | `v0.5-14b` |
| 4 | 5 opponent personas | 1 wk | `v0.6-personas` |
| 5 | All 20 cars LLM-driven | 2 wks | `v0.7-multiagent` |
| 6 | Self-play GRPO | 2 wks | `v0.8-selfplay` |
| 6.5 | Continual learning from races | 1 wk | `v0.85-continual` |
| 7 | Live race UI | 1 wk | `v0.9-liveui` |
| 8 | Commentary track | 1 wk | `v0.10-commentary` |
| 9 | Polish & public demo | 1 wk | `v1.0` |

**Total:** ~11 weeks part-time, ~4–5 weeks full-time.

Phases 0–3 are linear. Phase 2 (FastF1) and Phase 3 (14B) can run in parallel.
Phase 7 (UI) can start any time after Phase 1.

---

## Resumability convention

1. `ROADMAP.md` (this file) — static plan.
2. `STATUS.md` — current cursor, updated every session.
3. Git tag at the end of every phase (`v0.X-name`).
4. Per-sub-step commit so `git log` shows exact provenance.
5. Memory entry pointing at both files so any future Claude session resumes
   cleanly.

---

## Phase 0 — Foundation cleanup

**Goal:** clean, tested, documented baseline. Address known bugs from the
hackathon submission so they don't silently corrupt later phases.

### Deliverables
- `ROADMAP.md` (this file) — committed
- `STATUS.md` — committed, updated each session
- `/f1_LLM_racing` preview page live (vision + roadmap + progress)
- Postmortem ablation re-run on harder seeds OR claim formally dropped
- Gradio `/web` route fixed or formally dropped (replaced by Phase 7 UI)
- Colab notebook re-verified end-to-end on Free T4
- 110 MB tokenizer cruft removed from git history (`grpo_v1/checkpoint-*/tokenizer.json`)
- `scripts/{train.sh, eval.sh, serve.sh, race.sh}` — idempotent, resumable

### Open decisions
- Keep postmortem mechanic? Run ablation first, decide on data.
- Keep Gradio web UI or replace with Phase 7 SSE dashboard? Recommend replace.

### Tests / gates
- `pytest -q` passes (164 tests stay green)
- `python evaluate.py --quick` reproduces 0.618 ± 0.02
- Fresh `git clone` + `uv sync` + `python scripts/race.sh` runs smoke race in <2 min
- `/f1_LLM_racing` renders correctly at `https://f1.chinnaboina.com/f1_LLM_racing`

### Sub-steps
- 0.1 Write `ROADMAP.md` and `STATUS.md`
- 0.2 Build `/f1_LLM_racing` preview page (vision, roadmap, progress)
- 0.3 Run postmortem ablation on harder seeds; record verdict in `docs/postmortem-ablation.md`
- 0.4 Strip tokenizer cruft from git history (BFG or `git filter-repo`)
- 0.5 Decide & act on Gradio `/web`
- 0.6 Re-run Colab notebook on T4
- 0.7 Add `scripts/` wrappers
- 0.8 Tag `v0.2-foundation`

### Milestone
A clean, tested, documented baseline tagged `v0.2-foundation`. Public preview
page live so the project has a face during construction.

---

## Phase 1 — Full race length

**Goal:** env supports realistic 50–70 lap GP races without reward signal
collapsing.

### Deliverables
- `LAPS_PER_STEP` and `total_laps` parametrized per scenario family (12, 25, 55)
- Reward shaping rebalanced (current per-step `+0.02` revelation reward
  dominates the long-race signal — needs scaling by `1/total_laps`)
- Three new long scenario families: `monaco_full_gp`, `silverstone_full_gp`,
  `spa_full_wet`
- Expert solver verified ≥0.85 on long-race scenarios

### Open decisions
- Step granularity: 1 step = 1 lap, or sub-lap steps for special events?
- Recommendation: 1 step per lap + special-event steps (pit out, SC restart,
  start) that don't count as a lap.

### Tests / gates
- Expert solver ≥0.85 across all 3 new families × 5 seeds
- Reward distribution per-step roughly stable across the race (no flat zones)
- Random ≤0.45, untrained ≤0.55, trained-from-Phase-3 ≥ untrained + 0.15

### Sub-steps
- 1.1 Parametrize race length in `scenarios.py` and `generator.py`
- 1.2 Add 3 long-race scenario families
- 1.3 Re-tune reward shaping (`server/scoring.py`)
- 1.4 Verify expert solver on long races
- 1.5 Run current grpo_v2 on long races (will under-perform — expected)
- 1.6 Tag `v0.3-fullrace`

### Milestone
A clean Monaco GP runs to completion, expert ≥0.90, audit trail readable.

---

## Phase 2 — FastF1 grounding

**Goal:** replace synthetic calibration with real F1 telemetry from 2023–2025.

### Deliverables
- `data/fastf1_cache/` populated for 2023–2025 (~5 GB)
- `data/grounded/` with per-track tyre degradation curves, fuel-burn calibration,
  opponent pace baselines, real safety-car frequency, real weather distributions
- `server/physics.py` reads grounded curves, falls back to synthetic if missing
- "Replay 2024 Monaco" mode: load real GP, let strategist propose calls, compare
  to actual team decisions

### Open decisions
- Which seasons? 2023–2025 (current regs) safest. 2022 mixes regulatory eras.
- License/attribution: FastF1 free for non-commercial use — note in README.

### Tests / gates
- Predicted tyre-cliff lap matches real teams' first stops within ±2 laps,
  ≥80% of tracks
- Lap-time distribution matches real F1 within 0.5s mean error
- Replay mode runs 2024 Monaco GP, shows where strategist diverged from team

### Sub-steps
- 2.1 Add FastF1 to deps; build cache pipeline
- 2.2 Extract per-track tyre curves; commit to `data/grounded/`
- 2.3 Extract fuel burn, dirty-air, weather, SC frequency
- 2.4 Wire grounded curves into `physics.py`
- 2.5 Build replay mode (`evaluate.py --replay 2024-monaco`)
- 2.6 Tag `v0.4-grounded`

### Milestone
"Strategist replays the 2024 Monaco GP" demo — strategist's calls vs. team's
actual calls, side by side.

---

## Phase 5+ enhancement — Model chooses starting compound (earmarked)

Right now `starting_compound` is fixed per scenario in
`server/scenarios_long.py`. Real F1 teams *choose* their starting tyre as part
of pre-race strategy, and Devin Thomas's F1-RL paper (arXiv:2501.04068)
matches this — they fix starting compound via the simulator, not the policy.

**The richer move (Phase 5 or 6):** add `SET_STARTING_COMPOUND` as a special
pre-race action issued at lap 0. The model would observe scenario context
(track, weather forecast, opponent compounds) before the race begins and pick
its own opener. This is **multi-agent + pre-race agency** — fits naturally
with Phase 5 (all 20 cars LLM-driven, each with their own pre-race choice).

Skipped for now because:
1. It expands the action space and observation surface (architectural change)
2. The fixed-starting-compound design lets the model learn pure in-race
   strategy first
3. Scenario authors can express *forced* starting-compound choices as
   training signal (e.g., "you start on hard at Silverstone — make it work")

**Insight that surfaced this**: starting on a more durable compound is a
real strategic lever, distinct from per-stint mode management. Both are
needed. Documented in expert sequences as the "hard opener + conserve in
midrace stint" pattern.

---

## Phase 6.5+ data refinement — Bayesian state-space tire model (earmarked)

When linear regression on tire degradation isn't good enough, swap in
Beatson 2025 ([arXiv:2512.00640](https://arxiv.org/abs/2512.00640)). It
decomposes `lap_time = base + fuel(kg) + tire_pace(latent) + noise` with
state-space inference, giving us per-stint latent tire pace separated from
driver and traffic noise. The right answer when our R² stays low.

Earmarked for: post-multi-agent (Phase 5+) data refinement, alongside
per-lap weather filtering (B3 in `docs/calibration-decisions.md`). Not
needed for Phase 3 training — the env's behavior tolerates approximate
tire curves; what matters there is the *strategic patterns* in scenarios.

---

## Phase 3 upgrade path — HCAPO (earmarked, not started)

If 14B GRPO plateaus on long-race scenarios with ≤0.05 improvement over 4B,
implement HCAPO-style hindsight credit reweighting per
[arXiv:2603.08754](https://arxiv.org/abs/2603.08754). HCAPO beats vanilla GRPO
by +13.8% on ALFWorld and +7.7% on WebShop (long-horizon LLM-agent benchmarks)
using sparse terminal reward + post-hoc credit reweighting.

Why this is an *upgrade path* and not the default: it's a trainer modification
(adds a hindsight Q-value term to GRPO advantages), not an environment change.
Phase 1.3 reward philosophy stands; HCAPO sits on top of it.

---

## Phase 3 — Bigger model (Qwen3-14B)

**Goal:** upgrade from 4B to 14B and prove it's better.

### Deliverables
- SFT warm-start dataset rebuilt at 14B context (`sft_dataset_v4.jsonl`)
- `grpo_v3/` checkpoint, 500 steps, 14B + LoRA
- Eval: 4B vs 14B on the 6 original + 3 long scenarios
- Model published to HF Hub: `Deltasthic/f1-strategist-qwen3-14b-grpo`

### Open decisions
- Qwen3-14B vs Qwen2.5-14B-Instruct: Q3 has thinking mode you've already
  debugged; Q2.5 is more vanilla. Default Qwen3-14B unless Unsloth support breaks.
- LoRA rank: 16 (current) or 32 (more capacity, +20% VRAM)? Try 16 first.

### Tests / gates
- 14B trained ≥ 4B trained by ≥0.05 weighted score
- VRAM during GRPO stays under 28 GB
- Single training step ≤30s with batch=2, num_generations=8
- All five prior bugs verified absent (see `docs/prior-bugs-checklist.md` —
  authored in 0.1 sub-step)

### Sub-steps
- 3.1 Rebuild SFT dataset for 14B (`sft_dataset_v4.jsonl`)
- 3.2 SFT warm-start run
- 3.3 GRPO 500 steps
- 3.4 Merge LoRA, evaluate
- 3.5 Push to HF Hub
- 3.6 Tag `v0.5-14b`

### Milestone
Reward curve PNG, eval table, HF Hub link.

---

## Phase 4 — 5 opponent personas

**Goal:** opponents are LLM personas with personalities, not scripts.

### Deliverables
- 5 personas: Aggressor, Cautious, Veteran, Rookie, Wildcard
- One LoRA per persona, sharing the 14B base
- vLLM server with multi-LoRA enabled
- Race coordinator routes each car to its persona's LoRA

### How personalities are created
1. Author persona prompt cards (`docs/personas.md`).
2. Generate persona-flavored SFT data — partly by prompting the strong model
   (Claude or 14B) to act as that persona for ~500 simulated decisions, partly
   by tweaking reward weights:
   - Aggressor: bonus for overtakes, push mode, early pits
   - Cautious: bonus for tyre health, fuel margin, defensive lines
   - Veteran: bonus for stint length, low radio chatter
   - Rookie: lower bonus everywhere, occasional reward noise
   - Wildcard: bonus for contrarian calls (pit under green, opposite-tyre)
3. Train 5 LoRAs from the same Phase-3 base.
4. Optional short GRPO pass per persona to keep them competitive.

### Open decisions
- 20 cars assigned to 5 personas: fixed (4 each) or random per race?
- Recommendation: fixed, mapped to driver names for narrative consistency.

### Tests / gates
- Each persona identifiable from blind action traces (10-race manual eyeball test)
- vLLM serves 20 concurrent requests with different LoRAs at <500 ms p95
- Wildcard persona makes inadvisable plays >2× as often as Cautious

### Sub-steps
- 4.1 Author persona prompt cards
- 4.2 Generate persona-flavored SFT data
- 4.3 Train 5 LoRAs
- 4.4 Stand up vLLM with multi-LoRA
- 4.5 Wire race coordinator to LoRA-route per car
- 4.6 Tag `v0.6-personas`

### Milestone
A 20-car race where opponents are LLM personas, not scripts. Replay shows
distinguishable behaviors.

---

## Phase 5 — All 20 cars LLM-driven

**Goal:** every car (including ego) is an LLM. Env is symmetric.

### Deliverables
- `F1Observation` becomes per-car POV (each car sees its own state full,
  others' state filtered by track distance)
- `step()` accepts dict `{car_id: F1Action}` instead of one action
- Race coordinator collects 20 actions per step, advances world, returns 20 obs
- Per-car postmortem buffer

### Open decisions
- Sequential or parallel agent decisions per step?
- Recommendation: parallel + post-hoc conflict resolution (rare conflicts).

### Tests / gates
- 20-agent race completes in <3 min wall-clock
- Race result non-degenerate (different cars finish in different positions)
- Each car's audit trail internally consistent

### Sub-steps
- 5.1 Refactor `F1Observation` to POV-aware
- 5.2 Refactor `step()` to multi-action API; keep backwards-compat single-action path
- 5.3 Update environment to drive 20 agents per step
- 5.4 Add conflict resolution (pit-box, overtake)
- 5.5 Smoke test: 20-car race
- 5.6 Tag `v0.7-multiagent`

### Milestone
A full 20-car LLM-driven race that runs end-to-end with coherent classification.

---

## Phase 6 — Self-play GRPO

**Goal:** personas co-evolve through self-play; meta-game emerges.

### Deliverables
- Self-play loop: each step samples 5 personas, runs a race, attributes per-car
  reward, GRPO update on each persona's LoRA
- League-style training: keep last 3 checkpoints of each persona as opponents,
  sample uniformly (prevents collapse)
- 1000-step self-play run, all 5 personas

### Open decisions
- Train all 5 simultaneously or rotate one-at-a-time?
- Recommendation: rotate. Otherwise GPU OOMs and reward signal becomes noisy.

### Tests / gates
- After 1000 self-play steps, average score across all personas improves by ≥0.05
- No persona's action entropy collapses below 0.5 nats
- Manual replay shows new emergent behaviors (defensive blocking, slipstream)

### Sub-steps
- 6.1 Self-play orchestrator
- 6.2 Per-persona reward attribution
- 6.3 League snapshot system
- 6.4 1000-step run
- 6.5 Eval & tag `v0.8-selfplay`

### Milestone
Reward curves for all 5 personas side-by-side. Behaviors visibly differentiated
and improved.

---

## Phase 6.5 — Continual learning from races

**Goal:** every finished race becomes training signal. Model improves over
time without manual retraining.

### Approach
Buffered nightly retraining (Option C from the design conversation), not
online weight updates. Safe, debuggable, easy to roll back.

### Deliverables
- `data/race_buffer/` — append-only JSONL of every finished race
  `{race_id, persona, trajectory, reward, scenario, seed, timestamp, model_version}`
- `scripts/nightly_retrain.py` — runs at midnight: pulls last N races from
  buffer, runs short GRPO update (50 steps), saves new LoRA, swaps live model
- `scripts/replay_real_gp.py` — pulls a real GP from FastF1 (Phase 2),
  generates a "regret minimization" reward: how much score did the model
  leave on the table vs. what the actual team did
- Live model registry: `models/active/` symlink → current LoRA, atomically
  swapped at midnight

### Open decisions
- Buffer size: last 256 races, last 1024, or all-time?
  Recommendation: last 1024 with reservoir sampling for diversity.
- Retrain cadence: nightly, or only when buffer grows by >100 races?
  Recommendation: gated by ≥100 new races AND midnight tick.
- LoRA versioning: keep last 7 days of nightly LoRAs for rollback.

### Tests / gates
- Nightly retrain completes in <30 min on the 5090
- Live model swap is atomic (no race in flight serves a torn model state)
- Performance is monotone or flat — never worse than 7 days ago
- Catastrophic forgetting probe: re-eval on the original 6 scenarios after
  each retrain, score must not drop by >0.05

### Sub-steps
- 6.5.1 Race-buffer logger (hook into race coordinator)
- 6.5.2 `nightly_retrain.py` skeleton + cron entry
- 6.5.3 Atomic LoRA swap mechanism
- 6.5.4 FastF1 regret-minimization reward
- 6.5.5 Catastrophic-forgetting probe (auto-rollback if regression detected)
- 6.5.6 Tag `v0.85-continual`

### Milestone
The model that races on Monday is measurably better than the one that raced
the previous Monday, with no human intervention.

---

## Phase 7 — Live race UI

**Goal:** a web page where you watch a race unfold in real time.

### Deliverables
- `/race/live/<race_id>` SSE endpoint streaming race state ticks
- HTML page: track-map (canvas, dot per car), leaderboard, tyre-wear bars,
  weather badge, pit-stop event feed
- Spectate mode (watch in progress) and replay mode (scrub a finished race)
- **All hosted at `/f1_LLM_racing` first; promoted to `/` at v1.0**

### Open decisions
- Real-time speed: 1 lap = 1s wall-clock? Adjustable?
- Recommendation: slider 1×–60×.

### Tests / gates
- 20-car race renders smoothly at 10× speed in Chrome/Firefox/Safari
- Reconnect works (no race-state loss on transient blip)
- Replay scrubbing bug-free

### Sub-steps
- 7.1 SSE endpoint emitting `(t, race_state_diff)` ticks
- 7.2 Track-map canvas component
- 7.3 Leaderboard + tyre bars
- 7.4 Pit-stop / event feed
- 7.5 Replay mode
- 7.6 Tag `v0.9-liveui`

### Milestone
Tweet-able live demo at `f1.chinnaboina.com/f1_LLM_racing`.

---

## Phase 8 — Commentary track

**Goal:** each lap, a commentator LLM produces 2–3 sentences of commentary.

### Deliverables
- Commentator LLM (smaller model OK — Qwen3-4B Instruct, no fine-tune needed)
- Per-lap commentary generation hooked into race coordinator
- Commentary stream rendered in the web UI
- Optional: TTS via Piper or Coqui (CPU, free)

### Open decisions
- One commentator or two (color + play-by-play)? Start with one.
- Live or batch? Live, generated as race unfolds.

### Tests / gates
- Commentary factually correct (no hallucinated overtakes) — manual review of 5 races
- Commentary for lap N ready before lap N+1 finishes simulating

### Sub-steps
- 8.1 Commentary prompt template
- 8.2 Per-lap generation hook
- 8.3 Render in UI
- 8.4 (Optional) TTS
- 8.5 Tag `v0.10-commentary`

### Milestone
A race replay with rolling commentary. With TTS: a watchable, listenable race.

---

## Phase 9 — Polish & public demo

**Goal:** ship a public, repeatable, watchable F1 simulation.

### Deliverables
- Public landing page: pick track, weather, drivers → simulate → watch live
  with commentary
- Race archive (browse past races)
- Polished README with embedded demo GIF
- Blog v2: "Simulating a complete F1 race with multi-agent LLMs"
- **`/f1_LLM_racing` promoted to `/`. Old submission landing moved to `/submission`**

### Sub-steps
- 9.1 Public race-setup form
- 9.2 Race archive page
- 9.3 README polish
- 9.4 Blog v2
- 9.5 Promote `/f1_LLM_racing` → `/`; tag `v1.0`
- 9.6 **Decide on HF Space discoverability.** At v1.0, decide whether to
  publish a personal HF Space at `Anurax1321/f1racecraft` for OpenEnv
  community discoverability. Tradeoff: the model won't run on free CPU tier
  by then (14B base), so the Space would be a static landing page only.
  Likely value: yes — it's a free distribution channel and OpenEnv judges
  search the Space hub.

### Milestone
`v1.0` shipped, public, shareable.

---

## Deferred / Parking Lot

Items left from the hackathon era that aren't part of the "simulate a complete
F1 race" vision. Not abandoned — just not on the critical path. Pull from this
list only when there's a concrete reason to.

- **Re-verify Colab notebook on T4** (was sub-step 0.6) — was a hackathon W1
  deliverable. URLs are patched for the new repo so it's not actively broken.
  Pull this back if/when we want a public reproducibility tutorial. Until then,
  GPU training happens on the 5090 box, not Colab.
- **Public HF Space** — see Phase 9.6. Decision deferred to v1.0.
- **Old grpo_v1 model adapters** — exist on `Deltasthic` HF Hub + frozen on the
  `deltasthicc` git remote. Not needed locally; `grpo_v2` is the champion.
- **Strip 110 MB tokenizer cruft from git history** (was sub-step 0.4) —
  irrelevant on the new repo (Option C migration started clean).

---

## Prior bugs — never repeat

The hackathon model hit five real bugs. Every future training run must verify
these are absent. Checklist lives in `docs/prior-bugs-checklist.md` (authored
in sub-step 0.1).

1. **Silent scripted-fallback** — eval reported model score but actually ran a
   hand-coded heuristic. Always log "model called: <token>" on first step.
2. **Qwen3 thinking-mode trap** — `max_new_tokens=64` with reasoning-on
   produced unparseable output. Always match thinking mode train/eval, raise
   `max_new_tokens` to 256+ when thinking is on.
3. **Train/eval format mismatch** — `enable_thinking=False` only at eval
   broke the chat-template prefix. Pin the format in one place, used by both.
4. **`format_obs` stripped scenario disambiguation** — model couldn't tell
   `late_safety_car` from `dry_strategy_sprint` at lap 0. Always include
   `obs.message` and `obs.hint` verbatim.
5. **Cold GRPO collapse** — vanilla GRPO from base Qwen3 plateaued at 0.54
   (`frac_reward_zero_std → 1.0`). Always SFT warm-start before GRPO.

Each phase's "Tests / gates" section must verify the relevant bugs are absent.
