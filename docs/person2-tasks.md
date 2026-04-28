# Person 2 Queue — Tanish Shitanshu

**Owner of:** Scoring (bodies) + Expert solver + Training + Evaluation + Inference + Notebook + Demo assets + HF Space deployment + GPU server work.

You are the trainer and the publisher. Person 1 builds the world; you grade it, train against it, evaluate, and ship the demo.

This document is your day-by-day work queue. Tick the boxes as you go.

---

## Day 0 — Bootstrap (parallel with Person 1)

- [x] Clone repo, `pip install -e .` works
- [x] Read [`docs/architecture.md`](architecture.md) end-to-end (you need to know the action space and observation schema)
- [x] Read [`docs/reward-model.md`](reward-model.md) end-to-end (this is yours to implement)
- [x] Read [`TRAINING.md`](../TRAINING.md) end-to-end
- [x] Read [`GPU_HANDOFF.md`](../GPU_HANDOFF.md) end-to-end
- [x] **Run `scripts/calibrate_opponent_pace.py`** — produces `data/opponent_pace_calibration.json` from `archive.zip`. Validate output JSON.
- [x] **Run `scripts/calibrate_tyre_baseline.py`** — produces `data/tyre_compound_baseline.json`. Validate output JSON.
- [ ] **Create the empty HF Space** via:
  ```bash
  huggingface-cli login
  huggingface-cli repo create f1-strategist --type space --space_sdk docker --organization Deltasthic
  ```
  Push README only at this stage to confirm the metadata renders correctly. The build will fail (no Dockerfile yet) — that's expected.
- [ ] Verify `nvidia-smi` on the RTX 5090 server (your training target).

---

## Day 1 — Phase 2 (Scoring + Expert Solver)

You start AFTER Person 1 has Phase 1 done — you need their `models.py`, `client.py`, and a running `server/environment.py` skeleton. While you wait, you can write the inference baseline (1.4 below) against just `models.py`.

### 1.1 Scoring — `server/scoring.py`

Implement the six pure functions from [`docs/reward-model.md`](reward-model.md):

```python
def compute_race_result(...) -> float: ...
def compute_strategic_decisions(...) -> float: ...
def compute_tyre_management(...) -> float: ...
def compute_fuel_management(...) -> float: ...
def compute_comms_quality(...) -> float: ...
def compute_operational_efficiency(...) -> float: ...

def compute_multi_objective_scores(
    audit_trail: list[dict],
    ego_final_state: dict,
    scenario: dict,
    issue_resolutions: dict,
    inspection_calls: dict,
    pit_decisions: list,
    radio_calls: list,
    pit_wall_calls: list,
) -> dict:
    scores = {
        "race_result": compute_race_result(...),
        "strategic_decisions": compute_strategic_decisions(...),
        "tyre_management": compute_tyre_management(...),
        "fuel_management": compute_fuel_management(...),
        "comms_quality": compute_comms_quality(...),
        "operational_efficiency": compute_operational_efficiency(...),
    }
    weights = {
        "race_result": 0.35, "strategic_decisions": 0.20,
        "tyre_management": 0.15, "fuel_management": 0.10,
        "comms_quality": 0.10, "operational_efficiency": 0.10,
    }
    final = sum(scores[k] * weights[k] for k in scores)
    scores["weighted_final"] = round(min(max(final, 0.01), 0.99), 4)
    return {k: round(v, 4) for k, v in scores.items()}
```

**Strict no-LLM, no-random rule.** All six functions are pure and deterministic.

Unit test (`tests/test_scoring.py`):
- [x] Zero-issue case → `weighted_final` ≈ 0.65 (six dims at 0.65 each gives ~0.65 weighted)
- [x] Perfect case → `weighted_final` ≈ 0.99
- [x] DNF case → `weighted_final` ≤ 0.30
- [x] Each dimension can be tested in isolation

### 1.2 Expert solver — `baselines/expert_solver.py`

Rule-based gold strategist. Reads scenario at "god mode" (full hidden state visible). Emits the optimal action sequence for each family.

```python
class ExpertSolver:
    def solve(self, scenario: dict, env: F1StrategistEnv) -> list[F1Action]:
        family = scenario["scenario_family"]
        if family == "dry_strategy_sprint":
            return self._solve_dry(scenario, env)
        elif family == "weather_roulette":
            return self._solve_weather(scenario, env)
        elif family == "late_safety_car":
            return self._solve_sc(scenario, env)
        elif family == "championship_decider":
            return self._solve_championship(scenario, env)

    def _solve_dry(self, sc, env):
        # 1. Inspections at lap 1
        # 2. Read undercut threshold + opponent strategies
        # 3. Pit at the correct lap (depends on opponent #N's plan)
        # 4. Comms before and after pit
        # 5. DONE
        ...
```

Verify: `python -m baselines.expert_solver --task dry_strategy_sprint --seed 0` prints final score ≥ 0.92.

Save traces:
```bash
python -m baselines.expert_solver --task dry_strategy_sprint --seed 0 \
    --save baselines/trajectories/expert_dry_seed0.jsonl
# Repeat for all four families × 5 seeds = 20 expert trajectories
```

### 1.3 Smoke test — `tests/smoke_all_scenarios.py`

```python
def test_all_scenarios():
    for family in ["dry_strategy_sprint", "weather_roulette", "late_safety_car", "championship_decider"]:
        env = F1StrategistEnvironment()
        env.reset(options={"task": family, "seed": 42})

        # Expert run
        expert = ExpertSolver()
        for action in expert.solve(env._scenario, env):
            obs, reward, done = env.step(action)
            if done: break
        assert obs.score >= 0.85, f"{family} expert scored {obs.score}"

        # Panic run
        env.reset(options={"task": family, "seed": 42})
        for _ in range(20):
            obs, reward, done = env.step(F1Action(command="STAY_OUT"))
            if done: break
        assert obs.score <= 0.40, f"{family} panic scored {obs.score}"
```

### 1.4 Inference baseline — `inference.py`

LLM agent runner that calls a chat-completion API and parses the response into `F1Action`. Direct port of OpsTwin's pattern.

```python
SYSTEM_PROMPT = """You are an F1 race strategist. You see lap data, opponents,
weather, and your car state. You issue strategic commands. Respond with one
command per turn from this list:
- PIT_NOW <compound>
- STAY_OUT
- SET_MODE <push|race|conserve|fuel_save|tyre_save>
- INSPECT_TYRE_DEGRADATION
- CHECK_OPPONENT_STRATEGY <driver_number>
- REQUEST_FORECAST
- ASSESS_UNDERCUT_WINDOW
- INSPECT_FUEL_MARGIN
- DEFEND_POSITION
- ATTACK_AHEAD
- HOLD_GAP <driver_number>
- LET_BY <driver_number>
- RADIO_DRIVER "<message>"
- DRAFT_PIT_WALL "<message>"
- REQUEST_INFO <opponents|weather|tyres|fuel|pit_window|standings|audit>
- RECOMMEND_PIT <next_lap>
- DRS_PERMISSION <on|off>
- ESCALATE_TO_PRINCIPAL
- DONE
"""

def run_inference(model, task, n_episodes=1, verbose=False):
    env = F1StrategistEnv.from_local()
    for ep in range(n_episodes):
        obs = env.reset(options={"task": task, "seed": ep})
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        while not obs.done:
            history.append({"role": "user", "content": format_obs(obs)})
            response = model.generate(history, max_new_tokens=256)
            history.append({"role": "assistant", "content": response})
            action = parse_action(response)
            obs, reward, done = env.step(action)
        print(f"Episode {ep}: score = {obs.score}")
```

Test: `python inference.py --model Qwen/Qwen3-0.6B --task dry_strategy_sprint --n-episodes 1`.

---

## Day 2 morning — Phase 3 (Training)

This is the highest-priority deliverable. **A reward curve must exist before demo time.**

### 2.1 Train script — `train.py`

TRL GRPO with the `environment_factory` pattern. Mirrors OpsTwin's `train_sft_v3.py` shape but tuned for the F1 env.

```python
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from openenv import OpenEnvClient
from client import F1StrategistEnv

def make_env_factory(task: str):
    def factory(): return F1StrategistEnv.from_local()
    return factory

def main(args):
    config = GRPOConfig(
        model_name=args.model,
        reward_funcs=["environment"],
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        save_steps=50,
        logging_steps=10,
        output_dir=args.output_dir,
    )
    trainer = GRPOTrainer(
        config=config,
        env_factory=make_env_factory(args.task),
        tokenizer=AutoTokenizer.from_pretrained(args.model),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
```

Smoke run on the 5090:
```bash
source .venv/bin/activate
python train.py --model Qwen/Qwen3-0.6B --max-steps 50 --batch-size 1 --grad-accum 8 \
    --task dry_strategy_sprint --output-dir ./grpo_smoke
# Expected: reward curve trends up. If flat for 50 steps → reward bug, not training bug.
```

If smoke is good, full run:
```bash
python train.py --model Qwen/Qwen3-4B --max-steps 500 --batch-size 1 --grad-accum 32 \
    --task multi --output-dir ./grpo_v1
# 500 steps × ~30s/step = ~4 hours on the 5090
```

### 2.2 Optional — three-stage curriculum (mirrors OpsTwin V3)

If straight GRPO from a base model doesn't converge in 500 steps:

1. **Stage 1** — SFT warm-start on expert trajectories (`baselines/trajectories/*.jsonl`). 5 epochs, broad coverage. See [`TRAINING.md`](../TRAINING.md).
2. **Stage 2** — GRPO with shaped per-step rewards. 200 steps.
3. **Stage 3** — GRPO with sparse final-only rewards. 200 steps.

Use the SFT seed dataset from `python capture_everything.py --n-seeds 50`.

### 2.3 Evaluate — `evaluate.py`

```python
def main(args):
    modes = ["random", "untrained", "trained", "expert"]
    results = {mode: {} for mode in modes}

    for task in args.tasks:
        for mode in modes:
            scores = []
            for seed in range(args.n_seeds):
                score = run_one(task, mode, seed, args.model)
                scores.append(score)
            results[mode][task] = {"mean": np.mean(scores), "std": np.std(scores)}

    # Save canonical numbers
    json.dump(results, open(args.output_json, "w"), indent=2)

    # Bar chart
    plot_bar_chart(results, output_path=args.output_png)
```

Output:
- `results/eval_summary.json`
- `results/eval_curve.png`

Run: `python evaluate.py --model Deltasthic/f1-strategist-qwen3-4b-grpo --tasks dry_strategy_sprint weather_roulette late_safety_car championship_decider --n-seeds 5`

### 2.4 Notebook — `notebooks/f1_strategist_training_colab.ipynb`

Colab-runnable smoke version on Free T4. Cells:
1. Install: `pip install openenv-core trl unsloth transformers peft`
2. Pull env: `env = F1StrategistEnv.from_hf_space("Deltasthic/f1-strategist")`
3. Smoke train: 30 GRPO steps on Qwen3-0.6B
4. Plot reward curve
5. Print one full rollout transcript

Pin versions to match TRAINING.md:
- `transformers >= 4.51`
- `accelerate >= 1.3`
- `trl == 0.14.0`
- `peft >= 0.13`
- `liger-kernel < 0.6` (or skip if `transformers >= 4.52`)

### 2.5 Push trained model — `scripts/push_checkpoint.py`

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo("Deltasthic/f1-strategist-qwen3-4b-grpo", repo_type="model", exist_ok=True)
api.upload_folder(folder_path="./grpo_v1", repo_id="Deltasthic/f1-strategist-qwen3-4b-grpo")
```

---

## Day 2 afternoon — Phase 4 (Ablation) + Phase 5 (Demo & Deploy)

### 4.1 Postmortem ablation

Once Person 1 has postmortem retrieval working:
```bash
# Base trained policy on 10 held-out seeds
python evaluate.py --model Deltasthic/f1-strategist-qwen3-4b-grpo \
    --tasks dry_strategy_sprint weather_roulette late_safety_car \
    --n-seeds 10 --no-memory --output-json results/ablation_no_memory.json

# Same model + memory hints injected
python evaluate.py --model Deltasthic/f1-strategist-qwen3-4b-grpo \
    --tasks dry_strategy_sprint weather_roulette late_safety_car \
    --n-seeds 10 --use-memory --output-json results/ablation_with_memory.json

# Diff
python scripts/diff_ablation.py results/ablation_no_memory.json results/ablation_with_memory.json
```

Save the diff to `results/ablation.md` for the blog post.

### 5.1 Blog post — `demo-assets/blog-post.md`

Target: 600–900 words, HF Hub blog format. Structure:

1. **Hook** (100 words): "It's lap 8 of 12 at Spa, light rain has started..." — paint the scene, state the strategist framing
2. **Why F1 strategy is a good RL benchmark** (150 words): partial observability, long-horizon credit assignment, multi-objective
3. **The environment** (200 words): action space, observation, hidden state, six-dim scoring. Show ONE code snippet
4. **The training story** (150 words): random vs untrained vs trained vs expert, the bar chart, one qualitative example
5. **The self-improvement loop** (100 words): postmortem mechanism + ablation result
6. **Try it** (50 words): HF Space link, Colab link, GitHub link

Embed:
- `results/eval_curve.png`
- One visualizer GIF rendered by Person 1 showing the trained agent calling rain correctly
- Code snippet of the action space

### 5.2 Video script — `demo-assets/video-script.md`

Target: ≤ 1:50 runtime. Hook → demo → result.

```
[00:00–00:10] Title card. Voice: "Train an LLM to be an F1 race strategist."
[00:10–00:35] Untrained agent on Spa, weather scenario. It stays out into the rain.
              On-screen: lap times balloon, position drops, score lands at 0.30.
[00:35–01:00] Cut to trained agent, same seed. Calls REQUEST_FORECAST, then INSPECT,
              then PIT_NOW inter at lap 7 — perfect. Lap times stay clean. Score 0.91.
[01:00–01:25] Bar chart: random/untrained/trained/expert across all four scenarios.
              Voice: "20-point average improvement after 500 GRPO steps on a 5090."
[01:25–01:50] Card with HF Space, Colab, GitHub links. End on the F1 Strategist logo.
```

Record using OBS or QuickTime. Render the visualizer GIFs first; the video is mostly screen capture of those plus the eval bar chart.

### 5.3 Deploy HF Space

Follow [`DEPLOY_SPACE.md`](../DEPLOY_SPACE.md):
```bash
git remote add hfspace https://huggingface.co/spaces/Deltasthic/f1-strategist
git push hfspace main:main
# wait 3-5 min for build
python tests/smoke_http.py --base-url https://Deltasthic-f1-strategist.hf.space
```

Save URL to `demo-assets/hf-space-link.txt`.

### 5.4 Publish blog and video

- Blog: copy `demo-assets/blog-post.md` to a new HF Hub blog post draft, upload images, publish. Save URL to `demo-assets/hf-blog-link.txt`.
- Video: upload to YouTube as unlisted-or-public. Save URL to `demo-assets/youtube-link.txt`.
- Update README links in the relevant section.

### 5.5 Final pre-push

Walk through [`PRE_PUSH_CHECKLIST.md`](../PRE_PUSH_CHECKLIST.md) item by item before tagging the submission commit.

---

## Merge points with Person 1

| When | What you provide | What Person 1 needs |
|------|-----------------|---------------------|
| End of Day 1 morning | Skeleton signatures in `server/scoring.py` | Person 1 imports them in `environment.py` |
| Day 1 afternoon | Full scoring bodies + expert traces | Person 1 verifies expert ≥ 0.85 on each scenario |
| Day 2 morning | `train.py` running on the 5090 | – |
| Day 2 afternoon | Eval results + ablation numbers | Person 1 includes them in the postmortem story |

---

## Pitfalls to avoid

- **OOM on Qwen3-4B with batch > 1.** Use `batch=1, grad_accum=32` on the 5090. Confirmed working in OpsTwin V3.
- **CUDA wheel mismatch.** Blackwell sm_120 needs `torch+cu128`. Earlier wheels work on older GPUs but fail on the 5090. See [`TRAINING.md`](../TRAINING.md) §environment.
- **TRL `environment_factory` returns wrong tools.** If the trainer's tool schema doesn't include all ~20 actions, it can't emit them. Verify with `print(trainer.env.tool_schema)` before kicking off the full run.
- **HF Space build fails on first push.** Almost always the Dockerfile doesn't `COPY` something the env imports. Check the build logs in the Space's "Logs" tab.
- **Colab T4 OOM on Qwen3-1.7B+.** The notebook is for Qwen3-0.6B only. Don't try larger.
- **Reward curve flat = reward bug.** If 50 steps shows no improvement, stop and inspect raw rollouts. The model is almost never the problem this early.
- **Forgetting to commit `results/`.** The judges grade on what's in the repo. If `eval_curve.png` is on disk but not committed, it doesn't exist.

---

## Definition of done (Person 2)

- [x] All six scoring functions implemented and unit-tested
- [x] Expert solver scores ≥ 0.92 on all four scenarios
- [ ] At least one GRPO reward curve from a 500-step run on the 5090
- [ ] Trained model published to `Deltasthic/f1-strategist-qwen3-4b-grpo`
- [x] `results/eval_curve.png`, `eval_summary.json`, `training_loss_curve.png` generated locally
- [ ] HF Space deployed and serving `/reset` correctly
- [ ] Colab notebook runs end-to-end on Free T4
- [ ] Blog post published, video uploaded, all three links saved
- [ ] [`PRE_PUSH_CHECKLIST.md`](../PRE_PUSH_CHECKLIST.md) walked through and ticked
