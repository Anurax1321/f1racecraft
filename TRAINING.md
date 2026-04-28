# Training & Evaluation — F1 Strategist

This repo is ready for two levels of training:

1. A **verified local smoke path** that runs without a large GPU, exercises the
   environment and evaluator, and writes checkpoint-shaped artifacts under
   `grpo_smoke/`.
2. A **GPU handoff path** for a real Qwen3-4B GRPO run on the RTX 5090 server.
   The model repo
   [`Deltasthic/f1-strategist-qwen3-4b-grpo`](https://huggingface.co/Deltasthic/f1-strategist-qwen3-4b-grpo)
   should only be published after that run finishes.

See [`results/FINAL_RESULTS.md`](results/FINAL_RESULTS.md) for the current local
smoke comparison table and methodology notes.

> This recipe is the OpsTwin V3 stack with the F1 Strategist env swapped in. Same Python,
> same TRL/transformers/accelerate pins, same Unsloth path on the 5090. We are reusing it
> deliberately because it landed OpsTwin at 0.956 average final score.

## Compute

- 1× NVIDIA GPU with **≥ 32 GB VRAM** — we use the personal RTX 5090
- ~**3–4 h** wall-clock for the full 3-stage curriculum (early-stopped)
- ~**80 GB** disk for intermediate checkpoints; can prune after each stage

If you do not have a 32 GB GPU, the smaller path is:
- Qwen3-1.7B + LoRA on a 24 GB card (slower but feasible)
- Qwen3-0.6B + QLoRA on Colab Free T4 (smoke-only, will not reach reported numbers)

## Environment

```bash
# Python 3.12 + uv for dep management
python3 -m venv .venv && source .venv/bin/activate

# Torch with CUDA 12.8 wheels (REQUIRED for Blackwell sm_120; older CUDA wheels work on older GPUs)
uv pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch

# Training stack — pins matter, see "stack gotchas" below
uv pip install \
  'trl==0.14.0' 'transformers>=4.55.4' 'peft>=0.13.2' \
  'datasets>=3.2.0' 'accelerate>=1.3' \
  'liger-kernel<0.6' 'bitsandbytes>=0.49' \
  matplotlib protobuf python-dotenv

# Unsloth for LoRA-on-4B fast path
uv pip install "unsloth[cu128] @ git+https://github.com/unslothai/unsloth.git"

# F1 Strategist env deps (editable)
uv pip install -e .

# HuggingFace auth (for push + private models). Token needs write access.
# Do not commit .env; it is gitignored.
cp .env.example .env
# then edit .env and set HF_TOKEN
```

### Stack gotchas (lessons from OpsTwin V3)

- **`transformers` 4.47.0** does NOT recognise the Qwen3 architecture. Use **4.51 or newer**.
- **`accelerate` 1.2.1** is **incompatible with `transformers` 4.55** (missing `keep_torch_compile` kwarg in `Accelerator.unwrap_model`). Use **1.3 or newer**.
- **`liger-kernel ≥ 0.6`** requires `transformers ≥ 4.52`. Pin `<0.6` if you stay on 4.51.
- **`torch+cu128`** is required for the RTX 5090 (Blackwell sm_120). Earlier CUDA wheels fail to compile kernels.

## Quick smoke run (sanity check before the long run)

```bash
source .venv/bin/activate && set -a && source .env && set +a

# Verified CPU/local smoke. This does not train Qwen; it validates the env,
# evaluator, checkpoint layout, and reward-curve plotting.
python train.py \
    --model heuristic \
    --task multi \
    --max-steps 12 \
    --batch-size 1 --grad-accum 1 \
    --output-dir ./grpo_smoke

# Inspect the smoke run
python scripts/plot_training_curve.py --run-dir ./grpo_smoke --output results/training_loss_curve.png
# Open results/training_loss_curve.png
```

If this passes and produces a non-empty curve, the environment and evaluation loop are
ready for the full run.
If it's flat, **stop** — there's almost certainly a reward bug, not a training bug.
Inspect a couple of rollouts via `python rollout.py --task dry_strategy_sprint --model ./grpo_smoke --verbose`.

## Three-stage training (full run)

### Stage 1 — broad coverage SFT warm-start

This repo currently includes the SFT data generator but not a separate
`train_sft_v1.py` implementation. Use this stage only if you bring in the SFT
trainer from OpsTwin or another PEFT/SFT trainer; otherwise start with the
single-stage GRPO path below.

```bash
# First, generate the SFT dataset from procedural seeds + expert trajectories
python capture_everything.py \
    --tasks dry_strategy_sprint weather_roulette late_safety_car championship_decider \
    --n-seeds 100 \
    --output sft_dataset_v1.jsonl
# Should produce ~5,000 training turns

python train_sft_v1.py \
    --model Qwen/Qwen3-4B \
    --hf-model-id Deltasthic/f1-strategist-qwen3-4b-sft-v1 \
    --epochs 3 --batch-size 1 --grad-accum 32 --lr 1e-5 \
    --dataset sft_dataset_v1.jsonl \
    --output-dir ./sft_checkpoints_v1
```

Stop when rollout-callback avg plateaus. The best stage-1 checkpoint by rollout avg is
typically `checkpoint-150` to `checkpoint-200`. Pass that to stage 2.

### Stage 2 — GRPO with shaped rewards from the SFT warm start

```bash
python train.py \
    --base-checkpoint ./sft_checkpoints_v1/checkpoint-200 \
    --task multi \
    --max-steps 200 \
    --batch-size 1 --grad-accum 32 \
    --reward-mode shaped \
    --output-dir ./grpo_v1_stage2
```

Goal: anchor each scenario family at ≥ 0.55 and reduce inter-scenario oscillation.

### Stage 3 — GRPO with sparse final-only rewards

```bash
python train.py \
    --base-checkpoint ./grpo_v1_stage2/checkpoint-best \
    --task multi \
    --max-steps 200 \
    --batch-size 1 --grad-accum 32 \
    --reward-mode sparse \
    --output-dir ./grpo_v1_stage3
```

This is the long-horizon polish stage. The shaped rewards in stage 2 give the model the
basics; stage 3 forces it to reason end-to-end without crutches.

If any single scenario family drops below 0.50 during stage 3, abort the run and add a
floor guard (similar to OpsTwin V3's `GuardedRolloutEvalCallback`).

## Single-stage path (faster, for the hackathon timeline)

If 3-stage is too slow:

```bash
python train.py \
    --model Qwen/Qwen3-4B \
    --task multi \
    --max-steps 500 \
    --batch-size 1 --grad-accum 32 \
    --reward-mode shaped \
    --warm-start-sft ./sft_checkpoints_v1/checkpoint-200 \
    --output-dir ./grpo_v1
```

This combines SFT warm-start with shaped GRPO in one process. Less robust but takes ~3 hours
instead of ~4.

## Publishing

```bash
# Edit scripts/push_checkpoint.py:
#   CKPT = "./grpo_v1_stage3/checkpoint-best"
#   REPO = "Deltasthic/f1-strategist-qwen3-4b-grpo"
python scripts/push_checkpoint.py
```

The model card for the HF repo lives at `model_card.md`; copy it into the repo on push.

## Evaluation (against the published HF model)

```bash
python evaluate.py \
    --model Deltasthic/f1-strategist-qwen3-4b-grpo \
    --tasks dry_strategy_sprint weather_roulette late_safety_car championship_decider \
    --n-seeds 5 \
    --modes random untrained trained expert
# Outputs:
#   results/eval_summary.json  (canonical numbers, JSON)
#   results/eval_curve.png     (4-bar grouped chart per scenario)
```

## Plotting the training curve

```bash
python scripts/plot_training_curve.py
# Output: results/training_loss_curve.png
# Requires sft_checkpoints_*/  and  grpo_v1_*/  directories on disk.
```

## Notes on the evaluator

`evaluate.py` runs each (mode, task, seed) tuple through a fresh env reset. To prevent
contamination between modes:
- `random` mode: uniform sample from the action space
- `untrained` mode: same model code path as `trained` but `--model Qwen/Qwen3-4B` (no GRPO weights)
- `trained` mode: `--model Deltasthic/f1-strategist-qwen3-4b-grpo`
- `expert` mode: rule-based `baselines/expert_solver.py`, runs at god-mode

`max_new_tokens` is 384 (large enough for `<think>` blocks + a command). If it's
truncating, scores collapse — verify by rendering one rollout with `--verbose`.

## Artifacts

- `results/FINAL_RESULTS.md` — full untrained-vs-trained-vs-expert table + methodology notes
- `results/eval_summary.json` — canonical eval numbers
- `results/eval_curve.png` — bar chart for the blog
- `results/training_loss_curve.png` — loss + reward over training steps
- `results/ablation.md` — postmortem augmented vs base trained policy
- `training_log.txt`, `training_v1_stage2.log`, `training_v1_stage3.log` — full training logs

## Files NOT in git (regeneratable)

- `sft_dataset_v1*.jsonl` — produced by `capture_everything.py` on each run
- `sft_checkpoints_v1*/` — ~70 GB of checkpoint data
- `grpo_v1_stage*/` — ~50 GB
- `.venv/` — virtual environment

These are listed in `.gitignore`. Regenerate them on a fresh machine via the recipe above.

## Target numbers

| scenario | random | untrained Qwen3-4B | trained (target) | rule-based expert |
|---|---|---|---|---|
| dry_strategy_sprint | ~0.20 | ~0.35 | **≥ 0.70** | 0.95 |
| weather_roulette | ~0.20 | ~0.30 | **≥ 0.65** | 0.93 |
| late_safety_car | ~0.20 | ~0.40 | **≥ 0.65** | 0.92 |
| championship_decider | ~0.18 | ~0.30 | **≥ 0.55** | 0.90 |
| **average** | ~0.20 | ~0.34 | **≥ 0.65** | 0.92 |

Average improvement target: **≥ 0.30 over untrained**. Anything less and we have a
training pipeline issue, not a model-capability issue.

## Common failure modes

- **Reward curve flat after 100 steps** → reward bug. Inspect rollouts. Verify the scorer
  is being called with the right arguments. Verify `_pending_rewards` is being finalised.
- **Reward curve trends up but evaluation is no better than untrained** → train/eval mismatch.
  Check the evaluation `max_new_tokens` is at least as large as during training. Check the
  observation format hasn't drifted between train and eval.
- **One scenario family scores well, others tank** → curriculum imbalance. Add `family_weights`
  to the training-data sampler in stage 2. This is exactly the OpsTwin V3 stage-2 fix.
- **OOM on the 5090** → grad-accum is too small for the chosen batch_size. Drop `batch_size`
  to 1 first, then halve grad-accum. Unsloth's gradient checkpointing should be ON.
- **Training crashes at step 1 with "Qwen3 not recognised"** → `transformers` < 4.51. Upgrade.
- **Model emits malformed actions ("PT NOW", "soft pit")** → tokeniser issue. Verify Qwen3's
  chat template is correctly applied in `inference.py`. Look at one full prompt+response
  before kicking off training.
