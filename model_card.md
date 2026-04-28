# F1 Strategist — Qwen3-4B SFT+GRPO (`grpo_v2`)

This model is a Qwen3-4B LoRA checkpoint trained to act as a race strategist inside the
F1 Strategist OpenEnv environment. It emits one command per turn, such as
`REQUEST_FORECAST`, `PIT_NOW inter`, `SET_MODE push`, or `RADIO_DRIVER "Box this lap"`.

## Environment

- GitHub: `Deltasthicc/F1_Simulator_OpenENV`
- Space: `Deltasthic/f1-strategist`
- Live: https://f1.chinnaboina.com/
- Action schema: `models.F1Action`
- Observation schema: `models.F1Observation`
- Reward: six deterministic Python dimensions in `server/scoring.py` (no LLM judge)

## Held-out evaluation

Real LLM forward pass through the env, 6 scenario families × 5 seeds, greedy decode:

| Scenario | Random | Untrained Qwen3-4B | **This model** | Expert |
|---|---:|---:|---:|---:|
| Dry strategy sprint | 0.40 | 0.51 | **0.52** | 0.84 |
| Weather roulette | 0.34 | 0.41 | **0.97** | 0.95 |
| Late safety car | 0.33 | 0.53 | **0.65** | 0.94 |
| Championship decider | 0.21 | 0.27 | **0.56** | 0.97 |
| Virtual safety-car window | 0.33 | 0.38 | **0.47** | 0.97 |
| Tyre cliff management | 0.20 | 0.40 | **0.55** | 0.97 |
| **Average** | **0.30** | **0.42** | **0.62** | **0.94** |

**+0.20 average lift over untrained Qwen3-4B baseline.** Closes ~33% of the
random→expert gap. On the weather scenario the trained model (0.965) edges
out the rule-based expert (0.950) — investigation discipline + correct timing.

## Training recipe — what actually worked

Two-stage pipeline. The single-stage cold-GRPO baseline plateaued at 0.54 with
reward-variance collapse; layering SFT first was what broke through.

### Stage 1 — SFT warm-start

```bash
python capture_everything.py --tasks dry_strategy_sprint weather_roulette \
    late_safety_car championship_decider --n-seeds 100 \
    --output sft_dataset_v2.jsonl
python train_sft_v1.py --model unsloth/Qwen3-4B \
    --dataset sft_dataset_v2.jsonl --output-dir ./sft_checkpoints_v1 \
    --epochs 3 --batch-size 1 --grad-accum 32 --lr 1e-5
python scripts/merge_lora.py --adapter sft_checkpoints_v1/final \
    --out sft_checkpoints_v1/merged
```

Key choices: enriched `format_obs` includes `Briefing:` + `Hint:` (scenario
disambiguation), `enable_thinking=False` rendering at both train and eval (Qwen3's
reasoning mode breaks short-output evaluation otherwise).

### Stage 2 — GRPO from SFT base

```bash
python train.py --base-checkpoint sft_checkpoints_v1/merged \
    --task multi --max-steps 200 --batch-size 1 --grad-accum 16 \
    --reward-mode shaped --output-dir ./grpo_v2 --backend trl --no-unsloth
python scripts/merge_lora.py --adapter grpo_v2 --out grpo_v2/merged
```

GRPO knobs: `num_generations=8`, `beta=0.005`, `temperature=0.9`,
`max_completion_length=128`, `use_vllm=False`.

### Compatibility notes (read these before retraining)

- **TRL pinned to 0.18.2.** Newer TRL renames `max_new_tokens` → `max_completion_length`
  in GRPOConfig; `transformers` 5.5+ removes `TRANSFORMERS_CACHE`.
- **Unsloth disabled for GRPO.** Unsloth's compiled GRPO trainer assumes TRL ≥ 0.22
  (calls `truncate_with_protected_tokens`). With our pinned TRL, the Unsloth path
  raises `NameError`. Run with `--no-unsloth`. Slower (2.5×) but works.
- **vLLM disabled.** Not installed in our env. Re-enable when upgrading the stack.

## Evaluation

Run:

```bash
python evaluate.py \
  --model Deltasthic/f1-strategist-qwen3-4b-grpo \
  --tasks dry_strategy_sprint weather_roulette late_safety_car championship_decider \
  --n-seeds 5 \
  --modes random untrained trained expert
```

Outputs:

- `results/eval_summary.json`
- `results/eval_curve.png`
- `results/FINAL_RESULTS.md`

## Limitations

This model is a hackathon research artifact for a simulated strategy environment. It is
not a real motorsport decision system and should not be used for safety-critical or
commercial race operations.
