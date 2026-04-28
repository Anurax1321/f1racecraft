"""Build a rejection-sampling fine-tune (RFT) dataset.

For each (task, seed, step_idx) in the prompt source:
  1. Replay env up to step_idx with expert actions
  2. Sample N completions from the model (temperature 0.9)
  3. Score each completion by inserting it into env at step_idx and continuing
     with expert actions through episode end
  4. Keep the top-1 completion; write a row in TRL chat-messages format

Output: a JSONL file usable directly with `train_sft_v1.py --dataset <out>`.
"""

from __future__ import annotations

# Unsloth must be imported first.
try:
    import unsloth  # noqa: F401
except Exception:
    pass


import argparse
import json
import time
from pathlib import Path

import torch

from baselines.expert_solver import EXPERT_SEQUENCES
from inference import SYSTEM_PROMPT, format_obs, parse_action
from models import F1Action
from server.environment import F1StrategistEnvironment
from server.scenarios import SCENARIOS

CANONICAL_TASKS = [
    "dry_strategy_sprint",
    "weather_roulette",
    "late_safety_car",
    "championship_decider",
]


def _load_model(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lm = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    lm.eval()
    return lm, tokenizer


@torch.no_grad()
def _sample_n(lm, tokenizer, prompt_text: str, n: int, max_new_tokens: int):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(lm.device)
    out = lm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        num_return_sequences=n,
        pad_token_id=tokenizer.pad_token_id,
    )
    n_in = inputs["input_ids"].shape[-1]
    return [tokenizer.decode(seq[n_in:], skip_special_tokens=True) for seq in out]


def _score_action(task: str, seed: int, step_idx: int, action_command: str) -> float:
    family = SCENARIOS[task]["scenario_family"]
    expert_seq = EXPERT_SEQUENCES[family]

    env = F1StrategistEnvironment()
    obs = env.reset(seed=seed, options={"scenario": SCENARIOS[task]})

    for i, cmd in enumerate(expert_seq):
        if i >= step_idx or obs.done:
            break
        obs = env.step(F1Action(command=cmd))

    if obs.done:
        return float(obs.multi_objective_scores.get("weighted_final", obs.score))

    obs = env.step(F1Action(command=action_command))
    if obs.done:
        return float(obs.multi_objective_scores.get("weighted_final", obs.score))

    for cmd in expert_seq[step_idx + 1:]:
        obs = env.step(F1Action(command=cmd))
        if obs.done:
            break
    return float(obs.multi_objective_scores.get("weighted_final", obs.score))


def _build_prompts(tasks: list[str], n_seeds: int) -> list[dict]:
    rows = []
    for task in tasks:
        family = SCENARIOS[task]["scenario_family"]
        expert_seq = EXPERT_SEQUENCES[family]
        for seed in range(n_seeds):
            env = F1StrategistEnvironment()
            obs = env.reset(seed=seed, options={"scenario": SCENARIOS[task]})
            for step_idx, expert_cmd in enumerate(expert_seq):
                if obs.done:
                    break
                rows.append({
                    "task": task,
                    "seed": seed,
                    "step_idx": step_idx,
                    "user_content": format_obs(obs),
                })
                obs = env.step(F1Action(command=expert_cmd))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to SFT-merged model")
    parser.add_argument("--tasks", nargs="+", default=CANONICAL_TASKS)
    parser.add_argument("--n-seeds", type=int, default=20,
                        help="Seeds per task (≈step-count more rows per seed)")
    parser.add_argument("--n-samples", type=int, default=8,
                        help="N completions per prompt for rejection sampling")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", default="rft_dataset_v1.jsonl")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap total prompts (0 = no cap)")
    args = parser.parse_args()

    print(f"[rft] Building prompts: tasks={args.tasks} seeds={args.n_seeds}")
    prompts = _build_prompts(args.tasks, args.n_seeds)
    if args.limit and len(prompts) > args.limit:
        prompts = prompts[: args.limit]
    print(f"[rft] {len(prompts)} prompts; sampling {args.n_samples}/prompt")

    print(f"[rft] Loading model {args.model} …")
    lm, tokenizer = _load_model(args.model)
    print("[rft] Model ready.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_kept = 0
    n_filtered = 0
    sum_best_score = 0.0
    sum_mean_score = 0.0

    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(prompts):
            chat = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["user_content"]},
            ]
            prompt_text = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            try:
                completions = _sample_n(lm, tokenizer, prompt_text,
                                        args.n_samples, args.max_new_tokens)
            except Exception as exc:
                print(f"[rft] skip row {idx}: gen failed: {exc}")
                continue

            best_text = None
            best_score = -1.0
            scores = []
            for txt in completions:
                cmd = parse_action(txt).command
                if not cmd:
                    continue
                s = _score_action(row["task"], row["seed"], row["step_idx"], cmd)
                scores.append(s)
                if s > best_score:
                    best_score = s
                    best_text = cmd

            if best_text is None or not scores:
                n_filtered += 1
                continue

            mean_score = sum(scores) / len(scores)
            if best_score <= mean_score + 1e-6:
                # No advantage — all completions are equivalent; skip
                n_filtered += 1
                continue

            sum_best_score += best_score
            sum_mean_score += mean_score
            n_kept += 1

            f.write(json.dumps({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["user_content"]},
                    {"role": "assistant", "content": best_text},
                ],
                "task": row["task"],
                "seed": row["seed"],
                "step_idx": row["step_idx"],
                "best_score": round(best_score, 4),
                "mean_score": round(mean_score, 4),
                "advantage": round(best_score - mean_score, 4),
            }) + "\n")

            if (idx + 1) % 25 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / max(elapsed, 1.0)
                remaining = (len(prompts) - idx - 1) / max(rate, 0.001)
                avg_best = sum_best_score / max(n_kept, 1)
                avg_mean = sum_mean_score / max(n_kept, 1)
                print(f"[rft] {idx+1}/{len(prompts)}  kept={n_kept} filt={n_filtered}  "
                      f"avg_best={avg_best:.3f} avg_mean={avg_mean:.3f}  "
                      f"eta={remaining/60:.1f}m")

    avg_best = sum_best_score / max(n_kept, 1)
    avg_mean = sum_mean_score / max(n_kept, 1)
    print(f"\n[rft] Done. kept={n_kept} filtered={n_filtered}")
    print(f"[rft] avg best={avg_best:.3f}  avg mean={avg_mean:.3f}  uplift={avg_best-avg_mean:.3f}")
    print(f"[rft] Output: {out_path}")


if __name__ == "__main__":
    main()
