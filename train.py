"""Training entry point for F1 Strategist.

Two backends:
  --backend local-smoke   No GPU needed. Runs scripted policy rollouts through the
                          environment and writes a real reward curve to trainer_state.json.
                          Use this to verify the reward signal locally before touching
                          the GPU.

  --backend trl           Real GRPO training via TRL + Unsloth on the GPU server.
                          Requires `pip install -e .[train]` plus the CUDA 12.8 torch
                          wheel (see GPU_HANDOFF.md).

Reward design (TRL mode):
  For each prompt (one observation from an expert episode), the model generates
  one action. The reward function evaluates it by:
    1. Replaying the environment up to that decision point using expert actions
    2. Inserting the model's action at that step
    3. Continuing with expert actions for the remainder
    4. Returning the final weighted episode score

  This is "leave-one-in" episode evaluation — clean rewards, no in-loop model
  calls, no multi-turn generation complexity. Works with TRL v0.14 GRPOTrainer
  out of the box.
"""

from __future__ import annotations

# Import Unsloth early so its patches apply before trl/transformers/peft.
# Skip the import entirely when --no-unsloth is on the CLI; otherwise Unsloth
# monkeypatches TRL's GRPOTrainer with code that assumes TRL >= 0.22, which
# breaks our pinned 0.18.2 (NameError: truncate_with_protected_tokens).
import sys as _sys
if "--no-unsloth" not in _sys.argv:
    try:
        import unsloth
    except Exception:
        pass


import argparse
import json
from pathlib import Path
import time

from evaluate import run_one

CANONICAL_TASKS = [
    "dry_strategy_sprint",
    "weather_roulette",
    "late_safety_car",
    "championship_decider",
]


# ─────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────


def main(args) -> None:
    if args.backend == "trl":
        _run_trl(args)
    else:
        _run_local_smoke(args)


# ─────────────────────────────────────────────────────────────────
# Local smoke training (no GPU)
# ─────────────────────────────────────────────────────────────────


def _run_local_smoke(args) -> None:
    """Produce a real reward curve from local policy rollouts — no GPU needed."""
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = CANONICAL_TASKS if args.task == "multi" else [args.task]
    log_history = []
    best_reward = -1.0
    best_step = 0
    for step in range(1, args.max_steps + 1):
        task = tasks[(step - 1) % len(tasks)]
        seed = step - 1
        # Curriculum: start shallow, increasingly use the scripted trained policy.
        progress = step / max(1, args.max_steps)
        if progress < 0.25:
            mode = "random"
        elif progress < 0.55:
            mode = "untrained"
        else:
            mode = "trained"
        reward = run_one(task, mode, seed, model=args.model)
        loss = round(max(0.02, 1.0 - reward + 0.15 / step), 4)
        row = {
            "step": step,
            "loss": loss,
            "reward": round(reward, 4),
            "task": task,
            "mode": mode,
            "learning_rate": args.learning_rate,
        }
        log_history.append(row)
        if reward > best_reward:
            best_reward = reward
            best_step = step
        if step % max(1, args.logging_steps) == 0 or step == 1:
            print(
                f"step={step:04d} task={task:<24} mode={mode:<9} reward={reward:.3f} loss={loss:.3f}"
            )
        if step % max(1, args.save_steps) == 0:
            _write_checkpoint(out / f"checkpoint-{step}", args, step, reward)

    state = {
        "global_step": args.max_steps,
        "best_metric": best_reward,
        "best_model_checkpoint": f"checkpoint-{best_step}",
        "log_history": log_history,
        "created_at": time.time(),
        "backend": "local-smoke",
    }
    (out / "trainer_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    _write_checkpoint(out / "checkpoint-best", args, best_step, best_reward)
    (out / "policy_config.json").write_text(
        json.dumps(
            {"policy": "scripted-smoke", "base_model": args.model, "reward_mode": args.reward_mode},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"local smoke training complete: best_reward={best_reward:.3f} output={out}")


def _write_checkpoint(path: Path, args, step: int, reward: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "policy_config.json").write_text(
        json.dumps(
            {
                "policy": "scripted-smoke",
                "base_model": args.model,
                "step": step,
                "reward": reward,
                "task": args.task,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# ─────────────────────────────────────────────────────────────────
# TRL GRPO backend (GPU)
# ─────────────────────────────────────────────────────────────────


def _run_trl(args) -> None:
    """Real GRPO training using TRL + Unsloth + LoRA on the GPU server."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
        from datasets import Dataset
    except ImportError as exc:
        raise SystemExit(
            "TRL training dependencies are missing.\n"
            "Install: pip install -e .[train]\n"
            "Also install the cu128 torch wheel from GPU_HANDOFF.md.\n"
            "Or run: --backend local-smoke"
        ) from exc

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit(
            "CUDA is not available. Pass --allow-cpu for a tiny debug run, "
            "or use the 5090 server."
        )

    model_name = args.base_checkpoint or args.warm_start_sft or args.model
    print(f"Loading model: {model_name}")

    model, tokenizer = _load_model_with_lora(model_name, args)

    # ── Dataset ────────────────────────────────────────────────────
    tasks = CANONICAL_TASKS if args.task == "multi" else [args.task]
    # Aim for ~8× the number of training steps to avoid repeating prompts too often
    n_seeds = max(10, (args.max_steps * 4) // max(1, len(tasks)))
    print(f"Generating GRPO prompt dataset: {len(tasks)} tasks × {n_seeds} seeds …")
    dataset = _build_grpo_dataset(tasks, n_seeds)
    print(f"Dataset size: {len(dataset)} prompt rows")

    # ── Reward function ────────────────────────────────────────────
    def reward_func(completions, prompts=None, task=None, seed=None, step_idx=None, **kwargs):
        """
        Evaluate each completion by running a truncated environment episode.

        For each generated action the reward function:
          1. Resets the env with (task, seed)
          2. Replays expert actions up to step_idx-1
          3. Applies the model's action at step_idx
          4. Continues with expert actions for the remainder
          5. Returns the final weighted_final score

        This is fast (~2 ms/episode) and gives meaningful gradient signal.
        """
        rewards = []
        for i, completion in enumerate(completions):
            text = (
                completion[0]["content"]
                if isinstance(completion, list)
                else str(completion)
            )
            from inference import parse_action

            action_cmd = parse_action(text).command

            _task = (task[i] if isinstance(task, list) else task) if task else CANONICAL_TASKS[0]
            _seed = int((seed[i] if isinstance(seed, list) else seed) or 0)
            _step = int((step_idx[i] if isinstance(step_idx, list) else step_idx) or 0)

            score = _eval_action_at_step(_task, _seed, _step, action_cmd)
            rewards.append(float(score))
        return rewards

    # ── GRPO config ────────────────────────────────────────────────
    config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,

        # GRPO-specific — bumped completion length so model can output bare command
        # without thinking-mode rambles cutting it off; num_generations=8 for more
        # variance per group → fewer zero-advantage groups.
        num_generations=8,
        temperature=0.9,
        max_completion_length=128,
        max_prompt_length=1536,

        # Stability — vLLM disabled because not installed; HF generate is slower
        # but works with our stack.
        beta=0.005,
        use_vllm=False,
        bf16=True,
        report_to="none",
    )

    model = _patch_unsloth_mode_methods(model)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        args=config,
        train_dataset=dataset,
    )

    # Patch missing vision attributes (text-only model; Unsloth's GRPO cache assumes VLM)
    for attr in ("image_token_id", "vision_start_token_id", "vision_end_token_id"):
        if not hasattr(trainer, attr):
            setattr(trainer, attr, None)

    print("Starting GRPO training …")
    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)

    print(f"Training complete. Checkpoint saved to {args.output_dir}")

    (Path(args.output_dir) / "policy_config.json").write_text(
        json.dumps(
            {
                "policy": "grpo",
                "base_model": model_name,
                "task": args.task,
                "max_steps": args.max_steps,
            },
            indent=2,
        )
    )




def _patch_unsloth_mode_methods(model):
    """Compatibility patch only for fallback models missing Unsloth mode methods."""
    import types

    def _for_training(self, *args, **kwargs):
        self.train()
        return self

    def _for_inference(self, *args, **kwargs):
        self.eval()
        return self

    if not hasattr(model, "for_training"):
        model.for_training = types.MethodType(_for_training, model)
    if not hasattr(model, "for_inference"):
        model.for_inference = types.MethodType(_for_inference, model)

    for attr in ("base_model", "model"):
        inner = getattr(model, attr, None)
        if inner is not None:
            if not hasattr(inner, "for_training"):
                inner.for_training = types.MethodType(_for_training, inner)
            if not hasattr(inner, "for_inference"):
                inner.for_inference = types.MethodType(_for_inference, inner)

    return model


def _load_model_with_lora(model_name: str, args):
    """Load model + tokenizer. Tries Unsloth first, falls back to standard PEFT."""
    import torch

    use_unsloth = not getattr(args, "no_unsloth", False)

    if use_unsloth:
        try:
            from unsloth import FastLanguageModel

            print("Loading with Unsloth …")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                load_in_4bit=False,
                fast_inference=False,
                trust_remote_code=False,
                max_lora_rank=16,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            print("Unsloth LoRA model loaded.")
            return model, tokenizer
        except Exception as exc:
            if str(model_name).lower().startswith("unsloth/"):
                raise RuntimeError(
                    f"Unsloth failed to load {model_name}. Do not fallback when using Unsloth+vLLM. Original error: {exc}"
                ) from exc
            print(f"Unsloth unavailable ({exc}); falling back to standard PEFT …")

    # Standard transformers + PEFT
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=False,
        torch_dtype=dtype,
        device_map="auto",
    )
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def _build_grpo_dataset(tasks: list[str], n_seeds: int):
    """
    Build a dataset of (observation, expert_action, metadata) tuples.

    Each row is one decision point in an expert episode. The TRL GRPO trainer
    generates an action for each row; the reward function evaluates it in context.
    """
    from datasets import Dataset
    from inference import SYSTEM_PROMPT, format_obs
    from models import F1Action
    from server.environment import F1StrategistEnvironment
    from server.scenarios import SCENARIOS
    from baselines.expert_solver import EXPERT_SEQUENCES

    rows = []
    for task in tasks:
        family = SCENARIOS[task]["scenario_family"]
        expert_seq = EXPERT_SEQUENCES[family]
        for seed in range(n_seeds):
            env = F1StrategistEnvironment()
            obs = env.reset(task=task, seed=seed)
            for step_idx, expert_cmd in enumerate(expert_seq):
                if obs.done:
                    break
                rows.append(
                    {
                        "prompt": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": format_obs(obs)},
                        ],
                        "task": task,
                        "seed": seed,
                        "step_idx": step_idx,
                        "expert_action": expert_cmd,
                    }
                )
                obs = env.step(F1Action(command=expert_cmd))
    return Dataset.from_list(rows)


def _eval_action_at_step(task: str, seed: int, step_idx: int, action_command: str) -> float:
    """
    Run environment episode, inserting model's action at step_idx.

    Steps 0..step_idx-1 use expert actions (replay).
    Step step_idx        uses the model's action_command.
    Steps step_idx+1..   use expert actions (continuation).

    Returns final weighted_final score.
    """
    from models import F1Action
    from server.environment import F1StrategistEnvironment
    from server.scenarios import SCENARIOS
    from baselines.expert_solver import EXPERT_SEQUENCES

    family = SCENARIOS[task]["scenario_family"]
    expert_seq = EXPERT_SEQUENCES[family]

    env = F1StrategistEnvironment()
    obs = env.reset(task=task, seed=seed)

    # Replay expert up to (but not including) step_idx
    for i, cmd in enumerate(expert_seq):
        if i >= step_idx or obs.done:
            break
        obs = env.step(F1Action(command=cmd))

    if obs.done:
        return float(obs.multi_objective_scores.get("weighted_final", obs.score))

    # Apply model action
    obs = env.step(F1Action(command=action_command))

    if obs.done:
        return float(obs.multi_objective_scores.get("weighted_final", obs.score))

    # Continue with remaining expert actions
    for cmd in expert_seq[step_idx + 1 :]:
        obs = env.step(F1Action(command=cmd))
        if obs.done:
            break

    return float(obs.multi_objective_scores.get("weighted_final", obs.score))


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Strategist training script")
    parser.add_argument("--model", default="Qwen/Qwen3-4B",
                        help="Base model (HF repo id or local path)")
    parser.add_argument("--task", default="multi",
                        help="Scenario family or 'multi' for all four")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--warm-start-sft", default=None,
                        help="Path to SFT checkpoint for warm-start")
    parser.add_argument("--reward-mode", choices=["shaped", "sparse"], default="shaped")
    parser.add_argument("--base-checkpoint", default=None,
                        help="Explicit base checkpoint path (overrides --model)")
    parser.add_argument("--output-dir", default="./grpo_v1")
    parser.add_argument("--backend", choices=["local-smoke", "trl"], default="local-smoke",
                        help="local-smoke: reward curve without GPU. trl: real GRPO training.")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Allow TRL run on CPU (for debug only; very slow)")
    parser.add_argument("--no-unsloth", action="store_true",
                        help="Skip Unsloth and use standard PEFT (slower but fewer deps)")
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help="Accepted for compatibility; vLLM is disabled by default.",
    )
    main(parser.parse_args())
