"""SFT warm-start trainer for F1 Strategist.

Trains LoRA on Qwen3-4B from chat-format expert trajectories
(produced by `capture_everything.py`).

Recipe (from TRAINING.md):
  - LoRA r=16, target all attention + MLP projections
  - 3 epochs, lr 1e-5, batch 1 grad-accum 32
  - max_seq_length 2048
  - completion-only loss (so we only train on assistant tokens, not user prompt)

Output: ./sft_checkpoints_v1/ with epoch-wise checkpoints + final merged model.
"""

from __future__ import annotations

# Unsloth must be imported before trl/transformers/peft.
try:
    import unsloth  # noqa: F401
except Exception:
    pass


import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


def _load_model(model_name: str, no_unsloth: bool):
    if not no_unsloth:
        try:
            from unsloth import FastLanguageModel

            print(f"[sft] Loading {model_name} via Unsloth …")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                load_in_4bit=False,
                fast_inference=False,  # SFT path — no vLLM
                trust_remote_code=False,
                max_lora_rank=16,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            return model, tokenizer
        except Exception as exc:
            print(f"[sft] Unsloth failed ({exc}); falling back to PEFT …")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    lora_cfg = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen3-4B")
    parser.add_argument("--dataset", default="sft_dataset_v1.jsonl")
    parser.add_argument("--output-dir", default="./sft_checkpoints_v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--no-unsloth", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — SFT requires a GPU.")

    print(f"[sft] Loading dataset {args.dataset} …")
    ds = load_dataset("json", data_files=args.dataset, split="train")
    print(f"[sft] {len(ds)} training rows")
    print(f"[sft] First row keys: {list(ds[0].keys())}")

    model, tokenizer = _load_model(args.model, args.no_unsloth)

    # Pre-render messages → text using the chat template, so SFTTrainer's
    # text-field path is unambiguous (Unsloth's formatting_func interface is
    # restrictive about return shapes).
    # IMPORTANT: enable_thinking=False so train and eval format match. Qwen3's
    # default thinking-on inserts `<think>` markers the bare-command data
    # doesn't have — leads to a runaway rambling model at eval time.
    def _render(example):
        try:
            return {"text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False,
                add_generation_prompt=False, enable_thinking=False,
            )}
        except TypeError:
            return {"text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False,
            )}

    ds = ds.map(_render, remove_columns=[c for c in ds.column_names if c != "text"])
    print(f"[sft] Rendered to text-field. Sample (first 200 chars):")
    print(ds[0]["text"][:200].replace("\n", " | "))

    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=4,
        bf16=True,
        max_length=args.max_seq_length,
        packing=False,
        completion_only_loss=False,
        padding_free=False,
        dataset_text_field="text",
        report_to="none",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        gradient_checkpointing=False,  # Unsloth handles this
        dataset_num_proc=2,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=cfg,
        train_dataset=ds,
    )

    print("[sft] Starting training …")
    trainer.train()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out / "final"))
    tokenizer.save_pretrained(str(out / "final"))
    print(f"[sft] Saved final adapter to {out / 'final'}")

    (out / "sft_config.json").write_text(
        json.dumps(
            {
                "base_model": args.model,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "max_seq_length": args.max_seq_length,
                "dataset": args.dataset,
                "n_rows": len(ds),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
