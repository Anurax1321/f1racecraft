"""Merge GRPO LoRA adapter into base Qwen3-4B and save as a standalone checkpoint.

Output dir is ready for `evaluate.py --model <path>` and HF Hub upload.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter dir")
    parser.add_argument("--out", required=True, help="Output dir for merged model")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    adapter_dir = Path(args.adapter)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    base_id = cfg["base_model_name_or_path"]
    print(f"[merge] base = {base_id}")
    print(f"[merge] adapter = {adapter_dir}")
    print(f"[merge] out = {out_dir}")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    print(f"[merge] base loaded, dtype={dtype}")

    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    print("[merge] adapter loaded")

    merged = peft_model.merge_and_unload()
    print("[merge] merged")

    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    print(f"[merge] saved to {out_dir}")


if __name__ == "__main__":
    main()
