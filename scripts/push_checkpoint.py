"""Push a trained checkpoint folder to Hugging Face Hub."""

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./grpo_v1/checkpoint-best")
    parser.add_argument("--repo", default="Deltasthic/f1-strategist-qwen3-4b-grpo")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint folder not found: {checkpoint}")
    token = os.environ.get("HF_TOKEN")
    if args.dry_run:
        print(f"dry-run: would upload {checkpoint} to {args.repo}")
        return
    if not token:
        raise SystemExit("HF_TOKEN is not set. Put it in .env or export it before pushing.")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: pip install huggingface_hub") from exc
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(checkpoint), repo_id=args.repo, repo_type="model")
    card = Path("model_card.md")
    if card.exists():
        api.upload_file(
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="model",
        )
    print(f"uploaded {checkpoint} to {args.repo}")


if __name__ == "__main__":
    main()
