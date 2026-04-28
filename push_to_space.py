"""
push_to_space.py — bypass HF Spaces' 1 GB LFS quota by uploading via API.

Usage:
    $env:HF_TOKEN = "hf_xxxxx"        # PowerShell
    python push_to_space.py "commit message here"

If no commit message given, uses "update".

Skips: grpo_v1 LFS blobs, .git, .venv, __pycache__, captures, logs.
"""
import os, sys
from huggingface_hub import upload_folder

if not os.environ.get("HF_TOKEN"):
    sys.exit("FATAL: HF_TOKEN env var not set. Run `$env:HF_TOKEN = 'hf_...'` first.")

msg = sys.argv[1] if len(sys.argv) > 1 else "update"

upload_folder(
    folder_path=".",
    repo_id="Deltasthic/f1-strategist",
    repo_type="space",
    ignore_patterns=[
        "grpo_v1/**",
        "*.safetensors", "*.bin", "*.pt",
        ".venv/**", ".git/**", "__pycache__/**", "*.pyc",
        "captures/**", "*.log",
        ".pytest_cache/**", "*.egg-info/**",
        "node_modules/**",
        ".env", ".env.*",
    ],
    commit_message=msg,
)
print(f"uploaded: {msg}")
