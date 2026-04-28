# F1 Strategist server image.
# Starts the FastAPI app on port 8000.
FROM python:3.12-slim

WORKDIR /app

# System build tools needed by bitsandbytes / sentencepiece
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

# ── Core runtime deps ──────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "openenv-core>=0.2.3" \
    "fastapi>=0.104.0" \
    "uvicorn>=0.24.0" \
    "pydantic>=2.0.0" \
    "numpy>=1.26.0" \
    "pandas>=2.0.0" \
    "matplotlib>=3.8.0"

# ── Demo / visualisation deps ──────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "gradio>=4.0.0" \
    "imageio>=2.31.0" \
    "pillow>=10.0.0"

# ── LLM inference deps (CPU-only torch to keep image lean) ─────────────────
# CPU-only build is ~700 MB vs ~2 GB for CUDA build
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    "transformers>=4.55.4" \
    "peft>=0.13.2" \
    "accelerate>=1.3" \
    "sentencepiece" \
    "protobuf" \
    "psutil"

# ── Pre-cache Qwen3-0.6B so /simulate?model=qwen3 loads instantly ──────────
# ~1.5 GB download; baked into the image so runtime has zero network wait.
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
RUN python - <<'PYEOF'
import os, torch
os.environ.setdefault("HF_HOME", "/app/.cache/huggingface")
from transformers import AutoTokenizer, AutoModelForCausalLM
print("Pre-caching Qwen/Qwen3-0.6B ...")
AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True)
del m
print("Qwen/Qwen3-0.6B cached OK")
PYEOF

# ── Copy application code ──────────────────────────────────────────────────
COPY models.py     /app/models.py
COPY client.py     /app/client.py
COPY inference.py  /app/inference.py
COPY server        /app/server
COPY data          /app/data

# ── Copy grpo_v1 LoRA adapter (132 MB) ────────────────────────────────────
# Only copy the essential files; intermediate checkpoints are not needed.
COPY grpo_v1/adapter_config.json        /app/grpo_v1/adapter_config.json
COPY grpo_v1/adapter_model.safetensors  /app/grpo_v1/adapter_model.safetensors
COPY grpo_v1/added_tokens.json          /app/grpo_v1/added_tokens.json
COPY grpo_v1/chat_template.jinja        /app/grpo_v1/chat_template.jinja
COPY grpo_v1/merges.txt                 /app/grpo_v1/merges.txt
COPY grpo_v1/policy_config.json         /app/grpo_v1/policy_config.json
COPY grpo_v1/special_tokens_map.json    /app/grpo_v1/special_tokens_map.json
COPY grpo_v1/tokenizer_config.json      /app/grpo_v1/tokenizer_config.json
COPY grpo_v1/tokenizer.json             /app/grpo_v1/tokenizer.json
COPY grpo_v1/vocab.json                 /app/grpo_v1/vocab.json
COPY grpo_v1/training_args.bin          /app/grpo_v1/training_args.bin

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=1

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
