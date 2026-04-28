# Deploying F1 Strategist to a HuggingFace Space

This is the step-by-step procedure to publish the environment as a Docker-SDK Space.
**This is a minimum requirement for the Meta PyTorch OpenEnv Hackathon (W2).**

The Space serves the OpenEnv HTTP API (`/reset`, `/step`, `/health`) that any client can
hit, including the `client.py` in this repo and any OpenEnv-compliant training loop.
Optionally it also exposes a Gradio web UI at `/web` for interactive demos.

## Prerequisites

1. A HuggingFace account. The team uses the `Deltasthic` org.
2. A HuggingFace **write** token: https://huggingface.co/settings/tokens
3. The `huggingface_hub` Python package: `pip install "huggingface_hub>=0.24"`
4. This repo checked out locally with all artefacts committed (`models.py`, `server/`, `Dockerfile`, README frontmatter).

## One-time setup

```bash
huggingface-cli login
```

Paste your write token when prompted. **Do NOT commit the token** to any repo.

## Step 1: Create the Space (one-time)

```bash
huggingface-cli repo create f1-strategist \
    --type space \
    --space_sdk docker \
    --organization Deltasthic
```

This creates the empty repo at `https://huggingface.co/spaces/Deltasthic/f1-strategist`.
If you are deploying to your personal account instead, omit the `--organization` flag.

## Step 2: Push the code

```bash
# From the repo root
git remote add hfspace https://huggingface.co/spaces/Deltasthic/f1-strategist
git push hfspace main:main
```

HF Spaces will detect the `sdk: docker` line in `README.md`, build the Dockerfile, and
start the container. Build takes 3–5 minutes. Visit the Space URL, the **App** tab, and
wait for status = Running.

If you prefer token-in-URL (less secure; rotates easily):

```bash
HF_TOKEN=hf_xxx
git push https://user:${HF_TOKEN}@huggingface.co/spaces/Deltasthic/f1-strategist main:main
```

## Step 3: Smoke-test the live Space

Once the Space is Running, its URL is `https://Deltasthic-f1-strategist.hf.space`
(HF rewrites `/` to `-` in the org-plus-name pair).

```bash
python tests/smoke_http.py --base-url https://Deltasthic-f1-strategist.hf.space
```

Expected output: all 5 HTTP checks pass:
- `/health` returns 200 with `{"status": "ok"}`
- `/reset` returns 200 with a valid `F1Observation` payload (`total_issues_count > 0`)
- `/step` with `STAY_OUT` returns 200 and `done == False`
- `/step` with garbage returns 200 and `reward == -0.02`
- `/step` with `DONE` flips `done == True`

If `/reset` or `/step` return HTTP 500, the container likely crashed at import time.
Check the Space **Logs** tab for the traceback and patch either the Dockerfile or
`server/`.

After confirming, save the URL to `demo-assets/hf-space-link.txt` so the blog and video
deliverables can reference it:

```bash
echo "https://huggingface.co/spaces/Deltasthic/f1-strategist" > demo-assets/hf-space-link.txt
```

## Step 4 (optional): Enable the Gradio web UI

The OpenEnv server can serve a Gradio panel at `/web` for interactive demos. To enable
on the Space:

1. Add `ENV ENABLE_WEB_INTERFACE=1` to the Dockerfile (already there in the template).
2. Push.
3. Visit `https://Deltasthic-f1-strategist.hf.space/web`.
4. Click `Reset`, type strategic commands into the text box, watch the race state update.

This is the simplest way for judges and viewers to interact with the env without writing
any client code.

## Updating the Space

Any `git push hfspace` triggers a rebuild automatically. No manual redeploy step.

```bash
# Make changes locally, commit
git add server/ models.py
git commit -m "fix: tyre wear curve correction"

# Push to GitHub for the team
git push origin main

# Push to the Space to trigger rebuild
git push hfspace main:main

# Wait 3-5 min, smoke-test
python tests/smoke_http.py --base-url https://Deltasthic-f1-strategist.hf.space
```

## Known constraints for HF Spaces Docker builds

- **Ephemeral filesystem.** Writes to paths other than `/tmp` or `/data` are lost on
  restart. The environment itself does not persist state, so this is fine. Postmortems
  written to `baselines/trajectories/postmortems.jsonl` *will* be lost on restart on
  the Space; for production we'd persist to `/data`. For the demo this is acceptable
  because postmortem retrieval works within a session.
- **Base image size.** The current `python:3.12-slim + openenv-core + FastAPI + numpy +
  matplotlib + pandas` footprint is about 600 MB. Well under the 50 GB free-tier cap.
- **No GPU.** The serving container is CPU-only. Inference and training happen on the
  client side (Colab, RTX 5090, whatever the user supplies). The env itself is pure
  Python and runs comfortably on CPU.
- **`app_port` must match Dockerfile `EXPOSE`.** Both are 8000. Do not change one
  without the other.
- **README YAML frontmatter must be at the top.** HF Spaces parses the first 24 lines
  for `sdk:` and `app_port:`. Any text before the YAML block breaks the build.

## Rollback

To take the Space offline without deleting it, click **Settings → Pause this Space** in
the HF web UI. To re-enable, click **Restart this Space**. No git action needed.

To delete the Space entirely, use the **Settings → Delete this Space** button. There is
no CLI for delete.

## Troubleshooting

- **Build fails at `pip install openenv-core`** → the version pin in the Dockerfile is
  incompatible with PyPI. Check `pip install openenv-core==<version>` works locally.
- **`/reset` returns 500 with `ImportError: cannot import name 'F1Action'`** → `models.py`
  is not being copied into the container. Check the Dockerfile `COPY` lines.
- **`/step` returns `done=True` after one call** → server is creating a fresh env per
  request instead of using the shared singleton. Check `server/app.py` matches the
  OpsTwin pattern (single `_shared_env = F1StrategistEnvironment()` at module scope).
- **Gradio interface hangs** → `ENABLE_WEB_INTERFACE` env var not set. Check
  `os.environ.get("ENABLE_WEB_INTERFACE")` in `server/app.py`.

## What to do after a successful deploy

1. Save the Space URL to `demo-assets/hf-space-link.txt`
2. Update the README "Try it" section with the Space URL
3. Commit and push to GitHub (NOT to hfspace) so the team has a paper trail
4. Share the URL in the team chat and tag it in `TODO.md` under W2
