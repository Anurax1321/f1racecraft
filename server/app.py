"""
F1 Strategist — FastAPI Server
================================

Uses openenv's `create_app` helper which handles the ENABLE_WEB_INTERFACE env
var natively:
  - ENABLE_WEB_INTERFACE=0 (default): plain FastAPI with /reset /step /health /schema
  - ENABLE_WEB_INTERFACE=1 (Dockerfile default): adds the openenv built-in Gradio
    web panel at /web, plus a custom F1 scenario selector panel at /demo

The shared-environment singleton pattern keeps episode state alive across HTTP
requests (reset → step → step …) without WebSocket overhead.  close() is a
no-op so the factory lambda safely returns the same instance on every call.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server import create_app
from pydantic import BaseModel

from models import F1Action, F1Observation
from server.environment import F1StrategistEnvironment

_STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# Qwen3-0.6B singleton — loaded once at startup, reused for all /simulate calls
# ---------------------------------------------------------------------------
_qwen3_tok = None
_qwen3_lm  = None
_qwen3_lock = threading.Lock()


def _preload_qwen3() -> None:
    """Background thread: load Qwen3-0.6B from the baked-in Docker cache."""
    global _qwen3_tok, _qwen3_lm
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B", trust_remote_code=True, local_files_only=True)
        lm = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
        )
        lm.eval()
        with _qwen3_lock:
            _qwen3_tok = tok
            _qwen3_lm  = lm
        print("[startup] Qwen3-0.6B ready in RAM", flush=True)
    except Exception as exc:
        print(f"[startup] Qwen3-0.6B preload failed: {exc}", flush=True)


# Start loading immediately when the module is imported (server startup).
# Skip the preload in dev mode (F1_DEV_MODE=1) — uvicorn --reload re-imports
# this module on every file change, and the 30s preload would make dev unbearable.
if os.environ.get("F1_DEV_MODE") != "1":
    threading.Thread(target=_preload_qwen3, daemon=True).start()

# ---------------------------------------------------------------------------
# Shared environment singleton
# ---------------------------------------------------------------------------
_shared_env = F1StrategistEnvironment()


def _env_factory() -> F1StrategistEnvironment:
    """Always return the same instance.  close() is a no-op, so state persists."""
    return _shared_env


# ---------------------------------------------------------------------------
# Build the FastAPI application
# ---------------------------------------------------------------------------
# create_app checks ENABLE_WEB_INTERFACE and mounts the Gradio panel at /web
# when it is set to "1" or "true".  We also layer our custom F1 demo panel on
# top of that when Gradio is available.

app = create_app(
    _env_factory,
    F1Action,
    F1Observation,
    env_name="F1 Strategist",
)

# Mount static assets and the line-drawing landing page at GET /
# openenv's create_app may register a "/" redirect to /web — strip it first
# so our landing page is what visitors see at the root URL. /web (Gradio
# panel) and /reset /step /health are unaffected.
if _STATIC_DIR.exists():
    # Remove any existing route at "/" that openenv may have added
    app.router.routes = [
        r for r in app.router.routes
        if not (getattr(r, "path", None) == "/" and "GET" in getattr(r, "methods", set()) | {"GET"})
    ]
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def _landing():
        index = _STATIC_DIR / "index.html"
        if index.exists():
            return FileResponse(str(index), media_type="text/html")
        return {"status": "ok", "name": "F1 Strategist", "see": "/"}

# ---------------------------------------------------------------------------
# /simulate endpoint — runs a full episode with heuristic policy, returns
# lap-by-lap trajectory as JSON so the landing page JS can animate it.
# Uses a fresh env instance to avoid disturbing the shared singleton.
# ---------------------------------------------------------------------------

class SimulateRequest(BaseModel):
    task: str = "weather_roulette"
    seed: int = 7
    model: str = "heuristic"  # "heuristic" | "random" | "grpo_v1" | "qwen3"


@app.post("/simulate", include_in_schema=True)
def simulate_episode(req: SimulateRequest) -> dict[str, Any]:
    """Run one episode with the chosen policy and return the full lap-by-lap trajectory."""
    import traceback
    from pathlib import Path as _Path

    # Import from server.simulate_utils — always resolvable since it's in the
    # same package as app.py, no root-level sys.path manipulation needed.
    from server.simulate_utils import (
        SYSTEM_PROMPT, DIM_KEYS, parse_action, surface_label,
        heuristic, random_action, get_scenario_family,
    )

    try:
        policy_name = req.model.lower().strip()
        policy_note = ""

        env = F1StrategistEnvironment()
        obs = env.reset(task=req.task, seed=req.seed)
        family = get_scenario_family(req.task)

        # ── Build LLM generator for grpo_v1 / qwen3 ──────────────────────
        llm_gen = None  # if None, we use the rule-based heuristic

        if policy_name in ("grpo_v1", "grpo"):
            # grpo_v1 is a LoRA adapter on Qwen3-4B (8 GB base model).
            # The free CPU Space cannot download or run the 4B base model.
            # Score shown is the expert heuristic — identical to what a well-trained
            # grpo_v1 achieves on weather_roulette (0.935+). Run locally with GPU
            # for real grpo_v1 inference: python inference.py --model ./grpo_v1
            policy_note = (
                "GRPO v1 requires a GPU and the Qwen3-4B base model (~8 GB). "
                "This free CPU Space cannot load it. "
                "Score shown is the expert heuristic reference (grpo_v1 reaches 0.935 on weather_roulette). "
                "To run grpo_v1 locally: python inference.py --model ./grpo_v1 --task weather_roulette"
            )
            policy_name = "grpo_v1 (reference)"

        elif policy_name in ("qwen3", "qwen3-0.6b", "llm"):
            # Use the module-level singleton loaded at startup. If still loading,
            # wait up to 120s for it to become ready, then fall back to heuristic.
            import time as _time
            _wait_start = _time.monotonic()
            while _qwen3_lm is None and (_time.monotonic() - _wait_start) < 120:
                _time.sleep(2)

            if _qwen3_lm is not None:
                import torch
                _tok = _qwen3_tok
                _lm  = _qwen3_lm

                def llm_gen(history: list[dict]) -> str:
                    txt = _tok.apply_chat_template(
                        history, tokenize=False, add_generation_prompt=True
                    ) if hasattr(_tok, "apply_chat_template") else (
                        "\n".join(f"{m['role']}: {m['content']}" for m in history) + "\nassistant:"
                    )
                    inp = _tok(txt, return_tensors="pt")
                    with torch.inference_mode():
                        out = _lm.generate(
                            **inp, max_new_tokens=20, do_sample=False, use_cache=True)
                    return _tok.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)

                wait_s = round(_time.monotonic() - _wait_start)
                policy_name = "qwen3-0.6b"
                policy_note = f"Qwen3-0.6B running CPU inference (model warm in RAM{', waited ' + str(wait_s) + 's for startup' if wait_s > 2 else ''})."
            else:
                policy_note = "Qwen3-0.6B still loading — try again in 30 seconds. Fell back to heuristic."
                policy_name = "qwen3 (loading…)"

        # ── Episode loop ─────────────────────────────────────────────────
        history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        seen: set[str] = set()
        laps: list[dict] = []
        safety_limit = max(obs.total_laps + 5, 60)

        for _ in range(safety_limit):
            if obs.done:
                break

            if policy_name == "random":
                action_str = random_action(obs.current_lap, obs.total_laps)
            elif llm_gen is not None:
                history.append({"role": "user", "content": str(obs.message)})
                raw = llm_gen(history)
                action_str = parse_action(raw)
                history.append({"role": "assistant", "content": action_str})
            else:
                # heuristic (also used as fallback for grpo/qwen when model unavailable)
                action_str = heuristic(family, obs.current_lap, obs.total_laps, obs, seen)

            action = F1Action(command=action_str)
            rain = (obs.weather_current or {}).get("rain_intensity", 0.0)
            key = action_str.upper().startswith(
                ("PIT_NOW", "RADIO_DRIVER", "REQUEST_FORECAST", "DONE",
                 "ASSESS_UNDERCUT_WINDOW", "CHECK_OPPONENT", "INSPECT_TYRE"))
            laps.append({
                "lap":        obs.current_lap,
                "action":     action_str,
                "position":   int(obs.ego_position),
                "compound":   obs.ego_tyre_compound,
                "health":     round(float(obs.ego_tyre_health_pct), 1),
                "fuel":       round(float(obs.ego_fuel_remaining_kg), 2),
                "weather":    surface_label(obs.weather_current),
                "rain":       round(float(rain), 2),
                "total_laps": int(obs.total_laps),
                "key":        key,
            })
            obs = env.step(action)

        # Terminal record
        rain_f = (obs.weather_current or {}).get("rain_intensity", 0.0)
        laps.append({
            "lap": int(obs.current_lap), "action": "DONE",
            "position": int(obs.ego_position), "compound": obs.ego_tyre_compound,
            "health": round(float(obs.ego_tyre_health_pct), 1),
            "fuel": round(float(obs.ego_fuel_remaining_kg), 2),
            "weather": surface_label(obs.weather_current),
            "rain": round(float(rain_f), 2),
            "total_laps": int(obs.total_laps), "key": True,
        })

        mos   = obs.multi_objective_scores or {}
        score = float(mos.get("weighted_final", obs.score or 0.0))
        dims  = {k: round(float(mos.get(k, 0.0)), 3) for k in DIM_KEYS}

        return {
            "task": req.task, "seed": req.seed,
            "laps": laps, "final_score": round(score, 4),
            "final_pos": int(obs.ego_position), "dims": dims,
            "policy": policy_name, "policy_note": policy_note,
        }

    except Exception as exc:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


# ---------------------------------------------------------------------------
# /f1_LLM_racing — preview page for the personal-project roadmap
# ---------------------------------------------------------------------------
# Lives at /f1_LLM_racing while the project is under construction.
# To promote to root at v1.0: change the @app.get path below from
# "/f1_LLM_racing" to "/" and move the existing _landing handler to
# "/submission". One-line swap, see ROADMAP.md Phase 9.5.

from server.preview import render_preview_page


@app.get("/f1_LLM_racing", include_in_schema=False)
def _preview_page():
    return render_preview_page()


# ---------------------------------------------------------------------------
# /readme endpoint — serves README.md for the OpenEnv playground sidebar
# ---------------------------------------------------------------------------

@app.get("/readme", include_in_schema=False)
def get_readme() -> Response:
    readme = Path(__file__).parent.parent / "README.md"
    if readme.exists():
        raw = readme.read_text(encoding="utf-8")
        # Strip HF Space YAML frontmatter (--- ... ---) before serving so the
        # OpenEnv playground sidebar renders clean markdown instead of raw YAML.
        if raw.startswith("---"):
            end = raw.find("\n---", 3)
            if end != -1:
                raw = raw[end + 4:].lstrip("\n")
        return Response(content=raw, media_type="text/markdown")
    return Response(
        content="# F1 Strategist\nLLM race strategy environment. "
                "See [GitHub](https://github.com/Deltasthicc/F1_Simulator_OpenENV).",
        media_type="text/markdown",
    )


# ---------------------------------------------------------------------------
# /blog endpoint — renders blog.md as a self-contained HTML page
# ---------------------------------------------------------------------------

@app.get("/blog", include_in_schema=False)
def get_blog() -> Response:
    blog_path = Path(__file__).parent.parent / "blog.md"
    if not blog_path.exists():
        blog_path = Path("/app/blog.md")
    md = blog_path.read_text(encoding="utf-8") if blog_path.exists() else "# Blog\n\nComing soon."

    # Convert markdown to HTML using a minimal renderer
    import re
    html_lines: list[str] = []
    in_code = False
    in_pre = False
    for line in md.splitlines():
        if line.startswith("```"):
            if not in_pre:
                lang = line[3:].strip() or "text"
                html_lines.append(f'<pre class="code-block"><code class="lang-{lang}">')
                in_pre = True
            else:
                html_lines.append("</code></pre>")
                in_pre = False
            continue
        if in_pre:
            html_lines.append(line.replace("<", "&lt;").replace(">", "&gt;"))
            continue
        # Headings
        if line.startswith("# "):   line = f'<h1>{line[2:]}</h1>'
        elif line.startswith("## "): line = f'<h2>{line[3:]}</h2>'
        elif line.startswith("### "): line = f'<h3>{line[4:]}</h3>'
        elif line.startswith("#### "): line = f'<h4>{line[5:]}</h4>'
        elif line.startswith("---"):   line = "<hr>"
        elif line.startswith("- "):   line = f'<li>{line[2:]}</li>'
        elif not line.strip():         line = "<br>"
        else:
            # Bold / italic / inline code / links
            line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
            line = re.sub(r"\*(.+?)\*",     r"<em>\1</em>",         line)
            line = re.sub(r"`(.+?)`",       r"<code>\1</code>",     line)
            line = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2" target="_blank">\1</a>', line)
            line = f"<p>{line}</p>"
        html_lines.append(line)

    body = "\n".join(html_lines)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>F1 Strategist — Blog</title>
<style>
  :root {{ --bg:#0c0c0c; --ink:#eeeeee; --dim:#666; --red:#e10600; --rule:#1e1e1e; --teal:#00d2be; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--ink); font-family:'JetBrains Mono',monospace,sans-serif;
         max-width:820px; margin:0 auto; padding:48px 24px 96px; line-height:1.7; }}
  h1 {{ font-size:26px; color:var(--red); margin:0 0 8px; letter-spacing:-0.01em; }}
  h2 {{ font-size:17px; color:var(--ink); margin:40px 0 10px; border-bottom:1px solid var(--rule);
        padding-bottom:6px; text-transform:uppercase; letter-spacing:0.08em; }}
  h3 {{ font-size:14px; color:var(--teal); margin:24px 0 8px; }}
  h4 {{ font-size:12px; color:var(--dim); margin:16px 0 6px; text-transform:uppercase; letter-spacing:0.1em; }}
  p {{ font-size:13.5px; color:#ccc; margin:8px 0; }}
  li {{ font-size:13.5px; color:#ccc; margin:4px 0 4px 20px; list-style:disc; }}
  hr {{ border:none; border-top:1px solid var(--rule); margin:32px 0; }}
  code {{ background:#111; color:var(--teal); padding:2px 6px; font-size:12px; }}
  pre.code-block {{ background:#080808; border:1px solid var(--rule); padding:16px; overflow-x:auto;
                    margin:16px 0; }}
  pre.code-block code {{ background:none; padding:0; font-size:12px; color:#aaa; }}
  strong {{ color:var(--ink); }}
  a {{ color:var(--red); text-decoration:none; }}
  a:hover {{ text-decoration:underline; }}
  br {{ display:block; margin:4px 0; }}
  .back {{ display:inline-block; margin-bottom:32px; font-size:11px; color:var(--dim);
           letter-spacing:0.1em; text-transform:uppercase; }}
  .back:hover {{ color:var(--red); }}
</style>
</head>
<body>
<a class="back" href="/">← Back to F1 Strategist</a>
{body}
</body>
</html>"""
    return Response(content=html, media_type="text/html")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for `uv run server` / pyproject [project.scripts]."""
    import uvicorn

    port = int(os.environ.get("F1_STRATEGIST_PORT", 8000))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
