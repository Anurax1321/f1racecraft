"""
HTTP smoke test against the OpenEnv server.

Boots a local server (or hits a passed --base-url) and confirms:
    1. /health returns 200 with {"status": "ok"}
    2. /reset returns 200 with valid F1Observation (total_issues_count > 0)
    3. /step with STAY_OUT returns 200, done == False
    4. /step with garbage returns reward == -0.02
    5. /step with DONE flips done == True

CLI:
    # Local
    python tests/smoke_http.py
    # Remote
    python tests/smoke_http.py --base-url https://Deltasthic-f1-strategist.hf.space
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from urllib.request import urlopen, Request


def post_json(url: str, payload: dict) -> dict:
    req = Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def get_json(url: str) -> dict:
    with urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode())


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_health(base_url: str, timeout_s: float = 45.0) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            h = get_json(f"{base_url}/health")
            if h.get("status") in {"ok", "healthy"}:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"server did not become healthy: {last_error}")


def start_local_server() -> tuple[str, subprocess.Popen]:
    port = find_free_port()
    env = os.environ.copy()
    env.setdefault("ENABLE_WEB_INTERFACE", "1")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    wait_for_health(base_url)
    return base_url, proc


def run(base_url: str) -> int:
    failures = 0

    # 1. /health
    try:
        h = get_json(f"{base_url}/health")
        assert h.get("status") in {"ok", "healthy"}, f"health: {h}"
        print("  ok: /health")
    except Exception as e:
        print(f"FAIL: /health -> {e}")
        failures += 1

    # 2. /reset
    try:
        obs = post_json(f"{base_url}/reset", {})
        total = obs.get("observation", {}).get("total_issues_count", 0)
        assert total > 0, f"reset returned total_issues_count={total}"
        print(f"  ok: /reset (total_issues_count={total})")
    except Exception as e:
        print(f"FAIL: /reset -> {e}")
        failures += 1

    # 3. /step STAY_OUT
    try:
        r = post_json(f"{base_url}/step", {"action": {"command": "STAY_OUT"}})
        assert r.get("done") is False, f"step STAY_OUT: done={r.get('done')}"
        print("  ok: /step STAY_OUT")
    except Exception as e:
        print(f"FAIL: /step STAY_OUT -> {e}")
        failures += 1

    # 4. /step garbage → -0.02
    try:
        r = post_json(f"{base_url}/step", {"action": {"command": "ZZZZ_NOT_A_COMMAND"}})
        assert abs(r.get("reward", 0.0) - (-0.02)) < 1e-6, f"step garbage: reward={r.get('reward')}"
        print("  ok: /step garbage (-0.02)")
    except Exception as e:
        print(f"FAIL: /step garbage -> {e}")
        failures += 1

    # 5. /step DONE
    try:
        r = post_json(f"{base_url}/step", {"action": {"command": "DONE"}})
        assert r.get("done") is True, f"step DONE: done={r.get('done')}"
        print("  ok: /step DONE")
    except Exception as e:
        print(f"FAIL: /step DONE -> {e}")
        failures += 1

    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=None,
        help="Existing server URL. If omitted, the smoke test starts a local server.",
    )
    args = parser.parse_args()
    proc = None
    try:
        base_url = args.base_url
        if base_url is None:
            base_url, proc = start_local_server()
        else:
            wait_for_health(base_url)
        print(f"Smoke testing {base_url}")
        failures = run(base_url)
        if failures:
            print(f"\n{failures} check(s) failed.")
            sys.exit(1)
        print("\nAll 5 checks passed.")
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
