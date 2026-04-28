# tests/

Smoke and unit tests. Run all from the repo root:

```bash
pytest tests/                          # unit tests
python tests/smoke_http.py             # HTTP smoke (local server)
python tests/smoke_all_scenarios.py    # per-scenario discriminative-reward smoke
```

| File | Phase | What it checks |
|---|---|---|
| `smoke_http.py`          | 1   | 5 HTTP checks (health, reset, step, garbage, done) |
| `smoke_all_scenarios.py` | 2   | expert ≥ 0.85, panic ≤ 0.40 per scenario family |
| `test_environment.py`    | 1   | reset/step invariants |
| `test_scoring.py`        | 2   | scoring is pure and deterministic |

`smoke_http.py` accepts `--base-url` so you can point it at the live HF Space:

```bash
python tests/smoke_http.py --base-url https://Deltasthic-f1-strategist.hf.space
```

This is the test the deploy procedure runs after each Space push.
