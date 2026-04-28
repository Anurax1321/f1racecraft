# Historical Race Replays

Frozen historical-race scenarios for **post-training comparison**.

After the GRPO checkpoint exists, we run the trained strategist against a real
race scenario (e.g. 2024 Spa, where Hamilton actually pitted lap 12 for inters)
and compare the agent's calls against the actual outcome. Storytelling demo.

## How to populate

```bash
pip install fastf1
python scripts/build_historical_replay.py \
    --year 2024 --event Spa --session R \
    --output data/historical_replays/spa_2024.json
```

`scripts/build_historical_replay.py` reads from FastF1's local cache (no
runtime API calls during rollouts — that's forbidden by `CLAUDE.md`).

## Schema

See the docstring at the top of `scripts/build_historical_replay.py`. Each
file follows the same shape as the hand-authored scenarios in
`server/scenarios.py` plus extra fields capturing what *actually* happened in
the real race so we can score the trained agent against reality.

## What we'll do with this

Once we have it (post-training):

1. Run the trained agent on this scenario via `inference.py`
2. Compare its pit-lap call to the actual race winner's pit lap
3. Compute "agent-vs-actual" deltas for the blog and video
4. One paragraph in `demo-assets/blog-post.md` about the historical replay
