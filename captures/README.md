# captures/

Per-rollout traces and visualizer outputs. **Most files here are gitignored** —
they're regenerated on demand. Only files explicitly moved to `demo-assets/`
end up in the submission.

## Generation

```bash
# A single rollout, JSON + animated GIF
python rollout.py --task weather_roulette --seed 7 --mode trained --render
# Produces:
#   captures/rollout_<timestamp>.jsonl
#   captures/rollout_<timestamp>.log
#   captures/rollout_<timestamp>.gif
```

To ship a GIF with the blog/video:

```bash
mv captures/rollout_<timestamp>.gif demo-assets/trained-spa.gif
```

## Folder is intentionally empty in git

Only `README.md` is tracked here. See `.gitignore` — `captures/*.gif`,
`captures/*.mp4`, `captures/*.jsonl` are all excluded.
