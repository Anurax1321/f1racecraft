# scripts/

Helper scripts for setup, calibration, and post-training analysis. None of these
run inside the HF Space container — all are operator-side, GPU-server or local.

| Script | Owner | When |
|---|---|---|
| `extract_track_csvs.py`        | Person 1 | Phase 0 setup |
| `calibrate_opponent_pace.py`   | Person 2 | Phase 0 setup |
| `calibrate_tyre_baseline.py`   | Person 2 | Phase 0 setup |
| `plot_training_curve.py`       | Person 2 | After each training run |
| `push_checkpoint.py`           | Person 2 | After Phase 3 finishes |
| `diff_ablation.py`             | Person 2 | After Phase 4 finishes |

Each script has its own argparse `--help`. Read it before invoking on the GPU
server — most have safe defaults but a few (`push_checkpoint.py`) push to
public HF repos.
