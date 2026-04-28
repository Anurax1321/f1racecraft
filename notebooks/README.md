# notebooks/

| File | Owner | What |
|---|---|---|
| `f1_strategist_training_colab.ipynb` | Person 2 | Colab Free-T4 smoke (W1 minimum) |

The notebook is **the W1 minimum requirement** ("minimal TRL training script
runnable in Colab"). Don't break it. Run it end-to-end on a fresh Colab Free
T4 before final tagging.

## Pinning

Versions in the notebook must match `TRAINING.md` §environment. If you upgrade
one (`transformers`, `trl`, `peft`, `accelerate`, `liger-kernel`), upgrade
all four together — see the gotchas in `TRAINING.md`.
