"""Render a side-by-side untrained-vs-trained comparison GIF.

For a given task + seed, runs both modes, renders the two rollout traces
into individual GIFs, then stacks them frame-by-frame into a single GIF
that judges/blog readers can scan in one glance.

Usage:
    python scripts/render_compare.py --task weather_roulette --seed 7
    python scripts/render_compare.py --task dry_strategy_sprint --seed 0 \\
        --output captures/before_after_dry.gif

Output:
    captures/<task>_untrained_seed<seed>.jsonl/.gif   (intermediates)
    captures/<task>_trained_seed<seed>.jsonl/.gif     (intermediates)
    captures/before_after_<task>.gif                  (combined)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rollout import run_rollout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="weather_roulette",
                        choices=["dry_strategy_sprint", "weather_roulette",
                                 "late_safety_car", "championship_decider"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="heuristic",
                        help="Passed through to rollout.py (heuristic for offline demos)")
    parser.add_argument("--output", default=None,
                        help="Defaults to captures/before_after_<task>.gif")
    args = parser.parse_args()

    captures = REPO_ROOT / "captures"
    captures.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] running untrained rollout (task={args.task}, seed={args.seed})")
    untrained_score, untrained_jsonl = run_rollout(
        model=args.model, task=args.task, seed=args.seed,
        mode="untrained", render=True, verbose=False,
    )

    print(f"[2/3] running trained rollout (task={args.task}, seed={args.seed})")
    trained_score, trained_jsonl = run_rollout(
        model=args.model, task=args.task, seed=args.seed,
        mode="trained", render=True, verbose=False,
    )

    print(f"[3/3] stacking GIFs into combined output")
    untrained_gif = untrained_jsonl.with_suffix(".gif")
    trained_gif = trained_jsonl.with_suffix(".gif")
    output = Path(args.output) if args.output else captures / f"before_after_{args.task}.gif"
    _stack_gifs_vertically(untrained_gif, trained_gif, output,
                           untrained_score=untrained_score,
                           trained_score=trained_score,
                           task=args.task)
    print(f"[done] wrote {output}")
    print(f"        untrained={untrained_score:.3f}  trained={trained_score:.3f}  "
          f"Δ={trained_score - untrained_score:+.3f}")


def _stack_gifs_vertically(top_path: Path, bottom_path: Path, output: Path,
                           untrained_score: float, trained_score: float,
                           task: str) -> None:
    """Read two GIFs, concat them with labels, write a new combined GIF."""
    from PIL import Image, ImageDraw, ImageFont

    top_frames = _load_gif_frames(top_path)
    bottom_frames = _load_gif_frames(bottom_path)
    n = max(len(top_frames), len(bottom_frames))
    # Pad shorter sequence by holding the last frame
    while len(top_frames) < n:
        top_frames.append(top_frames[-1].copy())
    while len(bottom_frames) < n:
        bottom_frames.append(bottom_frames[-1].copy())

    # Match widths (pad the narrower with white)
    max_w = max(top_frames[0].width, bottom_frames[0].width)

    label_h = 38
    title = f"{task.replace('_', ' ').title()} — same seed, both policies"
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except IOError:
        font = ImageFont.load_default()
        font_small = font

    combined_frames = []
    for tf, bf in zip(top_frames, bottom_frames):
        tf = _pad_to_width(tf, max_w)
        bf = _pad_to_width(bf, max_w)
        total_h = label_h + tf.height + label_h + bf.height
        canvas = Image.new("RGB", (max_w, total_h), color="white")
        draw = ImageDraw.Draw(canvas)

        # Top label
        draw.rectangle([0, 0, max_w, label_h], fill="#ef8a17")
        draw.text((12, 8),
                  f"BEFORE training (untrained Qwen)   final score: {untrained_score:.3f}",
                  fill="white", font=font)
        canvas.paste(tf, (0, label_h))

        # Bottom label
        draw.rectangle([0, label_h + tf.height, max_w, label_h + tf.height + label_h],
                       fill="#1a8754")
        draw.text((12, label_h + tf.height + 8),
                  f"AFTER GRPO training                final score: {trained_score:.3f}   "
                  f"(Δ {trained_score - untrained_score:+.3f})",
                  fill="white", font=font)
        canvas.paste(bf, (0, label_h + tf.height + label_h))

        # Title strip in the very bottom corner
        draw.text((max_w - 290, label_h + tf.height + 10),
                  title, fill="white", font=font_small)
        combined_frames.append(canvas)

    combined_frames[0].save(
        output, save_all=True, append_images=combined_frames[1:],
        duration=600, loop=0, optimize=True,
    )


def _load_gif_frames(gif_path: Path) -> list:
    from PIL import Image

    img = Image.open(gif_path)
    frames = []
    try:
        while True:
            frames.append(img.convert("RGB").copy())
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return frames


def _pad_to_width(img, target_w: int):
    from PIL import Image

    if img.width >= target_w:
        return img
    new = Image.new("RGB", (target_w, img.height), color="white")
    new.paste(img, ((target_w - img.width) // 2, 0))
    return new


if __name__ == "__main__":
    main()
