# demo-assets/

Public artefacts that go with the submission. Everything in this folder is
referenced by the README, the blog, or the video.

## Contents

| File | Owner | When | What |
|---|---|---|---|
| `blog-post.md`              | Person 2 | Phase 5 | HF Hub blog draft |
| `video-script.md`           | Person 2 | Phase 5 | YouTube video script (~1:45) |
| `hf-space-link.txt`         | Person 2 | After Space deploy | URL of the live Space |
| `hf-blog-link.txt`          | Person 2 | After blog publish | URL of the published blog |
| `youtube-link.txt`          | Person 2 | After video upload | URL of the video |
| `submission-confirmation.png` | Both | After form submit | Screenshot of submission |
| `*.gif`                     | Person 1 | Phase 5 | Visualizer rollouts (shipped with blog) |

## Workflow

1. Person 1 generates GIFs via `python rollout.py --task <name> --mode <mode> --render`
   then moves the result here: `mv captures/*.gif demo-assets/`
2. Person 2 polishes `blog-post.md`, embeds GIFs, publishes to HF Hub, saves URL
   to `hf-blog-link.txt`
3. Person 2 records the video using the GIFs + the eval bar chart, uploads to
   YouTube unlisted, saves URL to `youtube-link.txt`
4. Person 2 updates the README "Try it" section to reference these URLs

The `.txt` files start as placeholders; populate them after each publish step.
