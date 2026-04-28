#!/usr/bin/env bash
# Polls origin for new commits on the current branch and restarts the
# f1-strategist service if the remote has advanced. Safe-by-default:
# uses --ff-only so unpushed local work is never silently overwritten.
set -euo pipefail

REPO=/home/anurag/projects/F1_Simulator_OpenENV
SERVICE=f1-strategist.service

cd "$REPO"

git fetch --quiet origin

BRANCH=$(git rev-parse --abbrev-ref HEAD)
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse "origin/$BRANCH" 2>/dev/null || echo "$LOCAL")

if [ "$LOCAL" = "$REMOTE" ]; then
  exit 0
fi

echo "f1-strategist-update: $BRANCH advanced ($LOCAL -> $REMOTE), restarting"
# Service ExecStartPre will perform the actual ff-only pull as user anurag.
systemctl restart "$SERVICE"
