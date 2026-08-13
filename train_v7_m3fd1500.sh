#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/jack/miniconda3/envs/main/bin/python}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

exec "$PYTHON" "$SCRIPT_DIR/train_strong_baseline_continuous_causal.py" \
  --baseline-checkpoint /data/best_train_loss.pt \
  --init-geometry-checkpoint /data/baseline.pt \
  --data-root MSRS=/data/dataset/MSRS/train \
  --data-root FMB=/data/dataset/FMB/train \
  --data-root RoadScene=/data/dataset/RoadScene \
  --data-root M3FD=/data/dataset/M3FD \
  --output-dir /data/v7/runs/continuous_causal_v7_m3fd1500 \
  --max-images 1500 \
  --epochs 3 \
  --device cuda
