#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/jack/miniconda3/envs/main/bin/python}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

exec "$PYTHON" "$SCRIPT_DIR/train_v10.py" \
  --encoder-checkpoint /data/best_train_loss.pt \
  --data-root MSRS=/data/v7/data/MSRS \
  --data-root FMB=/data/v7/data/FMB \
  --data-root RoadScene=/data/v7/data/RoadScene \
  --data-root M3FD=/data/v7/data/M3FD \
  --output-dir /data/v10/runs/dual_encoder_3x3_m3fd1500 \
  --max-images 1500 \
  --samples-per-epoch 3000 \
  --epochs 3 \
  --image-size 256 \
  --device cuda
