#!/usr/bin/env bash
set -euo pipefail

BIN=${BIN:-./build/topk_bench}
CACHE=${CACHE:-results/topk_cache_v5.csv}

for K in 1 2 4 8 16; do
  echo "============================================================"
  echo "B=128 N=65536 K=$K"
  "$BIN" 128 65536 "$K" \
    --cache "$CACHE"
done
