#!/usr/bin/env bash
set -euo pipefail

BIN=${BIN:-./build/topk_bench}
B=${B:-128}
N=${N:-65536}
K=${K:-16}
NCU=${NCU:-ncu}

echo "============================================"
echo "V2: per-thread Top-K + warp hierarchical merge"
echo "============================================"

sudo "$NCU" --set full \
  "$BIN" "$B" "$N" "$K" \
  --kernel warp \
  --profile-once

echo
echo "============================================"
echo "V3: warp-batch threshold + bitonic selection"
echo "============================================"

sudo "$NCU" --set full \
  "$BIN" "$B" "$N" "$K" \
  --kernel batch \
  --profile-once
