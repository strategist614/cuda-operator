#!/usr/bin/env bash
set -euo pipefail

BIN=${BIN:-./build/topk_bench}
B=${B:-128}
N=${N:-65536}
K=${K:-16}
NCU=${NCU:-ncu}

echo "============================================"
echo "V3 fixed Top-16 WarpSelect"
echo "============================================"

sudo "$NCU" --set full \
  "$BIN" "$B" "$N" "$K" \
  --kernel batch \
  --profile-once

echo
echo "============================================"
echo "V4 K-specialized WarpSelect"
echo "============================================"

sudo "$NCU" --set full \
  "$BIN" "$B" "$N" "$K" \
  --kernel specialized \
  --profile-once
