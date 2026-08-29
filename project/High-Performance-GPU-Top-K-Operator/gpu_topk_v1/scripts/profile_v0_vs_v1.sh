#!/usr/bin/env bash
set -euo pipefail

BIN=${BIN:-./build/topk_bench}
B=${B:-128}
N=${N:-65536}
K=${K:-16}
NCU=${NCU:-ncu}

echo "============================================"
echo "V0 naive"
echo "============================================"

sudo "$NCU" --set full \
  "$BIN" "$B" "$N" "$K" \
  --kernel naive \
  --profile-once

echo
echo "============================================"
echo "V1 register + block merge"
echo "============================================"

sudo "$NCU" --set full \
  "$BIN" "$B" "$N" "$K" \
  --kernel register \
  --profile-once
