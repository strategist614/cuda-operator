#!/usr/bin/env bash
set -euo pipefail

BIN=${BIN:-./build/topk_bench}
B=${B:-128}
N=${N:-65536}
K=${K:-4}
KERNEL=${KERNEL:-warpselect_k4_v4}
NCU=${NCU:-ncu}

sudo "$NCU" --set full \
  "$BIN" "$B" "$N" "$K" \
  --kernel "$KERNEL" \
  --profile-once
