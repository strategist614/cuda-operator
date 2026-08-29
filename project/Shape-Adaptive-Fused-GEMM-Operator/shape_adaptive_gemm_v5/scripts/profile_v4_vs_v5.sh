#!/usr/bin/env bash
set -euo pipefail

BIN=${BIN:-./build/shape_gemm}
M=${M:-128}
N=${N:-4096}
K=${K:-4096}

PIPE=${PIPE:-m128_n64_k16_t8x4_pipe}
WARP=${WARP:-m128_n64_k16_t8x4_warp}

NCU=${NCU:-ncu}

echo "============================================"
echo "V4 pipeline: $PIPE"
echo "============================================"

sudo "$NCU" --set full \
  "$BIN" "$M" "$N" "$K" \
  --kernel "$PIPE" \
  --profile-once

echo
echo "============================================"
echo "V5 warp pipeline: $WARP"
echo "============================================"

sudo "$NCU" --set full \
  "$BIN" "$M" "$N" "$K" \
  --kernel "$WARP" \
  --profile-once
