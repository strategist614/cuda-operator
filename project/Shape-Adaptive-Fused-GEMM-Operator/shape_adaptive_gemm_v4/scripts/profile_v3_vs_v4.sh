#!/usr/bin/env bash
set -euo pipefail

BIN=${BIN:-./build/shape_gemm}
M=${M:-128}
N=${N:-4096}
K=${K:-4096}

VEC4=${VEC4:-m128_n64_k16_t8x4_vec4}
PIPE=${PIPE:-m128_n64_k16_t8x4_pipe}

NCU=${NCU:-ncu}

echo "============================================"
echo "V3-style Vec4: $VEC4"
echo "============================================"

sudo "$NCU" --set full \
  "$BIN" "$M" "$N" "$K" \
  --kernel "$VEC4" \
  --profile-once

echo
echo "============================================"
echo "V4 Software Pipeline: $PIPE"
echo "============================================"

sudo "$NCU" --set full \
  "$BIN" "$M" "$N" "$K" \
  --kernel "$PIPE" \
  --profile-once
