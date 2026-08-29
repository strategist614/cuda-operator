#!/usr/bin/env bash
set -euo pipefail

BIN=${BIN:-./build/shape_gemm}
M=${M:-128}
N=${N:-4096}
K=${K:-4096}

SCALAR=${SCALAR:-m128_n128_k8_t8x8_scalar}
VEC4=${VEC4:-m128_n128_k8_t8x8_vec4}

NCU=${NCU:-ncu}

echo "============================================"
echo "Scalar path: $SCALAR"
echo "============================================"

sudo "$NCU" --set full \
  "$BIN" "$M" "$N" "$K" \
  --kernel "$SCALAR" \
  --profile-once

echo
echo "============================================"
echo "Vectorized path: $VEC4"
echo "============================================"

sudo "$NCU" --set full \
  "$BIN" "$M" "$N" "$K" \
  --kernel "$VEC4" \
  --profile-once
