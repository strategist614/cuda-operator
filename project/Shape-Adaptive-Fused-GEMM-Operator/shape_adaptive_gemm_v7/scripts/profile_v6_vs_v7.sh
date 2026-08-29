#!/usr/bin/env bash
set -euo pipefail

BIN=${BIN:-./build/shape_gemm}
M=${M:-128}
N=${N:-4096}
K=${K:-4096}
NCU=${NCU:-ncu}

V6=${V6:-tcv6_m128_n64_k16_w64x32}
V7=${V7:-tcv7_m128_n64_k16_w64x32_fp}

echo "============================================"
echo "V6 baseline: $V6"
echo "============================================"

sudo "$NCU" --set full \
  "$BIN" "$M" "$N" "$K" \
  --dtype fp16 \
  --kernel "$V6" \
  --profile-once

echo
echo "============================================"
echo "V7 fragment pipeline: $V7"
echo "============================================"

sudo "$NCU" --set full \
  "$BIN" "$M" "$N" "$K" \
  --dtype fp16 \
  --kernel "$V7" \
  --profile-once
