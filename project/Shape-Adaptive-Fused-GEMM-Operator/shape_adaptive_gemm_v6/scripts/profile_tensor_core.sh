#!/usr/bin/env bash
set -euo pipefail

BIN=${BIN:-./build/shape_gemm}
M=${M:-128}
N=${N:-4096}
K=${K:-4096}

KERNEL=${KERNEL:-tc_m128_n128_k32_w64x64}
NCU=${NCU:-ncu}

sudo "$NCU" --set full \
  "$BIN" "$M" "$N" "$K" \
  --dtype fp16 \
  --kernel "$KERNEL" \
  --profile-once
