#!/usr/bin/env bash
set -euo pipefail

BIN=${BIN:-./build/shape_gemm}
M=${M:-128}
N=${N:-4096}
K=${K:-4096}

KERNEL_A=${KERNEL_A:-m64_n64_k16_t4x4}
KERNEL_B=${KERNEL_B:-m128_n128_k8_t8x8}

NCU=${NCU:-ncu}

echo "Profiling $KERNEL_A"
sudo "$NCU" --set full \
  "$BIN" "$M" "$N" "$K" \
  --kernel "$KERNEL_A" \
  --profile-once

echo
echo "Profiling $KERNEL_B"
sudo "$NCU" --set full \
  "$BIN" "$M" "$N" "$K" \
  --kernel "$KERNEL_B" \
  --profile-once
