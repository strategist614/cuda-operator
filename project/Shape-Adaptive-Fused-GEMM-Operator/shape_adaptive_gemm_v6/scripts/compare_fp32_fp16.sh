#!/usr/bin/env bash
set -euo pipefail

BIN=${BIN:-./build/shape_gemm}
M=${M:-128}
N=${N:-4096}
K=${K:-4096}

echo "============================================"
echo "V5 FP32 SIMT"
echo "============================================"
"$BIN" "$M" "$N" "$K" \
  --dtype fp32 \
  --retune

echo
echo "============================================"
echo "V6 FP16 Tensor Core"
echo "============================================"
"$BIN" "$M" "$N" "$K" \
  --dtype fp16 \
  --retune
