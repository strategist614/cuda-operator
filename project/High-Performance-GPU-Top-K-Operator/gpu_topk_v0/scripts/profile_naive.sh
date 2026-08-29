#!/usr/bin/env bash
set -euo pipefail

BIN=${BIN:-./build/topk_bench}
B=${B:-128}
N=${N:-65536}
K=${K:-16}
NCU=${NCU:-ncu}

sudo "$NCU" --set full \
  "$BIN" "$B" "$N" "$K" \
  --warmup 1 \
  --repeat 1 \
  --groups 1
