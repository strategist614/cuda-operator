# Shape-Adaptive GEMM V4

V4 在 V3 的 memory-path specialization 基础上加入：

- `scalar` path
- `float4 vec4` path
- `software-pipelined vec4` path
- double-buffered shared memory
- global -> register prefetch
- shared-memory ping-pong stages
- V3 vs V4 NCU comparison script
- 独立 `gemm_cache_v4.csv`

目标 GPU：RTX 2080 / Turing / SM75。

**V4 没有使用 `cp.async`。**
`cp.async` 是后续架构的能力；SM75 版本使用软件流水：

```text
Current tile:
Shared -> Register -> FMA

同时提前发起：

Next tile:
Global -> Register prefetch

当前 tile compute 完成后：

Prefetched registers
       ↓
next shared stage
       ↓
__syncthreads()
       ↓
stage swap
```

---

## 1. 版本演进

```text
V0
Basic tiled GEMM

↓

V1
Kernel Registry
+ Autotuner

↓

V2
Performance Cache
+ Runtime Dispatch

↓

V3
Scalar / float4 Memory Paths
+ Alignment-aware Dispatch

↓

V4
Register Prefetch
+ Double-buffer Shared Memory
+ Software Pipeline
```

---

## 2. Kernel Registry

每一个 tile 现在有三条 path。

例如 V3 最优配置：

```text
BM = 128
BN = 64
BK = 16

TM = 8
TN = 4
```

在 V4 中变成：

```text
m128_n64_k16_t8x4_scalar
m128_n64_k16_t8x4_vec4
m128_n64_k16_t8x4_pipe
```

所以可以做非常干净的对照：

```text
same tile
same thread tile
same compute loop

only change:

memory / pipeline path
```

---

## 3. V4 software pipeline

V3：

```text
Global Load Tile 0
        ↓
Shared Tile 0
        ↓
__syncthreads
        ↓
Compute Tile 0
        ↓
__syncthreads
        ↓
Global Load Tile 1
```

load 和 compute 基本是阶段式串行。

V4：

```text
Prologue:
Global Tile 0
    ↓
Shared Stage 0

Mainloop:

Global Tile 1
    ↓
Register Prefetch
          \
           \ independent work
            \
Shared Stage 0
    ↓
Compute Tile 0
    ↓
FMA
          /
         /
Prefetch registers
    ↓
Shared Stage 1
    ↓
barrier
    ↓
swap stages
```

shared memory：

```cpp
As[2][BM][BK]
Bs[2][BK][BN]
```

使用 ping-pong：

```text
iteration 0:
read stage 0
prepare stage 1

iteration 1:
read stage 1
prepare stage 0
```

---

## 4. 为什么先 Global -> Register？

RTX 2080 没有 Ampere/Hopper 风格的 `cp.async`。

因此这里使用的是 software prefetch：

```text
Global Load
    ↓
prefetch register
```

然后执行当前 tile 的独立 FMA。

GPU warp scheduler / scoreboard 有机会让部分 global-load latency 与独立计算重叠。

计算结束：

```text
prefetch registers
      ↓
next shared stage
```

这不等同于硬件 async copy，但属于 Turing 上可研究的软件流水方式。

---

## 5. 代价

Pipeline 不是免费的。

### Shared Memory

普通 vec4：

```text
(BM*BK + BK*BN) * sizeof(float)
```

pipeline：

```text
2 *
(BM*BK + BK*BN) * sizeof(float)
```

例如：

```text
128x64x16
```

V3：

```text
12 KB shared/block
```

V4 pipe：

```text
24 KB shared/block
```

可能减少 active blocks / SM。

### Registers

还需要：

```text
a_prefetch[]
b_prefetch[]
```

所以 registers/thread 也可能上升。

因此存在 tradeoff：

```text
latency hiding ↑

vs

register pressure ↑
shared memory ↑
occupancy ↓
```

这也是为什么 V4 仍然让 autotuner 同时保留：

```text
scalar
vec4
pipe
```

---

# 6. 编译

```bash
rm -rf build

mkdir build
cd build

cmake ..
cmake --build . -j

cd ..
```

---

# 7. 先跑完整 Autotune

```bash
./build/shape_gemm \
  128 4096 4096 \
  --retune
```

默认 V4 cache：

```text
results/gemm_cache_v4.csv
```

V4 有：

```text
9 tile configs
x
3 paths

=
27 candidates
```

Autotuner 会自己判断：

```text
scalar
vec4
pipe
```

哪一种组合最快。

---

# 8. 最关键的三组对照

V3 的冠军：

```text
m128_n64_k16_t8x4_vec4

V3:
~2.044 ms
~2.102 TFLOPS
~25.6% cuBLAS
```

V4 建议直接测：

```bash
./build/shape_gemm \
  128 4096 4096 \
  --kernel m128_n64_k16_t8x4_scalar
```

```bash
./build/shape_gemm \
  128 4096 4096 \
  --kernel m128_n64_k16_t8x4_vec4
```

```bash
./build/shape_gemm \
  128 4096 4096 \
  --kernel m128_n64_k16_t8x4_pipe
```

记录：

```text
scalar latency
vec4 latency
pipe latency

TFLOPS

cuBLAS %
```

---

# 9. NCU：V3 vs V4

V3-style vec4：

```bash
sudo /usr/local/cuda-13.2/bin/ncu \
  --set full \
  ./build/shape_gemm \
  128 4096 4096 \
  --kernel m128_n64_k16_t8x4_vec4 \
  --profile-once
```

V4 pipe：

```bash
sudo /usr/local/cuda-13.2/bin/ncu \
  --set full \
  ./build/shape_gemm \
  128 4096 4096 \
  --kernel m128_n64_k16_t8x4_pipe \
  --profile-once
```

或者：

```bash
NCU=/usr/local/cuda-13.2/bin/ncu \
./scripts/profile_v3_vs_v4.sh
```

---

# 10. V4 最应该看的指标

重点比较：

```text
Registers / Thread

Shared Memory / Block

Achieved Occupancy

Active Warps / SM
Eligible Warps / Scheduler

SM Throughput

Memory Throughput
L1/TEX Throughput
L2 Throughput
DRAM Throughput

Warp Stall Reasons:
Long Scoreboard
Short Scoreboard
Wait
Barrier
LG Throttle
MIO Throttle
Not Selected
```

---

# 11. 要验证的 Hypothesis

V4 hypothesis：

```text
V3:
load next tile only after finishing current tile

↓

Global memory latency exposed
```

V4：

```text
issue next global load
       ↓
hold in registers
       ↓
compute current tile
       ↓
commit next shared stage
```

预期：

```text
Long Scoreboard ↓ ?
Eligible Warps ↑ ?
SM utilization ↑ ?
latency ↓ ?
```

但是如果观察到：

```text
Registers/thread ↑↑
Occupancy ↓
Barrier stall ↑
```

并且 pipe 比 vec4 更慢，那么结论就是：

```text
software pipeline hides some memory latency,
but resource pressure dominates on this tile.
```

这同样是有效的优化实验。

---

# 12. Shape Sweep

```bash
python3 scripts/sweep.py
```

生成：

```text
results/gemm_cache_v4.csv
```

后面可以统计：

```text
哪些 shape:
scalar wins

哪些 shape:
vec4 wins

哪些 shape:
pipe wins
```

这会进一步证明：

```text
最优 kernel 不仅取决于 BM/BN/BK，
还取决于 memory/pipeline strategy。
```

---

# 13. 下一阶段

如果 V4 pipeline 有明确收益：

```text
V5:
warp-level tiling
+
shared-memory layout / bank-conflict optimization
```

如果 V4 收益不明显：

先根据 NCU 判断到底卡在：

```text
register pressure
shared memory
barrier
dependency
instruction issue
```

然后再决定 V5。

之后再进入：

```text
FP16 / WMMA / Tensor Core
```

而不是现在就直接跳过去。
