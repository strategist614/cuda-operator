# Shape-Adaptive GEMM V5

V5 在 V4 的 `float4 + software pipeline` 基础上继续进入 **warp-level tiling + shared-memory layout optimization**。

目标 GPU：

```text
NVIDIA GeForce RTX 2080
Turing
SM75
```

---

# V5 新增内容

V4 最优：

```text
BM=128
BN=64
BK=16

TM=8
TN=4

path=pipe
```

V5 为这类 tile 新增显式 warp tile：

```text
WM=32
WN=32
```

对应：

```text
Block Tile:
128 x 64

↓

4 warp rows
x
2 warp cols

=

8 warps
```

每个 warp：

```text
32 x 32 output tile
```

每个 thread：

```text
8 x 4 output tile
```

所以：

```text
WM/TM = 4
WN/TN = 8

4 x 8
=
32 lanes
```

一个 warp 正好完整覆盖一个 warp tile。

---

# 为什么做 Warp-Level Tiling？

V4 的线程映射主要来自：

```text
thread_row
thread_col
```

直接平铺整个 block tile。

V5 显式变成：

```text
block tile
    ↓
warp tile
    ↓
thread tile
```

即：

```text
CTA Tile
  ↓
Warp Tile
  ↓
Thread Tile
```

这更接近高性能 GEMM kernel 常见的层级组织。

---

# Shared Memory Padding

V4：

```cpp
As[2][BM][BK]
Bs[2][BK][BN]
```

V5 warp path：

```cpp
As[2][BM][BK + 1]
Bs[2][BK][BN + 1]
```

也就是给 shared-memory row stride 增加一个 FP32 padding。

目标是改变 bank 映射：

```text
row stride = BK
```

变成：

```text
row stride = BK + 1
```

以及：

```text
BN
→
BN + 1
```

从而测试 shared-memory access conflict 是否能下降。

注意：

**padding 不保证一定提升。**

V5 保留 V4 pipe path，autotuner 会直接判断：

```text
pipe
vs
warp + padded pipe
```

哪个更快。

---

# V5 Kernel Paths

V4 原有：

```text
scalar
vec4
pipe
```

V5 新增：

```text
warp
```

其中 `warp` 表示：

```text
float4 global load
+
register prefetch
+
double-buffer shared memory
+
explicit warp-level tiling
+
padded shared-memory layout
```

---

# Warp Candidates

V5 没有盲目给所有 tile 加 warp path。

目前只注册 4 组较重要的配置：

## 1

```text
BM=64
BN=64
BK=16

TM=4
TN=4

WM=16
WN=32
```

## 2

```text
BM=64
BN=128
BK=16

TM=4
TN=8

WM=16
WN=64
```

## 3 — V4 Champion

```text
BM=128
BN=64
BK=16

TM=8
TN=4

WM=32
WN=32
```

对应 kernel：

```text
m128_n64_k16_t8x4_warp
```

## 4

```text
BM=128
BN=128
BK=8

TM=8
TN=8

WM=32
WN=64
```

---

# 编译

```bash
rm -rf build

mkdir build
cd build

cmake ..
cmake --build . -j

cd ..
```

---

# 第一次运行

```bash
./build/shape_gemm \
  128 4096 4096 \
  --retune
```

V5 使用独立 cache：

```text
results/gemm_cache_v5.csv
```

所以不会错误复用 V4 的 tuning result。

---

# 最重要的 A/B Test

你的 V4 best：

```text
m128_n64_k16_t8x4_pipe
```

V5：

```text
m128_n64_k16_t8x4_warp
```

直接测：

```bash
./build/shape_gemm \
  128 4096 4096 \
  --kernel m128_n64_k16_t8x4_pipe
```

然后：

```bash
./build/shape_gemm \
  128 4096 4096 \
  --kernel m128_n64_k16_t8x4_warp
```

V4 baseline：

```text
~1.886 ms
~2.278 TFLOPS
~27.9% cuBLAS
```

V5 的问题就是：

```text
explicit warp tiling
+
shared-memory padding

能不能继续压低 1.886 ms？
```

---

# NCU Comparison

```bash
NCU=/usr/local/cuda-13.2/bin/ncu \
./scripts/profile_v4_vs_v5.sh
```

或者手动：

```bash
sudo /usr/local/cuda-13.2/bin/ncu \
  --set full \
  ./build/shape_gemm \
  128 4096 4096 \
  --kernel m128_n64_k16_t8x4_pipe \
  --profile-once
```

```bash
sudo /usr/local/cuda-13.2/bin/ncu \
  --set full \
  ./build/shape_gemm \
  128 4096 4096 \
  --kernel m128_n64_k16_t8x4_warp \
  --profile-once
```

---

# V5 最值得看什么？

## Shared Memory

重点观察：

```text
Shared Load Throughput

Shared Store Throughput

Bank Conflicts

Short Scoreboard
```

如果 padding 有效，希望看到：

```text
bank conflicts ↓
```

或者：

```text
shared-memory dependency stall ↓
```

---

## Warp Scheduling

重点观察：

```text
Eligible Warps / Scheduler

Active Warps

Issued Warps

Not Selected

Wait
```

warp-level mapping 可能改变：

```text
shared access pattern
instruction locality
warp dependency behavior
```

---

## Resource Cost

V5 padding 会稍微增加 shared memory：

例如 V4 winner：

```text
BM=128
BN=64
BK=16
```

V4 double buffer：

```text
2 * (128*16 + 16*64) * 4
=
24 KB
```

V5 padded：

```text
2 *
(
128*(16+1)
+
16*(64+1)
)
*
4
≈
25.1 KB
```

增加不大，但仍然需要看：

```text
Active blocks / SM
Occupancy
```

---

# Hypothesis

V5 要验证：

```text
V4:
flat thread mapping
+
non-padded shared layout
```

↓

```text
V5:
CTA → Warp → Thread hierarchy
+
padded shared memory
```

如果 V5 更快：

```text
shared-memory access efficiency ↑
and/or
warp scheduling efficiency ↑
```

如果 V5 反而慢：

```text
padding / mapping overhead
>
bank-conflict improvement
```

同样是有价值的结论。

---

# 当前版本路线

```text
V0
Basic tiled GEMM

↓

V1
Kernel Registry
+
Autotuner

↓

V2
Performance DB
+
Runtime Cache

↓

V3
float4 Vectorized Load

↓

V4
Software Pipeline
+
Double Buffer

↓

V5
Warp-Level Tiling
+
Shared Memory Padding
```

---

# 下一步

如果 V5 对 FP32 SIMT 路径还有收益，可以继续细化：

```text
warp tile variants
shared layout
register pressure
```

但做到 V5 后，项目已经足够深入。

之后更有价值的主线应该进入：

```text
FP16
+
WMMA / Tensor Core
+
Tensor-Core Kernel Registry
+
Autotuning between SIMT and Tensor Core families
```

也就是从：

```text
mini FP32 SGEMM library
```

继续变成：

```text
multi-family GEMM kernel library
```
