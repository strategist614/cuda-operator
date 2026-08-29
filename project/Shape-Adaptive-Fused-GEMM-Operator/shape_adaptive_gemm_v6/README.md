# Shape-Adaptive GEMM V6

V6 是这个项目第一次真正进入 **Tensor Core GEMM Library** 阶段。

它不是一个独立的 WMMA demo，而是在现有 V5 library architecture 中加入第二个完整 kernel family：

```text
                    Shape-Adaptive GEMM V6
                              |
               +--------------+--------------+
               |                             |
               v                             v
         FP32 SIMT Family             FP16 Tensor Core Family
               |                             |
        register tiling                  WMMA 16x16x16
        float4 load                      warp MMA tiles
        software pipeline               half8 vector load
        warp tiling                      shared double buffer
        smem layout                      software pipeline
               |                             |
               +--------------+--------------+
                              |
                           Autotuner
                              |
                        Performance DB
                              |
                       Runtime Dispatch
```

---

# 1. V6 默认模式

V6 默认运行：

```text
FP16 input
FP32 accumulation
FP32 output
Tensor Core WMMA
```

所以：

```bash
./build/shape_gemm 128 4096 4096 --retune
```

等价于：

```bash
./build/shape_gemm \
  128 4096 4096 \
  --dtype fp16 \
  --retune
```

如果想回到 V5 FP32 SIMT family：

```bash
./build/shape_gemm \
  128 4096 4096 \
  --dtype fp32 \
  --retune
```

---

# 2. Tensor Core Kernel Hierarchy

V6 Tensor Core kernel 使用三级 tile：

```text
CTA Tile
   |
   v
Warp Tile
   |
   v
WMMA Instruction Tile
```

Instruction tile 固定：

```text
16 x 16 x 16
```

例如：

```text
tc_m128_n128_k32_w64x64
```

表示：

```text
CTA:
M = 128
N = 128
K = 32

Warp:
M = 64
N = 64

MMA:
16 x 16 x 16
```

一个 `64 x 64` warp tile 内部：

```text
4 x 4
=
16 个 16x16 accumulator fragments
```

每次 K 前进：

```text
16
```

执行：

```cpp
wmma::load_matrix_sync(...)
wmma::mma_sync(...)
```

---

# 3. Tensor Core Registry

V6 当前注册：

```text
tc_m64_n64_k16_w32x32
tc_m64_n64_k32_w32x32

tc_m128_n64_k16_w64x32
tc_m128_n64_k32_w64x32
tc_m128_n64_k32_w32x32

tc_m64_n128_k16_w32x64
tc_m64_n128_k32_w32x64

tc_m128_n128_k16_w64x64
tc_m128_n128_k32_w64x64
tc_m128_n128_k32_w32x64
```

查看：

```bash
./build/shape_gemm \
  --dtype fp16 \
  --list-kernels
```

FP32 registry：

```bash
./build/shape_gemm \
  --dtype fp32 \
  --list-kernels
```

---

# 4. Tensor Core Software Pipeline

RTX 2080 是 Turing / SM75。

因此 V6 **不使用 `cp.async`**。

V6 使用：

```text
Global FP16
   |
   v
half8 / 16-byte vector load
   |
   v
register prefetch
   |
   v
double-buffer shared memory
   |
   v
WMMA fragments
   |
   v
Tensor Core
```

shared memory：

```cpp
half As[2][BM][BK];
half Bs[2][BK][BN];
```

mainloop：

```text
Stage 0:
current WMMA compute

at the same time:
next Global -> Register prefetch

then:
Register -> Stage 1
barrier
swap

Stage 1:
current WMMA compute

at the same time:
next Global -> Register prefetch
```

这是 SM75 上的软件流水版本。

---

# 5. Vectorized FP16 Load

FP16 每个元素：

```text
2 bytes
```

V6 global load 使用：

```text
8 x half
=
16 bytes
```

通过一个 `int4` 完成一次 16-byte vector transfer。

因此 Tensor Core fast path 要求：

```text
M % CTA_M == 0
N % CTA_N == 0
K % CTA_K == 0

M/N/K also compatible with WMMA 16-element granularity
```

V6.0 的目标是先建立一个 **高性能 exact-tile fast path**。

一般 shape 的尾块处理后续可以作为 fallback extension；
V5 FP32 SIMT family 仍然保留在同一项目中。

---

# 6. WMMA Shared-Memory Requirements

V6 shared tile 显式做：

```cpp
__align__(32)
```

并保证：

```text
BK
BN
```

是适合 WMMA leading dimension 的 16-element multiples。

这让：

```cpp
wmma::load_matrix_sync(...)
```

从 shared memory 加载 16x16 fragment。

---

# 7. 编译

V6 默认：

```text
CMAKE_CUDA_ARCHITECTURES = 75
```

对应 RTX 2080。

编译：

```bash
rm -rf build

mkdir build
cd build

cmake ..
cmake --build . -j

cd ..
```

如需覆盖架构：

```bash
cmake \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  ..
```

---

# 8. 第一条必须跑的命令

```bash
./build/shape_gemm \
  128 4096 4096 \
  --retune
```

预期：

```text
GPU: NVIDIA GeForce RTX 2080
dtype: fp16
Family: FP16 Tensor Core / FP32 accumulate

Mode: cache miss -> Tensor Core autotune

tc_...
tc_...
tc_...

Best Tensor Core kernel:
...
```

然后：

```text
Correctness vs cuBLAS FP16->FP32

Max abs error
Max relative error
allclose
```

最后：

```text
cuBLAS Tensor Core baseline

latency
TFLOPS
Relative performance
```

---

# 9. cuBLAS Baseline

V6 FP16 baseline 不再使用：

```text
cublasSgemm
```

而是：

```cpp
cublasGemmEx(...)
```

数据：

```text
A = FP16
B = FP16
C = FP32
Compute = FP32
```

并在 Turing 上请求 Tensor-Op GEMM algorithm。

所以比较的是：

```text
Our FP16 Tensor Core GEMM
vs
cuBLAS FP16 Tensor Core GEMM
```

而不是不公平的：

```text
FP16 Tensor Core
vs
FP32 SGEMM
```

---

# 10. Correctness

Tensor Core 输入是 FP16，因此不要再要求：

```text
1e-6
```

级别 FP32 输入误差。

V6 输出：

```text
Max abs error
Max relative error
```

并提供：

```text
allclose(
    atol = 1e-2,
    rtol = 1e-2
)
```

cuBLAS 使用相同 FP16 A/B 和 FP32 accumulation/output。

---

# 11. Cache

V6 使用统一：

```text
results/gemm_cache_v6.csv
```

key 包含：

```text
GPU
Compute Capability
dtype family
M
N
K
epilogue
```

所以同一个数据库可以同时保存：

```text
fp32_simt
```

和：

```text
fp16_tc
```

例如：

```text
RTX2080,7,5,fp32_simt,...
RTX2080,7,5,fp16_tc,...
```

---

# 12. Cache Hit

第一次：

```bash
./build/shape_gemm \
  128 4096 4096 \
  --retune
```

第二次：

```bash
./build/shape_gemm \
  128 4096 4096
```

应该直接：

```text
Mode: cache hit
Cached Tensor Core kernel: ...
```

---

# 13. Force Kernel

例如：

```bash
./build/shape_gemm \
  128 4096 4096 \
  --dtype fp16 \
  --kernel tc_m128_n128_k32_w64x64
```

---

# 14. Nsight Compute

单 launch：

```bash
sudo /usr/local/cuda-13.2/bin/ncu \
  --set full \
  ./build/shape_gemm \
  128 4096 4096 \
  --dtype fp16 \
  --kernel tc_m128_n128_k32_w64x64 \
  --profile-once
```

或者：

```bash
NCU=/usr/local/cuda-13.2/bin/ncu \
KERNEL=tc_m128_n128_k32_w64x64 \
./scripts/profile_tensor_core.sh
```

---

# 15. NCU 重点

V6 开始重点看：

```text
Tensor Core utilization

SM Throughput

Registers / Thread

Shared Memory / Block

Achieved Occupancy

Eligible Warps
Active Warps

Warp Stall Reasons

Shared Memory Throughput
L2 Throughput
DRAM Throughput
```

你要回答：

```text
Tensor Core 有没有吃满？

如果没吃满：

为什么？
```

例如：

```text
shared-memory supply
register pressure
too few warps
pipeline bubble
load dependency
barrier
```

---

# 16. FP32 vs FP16

可以直接：

```bash
./scripts/compare_fp32_fp16.sh
```

它会分别 autotune：

```text
V5 FP32 SIMT
```

和：

```text
V6 FP16 Tensor Core
```

注意：

这两个使用不同输入 dtype，

所以这不是同精度算法的 apples-to-apples 性能竞争，

而是展示 library 同时管理：

```text
general FP32 SIMT family
+
high-throughput FP16 Tensor Core family
```

---

# 17. FP16 Shape Sweep

```bash
python3 scripts/sweep_fp16.py
```

将自动填充：

```text
results/gemm_cache_v6.csv
```

然后可以研究：

```text
Shape
   ↓
CTA Tile
   ↓
Warp Tile
   ↓
BK / pipeline
   ↓
Best Tensor Core Config
```

---

# 18. 项目最终结构

```text
Shape-Adaptive GEMM V6
|
+-- FP32 SIMT
|   |
|   +-- scalar
|   +-- float4
|   +-- software pipeline
|   +-- warp tiling
|   +-- shared layout
|
+-- FP16 Tensor Core
|   |
|   +-- half8 vector load
|   +-- double-buffer shared memory
|   +-- register prefetch
|   +-- WMMA 16x16x16
|   +-- CTA tile
|   +-- Warp tile
|   +-- software pipeline
|
+-- Kernel Registry
|
+-- Autotuner
|
+-- Performance Cache
|
+-- Runtime Dispatch
|
+-- Correctness Gate
|
+-- cuBLAS Baselines
|
+-- NCU Profile Mode
```

这已经可以作为项目的主要完成版。

下一步不应该急着继续编号 V7，

而应该先跑 V6：

```text
1. compile
2. correctness
3. autotune
4. cuBLAS Tensor Core comparison
5. NCU
```

然后根据真实 profiler 数据决定 Tensor Core 路线还缺什么。
