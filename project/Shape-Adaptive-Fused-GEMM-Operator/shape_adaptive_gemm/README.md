# Shape-Adaptive Fused GEMM Operator

一个用于 GPU 算子优化实习项目的最小可运行工程。

核心目标不是“实现一个 GEMM”，而是研究：

> 不同矩阵 shape 下，最优 CUDA kernel 配置为什么不同，以及如何通过 runtime dispatch / benchmark 驱动选择合适的 kernel family。

## 当前版本

实现 3 个 FP32 CUDA GEMM kernel family：

- `small_m`
  - `BM=16, BN=64, BK=16`
  - `TM=1, TN=4`
  - 面向小 M

- `regular`
  - `BM=64, BN=64, BK=16`
  - `TM=4, TN=4`
  - 面向普通 GEMM

- `skinny_n`
  - `BM=64, BN=16, BK=16`
  - `TM=4, TN=1`
  - 面向小 N

并包含：

- shared-memory tiling
- per-thread register tile
- shape-based dispatcher
- cuBLAS correctness baseline
- cuBLAS performance baseline
- all-kernel benchmark mode
- Bias fusion
- Bias + SiLU fusion
- shape sweep 脚本

## 目录

```text
shape_adaptive_gemm/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── benchmark.h
│   ├── common.h
│   └── gemm.h
├── src/
│   ├── benchmark.cu
│   ├── gemm_kernels.cu
│   └── main.cu
└── scripts/
    └── run_shapes.sh
```

## 编译

要求：

- CUDA Toolkit
- cuBLAS
- CMake >= 3.20
- 支持 CUDA 的 NVIDIA GPU

```bash
mkdir -p build
cd build

cmake ..
cmake --build . -j
```

生成：

```bash
./shape_gemm
```

## 运行

### 普通 dispatcher

```bash
./build/shape_gemm 128 4096 4096
```

### 同一 shape 跑全部 kernel

这是项目里非常重要的模式。

```bash
./build/shape_gemm 16 4096 4096 --all-kernels
```

你应该观察：

```text
small_m
regular
skinny_n
cuBLAS
```

在同一个 `(M,N,K)` 下的 latency 差异。

### Bias fusion

```bash
./build/shape_gemm 512 512 512 --bias
```

### Bias + SiLU fusion

```bash
./build/shape_gemm 512 512 512 --silu
```

融合模式暂时不与 cuBLAS 做直接 correctness/performance 对比，
因为当前 baseline 只实现纯 GEMM。

## Shape sweep

```bash
chmod +x scripts/run_shapes.sh
./scripts/run_shapes.sh
```

或者：

```bash
./scripts/run_shapes.sh /path/to/shape_gemm
```

## 当前 dispatcher

当前规则只是一个故意保留的简单 baseline：

```cpp
if (M <= 32)
    small_m;
else if (N <= 32)
    skinny_n;
else
    regular;
```

它不是最终答案。

项目真正值得研究的问题是：

> `M <= 32` 这个阈值应该怎么得到？

下一阶段应对多个 shape 跑全部 kernel：

```text
M = 8,16,32,64,128,256,512,...
N = ...
K = ...
```

记录最优 kernel，然后建立：

```text
(M,N,K)
   ↓
performance database
   ↓
dispatcher
```

## 推荐后续版本

### V1 — Benchmark-driven dispatcher

保存：

```text
M,N,K,kernel,latency
```

根据实测结果而不是手写阈值选 kernel。

### V2 — Memory path

加入：

- `float4` vectorized load
- shared-memory padding
- bank conflict 分析
- global load efficiency 分析

### V3 — Pipeline

加入：

- double buffering
- software pipelining
- async copy（根据 GPU 架构决定是否使用）

### V4 — FP16 Tensor Core

加入：

- FP16 input
- FP32 accumulate
- WMMA / mma
- CUTLASS baseline

### V5 — Epilogue

继续扩展：

- Bias
- ReLU
- SiLU
- Residual
- reduction / auxiliary output

### V6 — Offline autotuner

搜索：

```text
BM
BN
BK
TM
TN
num_warps
pipeline stages
```

输出最佳配置缓存。

## NCU 建议观察指标

至少观察：

- SM Throughput
- Memory Throughput
- DRAM Throughput
- L2 Hit Rate
- Occupancy
- Registers / Thread
- Shared Memory / Block
- Eligible Warps
- Active Warps
- Warp Stall Reasons

你最终应该形成：

```text
Profiler
   ↓
Bottleneck
   ↓
Hypothesis
   ↓
Kernel change
   ↓
Benchmark
   ↓
Profiler
```

而不是只比较 latency。

## 项目定位

最终这个项目应该证明：

1. 能写 CUDA GEMM kernel。
2. 理解 shared memory / register tiling。
3. 理解 shape 对 kernel 配置的影响。
4. 能建立多 kernel family。
5. 能做 runtime dispatch。
6. 能通过 benchmark / NCU 做性能工程闭环。
7. 后续能进一步升级到 Tensor Core / CUTLASS / autotuning。

不要把目标设成“第一版超过 cuBLAS”。

第一版最重要的是先建立正确的实验框架，然后再逐步分析：
为什么某个 shape 下某种 tile 更好。
