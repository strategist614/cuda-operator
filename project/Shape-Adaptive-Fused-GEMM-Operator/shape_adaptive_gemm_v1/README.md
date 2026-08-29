# Shape-Adaptive GEMM V1

这是一个面向 GPU 算子优化实习项目的小型 GEMM Library 原型。

V1 的核心不是继续手写一个 GEMM kernel，而是把项目重构为：

```text
Generic GEMM Kernel Template
        ↓
Kernel Instances
        ↓
Kernel Registry
        ↓
Hardware Filter
        ↓
Autotuner
        ↓
Best Kernel
```

## 当前能力

- FP32 CUDA GEMM
- Shared Memory Tiling
- Register Tiling
- 9 个 Kernel Config
- Kernel Registry
- Hardware Constraint Filter
- Offline Benchmark Autotuner
- cuBLAS Correctness Baseline
- cuBLAS Performance Baseline
- Fused Epilogue 接口保留：
  - NONE
  - BIAS
  - BIAS_SILU

## Kernel Registry

当前包含：

```text
m16_n64_k16_t1x4
m32_n64_k16_t2x4
m64_n64_k16_t4x4
m64_n128_k16_t4x8
m128_n64_k16_t8x4
m128_n128_k8_t8x8
m64_n16_k16_t4x1
m64_n32_k16_t4x2
m32_n128_k8_t2x8
```

例如：

```text
m64_n128_k16_t4x8
```

表示：

```text
BM = 64
BN = 128
BK = 16

TM = 4
TN = 8
```

## 编译

建议不要复用旧 build：

```bash
rm -rf build

mkdir build
cd build

cmake ..
cmake --build . -j
```

## 运行

例如：

```bash
./shape_gemm 128 4096 4096
```

程序会：

1. 打印 GPU 信息。
2. 打印 Kernel Registry。
3. 对 Registry 中所有硬件合法 kernel 做 benchmark。
4. 自动选择 latency 最低的 kernel。
5. 用最佳 kernel 与 cuBLAS 比较 correctness。
6. 与 cuBLAS 比较 TFLOPS。

## 你应该重点观察什么

不要只看最终 Best Kernel。

记录：

```text
shape
kernel config
latency
TFLOPS
best kernel
cuBLAS
relative performance
```

例如：

```text
M=128 N=4096 K=4096

m16_n64_k16_t1x4      ...
m32_n64_k16_t2x4      ...
m64_n64_k16_t4x4      ...
m64_n128_k16_t4x8     ...
...
```

这会成为后面分析：

```text
Shape
  ↓
Tile Configuration
  ↓
GPU Resource Usage
  ↓
Performance
```

的实验数据。

## V1 后面的路线

建议顺序：

```text
V1
Kernel Registry + Autotuner

↓

V2
Autotune Cache + Shape Sweep
Performance Database

↓

V3
Vectorized Global Load
Shared Memory Layout

↓

V4
Double Buffer
Software Pipeline

↓

V5
FP16 / Tensor Core

↓

V6
CUTLASS Baseline
Fused Epilogue

↓

V7
Profiler-aware Search / Pruning
```

## RTX 2080

当前版本使用 FP32 CUDA Core 路径，可以在 RTX 2080 / SM75 上运行。

后续 Tensor Core 版本再针对 Turing 的 WMMA / Tensor Core 单独增加 kernel family。
