# Shape-Adaptive GEMM V3

V3 开始进入 **kernel 性能优化主线**。

在 V2 的：

- Kernel Registry
- Autotuner
- Performance Cache
- Runtime Dispatch
- NCU profile mode

基础上，V3 增加了：

- Scalar memory path
- `float4` vectorized global-load path
- 16-byte alignment / stride compatibility check
- Scalar 与 Vec4 kernel 成对注册
- Autotuner 自动比较两种 memory path
- 独立 V3 cache，避免复用 V2 的旧 benchmark
- Scalar vs Vec4 NCU 对照脚本

---

## V3 核心实验

同一个 compute tile：

```text
BM=128
BN=128
BK=8
TM=8
TN=8
```

现在有两个版本：

```text
m128_n128_k8_t8x8_scalar
m128_n128_k8_t8x8_vec4
```

它们的：

```text
Block Tile
Thread Tile
Shared Memory Size
Register Accumulator
Compute Loop
```

都保持一致。

**唯一主要变量是 Global -> Shared 的加载路径。**

这样才能做相对干净的 A/B 实验：

```text
Scalar Load
    VS
float4 Vectorized Load
```

---

## Vectorized Load

V2 类似：

```cpp
As[r][c] = A[...];
Bs[r][c] = B[...];
```

V3 vec4 fast path：

```cpp
float4 v =
    *reinterpret_cast<const float4*>(ptr);
```

一次 global load 获取：

```text
4 x FP32
=
16 bytes
```

然后写入 shared memory。

---

## Fast Path 条件

Vec4 kernel 只有满足以下条件才参与 autotune：

```text
A base address 16-byte aligned
B base address 16-byte aligned

K % 4 == 0
N % 4 == 0
```

`cudaMalloc` 通常天然满足 base alignment。

`K/N` 的约束保证每一行起始地址仍然维持 16-byte alignment。

如果不满足：

```text
Vec4 candidate
   ↓
SKIP(alignment/stride)
```

Scalar kernel 仍然可以处理。

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

# 第一次跑

建议首先：

```bash
./build/shape_gemm \
  128 4096 4096 \
  --retune
```

V3 默认 cache：

```text
results/gemm_cache_v3.csv
```

所以不会错误使用 V2 的旧结果。

Autotuner 会同时 benchmark：

```text
*_scalar
*_vec4
```

共 18 个候选 kernel。

---

# 重点比较

你的 V2 best kernel 是：

```text
m128_n128_k8_t8x8
```

所以 V3 最重要的是比较：

```bash
./build/shape_gemm \
  128 4096 4096 \
  --kernel m128_n128_k8_t8x8_scalar
```

和：

```bash
./build/shape_gemm \
  128 4096 4096 \
  --kernel m128_n128_k8_t8x8_vec4
```

记录：

```text
latency
TFLOPS
relative to cuBLAS
```

---

# NCU 对照

只 launch 一次 Scalar：

```bash
sudo ncu --set full \
  ./build/shape_gemm \
  128 4096 4096 \
  --kernel m128_n128_k8_t8x8_scalar \
  --profile-once
```

Vec4：

```bash
sudo ncu --set full \
  ./build/shape_gemm \
  128 4096 4096 \
  --kernel m128_n128_k8_t8x8_vec4 \
  --profile-once
```

或者：

```bash
./scripts/profile_scalar_vs_vec4.sh
```

你的机器如果需要完整 NCU 路径：

```bash
NCU=/usr/local/cuda-13.2/bin/ncu \
./scripts/profile_scalar_vs_vec4.sh
```

---

# V3 最值得看的 NCU 指标

先集中在：

```text
SM Throughput
Memory Throughput

Registers / Thread
Achieved Occupancy

Global Load Instructions
Global Memory Transactions

L1/L2 Throughput

Eligible Warps
Active Warps

Warp Stall Reasons
```

我们要验证的 hypothesis 是：

```text
float4
  ↓
global-load instruction count ↓
  ↓
load path overhead ↓
  ↓
GEMM latency ?
```

注意：

**Vec4 不保证一定明显更快。**

因为 GEMM 当前可能主要受：

```text
shared-memory access
instruction dependency
register pressure
synchronization
compute scheduling
```

限制。

如果 `float4` 几乎没提升，这反而是很有价值的结果：

```text
Memory-path optimization did not move the bottleneck.
```

然后 V4 才应该进入：

```text
Double Buffer
+
Software Pipeline
```

---

# Shape Sweep

```bash
python3 scripts/sweep.py
```

它会生成：

```text
results/gemm_cache_v3.csv
```

现在 performance DB 里会自动体现：

```text
某些 shape -> scalar
某些 shape -> vec4
```

---

# 项目版本演进

```text
V0
3 hard-coded kernels

↓

V1
Kernel Registry
+
Autotuner

↓

V2
Performance Cache
+
Runtime Dispatch
+
Stable Benchmark
+
Profiler Mode

↓

V3
Scalar / Vec4 Memory Paths
+
Alignment-aware Candidate Filter
+
Memory-path A/B Benchmark
+
NCU Before / After
```

下一阶段：

```text
V4
Double Buffer
+
Software Pipeline
```

V3 的意义不是简单加一个 `float4`，而是第一次建立：

```text
Optimization Hypothesis
        ↓
A/B Kernel
        ↓
Autotune Benchmark
        ↓
NCU Validation
        ↓
Conclusion
```

这个性能工程闭环。
