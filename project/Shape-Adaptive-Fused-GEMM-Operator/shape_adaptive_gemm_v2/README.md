# Shape-Adaptive GEMM V2

这是一个面向 GPU 算子优化实习的小型高性能 GEMM Library 原型。

V2 在 V1 的 Kernel Registry + Autotuner 基础上加入了：

- GPU / Compute Capability 感知的 tune cache
- `M,N,K -> best kernel` runtime dispatch
- cache miss 自动 autotune
- cache hit 直接复用历史最优 kernel
- `--retune` 强制重新搜索
- `--kernel NAME` 单 kernel 模式
- `--profile-once` NCU 干净采样模式
- 多组 benchmark 取 median
- shape sweep 脚本
- CSV performance database

## 架构

```text
gemm(M,N,K)
     |
     v
Tune Cache
  /      \
hit      miss
 |         |
 |     Kernel Registry
 |         |
 |     Hardware Filter
 |         |
 |      Autotune
 |         |
 |     Save Result
 |         |
 +----+----+
      |
      v
 Best Kernel
      |
      v
   Launch
```

## 当前 kernel registry

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

当前仍然是 FP32 CUDA Core GEMM，后续再新增 Tensor Core family。

## 编译

```bash
rm -rf build
mkdir build
cd build

cmake ..
cmake --build . -j
```

回到项目根目录：

```bash
cd ..
```

## 第一次运行：cache miss

```bash
./build/shape_gemm 128 4096 4096
```

第一次没有缓存时会：

```text
cache miss
  ↓
9 个 kernel benchmark
  ↓
选择 fastest kernel
  ↓
写入 results/gemm_cache.csv
```

CSV 类似：

```text
gpu,cc_major,cc_minor,M,N,K,epilogue,kernel,latency_ms,tflops
NVIDIA GeForce RTX 2080,7,5,128,4096,4096,0,m128_n128_k8_t8x8,2.53,1.69
```

## 第二次运行：cache hit

再次：

```bash
./build/shape_gemm 128 4096 4096
```

此时不会重新遍历所有 kernel：

```text
cache hit
  ↓
m128_n128_k8_t8x8
  ↓
直接 runtime dispatch
```

## 强制重新 autotune

```bash
./build/shape_gemm 128 4096 4096 --retune
```

## 自定义 cache

```bash
./build/shape_gemm \
  128 4096 4096 \
  --cache results/rtx2080.csv
```

Cache key 包含：

```text
GPU name
Compute Capability
M
N
K
Epilogue
```

所以不同 GPU 不会错误复用同一个 tuning result。

## 单 kernel benchmark

```bash
./build/shape_gemm \
  128 4096 4096 \
  --kernel m128_n128_k8_t8x8
```

这对研究某一个 kernel 特别重要。

## NCU 模式

为了防止 NCU 把 autotuner 里的所有 kernel 都 profile 一遍：

```bash
./build/shape_gemm \
  128 4096 4096 \
  --kernel m128_n128_k8_t8x8 \
  --profile-once
```

然后：

```bash
sudo ncu --set full \
  ./build/shape_gemm \
  128 4096 4096 \
  --kernel m128_n128_k8_t8x8 \
  --profile-once
```

比较两个 kernel：

```bash
./scripts/profile_compare.sh
```

如果 `sudo` 下找不到 `ncu`：

```bash
NCU=/usr/local/cuda-13.2/bin/ncu \
./scripts/profile_compare.sh
```

## Shape Sweep

```bash
python3 scripts/sweep.py
```

它会对一批典型 shape 强制重新 autotune，并不断填充：

```text
results/gemm_cache.csv
```

可以调整：

```bash
python3 scripts/sweep.py \
  --bin ./build/shape_gemm \
  --cache results/rtx2080.csv \
  --repeat 50 \
  --groups 5
```

## 为什么 benchmark 改成 median

V1 是单次平均值。

V2：

```text
group 0 -> 50 launches -> avg
group 1 -> 50 launches -> avg
group 2 -> 50 launches -> avg
group 3 -> 50 launches -> avg
group 4 -> 50 launches -> avg

             ↓

           median
```

这样能减少：

- GPU Boost 波动
- 温度波动
- 偶发系统干扰

对 autotuner 更可靠。

## V2 的项目意义

V1：

```text
Kernel Registry
+
Autotuner
```

V2：

```text
Kernel Registry
+
Offline Autotuning
+
Performance Database
+
Runtime Dispatch
+
Profiler Mode
```

现在已经开始具备一个 mini GEMM library 的运行逻辑。

## 下一阶段 V3

V3 不再主要改 library architecture，而开始真正提高 kernel 上限：

```text
Vectorized Global Load
        ↓
Shared Memory Layout
        ↓
Bank Conflict Analysis
        ↓
NCU Before / After
```

然后：

```text
V4: Double Buffer / Software Pipeline
V5: Warp Tiling
V6: FP16 / Tensor Core
V7: CUTLASS baseline + richer epilogue
```
