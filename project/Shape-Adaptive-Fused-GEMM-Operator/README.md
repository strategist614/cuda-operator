# Shape-Adaptive Fused GEMM Operator

这是一个面向 CUDA 算子优化学习与实验的 Shape-Adaptive GEMM 项目。项目从多个固定 shape kernel 和简单规则分发开始，逐步加入 Kernel Registry、Autotuner、性能缓存、向量化访存、软件流水、warp-level tiling、FP16 Tensor Core kernel family，以及 CTA 与 warp fragment 两级流水。

项目关注的不只是实现矩阵乘法，而是研究：

> 面对不同的 `(M, N, K)`、数据类型和 GPU 资源约束，如何选择合适的 kernel 配置，并用正确性验证、benchmark 和 profiler 数据驱动优化。

## 核心能力

- FP32 CUDA Core GEMM。
- FP16 输入、FP32 累加和输出的 Tensor Core WMMA GEMM。
- Shared-memory tiling 与 per-thread register tiling。
- Shape-aware kernel dispatch。
- Kernel Registry 与硬件约束过滤。
- Offline benchmark autotuner。
- GPU、Compute Capability 和 shape 感知的性能缓存。
- Scalar、`float4` 和 `half8` 向量化访存路径。
- Register prefetch、双缓冲 shared memory 和 SM75 软件流水。
- Warp-level tiling 与 shared-memory padding。
- Shared→WMMA fragment double buffering。
- WMMA-compatible padded shared-memory layout。
- SM75 native `mma.sync` 实验封装。
- cuBLAS 正确性和性能基线。
- Bias、Bias + SiLU 等融合 Epilogue 接口。
- Shape sweep、单 kernel benchmark 和 NCU profile 脚本。

## 版本演进

| 版本 | 主要内容 | 详细文档 |
| --- | --- | --- |
| 基础版 | `small_m`、`regular`、`skinny_n` 三类 FP32 kernel，规则式 shape dispatcher、cuBLAS baseline 和融合 Epilogue | [`shape_adaptive_gemm/`](shape_adaptive_gemm/README.md) |
| V1 | 通用 kernel 模板、9 个 tile 配置、Kernel Registry、硬件过滤和离线 Autotuner | [`shape_adaptive_gemm_v1/`](shape_adaptive_gemm_v1/README.md) |
| V2 | Tune Cache、cache miss 自动调优、cache hit runtime dispatch、CSV 性能数据库和 NCU 单次采样模式 | [`shape_adaptive_gemm_v2/`](shape_adaptive_gemm_v2/README.md) |
| V3 | Scalar/`float4` 访存路径、16-byte 对齐检查和两种 memory path 的自动 A/B benchmark | [`shape_adaptive_gemm_v3/`](shape_adaptive_gemm_v3/README.md) |
| V4 | Global→Register 预取、双缓冲 shared memory 和面向 Turing/SM75 的软件流水 | [`shape_adaptive_gemm_v4/`](shape_adaptive_gemm_v4/README.md) |
| V5 | 显式 warp-level tiling、shared-memory padding，以及 scalar/vec4/pipe/warp 路径对比 | [`shape_adaptive_gemm_v5/`](shape_adaptive_gemm_v5/README.md) |
| V6 | FP32 SIMT 与 FP16 Tensor Core 双 kernel family、WMMA、`half8` 加载和 Tensor Core Autotuner | [`shape_adaptive_gemm_v6/`](shape_adaptive_gemm_v6/README.md) |
| V7 | CTA pipeline 与 warp fragment pipeline 两级流水、fragment double buffering、WMMA-compatible padding、扩大后的 Tensor Core 配置空间和 SM75 native MMA 实验入口 | [`shape_adaptive_gemm_v7/`](shape_adaptive_gemm_v7/README.md) |

整体演进路线：

```text
基础版
固定 kernel family + 手写 shape 规则
        ↓
V1
Kernel Registry + Autotuner
        ↓
V2
Performance Cache + Runtime Dispatch
        ↓
V3
Scalar / float4 Memory Paths
        ↓
V4
Register Prefetch + Double Buffer + Software Pipeline
        ↓
V5
Warp-Level Tiling + Shared-Memory Layout
        ↓
V6
FP32 SIMT + FP16 Tensor Core WMMA
        ↓
V7
CTA Pipeline + Warp Fragment Pipeline
```

## 总体架构

```text
GEMM Request
(GPU, dtype, M, N, K, epilogue)
                 ↓
          Performance Cache
           ↙             ↘
       cache hit       cache miss
           ↓               ↓
     Best Kernel      Kernel Registry
                           ↓
                    Hardware Filter
                           ↓
                       Autotuner
                           ↓
                  Save Benchmark Result
           ↘               ↙
              Runtime Launch
                    ↓
       Correctness + cuBLAS Comparison
```

## 目录结构

```text
Shape-Adaptive-Fused-GEMM-Operator/
├── README.md
├── shape_adaptive_gemm/
├── shape_adaptive_gemm_v1/
├── shape_adaptive_gemm_v2/
├── shape_adaptive_gemm_v3/
├── shape_adaptive_gemm_v4/
├── shape_adaptive_gemm_v5/
├── shape_adaptive_gemm_v6/
└── shape_adaptive_gemm_v7/
```

各版本通常包含：

```text
version/
├── CMakeLists.txt
├── README.md
├── include/
├── src/
├── scripts/      # 部分版本提供
└── results/      # V2 及以后提供性能缓存示例
```

## 环境要求

- 支持 CUDA 的 NVIDIA GPU。
- CUDA Toolkit 和 cuBLAS。
- CMake 3.20 或更高版本。
- 支持 C++17 的主机编译器。

项目主要面向 RTX 2080 / Turing / SM75。V4–V7 在 SM75 上使用软件预取和双缓冲，不依赖 `cp.async`。V6–V7 的 FP16 路径需要支持 Tensor Core 和 WMMA；V7 还提供实验性的 SM75 native `mma.sync` 封装，但尚未将其作为默认 GEMM mainloop。

检查环境：

```bash
nvidia-smi
nvcc --version
cmake --version
```

## 快速开始

### 构建基础版本

```bash
cmake -S shape_adaptive_gemm -B shape_adaptive_gemm/build
cmake --build shape_adaptive_gemm/build -j
```

运行默认 dispatcher：

```bash
./shape_adaptive_gemm/build/shape_gemm 128 4096 4096
```

比较同一 shape 下的全部基础 kernel：

```bash
./shape_adaptive_gemm/build/shape_gemm 16 4096 4096 --all-kernels
```

### 构建当前版本 V7

```bash
cmake -S shape_adaptive_gemm_v7 -B shape_adaptive_gemm_v7/build
cmake --build shape_adaptive_gemm_v7/build -j
```

运行默认 FP16 Tensor Core 路径并重新调优：

```bash
./shape_adaptive_gemm_v7/build/shape_gemm 128 4096 4096 --dtype fp16 --retune
```

运行 FP32 SIMT 路径：

```bash
./shape_adaptive_gemm_v7/build/shape_gemm 128 4096 4096 --dtype fp32 --retune
```

查看可用 kernel：

```bash
./shape_adaptive_gemm_v7/build/shape_gemm --dtype fp16 --list-kernels
./shape_adaptive_gemm_v7/build/shape_gemm --dtype fp32 --list-kernels
```

对比 V6 baseline、V7 fragment pipeline 和 padded fragment pipeline：

```bash
./shape_adaptive_gemm_v7/build/shape_gemm 128 4096 4096 \
  --dtype fp16 --kernel tcv6_m128_n64_k16_w64x32

./shape_adaptive_gemm_v7/build/shape_gemm 128 4096 4096 \
  --dtype fp16 --kernel tcv7_m128_n64_k16_w64x32_frag

./shape_adaptive_gemm_v7/build/shape_gemm 128 4096 4096 \
  --dtype fp16 --kernel tcv7_m128_n64_k16_w64x32_fp
```

使用 V7 提供的 NCU A/B 脚本：

```bash
(cd shape_adaptive_gemm_v7 && \
  NCU=/usr/local/cuda/bin/ncu ./scripts/profile_v6_vs_v7.sh)
```

每个版本的参数和支持能力不同，运行前请阅读对应目录的 README。

## 实测性能对比

以下数据于 2026-08-29 在同一台机器、同一轮连续测试中获得。

### 测试环境

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 2080 |
| Compute Capability | 7.5 / SM75 |
| NVIDIA Driver | 595.84 |
| CUDA Toolkit | 13.2（`nvcc 13.2.86`） |
| Shape | `M=128, N=4096, K=4096` |
| FP32 计算量 | `2 × M × N × K` |
| FP16 模式 | FP16 输入、FP32 累加与输出 |

测试约定：

- 基础版到 V7 均测试 FP32 路径。
- V6、V7 额外测试 FP16 Tensor Core 路径；FP16 与 FP32 的结果分表展示，不直接混合排名。
- V2 及以后版本均使用 `--retune` 强制搜索，缓存写入 `/tmp`，未读取或改写仓库中的历史 CSV。
- 各程序使用 CUDA Event 计时。V2 及以后版本对每个候选执行 `5 groups × 50 launches`，选择各组平均耗时的中位数。
- 基础版和 V1 使用各自早期版本内置的计时流程，因此与 V2–V7 的数值适合观察总体趋势，不应视为严格的微秒级横向结论。
- cuBLAS 在每个版本的同次运行中重新测量，以降低不同测试时段带来的误差。

### FP32：基础版到 V7

| 版本 | 最佳 kernel / 路径 | 延迟（ms） | TFLOPS | 同次 cuBLAS TFLOPS | cuBLAS 相对性能 | 相对基础版 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 基础版 | `regular` | 2.41053 | 1.78176 | 8.19444 | 21.74% | 基准 |
| V1 | `m128_n128_k8_t8x8` | 2.54753 | 1.68593 | 8.19403 | 20.58% | -5.38% |
| V2 | `m128_n128_k8_t8x8` | 2.54493 | 1.68766 | 8.11654 | 20.79% | -5.28% |
| V3 | `m128_n64_k16_t8x4_vec4` | 2.04370 | 2.10156 | 8.15823 | 25.76% | +17.95% |
| V4 | `m128_n64_k16_t8x4_pipe` | 1.88752 | 2.27546 | 8.18643 | 27.80% | +27.71% |
| V5 | `m128_n64_k16_t8x4_warp` | 1.25936 | 3.41045 | 8.18352 | 41.67% | +91.41% |
| V6 | `m128_n64_k16_t8x4_warp` | 1.26354 | 3.39917 | 8.21743 | 41.37% | +90.78% |
| V7 | `m128_n64_k16_t8x4_warp` | 1.25882 | 3.41189 | 8.19171 | 41.65% | +91.49% |

所有 FP32 版本在该 shape 上的最大绝对误差均为 `2.6226e-06`。

FP32 结果说明：

1. V1、V2 的主要价值是 Kernel Registry、Autotuner、缓存和运行时分发架构；在这个固定 shape 上，它们没有直接带来 kernel 性能提升。
2. V3 的 `float4` 路径将性能从 V2 的 `1.68766` 提升到 `2.10156 TFLOPS`。
3. V4 软件流水进一步达到 `2.27546 TFLOPS`。
4. V5 的 warp-level tiling 是 FP32 路线中最大的单版本提升，达到 `3.41045 TFLOPS`，相比 V4 提升约 `49.9%`。
5. V6、V7 保留 V5-compatible FP32 family，三者约为 `3.40 TFLOPS`，差异处于本轮测试波动范围内。

### FP16 Tensor Core：V6 与 V7

| 版本 | 最佳 kernel | 延迟（ms） | TFLOPS | 同次 cuBLAS TFLOPS | cuBLAS 相对性能 |
| --- | --- | ---: | ---: | ---: | ---: |
| V6 | `tc_m128_n64_k16_w64x32` | 0.328585 | 13.0711 | 23.5985 | 55.39% |
| V7 | `tcv7_m64_n64_k32_w32x32_fp` | 0.189980 | 22.6075 | 23.7627 | 95.14% |

两个版本的 FP16→FP32 结果均通过正确性检查：最大绝对误差和最大相对误差均为 `0`，`allclose(atol=1e-2, rtol=1e-2)` 为 `PASS`。

V7 相比 V6：

- TFLOPS 提升约 `72.96%`。
- 延迟降低约 `42.18%`。
- 与同次 cuBLAS Tensor Core 的性能差距缩小到约 `4.86%`；延迟仅高约 `5.11%`。

V7 的 ablation 还显示：无 padding 的 `tcv7_m128_n64_k16_w64x32_frag` 为 `13.0486 TFLOPS`，与 V6 同结构 baseline 的 `13.0727 TFLOPS` 基本持平；加入 padded layout 后，同类 V7 kernel 达到 `19.8154 TFLOPS`，而扩大配置空间后选出的 `64×64×32` 最佳 kernel 达到 `22.6075 TFLOPS`。因此本次提升来自 fragment pipeline、shared-memory layout 与配置搜索的组合，不能只归因于 fragment double buffering。

### 复现命令

基础版与 V1：

```bash
./shape_adaptive_gemm/build/shape_gemm 128 4096 4096
./shape_adaptive_gemm_v1/build/shape_gemm 128 4096 4096
```

V2–V5 使用对应版本目录，并将缓存放到 `/tmp`：

```bash
./shape_adaptive_gemm_v5/build/shape_gemm \
  128 4096 4096 \
  --cache /tmp/shape_gemm_v5_bench.csv \
  --retune
```

V6、V7 分别测试两种数据类型：

```bash
./shape_adaptive_gemm_v7/build/shape_gemm \
  128 4096 4096 \
  --dtype fp32 \
  --cache /tmp/shape_gemm_v7_fp32_bench.csv \
  --retune

./shape_adaptive_gemm_v7/build/shape_gemm \
  128 4096 4096 \
  --dtype fp16 \
  --cache /tmp/shape_gemm_v7_fp16_bench.csv \
  --retune
```

性能结果会受到 GPU Boost、温度、功耗上限、驱动和 CUDA 版本影响。在其他机器上应重新调优，并使用同一环境中的 cuBLAS 结果作为基线。

## Autotune 与性能缓存

V2 及以后版本会根据 GPU、Compute Capability、shape、数据类型和 Epilogue 等信息选择或缓存最佳 kernel。典型流程如下：

```text
第一次运行
cache miss → benchmark candidates → 选择最快 kernel → 写入 CSV

再次运行
cache hit → 直接 dispatch 已缓存 kernel
```

使用 `--retune` 可以忽略已有结果并重新搜索。性能缓存位于各版本的 `results/` 目录，可作为实验记录和 shape-performance 数据库。

## 正确性与性能验证

推荐按照以下顺序开展实验：

1. 使用 cuBLAS 或 CPU reference 验证正确性。
2. 检查最大绝对误差和相对误差。
3. 完成 warm-up 后再进行多轮计时。
4. 使用 median 等稳健统计量降低频率、温度和系统抖动影响。
5. 固定 shape 与 tile，仅改变一个优化因素进行 A/B 对比。
6. 使用 Nsight Compute 检查吞吐、occupancy、寄存器、shared memory 和 warp stall。
7. 修改 kernel 后重新执行正确性和性能测试。

重点关注：

- 非 tile 整数倍的边界 shape。
- 很小的 `M` 或 `N`。
- 指针对齐与行 stride 是否满足向量化要求。
- Shared-memory 容量、寄存器压力和 occupancy。
- FP16/FP32 对应的合理数值误差。
- Autotuner 是否公平比较相同 workload。

## 推荐阅读顺序

如果目标是理解优化过程，建议按基础版、V1、V2……V7 的顺序阅读。每个版本只引入一组主要变化，便于对比：

```text
shape 规则
→ 配置注册
→ 数据驱动选择
→ 访存优化
→ 流水化
→ warp 映射
→ Tensor Core
→ warp fragment 流水与 operand feeding
```

如果只想查看当前能力，可以直接从 [`shape_adaptive_gemm_v7/README.md`](shape_adaptive_gemm_v7/README.md) 开始。

## 构建产物

各版本的 `build/` 目录由项目级 `.gitignore` 排除。请在本地重新配置和构建，不要依赖其他机器或旧环境生成的 CMake 缓存、目标文件和可执行文件。

`results/*.csv` 是性能缓存或实验数据示例。不同 GPU、驱动、CUDA 版本和时钟状态下的最优配置可能不同，应在目标环境中重新调优。

## 项目定位

这是一个学习和实验性质的 GEMM Library 原型，不是生产级通用算子库。生产使用前仍需补充更完整的边界测试、错误处理、多数据类型覆盖、跨架构验证、稳定 ABI 和系统化性能回归测试。
