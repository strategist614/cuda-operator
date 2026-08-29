# High-Performance GPU Top-K Operator

一个面向小 `K` 场景的高性能 CUDA Top-K 算子实验项目。项目从串行基线逐步演进到 warp 协作选择、编译期 `K` 特化，以及基于输入形状和 GPU 型号的自动调优与运行时分发。

当前推荐使用 `gpu_topk_v5`。它提供完整的 kernel registry、autotuner、性能缓存和 dispatcher；V0–V4 保留用于学习优化过程、性能对照和 NCU 分析。

## 问题定义

输入为 FP32 二维数组：

```text
input[B, N]
```

输出每一行最大的 `K` 个元素及其原始下标：

```text
top_values[B, K]
top_indices[B, K]
```

结果按 value 从大到小排列。当 value 相同时，原始下标较小的元素排在前面，因此输出是确定性的。V5 当前支持：

```text
dtype = FP32
1 <= K <= 16
B > 0
N > 0
K <= N
```

## 项目亮点

- Exact Top-K：同时校验输出值和原始下标，不是近似选择。
- Coalesced scan：warp 中的线程连续读取输入，提升全局内存访问效率。
- Warp-cooperative selection：一个 warp 协作维护候选集合，减少逐线程 Top-K 状态和串行插入链。
- Threshold reject：整批候选无法超过当前第 `K` 大值时直接跳过后续排序与合并。
- K-specialized kernels：为 `K=1/2/4/8/16` 提供编译期特化实现。
- Shape-aware dispatch：V5 根据 GPU、`B`、`N`、`K` 和 dtype 自动选择更快的 kernel。
- Persistent cache：调优结果写入 CSV，相同形状再次运行时直接命中缓存。

## 版本演进

| 版本 | 核心实现 | 主要目标 |
| --- | --- | --- |
| [V0](gpu_topk_v0/README.md) | 单个 block 处理一行，仅 thread 0 重复扫描 | 建立正确性与性能基线 |
| [V1](gpu_topk_v1/README.md) | 并行扫描、thread-local register Top-K、block merge | 消除串行扫描 |
| [V2](gpu_topk_v2/README.md) | warp shuffle merge、warp0 最终合并 | 降低 shared memory 和 block barrier 开销 |
| [V3](gpu_topk_v3/README.md) | warp batch selection、threshold reject、bitonic merge | 优化 selection 本体 |
| [V4](gpu_topk_v4/README.md) | `K=1/2/4/8/16` 模板特化 kernel family | 减少小 K 场景的无效计算 |
| [V5](gpu_topk_v5/README.md) | registry、autotuner、cache、runtime dispatcher | 根据 GPU 和输入形状自动选核 |

V5 的执行路径如下：

```text
Top-K request (GPU, dtype, B, N, K)
                    |
                    v
             performance cache
               /           \
          cache hit      cache miss / --retune
              |                 |
              |                 v
              |            autotuner
              |          /     |      \
              |      batch_v3  ...  K-specialized V4
              |                 |
              |                 v
              |           save best result
               \               /
                v             v
                 runtime dispatcher
                         |
                         v
                   exact Top-K
```

## 环境要求

- 支持 CUDA 的 NVIDIA GPU
- CUDA Toolkit 和 NVCC
- CMake `>= 3.20`
- 支持 C++17 的主机编译器
- Python 3（仅 sweep 脚本需要）
- NVIDIA Nsight Compute（仅 NCU profiling 需要）

各版本的 CMake 默认编译目标为 SM75：

```cmake
CMAKE_CUDA_ARCHITECTURES=75
```

如果使用其他架构，请在配置阶段显式指定，例如 Ampere SM80：

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80
```

## 快速开始

推荐直接构建 V5：

```bash
cd gpu_topk_v5

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

查看所有已注册 kernel：

```bash
./build/topk_bench --list-kernels
```

对 `B=128, N=65536, K=4` 执行自动调优：

```bash
./build/topk_bench 128 65536 4 --retune
```

调优完成后，结果会写入：

```text
results/topk_cache_v5.csv
```

再次运行同一形状时会直接使用缓存中的最优 kernel：

```bash
./build/topk_bench 128 65536 4
```

程序还会使用 CPU `std::partial_sort` 生成参考结果，并检查 GPU 输出的 value 和 index 是否完全匹配。

## V5 Kernel Registry

| Kernel | 支持范围 | 默认参与 autotune | 说明 |
| --- | ---: | :---: | --- |
| `register_v1` | `1 <= K <= 16` | 否 | V1 register + block merge，保留用于历史对照 |
| `warp_v2` | `1 <= K <= 16` | 否 | V2 register + warp merge，保留用于历史对照 |
| `batch_v3` | `1 <= K <= 16` | 是 | 通用 exact small-K WarpSelect |
| `warpselect_k1_v4` | `K = 1` | 是 | 编译期 K=1 特化 |
| `warpselect_k2_v4` | `K = 2` | 是 | 编译期 K=2 特化 |
| `warpselect_k4_v4` | `K = 4` | 是 | 编译期 K=4 特化 |
| `warpselect_k8_v4` | `K = 8` | 是 | 编译期 K=8 特化 |
| `warpselect_k16_v4` | `K = 16` | 是 | 编译期 K=16 特化 |

对于不属于 `1/2/4/8/16`、但仍满足 `K <= 16` 的请求，V5 使用 `batch_v3` 作为高性能通用候选。

## 常用命令

强制使用指定 kernel：

```bash
./build/topk_bench 128 65536 8 --kernel batch_v3
./build/topk_bench 128 65536 8 --kernel warpselect_k8_v4
```

控制 benchmark 参数：

```bash
./build/topk_bench 128 65536 4 \
  --retune \
  --warmup 5 \
  --repeat 50 \
  --groups 5 \
  --seed 123
```

指定独立缓存文件：

```bash
./build/topk_bench 128 65536 4 \
  --retune \
  --cache results/my_topk_cache.csv
```

批量测试预设形状并填充缓存：

```bash
python3 scripts/sweep.py
```

调优后立即验证 cache hit：

```bash
python3 scripts/sweep.py --cache-hit-pass
```

查看多个 K 的分发结果：

```bash
./scripts/show_dispatch.sh
```

## 命令行参数

| 参数 | 作用 | 默认值 |
| --- | --- | --- |
| `B N K` | 输入形状和 Top-K 大小 | 必填 |
| `--retune` | 忽略已有缓存并重新调优 | 关闭 |
| `--cache PATH` | 指定性能缓存路径 | `results/topk_cache_v5.csv` |
| `--kernel NAME` | 跳过自动分发，强制运行指定 kernel | 未指定 |
| `--list-kernels` | 输出注册表并退出 | 关闭 |
| `--profile-once` | 只 launch 一次已选 kernel，便于 profiler 捕获 | 关闭 |
| `--warmup N` | 调优前的预热次数 | `5` |
| `--repeat N` | 每组 kernel launch 次数 | `50` |
| `--groups N` | benchmark 组数 | `5` |
| `--seed N` | 随机输入种子 | `123` |

## Benchmark 方法

V5 的 kernel benchmark 使用 CUDA Event 计时，默认流程为：

```text
5 次 warmup
5 groups × 50 launches
每组计算单次 launch 的平均延迟
对 5 组结果排序并取中位数
```

报告的 latency 是 kernel-only latency，不包含输入生成、Host-to-Device、Device-to-Host 和 CPU reference 的时间。`GElem/s` 按以下方式计算：

```text
GElem/s = B * N / latency_seconds / 1e9
```

## 仓库记录的性能结果

以下是各版本文档中记录的 RTX 2080 / SM75、FP32、`B=128, N=65536, K=16` 历史结果：

| 版本 | 延迟 | 相对 V0 加速 |
| --- | ---: | ---: |
| V0 | `525.86 ms` | `1.0x` |
| V1 | `4.11993 ms` | `~127.6x` |
| V2 | `4.08528 ms` | `~128.7x` |
| V3 | `0.475832 ms` | `~1105x` |

V4 和 V5 的最优实现会随 `K` 与输入形状变化，因此不使用单个数字代表所有场景。当前 V5 缓存文件中记录的同一 GPU、`B=128, N=65536` 结果为：

| K | 选中 kernel | 延迟 | 吞吐 |
| ---: | --- | ---: | ---: |
| 1 | `warpselect_k1_v4` | `0.311782 ms` | `26.91 GElem/s` |
| 4 | `warpselect_k4_v4` | `0.418328 ms` | `20.05 GElem/s` |
| 8 | `batch_v3` | `0.475974 ms` | `17.62 GElem/s` |

这些数据来自仓库已有文档和 `gpu_topk_v5/results/topk_cache_v5.csv`，仅用于展示优化趋势。实际性能会受到 GPU 架构、频率、驱动、CUDA 版本和输入形状影响，请使用 `--retune` 在目标机器上重新测量。

## Nsight Compute Profiling

使用 V5 脚本捕获指定 kernel：

```bash
K=4 \
KERNEL=warpselect_k4_v4 \
NCU=ncu \
./scripts/profile_selected.sh
```

脚本默认测试 `B=128, N=65536`，可以通过环境变量覆盖：

```bash
B=64 N=16384 K=8 KERNEL=batch_v3 NCU=ncu \
./scripts/profile_selected.sh
```

脚本内部使用 `sudo`。如果当前环境已经允许普通用户采集 GPU performance counters，可以直接修改脚本去掉 `sudo`。

## 目录结构

```text
High-Performance-GPU-Top-K-Operator/
├── README.md
├── gpu_topk_v0/              # 串行 CUDA baseline
├── gpu_topk_v1/              # 并行扫描 + register Top-K
├── gpu_topk_v2/              # warp shuffle merge
├── gpu_topk_v3/              # warp batch selection
├── gpu_topk_v4/              # K-specialized kernels
└── gpu_topk_v5/
    ├── CMakeLists.txt
    ├── include/              # launcher、registry、autotuner、cache 接口
    ├── src/                  # CUDA kernels 与 V5 分发基础设施
    ├── scripts/              # sweep、dispatch 展示与 NCU 脚本
    ├── results/              # 自动调优缓存
    └── README.md             # V5 详细设计说明
```

## 设计结论

这个项目的核心结论不是“存在一个适合所有输入的 Top-K kernel”，而是：

```text
Top-K performance = f(GPU architecture, B, N, K, dtype)
```

V0–V4 展示了如何通过 profiler 和 A/B benchmark 逐步定位瓶颈；V5 则将多个 kernel family 组织成一个能够自动测量、缓存并按形状分发的完整执行路径。
