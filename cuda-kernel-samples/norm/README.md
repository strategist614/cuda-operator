# CUDA LayerNorm 与 RMSNorm 优化示例

本目录包含 LayerNorm 和 RMSNorm 的 FP32 CUDA 实现。每个算子各有一个便于理解的 shared-memory 基础版，以及一个使用 `float4` 和 warp shuffle reduction 的优化版。四个 `.cu` 文件均带有 `main()`，可以独立编译、运行和验证。

## 文件说明

| 文件 | Kernel | 主要实现 |
| --- | --- | --- |
| `01_layernorm_naive.cu` | `layernorm_naive` | 每行一个 block，标量加载，shared-memory tree reduction |
| `02_layernorm_optimized.cu` | `layernorm_float4` | `float4` 向量化读写，两级 warp shuffle reduction |
| `03_rmsnorm_naive.cu` | `rmsnorm_naive` | 每行一个 block，标量加载，shared-memory tree reduction |
| `04_rmsnorm_optimized.cu` | `rmsnorm_float4` | `float4` 向量化读写，两级 warp shuffle reduction |
| `norm_test_utils.cuh` | 公共 host 测试代码 | 确定性输入、CPU FP64 reference、误差检查和 CUDA Event 计时 |

当前测试参数定义在 `norm_test_utils.cuh` 中：

```cpp
ROWS = 4096;
HIDDEN = 1024;
THREADS = 256;
WARMUP = 10;
REPEATS = 100;
EPSILON = 1.0e-5f;
```

输入 `X` 和输出 `Y` 的布局均为连续行主序 `[ROWS, HIDDEN]`。`gamma` 和 `beta` 的长度为 `HIDDEN`，由所有行共享。

## 数学定义

对每一行 `x`，LayerNorm 计算：

```text
mean     = sum(x[i]) / HIDDEN
variance = sum(x[i] * x[i]) / HIDDEN - mean * mean
y[i]     = (x[i] - mean) * rsqrt(variance + epsilon)
           * gamma[i] + beta[i]
```

RMSNorm 不减均值，也没有 beta：

```text
mean_square = sum(x[i] * x[i]) / HIDDEN
y[i]        = x[i] * rsqrt(mean_square + epsilon) * gamma[i]
```

两个 CUDA 实现都使用 FP32 归约和输出。测试程序的 CPU reference 使用 FP64 计算统计量，并以 `max_abs_error <= 1e-4` 作为通过标准。

## 基础版

基础版为每一行启动一个 256-thread block。每个线程以 `blockDim.x` 为步长遍历该行，先计算线程局部统计量，再写入 shared memory：

```text
线程局部 sum / square_sum
    -> shared memory
    -> 256, 128, ..., 1 的树形归约
    -> thread 0 计算 mean/inverse_std 或 inverse_rms
    -> 全 block 写回归一化结果
```

LayerNorm 使用两段 `THREADS` 大小的动态 shared memory，同时归约 `sum` 和 `square_sum`；RMSNorm 只需要归约 `square_sum`。这种写法结构直观，但每一级归约都需要一次 `__syncthreads()`。

## 优化版

优化版保持一个 block 处理一行，并做两项主要调整。

### `float4` 向量化

每个线程一次加载或写回 4 个连续 FP32 元素，减少 load/store 指令和循环控制开销：

```cpp
const float4 value = x4[row4_offset + col4];
```

当前 `HIDDEN=1024` 是 4 的整数倍，且 `cudaMalloc` 返回满足 `float4` 要求的对齐地址。如果修改为不能被 4 整除的 hidden size，需要增加标量 tail path；当前代码通过 `static_assert` 阻止不满足条件的配置。

### Warp shuffle 两级归约

每个 warp 先通过 `__shfl_down_sync` 在寄存器中归约；lane 0 将 warp 结果写入 shared memory，再由第一个 warp 合并所有 warp 的结果：

```text
thread local values
    -> warp shuffle reduction
    -> 每个 warp 的 lane 0 写 shared memory
    -> warp 0 完成最终 reduction
```

相较基础版，优化版只在跨 warp 合并前后进行 block 同步，并且 shared memory 只保存每个 warp 的 partial result。LayerNorm 在同一轮 shuffle 中同时归约 `sum` 与 `square_sum`。

## 编译与运行

RTX 2080 的 compute capability 为 7.5，因此本次测试使用 `sm_75`：

```bash
nvcc -O3 -std=c++17 -arch=sm_75 01_layernorm_naive.cu -o 01_layernorm_naive
nvcc -O3 -std=c++17 -arch=sm_75 02_layernorm_optimized.cu -o 02_layernorm_optimized
nvcc -O3 -std=c++17 -arch=sm_75 03_rmsnorm_naive.cu -o 03_rmsnorm_naive
nvcc -O3 -std=c++17 -arch=sm_75 04_rmsnorm_optimized.cu -o 04_rmsnorm_optimized

./01_layernorm_naive
./02_layernorm_optimized
./03_rmsnorm_naive
./04_rmsnorm_optimized
```

在 RTX 3090 上测试时，将编译目标改为 `-arch=sm_86`。

## 测试方法

四个程序使用完全相同的确定性输入。每次测试先执行 10 次 warm-up，再用 CUDA Event 统计 100 次连续 kernel launch 的总时间；`average_ms` 不包括内存分配、Host/Device 拷贝或 CPU reference。

为了提供统一且容易比较的带宽指标，程序按输入 `X` 最少读取一次、输出 `Y` 写入一次计算 effective bandwidth：

```text
effective_bandwidth_gbps =
    ROWS * HIDDEN * 2 * sizeof(float) / (average_ms * 10^6)
```

这是算法最低数据量对应的有效带宽，不是 Nsight Compute 的实际 DRAM 吞吐；它不计入统计阶段对 X 的第二次读取，也不计入 gamma/beta 流量和 cache 复用。

此外，四个 kernel 均使用以下方式检查了一个正式 launch，Compute Sanitizer 全部报告 `ERROR SUMMARY: 0 errors`：

```bash
compute-sanitizer --tool memcheck --launch-skip 10 --launch-count 1 ./程序名
```

## RTX 2080 实测结果

测试环境：NVIDIA GeForce RTX 2080、compute capability 7.5、驱动 595.84、CUDA Toolkit 13.2，编译参数为 `-O3 -std=c++17 -arch=sm_75`。输入形状为 `4096 x 1024`。

| Kernel | 最大绝对误差 | 相对 L-infinity 误差 | 平均时间 (ms) | 有效带宽 (GB/s) | 相对基础版 | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `layernorm_naive_fp32` | 0.00005221 | 0.00001872 | 0.10828288 | 309.87753334 | 1.000x | PASS |
| `layernorm_float4_warp_fp32` | 0.00005984 | 0.00002145 | 0.08745440 | 383.67919516 | 1.238x | PASS |
| `rmsnorm_naive_fp32` | 0.00000036 | 0.00000018 | 0.10291328 | 326.04568232 | 1.000x | PASS |
| `rmsnorm_float4_warp_fp32` | 0.00000036 | 0.00000018 | 0.08712160 | 385.14480421 | 1.181x | PASS |

LayerNorm 优化版相对基础版减少 19.24% 的执行时间，有效带宽提高 23.82%；RMSNorm 优化版减少 15.34% 的执行时间，有效带宽提高 18.13%。收益主要来自 `float4` 降低指令数，以及 warp shuffle 减少 shared-memory 读写和 block barrier。

Checksum：

| Kernel | Reference checksum | Output checksum |
| --- | ---: | ---: |
| `layernorm_naive_fp32` | 8278.85492913 | 8278.85952489 |
| `layernorm_float4_warp_fp32` | 8278.85492913 | 8278.85443678 |
| `rmsnorm_naive_fp32` | 756.31290937 | 756.31664025 |
| `rmsnorm_float4_warp_fp32` | 756.31290937 | 756.31531235 |

LayerNorm 基础版和优化版的前 8 个输出分别为：

```text
naive:
-2.00631189,-2.00021124,-1.99396801,-1.98758245,
-1.98105645,-1.97439134,-1.96758854,-1.96064925

optimized:
-2.00631356,-2.00021315,-1.99396968,-1.98758423,
-1.98105824,-1.97439301,-1.96759009,-1.96065104
```

两个 RMSNorm 版本的前 8 个输出相同：

```text
0.29499763,0.29652083,0.29802224,0.29950151,
0.30095840,0.30239251,0.30380359,0.30519131
```

浮点归约不满足结合律，tree reduction 和 warp shuffle 的加法顺序不同，因此两个 LayerNorm 版本与 FP64 reference 的误差、checksum 不会逐位相同；它们均满足本测试的误差阈值。

## 限制与后续优化

- 当前 kernel 固定假设 `THREADS` 是 32 的整数倍；基础版的 tree reduction 还要求它是 2 的幂。
- 优化版要求 `HIDDEN % 4 == 0`。若要支持任意 hidden size，需要增加标量尾部处理。
- 当前方差公式为 `E[x^2] - E[x]^2`。对于均值很大而方差很小的数据，可改用 Welford reduction 提高数值稳定性。
- 当前每行只使用一个 block，适合常见的中等 hidden size。非常长的行可考虑多 block 分段归约，较短的行可考虑一个 warp 处理一行。
- 进一步优化方向包括 FP16/BF16 输入、向量化参数缓存、融合 residual/bias、按 hidden size 专门化 block 大小，以及在目标 GPU 上用 Nsight Compute 分析 memory throughput 和 warp stall。
