# CUDA GEMM 优化示例

本目录包含 4 个可独立编译、运行的 CUDA GEMM 示例，按照以下路线逐步优化：

```text
Naive FP32
    -> Shared-memory tiling FP32
    -> Warp/register tiling FP32
    -> WMMA Tensor Core（FP16 输入、FP32 累加）
```

这些程序用于学习 GEMM kernel 的数据复用和分块方法，不是通用矩阵乘法库。当前矩阵尺寸、预热次数和计时次数均为编译期常量。

## GEMM 定义

所有矩阵均按行主序存储，计算：

```text
A: [M, K]
B: [K, N]
C: [M, N]

C[row, col] = sum(A[row, k] * B[k, col]), k = 0 ... K - 1
```

当前 4 个程序统一使用：

```cpp
M = 512;
N = 512;
K = 512;
WARMUP = 5;
REPEATS = 20;
```

## 文件与实现

| 文件 | Kernel | 主要配置 | 实现要点 |
| --- | --- | --- | --- |
| `01_gemm_naive.cu` | `naive_fp32` | `16 x 16` threads/block | 每个线程直接从全局内存读取 A、B，并计算 C 的一个元素 |
| `02_gemm_shared_tile.cu` | `shared_tile_fp32` | `TILE=16` | A、B 按 `16 x 16` 子块加载到 shared memory，每个线程计算 C 的一个元素 |
| `03_gemm_register_tile.cu` | `register_tile_fp32` | `BM=32, BN=32, BK=8`，128 threads/block | 4 个 warp 覆盖一个 `32 x 32` 输出块，每线程用寄存器计算 `TM x TN = 2 x 4` 个元素 |
| `04_gemm_wmma.cu` | `wmma_fp16_acc_fp32` | `BM=64, BN=64, BK=16`，16 warps/block | 每个 warp 通过 WMMA 计算一个 `16 x 16 x 16` tile，FP16 输入、FP32 accumulator 和输出 |

### 1. Naive FP32

每个线程负责一个输出元素：

```cpp
const int row = blockIdx.y * blockDim.y + threadIdx.y;
const int col = blockIdx.x * blockDim.x + threadIdx.x;

float sum = 0.0f;
for (int k = 0; k < K; ++k) {
    sum += A[row * K + k] * B[k * N + col];
}
C[row * N + col] = sum;
```

这个版本是正确性和性能基线。它没有显式缓存数据，同一个 A/B 元素会被多个线程重复从全局内存读取。

### 2. Shared-memory tiling FP32

每个 block 协作加载 A、B 的 `16 x 16` tile，然后在 shared memory 中复用数据。计算沿 K 维以 16 为步长推进：

```text
load A tile and B tile
        -> __syncthreads()
        -> 16 次乘加
        -> __syncthreads()
        -> 下一个 K tile
```

加载时包含边界判断，越界位置补零；写回 C 时也会检查行列边界。

### 3. Warp/register tiling FP32

每个 block 使用 128 个线程（4 个 warp）计算一个 `32 x 32` 输出块：

```text
Block tile:    BM x BN x BK = 32 x 32 x 8
Warp tile:     WM x WN      = 16 x 16
Thread tile:   TM x TN      =  2 x 4
```

一个 warp 中的 lane 映射为：

```text
lane_row = lane / 4
lane_col = lane % 4

8 lane-row groups x 2 rows = 16 rows
4 lane-col groups x 4 cols = 16 cols
```

每个线程把 `2 x 4` 个累加值保存在寄存器中，从而在 shared-memory tiling 的基础上增加线程内数据复用。

### 4. WMMA Tensor Core

WMMA 版本先在 host 端把 FP32 输入转换为 FP16，再使用 Tensor Core 执行矩阵乘加：

```text
Input A/B:       FP16
Accumulator C:   FP32
WMMA tile:       16 x 16 x 16
Block tile:      64 x 64 x 16
Warp layout:     4 x 4（共 16 个 warp、512 个线程）
```

A、B tile 先加载到 shared memory。每个 warp 负责一个 `16 x 16` 输出 tile，结果先写入 shared-memory 中的 `64 x 64` FP32 缓冲区，再由整个 block 带边界检查地写回全局内存。

WMMA 需要支持 Tensor Core 的 GPU。示例编译目标为 `sm_75`；在其他 GPU 上应把 `-arch` 改成对应的 compute capability。

## 编译

在当前目录执行：

```bash
nvcc -O3 -std=c++17 -arch=sm_75 01_gemm_naive.cu -o 01_gemm_naive
nvcc -O3 -std=c++17 -arch=sm_75 02_gemm_shared_tile.cu -o 02_gemm_shared_tile
nvcc -O3 -std=c++17 -arch=sm_75 03_gemm_register_tile.cu -o 03_gemm_register_tile
nvcc -O3 -std=c++17 -arch=sm_75 04_gemm_wmma.cu -o 04_gemm_wmma
```

## 运行

```bash
./01_gemm_naive
./02_gemm_shared_tile
./03_gemm_register_tile
./04_gemm_wmma
```

程序返回 `0` 表示正确性验证通过，返回非零值表示失败或发生 CUDA 错误。

## 正确性与计时方法

4 个程序使用相同的确定性伪随机输入，随机状态种子为 `20260822`。CPU reference 使用 FP64 累加，GPU 输出与 reference 比较以下指标：

- 最大绝对误差 `max_abs_error`。
- 相对 L-infinity 误差 `relative_linf = max_abs_error / max(abs(reference))`。
- reference 与输出的 checksum。
- 输出矩阵的前 8 个值。

FP32 kernel 的通过阈值为 `max_abs_error <= 1e-3`；WMMA 的通过阈值为 `max_abs_error <= 5e-2`。WMMA reference 仍由原始 FP32 输入计算，因此它的误差同时包含 FP16 输入量化误差。

性能测试先进行 5 次 warm-up，再用 CUDA event 统计 20 次连续 kernel launch 的总时间。输出的 `average_ms` 是单次 kernel 的平均时间，不包含 host/device 内存分配、数据传输和 CPU reference 计算。

GFLOPS 的计算公式为：

```text
FLOPs  = 2 x M x N x K
GFLOPS = FLOPs / (average_ms x 10^6)
```

## 本次运行结果

矩阵尺寸为 `512 x 512 x 512`，4 个 kernel 均通过正确性验证：

| Kernel | 数据类型 | 最大绝对误差 | 相对 L-infinity 误差 | 平均时间 (ms) | GFLOPS | 相对 Naive 加速 | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `naive_fp32` | FP32 | 0.00003662 | 0.00000104 | 0.39880159 | 673.10527185 | 1.00x | PASS |
| `shared_tile_fp32` | FP32 | 0.00003662 | 0.00000104 | 0.25700480 | 1044.47644034 | 1.55x | PASS |
| `register_tile_fp32` | FP32 | 0.00003662 | 0.00000104 | 0.15227520 | 1762.83103084 | 2.62x | PASS |
| `wmma_fp16_acc_fp32` | FP16 / FP32 | 0.00888518 | 0.00025182 | 0.07969280 | 3368.37794509 | 5.00x | PASS |

Checksum：

| Kernel | Reference checksum | Output checksum |
| --- | ---: | ---: |
| `naive_fp32` | 3203.57104091 | 3203.57244194 |
| `shared_tile_fp32` | 3203.57104091 | 3203.57244194 |
| `register_tile_fp32` | 3203.57104091 | 3203.57244194 |
| `wmma_fp16_acc_fp32` | 3203.57104091 | 3204.11095608 |

前三个 FP32 kernel 的前 8 个输出值相同：

```text
4.70811033, -6.73353481, 17.86639595, 4.12949896,
-11.51244068, -10.26406097, -5.85342073, 14.09879971
```

WMMA kernel 的前 8 个输出值为：

```text
4.70817852, -6.73169994, 17.86602020, 4.12787628,
-11.51045609, -10.25916195, -5.85119247, 14.09688377
```

这组结果表明，随着 shared-memory、register tiling 和 Tensor Core 的引入，当前测试中的吞吐量依次提高。WMMA 使用 FP16 输入，因此不能把它与 FP32 kernel 视为完全相同精度下的性能对比。运行时间和 GFLOPS 依赖 GPU 型号、频率、CUDA 版本、编译目标以及系统负载，本表只记录本次运行结果。

## 修改测试配置

如需测试其他矩阵尺寸或运行次数，请修改每个源码顶部的常量并重新编译：

```cpp
constexpr int M = 512;
constexpr int N = 512;
constexpr int K = 512;
constexpr int WARMUP = 5;
constexpr int REPEATS = 20;
```

当前源码带有边界加载、补零和写回判断，可以处理非 tile 整数倍的 M/N/K；修改尺寸后仍应检查程序输出的 `status` 和误差指标。
