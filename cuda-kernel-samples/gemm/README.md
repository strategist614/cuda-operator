# CUDA GEMM Samples

本目录用于练习行主序 FP32 GEMM 以及 WMMA/Tensor Core 矩阵乘。代码按 naive、shared-memory tiling、warp/register tiling、WMMA 的方向逐步展开，目前仍是开发中的学习草稿，不是可直接使用的性能库。

## GEMM 定义

输入和输出采用行主序：

```text
A: [M, K]
B: [K, N]
C: [M, N]

C[row, col] = Σ A[row, k] × B[k, col]
```

对应一维索引：

```text
A[row, k] = A[row * K + k]
B[k, col] = B[k * N + col]
C[row, col] = C[row * N + col]
```

## 文件说明

| 文件 | 内容 | 当前状态 |
| --- | --- | --- |
| `gemm.cu` | naive、block tile、warp/register tile、WMMA 草稿和 host 端示例 | 当前无法编译 |
| `gemm_test.cu` | 独立整理 naive kernel，并预留空的 block-tile kernel | 可编译为 object，但没有 `main()` |
| `gemm` | 目录中已有的无扩展名可执行文件 | 历史编译产物，不保证对应当前源码 |

## Kernel 版本

### gemm_naive

每个 CUDA 线程计算 C 的一个元素：

```cpp
int row = blockDim.y * blockIdx.y + threadIdx.y;
int col = blockDim.x * blockIdx.x + threadIdx.x;

for (int k = 0; k < K; ++k) {
    sum += A[row * K + k] * B[k * N + col];
}
```

当前 `main()` 只启动这个版本，使用 16×16 threads 的二维 block。相邻线程访问 B 的相邻列，B 读取容易合并；同一行 A 的值会被多个线程重复读取。该版本适合作为正确性 baseline，但数据复用少、全局内存访问量高。

### gemm_block_tile

设计目标是使用 16×16 shared-memory tile：

1. 每个 block 负责 C 的一个 tile。
2. 协作把 A/B 子块加载到 shared memory。
3. 同步后在 tile 内累加。
4. 沿 K 维循环，最后写回 C。

当前实现尚未完成，主要问题包括：

- Kernel 参数之间错误使用分号而不是逗号。
- B tile 逻辑上应写成 `Bs[BK][BN]`。
- 内层乘加循环写成 `k < bk`，应根据当前 tile 的 K 宽度遍历，例如 `k < BK`。
- 尚未单独配置 launch 或执行正确性验证。

### gemm_all_tile

该草稿尝试组合：

- 32×32×8 的 block tile。
- 16×16 的 warp tile。
- 每线程 2×4 的 register tile。
- 多元素协作加载 shared memory。
- 寄存器 fragment 和循环展开。

按设计，一个 warp 的 32 个 lane 可以覆盖 16×16 输出：

```text
lane_row = lane / 4   → 8 组 row
lane_col = lane % 4   → 4 组 col
TM × TN = 2 × 4

8 × 2 = 16 rows
4 × 4 = 16 cols
```

2×2 个 warp tile 可覆盖 32×32 block tile，因此预期使用 4 个 warp，也就是 128 个线程。

当前源码存在 `lene_col/lane_col`、`blcok_row/block_row` 等变量拼写不一致，无法编译。实现完成后还需检查 shared-memory 加载覆盖、尾块补零、每个线程输出坐标和寄存器压力。

### gemm_wmma

该草稿尝试使用 `nvcuda::wmma` 和 Tensor Core：

```text
A/B input: half
C accumulator/output: float
WMMA tile: 16 × 16 × 16
Block tile: 128 × 128 × 16（当前设想）
```

核心 API：

- `wmma::fragment`：声明 A、B 和 accumulator fragment。
- `wmma::load_matrix_sync`：从 shared memory 加载 fragment。
- `wmma::mma_sync`：执行矩阵乘加。
- `wmma::store_matrix_sync`：把 accumulator 写回 C。

WMMA 通常需要 Volta（SM 70）或更新架构，编译时应选择实际 GPU 对应的 `-arch=sm_XX`，并显式包含 `<mma.h>`；half 类型相关代码也建议包含 `<cuda_fp16.h>`。

当前 WMMA 草稿还不能正确工作：

- 两处 `int r = ...:` 使用冒号而非分号。
- 加载 B tile 时误从 A 指针读取。
- B 在 shared memory 中按普通二维数组存放，却用 `wmma::col_major` fragment 解释，布局和 leading dimension 需要重新统一。
- 当前 warp 布局只有 2×4 个 warp tile，即只覆盖 32×64 输出，不能覆盖声明的 128×128 block tile。
- 没有 host 端 half 输入分配、数据转换、WMMA launch 和验证。
- 边界处 `wmma::store_matrix_sync` 会写完整 16×16 tile，仅检查 tile 左上角不足以保护非整 tile 尺寸。

要覆盖 128×128 block tile，需要让每个 warp 计算多个 16×16 tile，或缩小 block tile；不能简单在一个 CUDA block 中启动 8×8=64 个 warp，因为这会超过单 block 最大线程数。

## 当前编译状态

使用 CUDA Toolkit 13.2 验证：

```bash
nvcc -std=c++17 -c gemm_test.cu -o /tmp/gemm_test.o
```

该命令可以通过。由于 `gemm_test.cu` 没有 `main()`，它只能编译为 object，不能直接链接成独立程序。

当前 `gemm.cu` 编译失败：

```bash
nvcc -std=c++17 -c gemm.cu -o /tmp/gemm.o
```

主要错误和警告包括：

- `gemm_block_tile` 参数列表中的分号。
- `BM/BN/BK` 被多次 `#define`，产生宏重定义警告。
- `block_row`、`lane_col` 等未定义变量。
- WMMA 部分的冒号语法错误。
- `h_C` 分配语句末尾包含 Unicode `·` 字符。
- 前面的语法错误还会引发大量级联解析错误。

建议不要直接运行目录中的 `./gemm` 来判断当前源码状态，因为该二进制可能来自较早版本。

## 推荐修复顺序

1. 从 `gemm_test.cu` 的 naive kernel 开始，补充 `main()` 和 CPU reference。
2. 统一使用 `constexpr int` 或版本专属名称替代重复的 `BM/BN/BK` 宏。
3. 单独实现并验证 16×16 block-tile kernel。
4. 再实现 32×32 block + warp/register tile，确认 128-thread launch。
5. 最后独立整理 WMMA kernel，先限定 M/N/K 为 16 的倍数。
6. 每增加一个版本，都先执行随机输入正确性测试，再做 benchmark。

建议把不同 kernel 拆到独立文件或使用不同常量名，例如：

```text
gemm_naive.cu
gemm_shared_tile.cu
gemm_register_tile.cu
gemm_wmma.cu
```

这样可以避免宏重定义和未完成版本阻止整个文件编译。

## 正确性验证

只打印 C 的前 10 个元素不足以证明 GEMM 正确。推荐实现 CPU reference：

```cpp
for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C_ref[row * N + col] = sum;
    }
}
```

然后统计最大绝对误差和相对误差。FP32、half 输入/FP32 accumulator 的容差应分别设置，不能要求 WMMA 与 CPU FP32 的结果逐 bit 相同。

还应覆盖：

- M/N/K 不是 tile 整数倍的尺寸。
- 很小矩阵和非方阵。
- 全零、随机正负数和较大数值。
- 重复运行以及不同 GPU 架构。

## 性能测试

完成正确性后，可用 CUDA event 只测 kernel 时间：

```text
warm-up
cudaEventRecord(start)
重复 launch kernel
cudaEventRecord(stop)
cudaEventSynchronize(stop)
```

GEMM 的理论运算量为：

```text
FLOPs = 2 × M × N × K
TFLOPS = FLOPs / time_seconds / 1e12
```

比较 kernel 时应固定 M/N/K、dtype、编译架构、warm-up 次数和计时轮数，并同时记录：

- kernel 时间与 TFLOPS。
- shared memory 和寄存器占用。
- occupancy。
- global-memory load/store efficiency。
- shared-memory bank conflict。
- 与 cuBLAS 的正确性和性能差距。

## 资源管理注意事项

当前 `main()` 只适用于 M=N=K=1024 的方阵示例，且没有完整释放资源。整理成可运行程序时应：

- 分别按 `M*K`、`K*N`、`M*N` 计算 A/B/C 字节数。
- 对每次 kernel launch 使用 `cudaGetLastError()` 和同步错误检查。
- 调用 `cudaFree()` 释放 d_A/d_B/d_C。
- 调用 `delete[]` 释放 h_A/h_B/h_C。
- 把内存分配和 H2D/D2H 拷贝排除在纯 kernel benchmark 之外。
