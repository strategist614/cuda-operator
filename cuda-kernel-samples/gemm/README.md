# CUDA GEMM 优化示例

本目录包含 6 个可独立编译、运行的 CUDA GEMM 示例，按照以下路线逐步优化：

```text
Naive FP32
    -> Shared-memory tiling FP32
    -> Warp/register tiling FP32
    -> WMMA Tensor Core（FP16 输入、FP32 累加）
    -> WMMA warp reuse + vectorized load + direct store
    -> Ampere cp.async double buffering
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

## `gemm.cu` 汇总代码说明

[`gemm.cu`](gemm.cu) 集中保留了前 4 个 GEMM kernel，包括 3 个 FP32 版本和 1 个 WMMA 版本，展示从直接计算、shared-memory 分块、寄存器分块到 Tensor Core 的优化过程。它只包含 device kernel，不包含头文件、分块常量、host 端内存管理、kernel launch 和 `main()`，因此不能作为独立程序直接编译。可独立运行的完整程序位于 `01_gemm_naive.cu`、`02_gemm_shared_tile.cu`、`03_gemm_register_tile.cu`、`04_gemm_wmma.cu`、`05_gemm_wmma_optimized.cu` 和 `06_gemm_wmma_cp_async.cu`。

前三个 FP32 kernel 使用相同的接口：

```cpp
const float* A;  // [m, k_size]
const float* B;  // [k_size, n]
float* C;        // [m, n]
```

矩阵均为行主序，计算 `C = A x B`。当前接口没有 `alpha`、`beta`、转置选项或 batch 维度，计算结果会直接覆盖 `C`。

### `gemm_naive`

每个二维线程负责计算 `C` 的一个元素。线程遍历整个 K 维，每次直接从 global memory 读取一个 A 元素和一个 B 元素：

```text
thread(row, col)
    -> sum = A[row, 0] * B[0, col] + ...
    -> C[row, col] = sum
```

建议使用二维 block，例如 `dim3 block(16, 16)`，grid 在 M、N 两个方向分别向上取整。这个版本结构最简单，可作为正确性基线，但没有显式的数据复用。

> 注意：`gemm.cu` 当前把列坐标写成了 `blockIdx.x * blockIdx.x + threadIdx.x`，应为 `blockIdx.x * blockDim.x + threadIdx.x`。在修正之前，多数 grid 配置会产生错误或重复的列索引。独立示例 `01_gemm_naive.cu` 中已使用正确写法。

### `gemm_shared_tile`

每个 `TILE x TILE` block 负责 C 的一个同尺寸输出块。计算沿 K 维分段进行：

1. block 内线程协作把 A、B 的 tile 搬入 `As`、`Bs`。
2. 越界元素填 0，因此 M、N、K 不必是 `TILE` 的整数倍。
3. 同步后，每个线程复用 shared memory 中的数据完成 `TILE` 次乘加。
4. 遍历完所有 K tile 后，将结果写回 C。

该 kernel 要求定义编译期常量 `TILE`，并使用 `dim3 block(TILE, TILE)`。shared memory 用量为：

```text
2 x TILE x TILE x sizeof(float)
```

> 注意：每轮乘加结束后还需要一次 `__syncthreads()`，确保所有线程读取完当前 tile 后，才允许下一轮覆盖 `As` 和 `Bs`。`gemm.cu` 当前缺少这次同步，存在竞态；`02_gemm_shared_tile.cu` 已包含完整的两次同步。

### `gemm_register_tile`

这个版本使用三级分块，让一个线程计算多个 C 元素：

| 层级 | 尺寸 | 作用 |
| --- | --- | --- |
| Block tile | `BM x BN x BK` | 一个 block 分阶段处理的 A、B、C 区域 |
| Warp tile | `WM x WN` | 一个 warp 负责的输出区域 |
| Thread tile | `TM x TN` | 一个线程保存在寄存器中的累加区域 |

每轮 K 分块的执行过程为：

```text
全局内存 A/B
    -> block 协作加载 As[BM][BK]、Bs[BK][BN]
    -> 每个线程读取 a_frag[TM]、b_frag[TN]
    -> 外积累加到 acc[TM][TN]
    -> 下一个 BK 分块
    -> 带边界检查写回 C
```

源码中的 lane 映射把 32 个线程拆为 `8 x 4`：

```cpp
lane_row = lane / 4;  // 0..7
lane_col = lane % 4;  // 0..3
```

因此当前实现要求 `WM = 8 x TM`、`WN = 4 x TN`。`warp_row = warp_id / 2`、`warp_col = warp_id % 2` 又把 4 个 warp 排成 `2 x 2`，所以还要求 `BM = 2 x WM`、`BN = 2 x WN`，并以 128 个线程启动一个 block。完整示例采用：

```cpp
BM = 32; BN = 32; BK = 8;
WM = 16; WN = 16;
TM = 2;  TN = 4;
THREADS = 128;
```

对应的 launch 配置为：

```cpp
dim3 block(THREADS);
dim3 grid((n + BN - 1) / BN,
          (m + BM - 1) / BM);
gemm_register_tile<<<grid, block>>>(A, B, C, m, n, k_size);
```

相比 shared-memory tiling，每个线程用 `acc[TM][TN]` 保存多个输出，A fragment 和 B fragment 能在寄存器中重复参与乘加，从而进一步减少每个输出元素对应的 shared-memory 读取次数。代价是寄存器占用增加，并且分块参数必须满足上述线程映射约束。

### 四个 kernel 的区别

| Kernel | 每线程输出 | 数据复用位置 | 主要限制 |
| --- | ---: | --- | --- |
| `gemm_naive` | 1 | 主要依赖硬件 cache | `gemm.cu` 中列索引需修正 |
| `gemm_shared_tile` | 1 | Shared memory | block 必须为 `TILE x TILE`；需补第二次同步 |
| `gemm_register_tile` | `TM x TN` | Shared memory + registers | 需满足固定的 warp/lane 分块关系 |
| `gemm_wmma` | 每个 warp 一个 `16 x 16` tile | Shared memory + Tensor Core | FP16 输入、FP32 累加，并依赖 WMMA 常量与类型定义 |

## `gemm.cu` 完整代码

以下代码与当前 [`gemm.cu`](gemm.cu) 保持一致：

```cpp
__global__ void gemm_naive(const float* A, const float* B, float* C, int m,int n,int k_size){
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockIdx.x + threadIdx.x;

    if(row < m && col < n){
        float sum = 0.0f;

        for(int k = 0;k < k_size;k++) sum += A[row * k_size + k] * B[k * n + col];

        C[row * n + col] = sum;
    }
}

__global__ void gemm_shared_tile(const float* A, const float* B, float* C, int m,int n,int k_size){
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for(int bk = 0;bk < k_size; bk += TILE){
        const int a_col = bk + tx;
        const int b_row = bk + ty;
        As[ty][tx] = (row < m && a_col < k_size) ? A[row * k_size + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < k_size && col < n) ? B[b_row * n + col] : 0.0f;

        __syncthreads();
#pragma unroll
        for(int k = 0;k < TILE; ++k) sum += As[ty][k] * Bs[k][tx];
    }
    if(row < m && col < n) C[row * n + col] = sum;
}

__global__ void gemm_register_tile(const float* A, const float* B, float* C,int m, int n,int k_size){
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;
    const int lane_row = lane / 4;
    const int lane_col = lane % 4;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[TM][TN] = {};

    for(int bk = 0;bk < k_size;bk += BK){
        for(int i = tid ;i < BM * BK;i += blockDim.x){
            const int r = i / BK;
            const int c = i % BK;
            const int global_row = block_row + r;
            const int global_col = bk + c;
            As[r][c] = (global_row < m && global_col < k_size)
                           ? A[global_row * k_size + global_col]
                           : 0.0f;
        }
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            const int r = i / BN;
            const int c = i % BN;
            const int global_row = bk + r;
            const int global_col = block_col + c;
            Bs[r][c] = (global_row < k_size && global_col < n)
                           ? B[global_row * n + global_col]
                           : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for(int k = 0;k < BK;++k){
            float a_frag[TM];
            float b_frag[TN];
#pragma unroll
            for(int i = 0;i < TM;++i){
                const int row = warp_row * WM + lane_row * TM + i;
                a_frag[i] = As[row][k];
            }
#pragma unroll
            for(int j = 0;j < TN; ++j){
                const int col = warp_col * WN + lane_col * TN + j;
                b_frag[j] = Bs[k][col];
            }
#pragma unroll
            for(int i = 0;i < TM;i++)
#pragma unroll
                for(int j = 0;j < TN;j ++) acc[i][j] += a_frag[i] * b_frag[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int row = block_row + warp_row * WM + lane_row * TM + i;
            const int col = block_col + warp_col * WN + lane_col * TN + j;
            if (row < m && col < n) C[row * n + col] = acc[i][j];
        }
}

__global__ void gemm_wmma(const half* A, const half* B, float* C, int m, int n, int k_size){
    __shared__ __align__(32) half As[BM][BK];
    __shared__ __align__(32) half Bs[BK][BN];
    __shared__ __align__(32) float Cs[BM][BN];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for(int bk = 0; bk < k_size; bk += BK){
        for(int i = tid; i < BM * BK;i += blockDim.x){
            const int r = i / BK;
            const int c = i % BK;
            const int global_row = block_row + r;
            const int global_col = bk + c;

            As[r][c] = (global_row < m && global_col < k_size) ? A[global_row * k_size + global_col] : __float2half(0.0f);
        }

        for(int i = tid;i < BK * BN;i += blockDim.x){
            const int r = i / BN;
            const int c = i % BN;
            const int global_row = bk + r;
            const int global_col = block_col + c;

            Bs[r][c] = (global_row < k_size && global_col < n) ? B[global_row * n + global_col] : __float2half(0.0f);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        const int warp_m = warp_row * 16;
        const int warp_n = warp_col * 16;
        wmma::load_matrix_sync(a_frag, &As[warp_m][0], BK);
        wmma::load_matrix_sync(b_frag, &Bs[0][warp_n], BN);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }
    const int warp_m = warp_row * 16;
    const int warp_n = warp_col * 16;
    wmma::store_matrix_sync(&Cs[warp_m][warp_n], c_frag, BN, wmma::mem_row_major);
    __syncthreads();

    for (int i = tid; i < BM * BN; i += blockDim.x) {
        const int r = i / BN;
        const int c = i % BN;
        const int global_row = block_row + r;
        const int global_col = block_col + c;
        if (global_row < m && global_col < n)
            C[global_row * n + global_col] = Cs[r][c];
    }
}
```

当前 6 个程序统一使用：

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
| `05_gemm_wmma_optimized.cu` | `wmma_optimized_fp16_acc_fp32` | `BM=64, BN=64, BK=16`，8 warps/block | 每个 warp 计算 `16 x 32`，复用 A fragment，向量化加载，并让完整输出 tile 直接写回 global memory |
| `06_gemm_wmma_cp_async.cu` | `wmma_cp_async_fp16_acc_fp32` | `BM=64, BN=64, BK=32`，8 warps/block，`sm_80+` | `cp.async` 双缓冲下一块 A/B，当前 shared stage 同时执行 WMMA；针对对齐尺寸的 fast kernel |

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

### 5. Optimized WMMA Tensor Core

05 保持 04 的 `64 x 64 x 16` block tile、FP16 输入和 FP32 accumulator，但重新安排了 warp 工作量：

```text
04: 16 warps/block x 每 warp 1 个 16 x 16 输出 tile
05:  8 warps/block x 每 warp 2 个 16 x 16 输出 tile
                         = 每 warp 负责 16 x 32
```

每个 warp 只加载一次 A fragment，然后与两个相邻的 B fragment 分别执行 `mma_sync`。这样可以把 A fragment 的 WMMA load 次数减半，同时把 block 线程数从 512 降到 256。

05 还包含两条 fast path：

- 完整 A/B tile 使用 `int4` 搬运，每条指令加载 8 个 half（16 bytes）。`sm_75` SASS 中可以看到 `LDG.E.128`。
- 完整的 `16 x 16` 输出 tile 由 `wmma::store_matrix_sync` 直接写入 C，避免 04 固定经过 `64 x 64` FP32 shared-memory 缓冲；只有边界 tile 才使用每 warp 独占的 shared-memory scratch。

边界 M/N/K 和不满足向量对齐的 leading dimension 会自动回退到标量、带补零的 shared-memory 加载；边界输出也会先写入 scratch，再逐元素检查后写回。

使用 CUDA Toolkit 13.2、目标 `sm_75` 的静态编译结果：

| 版本 | Threads/block | Registers/thread | Static shared memory | Spill |
| --- | ---: | ---: | ---: | ---: |
| 04 | 512 | 29 | 20,480 bytes | 0 |
| 05 | 256 | 48 | 12,288 bytes | 0 |

05 的单线程寄存器数增加，因为一个 warp 同时保存两个 accumulator fragment；但 block 线程数减半，估算的 registers/block 从 14,848 降为 12,288，static shared memory 也减少 8 KiB。实际速度仍取决于 GPU 架构、矩阵尺寸、频率和 occupancy，应在目标 GPU 上用同一输入重复 benchmark，不能只根据静态资源占用判断性能。

### 6. Ampere `cp.async` Double Buffering

06 面向 RTX 3090（`sm_86`）及其他 Ampere 或更新架构。在 05 的每 warp `16 x 32` 输出基础上，把 K stage 从 16 扩大到 32，并为 A/B 各准备两套带 padding 的 shared-memory buffer：

```text
As[2][64][32 + 8]
Bs[2][32][64 + 8]
```

每个 256-thread block 中，每个线程为 A 和 B 各发出一次 16-byte `cp.async`。流水过程为：

```text
prologue: cp.async tile 0 -> stage 0，等待完成

loop:
    cp.async tile N+1 -> write stage
    对 read stage 执行两组 K=16 WMMA
    cp.async.wait_group 0
    __syncthreads()
    交换 read/write stage
```

因此下一块 global→shared 传输可以和当前块的 Tensor Core 计算重叠。`BK=32` 让每个 stage 连续执行两个 WMMA K slice，也把 K-loop 的 block 同步轮数从 32 降到 16。`SKEW_HALF=8` 让 shared-memory 行跨度发生偏移，用于降低连续 WMMA load 映射到相同 bank 的风险；最终效果仍应由 Nsight Compute 的 shared wavefront/bank-conflict 指标确认。

06 是固定对齐尺寸的 fast kernel：M 必须是 64 的倍数，N 必须是 64 的倍数，K 必须是 32 的倍数，并要求 `sm_80+`。它不包含 05 的边界 fallback 和 edge scratch，因此当前 `512 x 512 x 512` 会全程走无分支流水路径。

使用 CUDA Toolkit 13.2、目标 `sm_86` 的静态编译结果：

| 版本 | Threads/block | Registers/thread | Static shared memory | Spill |
| --- | ---: | ---: | ---: | ---: |
| 05 (`sm_86`) | 256 | 40 | 12,288 bytes | 0 |
| 06 (`sm_86`) | 256 | 40 | 19,456 bytes | 0 |

06 的 shared memory 增加来自双缓冲和 padding，但寄存器数没有高于 05。SASS 已确认生成 `LDGSTS.E.BYPASS.128`、`HMMA.16816.F32` 和 `DEPBAR`：异步 global→shared copy 在 HMMA 前发出，在当前 stage 计算完成后才等待下一 stage。

## 编译

在当前目录执行：

```bash
nvcc -O3 -std=c++17 -arch=sm_75 01_gemm_naive.cu -o 01_gemm_naive
nvcc -O3 -std=c++17 -arch=sm_75 02_gemm_shared_tile.cu -o 02_gemm_shared_tile
nvcc -O3 -std=c++17 -arch=sm_75 03_gemm_register_tile.cu -o 03_gemm_register_tile
nvcc -O3 -std=c++17 -arch=sm_75 04_gemm_wmma.cu -o 04_gemm_wmma
nvcc -O3 -std=c++17 -arch=sm_75 05_gemm_wmma_optimized.cu -o 05_gemm_wmma_optimized
nvcc -O3 -std=c++17 -arch=sm_86 06_gemm_wmma_cp_async.cu -o 06_gemm_wmma_cp_async
```

## 运行

```bash
./01_gemm_naive
./02_gemm_shared_tile
./03_gemm_register_tile
./04_gemm_wmma
./05_gemm_wmma_optimized
./06_gemm_wmma_cp_async
```

程序返回 `0` 表示正确性验证通过，返回非零值表示失败或发生 CUDA 错误。

## 正确性与计时方法

6 个程序使用相同的确定性伪随机输入，随机状态种子为 `20260822`。CPU reference 使用 FP64 累加，GPU 输出与 reference 比较以下指标：

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

以下包含原有 01–04 与本次 05 在 `512 x 512 x 512` 矩阵上的运行记录，5 个 kernel 均通过正确性验证。05 的测试环境为 NVIDIA GeForce RTX 2080（compute capability 7.5）、驱动 595.84、CUDA Toolkit 13.2，使用 `-arch=sm_75` 编译：

| Kernel | 数据类型 | 最大绝对误差 | 相对 L-infinity 误差 | 平均时间 (ms) | GFLOPS | 相对 Naive 加速 | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `naive_fp32` | FP32 | 0.00003662 | 0.00000104 | 0.39880159 | 673.10527185 | 1.00x | PASS |
| `shared_tile_fp32` | FP32 | 0.00003662 | 0.00000104 | 0.25700480 | 1044.47644034 | 1.55x | PASS |
| `register_tile_fp32` | FP32 | 0.00003662 | 0.00000104 | 0.15227520 | 1762.83103084 | 2.62x | PASS |
| `wmma_fp16_acc_fp32` | FP16 / FP32 | 0.00888518 | 0.00025182 | 0.07969280 | 3368.37794509 | 5.00x | PASS |
| `wmma_optimized_fp16_acc_fp32` | FP16 / FP32 | 0.00888518 | 0.00025182 | 0.05668160 | 4735.84860275 | 7.04x | PASS |

Checksum：

| Kernel | Reference checksum | Output checksum |
| --- | ---: | ---: |
| `naive_fp32` | 3203.57104091 | 3203.57244194 |
| `shared_tile_fp32` | 3203.57104091 | 3203.57244194 |
| `register_tile_fp32` | 3203.57104091 | 3203.57244194 |
| `wmma_fp16_acc_fp32` | 3203.57104091 | 3204.11095608 |
| `wmma_optimized_fp16_acc_fp32` | 3203.57104091 | 3204.11095608 |

前三个 FP32 kernel 的前 8 个输出值相同：

```text
4.70811033, -6.73353481, 17.86639595, 4.12949896,
-11.51244068, -10.26406097, -5.85342073, 14.09879971
```

04、05 WMMA kernel 的前 8 个输出值相同：

```text
4.70817852, -6.73169994, 17.86602020, 4.12787628,
-11.51045609, -10.25916195, -5.85119247, 14.09688377
```

这组结果表明，随着 shared-memory、register tiling 和 Tensor Core 的引入，当前测试中的吞吐量依次提高。05 相比 04 的平均时间从 `0.07969280 ms` 降到 `0.05668160 ms`，时间减少 28.87%，吞吐量提高 1.406 倍；相对 naive 达到 7.04 倍加速。04、05 的误差、checksum 和前 8 个输出完全一致，说明新增的 warp tile 复用、128-bit 加载和直接写回没有改变当前输入上的数值结果。

### Nsight Compute：04 与 05 对比

在同一台 RTX 2080 上使用 Nsight Compute 的 `basic` 指标集采集 04、05。程序会先执行 5 次 warm-up，因此跳过前 5 次 launch，只分析随后第 1 次正式 launch：

```bash
sudo ncu --set basic --launch-skip 5 --launch-count 1 ./04_gemm_wmma
sudo ncu --set basic --launch-skip 5 --launch-count 1 ./05_gemm_wmma_optimized
```

两个 kernel 在 profiler 下均输出 `status=PASS`。主要硬件指标如下：

| Nsight Compute 指标 | 04 `gemm_wmma` | 05 `gemm_wmma_optimized` | 变化 |
| --- | ---: | ---: | ---: |
| Kernel duration | 73.79 us | 50.69 us | 减少 31.30% |
| Elapsed cycles | 124,978 | 85,867 | 减少 31.29% |
| Block size | 512 threads | 256 threads | 减半 |
| Grid size | 64 blocks | 64 blocks | 不变 |
| Registers/thread | 29 | 48 | 增加 19 |
| Static shared memory/block | 20.48 KB | 12.29 KB | 减少约 40.0% |
| Memory throughput | 30.47% | 37.36% | 提高 |
| DRAM throughput | 4.14% | 7.19% | 提高 |
| L1/TEX throughput | 60.95% | 74.73% | 提高 |
| L2 throughput | 7.91% | 11.53% | 提高 |
| Theoretical occupancy | 100% | 100% | 不变 |
| Achieved occupancy | 69.45% | 35.01% | 受小 grid 和较少 warp 影响 |
| Active warps/SM | 22.23 | 11.20 | 约减半 |
| Waves/SM | 0.70 | 0.35 | 减半 |

`Duration` 和 `Elapsed Cycles` 都表明 05 的单次 kernel 硬件执行时间约为 04 的 `0.687` 倍，即 Nsight Compute 采样中获得约 `1.456x` 加速。这与脱离 profiler 后 CUDA Event 测得的 05 加速趋势一致，说明性能提升并非普通运行计时的偶然波动。

05 的 achieved occupancy 较低不代表资源限制变严重：两个版本的 theoretical occupancy 都是 100%。当前 `512 x 512` 输出只产生 64 个 block，而 RTX 2080 有 46 个 SM，平均只有约 1.39 个 block/SM。05 又把每个 block 从 16 个 warp 减少到 8 个 warp，因此在相同 grid 下 `Waves/SM` 和 active warps 近似减半。05 虽然同时驻留的 warp 更少，但通过每个 warp 计算两个相邻 WMMA tile、复用 A fragment、128-bit 加载、减少 shared memory 和直接写回，仍用更少的周期完成计算。

注意：`ncu` 为采集 `basic` 指标执行了 9 次 replay。profiler 运行期间程序自身打印的几十毫秒 `average_ms` 包含 replay、暂停和采集开销，不能作为单次 kernel 性能；分析 NCU 结果应看报告中的 `Kernel duration`，常规性能比较则使用脱离 `ncu` 后的 CUDA Event 结果。当前问题规模还不足以填满整个 GPU，如需分析稳定的 Tensor Core、访存和 stall 指标，应使用更大的矩阵，并让 grid 包含更多 block。

WMMA 使用 FP16 输入，因此不能把它与 FP32 kernel 视为完全相同精度下的性能对比。运行时间和 GFLOPS 也依赖 GPU 型号、频率、CUDA 版本、编译目标以及系统负载，本表只记录对应运行结果。06 要求 `sm_80+`，无法在当前 RTX 2080 上执行；请在 RTX 3090 上同时重新运行 04、05、06，以相同环境比较 `average_ms`、GFLOPS 与 `status`。

## 修改测试配置

如需测试其他矩阵尺寸或运行次数，请修改每个源码顶部的常量并重新编译：

```cpp
constexpr int M = 512;
constexpr int N = 512;
constexpr int K = 512;
constexpr int WARMUP = 5;
constexpr int REPEATS = 20;
```

01–05 带有边界加载、补零和写回判断，可以处理非 tile 整数倍的 M/N/K。06 是为 `cp.async` 流水准备的对齐 fast kernel，修改尺寸时必须保持 `M % 64 == 0`、`N % 64 == 0`、`K % 32 == 0`。所有版本修改尺寸后都应重新检查 `status` 和误差指标。
