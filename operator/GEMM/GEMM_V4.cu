#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>


#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA Error: "                       \
                      << cudaGetErrorString(err)               \
                      << " at " << __FILE__                    \
                      << ":" << __LINE__                       \
                      << std::endl;                            \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)


// ============================================================
// GEMM V4 Configuration
//
// V3:
//   Shared Memory Tiling
//   +
//   4x4 Register Tiling
//
// V4:
//   在 V3 基础上增加:
//
//   Global Memory
//        ↓
//      float4
//        ↓
//   Shared Memory
//
// 一个 block:
//      16 x 16 threads
//      = 256 threads
//
// 一个 thread:
//      4 x 4 C
//      = 16 outputs
//
// 一个 block:
//      64 x 64 C
//
// K tile:
//      16
//
// ============================================================

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 16;

constexpr int THREAD_M = 4;
constexpr int THREAD_N = 4;

constexpr int VEC_SIZE = 4;


// ============================================================
// 辅助函数:
//
// 尝试一次读取 float4
//
// 如果:
//   1. 没越界
//   2. 地址满足 float4 对齐
//
// 就走 vectorized load
//
// 否则走 scalar fallback
//
// 这样代码不仅能跑 256x256，
// M/N/K 不是 4 的倍数时也不会直接炸掉。
// ============================================================

__device__ __forceinline__
float4 loadFloat4Safe(
    const float* ptr,
    int row,
    int col,
    int rows,
    int cols,
    int ld) {

    float4 result =
        make_float4(
            0.0f,
            0.0f,
            0.0f,
            0.0f
        );


    // row 已经越界
    if (row >= rows) {
        return result;
    }


    const int index =
        row * ld + col;


    // ========================================================
    // Fast Path
    //
    // 连续 4 个 float 全部合法
    // 并且 float index 是 4 的倍数
    //
    // 一个 float = 4 bytes
    //
    // index % 4 == 0
    //
    // =>
    //
    // byte offset % 16 == 0
    //
    // ========================================================

    if (
        col + 3 < cols
        &&
        (index % 4 == 0)
    ) {

        const float4* ptr4 =
            reinterpret_cast<const float4*>(
                ptr + index
            );

        return *ptr4;
    }


    // ========================================================
    // Slow Path
    //
    // 边界情况
    //
    // 一个一个读
    //
    // ========================================================

    if (col + 0 < cols) {
        result.x =
            ptr[index + 0];
    }

    if (col + 1 < cols) {
        result.y =
            ptr[index + 1];
    }

    if (col + 2 < cols) {
        result.z =
            ptr[index + 2];
    }

    if (col + 3 < cols) {
        result.w =
            ptr[index + 3];
    }


    return result;
}


// ============================================================
// 辅助函数:
//
// 尝试 float4 写回 C
//
// 如果地址和边界允许:
//
//      4 个 C
//          ↓
//      一个 float4
//          ↓
//      Global Memory
//
// 否则 scalar fallback
// ============================================================

__device__ __forceinline__
void storeFloat4Safe(
    float* ptr,
    int row,
    int col,
    int rows,
    int cols,
    int ld,
    float v0,
    float v1,
    float v2,
    float v3) {

    if (row >= rows) {
        return;
    }


    const int index =
        row * ld + col;


    // Fast Path

    if (
        col + 3 < cols
        &&
        (index % 4 == 0)
    ) {

        float4 value =
            make_float4(
                v0,
                v1,
                v2,
                v3
            );


        float4* ptr4 =
            reinterpret_cast<float4*>(
                ptr + index
            );


        *ptr4 =
            value;


        return;
    }


    // Slow Path

    if (col + 0 < cols) {
        ptr[index + 0] = v0;
    }

    if (col + 1 < cols) {
        ptr[index + 1] = v1;
    }

    if (col + 2 < cols) {
        ptr[index + 2] = v2;
    }

    if (col + 3 < cols) {
        ptr[index + 3] = v3;
    }
}


// ============================================================
// GEMM V4
//
// C[M,N] = A[M,K] * B[K,N]
//
// 1. Shared Memory Tiling
// 2. 4x4 Register Tiling
// 3. float4 Global Load
// 4. float4 Global Store
//
// ============================================================

__global__ void gemmV4(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K) {


    // ========================================================
    // Thread Index
    //
    // block:
    //
    // 16 x 16
    //
    // tx = 0~15
    // ty = 0~15
    //
    // ========================================================

    const int tx =
        threadIdx.x;

    const int ty =
        threadIdx.y;


    // --------------------------------------------------------
    // 2D thread ID
    //
    //      ↓
    //
    // linear thread ID
    //
    // 0 ~ 255
    // --------------------------------------------------------

    const int tid =
        ty * blockDim.x
        + tx;


    // ========================================================
    // Shared Memory
    //
    // A Tile:
    //
    //      64 x 16
    //
    // B Tile:
    //
    //      16 x 64
    //
    // ========================================================

    __shared__
    float As[BLOCK_M][BLOCK_K];


    __shared__
    float Bs[BLOCK_K][BLOCK_N];


    // ========================================================
    // 当前 thread 负责 C Tile 中哪个 4x4
    //
    // ty = 0
    //
    // thread_row = 0
    //
    // ty = 1
    //
    // thread_row = 4
    //
    // ...
    //
    // ========================================================

    const int thread_row =
        ty * THREAD_M;


    const int thread_col =
        tx * THREAD_N;


    // ========================================================
    // 当前 block 对应整个 C 的左上角
    // ========================================================

    const int block_row =
        blockIdx.y * BLOCK_M;


    const int block_col =
        blockIdx.x * BLOCK_N;


    // ========================================================
    // Register Tile
    //
    // 当前 thread 最后负责:
    //
    // c00 c01 c02 c03
    // c10 c11 c12 c13
    // c20 c21 c22 c23
    // c30 c31 c32 c33
    //
    // ========================================================

    float c00 = 0.0f;
    float c01 = 0.0f;
    float c02 = 0.0f;
    float c03 = 0.0f;

    float c10 = 0.0f;
    float c11 = 0.0f;
    float c12 = 0.0f;
    float c13 = 0.0f;

    float c20 = 0.0f;
    float c21 = 0.0f;
    float c22 = 0.0f;
    float c23 = 0.0f;

    float c30 = 0.0f;
    float c31 = 0.0f;
    float c32 = 0.0f;
    float c33 = 0.0f;


    // ========================================================
    // K Tile 数量
    // ========================================================

    const int num_k_tiles =
        (K + BLOCK_K - 1)
        / BLOCK_K;


    // ========================================================
    // K Tile Loop
    // ========================================================

    for (
        int tile = 0;
        tile < num_k_tiles;
        ++tile
    ) {


        // ====================================================
        //
        //          V4 最关键的地方
        //
        // ====================================================
        //
        // As:
        //
        // 64 x 16
        //
        // = 1024 float
        //
        // 但是:
        //
        // 1024 / 4
        //
        // = 256 float4
        //
        // 恰好:
        //
        // 256 threads
        //
        // 所以:
        //
        // 每个 thread
        //
        //      ↓
        //
        // 加载一个 float4
        //
        //      ↓
        //
        // 4 个 A
        //
        // ====================================================


        // ----------------------------------------------------
        // As 一行有:
        //
        // BLOCK_K = 16 floats
        //
        // 16 / 4
        //
        // = 4 个 float4
        //
        // ----------------------------------------------------

        constexpr int A_VECS_PER_ROW =
            BLOCK_K / VEC_SIZE;


        // ----------------------------------------------------
        // tid:
        //
        // 0 ~ 255
        //
        //
        // tid / 4
        //
        // =>
        //
        // A shared row
        //
        //
        // tid % 4
        //
        // =>
        //
        // 当前 row 的第几个 float4
        //
        // ----------------------------------------------------

        const int a_smem_row =
            tid / A_VECS_PER_ROW;


        const int a_vec_col =
            tid % A_VECS_PER_ROW;


        const int a_smem_col =
            a_vec_col * VEC_SIZE;


        // ----------------------------------------------------
        // global A 坐标
        // ----------------------------------------------------

        const int a_global_row =
            block_row
            + a_smem_row;


        const int a_global_col =
            tile * BLOCK_K
            + a_smem_col;


        // ----------------------------------------------------
        // 一次读 4 个 A
        // ----------------------------------------------------

        const float4 a_vec =
            loadFloat4Safe(
                A,
                a_global_row,
                a_global_col,
                M,
                K,
                K
            );


        // ----------------------------------------------------
        // float4:
        //
        // a_vec.x
        // a_vec.y
        // a_vec.z
        // a_vec.w
        //
        //      ↓
        //
        // Shared Memory
        //
        // ----------------------------------------------------

        As[a_smem_row][a_smem_col + 0] =
            a_vec.x;

        As[a_smem_row][a_smem_col + 1] =
            a_vec.y;

        As[a_smem_row][a_smem_col + 2] =
            a_vec.z;

        As[a_smem_row][a_smem_col + 3] =
            a_vec.w;



        // ====================================================
        // B Tile
        //
        // 16 x 64
        //
        // = 1024 floats
        //
        // = 256 float4
        //
        // 同样:
        //
        // 一个 thread
        //
        //      ↓
        //
        // 一个 float4
        //
        // ====================================================


        // ----------------------------------------------------
        // Bs 每行:
        //
        // 64 float
        //
        // =
        //
        // 16 float4
        //
        // ----------------------------------------------------

        constexpr int B_VECS_PER_ROW =
            BLOCK_N / VEC_SIZE;


        const int b_smem_row =
            tid / B_VECS_PER_ROW;


        const int b_vec_col =
            tid % B_VECS_PER_ROW;


        const int b_smem_col =
            b_vec_col * VEC_SIZE;


        // ----------------------------------------------------
        // Global B 坐标
        // ----------------------------------------------------

        const int b_global_row =
            tile * BLOCK_K
            + b_smem_row;


        const int b_global_col =
            block_col
            + b_smem_col;


        // ----------------------------------------------------
        // 一次读 4 个 B
        // ----------------------------------------------------

        const float4 b_vec =
            loadFloat4Safe(
                B,
                b_global_row,
                b_global_col,
                K,
                N,
                N
            );


        // ----------------------------------------------------
        // float4
        //
        //      ↓
        //
        // Shared Memory
        // ----------------------------------------------------

        Bs[b_smem_row][b_smem_col + 0] =
            b_vec.x;

        Bs[b_smem_row][b_smem_col + 1] =
            b_vec.y;

        Bs[b_smem_row][b_smem_col + 2] =
            b_vec.z;

        Bs[b_smem_row][b_smem_col + 3] =
            b_vec.w;


        // ====================================================
        // 到这里:
        //
        // A/B Tile 已经搬进 Shared Memory
        //
        // ====================================================

        __syncthreads();


        // ====================================================
        // Compute
        //
        // 注意:
        //
        // 这一部分和 V3 基本一样
        //
        // V4 优化的是:
        //
        // Global -> Shared
        //
        // 不是改变 GEMM 数学
        //
        // ====================================================

#pragma unroll

        for (
            int k = 0;
            k < BLOCK_K;
            ++k
        ) {


            // =================================================
            // A:
            //
            // 从 Shared Memory
            //
            //      ↓
            //
            // Registers
            //
            // =================================================

            const float a0 =
                As[
                    thread_row + 0
                ][k];


            const float a1 =
                As[
                    thread_row + 1
                ][k];


            const float a2 =
                As[
                    thread_row + 2
                ][k];


            const float a3 =
                As[
                    thread_row + 3
                ][k];


            // =================================================
            // B:
            //
            // 从 Shared Memory
            //
            //      ↓
            //
            // Registers
            // =================================================

            const float b0 =
                Bs[
                    k
                ][
                    thread_col + 0
                ];


            const float b1 =
                Bs[
                    k
                ][
                    thread_col + 1
                ];


            const float b2 =
                Bs[
                    k
                ][
                    thread_col + 2
                ];


            const float b3 =
                Bs[
                    k
                ][
                    thread_col + 3
                ];


            // =================================================
            // 4x4 Outer Product
            //
            //
            //            b0    b1    b2    b3
            //
            // a0        c00   c01   c02   c03
            //
            // a1        c10   c11   c12   c13
            //
            // a2        c20   c21   c22   c23
            //
            // a3        c30   c31   c32   c33
            //
            // =================================================


            c00 += a0 * b0;
            c01 += a0 * b1;
            c02 += a0 * b2;
            c03 += a0 * b3;


            c10 += a1 * b0;
            c11 += a1 * b1;
            c12 += a1 * b2;
            c13 += a1 * b3;


            c20 += a2 * b0;
            c21 += a2 * b1;
            c22 += a2 * b2;
            c23 += a2 * b3;


            c30 += a3 * b0;
            c31 += a3 * b1;
            c32 += a3 * b2;
            c33 += a3 * b3;
        }


        // ----------------------------------------------------
        // 等整个 block 都使用完当前 tile
        //
        // 下一轮才能覆盖 As / Bs
        // ----------------------------------------------------

        __syncthreads();
    }


    // ========================================================
    // 当前 thread 最终 global C 坐标
    // ========================================================

    const int row0 =
        block_row
        + thread_row
        + 0;

    const int row1 =
        block_row
        + thread_row
        + 1;

    const int row2 =
        block_row
        + thread_row
        + 2;

    const int row3 =
        block_row
        + thread_row
        + 3;


    const int col =
        block_col
        + thread_col;


    // ========================================================
    // V4:
    //
    // Register
    //
    //     ↓
    //
    // float4
    //
    //     ↓
    //
    // Global Memory C
    //
    //
    // 每一行的:
    //
    // c00 c01 c02 c03
    //
    // 可以组成一个 float4
    //
    // ========================================================


    storeFloat4Safe(
        C,
        row0,
        col,
        M,
        N,
        N,
        c00,
        c01,
        c02,
        c03
    );


    storeFloat4Safe(
        C,
        row1,
        col,
        M,
        N,
        N,
        c10,
        c11,
        c12,
        c13
    );


    storeFloat4Safe(
        C,
        row2,
        col,
        M,
        N,
        N,
        c20,
        c21,
        c22,
        c23
    );


    storeFloat4Safe(
        C,
        row3,
        col,
        M,
        N,
        N,
        c30,
        c31,
        c32,
        c33
    );
}


// ============================================================
// CPU Reference
// ============================================================

void gemmCPU(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M,
    int N,
    int K) {

    C.resize(M * N);


    for (
        int row = 0;
        row < M;
        ++row
    ) {

        for (
            int col = 0;
            col < N;
            ++col
        ) {

            float sum =
                0.0f;


            for (
                int k = 0;
                k < K;
                ++k
            ) {

                sum +=
                    A[
                        row * K
                        + k
                    ]
                    *
                    B[
                        k * N
                        + col
                    ];
            }


            C[
                row * N
                + col
            ] =
                sum;
        }
    }
}


// ============================================================
// Main
// ============================================================

int main() {


    // ========================================================
    // 为了让 V4 fast path 100% 生效，
    //
    // 这里先使用:
    //
    // M=N=K=256
    //
    // 都是 4 的倍数
    //
    // ========================================================

    constexpr int M =
        256;

    constexpr int N =
        256;

    constexpr int K =
        256;


    // ========================================================
    // Host
    // ========================================================

    std::vector<float>
        h_A(M * K);


    std::vector<float>
        h_B(K * N);


    std::vector<float>
        h_C(M * N);


    for (
        int i = 0;
        i < M * K;
        ++i
    ) {

        h_A[i] =
            static_cast<float>(
                i % 13
            )
            / 10.0f;
    }


    for (
        int i = 0;
        i < K * N;
        ++i
    ) {

        h_B[i] =
            static_cast<float>(
                i % 7
            )
            / 10.0f;
    }


    // ========================================================
    // Device
    // ========================================================

    float* d_A =
        nullptr;

    float* d_B =
        nullptr;

    float* d_C =
        nullptr;


    CUDA_CHECK(
        cudaMalloc(
            &d_A,
            M * K * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_B,
            K * N * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_C,
            M * N * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_A,
            h_A.data(),
            M * K * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_B,
            h_B.data(),
            K * N * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    // ========================================================
    // block
    //
    // 64 / 4
    //
    // =
    //
    // 16
    //
    //
    // block:
    //
    // 16 x 16
    //
    // =
    //
    // 256 threads
    // ========================================================

    dim3 block(
        BLOCK_N / THREAD_N,
        BLOCK_M / THREAD_M
    );


    // ========================================================
    // 每个 block:
    //
    // 64 x 64 C
    // ========================================================

    dim3 grid(

        (N + BLOCK_N - 1)
        / BLOCK_N,

        (M + BLOCK_M - 1)
        / BLOCK_M
    );


    // ========================================================
    // Launch
    // ========================================================

    gemmV4<<<grid, block>>>(
        d_A,
        d_B,
        d_C,
        M,
        N,
        K
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    // ========================================================
    // GPU -> CPU
    // ========================================================

    CUDA_CHECK(
        cudaMemcpy(
            h_C.data(),
            d_C,
            M * N * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    // ========================================================
    // CPU Reference
    // ========================================================

    std::vector<float>
        cpu_C;


    gemmCPU(
        h_A,
        h_B,
        cpu_C,
        M,
        N,
        K
    );


    // ========================================================
    // Check
    // ========================================================

    float max_error =
        0.0f;


    for (
        int i = 0;
        i < M * N;
        ++i
    ) {

        const float error =
            std::fabs(
                h_C[i]
                - cpu_C[i]
            );


        max_error =
            std::max(
                max_error,
                error
            );
    }


    std::cout
        << "Block threads: "
        << block.x
        << " x "
        << block.y
        << " = "
        << block.x * block.y
        << '\n';


    std::cout
        << "C tile per block: "
        << BLOCK_M
        << " x "
        << BLOCK_N
        << '\n';


    std::cout
        << "C tile per thread: "
        << THREAD_M
        << " x "
        << THREAD_N
        << '\n';


    std::cout
        << "Vector size: float4 = "
        << sizeof(float4)
        << " bytes"
        << '\n';


    std::cout
        << "Max error: "
        << max_error
        << '\n';


    // ========================================================
    // Free
    // ========================================================

    CUDA_CHECK(
        cudaFree(d_A)
    );


    CUDA_CHECK(
        cudaFree(d_B)
    );


    CUDA_CHECK(
        cudaFree(d_C)
    );


    return 0;
}