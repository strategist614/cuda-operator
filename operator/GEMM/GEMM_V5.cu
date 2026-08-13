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
// GEMM V5
//
// V1:
//   Shared Memory Tiling
//
// V2:
//   2x2 Register Tiling
//
// V3:
//   4x4 Register Tiling
//
// V4:
//   float4 Global Load / Store
//
// V5:
//   Double Buffer
//   +
//   Software Prefetch
//
// ============================================================

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 16;

constexpr int THREAD_M = 4;
constexpr int THREAD_N = 4;

constexpr int VEC_SIZE = 4;


// ============================================================
// Safe float4 load
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


    if (row >= rows) {
        return result;
    }


    const int index =
        row * ld + col;


    // --------------------------------------------------------
    // Fast path:
    //
    // 连续 4 个 float 都合法
    // 并且 16-byte aligned
    // --------------------------------------------------------

    if (
        col + 3 < cols
        &&
        (index % 4 == 0)
    ) {

        return
            *reinterpret_cast<const float4*>(
                ptr + index
            );
    }


    // --------------------------------------------------------
    // Tail / unaligned fallback
    // --------------------------------------------------------

    if (col + 0 < cols)
        result.x = ptr[index + 0];

    if (col + 1 < cols)
        result.y = ptr[index + 1];

    if (col + 2 < cols)
        result.z = ptr[index + 2];

    if (col + 3 < cols)
        result.w = ptr[index + 3];


    return result;
}


// ============================================================
// Safe float4 store
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


    if (
        col + 3 < cols
        &&
        (index % 4 == 0)
    ) {

        const float4 value =
            make_float4(
                v0,
                v1,
                v2,
                v3
            );


        *reinterpret_cast<float4*>(
            ptr + index
        ) = value;


        return;
    }


    if (col + 0 < cols)
        ptr[index + 0] = v0;

    if (col + 1 < cols)
        ptr[index + 1] = v1;

    if (col + 2 < cols)
        ptr[index + 2] = v2;

    if (col + 3 < cols)
        ptr[index + 3] = v3;
}


// ============================================================
// GEMM V5 Kernel
// ============================================================

__global__
void gemmV5(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K) {


    // ========================================================
    // Thread information
    // ========================================================

    const int tx =
        threadIdx.x;

    const int ty =
        threadIdx.y;


    const int tid =
        ty * blockDim.x
        + tx;


    // ========================================================
    // IMPORTANT:
    //
    // V4:
    //
    // As[64][16]
    // Bs[16][64]
    //
    //
    // V5:
    //
    // 两套 Shared Memory
    //
    // buffer 0
    // buffer 1
    //
    // ========================================================

    __shared__
    float As[2][BLOCK_M][BLOCK_K];


    __shared__
    float Bs[2][BLOCK_K][BLOCK_N];


    // ========================================================
    // 当前 thread 负责 4x4 C
    // ========================================================

    const int thread_row =
        ty * THREAD_M;


    const int thread_col =
        tx * THREAD_N;


    // ========================================================
    // 当前 block 在 C 中的位置
    // ========================================================

    const int block_row =
        blockIdx.y * BLOCK_M;


    const int block_col =
        blockIdx.x * BLOCK_N;


    // ========================================================
    // Register Tile
    //
    // 16 个 accumulator
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
    // vector mapping
    // ========================================================

    constexpr int A_VECS_PER_ROW =
        BLOCK_K / VEC_SIZE;      // 16 / 4 = 4


    constexpr int B_VECS_PER_ROW =
        BLOCK_N / VEC_SIZE;      // 64 / 4 = 16


    // --------------------------------------------------------
    // 每个 thread 负责 As 中一个 float4
    // --------------------------------------------------------

    const int a_smem_row =
        tid / A_VECS_PER_ROW;


    const int a_smem_col =
        (tid % A_VECS_PER_ROW)
        * VEC_SIZE;


    // --------------------------------------------------------
    // 每个 thread 负责 Bs 中一个 float4
    // --------------------------------------------------------

    const int b_smem_row =
        tid / B_VECS_PER_ROW;


    const int b_smem_col =
        (tid % B_VECS_PER_ROW)
        * VEC_SIZE;


    // ========================================================
    // K tile 数
    // ========================================================

    const int num_tiles =
        (K + BLOCK_K - 1)
        / BLOCK_K;


    // ========================================================
    // 第一步:
    //
    // 先加载 tile 0
    //
    // 没有 tile 0 就没东西可以计算
    // ========================================================

    {
        const int a_global_row =
            block_row
            + a_smem_row;


        const int a_global_col =
            a_smem_col;


        const float4 a_vec =
            loadFloat4Safe(
                A,
                a_global_row,
                a_global_col,
                M,
                K,
                K
            );


        As[0][a_smem_row][a_smem_col + 0] =
            a_vec.x;

        As[0][a_smem_row][a_smem_col + 1] =
            a_vec.y;

        As[0][a_smem_row][a_smem_col + 2] =
            a_vec.z;

        As[0][a_smem_row][a_smem_col + 3] =
            a_vec.w;



        const int b_global_row =
            b_smem_row;


        const int b_global_col =
            block_col
            + b_smem_col;


        const float4 b_vec =
            loadFloat4Safe(
                B,
                b_global_row,
                b_global_col,
                K,
                N,
                N
            );


        Bs[0][b_smem_row][b_smem_col + 0] =
            b_vec.x;

        Bs[0][b_smem_row][b_smem_col + 1] =
            b_vec.y;

        Bs[0][b_smem_row][b_smem_col + 2] =
            b_vec.z;

        Bs[0][b_smem_row][b_smem_col + 3] =
            b_vec.w;
    }


    // --------------------------------------------------------
    // tile 0 已经全部准备好
    // --------------------------------------------------------

    __syncthreads();


    // ========================================================
    // current_buffer:
    //
    // 当前正在拿来计算的 shared buffer
    //
    // 一开始是 buffer 0
    // ========================================================

    int current_buffer =
        0;


    // ========================================================
    // MAIN LOOP
    // ========================================================

    for (
        int tile = 0;
        tile < num_tiles;
        ++tile
    ) {


        // ====================================================
        // 下一个 buffer
        //
        // 0 -> 1
        //
        // 1 -> 0
        //
        // ====================================================

        const int next_buffer =
            current_buffer ^ 1;


        const bool has_next =
            (tile + 1)
            < num_tiles;


        // ====================================================
        // Prefetch Registers
        //
        // 重点!!!
        //
        // 下一块 tile
        //
        // Global Memory
        //
        //       ↓
        //
        // Registers
        //
        //
        // 注意:
        //
        // 现在先不写 Shared Memory
        //
        // ====================================================

        float4 next_a =
            make_float4(
                0.0f,
                0.0f,
                0.0f,
                0.0f
            );


        float4 next_b =
            make_float4(
                0.0f,
                0.0f,
                0.0f,
                0.0f
            );


        if (has_next) {


            // ------------------------------------------------
            // 下一块 A
            // ------------------------------------------------

            const int next_a_global_row =
                block_row
                + a_smem_row;


            const int next_a_global_col =
                (tile + 1)
                * BLOCK_K
                + a_smem_col;


            next_a =
                loadFloat4Safe(
                    A,
                    next_a_global_row,
                    next_a_global_col,
                    M,
                    K,
                    K
                );


            // ------------------------------------------------
            // 下一块 B
            // ------------------------------------------------

            const int next_b_global_row =
                (tile + 1)
                * BLOCK_K
                + b_smem_row;


            const int next_b_global_col =
                block_col
                + b_smem_col;


            next_b =
                loadFloat4Safe(
                    B,
                    next_b_global_row,
                    next_b_global_col,
                    K,
                    N,
                    N
                );
        }


        // ====================================================
        //
        // COMPUTE CURRENT TILE
        //
        // ====================================================
        //
        // 此时:
        //
        // 当前 tile:
        //
        // Shared Memory
        //      ↓
        // 正在计算
        //
        //
        // 下一 tile:
        //
        // Global Memory
        //      ↓
        // next_a / next_b registers
        //
        // ====================================================


#pragma unroll

        for (
            int k = 0;
            k < BLOCK_K;
            ++k
        ) {


            // =================================================
            // Shared -> Register
            //
            // A 4 values
            // =================================================

            const float a0 =
                As[
                    current_buffer
                ][
                    thread_row + 0
                ][
                    k
                ];


            const float a1 =
                As[
                    current_buffer
                ][
                    thread_row + 1
                ][
                    k
                ];


            const float a2 =
                As[
                    current_buffer
                ][
                    thread_row + 2
                ][
                    k
                ];


            const float a3 =
                As[
                    current_buffer
                ][
                    thread_row + 3
                ][
                    k
                ];


            // =================================================
            // Shared -> Register
            //
            // B 4 values
            // =================================================

            const float b0 =
                Bs[
                    current_buffer
                ][
                    k
                ][
                    thread_col + 0
                ];


            const float b1 =
                Bs[
                    current_buffer
                ][
                    k
                ][
                    thread_col + 1
                ];


            const float b2 =
                Bs[
                    current_buffer
                ][
                    k
                ][
                    thread_col + 2
                ];


            const float b3 =
                Bs[
                    current_buffer
                ][
                    k
                ][
                    thread_col + 3
                ];


            // =================================================
            // 4x4 outer product
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


        // ====================================================
        // 当前 tile 计算完成
        //
        // 如果还有下一 tile:
        //
        // 把刚才预取到 Register 的数据
        //
        //        ↓
        //
        // 写入另一个 Shared Buffer
        // ====================================================

        if (has_next) {


            // ------------------------------------------------
            // next A register
            //
            //      ↓
            //
            // inactive As buffer
            // ------------------------------------------------

            As[
                next_buffer
            ][
                a_smem_row
            ][
                a_smem_col + 0
            ] = next_a.x;


            As[
                next_buffer
            ][
                a_smem_row
            ][
                a_smem_col + 1
            ] = next_a.y;


            As[
                next_buffer
            ][
                a_smem_row
            ][
                a_smem_col + 2
            ] = next_a.z;


            As[
                next_buffer
            ][
                a_smem_row
            ][
                a_smem_col + 3
            ] = next_a.w;



            // ------------------------------------------------
            // next B register
            //
            //      ↓
            //
            // inactive Bs buffer
            // ------------------------------------------------

            Bs[
                next_buffer
            ][
                b_smem_row
            ][
                b_smem_col + 0
            ] = next_b.x;


            Bs[
                next_buffer
            ][
                b_smem_row
            ][
                b_smem_col + 1
            ] = next_b.y;


            Bs[
                next_buffer
            ][
                b_smem_row
            ][
                b_smem_col + 2
            ] = next_b.z;


            Bs[
                next_buffer
            ][
                b_smem_row
            ][
                b_smem_col + 3
            ] = next_b.w;


            // =================================================
            // 所有 thread 都必须把下一块 Shared Tile
            // 写完
            //
            // 下一轮才能读取
            // =================================================

            __syncthreads();


            // =================================================
            // Ping-Pong
            //
            // 当前 buffer:
            //
            // 0 -> 1
            //
            // 或:
            //
            // 1 -> 0
            // =================================================

            current_buffer =
                next_buffer;
        }
    }


    // ========================================================
    // Store C
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
    // Register -> float4 -> Global
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


    for (int row = 0;
         row < M;
         ++row) {

        for (int col = 0;
             col < N;
             ++col) {

            float sum =
                0.0f;


            for (int k = 0;
                 k < K;
                 ++k) {

                sum +=
                    A[row * K + k]
                    *
                    B[k * N + col];
            }


            C[row * N + col] =
                sum;
        }
    }
}


// ============================================================
// Main
// ============================================================

int main() {

    // --------------------------------------------------------
    // 先用比较规整的尺寸进行 benchmark
    // --------------------------------------------------------

    constexpr int M = 512;
    constexpr int N = 512;
    constexpr int K = 512;


    std::vector<float>
        h_A(M * K);

    std::vector<float>
        h_B(K * N);

    std::vector<float>
        h_C(M * N);


    for (int i = 0;
         i < M * K;
         ++i) {

        h_A[i] =
            static_cast<float>(
                i % 13
            )
            / 10.0f;
    }


    for (int i = 0;
         i < K * N;
         ++i) {

        h_B[i] =
            static_cast<float>(
                i % 7
            )
            / 10.0f;
    }


    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;


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
    // 16 x 16 = 256 threads
    // ========================================================

    dim3 block(
        BLOCK_N / THREAD_N,
        BLOCK_M / THREAD_M
    );


    // ========================================================
    // 每 block 负责 64 x 64 C
    // ========================================================

    dim3 grid(
        (N + BLOCK_N - 1)
            / BLOCK_N,

        (M + BLOCK_M - 1)
            / BLOCK_M
    );


    // ========================================================
    // Warmup
    // ========================================================

    for (int i = 0;
         i < 10;
         ++i) {

        gemmV5<<<grid, block>>>(
            d_A,
            d_B,
            d_C,
            M,
            N,
            K
        );
    }


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    // ========================================================
    // CUDA Event Benchmark
    // ========================================================

    cudaEvent_t start;
    cudaEvent_t stop;


    CUDA_CHECK(
        cudaEventCreate(&start)
    );


    CUDA_CHECK(
        cudaEventCreate(&stop)
    );


    constexpr int ITERATIONS =
        100;


    CUDA_CHECK(
        cudaEventRecord(start)
    );


    for (int i = 0;
         i < ITERATIONS;
         ++i) {

        gemmV5<<<grid, block>>>(
            d_A,
            d_B,
            d_C,
            M,
            N,
            K
        );
    }


    CUDA_CHECK(
        cudaEventRecord(stop)
    );


    CUDA_CHECK(
        cudaEventSynchronize(stop)
    );


    float total_ms =
        0.0f;


    CUDA_CHECK(
        cudaEventElapsedTime(
            &total_ms,
            start,
            stop
        )
    );


    const float avg_ms =
        total_ms
        / ITERATIONS;


    // GEMM:
    //
    // 一个 multiply
    // +
    // 一个 add
    //
    // ≈ 2 FLOPs

    const double flops =
        2.0
        * static_cast<double>(M)
        * static_cast<double>(N)
        * static_cast<double>(K);


    const double gflops =
        flops
        /
        (avg_ms * 1.0e6);


    // ========================================================
    // Copy result
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
    // CPU reference
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


    for (int i = 0;
         i < M * N;
         ++i) {

        max_error =
            std::max(
                max_error,
                std::fabs(
                    h_C[i]
                    - cpu_C[i]
                )
            );
    }


    std::cout
        << "Block: "
        << block.x
        << " x "
        << block.y
        << " = "
        << block.x * block.y
        << " threads\n";


    std::cout
        << "Block C tile: "
        << BLOCK_M
        << " x "
        << BLOCK_N
        << '\n';


    std::cout
        << "Thread C tile: "
        << THREAD_M
        << " x "
        << THREAD_N
        << '\n';


    std::cout
        << "Average time: "
        << avg_ms
        << " ms\n";


    std::cout
        << "Performance: "
        << gflops
        << " GFLOPS\n";


    std::cout
        << "Max error: "
        << max_error
        << '\n';


    CUDA_CHECK(
        cudaEventDestroy(start)
    );


    CUDA_CHECK(
        cudaEventDestroy(stop)
    );


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