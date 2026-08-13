#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>


// ============================================================
// CUDA Error Check
// ============================================================

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr                                         \
                << "CUDA Error: "                             \
                << cudaGetErrorString(err)                    \
                << " at "                                     \
                << __FILE__                                   \
                << ":"                                        \
                << __LINE__                                   \
                << std::endl;                                 \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)


// ============================================================
// V7 Configuration
// ============================================================
//
// Block Tile:
//
//      128 x 64
//
// K Tile:
//
//      16
//
// 一个 Block:
//
//      8 warps
//      256 threads
//
// Warp 排布:
//
//      4 x 2
//
// 每个 Warp Tile:
//
//      32 x 32
//
// 每个 Thread/Lane:
//
//      8 x 4
//
// ============================================================

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 16;


// ============================================================
// Warp Tile
// ============================================================

constexpr int WARP_M = 32;
constexpr int WARP_N = 32;


// ============================================================
// Warp arrangement inside one block
//
// 4 rows
// 2 cols
//
// Warp0 Warp1
// Warp2 Warp3
// Warp4 Warp5
// Warp6 Warp7
// ============================================================

constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;

constexpr int NUM_WARPS =
    WARPS_M * WARPS_N;


// ============================================================
// Thread Tile
//
// 每个 thread:
//
//      8 x 4
//
// =
//
//      32 outputs
// ============================================================

constexpr int THREAD_M = 8;
constexpr int THREAD_N = 4;


// ============================================================
// 32 threads / warp
// ============================================================

constexpr int WARP_SIZE = 32;


// ============================================================
// 一个 Block:
//
// 8 warps * 32
//
// =
//
// 256 threads
// ============================================================

constexpr int THREADS_PER_BLOCK =
    NUM_WARPS * WARP_SIZE;


// ============================================================
// Vector Load
// ============================================================

constexpr int VEC_SIZE = 4;


// ============================================================
// safe float4 load
// ============================================================

__device__ __forceinline__
float4 loadFloat4Safe(
    const float* ptr,
    int row,
    int col,
    int rows,
    int cols,
    int ld)
{
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


    // ========================================================
    // fast path
    // ========================================================

    if (
        col + 3 < cols
        &&
        index % 4 == 0
    ) {

        return
            *reinterpret_cast<const float4*>(
                ptr + index
            );
    }


    // ========================================================
    // boundary / unaligned fallback
    // ========================================================

    if (col + 0 < cols) {
        result.x = ptr[index + 0];
    }

    if (col + 1 < cols) {
        result.y = ptr[index + 1];
    }

    if (col + 2 < cols) {
        result.z = ptr[index + 2];
    }

    if (col + 3 < cols) {
        result.w = ptr[index + 3];
    }

    return result;
}


// ============================================================
// safe float4 store
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
    float v3)
{
    if (row >= rows) {
        return;
    }

    const int index =
        row * ld + col;


    if (
        col + 3 < cols
        &&
        index % 4 == 0
    ) {

        *reinterpret_cast<float4*>(
            ptr + index
        ) =
            make_float4(
                v0,
                v1,
                v2,
                v3
            );

        return;
    }


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
// GEMM V7
//
// C = A * B
//
// A: M x K
// B: K x N
// C: M x N
//
// ============================================================

__global__
void gemmV7WarpTiling(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K)
{
    // ========================================================
    // 当前 thread ID
    //
    // 0 ~ 255
    // ========================================================

    const int tid =
        threadIdx.x;


    // ========================================================
    // 第一个关键:
    //
    // thread
    //
    //       ↓
    //
    // warp
    //
    // ========================================================

    const int warp_id =
        tid / WARP_SIZE;


    // ========================================================
    // 当前 thread 是 warp 里面第几个 lane
    //
    // 0 ~ 31
    // ========================================================

    const int lane_id =
        tid % WARP_SIZE;


    // ========================================================
    // Shared Memory
    //
    // A:
    //
    // 128 x 16
    //
    // B:
    //
    // 16 x 64
    //
    // ========================================================

    __shared__ __align__(16)
    float As[BLOCK_M][BLOCK_K];


    __shared__ __align__(16)
    float Bs[BLOCK_K][BLOCK_N];


    // ========================================================
    // Block 在整个 C Matrix 的位置
    // ========================================================

    const int block_row =
        blockIdx.y * BLOCK_M;


    const int block_col =
        blockIdx.x * BLOCK_N;


    // ========================================================
    //
    //             WARP TILING
    //
    // ========================================================
    //
    // 8 warps 排成:
    //
    //
    //       warp_n →
    //
    //      0       1
    //
    // 0   Warp0   Warp1
    //
    // 1   Warp2   Warp3
    //
    // 2   Warp4   Warp5
    //
    // 3   Warp6   Warp7
    //
    //
    // ========================================================

    const int warp_m =
        warp_id / WARPS_N;


    const int warp_n =
        warp_id % WARPS_N;


    // ========================================================
    // 当前 Warp Tile 左上角
    //
    // 在 Block Tile 内的坐标
    //
    // ========================================================

    const int warp_row =
        warp_m * WARP_M;


    const int warp_col =
        warp_n * WARP_N;


    // ========================================================
    //
    //                LANE TILING
    //
    // ========================================================
    //
    // 一个 warp 有 32 lanes。
    //
    // 我们把它看成:
    //
    //      4 x 8
    //
    //
    // lane:
    //
    //  0  1  2  3  4  5  6  7
    //
    //  8  9 10 11 12 13 14 15
    //
    // 16 17 18 19 20 21 22 23
    //
    // 24 25 26 27 28 29 30 31
    //
    //
    // 每个 lane:
    //
    //      8 rows
    //
    //      4 cols
    //
    // ========================================================

    constexpr int LANES_N =
        WARP_N / THREAD_N;

    // 32 / 4 = 8


    const int lane_m =
        lane_id / LANES_N;


    const int lane_n =
        lane_id % LANES_N;


    // ========================================================
    // 当前 thread 的 8x4 Tile
    //
    // 在 block tile 内的左上角
    // ========================================================

    const int thread_row =
        warp_row
        +
        lane_m * THREAD_M;


    const int thread_col =
        warp_col
        +
        lane_n * THREAD_N;


    // ========================================================
    // Register Tile
    //
    // 每个 thread:
    //
    // 8 x 4
    //
    // =
    //
    // 32 accumulators
    //
    //
    // acc:
    //
    // row0: 4
    // row1: 4
    // ...
    // row7: 4
    //
    // ========================================================

    float acc[THREAD_M][THREAD_N];


#pragma unroll
    for (int i = 0;
         i < THREAD_M;
         ++i)
    {

#pragma unroll
        for (int j = 0;
             j < THREAD_N;
             ++j)
        {
            acc[i][j] =
                0.0f;
        }
    }


    // ========================================================
    // K direction tiles
    // ========================================================

    const int num_k_tiles =
        (K + BLOCK_K - 1)
        /
        BLOCK_K;


    // ========================================================
    // Cooperative Load Mapping
    // ========================================================
    //
    // A Tile:
    //
    // 128 x 16
    //
    // =
    //
    // 2048 float
    //
    // =
    //
    // 512 float4
    //
    //
    // 256 threads
    //
    // =>
    //
    // 每 thread 搬 2 个 float4
    //
    //
    // B Tile:
    //
    // 16 x 64
    //
    // =
    //
    // 1024 float
    //
    // =
    //
    // 256 float4
    //
    //
    // =>
    //
    // 每 thread 搬 1 个 float4
    //
    // ========================================================

    constexpr int A_VECS_PER_ROW =
        BLOCK_K / VEC_SIZE;

    // 16 / 4 = 4


    constexpr int A_TOTAL_VECS =
        BLOCK_M * A_VECS_PER_ROW;

    // 128 * 4 = 512


    constexpr int B_VECS_PER_ROW =
        BLOCK_N / VEC_SIZE;

    // 64 / 4 = 16


    constexpr int B_TOTAL_VECS =
        BLOCK_K * B_VECS_PER_ROW;

    // 16 * 16 = 256


    // ========================================================
    // K Tile Main Loop
    // ========================================================

    for (
        int tile = 0;
        tile < num_k_tiles;
        ++tile
    ) {

        // ====================================================
        // Step 1:
        //
        // Global A
        //
        //      ↓
        //
        // Shared A
        //
        // ====================================================

        for (
            int vec_index = tid;
            vec_index < A_TOTAL_VECS;
            vec_index += THREADS_PER_BLOCK
        ) {

            // ------------------------------------------------
            // A shared row
            // ------------------------------------------------

            const int smem_row =
                vec_index
                /
                A_VECS_PER_ROW;


            // ------------------------------------------------
            // 当前 row 中第几个 float4
            // ------------------------------------------------

            const int vec_col =
                vec_index
                %
                A_VECS_PER_ROW;


            const int smem_col =
                vec_col
                *
                VEC_SIZE;


            // ------------------------------------------------
            // Global A coordinate
            // ------------------------------------------------

            const int global_row =
                block_row
                +
                smem_row;


            const int global_col =
                tile * BLOCK_K
                +
                smem_col;


            const float4 value =
                loadFloat4Safe(
                    A,
                    global_row,
                    global_col,
                    M,
                    K,
                    K
                );


            As[smem_row][smem_col + 0] =
                value.x;

            As[smem_row][smem_col + 1] =
                value.y;

            As[smem_row][smem_col + 2] =
                value.z;

            As[smem_row][smem_col + 3] =
                value.w;
        }


        // ====================================================
        // Step 2:
        //
        // Global B
        //
        //      ↓
        //
        // Shared B
        //
        // ====================================================

        for (
            int vec_index = tid;
            vec_index < B_TOTAL_VECS;
            vec_index += THREADS_PER_BLOCK
        ) {

            const int smem_row =
                vec_index
                /
                B_VECS_PER_ROW;


            const int vec_col =
                vec_index
                %
                B_VECS_PER_ROW;


            const int smem_col =
                vec_col
                *
                VEC_SIZE;


            // ------------------------------------------------
            // Global B coordinate
            // ------------------------------------------------

            const int global_row =
                tile * BLOCK_K
                +
                smem_row;


            const int global_col =
                block_col
                +
                smem_col;


            const float4 value =
                loadFloat4Safe(
                    B,
                    global_row,
                    global_col,
                    K,
                    N,
                    N
                );


            Bs[smem_row][smem_col + 0] =
                value.x;

            Bs[smem_row][smem_col + 1] =
                value.y;

            Bs[smem_row][smem_col + 2] =
                value.z;

            Bs[smem_row][smem_col + 3] =
                value.w;
        }


        // ====================================================
        // 整个 Block 搬完 Shared Tile
        // ====================================================

        __syncthreads();


        // ====================================================
        //
        //                COMPUTE
        //
        // ====================================================

#pragma unroll
        for (
            int k = 0;
            k < BLOCK_K;
            ++k
        ) {

            // =================================================
            // 一个 thread 从 Shared A
            //
            // 读取 8 个 A
            //
            // =================================================

            float a_frag[THREAD_M];


#pragma unroll
            for (
                int i = 0;
                i < THREAD_M;
                ++i
            ) {

                a_frag[i] =
                    As[
                        thread_row + i
                    ][
                        k
                    ];
            }


            // =================================================
            // 从 Shared B
            //
            // 读取 4 个 B
            // =================================================

            float b_frag[THREAD_N];


#pragma unroll
            for (
                int j = 0;
                j < THREAD_N;
                ++j
            ) {

                b_frag[j] =
                    Bs[
                        k
                    ][
                        thread_col + j
                    ];
            }


            // =================================================
            //
            // Register Outer Product
            //
            //
            //            b0 b1 b2 b3
            //
            //      a0     x  x  x  x
            //
            //      a1     x  x  x  x
            //
            //      a2     x  x  x  x
            //
            //      a3     x  x  x  x
            //
            //      a4     x  x  x  x
            //
            //      a5     x  x  x  x
            //
            //      a6     x  x  x  x
            //
            //      a7     x  x  x  x
            //
            //
            // =
            //
            // 8 * 4
            //
            // =
            //
            // 32 FMA / k / thread
            //
            // =================================================

#pragma unroll
            for (
                int i = 0;
                i < THREAD_M;
                ++i
            ) {

#pragma unroll
                for (
                    int j = 0;
                    j < THREAD_N;
                    ++j
                ) {

                    acc[i][j]
                        +=
                        a_frag[i]
                        *
                        b_frag[j];
                }
            }
        }


        // ====================================================
        // 所有 thread 使用完当前 Shared Tile
        //
        // 下一 tile 才能覆盖
        // ====================================================

        __syncthreads();
    }


    // ========================================================
    //
    // Register -> Global C
    //
    // ========================================================

#pragma unroll
    for (
        int i = 0;
        i < THREAD_M;
        ++i
    ) {

        const int global_row =
            block_row
            +
            thread_row
            +
            i;


        const int global_col =
            block_col
            +
            thread_col;


        // ----------------------------------------------------
        // 当前 thread 每一行刚好有 4 个连续输出
        //
        // 所以 float4 store
        // ----------------------------------------------------

        storeFloat4Safe(
            C,
            global_row,
            global_col,
            M,
            N,
            N,

            acc[i][0],
            acc[i][1],
            acc[i][2],
            acc[i][3]
        );
    }
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
    int K)
{
    C.resize(
        M * N
    );


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
                        +
                        k
                    ]
                    *
                    B[
                        k * N
                        +
                        col
                    ];
            }


            C[
                row * N
                +
                col
            ] =
                sum;
        }
    }
}


// ============================================================
// Main
// ============================================================

int main()
{
    // ========================================================
    // Matrix size
    // ========================================================

    constexpr int M = 512;
    constexpr int N = 512;
    constexpr int K = 512;


    // ========================================================
    // Host memory
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
            /
            10.0f;
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
            /
            10.0f;
    }


    // ========================================================
    // Device memory
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
    // Block:
    //
    // 256 threads
    //
    // =
    //
    // 8 warps
    // ========================================================

    dim3 block(
        THREADS_PER_BLOCK
    );


    // ========================================================
    // 每个 block:
    //
    // 128 x 64 C
    // ========================================================

    dim3 grid(
        (N + BLOCK_N - 1)
            /
            BLOCK_N,

        (M + BLOCK_M - 1)
            /
            BLOCK_M
    );


    // ========================================================
    // Warmup
    // ========================================================

    for (
        int i = 0;
        i < 10;
        ++i
    ) {

        gemmV7WarpTiling<<<grid, block>>>(
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


    CUDA_CHECK(
        cudaGetLastError()
    );


    // ========================================================
    // Benchmark
    // ========================================================

    cudaEvent_t start;
    cudaEvent_t stop;


    CUDA_CHECK(
        cudaEventCreate(
            &start
        )
    );


    CUDA_CHECK(
        cudaEventCreate(
            &stop
        )
    );


    constexpr int ITERATIONS =
        100;


    CUDA_CHECK(
        cudaEventRecord(
            start
        )
    );


    for (
        int i = 0;
        i < ITERATIONS;
        ++i
    ) {

        gemmV7WarpTiling<<<grid, block>>>(
            d_A,
            d_B,
            d_C,
            M,
            N,
            K
        );
    }


    CUDA_CHECK(
        cudaEventRecord(
            stop
        )
    );


    CUDA_CHECK(
        cudaEventSynchronize(
            stop
        )
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
        /
        ITERATIONS;


    // ========================================================
    // GFLOPS
    // ========================================================

    const double flops =
        2.0
        *
        static_cast<double>(M)
        *
        static_cast<double>(N)
        *
        static_cast<double>(K);


    const double gflops =
        flops
        /
        (
            avg_ms
            *
            1.0e6
        );


    // ========================================================
    // Copy C
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
    // Verify
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
                -
                cpu_C[i]
            );


        max_error =
            std::max(
                max_error,
                error
            );
    }


    // ========================================================
    // Print
    // ========================================================

    std::cout
        << "Block tile: "
        << BLOCK_M
        << " x "
        << BLOCK_N
        << '\n';


    std::cout
        << "Warp tile: "
        << WARP_M
        << " x "
        << WARP_N
        << '\n';


    std::cout
        << "Thread tile: "
        << THREAD_M
        << " x "
        << THREAD_N
        << '\n';


    std::cout
        << "Warps per block: "
        << NUM_WARPS
        << '\n';


    std::cout
        << "Threads per block: "
        << THREADS_PER_BLOCK
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


    // ========================================================
    // Cleanup
    // ========================================================

    CUDA_CHECK(
        cudaEventDestroy(
            start
        )
    );


    CUDA_CHECK(
        cudaEventDestroy(
            stop
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_A
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_B
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_C
        )
    );


    return 0;
}