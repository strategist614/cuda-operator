// GEMM_V9_SharedTranspose.cu

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
// Block Tile
//
// 一个 block 负责 C:
//
//      128 x 64
//
// K 每次处理:
//
//      16
// ============================================================

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 16;


// ============================================================
// Warp Tile
//
// 一个 warp:
//
//      32 x 32
// ============================================================

constexpr int WARP_M = 32;
constexpr int WARP_N = 32;


// ============================================================
// 一个 block:
//
//      4 x 2 warps
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
// 一个 thread:
//
//      8 x 4
//
// = 32 个 C
// ============================================================

constexpr int THREAD_M = 8;
constexpr int THREAD_N = 4;


constexpr int WARP_SIZE = 32;

constexpr int THREADS_PER_BLOCK =
    NUM_WARPS * WARP_SIZE;


// ============================================================
// Vector width
// ============================================================

constexpr int VEC_SIZE = 4;


// ============================================================
// V9:
//
// Shared A 转置
//
// logical:
//
//      A tile = [128][16]
//
// V8:
//
//      As[128][17]
//
// V9:
//
//      AsT[16][129]
//
// 注意 129 = 128 + 1
//
// +1 仍然是 padding
// ============================================================

constexpr int A_TRANSPOSE_STRIDE =
    BLOCK_M + 1;


// ============================================================
// float4 safe load
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

    if (
        col + 3 < cols &&
        (index % 4 == 0)
    ) {
        return
            *reinterpret_cast<const float4*>(
                ptr + index
            );
    }

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
// float4 safe store
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
        col + 3 < cols &&
        (index % 4 == 0)
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
// GEMM V9
// ============================================================

__global__
void gemmV9SharedTranspose(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K)
{
    const int tid =
        threadIdx.x;

    // ========================================================
    // Warp ID + Lane ID
    // ========================================================

    const int warp_id =
        tid / WARP_SIZE;

    const int lane_id =
        tid % WARP_SIZE;


    // ========================================================
    // V9 Shared Memory
    //
    // A:
    //
    // 原来逻辑:
    //
    //      [M][K]
    //
    // Shared Memory 中:
    //
    //      [K][M]
    //
    // ========================================================

    __shared__ __align__(16)
    float AsT[
        BLOCK_K
    ][
        A_TRANSPOSE_STRIDE
    ];


    // ========================================================
    // B 保持原 layout
    // ========================================================

    __shared__ __align__(16)
    float Bs[
        BLOCK_K
    ][
        BLOCK_N
    ];


    // ========================================================
    // Block C tile
    // ========================================================

    const int block_row =
        blockIdx.y * BLOCK_M;

    const int block_col =
        blockIdx.x * BLOCK_N;


    // ========================================================
    // Warp 坐标
    // ========================================================

    const int warp_m =
        warp_id / WARPS_N;

    const int warp_n =
        warp_id % WARPS_N;


    const int warp_row =
        warp_m * WARP_M;

    const int warp_col =
        warp_n * WARP_N;


    // ========================================================
    // Warp 内 lane:
    //
    // 4 x 8
    //
    // 一个 lane:
    //
    // 8 x 4
    // ========================================================

    constexpr int LANES_N =
        WARP_N / THREAD_N;

    // 8


    const int lane_m =
        lane_id / LANES_N;

    const int lane_n =
        lane_id % LANES_N;


    // ========================================================
    // 当前 thread tile 左上角
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
    // Register accumulator
    //
    // 8 x 4
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
    // K Tile count
    // ========================================================

    const int num_k_tiles =
        (K + BLOCK_K - 1)
        /
        BLOCK_K;


    // ========================================================
    // A:
    //
    // 128 x 16
    //
    // 2048 floats
    //
    // = 512 float4
    //
    // 256 threads
    //
    // 每个 thread 平均两个 float4
    // ========================================================

    constexpr int A_VECS_PER_ROW =
        BLOCK_K / VEC_SIZE;

    // 4


    constexpr int A_TOTAL_VECS =
        BLOCK_M * A_VECS_PER_ROW;

    // 512


    // ========================================================
    // B:
    //
    // 16 x 64
    //
    // 1024 floats
    //
    // = 256 float4
    //
    // 每 thread 一个
    // ========================================================

    constexpr int B_VECS_PER_ROW =
        BLOCK_N / VEC_SIZE;

    // 16


    constexpr int B_TOTAL_VECS =
        BLOCK_K * B_VECS_PER_ROW;

    // 256


    // ========================================================
    // K tile loop
    // ========================================================

    for (
        int tile = 0;
        tile < num_k_tiles;
        ++tile
    ) {

        // ====================================================
        // STEP 1
        //
        // Global A
        //
        //      ↓
        //
        // Registers(float4)
        //
        //      ↓
        //
        // Transposed Shared A
        // ====================================================

        for (
            int vec_index = tid;
            vec_index < A_TOTAL_VECS;
            vec_index += THREADS_PER_BLOCK
        ) {

            // =================================================
            // Logical A tile row
            //
            // 0 ~ 127
            // =================================================

            const int a_row =
                vec_index /
                A_VECS_PER_ROW;


            // =================================================
            // 当前 row 的第几个 float4
            //
            // 0 1 2 3
            // =================================================

            const int a_vec =
                vec_index %
                A_VECS_PER_ROW;


            // =================================================
            // logical K position
            //
            // 0 / 4 / 8 / 12
            // =================================================

            const int a_k =
                a_vec * VEC_SIZE;


            // =================================================
            // Global A coordinate
            // =================================================

            const int global_row =
                block_row + a_row;

            const int global_col =
                tile * BLOCK_K
                +
                a_k;


            // =================================================
            // Global contiguous float4 load
            //
            // Global Memory 依然按照原本 row-major
            // 连续读取
            // =================================================

            const float4 value =
                loadFloat4Safe(
                    A,
                    global_row,
                    global_col,
                    M,
                    K,
                    K
                );


            // =================================================
            // 关键变化!!!
            //
            // V8:
            //
            // As[a_row][a_k]
            //
            //
            // V9:
            //
            // AsT[a_k][a_row]
            //
            //
            // 转置写入 Shared Memory
            // =================================================

            AsT[a_k + 0][a_row] =
                value.x;

            AsT[a_k + 1][a_row] =
                value.y;

            AsT[a_k + 2][a_row] =
                value.z;

            AsT[a_k + 3][a_row] =
                value.w;
        }


        // ====================================================
        // STEP 2
        //
        // Global B -> Shared B
        // ====================================================

        for (
            int vec_index = tid;
            vec_index < B_TOTAL_VECS;
            vec_index += THREADS_PER_BLOCK
        ) {

            const int b_row =
                vec_index /
                B_VECS_PER_ROW;


            const int b_vec =
                vec_index %
                B_VECS_PER_ROW;


            const int b_col =
                b_vec *
                VEC_SIZE;


            const int global_row =
                tile * BLOCK_K
                +
                b_row;


            const int global_col =
                block_col
                +
                b_col;


            const float4 value =
                loadFloat4Safe(
                    B,
                    global_row,
                    global_col,
                    K,
                    N,
                    N
                );


            Bs[b_row][b_col + 0] =
                value.x;

            Bs[b_row][b_col + 1] =
                value.y;

            Bs[b_row][b_col + 2] =
                value.z;

            Bs[b_row][b_col + 3] =
                value.w;
        }


        // ====================================================
        // 整个 Block:
        //
        // Shared Tile Ready
        // ====================================================

        __syncthreads();


        // ====================================================
        // Compute
        // ====================================================

#pragma unroll
        for (
            int k = 0;
            k < BLOCK_K;
            ++k
        ) {

            // =================================================
            // A fragment
            //
            // V8:
            //
            // As[thread_row + i][k]
            //
            //
            // V9:
            //
            // AsT[k][thread_row + i]
            //
            //
            // 注意逻辑数据完全一样!
            //
            // 只是 Shared Memory layout 不一样。
            // =================================================

            float a_frag[THREAD_M];


#pragma unroll
            for (
                int i = 0;
                i < THREAD_M;
                ++i
            ) {

                a_frag[i] =
                    AsT[
                        k
                    ][
                        thread_row + i
                    ];
            }


            // =================================================
            // B fragment
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
            // 8 x 4 outer product
            //
            // 8 A
            // x
            // 4 B
            //
            // =
            //
            // 32 FMA
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
        // 当前 tile 使用结束
        // ====================================================

        __syncthreads();
    }


    // ========================================================
    // Register -> Global C
    //
    // 一个 thread 8 rows
    //
    // 每 row 4 contiguous floats
    //
    // 使用 float4 store
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
// CPU GEMM Reference
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
                        row * K + k
                    ]
                    *
                    B[
                        k * N + col
                    ];
            }


            C[
                row * N + col
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
    constexpr int M = 512;
    constexpr int N = 512;
    constexpr int K = 512;


    std::vector<float>
        h_A(M * K);

    std::vector<float>
        h_B(K * N);

    std::vector<float>
        h_C(M * N);


    // ========================================================
    // Initialize
    // ========================================================

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
    // GPU memory
    // ========================================================

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
    // Block Tile:
    //
    // 128 x 64
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

        gemmV9SharedTranspose<<<grid, block>>>(
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

        gemmV9SharedTranspose<<<grid, block>>>(
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
    // FLOPS
    //
    // GEMM:
    //
    // 2*M*N*K
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
    // CPU Reference
    // ========================================================

    std::vector<float>
        h_ref;


    gemmCPU(
        h_A,
        h_B,
        h_ref,
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
                h_ref[i]
            );


        max_error =
            std::max(
                max_error,
                error
            );
    }


    // ========================================================
    // GPU Info
    // ========================================================

    cudaDeviceProp prop;


    CUDA_CHECK(
        cudaGetDeviceProperties(
            &prop,
            0
        )
    );


    // ========================================================
    // Print
    // ========================================================

    std::cout
        << "GPU: "
        << prop.name
        << '\n';


    std::cout
        << "Compute Capability: "
        << prop.major
        << "."
        << prop.minor
        << '\n';


    std::cout
        << "Block Tile: "
        << BLOCK_M
        << " x "
        << BLOCK_N
        << '\n';


    std::cout
        << "Warp Tile: "
        << WARP_M
        << " x "
        << WARP_N
        << '\n';


    std::cout
        << "Thread Tile: "
        << THREAD_M
        << " x "
        << THREAD_N
        << '\n';


    std::cout
        << "A Shared Layout: "
        << BLOCK_K
        << " x "
        << A_TRANSPOSE_STRIDE
        << " (transposed + padding)"
        << '\n';


    std::cout
        << "Average Time: "
        << avg_ms
        << " ms\n";


    std::cout
        << "Performance: "
        << gflops
        << " GFLOPS\n";


    std::cout
        << "Max Error: "
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