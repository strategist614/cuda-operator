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
// GEMM V3 Configuration
//
// 一个 block:
//     16 x 16 threads
//     = 256 threads
//
// 一个 thread:
//     计算 4 x 4 个 C
//
// 所以一个 block:
//     计算 64 x 64 个 C
//
// K 每次处理:
//     16
//
// ============================================================

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 16;

constexpr int THREAD_M = 4;
constexpr int THREAD_N = 4;


// ============================================================
// GEMM V3
//
// C[M,N] = A[M,K] * B[K,N]
//
// Shared Memory Tiling
// +
// 4x4 Register Tiling
// ============================================================

__global__ void gemmV3(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K) {

    // --------------------------------------------------------
    // Thread index
    //
    // tx = 0 ~ 15
    // ty = 0 ~ 15
    // --------------------------------------------------------

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int tid =
        ty * blockDim.x + tx;


    // --------------------------------------------------------
    // Shared Memory
    //
    // A tile:
    //     64 x 16
    //
    // B tile:
    //     16 x 64
    //
    // --------------------------------------------------------

    __shared__ float As[BLOCK_M][BLOCK_K];
    __shared__ float Bs[BLOCK_K][BLOCK_N];


    // --------------------------------------------------------
    // 当前 thread 在 block C tile 中负责区域的左上角
    //
    // ty=0 -> row 0
    // ty=1 -> row 4
    // ty=2 -> row 8
    //
    // tx=0 -> col 0
    // tx=1 -> col 4
    // tx=2 -> col 8
    // --------------------------------------------------------

    const int thread_row =
        ty * THREAD_M;

    const int thread_col =
        tx * THREAD_N;


    // --------------------------------------------------------
    // 当前 block 对应整个 C 的起点
    // --------------------------------------------------------

    const int block_row =
        blockIdx.y * BLOCK_M;

    const int block_col =
        blockIdx.x * BLOCK_N;


    // ========================================================
    // Register Tile
    //
    // 一个 thread 负责 4 x 4 = 16 个 C
    //
    // 尽量写成 scalar，
    // 方便编译器放进 register
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


    // --------------------------------------------------------
    // K 方向分 tile
    // --------------------------------------------------------

    const int num_k_tiles =
        (K + BLOCK_K - 1)
        / BLOCK_K;


    for (int tile = 0;
         tile < num_k_tiles;
         ++tile) {

        // ====================================================
        // Step 1:
        // cooperative load A
        //
        // As:
        //     64 x 16
        //     = 1024 floats
        //
        // 256 threads
        //
        // 平均每个 thread 搬 4 个 A
        // ====================================================

        for (int index = tid;
             index < BLOCK_M * BLOCK_K;
             index += blockDim.x * blockDim.y) {

            const int smem_row =
                index / BLOCK_K;

            const int smem_col =
                index % BLOCK_K;


            const int global_row =
                block_row + smem_row;

            const int global_col =
                tile * BLOCK_K
                + smem_col;


            if (global_row < M &&
                global_col < K) {

                As[smem_row][smem_col] =
                    A[
                        global_row * K
                        + global_col
                    ];

            } else {

                As[smem_row][smem_col] =
                    0.0f;
            }
        }


        // ====================================================
        // Step 2:
        // cooperative load B
        //
        // Bs:
        //     16 x 64
        //     = 1024 floats
        //
        // 平均每个 thread 搬 4 个 B
        // ====================================================

        for (int index = tid;
             index < BLOCK_K * BLOCK_N;
             index += blockDim.x * blockDim.y) {

            const int smem_row =
                index / BLOCK_N;

            const int smem_col =
                index % BLOCK_N;


            const int global_row =
                tile * BLOCK_K
                + smem_row;

            const int global_col =
                block_col
                + smem_col;


            if (global_row < K &&
                global_col < N) {

                Bs[smem_row][smem_col] =
                    B[
                        global_row * N
                        + global_col
                    ];

            } else {

                Bs[smem_row][smem_col] =
                    0.0f;
            }
        }


        // 等大家搬完
        __syncthreads();


        // ====================================================
        // Step 3:
        //
        // Register Tiling
        //
        // 每个 k:
        //
        // 从 Shared Memory:
        //
        // 读取:
        //     4 个 A
        //     4 个 B
        //
        // 得到:
        //     16 次 FMA
        //
        // ====================================================

        #pragma unroll

        for (int k = 0;
             k < BLOCK_K;
             ++k) {

            // ------------------------------------------------
            // 从 A tile 拿 4 个数
            //
            //        a0
            //        a1
            //        a2
            //        a3
            //
            // ------------------------------------------------

            const float a0 =
                As[thread_row + 0][k];

            const float a1 =
                As[thread_row + 1][k];

            const float a2 =
                As[thread_row + 2][k];

            const float a3 =
                As[thread_row + 3][k];


            // ------------------------------------------------
            // 从 B tile 拿 4 个数
            //
            // b0 b1 b2 b3
            //
            // ------------------------------------------------

            const float b0 =
                Bs[k][thread_col + 0];

            const float b1 =
                Bs[k][thread_col + 1];

            const float b2 =
                Bs[k][thread_col + 2];

            const float b3 =
                Bs[k][thread_col + 3];


            // =================================================
            // 4 x 4 Outer Product
            //
            //
            //             b0    b1    b2    b3
            //
            //      a0    c00   c01   c02   c03
            //
            //      a1    c10   c11   c12   c13
            //
            //      a2    c20   c21   c22   c23
            //
            //      a3    c30   c31   c32   c33
            //
            //
            // =================================================


            // row 0

            c00 += a0 * b0;
            c01 += a0 * b1;
            c02 += a0 * b2;
            c03 += a0 * b3;


            // row 1

            c10 += a1 * b0;
            c11 += a1 * b1;
            c12 += a1 * b2;
            c13 += a1 * b3;


            // row 2

            c20 += a2 * b0;
            c21 += a2 * b1;
            c22 += a2 * b2;
            c23 += a2 * b3;


            // row 3

            c30 += a3 * b0;
            c31 += a3 * b1;
            c32 += a3 * b2;
            c33 += a3 * b3;
        }


        // 当前 tile 大家都用完
        // 才允许下一轮覆盖 Shared Memory

        __syncthreads();
    }


    // ========================================================
    // Step 4:
    //
    // 当前 thread 最终负责:
    //
    // row0:
    //   c00 c01 c02 c03
    //
    // row1:
    //   c10 c11 c12 c13
    //
    // row2:
    //   c20 c21 c22 c23
    //
    // row3:
    //   c30 c31 c32 c33
    //
    // ========================================================

    const int row0 =
        block_row + thread_row + 0;

    const int row1 =
        block_row + thread_row + 1;

    const int row2 =
        block_row + thread_row + 2;

    const int row3 =
        block_row + thread_row + 3;


    const int col0 =
        block_col + thread_col + 0;

    const int col1 =
        block_col + thread_col + 1;

    const int col2 =
        block_col + thread_col + 2;

    const int col3 =
        block_col + thread_col + 3;


    // ========================================================
    // Register -> Global Memory
    // ========================================================


    // row 0

    if (row0 < M) {

        if (col0 < N)
            C[row0 * N + col0] = c00;

        if (col1 < N)
            C[row0 * N + col1] = c01;

        if (col2 < N)
            C[row0 * N + col2] = c02;

        if (col3 < N)
            C[row0 * N + col3] = c03;
    }


    // row 1

    if (row1 < M) {

        if (col0 < N)
            C[row1 * N + col0] = c10;

        if (col1 < N)
            C[row1 * N + col1] = c11;

        if (col2 < N)
            C[row1 * N + col2] = c12;

        if (col3 < N)
            C[row1 * N + col3] = c13;
    }


    // row 2

    if (row2 < M) {

        if (col0 < N)
            C[row2 * N + col0] = c20;

        if (col1 < N)
            C[row2 * N + col1] = c21;

        if (col2 < N)
            C[row2 * N + col2] = c22;

        if (col3 < N)
            C[row2 * N + col3] = c23;
    }


    // row 3

    if (row3 < M) {

        if (col0 < N)
            C[row3 * N + col0] = c30;

        if (col1 < N)
            C[row3 * N + col1] = c31;

        if (col2 < N)
            C[row3 * N + col2] = c32;

        if (col3 < N)
            C[row3 * N + col3] = c33;
    }
}


// ============================================================
// CPU reference
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

            float sum = 0.0f;

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

    constexpr int M = 256;
    constexpr int N = 256;
    constexpr int K = 256;


    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);


    for (int i = 0;
         i < M * K;
         ++i) {

        h_A[i] =
            static_cast<float>(i % 13)
            / 10.0f;
    }


    for (int i = 0;
         i < K * N;
         ++i) {

        h_B[i] =
            static_cast<float>(i % 7)
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


    // --------------------------------------------------------
    // 64 / 4 = 16
    //
    // 所以:
    //
    // block = 16 x 16
    //       = 256 threads
    // --------------------------------------------------------

    dim3 block(
        BLOCK_N / THREAD_N,
        BLOCK_M / THREAD_M
    );


    // --------------------------------------------------------
    // 每个 block 负责 64 x 64 C
    // --------------------------------------------------------

    dim3 grid(
        (N + BLOCK_N - 1)
            / BLOCK_N,

        (M + BLOCK_M - 1)
            / BLOCK_M
    );


    gemmV3<<<grid, block>>>(
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


    CUDA_CHECK(
        cudaMemcpy(
            h_C.data(),
            d_C,
            M * N * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    // CPU reference

    std::vector<float> cpu_C;

    gemmCPU(
        h_A,
        h_B,
        cpu_C,
        M,
        N,
        K
    );


    // Verify

    float max_error = 0.0f;

    for (int i = 0;
         i < M * N;
         ++i) {

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
        << "Max error: "
        << max_error
        << '\n';


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