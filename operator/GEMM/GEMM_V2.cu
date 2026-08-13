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
            std::cerr << "CUDA Error: "                       \
                      << cudaGetErrorString(err)               \
                      << " at " << __FILE__                    \
                      << ":" << __LINE__                       \
                      << std::endl;                            \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)


// ============================================================
// GEMM V2 Configuration
// ============================================================
//
// 一个 block 计算 C 的:
//
//      32 x 32
//
// 每个 thread 计算:
//
//      2 x 2
//
// 因此需要:
//
//      32 / 2 = 16 threads in x
//      32 / 2 = 16 threads in y
//
// 总线程数:
//
//      16 x 16 = 256 threads
//
// K 每次处理 16 个:
//
//      BLOCK_K = 16
//
// ============================================================

constexpr int BLOCK_M = 32;
constexpr int BLOCK_N = 32;
constexpr int BLOCK_K = 16;

constexpr int THREAD_M = 2;
constexpr int THREAD_N = 2;


// ============================================================
// GEMM V2
//
// C = A * B
//
// A: [M, K]
// B: [K, N]
// C: [M, N]
//
// V2:
//
// 1. Shared Memory Tiling
// 2. Register Tiling
// 3. 一个 thread 计算 2 x 2 个 C
//
// ============================================================

__global__ void gemmV2(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K) {

    // ========================================================
    // 当前 thread 在 block 中的位置
    //
    // tx: 0 ~ 15
    // ty: 0 ~ 15
    // ========================================================

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // 把二维 thread id 转成一维
    //
    // 0 ~ 255
    const int tid =
        ty * blockDim.x + tx;


    // ========================================================
    // Shared Memory
    //
    // A Tile:
    //
    //     32 x 16
    //
    // B Tile:
    //
    //     16 x 32
    //
    // ========================================================

    __shared__ float As[BLOCK_M][BLOCK_K];

    __shared__ float Bs[BLOCK_K][BLOCK_N];


    // ========================================================
    // 当前线程负责 C tile 中的位置
    //
    // 一个 thread 负责 2 x 2
    //
    // 比如:
    //
    // tx = 3
    // ty = 5
    //
    // thread_row = 10
    // thread_col = 6
    //
    // 负责:
    //
    // C[10][6]  C[10][7]
    // C[11][6]  C[11][7]
    //
    // ========================================================

    const int thread_row =
        ty * THREAD_M;

    const int thread_col =
        tx * THREAD_N;


    // ========================================================
    // 当前 block 在整个 C 矩阵中的起点
    // ========================================================

    const int block_row =
        blockIdx.y * BLOCK_M;

    const int block_col =
        blockIdx.x * BLOCK_N;


    // ========================================================
    // Register Tile
    //
    // 当前 thread 的 4 个输出结果
    //
    // 全程放在 thread private scalar 中
    //
    // ========================================================

    float c00 = 0.0f;
    float c01 = 0.0f;

    float c10 = 0.0f;
    float c11 = 0.0f;


    // ========================================================
    // K 方向分块
    //
    // 每次处理 BLOCK_K = 16
    //
    // ========================================================

    const int num_k_tiles =
        (K + BLOCK_K - 1)
        / BLOCK_K;


    for (int tile = 0;
         tile < num_k_tiles;
         ++tile) {

        // ====================================================
        // Step 1
        //
        // 256 threads 合作加载:
        //
        // As = 32 x 16 = 512 elements
        //
        // 每个 thread 搬 2 个 A
        //
        // ====================================================

        for (int index = tid;
             index < BLOCK_M * BLOCK_K;
             index += blockDim.x * blockDim.y) {

            // shared memory 坐标
            const int smem_row =
                index / BLOCK_K;

            const int smem_col =
                index % BLOCK_K;


            // global memory 坐标
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
        // Step 2
        //
        // 256 threads 合作加载:
        //
        // Bs = 16 x 32 = 512 elements
        //
        // 每个 thread 搬 2 个 B
        //
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


        // ====================================================
        // 等待整个 block 把 A/B Tile 搬完
        // ====================================================

        __syncthreads();


        // ====================================================
        // Step 3
        //
        // Register Tiling
        //
        // 每个 thread 计算 2 x 2 个结果
        //
        // ====================================================

        #pragma unroll

        for (int k = 0;
             k < BLOCK_K;
             ++k) {

            // =================================================
            // 从 Shared Memory 读取两个 A
            //
            // A:
            //
            // a0  ---->
            // a1  ---->
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


            // =================================================
            // 从 Shared Memory 读取两个 B
            //
            // B:
            //
            // b0 b1
            //
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


            // =================================================
            // 关键:
            //
            // 4 个 register accumulator
            //
            //         b0      b1
            //
            // a0     c00     c01
            //
            // a1     c10     c11
            //
            // =================================================

            c00 += a0 * b0;
            c01 += a0 * b1;

            c10 += a1 * b0;
            c11 += a1 * b1;
        }


        // ====================================================
        // 当前 tile 使用完
        //
        // 下一轮准备覆盖 shared memory
        // ====================================================

        __syncthreads();
    }


    // ========================================================
    // Step 4
    //
    // 计算当前 thread 最终负责的 global C 坐标
    // ========================================================

    const int row0 =
        block_row
        + thread_row;

    const int row1 =
        row0 + 1;


    const int col0 =
        block_col
        + thread_col;

    const int col1 =
        col0 + 1;


    // ========================================================
    // Step 5
    //
    // Register -> Global Memory
    // ========================================================

    if (row0 < M && col0 < N) {

        C[row0 * N + col0] =
            c00;
    }


    if (row0 < M && col1 < N) {

        C[row0 * N + col1] =
            c01;
    }


    if (row1 < M && col0 < N) {

        C[row1 * N + col0] =
            c10;
    }


    if (row1 < M && col1 < N) {

        C[row1 * N + col1] =
            c11;
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
    int K) {

    C.resize(
        M * N
    );


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
            ] = sum;
        }
    }
}


// ============================================================
// Main
// ============================================================

int main() {

    // ========================================================
    // Matrix Size
    //
    // A: [M, K]
    // B: [K, N]
    // C: [M, N]
    // ========================================================

    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 128;


    // ========================================================
    // Host Memory
    // ========================================================

    std::vector<float>
        h_A(M * K);

    std::vector<float>
        h_B(K * N);

    std::vector<float>
        h_C(M * N);


    // ========================================================
    // 初始化
    // ========================================================

    for (int i = 0;
         i < M * K;
         ++i) {

        h_A[i] =
            static_cast<float>(
                i % 13
            ) / 10.0f;
    }


    for (int i = 0;
         i < K * N;
         ++i) {

        h_B[i] =
            static_cast<float>(
                i % 7
            ) / 10.0f;
    }


    // ========================================================
    // Device Memory
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


    // ========================================================
    // CPU -> GPU
    // ========================================================

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
    // Block
    //
    // 一个 thread 算 2 x 2
    //
    // C tile = 32 x 32
    //
    // 所以:
    //
    // 32 / 2 = 16
    //
    // block:
    //
    // 16 x 16
    // =
    // 256 threads
    //
    // ========================================================

    dim3 block(
        BLOCK_N / THREAD_N,
        BLOCK_M / THREAD_M
    );


    // ========================================================
    // Grid
    //
    // 每个 block 负责 C 的 32 x 32
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

    gemmV2<<<grid, block>>>(
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
    // Verify
    // ========================================================

    float max_error =
        0.0f;


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


    std::cout
        << "First 10 CUDA results:\n";


    for (int i = 0;
         i < 10;
         ++i) {

        std::cout
            << h_C[i]
            << " ";
    }


    std::cout
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