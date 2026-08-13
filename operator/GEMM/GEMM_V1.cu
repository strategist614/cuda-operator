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
// Tile Size
//
// 一个 block:
// 16 x 16 = 256 threads
//
// 每个 block 计算 C 的一个 16 x 16 tile
// ============================================================

constexpr int TILE_SIZE = 16;


// ============================================================
// GEMM V1
//
// C = A * B
//
// A: [M, K]
// B: [K, N]
// C: [M, N]
//
// 优化：
// Shared Memory Tiling
// ============================================================

__global__ void gemmV1(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K) {

    // ========================================================
    // 当前线程在 block 里面的位置
    //
    // tx: 0 ~ 15
    // ty: 0 ~ 15
    // ========================================================

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;


    // ========================================================
    // 当前线程最终负责 C[row][col]
    // ========================================================

    const int row =
        blockIdx.y * TILE_SIZE + ty;

    const int col =
        blockIdx.x * TILE_SIZE + tx;


    // ========================================================
    // Shared Memory
    //
    // As:
    // A 当前的 16 x 16 tile
    //
    // Bs:
    // B 当前的 16 x 16 tile
    // ========================================================

    __shared__ float As[TILE_SIZE][TILE_SIZE];

    __shared__ float Bs[TILE_SIZE][TILE_SIZE];


    // ========================================================
    // 当前线程最终计算的 C[row][col]
    //
    // 一直放在 register 中累加
    // ========================================================

    float sum = 0.0f;


    // ========================================================
    // K 方向分成多个 tile
    //
    // 比如:
    //
    // K = 64
    // TILE_SIZE = 16
    //
    // 一共:
    //
    // tile 0: k =  0 ~ 15
    // tile 1: k = 16 ~ 31
    // tile 2: k = 32 ~ 47
    // tile 3: k = 48 ~ 63
    // ========================================================

    const int num_tiles =
        (K + TILE_SIZE - 1)
        / TILE_SIZE;


    for (int tile = 0;
         tile < num_tiles;
         ++tile) {


        // ====================================================
        // Step 1
        // 计算当前 tile 对应的 global memory index
        // ====================================================

        const int a_col =
            tile * TILE_SIZE + tx;

        const int b_row =
            tile * TILE_SIZE + ty;


        // ====================================================
        // Step 2
        // Global Memory -> Shared Memory
        //
        // 每个 thread 搬一个 A
        // 每个 thread 搬一个 B
        // ====================================================

        if (row < M && a_col < K) {

            As[ty][tx] =
                A[row * K + a_col];

        } else {

            As[ty][tx] =
                0.0f;
        }


        if (b_row < K && col < N) {

            Bs[ty][tx] =
                B[b_row * N + col];

        } else {

            Bs[ty][tx] =
                0.0f;
        }


        // ====================================================
        // Step 3
        // 等所有线程把 tile 搬完
        // ====================================================

        __syncthreads();


        // ====================================================
        // Step 4
        // 使用 Shared Memory 做 16 次乘加
        //
        // 当前 thread:
        //
        // As[ty][0..15]
        //
        // ×
        //
        // Bs[0..15][tx]
        // ====================================================

        #pragma unroll

        for (int k = 0;
             k < TILE_SIZE;
             ++k) {

            sum +=
                As[ty][k]
                *
                Bs[k][tx];
        }


        // ====================================================
        // Step 5
        //
        // 下一轮要覆盖 As / Bs
        //
        // 所以必须等当前 block 的所有 thread
        // 都使用完当前 tile
        // ====================================================

        __syncthreads();
    }


    // ========================================================
    // Step 6
    // 写最终结果
    // ========================================================

    if (row < M && col < N) {

        C[row * N + col] =
            sum;
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

    // ========================================================
    // Matrix Size
    //
    // A: [M, K]
    // B: [K, N]
    // C: [M, N]
    // ========================================================

    constexpr int M = 128;
    constexpr int K = 128;
    constexpr int N = 128;


    // ========================================================
    // Host Memory
    // ========================================================

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);


    // 初始化 A
    for (int i = 0;
         i < M * K;
         ++i) {

        h_A[i] =
            static_cast<float>(
                i % 10
            );
    }


    // 初始化 B
    for (int i = 0;
         i < K * N;
         ++i) {

        h_B[i] =
            static_cast<float>(
                i % 7
            );
    }


    // ========================================================
    // Device Memory
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


    // ========================================================
    // Host -> Device
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
    // Launch Configuration
    //
    // 一个 block:
    //
    // 16 x 16
    // =
    // 256 threads
    // ========================================================

    dim3 block(
        TILE_SIZE,
        TILE_SIZE
    );


    dim3 grid(
        (N + TILE_SIZE - 1)
            / TILE_SIZE,

        (M + TILE_SIZE - 1)
            / TILE_SIZE
    );


    // ========================================================
    // Launch GEMM V1
    // ========================================================

    gemmV1<<<grid, block>>>(
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
    // Device -> Host
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

    std::vector<float> cpu_C;


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


    std::cout << '\n';


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