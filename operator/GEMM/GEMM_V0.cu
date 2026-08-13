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
// GEMM V0
//
// C = A * B
//
// A: [M, K]
// B: [K, N]
// C: [M, N]
//
// 一个 CUDA thread 负责计算一个 C[row, col]
// ============================================================

__global__ void gemmV0(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K) {

    // 当前线程负责的输出矩阵坐标
    // row 是当前线程负责的输出矩阵 C 的行索引
    // col 是当前线程负责的输出矩阵 C 的列索引
    const int row =
        blockIdx.y * blockDim.y
        + threadIdx.y;

    const int col =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    // 防止线程越界
    if (row >= M || col >= N) {
        return;
    }

    // ========================================================
    // 计算 C[row, col]
    //
    // C[row, col]
    // =
    // A[row, 0] * B[0, col]
    // +
    // A[row, 1] * B[1, col]
    // +
    // ...
    // ========================================================

    float sum = 0.0f;

    for (int k = 0; k < K; ++k) {

        const float a =
            A[row * K + k];

        const float b =
            B[k * N + col];

        sum += a * b;
    }

    // 写回输出
    C[row * N + col] =
        sum;
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

    for (int row = 0; row < M; ++row) {

        for (int col = 0; col < N; ++col) {

            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {

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
// Print Matrix
// ============================================================

void printMatrix(
    const std::vector<float>& matrix,
    int rows,
    int cols) {

    for (int i = 0; i < rows; ++i) {

        for (int j = 0; j < cols; ++j) {

            std::cout
                << matrix[i * cols + j]
                << "\t";
        }

        std::cout << '\n';
    }
}


// ============================================================
// Main
// ============================================================

int main() {

    // ========================================================
    // Matrix Size
    //
    // A = [M, K]
    // B = [K, N]
    // C = [M, N]
    // ========================================================

    constexpr int M = 4;
    constexpr int K = 3;
    constexpr int N = 5;


    // ========================================================
    // Host Matrices
    // ========================================================

    std::vector<float> h_A = {

        // row 0
        1.0f, 2.0f, 3.0f,

        // row 1
        4.0f, 5.0f, 6.0f,

        // row 2
        7.0f, 8.0f, 9.0f,

        // row 3
        1.0f, 1.0f, 1.0f
    };


    std::vector<float> h_B = {

        // row 0
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,

        // row 1
        6.0f, 7.0f, 8.0f, 9.0f, 10.0f,

        // row 2
        11.0f, 12.0f, 13.0f, 14.0f, 15.0f
    };


    std::vector<float>
        h_C(M * N);

    std::vector<float>
        cpu_C;


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
    // 16 x 16 = 256 threads
    // ========================================================

    dim3 block(
        16,
        16
    );


    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );


    gemmV0<<<grid, block>>>(
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

    gemmCPU(
        h_A,
        h_B,
        cpu_C,
        M,
        N,
        K
    );


    // ========================================================
    // Print
    // ========================================================

    std::cout
        << "CUDA GEMM V0:\n";

    printMatrix(
        h_C,
        M,
        N
    );


    std::cout
        << "\nCPU Reference:\n";

    printMatrix(
        cpu_C,
        M,
        N
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
        << "\nMax error: "
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
