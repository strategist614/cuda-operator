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
// Kernel 1
//
// S = Q * K^T / sqrt(D)
//
// Q: [N, D]
// K: [N, D]
//
// S: [N, N]
//
// 一个 thread 计算一个 S[row][col]
// ============================================================

__global__
void qkKernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ S,
    int N,
    int D)
{
    // ========================================================
    // 当前 thread 负责 S[row][col]
    // ========================================================

    const int col =
        blockIdx.x * blockDim.x
        +
        threadIdx.x;

    const int row =
        blockIdx.y * blockDim.y
        +
        threadIdx.y;


    if (
        row >= N
        ||
        col >= N
    ) {
        return;
    }


    // ========================================================
    // dot:
    //
    // Q[row, :]
    //
    // dot
    //
    // K[col, :]
    //
    // 注意:
    //
    // 因为数学上是 Q * K^T
    //
    // 所以这里实际上取 K 的第 col 行
    // ========================================================

    float sum =
        0.0f;


    for (
        int d = 0;
        d < D;
        ++d
    ) {

        const float q =
            Q[
                row * D
                +
                d
            ];


        const float k =
            K[
                col * D
                +
                d
            ];


        sum +=
            q * k;
    }


    // ========================================================
    // scale
    //
    // 1 / sqrt(D)
    // ========================================================

    const float scale =
        rsqrtf(
            static_cast<float>(D)
        );


    S[
        row * N
        +
        col
    ] =
        sum * scale;
}


// ============================================================
// Kernel 2
//
// P = Softmax(S)
//
// S: [N, N]
// P: [N, N]
//
// Attention Softmax 是对每一行做
//
// 为了最容易理解:
//
//      一个 thread
//      负责一整行 Softmax
//
// 这个版本很慢!
// 但是非常适合 V0。
// ============================================================

__global__
void softmaxKernel(
    const float* __restrict__ S,
    float* __restrict__ P,
    int N)
{
    const int row =
        blockIdx.x * blockDim.x
        +
        threadIdx.x;


    if (row >= N) {
        return;
    }


    // ========================================================
    // Step 1
    //
    // 找这一行最大值
    //
    // stable softmax
    // ========================================================

    float max_value =
        -INFINITY;


    for (
        int col = 0;
        col < N;
        ++col
    ) {

        const float value =
            S[
                row * N
                +
                col
            ];


        max_value =
            fmaxf(
                max_value,
                value
            );
    }


    // ========================================================
    // Step 2
    //
    // exp(x - max)
    //
    // 同时求 sum
    // ========================================================

    float sum =
        0.0f;


    for (
        int col = 0;
        col < N;
        ++col
    ) {

        const float value =
            expf(
                S[
                    row * N
                    +
                    col
                ]
                -
                max_value
            );


        P[
            row * N
            +
            col
        ] =
            value;


        sum +=
            value;
    }


    // ========================================================
    // Step 3
    //
    // normalize
    // ========================================================

    for (
        int col = 0;
        col < N;
        ++col
    ) {

        P[
            row * N
            +
            col
        ]
        /=
        sum;
    }
}


// ============================================================
// Kernel 3
//
// O = P * V
//
// P: [N, N]
// V: [N, D]
//
// O: [N, D]
//
// 一个 thread 计算一个 O[row][d]
// ============================================================

__global__
void pvKernel(
    const float* __restrict__ P,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N,
    int D)
{
    const int d =
        blockIdx.x * blockDim.x
        +
        threadIdx.x;


    const int row =
        blockIdx.y * blockDim.y
        +
        threadIdx.y;


    if (
        row >= N
        ||
        d >= D
    ) {
        return;
    }


    float sum =
        0.0f;


    // ========================================================
    // O[row][d]
    //
    // =
    //
    // sum_j
    //
    // P[row][j]
    // *
    // V[j][d]
    // ========================================================

    for (
        int j = 0;
        j < N;
        ++j
    ) {

        const float p =
            P[
                row * N
                +
                j
            ];


        const float v =
            V[
                j * D
                +
                d
            ];


        sum +=
            p * v;
    }


    O[
        row * D
        +
        d
    ] =
        sum;
}


// ============================================================
// CPU Reference
// ============================================================

void attentionCPU(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    std::vector<float>& O,
    int N,
    int D)
{
    // ========================================================
    // S = Q K^T / sqrt(D)
    // ========================================================

    std::vector<float>
        S(N * N);


    const float scale =
        1.0f /
        std::sqrt(
            static_cast<float>(D)
        );


    for (
        int i = 0;
        i < N;
        ++i
    ) {

        for (
            int j = 0;
            j < N;
            ++j
        ) {

            float sum =
                0.0f;


            for (
                int d = 0;
                d < D;
                ++d
            ) {

                sum +=
                    Q[
                        i * D
                        +
                        d
                    ]
                    *
                    K[
                        j * D
                        +
                        d
                    ];
            }


            S[
                i * N
                +
                j
            ] =
                sum * scale;
        }
    }


    // ========================================================
    // P = Softmax(S)
    // ========================================================

    std::vector<float>
        P(N * N);


    for (
        int row = 0;
        row < N;
        ++row
    ) {

        float max_value =
            -INFINITY;


        for (
            int col = 0;
            col < N;
            ++col
        ) {

            max_value =
                std::max(
                    max_value,
                    S[
                        row * N
                        +
                        col
                    ]
                );
        }


        float sum =
            0.0f;


        for (
            int col = 0;
            col < N;
            ++col
        ) {

            const float value =
                std::exp(
                    S[
                        row * N
                        +
                        col
                    ]
                    -
                    max_value
                );


            P[
                row * N
                +
                col
            ] =
                value;


            sum +=
                value;
        }


        for (
            int col = 0;
            col < N;
            ++col
        ) {

            P[
                row * N
                +
                col
            ]
            /=
            sum;
        }
    }


    // ========================================================
    // O = P V
    // ========================================================

    O.resize(
        N * D
    );


    for (
        int row = 0;
        row < N;
        ++row
    ) {

        for (
            int d = 0;
            d < D;
            ++d
        ) {

            float sum =
                0.0f;


            for (
                int j = 0;
                j < N;
                ++j
            ) {

                sum +=
                    P[
                        row * N
                        +
                        j
                    ]
                    *
                    V[
                        j * D
                        +
                        d
                    ];
            }


            O[
                row * D
                +
                d
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
    // Attention shape
    //
    // N = sequence length
    // D = head dimension
    //
    // 单 head
    // ========================================================

    constexpr int N =
        128;


    constexpr int D =
        64;


    // ========================================================
    // Host Data
    // ========================================================

    std::vector<float>
        h_Q(N * D);


    std::vector<float>
        h_K(N * D);


    std::vector<float>
        h_V(N * D);


    std::vector<float>
        h_O(N * D);


    // ========================================================
    // 初始化
    // ========================================================

    for (
        int i = 0;
        i < N * D;
        ++i
    ) {

        h_Q[i] =
            static_cast<float>(
                (i % 17) - 8
            )
            /
            16.0f;


        h_K[i] =
            static_cast<float>(
                (i % 13) - 6
            )
            /
            16.0f;


        h_V[i] =
            static_cast<float>(
                (i % 11) - 5
            )
            /
            8.0f;
    }


    // ========================================================
    // Device pointers
    // ========================================================

    float* d_Q =
        nullptr;


    float* d_K =
        nullptr;


    float* d_V =
        nullptr;


    float* d_S =
        nullptr;


    float* d_P =
        nullptr;


    float* d_O =
        nullptr;


    // ========================================================
    // Allocate
    // ========================================================

    CUDA_CHECK(
        cudaMalloc(
            &d_Q,
            N * D * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_K,
            N * D * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_V,
            N * D * sizeof(float)
        )
    );


    // ========================================================
    // 注意:
    //
    // S 是 N x N
    //
    // P 也是 N x N
    //
    // 这两个就是以后 FlashAttention
    // 最想消灭的大中间矩阵
    // ========================================================

    CUDA_CHECK(
        cudaMalloc(
            &d_S,
            N * N * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_P,
            N * N * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_O,
            N * D * sizeof(float)
        )
    );


    // ========================================================
    // Host -> Device
    // ========================================================

    CUDA_CHECK(
        cudaMemcpy(
            d_Q,
            h_Q.data(),
            N * D * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_K,
            h_K.data(),
            N * D * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_V,
            h_V.data(),
            N * D * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    // ========================================================
    // Kernel 1 launch
    //
    // S = QK^T
    // ========================================================

    dim3 block_qk(
        16,
        16
    );


    dim3 grid_qk(
        (N + block_qk.x - 1)
        /
        block_qk.x,

        (N + block_qk.y - 1)
        /
        block_qk.y
    );


    qkKernel<<<
        grid_qk,
        block_qk
    >>>(
        d_Q,
        d_K,
        d_S,
        N,
        D
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    // ========================================================
    // Kernel 2 launch
    //
    // P = Softmax(S)
    //
    // V0:
    //
    // 一个 thread 一行
    // ========================================================

    constexpr int SOFTMAX_THREADS =
        128;


    const int softmax_blocks =
        (
            N
            +
            SOFTMAX_THREADS
            -
            1
        )
        /
        SOFTMAX_THREADS;


    softmaxKernel<<<
        softmax_blocks,
        SOFTMAX_THREADS
    >>>(
        d_S,
        d_P,
        N
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    // ========================================================
    // Kernel 3
    //
    // O = PV
    // ========================================================

    dim3 block_pv(
        16,
        16
    );


    dim3 grid_pv(
        (D + block_pv.x - 1)
        /
        block_pv.x,

        (N + block_pv.y - 1)
        /
        block_pv.y
    );


    pvKernel<<<
        grid_pv,
        block_pv
    >>>(
        d_P,
        d_V,
        d_O,
        N,
        D
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
            h_O.data(),
            d_O,
            N * D * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    // ========================================================
    // CPU Reference
    // ========================================================

    std::vector<float>
        h_ref;


    attentionCPU(
        h_Q,
        h_K,
        h_V,
        h_ref,
        N,
        D
    );


    // ========================================================
    // Check
    // ========================================================

    float max_error =
        0.0f;


    for (
        int i = 0;
        i < N * D;
        ++i
    ) {

        max_error =
            std::max(
                max_error,
                std::fabs(
                    h_O[i]
                    -
                    h_ref[i]
                )
            );
    }


    std::cout
        << "N = "
        << N
        << '\n';


    std::cout
        << "D = "
        << D
        << '\n';


    std::cout
        << "Attention matrix S shape = "
        << N
        << " x "
        << N
        << '\n';


    std::cout
        << "Max error = "
        << max_error
        << '\n';


    std::cout
        << "\nFirst output row:\n";


    for (
        int d = 0;
        d < std::min(D, 8);
        ++d
    ) {

        std::cout
            << h_O[d]
            << " ";
    }


    std::cout
        << '\n';


    // ========================================================
    // Cleanup
    // ========================================================

    CUDA_CHECK(
        cudaFree(
            d_Q
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_K
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_V
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_S
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_P
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_O
        )
    );


    return 0;
}