#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>


// ============================================================
// CUDA error check
// ============================================================

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA error: "                       \
                      << cudaGetErrorString(err)               \
                      << " at "                                \
                      << __FILE__                              \
                      << ":"                                  \
                      << __LINE__                              \
                      << std::endl;                            \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)


// ============================================================
// Block Tile
// ============================================================
//
// 一个 block 负责:
//
//      C 128 x 64
//
// K 每次处理:
//
//      16
//
// ============================================================

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 16;


// ============================================================
// Warp Tile
// ============================================================
//
// 一个 warp 负责:
//
//      32 x 32
//
// ============================================================

constexpr int WARP_M = 32;
constexpr int WARP_N = 32;


// ============================================================
// 一个 block 里面 warp 排布
//
//     Warp0 Warp1
//     Warp2 Warp3
//     Warp4 Warp5
//     Warp6 Warp7
//
// 4 x 2 = 8 warps
// ============================================================

constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;

constexpr int NUM_WARPS =
    WARPS_M * WARPS_N;


// ============================================================
// 每个 thread 负责:
//
//      8 x 4
//
// =
//
//      32 个 C
//
// ============================================================

constexpr int THREAD_M = 8;
constexpr int THREAD_N = 4;


constexpr int WARP_SIZE = 32;

constexpr int THREADS_PER_BLOCK =
    NUM_WARPS * WARP_SIZE;


// ============================================================
// float4
// ============================================================

constexpr int VEC_SIZE = 4;


// ============================================================
// V8 最重要的参数
//
// V7:
//
//      As[128][16]
//
// V8:
//
//      As[128][17]
//
// 多出来的 1 个 float 是 padding
// ============================================================

constexpr int A_SMEM_STRIDE =
    BLOCK_K + 1;      // 17


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
    // 16-byte aligned + 四个元素都合法
    // ========================================================

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


    // ========================================================
    // boundary fallback
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
// GEMM V8
//
// C = A * B
//
// A: M x K
// B: K x N
// C: M x N
//
// ============================================================

__global__
void gemmV8BankConflict(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K)
{
    // ========================================================
    // thread id
    //
    // 0 ~ 255
    // ========================================================

    const int tid =
        threadIdx.x;


    // ========================================================
    // 属于哪个 warp
    // ========================================================

    const int warp_id =
        tid / WARP_SIZE;


    // ========================================================
    // warp 内第几个 lane
    // ========================================================

    const int lane_id =
        tid % WARP_SIZE;


    // ========================================================
    // Shared Memory
    // ========================================================
    //
    // 注意:
    //
    // As 有 padding!
    //
    // 实际有效数据依然只有:
    //
    //      [128][16]
    //
    // 但是物理 stride 变成:
    //
    //      17
    //
    // ========================================================

    __shared__ __align__(16)
    float As[BLOCK_M][A_SMEM_STRIDE];


    // ========================================================
    // B 当前访问模式没有我们这里重点解决的
    // 那个 4-way conflict
    //
    // 所以暂时不 padding
    // ========================================================

    __shared__ __align__(16)
    float Bs[BLOCK_K][BLOCK_N];


    // ========================================================
    // 当前 block 在 C 中的左上角
    // ========================================================

    const int block_row =
        blockIdx.y * BLOCK_M;

    const int block_col =
        blockIdx.x * BLOCK_N;


    // ========================================================
    // Warp Tiling
    // ========================================================

    const int warp_m =
        warp_id / WARPS_N;

    const int warp_n =
        warp_id % WARPS_N;


    // ========================================================
    // warp tile 在 block 中的左上角
    // ========================================================

    const int warp_row =
        warp_m * WARP_M;

    const int warp_col =
        warp_n * WARP_N;


    // ========================================================
    // Warp 内 32 lanes 排成:
    //
    //      4 x 8
    //
    // 每 lane:
    //
    //      8 x 4
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
    // 当前 thread 的 8x4 tile
    // 在 block tile 内的位置
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
    // 一个 thread 保存:
    //
    //      8 x 4 = 32
    //
    // 个 C partial sum
    // ========================================================

    float acc[THREAD_M][THREAD_N];


#pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {

#pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {

            acc[i][j] =
                0.0f;
        }
    }


    // ========================================================
    // K tiles
    // ========================================================

    const int num_k_tiles =
        (K + BLOCK_K - 1)
        /
        BLOCK_K;


    // ========================================================
    // A tile:
    //
    // 128 x 16
    //
    // 有效数据仍然只有 2048 floats
    //
    // padding 不需要加载任何数据
    // ========================================================

    constexpr int A_VECS_PER_ROW =
        BLOCK_K / VEC_SIZE;

    // 16 / 4 = 4


    constexpr int A_TOTAL_VECS =
        BLOCK_M * A_VECS_PER_ROW;

    // 128 * 4 = 512


    // ========================================================
    // B tile:
    //
    // 16 x 64
    // ========================================================

    constexpr int B_VECS_PER_ROW =
        BLOCK_N / VEC_SIZE;

    // 64 / 4 = 16


    constexpr int B_TOTAL_VECS =
        BLOCK_K * B_VECS_PER_ROW;

    // 16 * 16 = 256


    // ========================================================
    // Main K Tile Loop
    // ========================================================

    for (
        int tile = 0;
        tile < num_k_tiles;
        ++tile
    ) {

        // ====================================================
        // Global A -> Shared A
        // ====================================================
        //
        // 注意:
        //
        // Global A 没 padding
        //
        // Shared As 有 padding
        //
        // 所以 global_col 还是:
        //
        //      0 4 8 12
        //
        // Shared 里面也只写:
        //
        //      As[row][0~15]
        //
        // As[row][16]
        //
        // 永远不存有效矩阵数据
        // ====================================================

        for (
            int vec_index = tid;
            vec_index < A_TOTAL_VECS;
            vec_index += THREADS_PER_BLOCK
        ) {

            const int smem_row =
                vec_index
                /
                A_VECS_PER_ROW;


            const int vec_col =
                vec_index
                %
                A_VECS_PER_ROW;


            const int smem_col =
                vec_col
                *
                VEC_SIZE;


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


            // =================================================
            // 写入 padded Shared Memory
            //
            // row 的真实 stride 是 17
            // =================================================

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
        // Global B -> Shared B
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
        // 等整个 Block 搬完
        // ====================================================

        __syncthreads();


        // ====================================================
        // Compute current tile
        // ====================================================

#pragma unroll
        for (
            int k = 0;
            k < BLOCK_K;
            ++k
        ) {

            // =================================================
            // Shared A -> registers
            //
            // 这里就是 V8 padding 真正发挥作用的位置
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
            // Shared B -> registers
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
        // 所有 thread 用完 Shared Tile
        // ====================================================

        __syncthreads();
    }


    // ========================================================
    // Register -> Global C
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
// CPU reference
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
    // ========================================================
    // Test shape
    // ========================================================

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
    // Device memory
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
    // 256 threads = 8 warps
    // ========================================================

    dim3 block(
        THREADS_PER_BLOCK
    );


    // ========================================================
    // 每个 block 计算 128x64
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

        gemmV8BankConflict<<<grid, block>>>(
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

        gemmV8BankConflict<<<grid, block>>>(
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
    // GEMM FLOPs
    //
    // 2 * M * N * K
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
    // CPU reference
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
    // Check error
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
    // Device info
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
        << "A shared stride: "
        << A_SMEM_STRIDE
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