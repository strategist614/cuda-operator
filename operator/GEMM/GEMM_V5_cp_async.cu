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
// GEMM configuration
//
// 一个 block:
//      16 x 16
//      = 256 threads
//
// 一个 thread:
//      4 x 4 C
//
// 一个 block:
//
//      64 x 64 C
//
// K tile:
//      16
//
// Double Buffer:
//      2 stages
// ============================================================

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 16;

constexpr int THREAD_M = 4;
constexpr int THREAD_N = 4;

constexpr int VEC_SIZE = 4;


// ============================================================
// cp.async helper
//
// 一次搬:
//
//      16 bytes
//
// =
//
//      4 x float
//
// 数据:
//
//      Global Memory
//            |
//            | cp.async
//            v
//      Shared Memory
//
// ============================================================

__device__ __forceinline__
void cp_async_16(
    void* smem_ptr,
    const void* gmem_ptr) {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    // --------------------------------------------------------
    // PTX 的 shared 地址不是普通 CUDA generic pointer。
    //
    // 所以:
    //
    // CUDA shared pointer
    //        ↓
    // shared address
    //
    // --------------------------------------------------------

    unsigned int smem_addr =
        static_cast<unsigned int>(
            __cvta_generic_to_shared(
                smem_ptr
            )
        );


    // --------------------------------------------------------
    // 真正的:
    //
    // cp.async
    //
    // global
    //    ↓
    // shared
    //
    // 16 bytes
    // --------------------------------------------------------

    asm volatile(
        "cp.async.cg.shared.global "
        "[%0], [%1], 16;\n"
        :
        : "r"(smem_addr),
          "l"(gmem_ptr)
    );

#else

    // ========================================================
    // sm_75 fallback
    //
    // 注意:
    //
    // 这里不是 cp.async。
    //
    // 只是为了让你当前机器也可以编译运行。
    // ========================================================

    float4 value =
        *reinterpret_cast<const float4*>(
            gmem_ptr
        );


    *reinterpret_cast<float4*>(
        smem_ptr
    ) = value;

#endif
}


// ============================================================
// commit group
// ============================================================

__device__ __forceinline__
void cp_async_commit() {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    asm volatile(
        "cp.async.commit_group;\n"
    );

#endif
}


// ============================================================
// wait group 0
//
// 意思:
//
// 之前提交的 cp.async group
//
// 全部完成
//
// ============================================================

__device__ __forceinline__
void cp_async_wait_all() {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    asm volatile(
        "cp.async.wait_group 0;\n"
    );

#endif
}


// ============================================================
// GEMM cp.async
// ============================================================

__global__
void gemmCpAsync(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K) {


    // ========================================================
    // Thread index
    // ========================================================

    const int tx =
        threadIdx.x;

    const int ty =
        threadIdx.y;


    const int tid =
        ty * blockDim.x
        + tx;


    // ========================================================
    // Double-buffered Shared Memory
    //
    //
    //           stage 0
    //
    //      As[0]
    //      Bs[0]
    //
    //
    //           stage 1
    //
    //      As[1]
    //      Bs[1]
    //
    //
    // ========================================================

    __shared__ __align__(16)
    float As[2][BLOCK_M][BLOCK_K];


    __shared__ __align__(16)
    float Bs[2][BLOCK_K][BLOCK_N];


    // ========================================================
    // 当前 thread 负责 C 的 4 x 4
    // ========================================================

    const int thread_row =
        ty * THREAD_M;


    const int thread_col =
        tx * THREAD_N;


    // ========================================================
    // 当前 block 对应 C 的位置
    // ========================================================

    const int block_row =
        blockIdx.y * BLOCK_M;


    const int block_col =
        blockIdx.x * BLOCK_N;


    // ========================================================
    // 16 个 Register Accumulators
    //
    // 一个 thread:
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
    // A Tile:
    //
    //      64 x 16
    //
    // 每行:
    //
    //      16 float
    //
    // =
    //
    //      4 float4
    //
    // ========================================================

    constexpr int A_VECS_PER_ROW =
        BLOCK_K / VEC_SIZE;


    // tid:
    //
    // 0  -> row0 col0
    // 1  -> row0 col4
    // 2  -> row0 col8
    // 3  -> row0 col12
    //
    // 4  -> row1 col0
    //
    // ...

    const int a_smem_row =
        tid / A_VECS_PER_ROW;


    const int a_smem_col =
        (tid % A_VECS_PER_ROW)
        * VEC_SIZE;


    // ========================================================
    // B Tile:
    //
    //      16 x 64
    //
    // 每行:
    //
    //      64 float
    //
    // =
    //
    //      16 float4
    //
    // ========================================================

    constexpr int B_VECS_PER_ROW =
        BLOCK_N / VEC_SIZE;


    const int b_smem_row =
        tid / B_VECS_PER_ROW;


    const int b_smem_col =
        (tid % B_VECS_PER_ROW)
        * VEC_SIZE;


    // ========================================================
    // K tiles
    //
    // 这里教学版要求:
    //
    // K % BLOCK_K == 0
    //
    // ========================================================

    const int num_tiles =
        K / BLOCK_K;


    // ========================================================
    //
    // STEP 1
    //
    // Prefetch tile 0
    //
    //
    // Global tile0
    //
    //       |
    //       | cp.async
    //       v
    //
    // Shared Buffer 0
    //
    // ========================================================


    {
        // ----------------------------------------------------
        // A tile 0
        // ----------------------------------------------------

        const float* A_src =
            A
            +
            (block_row + a_smem_row)
                * K
            +
            a_smem_col;


        float* A_dst =
            &As[
                0
            ][
                a_smem_row
            ][
                a_smem_col
            ];


        cp_async_16(
            A_dst,
            A_src
        );


        // ----------------------------------------------------
        // B tile 0
        // ----------------------------------------------------

        const float* B_src =
            B
            +
            b_smem_row
                * N
            +
            block_col
            +
            b_smem_col;


        float* B_dst =
            &Bs[
                0
            ][
                b_smem_row
            ][
                b_smem_col
            ];


        cp_async_16(
            B_dst,
            B_src
        );


        // ----------------------------------------------------
        // A copy + B copy
        //
        // 算成一个 async group
        // ----------------------------------------------------

        cp_async_commit();


        // ----------------------------------------------------
        // 第一块必须真的到 Shared
        //
        // 因为马上要计算它
        // ----------------------------------------------------

        cp_async_wait_all();


        // ----------------------------------------------------
        // wait_group 是 per-thread async completion。
        //
        // 还需要 block barrier:
        //
        // 保证 256 threads
        // 全都已经准备好 Shared tile。
        // ----------------------------------------------------

        __syncthreads();
    }


    // ========================================================
    //
    // MAIN LOOP
    //
    // ========================================================

    for (
        int tile = 0;
        tile < num_tiles;
        ++tile
    ) {


        // ====================================================
        // Ping-Pong
        //
        // tile0:
        //
        // current = 0
        // next    = 1
        //
        //
        // tile1:
        //
        // current = 1
        // next    = 0
        //
        // ====================================================

        const int current =
            tile & 1;


        const int next =
            current ^ 1;


        const bool has_next =
            (tile + 1)
            < num_tiles;


        // ====================================================
        //
        // STEP 2
        //
        // 先发起下一块 cp.async
        //
        //
        //             Global tile next
        //
        //                    |
        //                    |
        //                  async
        //                    |
        //                    v
        //
        //             Shared[next]
        //
        //
        // 与此同时:
        //
        // Shared[current]
        //
        //       ↓
        //
        // Compute
        //
        // ====================================================

        if (has_next) {


            // ------------------------------------------------
            // Next A tile
            // ------------------------------------------------

            const float* A_src =
                A
                +
                (block_row + a_smem_row)
                    * K
                +
                (tile + 1)
                    * BLOCK_K
                +
                a_smem_col;


            float* A_dst =
                &As[
                    next
                ][
                    a_smem_row
                ][
                    a_smem_col
                ];


            cp_async_16(
                A_dst,
                A_src
            );


            // ------------------------------------------------
            // Next B tile
            // ------------------------------------------------

            const float* B_src =
                B
                +
                (
                    (tile + 1)
                    * BLOCK_K
                    + b_smem_row
                )
                    * N
                +
                block_col
                +
                b_smem_col;


            float* B_dst =
                &Bs[
                    next
                ][
                    b_smem_row
                ][
                    b_smem_col
                ];


            cp_async_16(
                B_dst,
                B_src
            );


            // ------------------------------------------------
            // 提交下一 tile 的 async copies
            // ------------------------------------------------

            cp_async_commit();
        }


        // ====================================================
        //
        // STEP 3
        //
        // Compute CURRENT Tile
        //
        //
        // 此时概念上:
        //
        //
        // Global tile next
        //
        //      |
        //      | cp.async
        //      |
        //      +-----------> Shared[next]
        //
        //
        // 同时:
        //
        //
        // Shared[current]
        //      |
        //      v
        // Registers
        //      |
        //      v
        // FMA
        //
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
            // Shared
            //    ↓
            // Registers
            // =================================================

            const float a0 =
                As[
                    current
                ][
                    thread_row + 0
                ][
                    k
                ];


            const float a1 =
                As[
                    current
                ][
                    thread_row + 1
                ][
                    k
                ];


            const float a2 =
                As[
                    current
                ][
                    thread_row + 2
                ][
                    k
                ];


            const float a3 =
                As[
                    current
                ][
                    thread_row + 3
                ][
                    k
                ];


            // =================================================
            // B:
            //
            // Shared
            //    ↓
            // Registers
            // =================================================

            const float b0 =
                Bs[
                    current
                ][
                    k
                ][
                    thread_col + 0
                ];


            const float b1 =
                Bs[
                    current
                ][
                    k
                ][
                    thread_col + 1
                ];


            const float b2 =
                Bs[
                    current
                ][
                    k
                ][
                    thread_col + 2
                ];


            const float b3 =
                Bs[
                    current
                ][
                    k
                ][
                    thread_col + 3
                ];


            // =================================================
            // 4 x 4 Register Tile
            //
            //
            //             b0   b1   b2   b3
            //
            // a0          c00  c01  c02  c03
            //
            // a1          c10  c11  c12  c13
            //
            // a2          c20  c21  c22  c23
            //
            // a3          c30  c31  c32  c33
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


        // ====================================================
        //
        // STEP 4
        //
        // 当前 tile 算完了。
        //
        // 真正准备使用下一 tile 之前:
        //
        // WAIT
        //
        // ====================================================

        if (has_next) {


            // ------------------------------------------------
            // 如果 next cp.async 还没完成:
            //
            // 在这里等。
            //
            //
            // 如果 next 已经在 compute current 期间完成:
            //
            // 基本可以直接继续。
            //
            // ------------------------------------------------

            cp_async_wait_all();


            // ------------------------------------------------
            // 确保整个 block 的 256 个线程
            //
            // 都已经完成 next shared tile
            // ------------------------------------------------

            __syncthreads();
        }
    }


    // ========================================================
    //
    // Store C
    //
    // 每个 thread:
    //
    // 4 rows
    //
    // 每行:
    //
    // 4 contiguous floats
    //
    // 所以直接 float4 store
    //
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


    *reinterpret_cast<float4*>(
        C + row0 * N + col
    ) =
        make_float4(
            c00,
            c01,
            c02,
            c03
        );


    *reinterpret_cast<float4*>(
        C + row1 * N + col
    ) =
        make_float4(
            c10,
            c11,
            c12,
            c13
        );


    *reinterpret_cast<float4*>(
        C + row2 * N + col
    ) =
        make_float4(
            c20,
            c21,
            c22,
            c23
        );


    *reinterpret_cast<float4*>(
        C + row3 * N + col
    ) =
        make_float4(
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
    // 教学版本:
    //
    // 必须满足:
    //
    // M % 64 == 0
    // N % 64 == 0
    // K % 16 == 0
    //
    // 并且 N / K 满足 float4 alignment。
    //
    // ========================================================

    constexpr int M = 512;
    constexpr int N = 512;
    constexpr int K = 512;


    if (
        M % BLOCK_M != 0
        ||
        N % BLOCK_N != 0
        ||
        K % BLOCK_K != 0
        ||
        N % 4 != 0
        ||
        K % 4 != 0
    ) {

        std::cerr
            << "Matrix dimensions do not satisfy "
            << "the teaching kernel requirements.\n";

        return 1;
    }


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
    // Launch configuration
    //
    //
    // BLOCK_N / THREAD_N
    //
    // =
    //
    // 64 / 4
    //
    // =
    //
    // 16
    //
    //
    // BLOCK_M / THREAD_M
    //
    // =
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
    //
    // ========================================================

    dim3 block(
        BLOCK_N / THREAD_N,
        BLOCK_M / THREAD_M
    );


    dim3 grid(
        N / BLOCK_N,
        M / BLOCK_M
    );


    // ========================================================
    // Warmup
    // ========================================================

    for (
        int i = 0;
        i < 10;
        ++i
    ) {

        gemmCpAsync<<<grid, block>>>(
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

        gemmCpAsync<<<grid, block>>>(
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
        / ITERATIONS;


    // ========================================================
    // GEMM FLOPs
    //
    // multiply + add
    //
    // ≈ 2 FLOPs
    // ========================================================

    const double total_flops =
        2.0
        *
        static_cast<double>(M)
        *
        static_cast<double>(N)
        *
        static_cast<double>(K);


    const double gflops =
        total_flops
        /
        (avg_ms * 1.0e6);


    // ========================================================
    // Result
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

        max_error =
            std::max(
                max_error,
                std::fabs(
                    h_C[i]
                    -
                    cpu_C[i]
                )
            );
    }


    // ========================================================
    // Output
    // ========================================================

    cudaDeviceProp prop;

    CUDA_CHECK(
        cudaGetDeviceProperties(
            &prop,
            0
        )
    );


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


    if (
        prop.major >= 8
    ) {

        std::cout
            << "cp.async hardware path: YES\n";

    } else {

        std::cout
            << "cp.async hardware path: NO "
            << "(fallback copy is used)\n";
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
        << "Block tile: "
        << BLOCK_M
        << " x "
        << BLOCK_N
        << '\n';


    std::cout
        << "Thread tile: "
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


    // ========================================================
    // Free
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