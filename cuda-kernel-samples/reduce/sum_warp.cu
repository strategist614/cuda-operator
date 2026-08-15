#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

using namespace std;

#define checkCudaErrors(call)                                  \
    do {                                                       \
        cudaError_t err = (call);                              \
        if (err != cudaSuccess) {                              \
            std::cerr << "CUDA Error: "                        \
                      << cudaGetErrorString(err)                \
                      << " at "                                 \
                      << __FILE__                               \
                      << ":"                                    \
                      << __LINE__                               \
                      << std::endl;                             \
            std::exit(EXIT_FAILURE);                           \
        }                                                      \
    } while (0)

#define CEIL(a, b) (((a) + (b) - 1) / (b))

__global__ void reduce_sum_float4(
    const float* __restrict__ a,
    float* __restrict__ out,
    int N)
{
    __shared__ float warp_sum[32];
    // 0~262143
    int tid =
        blockDim.x * blockIdx.x +
        threadIdx.x;

    int total_threads =
        blockDim.x * gridDim.x;

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    float sum = 0.0f;

    // ==========================================
    // 每 4 个 float 看成一个 float4
    // ==========================================

    int N4 = N / 4;
    // grid-stride loop
    // 先计算每个线程自己负责的 float4 的 local sum
    for (int i = tid; i < N4; i += total_threads) {

        float4 x =
            reinterpret_cast<const float4*>(a)[i];

        sum += x.x + x.y + x.z + x.w;
    }

    // ==========================================
    // 处理最后 N % 4 个元素
    // ==========================================

    int tail = N4 * 4;

    for (int i = tail + tid;
         i < N;
         i += total_threads)
    {
        sum += a[i];
    }

    // ==========================================
    // warp reduction
    // ==========================================

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(
            0xffffffff,
            sum,
            offset
        );
    }

    // 每个 warp 的 lane 0 保存结果
    if (lane == 0) {
        warp_sum[warp_id] = sum;
    }

    __syncthreads();

    // ==========================================
    // 第一个 warp reduction 所有 warp sums
    // ==========================================

    int num_warps =
        (blockDim.x + 31) / 32;

    if (warp_id == 0) {

        sum =
            (lane < num_warps)
            ? warp_sum[lane]
            : 0.0f;

        for (int offset = 16;
             offset > 0;
             offset >>= 1)
        {
            sum += __shfl_down_sync(
                0xffffffff,
                sum,
                offset
            );
        }

        // 只有 thread 0 写 block result
        if (lane == 0) {
            out[blockIdx.x] = sum;
        }
    }
}


int main()
{

    int N = 1 << 20;   // 1,048,576

    size_t bytes =
        N * sizeof(float);

    float* h_a =
        new float[N];

    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f;
    }

    float* d_a = nullptr;

    checkCudaErrors(
        cudaMalloc(&d_a, bytes)
    );

    checkCudaErrors(
        cudaMemcpy(
            d_a,
            h_a,
            bytes,
            cudaMemcpyHostToDevice
        )
    );

    int blocksize = 256;

    int num_threads =
        CEIL(N, 4);

    int gridsize =
        CEIL(num_threads, blocksize);

    gridsize = std::min(gridsize, 1024);

    cout << "first gridsize = "
         << gridsize
         << endl;

    float* d_partial = nullptr;

    checkCudaErrors(
        cudaMalloc(
            &d_partial,
            gridsize * sizeof(float)
        )
    );
    // 1024个 block
    reduce_sum_float4<<<
        gridsize,
        blocksize
    >>>(
        d_a,
        d_partial,
        N
    );

    checkCudaErrors(
        cudaGetLastError()
    );

    checkCudaErrors(
        cudaDeviceSynchronize()
    );

    float* d_result = nullptr;

    checkCudaErrors(
        cudaMalloc(
            &d_result,
            sizeof(float)
        )
    );

    reduce_sum_float4<<<
        1,
        blocksize
    >>>(
        d_partial,
        d_result,
        gridsize
    );

    checkCudaErrors(
        cudaGetLastError()
    );

    checkCudaErrors(
        cudaDeviceSynchronize()
    );

    float result = 0.0f;

    checkCudaErrors(
        cudaMemcpy(
            &result,
            d_result,
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    cout << "GPU result = "
         << result
         << endl;

    cout << "Expected   = "
         << static_cast<float>(N)
         << endl;

    cudaFree(d_a);
    cudaFree(d_partial);
    cudaFree(d_result);

    delete[] h_a;

    return 0;
}