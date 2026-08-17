/*
    naive sum
*/

// __global__ void add(const float* a, float *out, int n){
//     int tid = threadIdx.x;
//     int idx = blockDim.x * blockIdx.x + tid;

//     __shared__ float _sdata[256];

//     _sdata[tid] = (idx < n) ? a[idx] : 0.0f;

//     for(int offset = blockDim.x / 2; offset > 0; offset >>= 1){
//         _sdata[tid] = _sdata[tid + offset];
//     }

//     if(tid == 0) atomicAdd(out, _sdata[0]);
// }

#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>

using namespace std;

#define checkCudaErrors(call)                                  \
    do {                                                       \
        cudaError_t err = (call);                              \
        if (err != cudaSuccess) {                              \
            std::cerr << "CUDA Error: "                        \
                      << cudaGetErrorString(err)                \
                      << " at " << __FILE__                     \
                      << ":" << __LINE__                        \
                      << std::endl;                             \
            std::exit(EXIT_FAILURE);                           \
        }                                                      \
    } while (0)

/*
    warp shuffle sum
*/

__global__ void add_warp_shuffle(
    const float* a,
    float* out,
    int n)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    __shared__ float warp_sum[32];

    int lane = tid % 32;
    int warp_id = tid / 32;

    // 每个线程先读取一个元素
    float sum = (idx < n) ? a[idx] : 0.0f;

    // 每个 warp 内部 reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(
            0xffffffff,
            sum,
            offset
        );
    }

    // 每个 warp 的 lane0 保存 warp sum
    if (lane == 0) {
        warp_sum[warp_id] = sum;
    }

    __syncthreads();

    // warp0 对所有 warp sum 再 reduction
    if (warp_id == 0) {

        int num_warps =
            (blockDim.x + 31) / 32;

        sum = (lane < num_warps)
            ? warp_sum[lane]
            : 0.0f;

        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(
                0xffffffff,
                sum,
                offset
            );
        }

        // 每个 block 的最终结果加到 out[0]
        if (lane == 0) {
            atomicAdd(out, sum);
        }
    }
}


int main()
{
    int N = 1024;
    size_t bytes = N * sizeof(float);

    float* h_a = new float[N];

    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
    }

    float* d_a = nullptr;
    float* d_out = nullptr;

    checkCudaErrors(
        cudaMalloc(&d_a, bytes)
    );

    checkCudaErrors(
        cudaMalloc(&d_out, sizeof(float))
    );

    // atomicAdd 必须从 0 开始
    checkCudaErrors(
        cudaMemset(d_out, 0, sizeof(float))
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
    int gridsize =
        (N + blocksize - 1) / blocksize;

    add_warp_shuffle<<<gridsize, blocksize>>>(
        d_a,
        d_out,
        N
    );

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    float h_out = 0.0f;

    checkCudaErrors(
        cudaMemcpy(
            &h_out,
            d_out,
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    cout << "GPU result = " << h_out << endl;

    cout << "Expected = "
         << (1023.0f * 1024.0f / 2.0f)
         << endl;

    cudaFree(d_a);
    cudaFree(d_out);

    delete[] h_a;

    return 0;
}