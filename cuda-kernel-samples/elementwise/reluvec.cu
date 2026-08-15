#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

using namespace std;

// ============================================================

#define checkCudaErrors(call)                                      \
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
// 1. 向上取整
#define CEIL(a, b) (((a) + (b) - 1) / (b))

// 2. FLOAT4，用于向量化访存
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])


__global__ void relu_4float(float *a, float *b, int N){
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if(idx + 3 < N){
        float4 a4 = FLOAT4(a[idx]);
        float4 b4;

        b4.x = fmaxf(0.0f, a4.x);
        b4.y = fmaxf(0.0f, a4.y);
        b4.z = fmaxf(0.0f, a4.z);
        b4.w = fmaxf(0.0f, a4.w);

        FLOAT4(b[idx]) = b4;
    }
}
int main()
{
    int N = 1024;
    const size_t bytes = N * sizeof(float);
    float *h_a = new float[N];
    float *h_b = new float[N];

    for(int i = 0;i < N;i++) h_a[i] = static_cast<float>(i);

    float *d_a = nullptr;
    float *d_b = nullptr;

    checkCudaErrors(cudaMalloc(&d_a, bytes));
    checkCudaErrors(cudaMalloc(&d_b, bytes));

    checkCudaErrors(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    int blocksize = 256;
    int num = CEIL(N, 4);
    int gridsize = CEIL(num, blocksize);

    relu_4float<<<gridsize, blocksize>>>(d_a, d_b, N);

    checkCudaErrors(cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost));

    for(int i = 0;i < N;i++) cout << h_b[i] << ' ';

    return 0;
}