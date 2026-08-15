#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>


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

using namespace std;

__global__ void relu(float *a, float *b, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N){
        b[idx] = fmax(0.0f, a[idx]);
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
    int gridsize = CEIL(N, blocksize);

    relu<<<gridsize, blocksize>>>(d_a, d_b, N);

    checkCudaErrors(cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost));

    for(int i = 0;i < N;i++) cout << h_b[i] << ' ';
    return 0;
}