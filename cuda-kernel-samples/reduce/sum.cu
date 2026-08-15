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

__global__ void add(float *a, float *b, float *c, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        c[idx] = a[idx] + b[idx];
    }
}
int main()
{
    int N = 1024;
    const size_t bytes = N * sizeof(float);
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    checkCudaErrors(cudaMalloc(&d_a, bytes));
    checkCudaErrors(cudaMalloc(&d_b, bytes));
    checkCudaErrors(cudaMalloc(&d_c, bytes));

    for(int i = 0;i < N;i++) h_a[i] = static_cast<float>(i), h_b[i] = static_cast<float>(2 * i);

    checkCudaErrors(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int blocksize = 256;
    int gridsize = CEIL(N, blocksize);

    add<<<gridsize, blocksize>>>(d_a, d_b, d_c, N);

    checkCudaErrors(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    for(int i = 0;i < 10;i ++) cout << h_c[i] <<' ';
    return 0;
}