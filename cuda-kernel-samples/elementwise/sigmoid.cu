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

using namespace std;

__global__ void sigmoid(float *a, float *b, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = a[idx];

        if (x >= 0.0f) {
            b[idx] = 1.0f / (1.0f + expf(-x));
        } else {
            float e = expf(x);
            b[idx] = e / (1.0f + e);
        }
    }
}

int main(){
    int N = 1024;
    const size_t bytes = N * sizeof(float);
    float *h_a = new float[N];
    float *h_b = new float[N];

    for(int i = 0;i < N;i ++) h_a[i] = i;

    float *d_a = nullptr;
    float *d_b = nullptr;

    checkCudaErrors(cudaMalloc(&d_a, bytes));
    checkCudaErrors(cudaMalloc(&d_b, bytes));

    checkCudaErrors(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    int blocksize = 256;
    int gridsize = (N + blocksize - 1) / blocksize;

    sigmoid<<<gridsize, blocksize>>>(d_a, d_b, N);

    checkCudaErrors(cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost));

    for(int i = 0;i < 10;i ++) cout << h_b[i] << " ";
    return 0;
}

