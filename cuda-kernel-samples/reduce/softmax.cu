#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

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

__device__ float warp_reduce_max(float val){
    for(int offset = 16;offset > 0;offset >>= 1){
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float warp_reduce_sum(float sum){
    for(int offset = 16;offset >0;offset >>= 1){
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    return sum;
}

__global__ void softmax(const float * x, float * y, int n){
    __shared__ float warp_data[32];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid / 32;

    int num_warps = (blockDim.x + 31) / 32;

    float local_max = -INFINITY;
    for(int i = tid; i < n;i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }

    local_max = warp_reduce_max(local_max);

    if(lane == 0){
        warp_data[warp_id] = local_max;
    }

    __syncthreads();

    float max_val = -INFINITY;
    if(warp_id == 0){
        max_val = (lane < num_warps) ? warp_data[lane] : -INFINITY;

        max_val = warp_reduce_max(max_val);

        if(lane == 0) warp_data[0] = max_val;
    }
    __syncthreads();

    max_val = warp_data[0];

    float local_sum = 0.0f;

    for(int i = tid; i < n;i += blockDim.x){
        local_sum += expf(x[i] - max_val);
    }

    local_sum = warp_reduce_sum(local_sum);

    if(lane == 0) warp_data[warp_id] = local_sum;

    __syncthreads();

    float sum = 0.0f;

    if (warp_id == 0) {

        sum =
            (lane < num_warps)
            ? warp_data[lane]
            : 0.0f;

        sum = warp_reduce_sum(sum);

        if (lane == 0) {
            warp_data[0] = sum;
        }
    }

    __syncthreads();

    sum = warp_data[0];

    for (int i = tid; i < n; i += blockDim.x) {

        y[i] =
            expf(x[i] - max_val) / sum;
    }
}

int main()
{
    const int N = 1024;
    const size_t bytes = N * sizeof(float);
    float *h_x = new float[N];
    float *h_y = new float[N];

    for(int i = 0;i < N;i ++) h_x[i] = static_cast<float>(i % 10);

    float *d_x = nullptr;
    float *d_y = nullptr;

    checkCudaErrors(cudaMalloc(&d_x, bytes));
    checkCudaErrors(cudaMalloc(&d_y, bytes));

    checkCudaErrors(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    int blocksize = 256;

    softmax<<<1, blocksize>>>(d_x, d_y, N);

    checkCudaErrors(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    for(int i = 0;i < 10;i ++) cout << h_y[i] << ' ';
    return 0;
}