#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                             \
    do {                                                             \
        cudaError_t err = call;                                      \
        if (err != cudaSuccess) {                                    \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)   \
                      << " at " << __FILE__ << ":" << __LINE__       \
                      << std::endl;                                  \
            std::exit(EXIT_FAILURE);                                 \
        }                                                            \
    } while (0)

// 输入:  HWC, uint8,  shape = [H, W, 3]
// 输出:  CHW, float, shape = [3, H, W]
//
// out[c, h, w] = (in[h, w, c] / 255.0 - mean[c]) / std[c]
__global__ void hwc_to_chw_norm_kernel(
    const unsigned char* __restrict__ input,
    float * __restrict__ output,
    int height,
    int width,
    float mean0,
    float mean1,
    float std0,
    float std1,
    float std2
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int hw = height * width;

    if(idx >= hw){
        return;
    }


}