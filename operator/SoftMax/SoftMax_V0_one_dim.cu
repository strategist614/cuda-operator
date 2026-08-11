#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>

using namespace std;

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr                                         \
                << "CUDA error: "                             \
                << cudaGetErrorString(err)                    \
                << " at " << __FILE__                         \
                << ":" << __LINE__                            \
                << std::endl;                                 \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)


__global__ void softmax1D(
    const float* input,
    float* output,
    int n
){
    extern __shared__ float shared[];

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    float max_val = -INFINITY;
    
    for(auto i = tid; i < n; i += block_size){
        max_val = fmaxf(max_val, input[i]);
    }

    shared[tid] = max_val;
    __syncthreads();

    for(int offset = block_size / 2; offset > 0; offset /= 2){
        if(tid < offset){
            shared[tid] = fmaxf(shared[tid], shared[tid + offset]);
        }
        __syncthreads();
    }

    const float global_max = shared[0];

    float sum_exp = 0.0f;

    for(auto i = tid; i < n; i += block_size){
        const float value = expf(input[i] - global_max);

        output[i] = value;
        sum_exp += value;
    }

    shared[tid] = sum_exp;
    __syncthreads();

    for(int offset = block_size / 2; offset > 0; offset /= 2){
        if(tid < offset){
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

     const float sum = shared[0];

     const float inv_sum = 1.0f / sum;

     for(auto i = tid; i < n; i += block_size){
         output[i] *= inv_sum;
     }
}

void softmaxCPU(
    const std::vector<float>& input,
    std::vector<float>& output) {

    const int n =
        static_cast<int>(input.size());

    output.resize(n);

    float max_val = -INFINITY;

    for (int i = 0; i < n; ++i) {
        max_val =
            std::max(max_val, input[i]);
    }

    float sum = 0.0f;

    for (int i = 0; i < n; ++i) {

        output[i] =
            std::exp(input[i] - max_val);

        sum += output[i];
    }

    for (int i = 0; i < n; ++i) {
        output[i] /= sum;
    }
}


// ============================================================
// main
// ============================================================

int main() {

    // --------------------------------------------------------
    // 输入
    // --------------------------------------------------------

    std::vector<float> h_input = {
        1.0f,
        2.0f,
        3.0f,
        4.0f
    };

    const int n =
        static_cast<int>(h_input.size());

    std::vector<float> h_output(n);

    // --------------------------------------------------------
    // GPU memory
    // --------------------------------------------------------

    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            n * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_output,
            n * sizeof(float)
        )
    );

    // --------------------------------------------------------
    // Host -> Device
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            n * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    // --------------------------------------------------------
    // Kernel launch
    //
    // 一维 Softmax:
    // 一个 block 处理整个 vector
    // --------------------------------------------------------

    const int threads = 256;

    const int blocks = 1;

    const size_t shared_memory =
        threads * sizeof(float);

    softmax1D<<<
        blocks,
        threads,
        shared_memory
    >>>(
        d_input,
        d_output,
        n
    );

    CUDA_CHECK(
        cudaGetLastError()
    );

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );

    // --------------------------------------------------------
    // Device -> Host
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            h_output.data(),
            d_output,
            n * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    // --------------------------------------------------------
    // Print CUDA result
    // --------------------------------------------------------

    std::cout << "CUDA Softmax:\n";

    for (float value : h_output) {
        std::cout << value << " ";
    }

    std::cout << "\n";

    // --------------------------------------------------------
    // CPU reference
    // --------------------------------------------------------

    std::vector<float> cpu_output;

    softmaxCPU(
        h_input,
        cpu_output
    );

    std::cout << "CPU Softmax:\n";

    for (float value : cpu_output) {
        std::cout << value << " ";
    }

    std::cout << "\n";

    // --------------------------------------------------------
    // Free GPU memory
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaFree(d_input)
    );

    CUDA_CHECK(
        cudaFree(d_output)
    );

    return 0;
}