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
    float* __restrict__ output,
    int height,
    int width,
    float mean0,
    float mean1,
    float mean2,
    float std0,
    float std1,
    float std2
) {
    // 当前第几个像素了 每个线程处理当前哪个像素
    // 每个线程负责一个像素的三个通道
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int hw = height * width;
    // 线程数通常会比真实像素多一点
    if (idx >= hw) {
        return;
    }

    // idx 对应 CHW 里面的空间位置 h * W + w
    // HWC 里面一个像素有 3 个通道，所以输入起点是 idx * 3
    int in_base = idx * 3;

    unsigned char r = input[in_base + 0];
    unsigned char g = input[in_base + 1];
    unsigned char b = input[in_base + 2];
    // 强制转换
    float rf = static_cast<float>(r) / 255.0f;
    float gf = static_cast<float>(g) / 255.0f;
    float bf = static_cast<float>(b) / 255.0f;

    /*
        CHW: channel plane 是连续的
        三个通道是分开存的
        R 平面: output[0 ... hw-1]
        G 平面: output[hw ... 2*hw-1]
        B 平面: output[2*hw ... 3*hw-1]
    */
    output[0 * hw + idx] = (rf - mean0) / std0;
    output[1 * hw + idx] = (gf - mean1) / std1;
    output[2 * hw + idx] = (bf - mean2) / std2;
}


void launch_hwc_to_chw_norm(
    const unsigned char* d_input,
    float* d_output,
    int height,
    int width,
    const float mean[3],
    const float std[3],
    cudaStream_t stream = 0
) {
    int hw = height * width;

    int block = 256;
    int grid = (hw + block - 1) / block;

    hwc_to_chw_norm_kernel<<<grid, block, 0, stream>>>(
        d_input,
        d_output,
        height,
        width,
        mean[0],
        mean[1],
        mean[2],
        std[0],
        std[1],
        std[2]
    );

    CHECK_CUDA(cudaGetLastError());
}


// CPU 版本，用来验证 CUDA 结果
void cpu_hwc_to_chw_norm(
    const std::vector<unsigned char>& input,
    std::vector<float>& output,
    int height,
    int width,
    const float mean[3],
    const float std[3]
) {
    int hw = height * width;

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            int pixel_idx = h * width + w;

            for (int c = 0; c < 3; ++c) {
                int in_idx = (h * width + w) * 3 + c;
                int out_idx = c * hw + pixel_idx;

                float value = static_cast<float>(input[in_idx]) / 255.0f;
                output[out_idx] = (value - mean[c]) / std[c];
            }
        }
    }
}


int main() {
    int height = 4;
    int width = 5;
    int channels = 3;

    int input_size = height * width * channels;
    int output_size = channels * height * width;

    std::vector<unsigned char> h_input(input_size);
    std::vector<float> h_output(output_size);
    std::vector<float> h_ref(output_size);

    // 构造一个假图像
    for (int i = 0; i < input_size; ++i) {
        h_input[i] = static_cast<unsigned char>(i % 256);
    }

    // 常见 ImageNet normalize 参数
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3]  = {0.229f, 0.224f, 0.225f};

    unsigned char* d_input = nullptr;
    float* d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, input_size * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_output, output_size * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(
        d_input,
        h_input.data(),
        input_size * sizeof(unsigned char),
        cudaMemcpyHostToDevice
    ));

    launch_hwc_to_chw_norm(
        d_input,
        d_output,
        height,
        width,
        mean,
        std
    );

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(
        h_output.data(),
        d_output,
        output_size * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    cpu_hwc_to_chw_norm(
        h_input,
        h_ref,
        height,
        width,
        mean,
        std
    );

    float max_error = 0.0f;

    for (int i = 0; i < output_size; ++i) {
        float error = std::fabs(h_output[i] - h_ref[i]);
        if (error > max_error) {
            max_error = error;
        }
    }

    std::cout << "Max error: " << max_error << std::endl;

    std::cout << "\nOutput CHW tensor:" << std::endl;

    int hw = height * width;

    for (int c = 0; c < 3; ++c) {
        std::cout << "Channel " << c << ":" << std::endl;

        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int idx = c * hw + h * width + w;
                std::cout << h_output[idx] << " ";
            }
            std::cout << std::endl;
        }
    }

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}