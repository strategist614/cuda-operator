#include "norm_test_utils.cuh"

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void rmsnorm_float4(const float* __restrict__ x,
                               const float* __restrict__ gamma,
                               float* __restrict__ y,
                               int hidden_size,
                               float epsilon) {
    __shared__ float warp_square_sum[32];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int warp_count = blockDim.x >> 5;
    const int hidden4 = hidden_size >> 2;
    const int row4_offset = blockIdx.x * hidden4;

    const float4* x4 = reinterpret_cast<const float4*>(x);
    float local_square_sum = 0.0f;
    for (int col4 = tid; col4 < hidden4; col4 += blockDim.x) {
        const float4 value = x4[row4_offset + col4];
        local_square_sum = fmaf(value.x, value.x, local_square_sum);
        local_square_sum = fmaf(value.y, value.y, local_square_sum);
        local_square_sum = fmaf(value.z, value.z, local_square_sum);
        local_square_sum = fmaf(value.w, value.w, local_square_sum);
    }

    local_square_sum = warp_reduce_sum(local_square_sum);
    if (lane == 0) {
        warp_square_sum[warp_id] = local_square_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_square_sum =
            lane < warp_count ? warp_square_sum[lane] : 0.0f;
        local_square_sum = warp_reduce_sum(local_square_sum);
        if (lane == 0) {
            warp_square_sum[0] =
                rsqrtf(local_square_sum / hidden_size + epsilon);
        }
    }
    __syncthreads();

    const float inverse_rms = warp_square_sum[0];
    const float4* gamma4 = reinterpret_cast<const float4*>(gamma);
    float4* y4 = reinterpret_cast<float4*>(y);
    for (int col4 = tid; col4 < hidden4; col4 += blockDim.x) {
        const float4 value = x4[row4_offset + col4];
        const float4 scale = gamma4[col4];
        float4 output;
        output.x = value.x * inverse_rms * scale.x;
        output.y = value.y * inverse_rms * scale.y;
        output.z = value.z * inverse_rms * scale.z;
        output.w = value.w * inverse_rms * scale.w;
        y4[row4_offset + col4] = output;
    }
}

int main() {
    auto launch = [](const float* x,
                     const float* gamma,
                     const float*,
                     float* y) {
        rmsnorm_float4<<<norm_test::ROWS, norm_test::THREADS>>>(
            x, gamma, y, norm_test::HIDDEN, norm_test::EPSILON);
    };

    return norm_test::run("rmsnorm_float4_warp_fp32", 2,
                          norm_test::rmsnorm_reference, launch);
}
