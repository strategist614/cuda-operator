/*
    reduction sum
*/
#include<cuda_runtime.h>

__global__ void reduce_sum(const float* input, float* output, int n){
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if(idx < n) sdata[tid] = input[idx];
    else sdata[tid] = 0.0f;
    __syncthreads();

    for(int stride = blockDim.x / 2;stride > 0; stride >>= 1){
        if(tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
/*
int threads = 256;

int blocks = (n + threads - 1) / threads;

reduce_sum<<<blocks, threads, threads * sizeof(float)>>>(
    input,
    partial_sum,
    n
);

reduce_sum<<<1, 256, 256 * sizeof(float)>>>(
    partial_sum,
    output,
    blocks
);
*/


__global__ void reduce_sum(const float* input, float* output, int n){
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;

    __syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if(tid < n) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if(tid == 0) atomicAdd(output, sdata[0]);
}

__device__ __forceinline__ float warpReduceSum(float val){
    for(int offset = 16;offset > 0;offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_sum(const float* input, float* input, int n){
    __shared__ float warpSums[32];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    int lane = tid % 32;
    int warp_id = tid / 32;

    float val = 0.0f;
    if(idx < n) val = input[idx];

    val = warpReduceSum(val);

    if (lane == 0) {
        warpSums[warp_id] = val;
    }

    __syncthreads();
    if(warp_id == 0){
        int numWarps = (blockDim.x + 31) / 32;

        val = (lane < numWarps)
            ? warpSums[lane]
            : 0.0f;

        val = warpReduceSum(val);

        if (lane == 0) {
            output[blockIdx.x] = val;
        }
    }
}

/*
    softmax
*/

__device__ __forceinline__ float warpReduceMax(float val){
    for(int offset = 16; offset > 0; offset>>=1){
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val){
    for(int offset = 16; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val ,offset);
    }
    return val;
}

__global__ void softmax(
    const float* input,
    float* output,
    int m,
    int n
){
    __shared__ float sdata[32];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= m) return;

    float local_max = -INFINITY;

    for (int i = tid; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, input[row * n + i]);
    }
    int warp_id = tid / 32;
    int lane = tid % 32;

    local_max = warpReduceMax(local_max);

    if(lane == 0){
        sdata[warp_id] = local_max;
    }

    __syncthreads();

    float row_max = -INFINITY;

    if(warp_id == 0){
        int numWarps = (blockDim.x + 31) / 32;
        
        if(lane < numWarps) row_max = sdata[lane];

        row_max = warpReduceMax(row_max);

        if(lane == 0){
            sdata[0] = row_max;
        }
    }
    __syncthreads();
    row_max = sdata[0];

    float local_sum = 0.0f;

    for(int i = tid;i < n;i += blockDim.x){
        local_sum += expf(input[row * n + i] - row_max);
    }

    local_sum = warpReduceSum(local_sum);

    if(lane == 0){
        sdata[warp_id] = local_sum;
    }
    __syncthreads();
    float row_sum = 0.0f;
    if(warp_id == 0){
        int numWarps = (blockDim.x + 31) / 32;
        if(lane < numWarps) row_sum = sdata[lane];
        row_sum = warpReduceSum(row_sum);
        if(lane == 0){
            sdata[0] = row_sum;
        }
    }

    __syncthreads();

    row_sum = sdata[0];

    for(int i = tid; i < n; i += blockDim.x){
        output[row * n + i] =
            expf(input[row * n + i] - row_max)
            / row_sum;
    }

}

/*
    online softmax
*/

__device__ __forceinline__ void warpOnlineReduce(float& m, float& l){
    for(int offset = 16; offset > 0; offset >> =1){
        float other_m = __shfl_down_sync(0xffffffff, m, offset);

        float other_l = __shfl_down_sync(0xffffffff, l, offset);

        float new_m = fmaxf(m, other_m);

        l = l * expf(m - new_m) + other_l * expf(other_m - new_m);
        m = new_m;
    }
}

__global__ void online_softmax(const float* input, float* output, int m, int n){
    __shared__ float max[32];
    __shared__ float ssum[32];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    int lane = tid % 32;
    int warp_id = tid / 32;

    if(row >= m) return;
    
    float local_max = -INFINITY;
    float local_sum = 0.0f;

    for(int i = tid;i <= n;i += blockDim.x){
        float x = input[row * n + i];

        float new_max = fmaxf(local_max, x);

        local_sum = local_sum * expf(local_max - new_max) + expf(x - new_max);

        local_max = new_max;
    }

    warpOnlineReduce(local_max, local_sum);

    if(lane == 0){
        smax[warp_id] = local_max;
        ssum[warp_id] = local_sum;
    }

    __syncthreads();

    float row_max = -INFINITY;
    float row_sum = 0.0f;

    if(warp_id == 0){
        int numWarps = (blockDim + 31) / 32;
        if(lane < numWarps){
            row_max = smax[lane];
            row_sum = ssum[lane];
        }

        warpOnlineReduce(row_max, row_sum);

        if(lane == 0){
            smax[0] = row_max;
            ssum[0] = row_sum;
        }
    }

    __syncthreads();

    row_max = smax[0];
    row_sum = ssum[0];

    for (int i = tid; i < n; i += blockDim.x) {
        float x = input[row * n + i];

        output[row * n + i] =
            expf(x - row_max) / row_sum;
    }
}

/*
    layernorm
*/

__device__ float warpReduceSum(float val){
    for(int offset = 16;offset >0;offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void layernorm(const float* input, const float* gamma, const float* beta, float* output, int m,int n,float eps){
    __shared__ float sdata[32];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int lane = tid % 32;
    const int warp_id = tid / 32;

    float local_sum = 0.0f;

    for(int i = tid;i < n;i += blockDim.x){
        float x = input[row * n + i];

        local_sum += x;
    }

    local_sum = warpReduceSum(local_sum);

    if(lane == 0){
        sdata[warp_id] = local_sum;
    }

    __syncthreads();
    
    float row_sum = 0.0f;
    
    if(warp_id == 0){
        int numWarps = (blockDim.x + 31) / 32;

        if(lane < numWarps) row_sum = sdata[lane];

        row_sum = warpReduceSum(row_sum);

        if(lane == 0){
            sdata[0] = row_sum;
        }
    }
    __syncthreads();

    float mean = sdata[0] / n;

    float local_var = 0.0f;
    for(int i =tid;i < n;i+=blockDim.x){
        float x = input[row * n + i];
        float diff = x - mean;
        local_var += diff * diff;
    }

    local_var = warpReduceSum(local_var);

    if(lane == 0){
        sdata[warp_id] = local_var;
    }

    __syncthreads();
    float row_var = 0.0f;
    if(warp_id == 0){
        int numWarps = (blockDim.x + 31) /32;
        if(lane < numWarps) row_var = sdata[lane];

        row_var = warpReduceSum(row_var);

        if(lane == 0){
            sdata[0] = row_var;
        }
    }

    __syncthreads();
    float var = sdata[0] / n;

    float inv_std = rsqrtf(var + eps);

    for (int i = tid; i < n; i += blockDim.x) {

        float x = input[row * n + i];

        float normalized =
            (x - mean) * inv_std;

        output[row * n + i] =
            normalized * gamma[i] + beta[i];
    }
}  

/*
    rmsnorm
*/

__device__ float warpReduceSum(float val){
    for(int offset = 16;offset > 0;offset>>=1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void Rmsnorm(const float* input, const float* gamma, float* output, int m, int n, float eps){
    __shared__ float sdata[32];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    const int warp_id = tid / 32;
    const int lane = tid % 32;

    float local_sum = 0.0f;

    for(int i = tid;i < n;i += blockDim.x){
        local_sum += input[row * n + i] * input[row * n + i];
    }

    local_sum = warpReduceSum(local_sum);

    if(lane == 0){
        sdata[warp_id] = local_sum;
    }
    __syncthreads();
    float row_sum = 0.0f;
    if(warp_id == 0){
        int numWarps = (blockDim.x + 31) /32;
        if(lane < numWarps) row_sum = sdata[lane];

        row_sum = warpReduceSum(row_sum);
        if(lane == 0){
            sdata[0] = row_sum;
        }
    }
    __syncthreads();

    float var = sdata[0] / n;
    float inv_std = rsqrtf(var + eps);
    for(int i =tid;i < n;i+=blockDim.x){
        float x = input[row * n + i];
        output[row * n + i] = x * inv_std * gamma[i];
    }
}

/*
    matrix transpose
*/

__global__ void transpose_naive(const float* input, float* output, int m, int n){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < n && y < m){
        output[x * m + y] = input[y * n + x];
    }
}

__global__ void transpose_tile(const float* input, float* output, int m, int n){
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    __shared__ float tile[32][33];

    if(x < n && y < m){
        tile[y][x] = input[y * n + x];
    }

    __syncthreads();
    
    int out_x = blockIdx.y * 32 + threadIdx.x;
    int out_y = blockIdx.x * 32 + threadIdx.y;

    if(out_x < m && out_y < n){
        output[out_y * m + out_x] = tile[threadIdx.x][threadIdx.y];
    }
}

/*
    gemm
*/

__global__ void gemm_naive(const float* A, const float* B, float* C, int m, int n,int k_size){
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < m && col < n){
        float sum = 0.0f;
        for(int k = 0;k < k_size;k++){
            sum += A[row * k_size + k] * B[k * n + col];
        }

        C[row * n + col] = sum;
    }
}

__global__ void gemm_shared_tile(const float* A, const float* B, float* C,int m,int n,int k_size){
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = TILE * blockIdx.y + ty;
    const int col = TILE * blockIdx.x + tx;

    float sum = 0.0f;
    for(int bk = 0;bk < k_size;bk += TILE){
        const int a_col = tx + bk;
        const int b_row = ty + bk;

        As[ty][tx] = (row < m && a_col < k_size) ? A[row * k_size + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < k_size && col < n) ? B[b_row * n + col] : 0.0f;

        __syncthreads();

#pragma unroll
        for(int k = 0;k < TILE;k ++) sum += As[ty][k] * Bs[k][tx];
        __syncthreads(); 
    }

    if(row < m && col < n) C[row * n + col] = sum;
}

__global__ void gemm_register_tile(const float* A, const float *B, float* C,int m,int n,int k_size){
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32; 
    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;


    const int lane_row = lane / 4;
    const int lane_col = lane % 4;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[TM][TN] = {};
    for(int bk = 0;bk < k_size;bk+=BK){
        for(int i = tid;i < BM * BK;i+=blockDim.x){
            const int r = i / BK;
            const int c = i % BK;

            const int global_row = block_row + r;
            const int global_col = bk + c;

            As[r][c] = (global_row < m && global_col < k_size) ? A[global_row * k_size + global_col] : 0.0f;
        }

        for(int i = tid;i < BK *BN;i+=blockDim.x){
            const int r = i / BN;
            const int c = i % BN;

            const int global_row = bk + r;
            const int global_col = block_col + c;

            Bs[r][c] = (global_row < k_size&&global_col < n) ? B[global_row*n+global_col] : 0.0f;
        }

        __syncthreads();

        for(int k = 0;k < BK;++k){
            float a_frag[TM];
            float b_frag[TN];
            for(int i = 0;i < TM;i++){
                const int row = warp_row * WM + lane_row * TM + i;
                a_frag[i] = As[row][k];
            }
            for(int j = 0;j < TN;j++){
                const int col = warp_col * WN + lane_col * TN + j;
                b_frag[j] = Bs[k][col];
            }

            for(int i = 0;i < TM;i++){
                for(int j = 0;j < TN;j++){
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        __syncthreads();
    }

    for(int i = 0;i < TM;i++){
        for(int j = 0;j < TN;j++){
            const int row = block_row + warp_row * WM + lane_row * TM + i;
            const int col = block_col + warp_col * WN + lane_col * TN + j;
            if(row < m && col < n) C[row * n + col] = acc[i][j];
        }
    }

}

/* 
    prefix sum/scan
*/

__global__ void scan(const float* input, float* output, int n){
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;

    __syncthreads();

    for(int offset = 1;offset < blockDim.x;offset <<= 1){
        float val = 0.0f;
        if(tid >= offset){
            val = sdata[tid - offset];
        }

        __syncthreads();
        if(tid >= offset){
            sdata[tid] += val;
        }

        __syncthreads();
    }
    if(idx < n) output[idx] = sdata[tid];
}

/*
    Histogram
*/

__global__ void histogram_naive(const int* input, int* count, int n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < n){
        int value = input[idx];

        atomicAdd(&count[value], 1);
    }
}

__global__ void histogram_shared(const int *input, int* count, int n){
    __shared__ int local_count[10];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if(tid < 10) local_count[tid] = 0;

    __syncthreads();

    if(idx < n){
        int value = input[idx];
        atomicAdd(&local_count[value], 1);
    }

    __syncthreads();

    if(tid < 10){
        atomicAdd(&count[tid], local_count[tid]);
    }
}

/*
    top-k
*/
__device__ float warpReduceMax(float val){
    for(int offset =16;offset > 0;offset >>= 1){
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ void warp_bitonic_sort(float &value, int& index, int lane){
    for(int size = 2;size <= 32;size <<= 1){
        for(int stride = size >> 1;stride >0;stride >>= 1){
            float other_value = __shfl_xor_sync(0xffffffff, value, stride);

            int other_index = __shfl_xor_sync(0xffffffff, index, stride);

            bool other_is_better = better(other_value, other_index, value, index);

            bool lower_lane = (lane & stride) == 0;
            bool descending = (lane & size) == 0;
            bool keep_better = lower_lane == descending;

            if (keep_better)
            {
                if (other_is_better)
                {
                    value =
                        other_value;

                    index =
                        other_index;
                }
            }
            else
            {
                if (!other_is_better)
                {
                    value =
                        other_value;

                    index =
                        other_index;
                }
            }
        }
    }
}

__global__ void top_kernel(const float* input, float* out_val, int* out_idx, int B, int N){
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    float topk_value = -INFINITY;
    int topk_index = -1;

    for(int base = warp_id*32; base < N; base+= blockDim.x){
        int idx = base + lane;
        float x = (idx < N) ? input[row * N + idx] : -INFINITY;

        float batch_max = warpReduceMax(x);
        batch_max = __shfl_sync(0xffffffff, batch_max, 0);
        
        float threshold = __shfl_sync(0xffffffff, topk_value, K - 1);

        if(batch_max <= threshold) continue;

        warp_bitonic_sort(x, x_idx, lane);

        topk_value = warp_topk_merge(topk_value, topk_index, x, x_idx, lane);
    }
}