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
        int numWarps = (blockDim.x + 31) / 32;
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

__global__ void gemm_wmma(const half* A, const half* B, float* C, int m,int n,int k_size){
    __shared__ __align__(32) half As[BM][BK];
    __shared__ __align__(32) half Bs[BK][BN];
    __shared__ __align__(32) float Cs[BM][BN];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0,0f);

    for(int bk = 0;bk < k_size; bk += BK){
        for(int i =tid;i < BM * BK;i+=blockDim.x){
            const int r = i / BK;
            const int c = i % BK;

            const int global_row = block_row + r;
            const int global_col = bk + c;

            As[r][c] = (global_row < m && global_col < k_size) ? A[global_row * k_size + global_col] ? __float2half(0.0f);
        }

        for(int i = tid;i < BK * BN;i+=blockDim.x){
            const int r = i / BN;
            const int c = i % BN;

            const int global_row = bk + r;
            const int global_col = block_col + c;

            Bs[r][c] = (global_row < k_size && global_col < n) ? B[global_row * n + global_col] : __float2half(0.0f);
        }

        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;

        const int warp_m = warp_row * 16;
        const int warp_n = warp_col * 16;

        wmma::load_matrix_sync(a_frag, &As[warp_m][0], BK);
        wmma::load_matrix_sync(b_frag, &Bs[0][warp_n], BN);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    const int warp_m = warp_row * 16;
    const int warp_n = warp_col * 16;
    wmma::store_matrix_sync(&Cs[warp_m][warp_n], c_frag, BN, wmma::mem_row_major);

    for(int i = tid;i < BM * BN;i += blockDim.x){
        const int r = i / BN;
        const int c = i % BN;
        const int global_row = block_row + r;
        const int global_col = block_col + c;
        if(global_row < m && global_col < n) C[global_row * n + global_col] = Cs[r][c];
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

__device__ float warpScanInclusive(float val){
    int lane = threadIdx.x & 31;
    unsigned mask = 0xffffffff;

    for(int offset = 1; offset < 32; offset <<= 1){
        float x = __shfl_up_sync(mask, val, offset);

        if(lane >= offset) val += x;
    }
    return val;
}

__global__ void warpScan(const float* input, float* output, int n){
    __shared__ float warp_sum[32];

    const int tid = threadIdx.x;
    const int row = blockDim.x;

    const int lane = tid & 31;
    const int warp_id = tid / 32;

    float val = (idx < n) ? input[idx] : 0.0f;

    val = warpScanInclusive(val);

    if(lane == 31) warp_sum[warp_id] = val;

    __syncthreads();
    
    float row_val = 0.0f;

    if(warp_id == 0){
        int numWarps = (blockDim.x + 31) / 32;

        if(lane < numWarps) row_val = warp_sum[lane];

        row_val = warpScanInclusive(row_val);
        if(lane == 0) warp_sum[lane] = row_val;
    }
    __syncthreads();

    if(warp_id > 0) val += warp_sum[warp_id - 1];

    if(idx < n) output[idx] = val;
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
    for(int offset= 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ void warp_bitonic_sort(float& x, int& x_idx, int lane){
    unsigned mask = 0xffffffff;

    for(int k = 2; k <= 32; k <<= 1){
        for(int j = k >> 1;j > 0;j >>= 1){
            float other_x = __shfl_xor_sync(mask, x, j);

            float other_idx = __shfl_xor_sync(mask, x_idx, j);

            bool descending = ((lane & k) == 0);

            bool low_lane = ((lane & j) == 0);

            bool take_other = false;

            if(descending){
                if(low_lane) {
                    if(other_x > x) take_other = true;
                }else {
                    if(other_x < x) take_other = true;
                }
            }else {
                if(low_lane){
                    if(other_x < x) take_other = true;
                }else {
                    if(other_x > x) take_other = true;
                }
            }

            if(take_other){
                x = other_x;
                x_idx = other_idx;
            }
        }
    }
}

__device__ 

__global__ void top_kernel(const float* input, float* out_val, int* out_idx, int B, int N){
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    float topk_value = -INFINITY;
    int topk_index = -1;

    for(int base = warp_id * 32; base < N; base+= blockDim.x){
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

__device__ void warpArgmax(float &val, int& idx){
    unsigned mask = 0xffffffff;
    for(int offset= 16; offset > 0;offset >>= 1){
        float other_val = __shfl_down_sync(mask, val, offset);

        float other_idx = __shfl_down_sync(mask, idx, offset);

        if(other_val > val){
            val = other_val;
            idx = other_idx;
        }
    }
}

__global__ void Argmax(const float* input, int* output, int B, int N){
    __shared__ float warp_max[32];
    __shared__ int warp_idx[32];
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const int lane = tid % 32;
    const int warp_id = tid / 32;

    if(row >= B) return;

    float local_max = -INFINITY;
    float local_idx = -1;

    for(int i = tid;i < N;i += blockDim.x){
        float x = input[row * N + i];

        if(x > local_max){
            local_max = x;
            local_idx = i;
        }
    }

    warpArgmax(local_max, local_idx);

    if(lane == 0) {
        warp_max[warp_id] = local_max;
        warp_idx[warp_id] = local_sum;
    }

    __syncthreads();
    float row_max = -INFINITY;
    int row_idx = -1;
    if(warp_id == 0){
        int numWarps = (blockDim.x + 31) / 32;

        if(lane < numWarps) {
            row_max = warp_max[lane];
            row_idx = warp_idx[lane];
        }

        warpArgmax(row_max, row_idx);

        if(lane == 0){
            warp_max[0] = row_max;
            warp_idx[0] = row_idx;
        }
    }
    __syncthreads();
}

__device__ float warpReduceSum(float val){
    for(int offset = 16; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void gemv(const float* A, const float* x, float* y, int M, int K){
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp_id = tid / 32;

    int numWarps = (blockDim.x + 31) / 32;

    const int row = blockIdx.x * numWarps + warp_id;

    if(row >= M) return ;

    float sum = 0.0f;

    for(int k = lane; k < K; k += 32){
        sum += A[row * K + k] * x[k];
    }

    sum = warpReduceSum(sum);

    if(lane == 0) y[row] = sum;
}

__global__ void rope(const float* Q, const float* K, const float* cos_table, const float* sin_table, float* Q_out, float* K_out, int tokes, int num_heads, int head_dim){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int pairs = head_dim / 2;
    int total_pairs = tokens * num_heads * pairs;

    if(idx >= total_pairs) return;

    int pair = idx % pairs;
    int tmp  = idx / pairs;

    int head = tmp % num_heads;
    int token = tmp / num_heads;

    int base = (token * num_heads + head) * head_dim;

    int i0 = base + 2 * pair;
    int i1 = i0 + 1;

    float c = cos_table[token * pairs + pair];
    float s = sin_table[token * pairs + pair];

    float q0 = Q[i0];
    float q1 = Q[i1];

    Q_out[i0] = q0 * c - q1 * s;
    Q_out[i1] = q0 * s + q1 * c;

    float k0 = K[i0];
    float k1 = K[i1];

    K_out[k0] = k0 * c - k1 * s;
    K_out[k1] = k0 * s + k1 * c;
}

#define BM 16
#define BN 16
#define D 64
#define THREADS 128

__global__ void flash_attention_wmma(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int S,
    bool causal
){
    int tid = threadIdx.x;
    int warp = tid / 32;

    int q_start = blockIdx.x * BM;

    __shared__ __align__(32) half sQ[BM * D];
    __shared__ __align__(32) half sK[BN * D];
    __shared__ __align__(32) half sV[BN * D];
    __shared__ __align__(32) float sScore[BM * BN];
    __shared__ __align__(32) half sP[BM * BN];
    __shared__ __align__(32) float sPV[BM * D];
    __shared__ __align__(32) float sO[BM * D];

    __shared__ float sM[BM];
    __shared__ float sL[BM];
    __shared__ float sAlpha[BM];

    for(int i = tid; i < BM; i += blockDim.x){
        sM[i] = -INFINITY;
        sL[i] = 0.0f;
    }

    for(int i = tid; i < BM * D; i += blockDim.x){
        sO[i] = 0.0f;
    }

    __syncthreads();

    for(int i = tid; i < BM * D; i += blockDim.x){
        int row = i / D;
        int d = i % D;
        int q = q_start + row;

        if(q < S) sQ[i] = Q[q * D + d];
        else sQ[i] = __float2half(0.0f);
    }

    __syncthreads();

    for(int kv_start = 0; kv_start < S; kv_start += BN){

        for(int i = tid; i < BN * D; i += blockDim.x){
            int row = i / D;
            int d = i % D;
            int k = kv_start + row;

            if(k < S){
                sK[i] = K[k * D + d];
                sV[i] = V[k * D + d];
            }else{
                sK[i] = __float2half(0.0f);
                sV[i] = __float2half(0.0f);
            }
        }

        __syncthreads();

        if(warp == 0){
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> score_frag;
            wmma::fill_fragment(score_frag, 0.0f);

            for(int kk = 0; kk < D; kk += 16){
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;

                wmma::load_matrix_sync(q_frag, &sQ[kk], D);
                wmma::load_matrix_sync(k_frag, &sK[kk], D);

                wmma::mma_sync(score_frag, q_frag, k_frag, score_frag);
            }

            wmma::store_matrix_sync(sScore, score_frag, BN, wmma::mem_row_major);
        }

        __syncthreads();

        if(tid < BM){
            int row = tid;
            int q = q_start + row;

            if(q >= S){
                sAlpha[row] = 0.0f;

                for(int col = 0; col < BN; col++){
                    sP[row * BN + col] = __float2half(0.0f);
                }
            }else{
                float tile_max = -INFINITY;
                bool has_valid = false;

                for(int col = 0; col < BN; col++){
                    int k = kv_start + col;

                    bool valid = (k < S) && (!causal || k <= q);

                    if(valid){
                        float score = sScore[row * BN + col] * rsqrtf((float)D);
                        tile_max = fmaxf(tile_max, score);
                        has_valid = true;
                    }
                }

                if(!has_valid){
                    sAlpha[row] = 1.0f;

                    for(int col = 0; col < BN; col++){
                        sP[row * BN + col] = __float2half(0.0f);
                    }
                }else{
                    float old_m = sM[row];
                    float new_m = fmaxf(old_m, tile_max);

                    float alpha;
                    if(old_m == -INFINITY) alpha = 0.0f;
                    else alpha = expf(old_m - new_m);

                    sAlpha[row] = alpha;

                    float tile_sum = 0.0f;

                    for(int col = 0; col < BN; col++){
                        int k = kv_start + col;
                        bool valid = (k < S) && (!causal || k <= q);

                        float p = 0.0f;

                        if(valid){
                            float score = sScore[row * BN + col] * rsqrtf((float)D);
                            p = expf(score - new_m);
                        }

                        tile_sum += p;
                        sP[row * BN + col] = __float2half_rn(p);
                    }

                    sL[row] = alpha * sL[row] + tile_sum;
                    sM[row] = new_m;
                }
            }
        }

        __syncthreads();

        if(warp < 4){
            int col_start = warp * 16;

            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> pv_frag;

            wmma::fill_fragment(pv_frag, 0.0f);

            wmma::load_matrix_sync(p_frag, sP, BN);
            wmma::load_matrix_sync(v_frag, &sV[col_start], D);

            wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);

            wmma::store_matrix_sync(
                &sPV[col_start],
                pv_frag,
                D,
                wmma::mem_row_major
            );
        }

        __syncthreads();

        for(int i = tid; i < BM * D; i += blockDim.x){
            int row = i / D;

            sO[i] = sAlpha[row] * sO[i] + sPV[i];
        }

        __syncthreads();
    }

    for(int i = tid; i < BM * D; i += blockDim.x){
        int row = i / D;
        int d = i % D;
        int q = q_start + row;

        if(q < S){
            O[q * D + d] = __float2half_rn(sO[i] / sL[row]);
        }
    }
}

__global__ void swiglu_half2(const half* a, const half* b, half* out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int i = idx * 2;

    if(i  + 1 < n){
        half2 x = reinterpret_cast<const half2 *>(a)[idx];
        half2 y = reinterpret_cast<const half2 *>(b)[idx];

        float2 xf = __half22float2(x);
        float2 yf = __half22float2(y);

        float2 of;
        of.x = (xf.x / (1.0f + expf(-xf.x))) * yf.x;
        of.y = (xf.y / (1.0f + expf(-xf.y))) * yf.y;

        reinterpret_cast<half2*>(out)[idx] = __floats2half2_rn(of.x, of.y);
    }
}

__global__ void embedding_gather(const float* embedding, const int* ids, float* output, int N, int D){
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if(row >= N) return;

    int src_row = ids[row];

    for(int col = tid; col < N; col += blockDim.x){
        output[row * N + col] = input[src_row * N + col];
    }
}

__global__ void scatter_add(const float* value, const int* index, float* output, int n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < n){
        atomicAdd(&output[index[idx]], value[idx]);
    }
}

__device__ __forceinline__ float warpReduceSum(float val){
    for(int offset = 16; offset > 0;offset >>= 1){ 
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void fused_add_rmsnorm(const float* x, const float* residual, const float* gamma, float* output, int M, int N, float eps){
    __shared__ float warp_sum[32];
    __shared__ float inv_rms;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    if(row >= M) return;

    float local_sum = 0.0f;

    for(int i = tid;i < N;i+=blockDim.x)
    {
        float z = x[row * N + i] + residual[row * N + i];
        local_sum += z * z;
    }

    local_sum = warpReduceSum(local_sum);

    if(lane == 0) warp_sum[warp_id] = local_sum;
    __syncthreads();
    float row_sum = 0.0f;
    if(warp_id == 0){
        int numWarps = (blockDim.x + 31) / 32;

        if(lane < numWarps) row_sum = warp_sum[lane];

        row_sum = warpReduceSum(row_sum);

        if(lane == 0) warp_sum[0] = row_sum; 
    }
    __syncthreads();
    row_sum = warp_sum[0];
    inv_rms = rsqrtf(row_sum / N + eps);
    for(int i = tid;i < N;i+=blockDim.x){
        float z = x[row * N + i] + residual[row * N + i];

        output[row * N + i] = z * inv_rms * gamma[i];
    }

}

__device__ float warpReduceMax(float val){
    for(int offset = 16; offset > 0; offset >>= 1){
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__global__ void dynamic_quantize(const float* input, int8_t* output, float* scales, int M, int N){
    __shared__ float warp_max[32];
    __shared__ float scale;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    if(row >= M) return;

    float local_max = -INFINITY;
    for(int i = tid;i < N:i += blockDim.x){
        local_max = fmaxf(local_max, input[row * N + i]);
    }

    if(lane == 0) warp_max[warp_id] = local_max;
    __syncthreads();
    float row_max = -INFINITY;
    if(warp_id == 0){
        int numWarps = (blockDim.x + 31) / 32;
        if(lane < numWarps) row_max = warp_max[lane];

        row_max = warpReduceMax(row_max);

        if(lane == 0) {
            if(max_val == 0.0f){
                scale = 1.0f;
            }else{
                scale = max_val / 127.0f;
            }
            scales[row] = scale;
        }
    }

    __syncthreads();

    float s = scalel
    for(int i = tid;i < N;i+=blockDim.x){
        float q = roundf(input[row * N + i] / s);

        q = fmaxf(-127.0f, fminf(127.0f, q));

        output[row * N + i] = static_cast<int8_t>(q);
    }
}
