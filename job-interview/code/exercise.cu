/*
    reduction sum
*/

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

}
