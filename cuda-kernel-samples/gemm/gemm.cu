__global__ void gemm_naive(const float* A, const float* B, float* C, int m,int n,int k_size){
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockIdx.x + threadIdx.x;

    if(row < m && col < n){
        float sum = 0.0f;

        for(int k = 0;k < k_size;k++) sum += A[row * k_size + k] * B[k * n + col];

        C[row * n + col] = sum;
    }
}

__global__ void gemm_shared_tile(const float* A, const float* B, float* C, int m,int n,int k_size){
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for(int bk = 0;bk < k_size; bk += TILE){
        const int a_col = bk + tx;
        const int b_row = bk + ty;
        As[ty][tx] = (row < m && a_col < k_size) ? A[row * k_size + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < k_size && col < n) ? B[b_row * n + col] : 0.0f;

        __syncthreads();
#pragma unroll
        for(int k = 0;k < TILE; ++k) sum += As[ty][k] * Bs[k][tx];
    }
    if(row < m && col < n) C[row * n + col] = sum;
}
