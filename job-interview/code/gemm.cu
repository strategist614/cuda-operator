__global__ void gemm_naive(const float* A, const float* B, int m, int n, int k_size){
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int rol = blockDim.x * blockIdx.y + threadIdx.x;

    if(row < m && col < n){
        float sum = 0.0f;
        for(int k = 0;k < k_size;k++) sum += A[row * k_size + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

__global__ void gemm_shared_tile(const float* A, const float* B, int m, int n, int k_size){
    __shared__ float As[BK][BK];
    __shared__ float Bs[BK][BK];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;


    const int block_row = blockIdx.y * BK + ty;
    const int block_col = blockIdx.x * BK + tx;

    float sum = 0.0f;
    for(int k = 0; k < k_size;k += BK){
        const int a_col = tx + bk;
        const int b_row = ty + bk;


    }
}
