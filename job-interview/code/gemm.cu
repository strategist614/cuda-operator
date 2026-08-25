__global__ void gemm_naive(const float* A, const float* B, int m, int n, int k_size){
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int rol = blockDim.x * blockIdx.y + threadIdx.x;

    if(row < m && col < n){
        float sum = 0.0f;
        for(int k = 0;k < k_size;k++) sum += A[row * k_size + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

__global__ void gemm_shared_tile(const float* A, const float* B, float* C, int m, int n, int k_size){
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockDim.y * blockIdx.y + ty;
    const int col = blockDim.x * blockIdx.x + tx;

    for(int bk = 0;bk < k_size;bk += TILE){
        const int a_col = tx + bk;
        const int b_row = ty + bk;

        As[ty][tx] = (row < m && a_col < k_size) ? A[row * k_size + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < k_size && col < n) ? B[b_row * n + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for(int k = 0;k < TILE;k++) sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if(row < m && col < n) C[row * n + col] = sum;
}

__global__ void gemm_register_tile(const float* A, const float*B, float* C,int n, int m,int k_size){
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

    for(int bk = 0;bk < k_size;bk += BK){
        for(int i = 0;i < BM * BK;i += blockDim.x){
            const int r = i / BK;
            const int c = i % BK;

            const int global_row = block_row + r;
            const int global_col = bk + c;

            As[r][c] = (global_row < m && global_col < k_size) : A[global_row * k_size + global_col] : 0.0f;
        }
        for(int i =0;i < BK*BN;i+= blockDim.x){
            const int r = i / BN;
            const int c = i % BN;

            const int global_row = bk + r;
            const int global_col = block_col + c;

            Bs[r][c] = (global_row < k_size && global_col < n) : B[global_row * n + global_col] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for(int k = 0;k < BK;++k){
            float a_frag[TM];
            float b_frag[TN];
#pragma unroll
            for(int i = 0;i < TM;++i){
                const int row = warp_row * WM + lane_row * TM + i;
                a_frag[i] = As[row][k];
            }
#pragma unroll
            for(int j = 0;j < TN; ++j){
                const int col = warp_col * WN + lane_col * TN + j;
                b_frag[j] = Bs[k][col];
            }
#pragma unroll
            for(int i = 0;i < TM;i++)
#pragma unroll
                for(int j = 0;j < TN;j ++) acc[i][j] += a_frag[i] * b_frag[j];
        }
        __syncthreads();

    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int row = block_row + warp_row * WM + lane_row * TM + i;
            const int col = block_col + warp_col * WN + lane_col * TN + j;
            if (row < m && col < n) C[row * n + col] = acc[i][j];
        }
}

__global__ void gemm_wmma(const float* A, const float* B, float* C, int n, int m, int k_size){ 
    __shared__ __align__(32) half As[BM][BK];
    __shared__ __align__(32) half Bs[BK][BN];
    __shared__ __align__(32) float Cs[BM][BN];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for(int bk = 0;bk < k_size;bk += BK){
        for(int i = tid;i < BM * BK;i += blockDim.x){
            const int r = i / BK;
            const int c = i % BK;
            const int global_row = block_row + r;
            const int global_col = bk + c;

            As[r][c] = (global_row < m && global_col < k_size) ? A[global_row * k_size + global_col] : __float2half(0.0f);
        }
        for(int i = tid;i < BK *BN;i += blockDim.x){
            const int r = i / BN;
            const int c = i % BN;
            const int global_row = bk + r;
            const int global_col = block_col + c;

            Bs[r][c] = (global_row < k_size && global_col < n) ? B[global_row * n + global_col] : __float2half(0,0f);
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
    __syncthreads();

    for(int i =tid;i < BM*BN;i+= blockDim.x){
        const int r = i / BN;
        const int c = i % BN;
        const int global_row = block_row + r;
        const int global_col = block_col + c;
        if(global_row < m && global_col < n) C[global_row * n + global_col] = Cs[r][c];
    }
    
}