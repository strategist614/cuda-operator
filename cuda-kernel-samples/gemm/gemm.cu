#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <mma.h>


using namespace std;


// ============================================================

#define checkCudaErrors(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr                                         \
                << "CUDA Error: "                             \
                << cudaGetErrorString(err)                    \
                << " at "                                     \
                << __FILE__                                   \
                << ":"                                        \
                << __LINE__                                   \
                << std::endl;                                 \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)
// 1. 向上取整
#define CEIL(a, b) (((a) + (b) - 1) / (b))

// 2. FLOAT4，用于向量化访存
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

__global__ void gemm_naive(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < M && col < N){
        float sum = 0.0f;
        for(int k = 0;k < K;k++) sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}


#define BM 16
#define BN 16
#define BK 16

__global__ void gemm_block_tile(
    const float* A;
    const float* B;
    float * C;
    int M,
    int N,
    int K
){
    __shared__ float As[BM][BK];
    __shared__ float Bs[BM][BK];    

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockDim.y * blockIdx.y + ty;
    int col = blockDim.x * blockIdx.x + tx;

    float sum = 0.0f;

    for(int bk = 0; bk < K;bk += BK){
        int a_col = bk + tx;
        int b_row = bk + ty;

        As[ty][tx] = (row < M) && (a_col < K) ? A[row * K + a_col] : 0.0f;

        Bs[ty][tx] = (b_row < K) && (col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for(int k = 0;k < bk;k++) sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if(row < M && col < N) C[row * N + col] = sum;
}


#define BM 32
#define BN 32
#define BK 8

#define WM 16
#define WN 16

#define TM 2
#define TN 4

__global__ void gemm_all_tile(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
){
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    int warp_row = warp_id / 2;
    int warp_col = warp_id % 2;

    int lane_row = lane / 4;
    int lene_col = lane % 4;

    float acc[TM][TN] = {0.0f};

    int blcok_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    for(int bk = 0;bk < K;bk += BK){
        for (int i = tid;
             i < BM * BK;
             i += blockDim.x)
        {
            int r = i / BK;
            int c = i % BK;

            int global_row = block_row + r;
            int global_col = bk + c;

            As[r][c] =
                (global_row < M && global_col < K)
                ? A[global_row * K + global_col]
                : 0.0f;
        }

        for (int i = tid;
             i < BK * BN;
             i += blockDim.x)
        {
            int r = i / BN;
            int c = i % BN;

            int global_row = bk + r;
            int global_col = block_col + c;

            Bs[r][c] =
                (global_row < K && global_col < N)
                ? B[global_row * N + global_col]
                : 0.0f;
        }


        __syncthreads();

        #pragma unroll
        for(int k =0;k < BK;++k){

            float a_frag[TM];

            #pragma unroll
            for(int i = 0;i < TM;i++){
                int row = warp_row * WM + lane_row * TM + i;
                a_frag[i] = As[row][k];
            }

            float b_frag[TN];

            #pragma unroll
            for (int j = 0; j < TN; ++j) {

                int col =
                    warp_col * WN
                    + lane_col * TN
                    + j;

                b_frag[j] = Bs[k][col];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {

                #pragma unroll
                for (int j = 0; j < TN; ++j) {

                    acc[i][j] +=
                        a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {

        #pragma unroll
        for (int j = 0; j < TN; ++j) {

            int row =
                block_row
                + warp_row * WM
                + lane_row * TM
                + i;

            int col =
                block_col
                + warp_col * WN
                + lane_col * TN
                + j;

            if (row < M && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}

using namespace nvcuda;

#define BM 128
#define BN 128
#define BK 16

__global__ void gemm_wmma(const half* A, const half* B, float* C, int M, int N, int K){
    __shared__ half As[BM][BK];
    __shared__ half Bs[BK][BN];

    int tid = threadIdx.x;

    int warp_id = tid / 32;

    int warp_row = warp_id / 4;
    int warp_col = warp_id % 4;

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for(int bk = 0;bk < K;bk += BK){
        for(int i = tid;i < BM * BK;i+=blockDim.x){
            int r = i / BK:
            int c = i % BK;

            int gr = block_row + r;
            int gc = bk + c;

            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
        }

        for(int i = tid;i < BK * BN;i+=blockDim.x){
            int r = i / BN:
            int c = i % BN;

            int gr = bk + r;
            int gc = block_col + c;

            Bs[r][c] = (gr < K && gc < N) ? A[gr * N + gc] : __float2half(0.0f);
        }
        __syncthreads();

        for(int k=0;k<BK;k+=16)
        {


            wmma::fragment<wmma::matrix_a,16,16,16, half, wmma::row_major> a_frag;

            wmma::fragment<wmma::matrix_b,16,16,16, half, wmma::col_major> b_frag;

            int warp_m = warp_row*16;


            int warp_n = warp_col*16;

            wmma::load_matrix_sync( a_frag, &As[warp_m][k], BK);

            wmma::load_matrix_sync( b_frag, &Bs[k][warp_n], BN);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();

    }
    int c_row = block_row + warp_row*16;


    int c_col = block_col + warp_col*16;

    if(c_row<M && c_col<N)
    {
        wmma::store_matrix_sync( &C[c_row*N+c_col], c_frag, N, wmma::mem_row_major);
    }
}

int main()
{
    int N = 1024;
    int M = 1024;
    int bytes = M * N * sizeof(float);
    float *h_A = new float[M * N];
    float *h_B = new float[N * M];   
    float *h_C = new float[M * M];·

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    checkCudaErrors(cudaMalloc(&d_A, bytes));
    checkCudaErrors(cudaMalloc(&d_B, bytes));
    checkCudaErrors(cudaMalloc(&d_C, bytes));

    for(int i = 0;i < M * N;i ++) h_A[i] = static_cast<float>(i % 10);
    for(int i = 0;i < N * M;i ++) h_B[i] = static_cast<float>(i % 10);

    checkCudaErrors(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(
        (N + 15) / 16,
        (M + 15) / 16
    );

    gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, N);

    cudaGetLastError();
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    for(int i = 0;i < 10;i ++) cout << h_C[i] << " ";
    return 0;
}