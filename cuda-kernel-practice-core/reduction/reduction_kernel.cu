#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// 根据模版类型获取动态的 shared memory 指针
template <class T> struct SharedMemory
{
    __device__ inline operator T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// double 对内存对齐的要求更高 单独写 保证double shared memory对齐更安全
template <> struct SharedMemory<double>
{
    __device__ inline operator double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

// warp内求和
// offset = 16: 线程0加线程16，线程1加线程17
// offset = 8 : 线程0再加线程8
// 最后输出是 lane 0
template <class T> __device__ __forceinline__ T warpReduceSum(unsigned int mask, T mySum)
{
    // __shfl_down_sync 不经过 shared memory 直接在线程之间交换寄存器数据
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        mySum += __shfl_down_sync(mask, mySum, offset); 
    }
    return mySum;
}
// GPU计算能力如果是8.0
#if __CUDA_ARCH__ >= 800

template <> __device__ __forceinline__ int warpReduceSum<int>(unsigned int mask, int mySum)
{
    // 数据类型是 int 不用手写 shuffle 循环 直接使用硬件级别的
    mySum = __reduce_add_sync(mask, mySum);
    return mySum;
}
#endif

/*
所有kerne目标：
输入：g_idata
输出：g_odata
长度：n

1. 每个线程从 global memory 读一个或多个元素
2. 放入 shared memory 或 mySum
3. block 内部做归约
4. block 的 0 号线程写出结果
*/
template <class T> __global__ void reduce0(T *g_idata, T *g_odata, unsigned int n)
{
    // 当前 block
    cg::thread_block cta   = cg::this_thread_block();
    // 当前 shared memory
    T               *sdata = SharedMemory<T>();
    // 当前线程在 block 里面的编号
    unsigned int tid = threadIdx.x;
    // 当前线程对应的全局数组的下标
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;
    // 把 global memory 的值搬到 shared memory
    // 填0是因为不影响加法的总和 还有就是最后一个block可能不满 所以填0
    sdata[tid] = (i < n) ? g_idata[i] : 0;

    cg::sync(cta);
    // 多路合并相加 充分利用其并行性
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {

        if ((tid % (2 * s)) == 0) {
            sdata[tid] += sdata[tid + s];
        }

        cg::sync(cta);
    }
    // 足后 lane0 输出结果到 global memory
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

template <class T> __global__ void reduce1(T *g_idata, T *g_odata, unsigned int n)
{

    cg::thread_block cta   = cg::this_thread_block();
    T               *sdata = SharedMemory<T>();

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    cg::sync(cta);

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        // 减少取模开销
        int index = 2 * s * tid;

        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }

        cg::sync(cta);
    }

    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

template <class T> __global__ void reduce2(T *g_idata, T *g_odata, unsigned int n)
{

    cg::thread_block cta   = cg::this_thread_block();
    T               *sdata = SharedMemory<T>();

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    cg::sync(cta);

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }

        cg::sync(cta);
    }

    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

template <class T> __global__ void reduce3(T *g_idata, T *g_odata, unsigned int n)
{

    cg::thread_block cta   = cg::this_thread_block();
    T               *sdata = SharedMemory<T>();

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    T mySum = (i < n) ? g_idata[i] : 0;

    if (i + blockDim.x < n)
        mySum += g_idata[i + blockDim.x];

    sdata[tid] = mySum;
    cg::sync(cta);

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }

        cg::sync(cta);
    }

    if (tid == 0)
        g_odata[blockIdx.x] = mySum;
}

template <class T, unsigned int blockSize> __global__ void reduce4(T *g_idata, T *g_odata, unsigned int n)
{

    cg::thread_block cta   = cg::this_thread_block();
    T               *sdata = SharedMemory<T>();

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    T mySum = (i < n) ? g_idata[i] : 0;

    if (i + blockSize < n)
        mySum += g_idata[i + blockSize];

    sdata[tid] = mySum;
    cg::sync(cta);

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }

        cg::sync(cta);
    }

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    if (cta.thread_rank() < 32) {

        if (blockSize >= 64)
            mySum += sdata[tid + 32];

        for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
            mySum += tile32.shfl_down(mySum, offset);
        }
    }

    if (cta.thread_rank() == 0)
        g_odata[blockIdx.x] = mySum;
}

template <class T, unsigned int blockSize> __global__ void reduce5(T *g_idata, T *g_odata, unsigned int n)
{

    cg::thread_block cta   = cg::this_thread_block();
    T               *sdata = SharedMemory<T>();

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockSize * 2) + threadIdx.x;

    T mySum = (i < n) ? g_idata[i] : 0;

    if (i + blockSize < n)
        mySum += g_idata[i + blockSize];

    sdata[tid] = mySum;
    cg::sync(cta);

    if ((blockSize >= 512) && (tid < 256)) {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    cg::sync(cta);

    if ((blockSize >= 256) && (tid < 128)) {
        sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

    cg::sync(cta);

    if ((blockSize >= 128) && (tid < 64)) {
        sdata[tid] = mySum = mySum + sdata[tid + 64];
    }

    cg::sync(cta);

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    if (cta.thread_rank() < 32) {

        if (blockSize >= 64)
            mySum += sdata[tid + 32];

        for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
            mySum += tile32.shfl_down(mySum, offset);
        }
    }

    if (cta.thread_rank() == 0)
        g_odata[blockIdx.x] = mySum;
}

template <class T, unsigned int blockSize, bool nIsPow2> __global__ void reduce6(T *g_idata, T *g_odata, unsigned int n)
{

    cg::thread_block cta   = cg::this_thread_block();
    T               *sdata = SharedMemory<T>();

    unsigned int tid      = threadIdx.x;
    unsigned int gridSize = blockSize * gridDim.x;

    T mySum = 0;

    if (nIsPow2) {
        unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
        gridSize       = gridSize << 1;

        while (i < n) {
            mySum += g_idata[i];

            if ((i + blockSize) < n) {
                mySum += g_idata[i + blockSize];
            }
            i += gridSize;
        }
    }
    else {
        unsigned int i = blockIdx.x * blockSize + threadIdx.x;
        while (i < n) {
            mySum += g_idata[i];
            i += gridSize;
        }
    }

    sdata[tid] = mySum;
    cg::sync(cta);

    if ((blockSize >= 512) && (tid < 256)) {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    cg::sync(cta);

    if ((blockSize >= 256) && (tid < 128)) {
        sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

    cg::sync(cta);

    if ((blockSize >= 128) && (tid < 64)) {
        sdata[tid] = mySum = mySum + sdata[tid + 64];
    }

    cg::sync(cta);

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    if (cta.thread_rank() < 32) {

        if (blockSize >= 64)
            mySum += sdata[tid + 32];

        for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
            mySum += tile32.shfl_down(mySum, offset);
        }
    }

    if (cta.thread_rank() == 0)
        g_odata[blockIdx.x] = mySum;
}

template <typename T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce7(const T *__restrict__ g_idata, T *__restrict__ g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    unsigned int tid        = threadIdx.x;
    unsigned int gridSize   = blockSize * gridDim.x;
    unsigned int maskLength = (blockSize & 31);
    maskLength              = (maskLength > 0) ? (32 - maskLength) : maskLength;
    const unsigned int mask = (0xffffffff) >> maskLength;

    T mySum = 0;

    if (nIsPow2) {
        unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
        gridSize       = gridSize << 1;

        while (i < n) {
            mySum += g_idata[i];

            if ((i + blockSize) < n) {
                mySum += g_idata[i + blockSize];
            }
            i += gridSize;
        }
    }
    else {
        unsigned int i = blockIdx.x * blockSize + threadIdx.x;
        while (i < n) {
            mySum += g_idata[i];
            i += gridSize;
        }
    }

    mySum = warpReduceSum<T>(mask, mySum);

    if ((tid % warpSize) == 0) {
        sdata[tid / warpSize] = mySum;
    }

    __syncthreads();

    const unsigned int shmem_extent  = (blockSize / warpSize) > 0 ? (blockSize / warpSize) : 1;
    const unsigned int ballot_result = __ballot_sync(mask, tid < shmem_extent);
    if (tid < shmem_extent) {
        mySum = sdata[tid];

        mySum = warpReduceSum<T>(ballot_result, mySum);
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = mySum;
    }
}

template <typename T, typename Group> __device__ T cg_reduce_n(T in, Group &threads)
{
    return cg::reduce(threads, in, cg::plus<T>());
}

template <class T> __global__ void cg_reduce(T *g_idata, T *g_odata, unsigned int n)
{

    T *sdata = SharedMemory<T>();

    cg::thread_block cta = cg::this_thread_block();

    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);

    unsigned int ctaSize     = cta.size();
    unsigned int numCtas     = gridDim.x;
    unsigned int threadRank  = cta.thread_rank();
    unsigned int threadIndex = (blockIdx.x * ctaSize) + threadRank;

    T threadVal = 0;
    {
        unsigned int i           = threadIndex;
        unsigned int indexStride = (numCtas * ctaSize);
        while (i < n) {
            threadVal += g_idata[i];
            i += indexStride;
        }
        sdata[threadRank] = threadVal;
    }

    {
        unsigned int ctaSteps = tile.meta_group_size();
        unsigned int ctaIndex = ctaSize >> 1;
        while (ctaIndex >= 32) {
            cta.sync();
            if (threadRank < ctaIndex) {
                threadVal += sdata[threadRank + ctaIndex];
                sdata[threadRank] = threadVal;
            }
            ctaSteps >>= 1;
            ctaIndex >>= 1;
        }
    }

    {
        cta.sync();
        if (tile.meta_group_rank() == 0) {
            threadVal = cg_reduce_n(threadVal, tile);
        }
    }

    if (threadRank == 0)
        g_odata[blockIdx.x] = threadVal;
}

template <class T, size_t BlockSize, size_t MultiWarpGroupSize>
__global__ void multi_warp_cg_reduce(T *g_idata, T *g_odata, unsigned int n)
{

    T         *sdata = SharedMemory<T>();
    __shared__ cg::block_tile_memory<BlockSize> scratch;

    auto cta = cg::this_thread_block(scratch);

    auto multiWarpTile = cg::tiled_partition<MultiWarpGroupSize>(cta);

    unsigned int gridSize  = BlockSize * gridDim.x;
    T            threadVal = 0;

    int nIsPow2 = !(n & n - 1);
    if (nIsPow2) {
        unsigned int i = blockIdx.x * BlockSize * 2 + threadIdx.x;
        gridSize       = gridSize << 1;

        while (i < n) {
            threadVal += g_idata[i];

            if ((i + BlockSize) < n) {
                threadVal += g_idata[i + blockDim.x];
            }
            i += gridSize;
        }
    }
    else {
        unsigned int i = blockIdx.x * BlockSize + threadIdx.x;
        while (i < n) {
            threadVal += g_idata[i];
            i += gridSize;
        }
    }

    threadVal = cg_reduce_n(threadVal, multiWarpTile);

    if (multiWarpTile.thread_rank() == 0) {
        sdata[multiWarpTile.meta_group_rank()] = threadVal;
    }
    cg::sync(cta);

    if (threadIdx.x == 0) {
        threadVal = 0;
        for (int i = 0; i < multiWarpTile.meta_group_size(); i++) {
            threadVal += sdata[i];
        }
        g_odata[blockIdx.x] = threadVal;
    }
}

extern "C" bool isPow2(unsigned int x);

template <class T> void reduce(int size, int threads, int blocks, int whichKernel, T *d_idata, T *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    if (threads < 64 && whichKernel == 9) {
        whichKernel = 7;
    }

    switch (whichKernel) {
    case 0:
        reduce0<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

    case 1:
        reduce1<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

    case 2:
        reduce2<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

    case 3:
        reduce3<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

    case 4:
        switch (threads) {
        case 512:
            reduce4<T, 512><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 256:
            reduce4<T, 256><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 128:
            reduce4<T, 128><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 64:
            reduce4<T, 64><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 32:
            reduce4<T, 32><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 16:
            reduce4<T, 16><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 8:
            reduce4<T, 8><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 4:
            reduce4<T, 4><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 2:
            reduce4<T, 2><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 1:
            reduce4<T, 1><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;
        }

        break;

    case 5:
        switch (threads) {
        case 512:
            reduce5<T, 512><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 256:
            reduce5<T, 256><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 128:
            reduce5<T, 128><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 64:
            reduce5<T, 64><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 32:
            reduce5<T, 32><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 16:
            reduce5<T, 16><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 8:
            reduce5<T, 8><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 4:
            reduce5<T, 4><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 2:
            reduce5<T, 2><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 1:
            reduce5<T, 1><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;
        }

        break;

    case 6:
        if (isPow2(size)) {
            switch (threads) {
            case 512:
                reduce6<T, 512, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 256:
                reduce6<T, 256, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 128:
                reduce6<T, 128, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 64:
                reduce6<T, 64, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 32:
                reduce6<T, 32, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 16:
                reduce6<T, 16, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 8:
                reduce6<T, 8, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 4:
                reduce6<T, 4, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 2:
                reduce6<T, 2, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 1:
                reduce6<T, 1, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;
            }
        }
        else {
            switch (threads) {
            case 512:
                reduce6<T, 512, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 256:
                reduce6<T, 256, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 128:
                reduce6<T, 128, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 64:
                reduce6<T, 64, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 32:
                reduce6<T, 32, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 16:
                reduce6<T, 16, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 8:
                reduce6<T, 8, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 4:
                reduce6<T, 4, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 2:
                reduce6<T, 2, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 1:
                reduce6<T, 1, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;
            }
        }

        break;

    case 7:

        smemSize = ((threads / 32) + 1) * sizeof(T);
        if (isPow2(size)) {
            switch (threads) {
            case 1024:
                reduce7<T, 1024, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;
            case 512:
                reduce7<T, 512, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 256:
                reduce7<T, 256, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 128:
                reduce7<T, 128, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 64:
                reduce7<T, 64, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 32:
                reduce7<T, 32, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 16:
                reduce7<T, 16, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 8:
                reduce7<T, 8, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 4:
                reduce7<T, 4, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 2:
                reduce7<T, 2, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 1:
                reduce7<T, 1, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;
            }
        }
        else {
            switch (threads) {
            case 1024:
                reduce7<T, 1024, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;
            case 512:
                reduce7<T, 512, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 256:
                reduce7<T, 256, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 128:
                reduce7<T, 128, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 64:
                reduce7<T, 64, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 32:
                reduce7<T, 32, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 16:
                reduce7<T, 16, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 8:
                reduce7<T, 8, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 4:
                reduce7<T, 4, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 2:
                reduce7<T, 2, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 1:
                reduce7<T, 1, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;
            }
        }

        break;
    case 8:
        cg_reduce<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;
    case 9:
        constexpr int numOfMultiWarpGroups = 2;
        smemSize                           = numOfMultiWarpGroups * sizeof(T);
        switch (threads) {
        case 1024:
            multi_warp_cg_reduce<T, 1024, 1024 / numOfMultiWarpGroups>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 512:
            multi_warp_cg_reduce<T, 512, 512 / numOfMultiWarpGroups>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 256:
            multi_warp_cg_reduce<T, 256, 256 / numOfMultiWarpGroups>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 128:
            multi_warp_cg_reduce<T, 128, 128 / numOfMultiWarpGroups>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        case 64:
            multi_warp_cg_reduce<T, 64, 64 / numOfMultiWarpGroups>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

        default:
            printf("thread block size of < 64 is not supported for this kernel\n");
            break;
        }
        break;
    }
}

template void reduce<int>(int size, int threads, int blocks, int whichKernel, int *d_idata, int *d_odata);

template void reduce<float>(int size, int threads, int blocks, int whichKernel, float *d_idata, float *d_odata);

template void reduce<double>(int size, int threads, int blocks, int whichKernel, double *d_idata, double *d_odata);

#endif
