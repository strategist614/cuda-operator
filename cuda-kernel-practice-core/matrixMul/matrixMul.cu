#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h>

// 定义模版参数 BLOCK_SIZE kernel块的大小在编译时确定 后面调用的时候会给出大小
// wA矩阵A的宽度 wB矩阵B的宽度
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A, float *B, int wA, int wB){
    // 当前这个block负责矩阵C的第几列小块
    int bx = blockIdx.x;
    // 当前这个block负责矩阵C的第几行小块
    int by = blockIdx.y;

    // 当前这个线程在小组里面的位置 表示当前这个线程是在block里面第tx列、第ty行的位置
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // 计算A矩阵的起点 当前block在第by行小块
    // A的起始行是 BLOCK_SIZE * by
    // 一行有 wA 个threads
    // 所以起始位置是 wA * BLOCK_SIZE * by
    int aBegin = wA * BLOCK_SIZE * by;
    // 当前block下 从开头到结尾
    int aEnd = aBegin + wA - 1;
    // A的block是横着移动的 每次移动一个block的大小
    int aStep = BLOCK_SIZE;
    // 
    int bBegin = BLOCK_SIZE * bx;
    // 这里是和a的方式不一样 因为是行和列有不同 因为取的是列 所以每往下一行就要加个 BLOCK_SIZE
    int bStep = BLOCK_SIZE * wB;
    // 记录算出来的C的结果
    float Csub = 0;
    for(int a = aBegin,b=bBegin; a <= aEnd; a += aStep, b += bStep){
        // 共享数组大小必须在编译期确定
        // block内部的共享内存
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        // 这里ty tx是每个线程的处理 也就是C矩阵的一个元素值
        // 又因为As Bs是block共享内存 所以这里也是在多线程的往矩阵里面填数据
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];
        // 所有线程需要同步等待结果
        __syncthreads();
// 告诉编译器 下面这个 k 循环 可以展开，尽量优化
#pragma unroll
        // 矩阵运算
        // 当前 thread 拿着A的第ty行 拿B的第tx列 一项一项的相乘
        for(int k = 0;k < BLOCK_SIZE;++k){
            Csub += As[ty][k] * Bs[k][tx];
        }
        // 这里需要同步 不然 As Bs会出现覆盖的问题
        __syncthreads();
    }
    // 还原真正的C矩阵的位置
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    // 当前block的起点+当前thread在小块里面位置
    C[c + wB * ty + tx] = Csub;
}

void ConstantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}

int MatrixMultiply(int argc, char **argv, int block_size, const dim3 &dimsA, const dim3 &dimsB){
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A;
    // CPU 申请内存
    // CUDA 申请页锁定主机内存 CPU端内存
    // 这里提高CPU-GPU数据拷贝速度 支持真正的异步拷贝
    checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B;
    checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
    // GPU的任务队列 之后的拷贝和kernel都会放到这个队列里面
    // 保证任务是按顺序执行的
    cudaStream_t stream;

    const float valB = 0.01f;
    ConstantInit(h_A, size_A, 1.0f);
    ConstantInit(h_B, size_B, valB);
    // d_A, d_B, d_C 保存的是GPU显存的地址
    float *d_A, *d_B, *d_C;
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C;
    checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

    if (h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }
    // GPU上面申请显存
    // cudaError_t cudaMalloc(void **devPtr, size_t size) reinterpret_cast 需要做类型的强制转换
    // 把 d_A 变量的地址给cudaMalloc函数 可以往里面写GPU地址
    // 因为 d_A 本身的类型是 float * 所以&A 就是 float **
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));
    // 创建CUDA事件 等同于秒表的作用
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    // 创建GPU任务队列 不会被默认stream隐式阻塞
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // 把 h_A 从 CPU 拷贝到 GPU 的 d_A
    // 把这个拷贝任务放入 stream 中
    // 这个是异步的 CPU 不会等待拷贝任务做完 而是会继续往下执行
    // CPU不等待GPU GPU无法知道任务顺序 所以将这个拷贝任务加入了stream队列中
    checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    printf("Computing result using CUDA Kernel...\n");

    if (block_size == 16) {
        MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }
    else {
        MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }

    printf("done\n");
    // CPU等待这个 stream 里面的任务做完
    // 等GPU完成前面的拷贝和第一次kernel
    checkCudaErrors(cudaStreamSynchronize(stream));
    // 开始计时
    checkCudaErrors(cudaEventRecord(start, stream));

    int nIter = 300;

    for (int j = 0; j < nIter; j++) {
        if (block_size == 16) {
            MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
        else {
            MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
    }
    checkCudaErrors(cudaEventRecord(stop, stream));
    // CPU等待 stop 完成
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    float  msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul =
        2.0 * static_cast<double>(dimsA.x) * static_cast<double>(dimsA.y) * static_cast<double>(dimsB.x);
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
           " WorkgroupSize= %u threads/block\n",
           gigaFlops,
           msecPerMatrixMul,
           flopsPerMatrixMul,
           threads.x * threads.y);

    checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    printf("Checking computed result for correctness: ");
    bool correct = true;

    double eps = 1.e-6; // machine zero

    for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
        double abs_err    = fabs(h_C[i] - (dimsA.x * valB));
        double dot_length = dimsA.x;
        double abs_val    = fabs(h_C[i]);
        double rel_err    = abs_err / abs_val / dot_length;

        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], dimsA.x * valB, eps);
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    printf("\nNOTE: The CUDA Samples are not meant for performance "
           "measurements. Results may vary when GPU Boost is enabled.\n");

    if (correct) {
        return EXIT_SUCCESS;
    }
    else {
        return EXIT_FAILURE;
    }
}

int main(int argc, char **argv)
{
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
        printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
        printf("  Note: Outer matrix dimensions of A & B matrices"
               " must be equal.\n");

        exit(EXIT_SUCCESS);
    }

    int dev = findCudaDevice(argc, (const char **)argv);

    int block_size = 32;

    dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
    dim3 dimsB(5 * 2 * block_size, 5 * 2 * block_size, 1);

    if (checkCmdLineFlag(argc, (const char **)argv, "wA")) {
        dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "hA")) {
        dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "wB")) {
        dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "hB")) {
        dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
    }

    if (dimsA.x != dimsB.y) {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n", dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);
    // profiler是性能采样工具 这里是采样开始
    checkCudaErrors(cudaProfilerStart());
    int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);
    checkCudaErrors(cudaProfilerStop());

    exit(matrix_result);
    return 0;
}