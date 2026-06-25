#include <cuda_runtime.h>

#include <algorithm>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "reduction.h"

enum ReduceType { REDUCE_INT, REDUCE_FLOAT, REDUCE_DOUBLE };

template <class T> bool runTest(int argc, char **argv, ReduceType datatype);

#define MAX_BLOCK_DIM_SIZE 65535

#ifdef WIN32
#define strcasecmp strcmpi
#endif

extern "C" bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

const char *getReduceTypeString(const ReduceType type)
{
    switch (type) {
    case REDUCE_INT:
        return "int";
    case REDUCE_FLOAT:
        return "float";
    case REDUCE_DOUBLE:
        return "double";
    default:
        return "unknown";
    }
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n\n", argv[0]);

    char *typeInput = 0;
    getCmdLineArgumentString(argc, (const char **)argv, "type", &typeInput);

    ReduceType datatype = REDUCE_INT;

    if (0 != typeInput) {
        if (!strcasecmp(typeInput, "float")) {
            datatype = REDUCE_FLOAT;
        }
        else if (!strcasecmp(typeInput, "double")) {
            datatype = REDUCE_DOUBLE;
        }
        else if (strcasecmp(typeInput, "int")) {
            printf("Type %s is not recognized. Using default type int.\n\n", typeInput);
        }
    }

    cudaDeviceProp deviceProp;
    int            dev;

    dev = findCudaDevice(argc, (const char **)argv);

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    printf("Using Device %d: %s\n\n", dev, deviceProp.name);
    checkCudaErrors(cudaSetDevice(dev));

    printf("Reducing array of type %s\n\n", getReduceTypeString(datatype));

    bool bResult = false;

    switch (datatype) {
    default:
    case REDUCE_INT:
        bResult = runTest<int>(argc, argv, datatype);
        break;

    case REDUCE_FLOAT:
        bResult = runTest<float>(argc, argv, datatype);
        break;

    case REDUCE_DOUBLE:
        bResult = runTest<double>(argc, argv, datatype);
        break;
    }

    printf(bResult ? "Test passed\n" : "Test failed!\n");
}

template <class T> T reduceCPU(T *data, int size)
{
    T sum = data[0];
    T c   = (T)0.0;

    for (int i = 1; i < size; i++) {
        T y = data[i] - c;
        T t = sum + y;
        c   = (t - sum) - y;
        sum = t;
    }

    return sum;
}

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

#ifndef MIN
#define MIN(x, y) ((x < y) ? x : y)
#endif

void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
    cudaDeviceProp prop;
    int            device;
    checkCudaErrors(cudaGetDevice(&device));
    checkCudaErrors(cudaGetDeviceProperties(&prop, device));

    if (whichKernel < 3) {
        threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
        blocks  = (n + threads - 1) / threads;
    }
    else {
        threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
        blocks  = (n + (threads * 2 - 1)) / (threads * 2);
    }

    if ((float)threads * blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > prop.maxGridSize[0]) {
        printf("Grid size <%d> exceeds the device capability <%d>, set block size as "
               "%d (original %d)\n",
               blocks,
               prop.maxGridSize[0],
               threads * 2,
               threads);

        blocks /= 2;
        threads *= 2;
    }

    if (whichKernel >= 6) {
        blocks = MIN(maxBlocks, blocks);
    }
}

template <class T>
T benchmarkReduce(int                 n,
                  int                 numThreads,
                  int                 numBlocks,
                  int                 maxThreads,
                  int                 maxBlocks,
                  int                 whichKernel,
                  int                 testIterations,
                  bool                cpuFinalReduction,
                  int                 cpuFinalThreshold,
                  StopWatchInterface *timer,
                  T                  *h_odata,
                  T                  *d_idata,
                  T                  *d_odata)
{
    T    gpu_result   = 0;
    bool needReadBack = true;

    T *d_intermediateSums;
    checkCudaErrors(cudaMalloc((void **)&d_intermediateSums, sizeof(T) * numBlocks));

    for (int i = 0; i < testIterations; ++i) {
        gpu_result = 0;

        cudaDeviceSynchronize();
        sdkStartTimer(&timer);

        reduce<T>(n, numThreads, numBlocks, whichKernel, d_idata, d_odata);
        getLastCudaError("Kernel execution failed");

        if (cpuFinalReduction) {
            checkCudaErrors(cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(T), cudaMemcpyDeviceToHost));

            for (int i = 0; i < numBlocks; i++) {
                gpu_result += h_odata[i];
            }

            needReadBack = false;
        }
        else {
            int s      = numBlocks;
            int kernel = whichKernel;

            while (s > cpuFinalThreshold) {
                int threads = 0, blocks = 0;
                getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks, threads);
                checkCudaErrors(cudaMemcpy(d_intermediateSums, d_odata, s * sizeof(T), cudaMemcpyDeviceToDevice));
                reduce<T>(s, threads, blocks, kernel, d_intermediateSums, d_odata);

                if (kernel < 3) {
                    s = (s + threads - 1) / threads;
                }
                else {
                    s = (s + (threads * 2 - 1)) / (threads * 2);
                }
            }

            if (s > 1) {
                checkCudaErrors(cudaMemcpy(h_odata, d_odata, s * sizeof(T), cudaMemcpyDeviceToHost));

                for (int i = 0; i < s; i++) {
                    gpu_result += h_odata[i];
                }

                needReadBack = false;
            }
        }

        cudaDeviceSynchronize();
        sdkStopTimer(&timer);
    }

    if (needReadBack) {
        checkCudaErrors(cudaMemcpy(&gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost));
    }
    checkCudaErrors(cudaFree(d_intermediateSums));
    return gpu_result;
}

template <class T> void shmoo(int minN, int maxN, int maxThreads, int maxBlocks, ReduceType datatype)
{
    unsigned int bytes = maxN * sizeof(T);

    T *h_idata = (T *)malloc(bytes);

    for (int i = 0; i < maxN; i++) {
        if (datatype == REDUCE_INT) {
            h_idata[i] = (T)(rand() & 0xFF);
        }
        else {
            h_idata[i] = (rand() & 0xFF) / (T)RAND_MAX;
        }
    }

    int maxNumBlocks = MIN(maxN / maxThreads, MAX_BLOCK_DIM_SIZE);

    T *h_odata = (T *)malloc(maxNumBlocks * sizeof(T));

    T *d_idata = NULL;
    T *d_odata = NULL;

    checkCudaErrors(cudaMalloc((void **)&d_idata, bytes));
    checkCudaErrors(cudaMalloc((void **)&d_odata, maxNumBlocks * sizeof(T)));

    checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_odata, h_idata, maxNumBlocks * sizeof(T), cudaMemcpyHostToDevice));

    for (int kernel = 0; kernel < 8; kernel++) {
        reduce<T>(maxN, maxThreads, maxNumBlocks, kernel, d_idata, d_odata);
    }

    int testIterations = 100;

    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);

    printf("Time in milliseconds for various numbers of elements for each "
           "kernel\n\n\n");
    printf("Kernel");

    for (int i = minN; i <= maxN; i *= 2) {
        printf(", %d", i);
    }

    for (int kernel = 0; kernel < 8; kernel++) {
        printf("\n%d", kernel);

        for (int i = minN; i <= maxN; i *= 2) {
            sdkResetTimer(&timer);
            int numBlocks  = 0;
            int numThreads = 0;
            getNumBlocksAndThreads(kernel, i, maxBlocks, maxThreads, numBlocks, numThreads);

            float reduceTime;

            if (numBlocks <= MAX_BLOCK_DIM_SIZE) {
                benchmarkReduce(i,
                                numThreads,
                                numBlocks,
                                maxThreads,
                                maxBlocks,
                                kernel,
                                testIterations,
                                false,
                                1,
                                timer,
                                h_odata,
                                d_idata,
                                d_odata);
                reduceTime = sdkGetAverageTimerValue(&timer);
            }
            else {
                reduceTime = -1.0;
            }

            printf(", %.5f", reduceTime);
        }
    }
    sdkDeleteTimer(&timer);
    free(h_idata);
    free(h_odata);

    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));
}

template <class T> bool runTest(int argc, char **argv, ReduceType datatype)
{
    int  size              = 1 << 24; 
    int  maxThreads        = 256;    
    int  whichKernel       = 7;
    int  maxBlocks         = 64;
    bool cpuFinalReduction = false;
    int  cpuFinalThreshold = 1;

    if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
        size = getCmdLineArgumentInt(argc, (const char **)argv, "n");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "threads")) {
        maxThreads = getCmdLineArgumentInt(argc, (const char **)argv, "threads");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "kernel")) {
        whichKernel = getCmdLineArgumentInt(argc, (const char **)argv, "kernel");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "maxblocks")) {
        maxBlocks = getCmdLineArgumentInt(argc, (const char **)argv, "maxblocks");
    }

    printf("%d elements\n", size);
    printf("%d threads (max)\n", maxThreads);

    cpuFinalReduction = checkCmdLineFlag(argc, (const char **)argv, "cpufinal");

    if (checkCmdLineFlag(argc, (const char **)argv, "cputhresh")) {
        cpuFinalThreshold = getCmdLineArgumentInt(argc, (const char **)argv, "cputhresh");
    }

    bool runShmoo = checkCmdLineFlag(argc, (const char **)argv, "shmoo");

    if (runShmoo) {
        shmoo<T>(1, 33554432, maxThreads, maxBlocks, datatype);
    }
    else {
        unsigned int bytes = size * sizeof(T);

        T *h_idata = (T *)malloc(bytes);

        for (int i = 0; i < size; i++) {
            if (datatype == REDUCE_INT) {
                h_idata[i] = (T)(rand() & 0xFF);
            }
            else {
                h_idata[i] = (rand() & 0xFF) / (T)RAND_MAX;
            }
        }

        int numBlocks  = 0;
        int numThreads = 0;
        getNumBlocksAndThreads(whichKernel, size, maxBlocks, maxThreads, numBlocks, numThreads);

        if (numBlocks == 1) {
            cpuFinalThreshold = 1;
        }

        T *h_odata = (T *)malloc(numBlocks * sizeof(T));

        printf("%d blocks\n\n", numBlocks);

        T *d_idata = NULL;
        T *d_odata = NULL;

        checkCudaErrors(cudaMalloc((void **)&d_idata, bytes));
        checkCudaErrors(cudaMalloc((void **)&d_odata, numBlocks * sizeof(T)));

        checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_odata, h_idata, numBlocks * sizeof(T), cudaMemcpyHostToDevice));

        reduce<T>(size, numThreads, numBlocks, whichKernel, d_idata, d_odata);

        int testIterations = 100;

        StopWatchInterface *timer = 0;
        sdkCreateTimer(&timer);

        T gpu_result = 0;

        gpu_result = benchmarkReduce<T>(size,
                                        numThreads,
                                        numBlocks,
                                        maxThreads,
                                        maxBlocks,
                                        whichKernel,
                                        testIterations,
                                        cpuFinalReduction,
                                        cpuFinalThreshold,
                                        timer,
                                        h_odata,
                                        d_idata,
                                        d_odata);

        double reduceTime = sdkGetAverageTimerValue(&timer) * 1e-3;
        printf("Reduction, Throughput = %.4f GB/s, Time = %.5f s, Size = %u Elements, "
               "NumDevsUsed = %d, Workgroup = %u\n",
               1.0e-9 * ((double)bytes) / reduceTime,
               reduceTime,
               size,
               1,
               numThreads);

        T cpu_result = reduceCPU<T>(h_idata, size);

        int    precision = 0;
        double threshold = 0;
        double diff      = 0;

        if (datatype == REDUCE_INT) {
            printf("\nGPU result = %d\n", (int)gpu_result);
            printf("CPU result = %d\n\n", (int)cpu_result);
        }
        else {
            if (datatype == REDUCE_FLOAT) {
                precision = 8;
                threshold = 1e-8 * size;
            }
            else {
                precision = 12;
                threshold = 1e-12 * size;
            }

            printf("\nGPU result = %.*f\n", precision, (double)gpu_result);
            printf("CPU result = %.*f\n\n", precision, (double)cpu_result);

            diff = fabs((double)gpu_result - (double)cpu_result);
        }
        sdkDeleteTimer(&timer);
        free(h_idata);
        free(h_odata);

        checkCudaErrors(cudaFree(d_idata));
        checkCudaErrors(cudaFree(d_odata));

        if (datatype == REDUCE_INT) {
            return (gpu_result == cpu_result);
        }
        else {
            return (diff < threshold);
        }
    }

    return true;
}