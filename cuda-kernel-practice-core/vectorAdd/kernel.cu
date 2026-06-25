#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i] + 0.0f;
    }
}

int main()
{
    cudaError_t err = cudaSuccess;

    int numElements = 50000;
    size_t size = numElements * sizeof(float);

    printf("[Vector addition of %d elements]\n", numElements);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numElements; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;

    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy input vectors from host memory to device memory\n");

    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector A from host to device: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector B from host to device: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel execution failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy output vector from device memory to host memory\n");

    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector C from device to host: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numElements; i++) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done!\n");
    return 0;
}