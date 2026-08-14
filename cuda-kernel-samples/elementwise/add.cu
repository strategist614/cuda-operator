#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <iostream>


// 1. 向上取整
#define CEIL(a, b) ((a + b - 1) / (b))

// 2. FLOAT4，用于向量化访存，以下两种都可以
// c写法
#define FLOAT4(value) *(float4*)(&(value))

// c++写法
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])


int blocksize = 1024;

int gridsize = CEIL(N, blocksize);

void __global__ add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
float *a, *b, *c;
int main(){

    return 0;
}



