# MatrixMul

`matrixMul.cu` 是 tiled 矩阵乘示例，使用 shared memory 复用 A、B tile，并包含主机端初始化、计时和结果检查；`helper_cuda.h` 提供 CUDA 错误检查辅助代码。

```bash
nvcc -O3 matrixMul.cu -o matrixMul
./matrixMul
```

示例对矩阵尺寸和 block 配置有预设，更改参数时要同步检查 tile 整除条件与边界处理。

### `CUDA`走高效 `DMA` 通道

* 使用 `pinned memory` 
  普通写法：
  ```c++
  float *h_A = (float*)malloc(size);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  ```
  更快写法：
  ```c++
  float *h_A;
  cudaMallocHost((void**)&h_A, size);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaFreeHost(h_A);
  ```
  原因：
  ```
  malloc:
    普通 Host 内存 -> 临时 pinned buffer -> GPU

  cudaMallocHost:
    pinned Host 内存 -> GPU
  ```
* 使用 `cudaMemcpyAsync + stream`

* 复用 `pinned memory`

* 用双缓冲/多缓冲

> `CUDA`里面通常是 `x`表示列，`y`表示行
