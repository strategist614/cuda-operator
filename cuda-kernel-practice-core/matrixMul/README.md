# MatrixMul

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