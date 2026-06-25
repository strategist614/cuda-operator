### VectorAdd
* `cudaMalloc` 是在 `GPU` 中分配显存的
* `lockIdx.x` 表示`block`在整个`grid`的编号
* `blockDim.x` 表示每个`block`有多少个线程
* `blockIdx.x * blockDim.x` 表示当前到多少个`block`
* `blockIdx.x * blockDim.x + threadIdx.x` 表示偏移了多少个线程

`CPU` 端 `malloc` 分配 `h_A` / `h_B` / `h_C`
初始化输入数据
`GPU` 端 `cudaMalloc` 分配 `d_A` / `d_B` / `d_C`
`cudaMemcpy` 把 `A`、`B` 从 `CPU` 拷到 `GPU`
启动 `kernel`：
```c++
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
cudaDeviceSynchronize() 等待 kernel 执行完成
```
把结果 `d_C` 拷回 `h_C`
`CPU` 端验证    结果
释放显存和内存

`vector add`本身是访存带宽受限算子，不是计算受限算子
