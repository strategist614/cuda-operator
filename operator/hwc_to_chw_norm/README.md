## HWC to CHW

一个像素连续存入 `3` 个通道

```
pixel 0: input[0], input[1], input[2]
pixel 1: input[3], input[4], input[5]
pixel 2: input[6], input[7], input[8]
...
```

### naive

`nsys`结果：

`CUDA API Summary`:

```
Time (%)  Total Time (ns)  Num Calls  Avg (ns)       Name
94.4      322201485        2          161100742.5    cudaMalloc
2.4       8251011          1011       8161.2         cudaLaunchKernel
1.2       4120736          1          4120736.0      cudaEventSynchronize
1.2       4008376          2          2004188.0      cudaDeviceSynchronize
0.6       2070582          2          1035291.0      cudaMemcpy
0.2       579788           2          289894.0       cudaFree
```
第一次使用 `cudaMalloc` 会有点慢：
```
真实推理时不要把 cudaMalloc / cudaFree 放进每张图处理流程里。

程序开始：
    cudaMalloc 一次

每张图：
    cudaMemcpy
    kernel
    模型推理

程序结束：
    cudaFree
```

`Kernel Summary`:
```
CUDA GPU Kernel Summary

Total Time: 8849351 ns
Instances : 1011
Avg       : 8753.1 ns
Med       : 6528.0 ns
Min       : 5824 ns
Max       : 1654989 ns
Name      : hwc_to_chw_norm_kernel
```

`MemOps Summary`:
```
CUDA GPU MemOps Summary

Device-to-Host:
Total Time: 1.382731 ms
Size      : 4.915 MB

Host-to-Device:
Total Time: 0.082496 ms
Size      : 1.229 MB

H2D 输入拷贝：0.082 ms
D2H 输出拷贝：1.383 ms
kernel 平均：约 0.006~0.009 ms

把结果从 GPU 拷回 CPU 比 kernel 本身慢很多
```
总结：
```
1. kernel 功能正确：Max error = 0
2. kernel 本身很快：平均约 8.75 us，中位数约 6.53 us
3. cudaMalloc/cudaFree 不是 kernel 性能，应该排除出 benchmark
4. Device-to-Host 拷贝明显比 kernel 慢，真实推理中应该避免
5. 下一步不要先优化 kernel，而是优化整体流程
```