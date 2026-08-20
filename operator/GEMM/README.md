# GEMM Optimization

本目录以行主序 FP32 GEMM 为主线，按版本逐步增加数据复用和访存优化。

| 版本 | 主要思路 |
| --- | --- |
| `GEMM_V0.cu` | 每线程计算一个 C 元素的 naive 基线 |
| `GEMM_V1.cu` | 16×16 shared-memory tiling |
| `GEMM_V2.cu` | 每线程 2×2 register tile |
| `GEMM_V3.cu` | 64×64 block tile、4×4 thread tile |
| `GEMM_V4.cu` | `float4` 向量化加载/存储 |
| `GEMM_V5.cu` | 双缓冲流水线 |
| `GEMM_V5_cp_async.cu` | `cp.async` 异步 global→shared 拷贝实验 |
| `GEMM_V6_WarpTailing.cu` | 128×64 block 与 warp tiling |
| `GEMM_V7_BankConflict.cu` | shared-memory padding/布局，降低 bank conflict |
| `GEMM_V8_SharedTranspose.cu` | shared-memory 转置布局 |

```bash
nvcc -O3 -arch=sm_80 GEMM_V0.cu -o GEMM_V0
./GEMM_V0
```

异步拷贝版本通常要求 SM 80+。每个文件是独立实验，常量、benchmark 次数和正确性容差定义在各自源码中。比较版本时应使用同一 M/N/K、相同编译架构、warm-up 和计时方法，并同时观察寄存器、shared memory、occupancy 与有效 TFLOPS。
