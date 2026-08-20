# CUDA Operator Practice

这是一个 CUDA/Triton 算子学习仓库，内容从线程模型、访存与归约等基础练习，逐步延伸到 GEMM、归一化、Softmax 和 Attention 优化。代码以教学和实验为主，每个源文件通常同时包含 CPU 参考实现、CUDA kernel、正确性检查或简单 benchmark。

## 目录导航

| 目录 | 内容 |
| --- | --- |
| `cuda-kernel-practice-core/` | CUDA 官方风格基础练习：向量加、矩阵乘、归约、转置等 |
| `cuda-kernel-samples/` | 小型独立 kernel：elementwise、reduce、GEMM、transpose |
| `operator/` | 完整算子及逐版本优化：Attention、GEMM、LayerNorm、RMSNorm、Softmax |
| `triton/` | 使用 Triton 实现的 elementwise kernel |
| `low-level/` | block、warp、SM 等底层概念笔记 |
| `cutlass/` | CUTLASS 学习占位目录 |

仓库根目录的 `CUDA_C_Best_Practices_Guide.pdf` 可作为 CUDA 性能优化参考资料。

## 环境与运行

CUDA 示例一般要求 NVIDIA GPU、兼容驱动和 CUDA Toolkit，可用下面的通用方式单独编译：

```bash
nvcc -O3 path/to/example.cu -o example
./example
```

Triton 示例还需要 Python、PyTorch（CUDA 版本）和 Triton：

```bash
python triton/add.py
python triton/relu.py
```

不同文件对 GPU 架构、固定输入尺寸和编译选项的要求不同，请先阅读对应子目录 README。仓库内已有的无扩展名文件多为本地编译产物，并非统一构建系统的一部分。

## 推荐学习顺序

1. `low-level/` 与 `cuda-kernel-practice-core/vectorAdd/`
2. `cuda-kernel-samples/elementwise/`、`reduce/`、`transpose/`
3. `cuda-kernel-practice-core/reduction/` 与 `transpose/`
4. `operator/LayerNorm/`、`RMSNorm/`、`SoftMax/`
5. `operator/GEMM/` 与 `Attention/`

> 这是练习型仓库，部分文件仍是草稿或只针对特定尺寸。用于生产环境前，应补充边界覆盖、错误处理、数值精度测试和目标 GPU 上的性能验证。
