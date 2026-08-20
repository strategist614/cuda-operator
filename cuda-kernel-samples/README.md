# CUDA Kernel Samples

这里是一组小而独立的 CUDA 练习，每个 `.cu` 文件通常自带 `main`、输入构造和简单输出/验证，适合单文件编译。

| 目录 | 示例 |
| --- | --- |
| `elementwise/` | add、ReLU、Sigmoid 及 `float4` 向量化版本 |
| `reduce/` | sum、warp shuffle reduction、Softmax |
| `gemm/` | naive 与 tiled GEMM 草稿 |
| `transpose/` | naive 矩阵转置 |

```bash
nvcc -O3 elementwise/add.cu -o elementwise/add
./elementwise/add
```

目录中已有的无扩展名文件是编译产物。部分代码是学习草稿，边界处理和资源释放并不完整；具体限制见各子目录 README。
