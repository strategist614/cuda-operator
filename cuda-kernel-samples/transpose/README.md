# Transpose Sample

`transpose_naive.cu` 展示二维 grid/block 下的基础矩阵转置：每个线程读取输入 `(row, col)`，写到输出 `(col, row)`，并在主机端打印结果。

```bash
nvcc -O3 transpose_naive.cu -o transpose_naive
./transpose_naive
```

naive 写入跨步较大，主要用于建立正确性基线。进一步优化可参考 `cuda-kernel-practice-core/transpose/` 中的 shared-memory tiled、coalesced 和无 bank-conflict 版本。
