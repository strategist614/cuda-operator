# Softmax

本目录实现数值稳定的 Softmax：对每行先求最大值，再计算 `exp(x - max)` 的和，最后归一化。

| 文件 | 内容 |
| --- | --- |
| `SoftMax_CPU_one_dim.cpp` | 一维 CPU reference |
| `SoftMax_CPU_multi_dim.cpp` | 多维/按行 CPU reference |
| `SoftMax_V0_one_dim.cu` | 单行基础 CUDA 版本 |
| `SoftMax_V0.cu` | CUDA 基线实验 |
| `SoftMax_V1_one_dim_low_threads.cu` | 少线程版本 |
| `SoftMax_V1_one_dim_multi_threads.cu` | 多线程 block reduction |
| `SoftMax_V2_one_dim.cu` | warp/block reduction，每线程多元素 |
| `SoftMax_V2_one_dim_register.cu` | 将中间值保存在寄存器 |
| `online normalizer calculation for softmax.pdf` | online normalizer 参考资料 |

```bash
nvcc -O3 SoftMax_V2_one_dim.cu -o softmax_v2
./softmax_v2
```

部分版本固定 `BLOCK_SIZE=256`、每线程 4 个元素或最大长度 1024。更改 N 时需检查寄存器数组、shared memory 和尾部 mask。性能测试应保留减最大值步骤，并用相对/绝对误差共同检查数值结果。
