# Reduction Samples

本目录用于练习跨线程归约及 Softmax。

| 文件 | 内容 |
| --- | --- |
| `sum.cu` | 基础逐元素相加示例 |
| `sum_warp.cu` | `float4` 加载、线程内累加和 warp shuffle reduction |
| `sum_interview.cu` | 更紧凑的 warp-shuffle 求和练习 |
| `softmax.cu` | 先求最大值和指数和，再归一化的 Softmax 示例 |

```bash
nvcc -O3 sum_warp.cu -o sum_warp
./sum_warp
```

归约结果只在指定线程/输出位置有效，阅读时注意 block 数、warp 数和 partial sum 的二次归约。Softmax 必须先减去行最大值以避免指数溢出；修改输入长度时应同步检查 shared memory 大小和 kernel 的尺寸假设。
