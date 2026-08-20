# Elementwise Kernels

本目录实现逐元素 Add、ReLU 和 Sigmoid，每个算子分别提供标量版本和一次处理 4 个 `float` 的向量化版本。

| 文件 | 说明 |
| --- | --- |
| `add.cu` / `addvec.cu` | `c = a + b`；向量版包含尾部标量处理 |
| `relu.cu` / `reluvec.cu` | `y = max(x, 0)` |
| `sigmoid.cu` / `sigmoidvec.cu` | `y = 1 / (1 + exp(-x))` |

```bash
nvcc -O3 add.cu -o add && ./add
nvcc -O3 addvec.cu -o addvec && ./addvec
```

其他文件可用相同方式编译。`float4` 版本依赖 16 字节对齐；当前 ReLU/Sigmoid 向量版只处理完整的 4 元素分组，若把测试长度改成非 4 的倍数，需要补尾部处理。Sigmoid 标量版对正负输入分支计算以改善数值稳定性，而当前向量版未采用该分支。
