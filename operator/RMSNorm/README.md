# RMSNorm

RMSNorm 按行计算均方根并缩放：`y = x * rsqrt(mean(x²) + eps) * weight`。与 LayerNorm 相比，它不减均值，也没有 beta 项。

| 文件 | 主要思路 |
| --- | --- |
| `RMSNorm_V0.cu` | 基础 shared-memory reduction |
| `RMSNorm_V1.cu` | warp shuffle/block reduction |
| `RMSNorm_V2.cu` | 在 V1 基础上调整数据处理与复用 |
| `RMSNorm_V3.cu` | 更进一步的寄存器/访存优化 |

```bash
nvcc -O3 RMSNorm_V0.cu -o rmsnorm_v0
./rmsnorm_v0
```

各文件均为独立程序，并包含 CPU reference、误差检查和 benchmark。版本名与 kernel 内部函数名不一定完全一致，应以文件和 launch 处为准。修改 hidden size 后需检查固定的每线程处理数量、向量加载对齐和归约边界。
