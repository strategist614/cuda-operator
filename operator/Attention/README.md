# Attention

本目录从拆分 kernel 的 scaled dot-product attention 逐步演进到 tiled online-softmax 与 FlashAttention 风格融合实现。核心公式为 `softmax(QKᵀ / sqrt(d))V`。

| 文件 | 主要内容 |
| --- | --- |
| `attention_v0.cu` | GEMM、Softmax、输出 GEMM 分阶段基线 |
| `attention_v1.cu` | tiled QK GEMM 与后续阶段优化 |
| `attention_v2_online.cu` | online softmax，减少中间统计量遍历 |
| `attention_v3_tiled_online.cu` | Q/KV 分块与 online 更新 |
| `flashattention.cu` | 固定 `HEAD_DIM=64` 的融合实现实验 |
| `attention_gemm_v9.cu` | 更完整的高性能 GEMM/Attention 实验 |
| `flash attention.pdf` | 算法参考资料 |

单文件通常自带 reference、correctness check 和 benchmark：

```bash
nvcc -O3 -arch=sm_80 attention_v3_tiled_online.cu -o attention_v3
./attention_v3
```

`-arch` 必须与实际 GPU 匹配。多个版本使用固定 head dimension、tile 和 warp 布局；修改 shape 前应检查静态 shared memory、向量对齐、尾块和 mask。Attention 对数值误差敏感，性能比较时也应报告精度容差并排除初始化和首次 launch 开销。
