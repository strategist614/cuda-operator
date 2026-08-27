# CUDA Attention Samples

本目录使用 CUDA C++ 实现两个 scaled dot-product self-attention 前向算子，并通过 PyTorch C++/CUDA extension 暴露给 Python：

- `cuda_basic`：依次执行 `QK^T`、row-wise softmax 和 `PV` 三个 CUDA kernel。
- `cuda_flash_wmma`：以 `16 x 16` tile 执行在线 softmax，`QK^T` 和 `PV` 都使用 WMMA/Tensor Core，并且不生成完整的 attention score 矩阵。
- `cuda_flash_online_fallback`：FP32、非 16 对齐尺寸或不支持 WMMA 的 GPU 使用标量在线 softmax CUDA kernel。
- `torch_sdpa`：调用 PyTorch 自带的 `torch.nn.functional.scaled_dot_product_attention`，作为正确性和性能参照。

代码中没有使用 Triton。Python 仅用于 JIT 编译 CUDA extension、生成输入、校验结果和执行 benchmark。

## 文件说明

| 文件 | 内容 |
| --- | --- |
| `binding.cpp` | PyTorch extension 绑定、输入形状和数据类型检查 |
| `attention_cuda.cu` | 基础 Attention、WMMA 分块融合 Attention 和通用在线 softmax fallback |
| `benchmark.py` | 编译扩展，以 CUDA Event 对三个 provider 计时并检查结果 |

输入采用连续的 `[batch, heads, sequence, head_dim]` 布局。当前实现支持 FP16、FP32，要求 Q、K、V 形状一致且 `head_dim <= 256`，计算的是无 mask、无 dropout 的非 causal self-attention 前向过程：

```text
scores = Q K^T / sqrt(head_dim)
P      = softmax(scores, dim=-1)
O      = P V
```

## 基础版

基础版由三个 kernel 组成：

1. 每个线程计算一个 `QK^T` score，并以 FP32 写入 `[B,H,S,S]` 中间张量。
2. 每个 block 处理一行，先求最大值，再计算稳定 softmax；block 内归约使用 warp shuffle 和 shared memory。
3. 每个线程计算一个输出元素，沿 sequence 维累加 `P*V`。

它的逻辑直观，但 FP32 score 中间张量需要 `4*B*H*S*S` 字节，显存占用随序列长度平方增长，而且 score 会被多次写入和读出全局显存。

## WMMA 分块融合版

FP16 且 `sequence`、`head_dim` 都是 16 的倍数时，融合版自动走 WMMA 路径。每个 CUDA block 使用一个 warp 处理 16 行 query，并以 `16 x 16` key/value tile 遍历序列：

```text
Q[16,D] x K_tile^T[D,16] -> score_tile[16,16]  (WMMA)
score_tile -> online softmax tile
P_tile[16,16] x V_tile[16,D] -> output update   (WMMA)
```

两次矩阵乘法的 FP16 operands 都通过 Tensor Core 执行，WMMA accumulator、score、softmax 状态和输出累加使用 FP32。`PV` 前会把当前 tile 的未归一化概率转换为 FP16，以满足 WMMA operand 类型要求。

遍历 key/value tile 时不保存完整 score，而是为每个 query row 维护运行最大值 `m`、指数和 `l` 以及未归一化输出 `o`。对当前 tile 更新：

```text
m_new = max(m, s)
alpha = exp(m - m_new)
beta  = exp(s - m_new)
l_new = alpha * l + beta
o_new = alpha * o + beta * v
```

遍历结束后输出 `o/l`。这种在线更新保持数值稳定，同时把额外工作空间从 `O(B*H*S*S)` 降为每个 block 的 `O(16*D)` shared-memory 状态。

当前 WMMA 优化包括 `16 x 16` Q/K/V 分块、16 行 query 对 K/V tile 的复用、Tensor Core MMA、FP32 accumulator、在线 softmax 和单 kernel 融合。FP32、非 16 对齐尺寸或 compute capability 低于 7.0 时，代码自动调用逐 key 的标量在线 softmax kernel，以保证接口仍可使用。

这里仍是学习版 FlashAttention，而不是完整的生产级 FlashAttention-2。它只有一个 warp/block，没有 `cp.async`、shared-memory K/V 双缓冲、多 stage pipeline、warp specialization、反向传播、causal mask 或 dropout。因此 WMMA 版会显著快于原始标量版，但通常仍达不到 PyTorch/cuDNN 高度优化后端的吞吐。

## 运行 benchmark

环境需要 CUDA Toolkit、CUDA GPU，以及安装了 CUDA 版 PyTorch 的 Python 环境。在本目录运行：

```bash
python benchmark.py
```

默认参数是 `[B,H,S,D] = [1,8,512,64]`、FP16、10 次 warmup 和 50 次计时。可自行修改：

```bash
python benchmark.py \
    --batch 1 \
    --heads 8 \
    --sequence 1024 \
    --head-dim 64 \
    --dtype fp16 \
    --warmup 10 \
    --repetitions 50
```

首次运行会在 `/tmp/cuda_attention_torch_extensions` 中 JIT 编译 extension。计时使用 CUDA Event；报告的近似 GFLOPS 只统计 `QK^T` 和 `PV` 的乘加，即 `4*B*H*S*S*D`，不计 softmax 指数与归约操作。

## 当前测试结果

测试环境：NVIDIA GeForce RTX 2080（compute capability 7.5）、CUDA Toolkit 13.2、PyTorch 2.12.1+cu132。

```text
device=NVIDIA GeForce RTX 2080
shape=1x8x512x64 dtype=fp16
warmup=10 repetitions=50
basic_score_intermediate_mib=8.0000
provider=cuda_basic average_ms=4.34106354 gflops=123.6727 max_abs_error=0.00012207 status=PASS
provider=cuda_flash_wmma average_ms=0.23502975 gflops=2284.2678 max_abs_error=0.00012207 status=PASS
provider=torch_sdpa average_ms=0.04914944 gflops=10923.2352 status=PASS
flash_vs_basic=18.4703x
flash_vs_torch=0.2091x
status=PASS
```

在该尺寸上，WMMA 融合版相对基础版提速约 `18.47x`，相对改造前的标量在线版本 `2.84323059 ms` 提速约 `12.10x`，并省去 8 MiB 的 score 中间张量。PyTorch SDPA 仍约快 `4.78x`，因为它会根据硬件和输入选择更深度分块、向量化和流水线化的 CUDA 后端；本示例的目标是展示算法与内存访问差异，而不是替代生产算子。

额外使用 `[1,2,128,128]`、FP16 验证 WMMA 路径，最大绝对误差为 `2.4414e-4`；使用 `[1,2,129,48]`、FP32 验证通用 fallback，最大绝对误差为 `1.6e-7`，两组均通过。

性能数据会受到 GPU 型号、频率、CUDA/PyTorch 版本和输入尺寸影响；更换环境后应重新运行 benchmark，而不是直接沿用这里的数值。
