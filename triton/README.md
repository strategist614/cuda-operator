# Triton Kernels

本目录使用 Triton 编写简单的逐元素 GPU kernel，帮助对照 CUDA 中的 grid、block、线程索引和越界判断。

| 文件 | 状态 |
| --- | --- |
| `add.py` | 完整实现，包含 PyTorch reference 与正确性检查 |
| `relu.py` | 完整实现，包含 PyTorch reference 与正确性检查 |
| `sigmoid.py` | 仅有 kernel 函数骨架，尚未实现 |

环境需要支持 CUDA 的 PyTorch 和与其兼容的 Triton。运行：

```bash
python add.py
python relu.py
```

每个 Triton program 处理 `BLOCK_SIZE` 个元素，通过 `tl.program_id` 定位数据块，使用 mask 保护最后一个不完整块。示例假设输入位于 CUDA、shape 相同且内存连续。
