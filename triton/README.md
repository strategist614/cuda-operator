# Triton Kernels

本目录使用 Triton 编写简单的逐元素 GPU kernel，帮助对照 CUDA 中的 grid、block、线程索引和越界判断。

| 文件 | 状态 |
| --- | --- |
| `add.py` | 完整实现，包含 PyTorch reference 与正确性检查 |
| `relu.py` | 完整实现，包含 PyTorch reference 与正确性检查 |
| `sigmoid.py` | 仅有 kernel 函数骨架，尚未实现 |

## Conda 环境

本仓库当前使用 Conda 的 `main` 环境：

| 组件 | 版本 |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.12.1+cu132 |
| PyTorch CUDA | 13.2 |
| Triton | 3.7.1 |
| NumPy | 2.2.6 |

激活环境并检查 GPU 是否可用：

```bash
conda activate main
python -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__); print(torch.cuda.is_available())"
```

`torch.cuda.is_available()` 应输出 `True`。如果是 `False`，请检查 NVIDIA 驱动、GPU 访问权限，以及 PyTorch 所带 CUDA runtime 与驱动的兼容性。

## 运行示例

在 `main` 环境中运行：

```bash
conda activate main
cd triton
python add.py
python relu.py
```

每个 Triton program 处理 `BLOCK_SIZE` 个元素，通过 `tl.program_id` 定位数据块，使用 mask 保护最后一个不完整块。示例假设输入位于 CUDA、shape 相同且内存连续。
