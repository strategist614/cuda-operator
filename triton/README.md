# Triton Kernels

本目录使用 Triton 编写简单的逐元素 GPU kernel，帮助对照 CUDA 中的 grid、block、线程索引和越界判断。

| 文件 | 状态 |
| --- | --- |
| `add.py` | 完整实现，包含 PyTorch reference 与正确性检查 |
| `relu.py` | 完整实现，包含 PyTorch reference 与正确性检查 |
| `sigmoid.py` | Sigmoid kernel 与 Python wrapper 草稿，测试代码待修正 |

## Sigmoid

`sigmoid.py` 计划实现逐元素 Sigmoid：

```text
y = 1 / (1 + exp(-x))
```

与 Add、ReLU 示例相同，每个 Triton program 根据 `tl.program_id(axis=0)` 定位一个数据块，生成 `BLOCK_SIZE=256` 个 offsets，并通过 mask 保护最后一个不足 256 个元素的数据块。Python wrapper 负责创建输出 tensor、计算 grid 并启动 kernel。

当前文件仍处于调试阶段，直接运行前需要修正以下内容：

- `offsets` 应是 Triton tensor，而不是只含一个元素的 Python tuple。
- `tl.store` 需要同时传入输出地址、待写入的 `y` 和 mask。
- wrapper 的类型注解应使用 `torch.Tensor`，或显式导入 `Tensor`。
- `main()` 中的 reference 应改为 `torch.sigmoid(x)`。
- `main()` 应调用 `triton_sigmoid(x)`，而不是 `triton_relu(x)`。

修正后可使用 `torch.testing.assert_close` 对比 Triton 输出和 PyTorch reference，并运行：

```bash
python sigmoid.py
```

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
