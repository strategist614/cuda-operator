# Triton Kernels

本目录使用 Triton 编写简单的逐元素 GPU kernel，帮助对照 CUDA 中的 grid、block、线程索引和越界判断。

| 文件 | 状态 |
| --- | --- |
| `add.py` | 完整实现，包含 PyTorch reference 与正确性检查 |
| `relu.py` | 完整实现，包含 PyTorch reference 与正确性检查 |
| `sigmoid.py` | 完整实现，包含 PyTorch reference 与正确性检查 |

## Add

`add.py` 实现两个相同 shape 的连续 CUDA tensor 的逐元素相加：

```text
output[i] = x[i] + y[i]
```

`add_kernel` 通过 `tl.program_id(axis=0)` 获取当前 program 编号，然后生成该 program 负责的 offsets。kernel 分别加载 `x` 和 `y`，完成加法后写入输出。`triton_add()` 检查输入设备、shape 和连续性，使用 `torch.empty_like()` 创建输出，并通过 `triton.cdiv()` 计算 grid。

测试使用 1000 个元素，故意让长度不能被 `BLOCK_SIZE=256` 整除，以验证最后一个 program 的 mask。结果使用 `torch.testing.assert_close` 与 `x + y` 比较。

```bash
python add.py
```

## ReLU

`relu.py` 实现逐元素 ReLU：

```text
y[i] = max(x[i], 0)
```

`relu_kernel` 加载输入后使用 `tl.maximum(x, 0.0)` 完成计算。`triton_relu()` 要求输入位于 CUDA 且内存连续，按 256 个元素一个 program 启动 kernel。测试输入由 `torch.randn()` 生成，因此同时覆盖正数和负数，并使用 `torch.relu(x)` 作为 reference。

```bash
python relu.py
```

## Sigmoid

`sigmoid.py` 实现逐元素 Sigmoid：

```text
y = 1 / (1 + exp(-x))
```

`sigmoid_kernel` 使用 `tl.exp(-x)` 计算指数，然后将结果写回输出。`triton_sigmoid()` 负责创建输出、计算一维 grid 并启动 kernel。测试使用 `torch.sigmoid(x)` 作为 reference，并通过 `torch.testing.assert_close` 检查结果。

当前实现表达式清晰，适合学习 Triton 基础；如果用于极端输入或更严格的数值场景，还应针对大正数、大负数和不同 dtype 单独测试误差。

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
python sigmoid.py
```

## 三个示例的共同执行模型

三个示例都使用一维 grid，每个 Triton program 处理 `BLOCK_SIZE=256` 个元素：

```text
pid         = tl.program_id(axis=0)
offsets     = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask        = offsets < n_elements
program 数量 = ceil(n_elements / BLOCK_SIZE)
```

mask 用于保护最后一个不完整的数据块，避免越界读写。示例假设输入位于 CUDA 且内存连续；Add 还要求两个输入 shape 相同。这些脚本目前用于正确性学习，没有包含 warm-up、重复计时或与 PyTorch 的性能 benchmark。

## 为什么这里不需要显式向量化加载

在 CUDA C++ 中，常用 `float4` 让一个线程一次加载或存储 4 个连续的 `float`：

```cpp
float4 value = reinterpret_cast<const float4*>(input)[index];
```

本目录的 Add、ReLU 和 Sigmoid 没有写类似的显式向量类型，因为 Triton 的编程模型不同。下面的 `offsets` 不是单个标量下标，而是一个包含 `BLOCK_SIZE` 个连续下标的 Triton tensor：

```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
x = tl.load(x_ptr + offsets, mask=mask)
```

一次 `tl.load` 描述了整个 program 对一批连续元素的加载。编译器可以根据地址连续性、数据类型、对齐和目标 GPU，把这些访问降级为合适的向量化指令与合并的 global-memory transactions。因此，这三个 kernel 已经表达了批量连续访问，没有必要为了模仿 CUDA 再手写 `float4`。

对这三个逐元素算子来说，连续 offsets 还带来以下好处：

- 相邻执行单元访问相邻地址，便于 global memory coalescing。
- Add 的两个输入和一个输出都按相同连续下标访问。
- ReLU、Sigmoid 在加载后直接计算并写回，不需要 shared memory 做数据复用。
- mask 可以安全处理元素数量不是 `BLOCK_SIZE` 整数倍的尾部。

不过，不显式使用 `float4` 不代表性能一定自动达到最优。实际优化时仍需 benchmark，并根据 dtype、shape 和 GPU 调整 `BLOCK_SIZE`、`num_warps`；对于二维或非连续数据，还需要设计合适的 block 布局，并可通过 `tl.multiple_of`、`tl.max_contiguous` 等提示向编译器提供已知的对齐或连续性信息。是否生成理想访存指令，应以 Triton 编译结果和 profiler 数据为准。
