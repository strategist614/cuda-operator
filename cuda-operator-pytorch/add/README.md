# PyTorch Custom Operator：Vector Add

本目录实现一个可安装的 PyTorch 自定义算子 `cuda_operator::vector_add`，用于计算两个同形状 Tensor 的逐元素加法：

```text
output[i] = a[i] + b[i]
```

示例不仅包含 CUDA kernel，还演示了 PyTorch dispatcher、CPU/CUDA backend 注册、FakeTensor、autograd、`torch.library.opcheck` 和 `torch.compile` 的完整接入方式。

## 目录结构

```text
add/
├── csrc/
│   ├── operator.cpp       # 算子 schema、CPU/CUDA dispatch 注册
│   └── operator_cuda.cu   # CUDA kernel 和 CUDA backend 实现
├── cuda_operator/
│   ├── __init__.py        # Python API、FakeTensor 和 autograd 注册
│   └── _C*.so             # 本地构建产物，文件名与 Python/平台相关
├── tests/
│   └── test_operator.py   # CPU、CUDA、梯度和 torch.compile 测试
├── setup.py               # C++/CUDA Extension 构建配置
└── README.md
```

## 调用流程

```text
import cuda_operator
  -> import cuda_operator._C
  -> 执行 TORCH_LIBRARY 静态注册
  -> 注册 cuda_operator::vector_add schema
  -> 注册 CPU implementation
  -> 注册 CUDA implementation
  -> 注册 FakeTensor implementation 和 autograd formula

cuda_operator.vector_add(a, b)
  -> torch.ops.cuda_operator.vector_add(a, b)
  -> PyTorch dispatcher 根据 Tensor device 选择 CPU 或 CUDA backend
```

`PYBIND11_MODULE` 本身不导出 Python 函数。它创建一个可被 Python import 的扩展模块；真正的算子入口由 PyTorch dispatcher 管理。

## 支持范围

| 能力 | CPU backend | CUDA backend |
| --- | --- | --- |
| 逐元素加法 | 支持，内部调用 `at::add` | 支持，自定义 CUDA kernel |
| Dtype | 两个输入 dtype 相同，并由 `at::add` 处理 | `float16`、`float32`、`float64` |
| Shape | 两个输入必须完全相同 | 两个输入必须完全相同 |
| Broadcasting | 不支持 | 不支持 |
| 非连续 Tensor | 由 `at::add` 处理 | 内部转换为连续 Tensor |
| 空 Tensor | 支持 | 支持，不启动 kernel |
| Autograd | 支持 | 支持 |
| `torch.compile` | 已注册 FakeTensor 实现 | 已注册 FakeTensor 实现 |

CUDA 输入必须位于同一块 GPU。输出与输入具有相同 shape 和 dtype；CUDA backend 返回 contiguous layout。

## 环境要求

- Linux 或其他受 PyTorch C++ Extension 支持的系统。
- Python 3。
- 支持本示例所用 `torch.library` API 的较新 PyTorch。
- NVIDIA GPU 和可用的 NVIDIA 驱动。
- CUDA-enabled PyTorch。
- CUDA Toolkit，包括 `nvcc`。
- PyTorch 支持的 C++ 编译器。
- Ninja 和 pytest。

检查环境：

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
nvcc --version
ninja --version
python3 -m pytest --version
```

运行完整测试需要 `torch.cuda.is_available()` 返回 `True`。`nvcc` 所属 CUDA Toolkit 还需要与当前 PyTorch 使用的 CUDA 版本兼容。

## 构建

进入本目录：

```bash
cd cuda-operator-pytorch/add
```

### 方式一：原地构建

```bash
python3 setup.py build_ext --inplace
```

构建完成后会在 `cuda_operator/` 下生成类似文件：

```text
_C.cpython-310-x86_64-linux-gnu.so
```

具体文件名取决于 Python 版本、操作系统和 CPU 架构。仓库中已有的 `.so` 只适用于它的原始构建环境；切换 Python、PyTorch、CUDA 或平台后应重新构建。

### 方式二：以 editable package 安装

```bash
python3 -m pip install -e .
```

这种方式便于从其他目录执行 `import cuda_operator`。修改 C++/CUDA 源码后仍需重新构建 Extension。

### 指定 GPU 架构

如需显式控制编译目标，可在构建前设置 `TORCH_CUDA_ARCH_LIST`：

```bash
TORCH_CUDA_ARCH_LIST="7.5" python3 setup.py build_ext --inplace
```

应将 `7.5` 替换为实际 GPU 的 compute capability。

## 使用方法

```python
import torch
import cuda_operator

a = torch.randn(1024, device="cuda", dtype=torch.float32)
b = torch.randn_like(a)

output = cuda_operator.vector_add(a, b)
torch.testing.assert_close(output, a + b)
```

也可以直接调用 dispatcher 中的算子：

```python
output = torch.ops.cuda_operator.vector_add(a, b)
```

CPU 调用使用相同的 Python 接口：

```python
a = torch.randn(32, 64)
b = torch.randn_like(a)
output = cuda_operator.vector_add(a, b)
```

## 实现说明

### Schema 与 dispatcher

`operator.cpp` 先定义统一的算子 schema：

```cpp
TORCH_LIBRARY(cuda_operator, module)
{
    module.def("vector_add(Tensor a, Tensor b) -> Tensor");
}
```

随后为 CPU 和 CUDA dispatch key 注册不同实现：

```cpp
TORCH_LIBRARY_IMPL(cuda_operator, CPU, module)
{
    module.impl("vector_add", TORCH_FN(vector_add_cpu));
}

TORCH_LIBRARY_IMPL(cuda_operator, CUDA, module)
{
    module.impl("vector_add", TORCH_FN(vector_add_cuda));
}
```

因此用户只调用一个算子，PyTorch 会根据输入 Tensor 的 device 自动选择 backend。

### CUDA kernel

CUDA kernel 使用一维 grid，每个线程处理一个元素：

```cpp
template <typename scalar_t>
__global__ void vector_add_kernel(
    const scalar_t* a,
    const scalar_t* b,
    scalar_t* output,
    int64_t numel)
{
    const int64_t index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index < numel) {
        output[index] = a[index] + b[index];
    }
}
```

每个 block 使用 256 个线程，grid size 向上取整。`AT_DISPATCH_FLOATING_TYPES_AND_HALF` 根据输入 dtype 实例化 half、float 或 double kernel。

### Device、stream 与错误检查

CUDA backend 使用 `c10::cuda::CUDAGuard` 确保 kernel 在输入所在 GPU 上启动：

```cpp
const c10::cuda::CUDAGuard device_guard(a.device());
```

kernel 接入 PyTorch 当前 CUDA stream：

```cpp
const cudaStream_t stream =
    at::cuda::getCurrentCUDAStream(a.get_device()).stream();
```

这样可以保持与同一 stream 上其他 PyTorch CUDA 操作的执行顺序。`C10_CUDA_KERNEL_LAUNCH_CHECK()` 检查 launch error，但不会进行全局同步。

### 非连续输入

CUDA kernel 按连续地址访问数据，因此 backend 会先执行：

```cpp
const auto a_contiguous = a.contiguous();
const auto b_contiguous = b.contiguous();
```

这保证了结果正确，但非连续输入会产生额外复制开销。输出由 `at::empty_like(a_contiguous)` 分配，因此是连续布局。

### FakeTensor 与 `torch.compile`

`cuda_operator/__init__.py` 为算子注册 fake implementation：

```python
@torch.library.register_fake("cuda_operator::vector_add")
def _vector_add_fake(a, b):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == b.dtype)
    torch._check(a.device == b.device)

    return torch.empty_like(
        a,
        memory_format=torch.contiguous_format,
    )
```

FakeTensor 执行不运行真实 kernel，只根据输入推导输出的 shape、dtype、device 和 layout。这使 `opcheck` 和 `torch.compile` 能够追踪该算子。

### Autograd

对于 `output = a + b`：

```text
d(output) / d(a) = 1
d(output) / d(b) = 1
```

因此 backward 直接把 `grad_output` 返回给两个输入：

```python
def _backward(ctx, grad_output):
    return grad_output, grad_output
```

该公式通过 `torch.library.register_autograd` 注册，并由 double-precision `gradcheck` 验证。

## 测试

构建完成后运行完整测试：

```bash
python3 -m pytest tests -v
```

也可以直接运行测试脚本：

```bash
PYTHONPATH=. python3 tests/test_operator.py
```

当前测试覆盖：

| 测试 | 检查内容 |
| --- | --- |
| `test_forward_cpu` | CPU dispatcher 与 `torch.add` 结果一致 |
| `test_forward_cuda` | CUDA FP32 forward 正确性 |
| `test_non_contiguous_cuda` | 非连续 CUDA 输入正确性 |
| `test_opcheck` | schema、FakeTensor、autograd registration 等算子注册约定 |
| `test_gradcheck` | double 输入下的数值梯度正确性 |
| `test_torch_compile` | `fullgraph=True` 下能够编译和执行 |

仅运行 CPU 测试：

```bash
python3 -m pytest tests/test_operator.py::test_forward_cpu -v
```

所有测试通过时，直接执行脚本会输出：

```text
All tests passed.
```

## 构建配置

`setup.py` 使用 `CUDAExtension` 同时编译 C++ 和 CUDA 源码：

```python
CUDAExtension(
    name="cuda_operator._C",
    sources=[
        "csrc/operator.cpp",
        "csrc/operator_cuda.cu",
    ],
    extra_compile_args={
        "cxx": ["-O3"],
        "nvcc": ["-O3", "-lineinfo"],
    },
)
```

`-lineinfo` 为 CUDA 二进制保留源码行映射，便于 Nsight Compute、Nsight Systems 和 Compute Sanitizer 定位问题，同时不会像 `-G` 那样完全关闭优化。

## 当前限制

- 不支持 broadcasting；输入 shape 必须完全一致。
- CUDA backend 只支持 float16、float32 和 float64。
- 暂无 benchmark，不能据此认为自定义 kernel 比 `torch.add` 更快。
- CUDA 实现是标量加载/存储，尚未加入 `float4` 等向量化优化。
- 非连续 CUDA 输入需要额外的 contiguous copy。
- 测试未覆盖多个 CUDA device、空 Tensor、错误输入和全部 dtype 组合。
- Python fake implementation 只检查 shape、dtype 和 device 一致性，没有重复 CUDA backend 的全部 dtype 限制。

## 常见问题

### `ModuleNotFoundError: No module named 'torch'`

请在当前 Python 环境安装 PyTorch，并确认构建和运行测试使用同一个环境。

### 找不到 `nvcc` 或 `CUDA_HOME`

```bash
which nvcc
echo "$CUDA_HOME"
```

确认 CUDA Toolkit 的 `bin` 目录位于 `PATH` 中，并让 `CUDA_HOME` 指向正确的 Toolkit 根目录。

### CUDA Toolkit 与 PyTorch CUDA 版本不兼容

```bash
python3 -c "import torch; print(torch.version.cuda)"
nvcc --version
```

JIT/Extension 构建使用本机 CUDA Toolkit，而 PyTorch wheel 使用其构建时指定的 CUDA 版本；两者需要兼容。

### `import cuda_operator` 加载了旧扩展

确认当前工作目录、editable install 路径和 `cuda_operator._C` 指向预期位置。修改 C++/CUDA 源码后，重新执行：

```bash
python3 setup.py build_ext --inplace
```

如果更换了 Python、PyTorch 或 CUDA 环境，不应继续复用旧的 `_C*.so`。
