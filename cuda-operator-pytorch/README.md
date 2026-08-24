# PyTorch CUDA Extension：Elementwise Add

本目录演示如何使用 PyTorch C++/CUDA Extension 编写并即时编译一个自定义 CUDA 算子。示例实现两个同形状 FP32 CUDA Tensor 的逐元素加法：

```text
output[i] = a[i] + b[i]
```

这是一个用于学习 PyTorch、C++ binding 与 CUDA kernel 连接方式的最小示例，不是完整的算子库。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `binding.cpp` | 检查输入并通过 pybind11 导出 Python 接口 `forward` |
| `my_kernel.cu` | 实现 CUDA kernel、Tensor 连续化、输出分配及 kernel launch |
| `test.py` | 使用 `torch.utils.cpp_extension.load` JIT 编译扩展并验证结果 |

调用流程如下：

```text
test.py
  -> JIT 编译 binding.cpp + my_kernel.cu
  -> my_operator.forward(a, b)
  -> C++ 输入检查
  -> add_cuda(a, b)
  -> add_kernel<<<blocks, 256, 0, current_stream>>>
  -> 返回 output
```

## 环境要求

- NVIDIA GPU 和可用的 NVIDIA 驱动。
- Python 3。
- 支持 CUDA 的 PyTorch，而不是 CPU-only PyTorch。
- CUDA Toolkit，包括 `nvcc`。
- PyTorch 支持的 C++ 编译器。
- Ninja，用于加速和管理 JIT Extension 构建。

先检查环境：

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
nvcc --version
ninja --version
```

`torch.cuda.is_available()` 应输出 `True`。`nvcc` 所属 CUDA Toolkit 需要与当前 PyTorch 使用的 CUDA 版本兼容，否则 JIT 编译时可能出现 CUDA 版本不匹配错误。

## 运行

进入本目录后执行：

```bash
python3 test.py
```

首次运行时，`torch.utils.cpp_extension.load` 会编译：

```text
binding.cpp
my_kernel.cu
```

并加载生成的 Python 扩展 `my_operator_cuda`。编译产物默认保存在 PyTorch Extension 缓存目录中，后续运行在源码和编译参数未变化时会复用缓存。

测试程序创建两个 `1024 x 1024` 的 FP32 CUDA Tensor，分别计算 PyTorch reference 和自定义算子结果：

```python
reference = a + b
output = my_operator.forward(a, b)

torch.testing.assert_close(output, reference)
```

验证通过后会输出：

```text
结果正确
```

由于 `test.py` 设置了 `verbose=True`，首次运行还会显示 Extension 的详细构建日志；最后会打印输出 Tensor。

## Python 接口

JIT 编译后，扩展提供：

```python
output = my_operator.forward(a, b)
```

输入约束：

| 项目 | 要求 |
| --- | --- |
| Device | `a`、`b` 都必须是 CUDA Tensor |
| GPU | 两个 Tensor 必须位于同一块 GPU |
| Dtype | 两个 Tensor 都必须是 `torch.float32` |
| Shape | 两个 Tensor 的形状必须完全相同，不支持 broadcasting |
| Layout | 可以传入非连续 Tensor，内部会调用 `.contiguous()` |

不满足约束时，`binding.cpp` 中的 `TORCH_CHECK` 会抛出带具体原因的异常。

## CUDA 实现

### 线程映射

kernel 使用一维 grid，每个线程处理一个元素：

```cpp
int64_t index =
    static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

if (index < n) {
    output[index] = a[index] + b[index];
}
```

每个 block 使用 256 个线程，block 数量向上取整：

```cpp
constexpr int threads = 256;
int blocks = static_cast<int>((n + threads - 1) / threads);
```

边界判断使元素数量不需要是 256 的整数倍；空 Tensor 会直接返回，不启动 kernel。

### Device 与 stream

`add_cuda` 使用 `c10::cuda::CUDAGuard` 切换到输入 Tensor 所在 GPU，因此支持当前进程中的多 GPU 场景：

```cpp
c10::cuda::CUDAGuard device_guard(a.device());
```

kernel 在 PyTorch 当前 CUDA stream 上启动，而不是隐式使用默认 stream：

```cpp
cudaStream_t stream =
    at::cuda::getCurrentCUDAStream(a.get_device()).stream();
```

这样可以保持与同一 stream 上其他 PyTorch CUDA 操作的执行顺序。kernel launch 后通过 `C10_CUDA_KERNEL_LAUNCH_CHECK()` 检查启动错误。

## JIT 编译配置

`test.py` 使用以下方式构建扩展：

```python
my_operator = load(
    name="my_operator_cuda",
    sources=[
        str(root / "binding.cpp"),
        str(root / "my_kernel.cu"),
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)
```

`TORCH_EXTENSION_NAME` 会在构建时展开为 `my_operator_cuda`，因此 `binding.cpp` 不需要硬编码 Python module 名称。

如果希望把构建缓存放在指定目录，可以设置：

```bash
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions python3 test.py
```

## 当前限制

- 只支持 CUDA FP32 Tensor。
- 只实现逐元素加法，不支持 broadcasting、类型提升或 CPU fallback。
- 当前 pybind 接口只实现 forward，没有注册 autograd backward；不要把它当作可训练模型中的完整可微算子。
- 每次调用都会为非连续输入创建连续副本，这会带来额外开销。
- 测试只检查单一尺寸和 dtype，尚未覆盖空 Tensor、非连续 Tensor、多 GPU、错误输入以及不同形状。
- 当前没有 benchmark，不能仅凭实现形式判断它比 `torch.add` 更快。

## 常见问题

### `ModuleNotFoundError: No module named 'torch'`

当前 Python 环境没有安装 PyTorch。请安装支持本机驱动和 CUDA 环境的 PyTorch，并确认运行 `test.py` 时使用的是同一个 Python 环境。

### 找不到 `nvcc` 或 `CUDA_HOME`

确认 CUDA Toolkit 已安装，并检查：

```bash
which nvcc
echo "$CUDA_HOME"
```

必要时把 CUDA Toolkit 的 `bin` 目录加入 `PATH`，并将 `CUDA_HOME` 指向 Toolkit 根目录。

### CUDA 版本不匹配

分别检查 `torch.version.cuda` 和 `nvcc --version`。JIT Extension 使用本机 Toolkit 编译 CUDA 源码，两者需要兼容。

### 修改源码后仍加载旧结果

`load()` 会根据源码和构建参数管理版本与缓存。可设置一个新的 `TORCH_EXTENSIONS_DIR` 后重新运行，以确认使用全新构建目录。
