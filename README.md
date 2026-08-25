# CUDA Operator Practice

这是一个 CUDA、PyTorch Custom Operator、Triton 和 GPU 编译器学习仓库。内容从 CUDA 线程索引、访存、归约与矩阵转置开始，逐步扩展到 GEMM、归一化、Softmax、Attention、PyTorch C++/CUDA Extension、Triton kernel，以及一个用 Python 实现的教学型 Mini Triton 编译器。

仓库以学习记录和优化实验为主，不是统一构建、完整测试或生产部署的算子库。不同目录的完成度不同：既有能运行并验证结果的示例，也有编译器阶段实验、固定 shape 优化和仍不能编译的 kernel 草稿。

## 项目地图

| 目录 | 主要内容 | 当前定位 |
| --- | --- | --- |
| [`cuda-kernel-practice-core/`](cuda-kernel-practice-core/) | VectorAdd、MatrixMul、Reduction、Transpose，以及 scan/卷积/异步拷贝占位练习 | CUDA 基础与经典优化 |
| [`cuda-kernel-samples/`](cuda-kernel-samples/) | Elementwise、Reduction、Transpose、GEMM 小型示例 | 单文件 kernel 练习 |
| [`cuda-operator-pytorch/`](cuda-operator-pytorch/) | JIT Extension 与可安装的 dispatcher/custom op 示例 | PyTorch C++/CUDA 算子接入 |
| [`operator/`](operator/) | Attention、GEMM、LayerNorm、RMSNorm、Softmax、HWC→CHW normalize | 算子逐版本优化 |
| [`triton/`](triton/) | Elementwise、Norm、Softmax、普通/高优化 GEMM、benchmark 与汇编导出 | Triton 逐元素 kernel、行归约、Tensor Core 与编译结果观察 |
| [`mini-triton/`](mini-triton/) | Python AST、IR、PTX、Tensor/Layout/Thread/Address/Register lowering | 教学型 GPU DSL 编译器 |
| [`cutlass/`](cutlass/) | CUTLASS 学习规划 | 当前为空源码占位 |
| [`job-interview/`](job-interview/) | 面试相关 CUDA kernel 草稿 | 当前为未完成的 GEMM 练习 |

根目录的 [`CUDA_C_Best_Practices_Guide.pdf`](CUDA_C_Best_Practices_Guide.pdf) 是 CUDA 性能优化参考资料。

## 内容概览

### CUDA 基础练习

[`cuda-kernel-practice-core/`](cuda-kernel-practice-core/) 包含：

- VectorAdd：线程索引、内存分配、H2D/D2H 拷贝和结果验证。
- MatrixMul：shared-memory tiled matrix multiplication。
- Reduction：从低效分支/取模版本演进到 warp shuffle 和 cooperative groups。
- Transpose：naive、coalesced、无 bank conflict、diagonal 等版本。
- Scan、可分离卷积、global→shared async copy：当前为待实现练习。

### 独立 CUDA samples

[`cuda-kernel-samples/`](cuda-kernel-samples/) 适合单文件编译：

- Elementwise：Add、ReLU、Sigmoid 及 `float4` 版本。
- Reduce：sum、warp shuffle、Softmax。
- Transpose：naive 二维矩阵转置。
- GEMM：naive、shared-memory tile、warp/register tile、WMMA、optimized WMMA 和 Ampere `cp.async` 六个独立版本。

6 个独立 GEMM 程序均包含 host 端输入、CPU reference、误差检查和 CUDA Event benchmark。01–05 已在 `512 x 512 x 512` 矩阵上通过验证；05 增加 warp 内多输出 tile 复用、128-bit A/B 加载和直接写回，在 RTX 2080 上相比 04 提速 1.406 倍。06 进一步面向 RTX 3090 使用 `sm_86` `cp.async` 双缓冲和 `BK=32` 流水，已通过静态编译，仍需在目标 GPU 上补充运行数据。汇总文件 `gemm.cu` 只包含 device kernel，不能独立编译运行；实现说明、完整汇总代码和实测结果见 [`cuda-kernel-samples/gemm/README.md`](cuda-kernel-samples/gemm/README.md)。

### PyTorch Custom Operator

[`cuda-operator-pytorch/`](cuda-operator-pytorch/) 目前包含两种从 Python 调用 CUDA kernel 的方式：

| 目录 | 构建/注册方式 | 主要内容 |
| --- | --- | --- |
| [`simple/`](cuda-operator-pytorch/simple/) | `torch.utils.cpp_extension.load` JIT 编译与 pybind11 导出 | FP32 CUDA Add、输入检查、当前 device/stream 接入 |
| [`add/`](cuda-operator-pytorch/add/) | `setup.py` + `CUDAExtension` + `TORCH_LIBRARY` dispatcher | CPU/CUDA dispatch、FP16/FP32/FP64 CUDA kernel、FakeTensor、autograd、`opcheck`、`torch.compile` |

`simple/` 适合先理解 Python→C++ binding→CUDA kernel 的最短调用链；`add/` 展示更接近正式 PyTorch custom op 的包结构和注册方式。详细构建、接口约束和测试方法分别见两个目录的 README。

### 算子优化

[`operator/`](operator/) 按版本保存优化过程：

| 算子 | 主要实验 |
| --- | --- |
| GEMM | shared/register/warp tiling、向量化、bank conflict、异步拷贝 |
| Attention | 分阶段 attention、online softmax、tiled/FlashAttention 风格融合 |
| LayerNorm | block/warp reduction、寄存器复用、Welford、访存优化 |
| RMSNorm | sum-of-squares reduction 与逐版本优化 |
| Softmax | 数值稳定、warp/block reduction、online normalizer |
| HWC→CHW | layout conversion、uint8→FP32 和 normalization 融合 |

这些版本常针对固定 hidden size、head dimension、tile 或 GPU 架构。修改 shape 前，应检查静态 shared memory、寄存器数组、尾块 mask 和向量对齐。

### Triton kernels

[`triton/`](triton/) 当前包含：

- `add.py`、`relu.py`、`sigmoid.py`：逐元素 Triton kernel 与 PyTorch reference。
- `layernorm.py`、`rmsnorm.py`、`softmax.py`：最后一维 fused reduction kernel 与 PyTorch reference。
- `normalization_benchmark.py`：统一比较三个 Triton kernel 与 PyTorch 原生实现。
- `gemm.py`：固定 tile 和单 stage 的普通 Triton GEMM。
- `gemm_optimized.py`：面向 `sm_80+` 的 Tensor Core、grouped ordering、autotune 和流水双缓冲 GEMM。
- `gemm_sm86_compile_check.py`、`gemm_benchmark.py`：验证 `mma.sync`/`cp.async` 并测试 GEMM 性能。
- `relu_benchmark.py`：扫描 BLOCK_SIZE，统计寄存器、耗时和有效带宽。
- `relu_256.ptx`、`relu_256.sass`：ReLU 的编译产物，用于观察 global-memory 指令。
- `max.py`：当前为空文件，是后续练习占位。

Triton 使用 block tensor 描述批量连续访问，通常不需要像 CUDA C++ 一样显式写 `float4`。具体访存宽度仍由编译器、对齐、dtype 和目标 GPU 决定，应结合导出的 PTX/SASS 与 profiler 验证。

### Mini Triton 编译器

[`mini-triton/`](mini-triton/) 使用多个小版本展示 GPU DSL 编译过程：

```text
v0.1
Python AST → scalar SSA-like IR → real PTX → CUDA Driver → GPU
    ↓
v0.2
Tensor IR → type/shape verification → placeholder backend
    ↓
v0.3
Layout → Thread → Address → Register lowering → PTX simulation
```

- v0.1 根版本可以生成并通过 CUDA Driver API 运行真实的向量加 PTX。
- v0.2 重点研究 TensorType、PointerType 和 verification/optimization pass。
- v0.3 重点研究 BlockedLayout、thread mapping、地址和寄存器 lowering。
- `mini-triton-final` 内部标记为 v0.3.8，把多个阶段串进一个 pipeline，但 backend 仍是不可执行的 PTX 模拟。
- `mini-triton-v0.4/` 当前为空目录，占位后续开发。

不同小版本不是严格的功能超集，部分阶段会被单独抽出来实验。请从 [`mini-triton/README.md`](mini-triton/README.md) 开始，并以各版本 README 的“当前限制”为准。

### 面试练习草稿

[`job-interview/`](job-interview/) 用于保存面试相关的手写 kernel 练习。当前只有 `code/gemm.cu`，其中 naive 和 shared-memory GEMM 都尚未完成，存在缺少输出参数、变量拼写和未完成加载/计算逻辑等问题，不能作为可运行示例。需要完整版本时，应参考 [`cuda-kernel-samples/gemm/`](cuda-kernel-samples/gemm/) 中的六个独立程序。

## 环境

### CUDA 示例

通常需要：

- NVIDIA GPU 与兼容驱动。
- CUDA Toolkit 和 `nvcc`。
- 与源码特性匹配的 GPU 架构。例如 WMMA 通常要求 SM 70+，`cp.async` 通常要求 SM 80+。

检查环境：

```bash
nvidia-smi
nvcc --version
```

通用单文件编译方式：

```bash
nvcc -O3 -std=c++17 path/to/example.cu -o /tmp/example
/tmp/example
```

并非所有 `.cu` 文件当前都可编译；先阅读所在目录 README。

### PyTorch Extension 环境

[`cuda-operator-pytorch/`](cuda-operator-pytorch/) 额外需要 CUDA-enabled PyTorch、PyTorch 支持的 C++ 编译器和 Ninja。检查环境：

```bash
python3 -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())"
nvcc --version
ninja --version
```

运行简单 JIT 示例：

```bash
python3 cuda-operator-pytorch/simple/test.py
```

构建并测试 dispatcher/custom op 示例：

```bash
cd cuda-operator-pytorch/add
python3 setup.py build_ext --inplace
python3 -m pytest tests -v
```

完整测试需要可用的 NVIDIA GPU。PyTorch、CUDA Toolkit、Python 或平台变化后，应重新构建 `_C` Extension，不要复用其他环境生成的 `.so`。

### Triton 环境

仓库当前使用 Conda `main` 环境，具体 Python、PyTorch、CUDA、Triton 和 NumPy 版本记录在 [`triton/README.md`](triton/README.md)。

```bash
conda activate main
python -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__); print(torch.cuda.is_available())"
```

运行 Triton 示例：

```bash
conda run -n main python triton/add.py
conda run -n main python triton/relu.py
conda run -n main python triton/sigmoid.py
conda run -n main python triton/layernorm.py
conda run -n main python triton/rmsnorm.py
conda run -n main python triton/softmax.py
conda run -n main python triton/normalization_benchmark.py
conda run -n main python triton/gemm.py
conda run -n main python triton/gemm_optimized.py
conda run -n main python triton/gemm_sm86_compile_check.py
conda run -n main python triton/gemm_benchmark.py
```

`torch.cuda.is_available()` 应为 `True`。

### Mini Triton 环境

大多数 AST→IR→文本 backend demo 只依赖 Python 标准库，不要求 GPU。v0.1 的端到端 GPU demo 还需要 NumPy 和 cuda-python。

```bash
(cd mini-triton/mini-triton-v0.1 && conda run -n main python test_compiler.py)
(cd mini-triton/mini-triton-v0.2 && conda run -n main python demo.py)
(cd mini-triton/mini-triton-v0.3 && conda run -n main python demo.py)
```

## 正确性优先

优化 kernel 时建议保持下面的顺序：

```text
CPU/reference 实现
        ↓
naive GPU baseline
        ↓
随机输入与边界 shape 正确性
        ↓
warm-up + 稳定计时
        ↓
profile 找瓶颈
        ↓
一次只引入一种优化
        ↓
重新验证正确性和性能
```

推荐至少检查：

- 最大绝对误差和相对误差。
- 非 tile 整数倍、很小尺寸、非方阵和尾部数据。
- FP32、half/BF16 等不同 dtype 对应的合理容差。
- 每次 launch 的 `cudaGetLastError()` 和同步错误。

## 性能分析

### Nsight Systems

用于观察程序整体时间线：kernel launch、cudaMemcpy、CPU/GPU 同步和 kernel 间空隙。

```bash
nsys profile ./example
```

### Nsight Compute

用于分析单个 kernel：DRAM throughput、occupancy、warp stall、寄存器、shared memory 和 bank conflict。

```bash
ncu ./example
```

### CUDA Event / Triton benchmark

CUDA kernel 可用 CUDA Event 测量，Triton 可用 `triton.testing.do_bench`。计时应排除首次 JIT、内存分配和不需要比较的 H2D/D2H 拷贝。

## 推荐学习路线

1. [`cuda-kernel-practice-core/vectorAdd/`](cuda-kernel-practice-core/vectorAdd/)：掌握线程索引、内存管理和完整 CUDA 程序流程。
2. [`cuda-kernel-samples/elementwise/`](cuda-kernel-samples/elementwise/)：对比标量与 `float4` elementwise。
3. [`cuda-kernel-samples/reduce/`](cuda-kernel-samples/reduce/) 和 [`cuda-kernel-practice-core/reduction/`](cuda-kernel-practice-core/reduction/)：学习 shared memory 与 warp shuffle。
4. [`cuda-kernel-practice-core/transpose/`](cuda-kernel-practice-core/transpose/)：理解 coalescing 和 bank conflict。
5. [`cuda-kernel-samples/gemm/`](cuda-kernel-samples/gemm/)：从 naive 逐步学习 shared/register tiling 和 WMMA。
6. [`operator/LayerNorm/`](operator/LayerNorm/)、[`operator/RMSNorm/`](operator/RMSNorm/)、[`operator/SoftMax/`](operator/SoftMax/)：进入归约型真实算子。
7. [`operator/GEMM/`](operator/GEMM/) 与 [`operator/Attention/`](operator/Attention/)：研究更深入的分块、online softmax 和融合。
8. [`cuda-operator-pytorch/simple/`](cuda-operator-pytorch/simple/)：理解 Python、C++ binding 和 CUDA kernel 的调用链。
9. [`cuda-operator-pytorch/add/`](cuda-operator-pytorch/add/)：学习 dispatcher、autograd、FakeTensor 与 `torch.compile` 接入。
10. [`triton/`](triton/)：用 block tensor 模型重新实现 elementwise kernel。
11. [`mini-triton/`](mini-triton/)：从编译器角度理解 AST、IR、layout、thread 和 PTX。

## 仓库状态与注意事项

- 仓库没有统一构建入口；CUDA 示例多为单文件 `nvcc` 编译，PyTorch Extension、Triton 和 Mini Triton 使用各自的 Python 入口。
- 无扩展名文件多为本地编译产物，不应视为可复现构建结果。
- `cuda-operator-pytorch/add/cuda_operator/_C*.so` 是与 Python、PyTorch、CUDA 和平台绑定的本地构建产物，已通过 `.gitignore` 排除。
- `output/*.mir`、`*.tir`、`*.ptx` 可能是教学快照或占位 backend 输出，不一定能执行。
- `__pycache__/*.pyc` 是 Python 运行缓存。
- 部分源码仍有语法错误、固定尺寸假设、缺少边界处理或资源释放。
- 文档会明确记录当前限制；生产使用前必须补齐测试、错误处理、数值验证和目标 GPU benchmark。
