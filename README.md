# CUDA Operator Practice

这是一个 CUDA、Triton 和 GPU 编译器学习仓库。内容从 CUDA 线程模型、访存、归约与矩阵转置开始，逐步扩展到 GEMM、归一化、Softmax、Attention、Triton kernel，以及一个用 Python 实现的教学型 Mini Triton 编译器。

仓库以学习记录和优化实验为主，不是统一构建、完整测试或生产部署的算子库。不同目录的完成度不同：既有能运行并验证结果的示例，也有编译器阶段实验、固定 shape 优化和仍不能编译的 kernel 草稿。

## 项目地图

| 目录 | 主要内容 | 当前定位 |
| --- | --- | --- |
| [`low-level/`](low-level/) | block、warp、SM、occupancy、访存等底层概念 | 基础笔记 |
| [`cuda-kernel-practice-core/`](cuda-kernel-practice-core/) | VectorAdd、MatrixMul、Reduction、Transpose，以及 scan/卷积/异步拷贝占位练习 | CUDA 基础与经典优化 |
| [`cuda-kernel-samples/`](cuda-kernel-samples/) | Elementwise、Reduction、Transpose、GEMM 小型示例 | 单文件 kernel 练习 |
| [`operator/`](operator/) | Attention、GEMM、LayerNorm、RMSNorm、Softmax、HWC→CHW normalize | 算子逐版本优化 |
| [`triton/`](triton/) | Add、ReLU、Sigmoid、ReLU benchmark、PTX/SASS 导出 | Triton kernel 与编译结果观察 |
| [`mini-triton/`](mini-triton/) | Python AST、IR、PTX、Tensor/Layout/Thread/Address/Register lowering | 教学型 GPU DSL 编译器 |
| [`cutlass/`](cutlass/) | CUTLASS 学习规划 | 当前为空源码占位 |

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
- GEMM：naive、shared tile、warp/register tile 和 WMMA 草稿。

GEMM 目录中的 `gemm.cu` 当前仍存在编译错误，详细状态和修复顺序见 [`cuda-kernel-samples/gemm/README.md`](cuda-kernel-samples/gemm/README.md)。目录中的旧可执行文件不保证与当前源码一致。

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

1. [`low-level/`](low-level/)：理解 block、warp、SM 和访存。
2. [`cuda-kernel-practice-core/vectorAdd/`](cuda-kernel-practice-core/vectorAdd/)：掌握完整 CUDA 程序流程。
3. [`cuda-kernel-samples/elementwise/`](cuda-kernel-samples/elementwise/)：标量与 `float4` elementwise。
4. [`cuda-kernel-samples/reduce/`](cuda-kernel-samples/reduce/) 和 [`cuda-kernel-practice-core/reduction/`](cuda-kernel-practice-core/reduction/)：shared memory 与 warp shuffle。
5. [`cuda-kernel-practice-core/transpose/`](cuda-kernel-practice-core/transpose/)：coalescing 和 bank conflict。
6. [`operator/LayerNorm/`](operator/LayerNorm/)、[`operator/RMSNorm/`](operator/RMSNorm/)、[`operator/SoftMax/`](operator/SoftMax/)：归约型真实算子。
7. [`operator/GEMM/`](operator/GEMM/)：shared/register/warp tiling。
8. [`operator/Attention/`](operator/Attention/)：online softmax 与融合。
9. [`triton/`](triton/)：用 block tensor 模型重新实现 elementwise kernel。
10. [`mini-triton/`](mini-triton/)：从编译器角度理解 AST、IR、layout、thread 和 PTX。

## 仓库状态与注意事项

- 仓库没有统一 CMake/Makefile；多数示例独立编译运行。
- 无扩展名文件多为本地编译产物，不应视为可复现构建结果。
- `output/*.mir`、`*.tir`、`*.ptx` 可能是教学快照或占位 backend 输出，不一定能执行。
- `__pycache__/*.pyc` 是 Python 运行缓存。
- 部分源码仍有语法错误、固定尺寸假设、缺少边界处理或资源释放。
- 文档会明确记录当前限制；生产使用前必须补齐测试、错误处理、数值验证和目标 GPU benchmark。
