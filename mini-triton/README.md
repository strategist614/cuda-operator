# Mini Triton

Mini Triton 是一个用于学习 GPU DSL 编译器的渐进式 Python 项目。它借用 Triton 的部分表面语法，通过多个小版本分别探索 Python AST frontend、SSA-like IR、PTX codegen、tensor 类型、verification pass、layout 和 thread mapping。

本项目不是官方 Triton 的子集或兼容实现，也不依赖 Triton 编译器。各版本是相对独立的教学原型，代码优先追求编译流程清晰，而不是功能完整或性能最优。

## 总体演进

```text
v0.1
Python DSL → scalar SSA-like IR → real PTX → CUDA Driver → GPU
    ↓
v0.2
Python DSL → Tensor IR → type/shape passes → placeholder backend
    ↓
v0.3
Tensor IR → layout assignment → thread mapping → placeholder backend
```

v0.1 展示最小端到端 GPU 执行链路；v0.2 和 v0.3 则退回到更高层 IR，分别研究 tensor 语义与 layout lowering。因而更高版本并不代表所有低版本能力都已保留。

## 目录导航

| 目录 | 主要内容 | GPU 执行状态 |
| --- | --- | --- |
| [`mini-triton-v0.1/`](mini-triton-v0.1/) | 标量 SSA IR、PTXEmitter、CUDA Driver API 向量加 | 根版本可生成并运行真实 PTX |
| [`mini-triton-v0.1/mini-triton-v0.1.1/`](mini-triton-v0.1/mini-triton-v0.1.1/) | 模块化 frontend/backend/runtime 骨架 | PTX 不完整，demo 不启动 GPU |
| [`mini-triton-v0.2/`](mini-triton-v0.2/) | v0.2.1 Tensor IR 基础版本 | backend 只输出 opcode 注释 |
| [`mini-triton-v0.2/mini-triton-v0.2.2/`](mini-triton-v0.2/mini-triton-v0.2.2/) | TypeCheckPass、shape/dtype validation | backend 只输出 opcode 注释 |
| [`mini-triton-v0.2/mini-triton-v0.2.3/`](mini-triton-v0.2/mini-triton-v0.2.3/) | TypeShapePass、SimplifyPass，v0.2 收尾 | backend 只输出 opcode 注释 |
| [`mini-triton-v0.3/`](mini-triton-v0.3/) | BlockedLayout、LayoutPass、ThreadMappingPass | backend 只 dump lowering 后的 IR |

每个目录中的 README 记录该版本的源码结构、支持语法、运行方法和具体限制。

## 各阶段解决的问题

### v0.1：最小可执行编译链路

根 v0.1 把一个受限 Python kernel lowering 为标量 SSA-like IR，并生成包含 kernel 参数、寄存器、地址计算、predicate、global load/store 的 PTX。

它把教学 DSL 简化映射到 CUDA 一维执行模型：

| DSL | v0.1 lowering |
| --- | --- |
| `program_id(0)` | `blockIdx.x` / `%ctaid.x` |
| `arange(BLOCK)` | `threadIdx.x` / `%tid.x` |
| `offset < N` | PTX predicate |
| `load(..., mask)` | predicated `ld.global.f32` |
| `store(..., mask)` | predicated `st.global.f32` |

`demo.py` 通过 cuda-python 调用 CUDA Driver API，完成 PTX JIT、显存分配、kernel launch、结果拷回和 NumPy 验证。

子目录 v0.1.1 尝试把 frontend、IR、backend、compiler 和 runtime 拆成独立模块，但其 backend 是较早的骨架：kernel entry 没有参数、寄存器未声明、load/store 只是注释。因此应把它用于阅读模块边界，而不是用于 GPU correctness。

### v0.2：Tensor IR 与编译 pass

v0.2 将“一个 CUDA 线程处理一个标量”的表示提升为“一条 IR operation 处理一个 tensor block”：

```text
arange(256)
    ↓
tensor<256 × i32> offsets
    ↓
tensor_load X / tensor_load Y
    ↓
tensor_add
    ↓
tensor_store Z
```

版本演进：

- v0.2.1：加入 PointerType、TensorType、函数参数 lowering 和局部 shape inference。
- v0.2.2：加入 TypeCheckPass，验证 add 的 shape/dtype 与 load 的 pointer/offset 类型。
- v0.2.3：shape 改用 tuple，引入 TypeShapePass 与空实现 SimplifyPass。

v0.2 的重点是 frontend、类型传播和 pass pipeline。三个版本都没有定义 tensor block 到 CUDA threads/warps 的映射，因此 backend 只输出注释。

### v0.3：Layout 与线程映射

v0.3 开始回答 v0.2 遗留的问题：一个逻辑 tensor block 应怎样分配给 GPU 线程。

Frontend 为 `arange(256)` 创建尚未分配 layout 的 `LayoutTensorType`。随后：

1. `LayoutPass` 为 tensor 设置 `BlockedLayout(threads=128, elements_per_thread=2)`。
2. `ThreadMappingPass` 在 `make_tensor` 后插入 `thread_mapping` operation。
3. backend 打印附带 layout 和 mapping 的 IR。

当前 128 个线程 × 每线程 2 个元素正好覆盖 256 个元素，但这些信息还没有 lowering 成 `%tid.x`、寄存器或真实内存指令。

## 示例 DSL

各版本主要围绕向量加法展开。Tensor IR 版本使用：

```python
def add_kernel(X, Y, Z):
    offsets = arange(256)
    x = load(X, offsets)
    y = load(Y, offsets)
    z = x + y
    store(Z, offsets, z)
```

逻辑含义为：

```text
Z[0:256] = X[0:256] + Y[0:256]
```

真正 Triton 会让多个 program 覆盖完整输入并用 mask 保护尾部。当前 v0.2/v0.3 示例没有 `program_id`、N 或 mask，只描述一个固定 tensor block；只有根 v0.1 实现了多 block 索引和尾部 predicate。

## 环境

只运行 AST→IR→文本 backend 的版本通常只需要 Python 标准库。当前仓库使用的 Conda 环境为 `main`。

根 v0.1 的 GPU demo 还需要：

- NVIDIA GPU 与兼容驱动。
- NumPy。
- `cuda-python` / `cuda-bindings`。

当前环境对应版本记录在 v0.1 README 中。可以先检查：

```bash
conda run -n main python -c "from cuda.bindings import driver; import numpy; print(numpy.__version__)"
```

## 快速运行

建议从仓库根目录使用子 shell，确保同名模块互不干扰，生成文件也落在各自版本目录。

### v0.1

仅查看 IR/PTX：

```bash
(cd mini-triton/mini-triton-v0.1 && conda run -n main python test_compiler.py)
```

运行端到端 GPU demo：

```bash
(cd mini-triton/mini-triton-v0.1 && conda run -n main python demo.py)
```

成功时应输出 `correct: True`。

运行模块化 v0.1.1 编译演示：

```bash
(cd mini-triton/mini-triton-v0.1/mini-triton-v0.1.1 && conda run -n main python demo.py)
```

### v0.2

```bash
(cd mini-triton/mini-triton-v0.2 && conda run -n main python demo.py)
(cd mini-triton/mini-triton-v0.2/mini-triton-v0.2.2 && conda run -n main python demo.py)
(cd mini-triton/mini-triton-v0.2/mini-triton-v0.2.3 && conda run -n main python demo.py)
```

v0.2.1 和 v0.2.2 会更新各自 `output/` 下的 `.tir/.ptx` 快照；v0.2.3 只打印结果。

### v0.3

```bash
(cd mini-triton/mini-triton-v0.3 && conda run -n main python demo.py)
```

它会打印带 BlockedLayout 和 thread_mapping 的 Layout IR，以及 backend 的注释输出。

## 如何阅读源码

推荐顺序：

1. 运行 v0.1 `test_compiler.py`，理解 DSL、IR 和真实 PTX 的对应关系。
2. 阅读 v0.1 `demo.py`，理解 CUDA Driver API 如何加载和启动 PTX。
3. 阅读 v0.2.1 的 `tensor_types.py`、`ir.py`、`frontend.py`，理解 Tensor IR。
4. 对比 v0.2.2/v0.2.3 的 `passes.py`，理解 verification 与 optimization stage。
5. 阅读 v0.3 `layout.py` 和 `passes.py`，理解 tensor 到线程的映射为何需要独立 layout。
6. 最后查看各版本 `backend.py`，区分真实 codegen 与占位 dump。

在每个版本中，可以沿以下顺序跟踪一次编译：

```text
demo.py
  → compiler.py
    → frontend.py
      → ir.py / tensor_types.py / layout.py
    → passes.py
    → backend.py
```

## 生成文件

仓库中可能包含：

- `output/*.mir`：标量 Mini IR 快照。
- `output/*.tir`：Tensor IR 快照。
- `output/*.ptx`：真实 PTX 或占位 backend 输出，必须结合版本判断。
- `__pycache__/*.pyc`：Python 自动生成的缓存。

不要仅凭 `.ptx` 扩展名判断文件可执行。根 v0.1 的 PTX emitter 生成完整 kernel；v0.1.1、v0.2 和 v0.3 中的 backend 仍可能只输出不完整 PTX 或注释。

## 共同限制

- DSL 只支持极少的 Python AST 节点，不是通用 Python 编译器。
- dtype、pointer、shape 和错误诊断仍很简化。
- 没有广播、归约、多维 tensor、控制流或完整类型系统。
- 没有成熟的 pass manager、分析依赖和优化基础设施。
- 除根 v0.1 外，其他版本尚未打通真实 GPU codegen/runtime。
- 没有系统化单元测试、负例测试、benchmark 或多 GPU 架构验证。
- 多个目录存在同名 Python 模块，应从目标版本目录运行，避免错误导入。

## 后续方向

从 v0.3 继续开发时，可以按以下顺序推进：

1. 为 kernel 参数建立 scalar、`ptr<dtype>` 和 tensor 类型系统。
2. 补齐 program id、grid、N、mask 与多 program 语义。
3. 验证 `shape == threads × elements_per_thread` 等 layout 约束。
4. 把 thread mapping lowering 成 lane/thread 级 IR。
5. 实现 SSA value 到 PTX 寄存器的分配。
6. Lower tensor load/add/store 为带 predicate 的真实 PTX。
7. 接入 CUDA Driver runtime 并建立 GPU correctness tests。
8. 再逐步加入向量化、共享内存、归约和自动调优。
