# Mini Triton v0.2 Series

本目录保存 Mini Triton v0.2 系列的三个教学版本。整个系列围绕 tensor-level IR 展开：先在 v0.2.1 建立 Tensor IR，再在 v0.2.2 接入类型检查，最后由 v0.2.3 加入明确的 type/shape 与 simplify pass pipeline。

## 学习目标

v0.2 系列关注的是“如何在编译器 IR 中表达一整块 tensor”，而不是立即生成高性能 GPU 指令。通过三个版本可以观察：

1. Python 函数参数如何进入 kernel IR。
2. `arange`、`load`、`+`、`store` 如何变成 tensor operation。
3. shape 和 dtype 如何沿 operation 传播。
4. 类型检查为什么适合放在独立 pass 中。
5. frontend、verification、optimization、backend 如何组成编译 pipeline。

本系列不是官方 Triton 的子集或兼容实现；DSL、IR 和类型字符串都是仓库中的教学设计。

## 版本导航

| 版本 | 位置 | 主要变化 | 输出行为 |
| --- | --- | --- | --- |
| v0.2.1 | 当前目录 | PointerType、TensorType、函数参数 lowering、tensor load/add/store | 写入 `output/*.tir` 和 `output/*.ptx` |
| v0.2.2 | [`mini-triton-v0.2.2/`](mini-triton-v0.2.2/) | 新增 ScalarType 与 TypeCheckPass，检查 add shape/dtype 和 load 类型 | 写入 `output/*.tir` 和 `output/*.ptx` |
| v0.2.3 | [`mini-triton-v0.2.3/`](mini-triton-v0.2.3/) | tuple shape、TypeShapePass、SimplifyPass；v0.2 收尾版本 | 只返回并打印 IR/PTX，不自动写文件 |

建议按 v0.2.1 → v0.2.2 → v0.2.3 的顺序阅读。每个子版本都是独立的小型 Python 项目，使用各自目录中的同名模块，不应跨目录混合导入。

## 能力矩阵

| 能力 | v0.2.1 | v0.2.2 | v0.2.3 |
| --- | --- | --- | --- |
| 从函数签名提取 pointer 参数 | 支持 | 支持 | 支持 |
| `arange(size)` tensor | list shape | list shape | tuple shape |
| `tensor_load` / `tensor_add` / `tensor_store` | 支持 | 支持 | 支持 |
| add shape/dtype 检查 | 不支持 | `TypeCheckPass` | `TypeShapePass` |
| load pointer/offset 检查 | 不支持 | 支持 | 不支持 |
| 独立 simplify/optimizer pass | 无 | 无 | 有接口，当前为空实现 |
| 自动保存 IR/PTX | 支持 | 支持 | 不支持 |
| 真实 PTX codegen | 不支持 | 不支持 | 不支持 |
| CUDA runtime/GPU 执行 | 不支持 | 不支持 | 不支持 |

## 系列演进

```text
v0.2.1
Tensor IR + pointer arguments + local shape inference
    ↓
v0.2.2
TypeCheckPass + tensor operation validation
    ↓
v0.2.3
TypeShapePass + SimplifyPass pipeline
```

三个版本的 backend 都仍是占位实现：生成的 `.ptx` 或 PTX 字符串只包含 opcode 注释，不是可由 CUDA Driver API 加载的 PTX module。v0.2 系列的重点是 frontend、类型/shape 和 pass pipeline，而不是 GPU 执行。

## v0.2.1 基础版本

Mini Triton v0.2.1 是一个教学型 GPU DSL 编译器原型。本版本开始在 IR 中保留 Triton 风格的整块 tensor 语义：`arange()` 创建索引 tensor，`load()` 返回数据 tensor，`x + y` 表示 tensor 加法，`store()` 写回整个 tensor。

目录名是 `mini-triton-v0.2`，代码和生成文件中的版本标识为 `v0.2.1`。

当前目标是展示 Python DSL 到 Tensor IR 的 lowering，并将编译产物写入文件。PTX backend 仍是占位实现，只输出 operation 名称的注释，尚不能生成可加载执行的 GPU kernel。

## 相比 v0.1 的变化

- 从函数签名读取 kernel 参数，并把它们保存到 `FunctionIR.args`。
- 新增 `PointerType`，用于表示 X、Y、Z 等指针参数。
- 新增 `TensorType`，同时记录元素类型和 tensor shape。
- `arange(size)` 生成 `tensor<[size]xi32>` 索引值。
- `load(ptr, offsets)` 生成与 offsets shape 相同的 FP32 tensor。
- 普通 `+` 被 lowering 为 `tensor_add`，结果继承输入 tensor 类型。
- `store(ptr, offsets, value)` 被保留为无返回值的 `tensor_store`。
- IR dump 会同时打印函数参数、tensor 类型和 operation。

## 编译流程

```text
Python DSL source
        ↓ ast.parse()
Python AST
        ↓ Frontend
Tensor IR
        ├── dump → output/<name>.tir
        ↓ PTXBackend
占位 PTX 文本 → output/<name>.ptx
```

当前没有 optimizer/pass pipeline，也没有 runtime：

```text
Frontend().compile(src)
        ↓
FunctionIR
        ↓
PTXBackend().emit(ir)
```

## v0.2.1 文件结构

| 文件或目录 | 作用 |
| --- | --- |
| `tensor_types.py` | 定义 tensor 与 pointer 类型 |
| `ir.py` | 定义 `Value`、`Op`、`FunctionIR` |
| `frontend.py` | 将受限 Python AST lowering 为 Tensor IR |
| `backend.py` | 遍历 IR；目前只生成 opcode 注释 |
| `compiler.py` | 编译入口，并保存 `.tir` 与 `.ptx` 文件 |
| `demo.py` | 构造并编译一个 256 元素的 tensor add kernel |
| `output/add_kernel.tir` | demo 生成的 Tensor IR 快照 |
| `output/add_kernel.ptx` | demo 生成的占位 PTX 快照 |

`__pycache__/` 是 Python 自动生成的字节码缓存，不属于编译器实现。

## v0.2.1 Demo DSL

`demo.py` 编译下面的源码：

```python
def add_kernel(X, Y, Z):
    offsets = arange(256)
    x = load(X, offsets)
    y = load(Y, offsets)
    z = x + y
    store(Z, offsets, z)
```

这段 DSL 描述一个长度为 256 的向量加法：

```text
Z[offsets] = X[offsets] + Y[offsets]
offsets = [0, 1, ..., 255]
```

与真正的 Triton kernel 相比，当前 DSL 没有 `program_id`，因此只描述从 offset 0 开始的一个 tensor block；它还没有 N 和 mask，无法处理任意长度或尾部越界。

## v0.2.1 类型系统

### PointerType

`PointerType` 当前只表示一个通用的 `ptr`：

```python
@dataclass
class PointerType:
    dtype: str = "ptr"
```

Frontend 会把函数签名中的每个参数都定义为 pointer，包括 X、Y、Z。当前还不能表达 `ptr<f32>`、地址空间、const 属性或标量参数。

### TensorType

`TensorType` 记录元素类型与 shape：

```python
@dataclass
class TensorType:
    dtype: str
    shape: list
```

例如：

```text
tensor<[256]xi32>  # 256 个 i32 offset
tensor<[256]xf32>  # 256 个 FP32 数据元素
```

这是 IR 层的逻辑类型，当前 backend 还没有决定它应如何映射到 CUDA thread、warp、寄存器或向量指令。

## v0.2.1 IR 结构

### Value

`Value` 包含 SSA-like 名称和类型。Frontend 依次生成 `%0`、`%1`、`%2` 等名称：

```python
@dataclass
class Value:
    name: str
    ty: object
```

### Op

`Op` 保存 opcode、可选结果和 operands：

```python
@dataclass
class Op:
    opcode: str
    result: object
    args: tuple
```

`arange`、`tensor_load`、`tensor_add` 会生成结果；`tensor_store` 没有返回值。

### FunctionIR

`FunctionIR` 保存函数名、参数和 operation 列表。`dump()` 生成易读的 `.tir` 文本，但这不是 MLIR/Triton IR 的标准序列化格式。

demo 生成的 Tensor IR 为：

```text
func @add_kernel
  arg X:ptr
  arg Y:ptr
  arg Z:ptr
  %0:tensor<[256]xi32> = arange 256
  %1:tensor<[256]xf32> = tensor_load X:ptr %0:tensor<[256]xi32>
  %2:tensor<[256]xf32> = tensor_load Y:ptr %0:tensor<[256]xi32>
  %3:tensor<[256]xf32> = tensor_add %1:tensor<[256]xf32> %2:tensor<[256]xf32>
  tensor_store Z:ptr %0:tensor<[256]xi32> %3:tensor<[256]xf32>
```

## v0.2.1 Frontend lowering

`Frontend.compile()` 使用 `ast.parse()` 解析源码，并选择第一个 `FunctionDef`。它遍历函数参数，把参数注册到 symbol table，然后逐条编译函数体。

当前支持：

| Python/DSL 写法 | Tensor IR |
| --- | --- |
| 函数参数 | `Value(name, PointerType())` |
| `offsets = arange(size)` | `arange -> tensor<[size]xi32>` |
| `x = load(ptr, offsets)` | `tensor_load -> tensor<offsets.shape × f32>` |
| `z = x + y` | `tensor_add`，结果类型继承左操作数 |
| `store(ptr, offsets, value)` | `tensor_store` |
| 简单赋值 | 将结果写入 frontend symbol table |

只有 `ast.Add` 会被处理，其他二元运算最终会落入不支持路径。Frontend 也没有检查 tensor shape、dtype 或 pointer/value operand 是否兼容。

## v0.2.1 Backend 当前状态

`PTXBackend.emit()` 目前只输出版本注释和每条 IR 的 opcode：

```text
// Mini Triton v0.2.1
// Tensor IR
// arange
// tensor_load
// tensor_load
// tensor_add
// tensor_store
```

这个文件虽然使用 `.ptx` 扩展名，但不包含 PTX header、kernel entry、参数 ABI、寄存器、地址计算或任何可执行指令，不能交给 CUDA Driver API 加载。

要生成真实 PTX，backend 还需要决定 tensor operation 到 CUDA 执行模型的映射，例如：

- `arange(256)` 如何分配给 thread/warp。
- pointer 参数如何生成 `.param .u64` 和地址寄存器。
- tensor load/store 如何生成合并访存和尾部 predicate。
- tensor add 如何针对每个 lane 生成 FP32 指令。
- 多个 program 如何利用 `blockIdx` 覆盖完整输入。

## v0.2.1 Compiler API

公开编译入口是：

```python
from compiler import compile_kernel

ir, ptx = compile_kernel(src, "add_kernel")
```

`compile_kernel()` 执行以下步骤：

1. 通过 `Frontend().compile(src)` 构建 Tensor IR。
2. 在当前工作目录创建 `output/`。
3. 将 `ir.dump()` 写入 `output/<name>.tir`。
4. 调用 `PTXBackend().emit(ir)` 生成占位 PTX。
5. 将结果写入 `output/<name>.ptx`。
6. 返回 `(ir, ptx)`。

传入的 `name` 控制输出文件名，IR 中的函数名仍来自 DSL 源码。输出路径相对于执行命令时的当前工作目录。

## 运行各版本

三个版本都只使用 Python 标准库，不需要 Triton、PyTorch、CUDA Toolkit 或 NVIDIA GPU。从仓库根目录可分别运行：

```bash
conda run -n main python mini-triton/mini-triton-v0.2/demo.py
conda run -n main python mini-triton/mini-triton-v0.2/mini-triton-v0.2.2/demo.py
conda run -n main python mini-triton/mini-triton-v0.2/mini-triton-v0.2.3/demo.py
```

不过，v0.2.1 和 v0.2.2 的输出目录相对于当前工作目录。为了让生成文件保存在对应版本内，推荐进入各目录后运行：

```bash
cd mini-triton/mini-triton-v0.2
conda run -n main python demo.py

cd mini-triton-v0.2.2
conda run -n main python demo.py

cd ../mini-triton-v0.2.3
conda run -n main python demo.py
```

v0.2.1 和 v0.2.2 会打印 Tensor IR 与占位 PTX，并更新各自的：

```text
output/add_kernel.tir
output/add_kernel.ptx
```

v0.2.3 只打印结果，不创建 output 文件。各版本的详细 API、检查规则和限制请阅读对应 README。

### 快速验证全部版本

从仓库根目录运行下面的命令，可以确保 v0.2.1/v0.2.2 的输出写回各自目录，同时避免同名 Python 模块互相干扰：

```bash
(cd mini-triton/mini-triton-v0.2 && conda run -n main python demo.py)
(cd mini-triton/mini-triton-v0.2/mini-triton-v0.2.2 && conda run -n main python demo.py)
(cd mini-triton/mini-triton-v0.2/mini-triton-v0.2.3 && conda run -n main python demo.py)
```

预期三个命令都能打印 `func @add_kernel` 及 `arange`、`tensor_load`、`tensor_add`、`tensor_store`。这只能证明 AST→IR→占位 backend 流程可运行，不代表 GPU kernel 已经生成或执行。

## v0.2.1 当前限制

- 仅支持第一个 Python 函数；找不到函数时没有明确错误信息。
- 所有函数参数都被视为无元素类型的 pointer，尚不支持标量参数。
- 仅支持简单赋值、表达式语句、`arange`、`load`、`store` 和 tensor 加法。
- `arange()` 参数必须是 AST 整数字面量，不能使用变量或表达式。
- `load()` 固定返回 FP32 tensor。
- `tensor_add` 直接继承左操作数类型，没有 shape/dtype 校验。
- 没有 program id、grid、mask、broadcast、reduce、多维 tensor 或控制流。
- `Frontend` 实例重复编译时不会清空 ops、args、env 和 SSA id。
- Backend 只是 opcode dump，尚未生成真实 PTX。
- 没有优化 pass、runtime、数值正确性测试和性能 benchmark。

## v0.2 系列共同限制

- 只覆盖固定 tensor block 的 load/add/store，没有 program id、grid、N 或 mask。
- PointerType 没有元素 dtype，load 固定产生 FP32 tensor。
- 没有定义 tensor block 到 CUDA thread/warp 的 layout。
- 类型与 shape 检查仍不完整；v0.2.3 甚至移除了 v0.2.2 的 load 检查。
- 没有广播、归约、多维 layout、控制流或完整诊断系统。
- 所有 backend 都只输出注释，不能生成或运行真实 GPU kernel。
- 没有 runtime、GPU correctness test 或性能 benchmark。

## 下一步建议

1. 增加 `program_id(axis)`、`BLOCK` 和 `offsets < N` mask。
2. 为参数增加 `ptr<dtype>` 和标量类型，并执行 shape/dtype 校验。
3. 明确 tensor block 到 CUDA threads/warps 的布局规则。
4. 实现真实 PTX kernel ABI、寄存器分配和 pointer 地址计算。
5. Lower `tensor_load`、`tensor_add`、`tensor_store` 为可执行 PTX。
6. 加入 CUDA Driver runtime，运行并验证任意长度的向量加法。
7. 补充 frontend 单元测试、非法 DSL 测试和编译快照测试。
