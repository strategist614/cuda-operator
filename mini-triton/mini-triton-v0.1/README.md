# Mini Triton v0.1

这是一个用于理解 Triton 编译流程的最小教学实现。它接收一小部分 Python kernel 语法，通过 Python AST 构建自定义 SSA IR，再把 IR lowering 为可由 CUDA Driver API 加载的 PTX。

v0.1 不是 Triton 的兼容实现，也不依赖 Triton 编译器。它只实现了运行向量加法示例所需的最小语法和固定 ABI，重点是展示一条完整而容易阅读的编译链路：

```text
Python kernel source
        ↓ ast.parse
Python AST
        ↓ Frontend
Mini Triton SSA IR
        ↓ PTXEmitter
PTX
        ↓ CUDA Driver API
GPU kernel
```

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `mini_triton.py` | IR 数据结构、AST frontend、IR builder、PTX emitter 和公开编译接口 |
| `test_compiler.py` | 编译向量加 kernel，并打印生成的 IR 与 PTX；不启动 GPU kernel |
| `demo.py` | 完整端到端示例：编译、加载 PTX、分配显存、启动 kernel、拷回并验证结果 |

`__pycache__/` 是 Python 自动生成的字节码缓存，不属于编译器源码。

## Kernel 写法

两个示例编译下面的向量加 kernel：

```python
def add_kernel(X, Y, Z, N):
    pid = program_id(0)
    offs = pid * BLOCK + arange(BLOCK)
    mask = offs < N

    x = load(X, offs, mask)
    y = load(Y, offs, mask)
    z = x + y

    store(Z, offs, z, mask)
```

这里的接口模仿 Triton，但 v0.1 会直接映射到一维 CUDA 执行模型：

| Mini Triton 表达式 | v0.1 lowering |
| --- | --- |
| `program_id(0)` | `%ctaid.x`，相当于 `blockIdx.x` |
| `arange(BLOCK)` | `%tid.x`，相当于 `threadIdx.x` |
| `pid * BLOCK + arange(BLOCK)` | 当前 CUDA 线程处理的全局元素下标 |
| `offs < N` | PTX predicate，用于保护尾部元素 |
| `load(...)` | 带 predicate 的 `ld.global.f32` |
| `store(...)` | 带 predicate 的 `st.global.f32` |

与真正 Triton 中一个 program 操作一整个 tensor block 不同，这个教学版本把 `arange(BLOCK)` 简化成单个 CUDA 线程的 `threadIdx.x`。因此运行时必须使用与编译期 `BLOCK` 相同的 CUDA block size。

## 编译器结构

### 1. Mini Triton IR

`Value`、`Argument`、`Op` 和 `FunctionIR` 描述一个简单的 SSA IR。`IRBuilder.emit()` 为有结果的 operation 自动分配 `%0`、`%1` 等 SSA value。

上面的向量加 kernel 在 `BLOCK=256` 时会生成：

```text
func @add_kernel(X: ptr<f32>, Y: ptr<f32>, Z: ptr<f32>, N: i32) {
  %0:i32 = program_id 0
  %1:i32 = const 256
  %2:i32 = mul %0, %1
  %3:i32 = lane_id
  %4:i32 = add %2, %3
  %5:pred = cmp_lt %4, N
  %6:f32 = load X, %4, %5
  %7:f32 = load Y, %4, %5
  %8:f32 = add %6, %7
  store Z, %4, %8, %5
}
```

### 2. Python AST frontend

`Frontend.compile()` 使用 `ast.parse()` 读取源码，要求源码中恰好包含一个函数。frontend 维护变量到 IR value 的 symbol table，并逐条编译 assignment 或函数调用表达式。

v0.1 支持的语法如下：

- 简单变量赋值，例如 `x = load(...)`。
- 整数常量和编译期常量 `BLOCK`。
- `+`、`*` 和 `<`。
- `program_id(0)` 与 `arange(BLOCK)`。
- `load(base, offset, mask)` 和 `store(base, offset, value, mask)`。

不在上述范围内的语句或表达式会抛出 `SyntaxError`。引用未定义变量会抛出 `NameError`。

### 3. PTX emitter

`PTXEmitter` 为 IR value 分配 32-bit integer、64-bit address、FP32 和 predicate 寄存器，并逐条生成 PTX。默认目标是：

```text
.version 7.0
.target sm_70
.address_size 64
```

masked load 会先把目标浮点寄存器置零，再通过 predicate 执行 `ld.global.f32`；masked store 则通过同一个 predicate 保护 `st.global.f32`。FP32 元素地址按 `base + offset × 4` 计算。

### 4. 公开接口

调用 `compile_kernel()` 可同时得到 IR 对象和 PTX 字符串：

```python
from mini_triton import compile_kernel

ir, ptx = compile_kernel(SOURCE, block=256)
print(ir)
print(ptx)
```

## 环境

当前 `main` Conda 环境中与本示例直接相关的版本为：

| 组件 | 版本 |
| --- | --- |
| Python | 3.10.20 |
| NumPy | 2.2.6 |
| cuda-python | 13.3.1 |
| cuda-bindings | 13.3.1 |

运行 `test_compiler.py` 只需要 Python。运行 `demo.py` 还需要 NumPy、`cuda.bindings`、NVIDIA GPU 和兼容驱动。

## 运行

进入目录并激活环境：

```bash
conda activate main
cd mini-triton/v0.1
```

只查看生成的 IR 和 PTX：

```bash
python test_compiler.py
```

运行端到端 GPU 示例：

```bash
python demo.py
```

`demo.py` 使用 `BLOCK=256`、`N=1000`，所以会启动 `ceil(1000 / 256) = 4` 个 block，每个 block 256 个线程。最后一个 block 中超出 N 的线程通过 mask 跳过 global-memory 访问。

端到端流程包括：

1. 调用 `compile_kernel()` 生成 IR 和 PTX。
2. 使用 NumPy 创建 FP32 输入和 reference。
3. 初始化 CUDA Driver API 并取得 primary context。
4. 使用 `cuModuleLoadData()` 加载 PTX，取得 `add_kernel`。
5. 分配 device memory 并执行 H2D 拷贝。
6. 使用 `cuLaunchKernel()` 启动 kernel。
7. 执行 D2H 拷贝，并用 `np.allclose()` 验证 `Z = X + Y`。
8. 在 `finally` 中释放显存、module 和 primary context。

成功运行时，输出结尾应包含：

```text
correct: True
```

## 当前限制

- Kernel ABI 固定为 `X/Y/Z: ptr<f32>` 和 `N: i32`，不能从函数签名自动推导。
- 只支持一个一维 kernel、一个元素对应一个 CUDA 线程。
- `load`、`store` 和浮点运算只支持 FP32。
- 只支持 `+`、`*`、`<`，不支持减法、除法、循环、分支或多维索引。
- `arange()` 必须使用编译期的 `BLOCK`，`program_id()` 只支持 axis 0。
- 寄存器声明使用固定上限，没有 liveness、复用或寄存器压力优化。
- 没有常量折叠、公共子表达式消除、向量化、自动调优等优化 pass。
- PTX target 固定为 `sm_70`，没有根据本机 GPU 自动选择架构。
- 当前测试脚本打印编译产物，但没有单元测试框架、负例测试或性能 benchmark。

这个版本的价值在于把 frontend、IR、code generation 和 runtime launch 串成一条最小可运行链路。后续版本可以在此基础上逐步增加类型系统、多维 program、更多算子、优化 pass 和更接近 Triton 的 block-level 语义。
