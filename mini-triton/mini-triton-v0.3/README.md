# Mini Triton v0.3 Series

Mini Triton v0.3 系列探索如何把 v0.2 的逻辑 Tensor IR 逐步 lowering 到 GPU 线程、地址和寄存器层。系列从 BlockedLayout 开始，依次实验 thread mapping、address generation、register scalarization，最后在 `mini-triton-final` 中把这些阶段放进一个 compiler pipeline。

本系列仍是教学型编译器原型。所有 backend 都只生成 IR 注释或 PTX 风格草图，没有形成可由 CUDA Driver API 执行的完整 PTX module。

## 版本导航

| 版本 | 目录 | 核心实验 | Backend 状态 |
| --- | --- | --- | --- |
| v0.3 | 当前目录 | LayoutTensorType、BlockedLayout、LayoutPass、ThreadMappingPass | 打印 layout-lowered IR 注释 |
| v0.3.1 | [`mini-triton-v0.3.1/`](mini-triton-v0.3.1/) | ThreadMapping、LaunchConfig、thread_id/lane_id | 打印 launch 与 Thread IR 注释 |
| v0.3.2 | [`mini-triton-v0.3.2/`](mini-triton-v0.3.2/) | element→thread、AddressLoweringPass、地址生成模拟 | 追加三行 PTX 风格 load 草图 |
| v0.3.3 | [`mini-triton-v0.3.3/`](mini-triton-v0.3.3/) | Register IR、scalarization、虚拟寄存器 | 追加 load/add/store 寄存器草图 |
| Final / v0.3.8 | [`mini-triton-final/`](mini-triton-final/) | 集成 layout/thread/address/register stages | 硬编码 PTX 模拟，仍不可执行 |

建议按表中顺序阅读，并结合各子目录 README 查看详细源码分析和限制。

## 系列演进

```text
v0.3
Tensor IR + BlockedLayout
    ↓
v0.3.1
ThreadMapping + LaunchConfig + thread/lane markers
    ↓
v0.3.2
Thread → element/address lowering experiment
    ↓
v0.3.3
Tensor scalarization + virtual-register experiment
    ↓
Final / v0.3.8
LayoutPass → ThreadPass → AddressPass → RegisterAllocator
```

这些版本不是严格的增量继承：

- v0.3.1 没有继续使用 v0.3 的结构化 LayoutTensorType。
- v0.3.2 加入 address lowering，但移除了 v0.3.1 的 lane_id。
- v0.3.3 只运行 RegisterLoweringPass，没有串联 v0.3.2 的 thread/address passes。
- Final 才把 layout、thread、address、register 阶段放入同一个入口，但 pass 仍主要追加标记，没有真正重写高层数据流。

## 能力矩阵

| 能力 | v0.3 | v0.3.1 | v0.3.2 | v0.3.3 | Final |
| --- | --- | --- | --- | --- | --- |
| BlockedLayout | 类型中使用 | 文件中定义但未接入 | 无 | 无 | 使用 |
| LaunchConfig | 无 | 128 threads / 4 warps | 128 threads / 4 warps | 无 | layout 隐含 4 warps / 128 threads |
| Thread mapping | 显式 mapping op | FunctionIR metadata + op | FunctionIR metadata | RegisterMapping 未接入 | layout + thread/lane ops |
| Address lowering | 无 | 无 | 有描述与 PTX 草图 | 无 | 有抽象 address op |
| Register lowering | 无 | 无 | 无 | RegisterLoweringPass | 简单 RegisterAllocator |
| 真实 PTX module | 否 | 否 | 否 | 否 | 否 |
| CUDA runtime | 否 | 否 | 否 | 否 | 否 |

## 共同示例

各版本都围绕固定 256 元素的 tensor add：

```python
def add_kernel(X, Y, Z):
    offsets = arange(256)
    x = load(X, offsets)
    y = load(Y, offsets)
    z = x + y
    store(Z, offsets, z)
```

目标语义为：

```text
Z[0:256] = X[0:256] + Y[0:256]
```

所有 v0.3 示例都缺少 program id、N 和 mask，因此只能描述一个固定 tensor block，不能覆盖任意长度输入或安全处理尾部。

## v0.3 基础版本

当前目录中的根版本使用 `LayoutTensorType` 表示带 layout 的 tensor：

```python
@dataclass
class LayoutTensorType:
    dtype: str
    shape: tuple
    layout: object
```

Frontend 为 `arange(256)` 创建 layout 为空的 tensor。LayoutPass 随后设置：

```text
BlockedLayout(
  threads=128,
  elements_per_thread=2
)
```

128 个线程 × 每线程 2 个元素 = 256 个元素。

ThreadMappingPass 在 `make_tensor` 后插入：

```text
thread_mapping threads=128 elements_per_thread=2
```

但它没有把 tensor load/add/store 重写成逐线程 operation，也没有生成 address/register IR。

## v0.3 基础 pipeline

```text
Python DSL
    ↓ Frontend
LayoutTensorType（layout=None）
    ↓ LayoutPass
BlockedLayout(128 threads, 2 elements/thread)
    ↓ ThreadMappingPass
thread_mapping marker
    ↓ PTXBackend
layout-lowered IR comments
```

`compiler.py` 实际执行：

```python
ir = Frontend().compile(src)
LayoutPass().run(ir)
ThreadMappingPass().run(ir)
ptx = PTXBackend().emit(ir)
```

## v0.3 基础文件结构

| 文件 | 作用 |
| --- | --- |
| `layout.py` | 定义 BlockedLayout 与 LayoutTensorType |
| `ir.py` | 定义 Value、Op、FunctionIR |
| `frontend.py` | Python AST → Layout-aware Tensor IR |
| `passes.py` | 分配 layout 并插入 thread_mapping |
| `backend.py` | 把 lowering 后的 operation 输出为注释 |
| `compiler.py` | 运行 frontend、layout/thread passes 和 backend |
| `demo.py` | 打印 Layout IR 与 backend 输出 |

`__pycache__/` 是 Python 自动生成的字节码缓存，不属于编译器逻辑。

## v0.3 基础实现限制

- 函数参数只有字符串类型 `ptr`。
- `arange` size 虽从 AST 读取，但 load 会直接继承 offsets 的 i32 类型，数据 tensor 因而也显示为 i32。
- 任意二元表达式都会被当成 tensor_add。
- LayoutPass 直接修改共享 type object，没有类型不可变性或验证。
- ThreadMappingPass 只插入描述，不建立 SSA thread/element 映射。
- 没有 address、register 或真实 PTX lowering。

## 后续版本的关键观察

### v0.3.1：Thread IR

固定 launch 为 128 threads / 4 warps，并增加 thread_id、lane_id 标记。但标记追加在原 store 后，没有参与数据流。

### v0.3.2：Address IR

加入 `element_owner = element // 2` 和 `offset = thread_id * 2`。Backend 开始输出 `%tid.x` 和 load 草图，但把元素索引直接当成字节偏移，且只处理一次 load。

### v0.3.3：Register IR

加入 tensor scalarization 标记、Register/Mapping 与虚拟寄存器草图。但 compiler 只运行 register pass；两次 load 来自同一抽象地址的相邻元素，不是 X/Y 同 offset 加法。

### Final：阶段集成

Final 创建 4 warp × 32 thread × 2 elements/thread 的 layout，追加 thread/lane/address IR，并建立 SSA 名称到 `%rN` 的编号表。

不过 RegisterAllocator 不区分类型、不重写 operands，backend 也不读取分配表。硬编码 store 写向固定 `[addr]`，多线程会产生写冲突。

## 为什么所有 backend 仍不可执行

各版本存在的具体问题不同，但共同缺少：

- 完整 PTX header 和 `.entry` kernel。
- X/Y/Z 参数 ABI 与 base pointer 加载。
- 类型正确的寄存器声明与分配。
- program/thread 到元素的完整 SSA lowering。
- FP32 元素索引到 byte address 的正确换算。
- N、mask 和 predicated tail handling。
- 从 X/Y 加载同一 offset、相加并写入 Z 的完整数据流。

因此 README 或源码中的“PTX”应理解为 codegen simulation，而不是可交给 CUDA Driver JIT 的 module。

## 运行全部版本

这些版本只依赖 Python 标准库，不需要 GPU。建议从仓库根目录使用子 shell：

```bash
(cd mini-triton/mini-triton-v0.3 && conda run -n main python demo.py)
(cd mini-triton/mini-triton-v0.3/mini-triton-v0.3.1 && conda run -n main python demo.py)
(cd mini-triton/mini-triton-v0.3/mini-triton-v0.3.2 && conda run -n main python demo.py)
(cd mini-triton/mini-triton-v0.3/mini-triton-v0.3.3 && conda run -n main python demo.py)
(cd mini-triton/mini-triton-v0.3/mini-triton-final && conda run -n main python demo.py)
```

每个 demo 只打印内存中的 IR 和 backend 字符串，不创建 output 文件，不初始化 CUDA，也不执行数值验证。

## 推荐阅读顺序

在每个版本中按以下调用链阅读：

```text
demo.py
  → compiler.py
    → frontend.py
      → ir.py / layout.py / thread_ir.py / register_ir.py
    → passes.py
    → backend.py
```

跨版本建议重点对比：

1. Tensor type 如何从无 layout 变成 BlockedLayout。
2. ThreadMapping 如何从描述 operation 变成 launch metadata。
3. Address IR 是否真正连接 thread id 与 pointer。
4. Register allocation 是否按类型分配并重写 SSA operands。
5. Backend 是由 IR 驱动还是硬编码输出。

## 系列共同限制

- DSL 只支持极少 Python AST 节点。
- Tensor shape、dtype、layout 多为硬编码。
- 多数 pass 只在 operation 尾部追加标记，没有真正 lowering/rewriting。
- 子版本之间并非严格累积，部分能力会消失或退化。
- 没有系统化 type/layout verification、pass manager 或诊断。
- 没有合法 PTX、CUDA runtime、单元测试、GPU correctness test 或 benchmark。

## 下一步建议

1. 建立统一且类型化的 Tensor/Layout/Thread/Address/Register IR。
2. 让每个 pass 消费并替换上一层 operation，而不是追加注释标记。
3. 验证 `shape == warps × threads_per_warp × elements_per_thread`。
4. 增加 program id、N、mask 和多 block 语义。
5. 正确计算每线程每元素的 byte address。
6. 实现类型感知的 register allocation 和 SSA operand rewrite。
7. 让 backend 完全由 lowered IR 驱动，移除硬编码。
8. 生成完整 PTX module，并接入 CUDA Driver correctness tests。
