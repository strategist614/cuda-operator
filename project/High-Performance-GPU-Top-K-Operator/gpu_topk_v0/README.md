# GPU Top-K V0

这是 High-Performance GPU Top-K Operator 项目的第一版 baseline。

V0 的目标不是性能，而是建立后续所有优化都能复用的：

```text
CPU reference
+
CUDA correctness
+
stable benchmark
+
shape interface
+
profiling entry
```

---

## 1. 问题定义

输入：

```text
input[B, N]
```

输出：

```text
top_values[B, K]
top_indices[B, K]
```

按 value 从大到小排序。

相同 value 时：

```text
smaller original index first
```

所以结果是 deterministic 的。

---

## 2. V0 CUDA 算法

V0 故意采用非常 naive 的实现：

```text
one block / row

block 内只有 thread 0 真正工作

for rank in [0, K):
    扫描整行 N 个元素
    找当前最大且没有被选择过的元素
```

复杂度：

```text
O(B * K * N)
```

而且：

```text
127/128 threads 基本闲置
```

这正是后续 V1 要解决的问题。

---

## 3. 为什么 baseline 要故意简单

如果一开始就写复杂 warp-select，你后面很难回答：

```text
优化到底快在哪里？
```

V0 让 profiler 能清楚看到：

```text
极低的 warp utilization
大量 serial work
GPU parallelism 没被使用
```

V1 可以直接对照：

```text
V0:
one active thread / row

↓

V1:
all threads scan the row
+
thread-local register Top-K
+
block merge
```

---

## 4. CPU Reference

CPU 使用：

```cpp
std::partial_sort
```

生成完整 correctness reference。

比较：

```text
value
+
original index
```

不只是比较 Top-K value。

---

## 5. 编译

默认目标是用户当前的：

```text
RTX 2080 / SM75
```

编译：

```bash
rm -rf build
mkdir build
cd build

cmake ..
cmake --build . -j

cd ..
```

---

## 6. 第一条测试

建议先不要直接跑特别大的 K，因为 V0 是 O(K*N)。

```bash
./build/topk_bench \
  128 65536 16
```

参数：

```text
B = 128
N = 65536
K = 16
```

预期输出：

```text
GPU: ...
Top-K shape: B=128 N=65536 K=16

Correctness
Value + index match = PASS

Performance
CPU reference latency = ...
GPU naive latency = ...
```

---

## 7. Benchmark 方法

默认：

```text
5 groups
×
20 launches

取 group average 的中位数
```

可以修改：

```bash
./build/topk_bench \
  128 65536 16 \
  --warmup 5 \
  --repeat 20 \
  --groups 5
```

---

## 8. V0 的 normalized bandwidth

程序输出：

```text
Normalized input bandwidth
```

注意这不是严格 DRAM throughput。

V0 会为了找多个 rank 多次重新扫描输入，因此真实 memory traffic 比：

```text
B*N*sizeof(float)
```

更大。

这个指标只是：

```text
固定输入规模 / latency
```

方便 V0/V1/V2 做统一归一化比较。

真正 DRAM throughput 后面看 NCU。

---

## 9. Shape Sweep

```bash
python3 scripts/sweep.py
```

默认测试：

```text
B=32  N=1024   K=1
B=32  N=1024   K=8
B=32  N=4096   K=8
B=64  N=16384  K=16
B=128 N=65536  K=16
B=128 N=65536  K=32
```

V0 对大 K 会很慢，这是预期行为。

---

## 10. Nsight Compute

```bash
NCU=/usr/local/cuda-13.2/bin/ncu \
./scripts/profile_naive.sh
```

V0 重点不是看 Tensor Core，而是看：

```text
Warp Execution Efficiency
Active Threads / Warp

SM Throughput
Memory Throughput

Eligible Warps
Active Warps

Warp Stall Reasons
```

你应该会看到：

```text
blockDim = 128
但只有 threadIdx.x == 0 工作
```

所以这是一个非常差的 GPU mapping。

---

## 11. V0 最核心的瓶颈

代码：

```cpp
if (threadIdx.x != 0)
    return;
```

意味着：

```text
128-thread block

↓

只有一个 thread 做：
N × K selection
```

GPU 的并行能力几乎完全浪费。

其次每找一个 Top-K rank：

```text
重新扫描 N
```

因此：

```text
K ↑
latency 大致快速上升
```

后面 V1 会把：

```text
serial row scan
```

改成：

```text
parallel strided row scan
```

---

## 12. V1 预定优化

V1：

```text
one block / row

256 threads
    ↓

thread 0:
input[0], input[256], ...

thread 1:
input[1], input[257], ...

...

每个 thread:
register local candidates

    ↓

block candidate merge

    ↓

final Top-K
```

也就是第一次真正进入：

```text
Thread-local Register Top-K
```

这是下一个版本的核心。

---

## 13. 项目路线

```text
V0
Naive GPU baseline
+ CPU reference
+ benchmark

↓

V1
Thread-local register Top-K
+ block merge

↓

V2
Warp shuffle Top-K / WarpSelect

↓

V3
Register candidate queue
+ optimized warp merge

↓

V4
Hierarchical Top-K

↓

V5
Multiple algorithm families
+ runtime dispatcher

↓

V6
Radix / bucket selection

↓

V7
PyTorch extension
+ torch.topk benchmark
```
