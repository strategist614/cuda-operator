# High-Performance GPU Top-K V3

V3 开始真正优化 **selection 本体**。

此前：

```text
V0:
serial repeated selection
525.86 ms

V1:
parallel scan
+ per-thread register Top-16
+ block merge
4.11993 ms

V2:
same per-thread Top-16
+ warp-shuffle merge
4.08793 ms
```

V1 → V2 只有约：

```text
0.8%
```

说明 merge 已经不是主要瓶颈。

真正的大头是：

```text
每个 thread 都维护 float[16] + int[16]
+
每个 candidate 可能触发长串 insertion dependency
```

V3 因此把算法改成：

```text
Warp-Cooperative Batch Selection
```

---

## 1. V3 核心结构

```text
Input Row
   ↓
8 warps / block
   ↓
每个 warp 每轮读取 32 elements
   ↓
cheap warp threshold test
   ↓
不可能进入 Top-16？
   ├── yes → 整批 reject
   └── no
         ↓
   32-lane bitonic sort
         ↓
   batch Top-16
         ↓
   merge with current warp Top-16
         ↓
   exact warp Top-16

8 warp Top-16
   ↓
warp0 bitonic final merge
   ↓
Final Top-K
```

---

# 2. 最大变化：不再每 thread 保存 Top-16

V1/V2：

```cpp
float local_values[16];
int local_indices[16];
```

每个 thread 都有一份。

256 threads：

```text
256 × 16 values
+
256 × 16 indices
```

而 V3：

```text
一个 warp 只维护一份 Top-16
```

并分布在：

```text
lane 0 .. lane 15
```

所以逻辑上：

```text
32 threads
共同维护
16 candidates
```

不是：

```text
32 threads
各自维护
16 candidates
```

这会显著减少：

```text
register state
serial insertion
dependency chain
```

---

# 3. Warp 如何扫描输入

一个 block：

```text
256 threads
=
8 warps
```

warp 0：

```text
input[0..31]
input[256..287]
input[512..543]
...
```

warp 1：

```text
input[32..63]
input[288..319]
...
```

所以所有 warp 共同覆盖整行，而且：

```text
每个 input element
只读取一次
```

每轮 32 lanes 读取连续数据：

```text
lane0  → x[base+0]
lane1  → x[base+1]
...
lane31 → x[base+31]
```

保持 coalesced global load。

---

# 4. Threshold Reject

这是 V3 很关键的一层。

warp 当前已经有 Top-16。

其中：

```text
lane15
```

保存当前：

```text
16th best
=
threshold
```

每次加载新的 32 个值后，先不马上排序。

先通过：

```cpp
__shfl_down_sync
```

做一个 32-lane maximum reduction。

只需要：

```text
offset:
16
8
4
2
1
```

共 5 个 shuffle stages。

得到：

```text
batch maximum
```

如果：

```text
batch_max <= current_threshold
```

那么这一批 32 个元素：

```text
没有任何一个
能进入当前 Top-16
```

整个 batch 直接跳过。

因此 steady state 下：

```text
很多普通 batch
只付出：
load + 5-stage warp max
```

不再每个元素都做 Top-16 insertion。

---

# 5. 为什么 threshold reject 是 exact 的

当前 warp Top-16 threshold：

```text
T
```

如果某个 batch：

```text
max(batch) <= T
```

那么 batch 中所有：

```text
x <= T
```

当前已经至少有 16 个元素：

```text
>= T
```

所以这一批不可能改变 warp Top-16。

相同 value 时继续使用：

```text
smaller original index first
```

因此判断实际上基于：

```text
(value, index)
```

pair comparator，而不是只比较 float。

---

# 6. Batch Sort

如果这一批确实有竞争力：

```text
batch_max > threshold
```

才启动：

```text
32-lane bitonic sort
```

所有数据都留在：

```text
warp registers
```

通过：

```cpp
__shfl_xor_sync
```

做 compare-exchange network。

最终：

```text
lane0
lane1
...
lane15
```

得到这一批：

```text
Batch Top-16
```

---

# 7. 为什么只保留 Batch Top-16

一个 batch 有 32 个元素。

如果某元素连：

```text
batch Top-16
```

都进不了，

说明这个 batch 内已经有至少 16 个元素比它更好。

那么在整个 warp 的所有输入中，它也不可能成为：

```text
warp Top-16
```

所以：

```text
batch 32
↓
只留 batch Top-16
```

是严格 exact 的。

---

# 8. Warp Top-16 merge

当前：

```text
Current Warp Top-16
```

已经排序。

新的：

```text
Batch Top-16
```

也已经排序。

不需要重新 full sort 32 个任意元素。

V3 构造：

```text
Current Top-16 descending
+
Batch Top-16 reversed
```

形成 bitonic sequence。

然后只需要：

```text
stride:
16
8
4
2
1
```

5 个 compare-exchange stages。

得到：

```text
Top-16 of union
```

所以昂贵路径是：

```text
batch full sort
+
5-stage merge
```

而不是每个输入元素都做长度 16 insertion。

---

# 9. Block Final Merge

每个 warp 最后得到：

```text
exact Warp Top-16
```

一共有：

```text
8 lists
```

写入小 shared memory：

```text
8 × 16 values
8 × 16 indices
```

只有：

```text
1 次 __syncthreads()
```

然后 warp0 连续做 8 次 bitonic Top-16 merge。

最终 lane：

```text
0..15
```

得到 block / row Top-16。

---

# 10. Exactness

V3 不是 approximate Top-K。

准确性链：

```text
每个 32-element batch
保留 exact batch Top-16

↓

warp 合并所有 batch Top-16
得到 exact warp Top-16

↓

8 个 warp Top-16 合并
得到 exact row Top-16

↓

输出 first K
```

支持：

```text
1 <= K <= 16
```

内部始终维护 Top-16。

对于小 K 会多做一些工作，但保证一个统一 exact kernel。

---

# 11. 编译

```bash
rm -rf build
mkdir build
cd build

cmake ..
cmake --build . -j

cd ..
```

默认：

```text
SM75 / RTX 2080
```

---

# 12. 第一条命令

```bash
./build/topk_bench \
  128 65536 16
```

默认就是：

```text
--kernel batch
```

应该输出：

```text
Implementation:
warp_batch_threshold_bitonic_v3

Correctness:
PASS
```

---

# 13. V2/V3 A-B

最重要：

```bash
./build/topk_bench \
  128 65536 16 \
  --compare-v2
```

输出：

```text
GPU selected latency = ...

V2 comparison
V2 warp-merge latency = ...

V3 speedup over V2 = ...x
```

你的 V2 baseline：

```text
4.08793 ms
```

所以 V3 就以这个作为目标。

---

# 14. 所有版本仍保留

```bash
# V0
./build/topk_bench 128 65536 16 \
  --kernel naive
```

```bash
# V1
./build/topk_bench 128 65536 16 \
  --kernel register
```

```bash
# V2
./build/topk_bench 128 65536 16 \
  --kernel warp
```

```bash
# V3
./build/topk_bench 128 65536 16 \
  --kernel batch
```

---

# 15. NCU

```bash
NCU=/usr/local/cuda-13.2/bin/ncu \
./scripts/profile_v2_vs_v3.sh
```

重点比较：

```text
Registers / Thread

Achieved Occupancy

Active Warps

Eligible Warps

Long Scoreboard

Wait

Instruction count

Shared memory

Barrier stalls
```

V3 最希望看到：

```text
Registers / Thread ↓
```

以及：

```text
local insertion dependency 消失
```

代价是：

```text
shuffle instruction count ↑
```

最终是不是更快必须由 benchmark 决定。

---

# 16. 一个重要实验预期

V3 不保证所有 shape 都比 V2 快。

对于：

```text
N 很小
```

bitonic network 的固定成本可能太高。

对于：

```text
N 很大
K=16
```

threshold reject 才更容易发挥价值。

这正好为后面的 runtime dispatcher 铺路：

```text
small N
→ V2-like path

large N / small K
→ V3 batch-select path
```

---

# 17. 下一步

如果 V3 有明显提升，下一步可以继续研究：

```text
multi-items/lane
+
load ILP
+
threshold update frequency
```

如果 V3 反而变慢，则说明：

```text
full 32-lane sorting network cost
>
saved insertion cost
```

那 V4 应该转向：

```text
radix / threshold selection
```

而不是继续堆 bitonic。
