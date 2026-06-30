## 算子优化思路

```
先写对
↓
测基准
↓
找瓶颈
↓
针对瓶颈优化
↓
重新测
```

### memory-bound
主要是处理数据
优化重点是：
```
减少 global memory 访问
让读写连续
vectorized load/store
kernel fusion
```
* 保证 coalesced access 
* 减少 global memory 读取次数
* vectorized load/store
* 减少分支

### reduction-bound
这类计算是 很多个数归约成一个数

优化重点是：
```
warp-level reduction
减少 __syncthreads()
使用 __shfl_down_sync()
一个线程处理多个元素
避免 shared memory bank conflict
```

```
一个线程处理多个元素
先在寄存器里累加
warp 内用 shuffle reduction
block 内再合并 warp 结果
减少 __syncthreads()
mean 和 variance 尽量一次读入完成
```
### compute-bound
一个数据从 `global memory` 读进来之后，要尽量多用几次
优化重点是：
```
shared memory tiling
register tiling
tensor core
double buffering
wmma / mma
数据复用
```

```
shared memory tiling
register tiling
warp tiling
double buffering
tensor core
```
## 步骤
### naive版本
### 建立baseline
### 用 profile 找瓶颈
一般判断方法：
```
DRAM throughput 高，SM utilization 不高
    -> memory-bound

SM utilization 高，计算指令很多
    -> compute-bound

occupancy 很低
    -> 可能 register 太多 / shared memory 太多

warp stall memory 很高
    -> 访存拖慢

warp stall sync 很高
    -> 同步太多
```
## 优化顺序
```
1. 正确性
2. 减少 kernel 数量
3. 减少 global memory 访问
4. 保证 coalesced access
5. 增加数据复用
6. 减少同步
7. 减少分支
8. 控制 register / shared memory 使用
9. vectorized load/store
10. 特殊化尺寸
```

## profiler
### Nsight Systems
整个程序的时间线
```
程序整体时间花在哪里？
CPU 在等 GPU 吗？
cudaMemcpy 花了多久？
kernel 之间有没有空隙？
是不是频繁 launch kernel？
```

```
nsys profile 
```
### Nsight Compute

某一个 CUDA kernel 的细节
```
这个 kernel 内部到底慢在哪里？
global memory 访问效率怎么样？
DRAM throughput 多少？
occupancy 多少？
warp stall 原因是什么？
寄存器用了多少？
shared memory 有没有 bank conflict？
```

```
ncu 
```