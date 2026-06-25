## Reduction

### Sum
`CUDA` 里，一个 `warp` 通常有 32 个线程。

```
一个 warp = 32 个线程
lane id = 0, 1, 2, ..., 31
```

每个线程都有自己的寄存器变量：
```
mySum

lane 0  : a0
lane 1  : a1
lane 2  : a2
...
lane 31 : a31

a0 + a1 + a2 + ... + a31
```

`__shfl_down_sync__`当前线程去拿“自己下面`offset`个`lane`”的`mySum`值

例如
```
__shfl_down_sync(mask, mySum, 16)

lane 0  拿 lane 16 的 mySum
lane 1  拿 lane 17 的 mySum
lane 2  拿 lane 18 的 mySum
...
lane 15 拿 lane 31 的 mySum

当前线程的 mySum = 当前线程的 mySum + 后面 offset 个线程的 mySum
```

第一轮 `offset=16`
```
warpSize = 32
offset = warpSize / 2 = 16
mySum += __shfl_down_sync(mask, mySum, 16);

lane 0  : a0  + a16
lane 1  : a1  + a17
lane 2  : a2  + a18
...
lane 15 : a15 + a31

```
第二轮 `offset=8`
```
lane 0 当前值 = a0 + a16
lane 8 当前值 = a8 + a24

lane 0 = a0 + a16 + a8 + a24
```
第三轮 `offset=4`
```
lane 0 = a0 + a8 + a16 + a24
lane 4 = a4 + a12 + a20 + a28

lane 0 = a0 + a4 + a8 + a12 + a16 + a20 + a24 + a28
```
第四轮 `offset=2`
```
lane2 = a2 + a6 + a10 + a14 + a18 + a22 + a26 + a30

lane0 = lane0 + lane2 = 
a0 + a2 + a4 + a6 + a8 + a10 + a12 + a14
+ a16 + a18 + a20 + a22 + a24 + a26 + a28 + a30
```
第五轮 `offset=1`
```
lane1 = a1 + a3 + a5 + a7 + ... + a31
lane0 = lane0 + lane1 = a0 + a1 + a2 + a3 + ... + a31
```

`__shfl_down_sync`线程之间直接交换寄存器数据

减少了
```
shared memory 读写
block 级同步
shared memory bank conflict
```

注意它只能做一个`warp`内部的数据交换，不能跨 `warp`

`mask`
```
unsigned int mask

__shfl_down_sync(mask, mySum, offset)

mask表示哪些 lane 参与这次shuffle

如果是 0xffffffff
表示 32 个 lane 全部参与

```
根据`blocksize`生成`mask`
```C++
unsigned int maskLength = (blockSize & 31);
maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
const unsigned int mask = (0xffffffff) >> maskLength;
```

### reduce0

低效原因：

* 用了 `%` 取模
* 活跃线程是间隔分布的
  每轮的`warp`利用率很低

`GPU` 上一个 `warp` 的 `32` 个线程通常是一起执行同一条指令
