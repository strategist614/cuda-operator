## transpose

#### CUDA内存
```c++
threads(TILE_DIM, BLOCK_ROWS)
```
这里是 `threads(32, 16)`
表示 `threadIdx.y` 是固定的 然后
`threadIdx.x = 0,1,2..31`，所以`xIndex`连续在变化，`yIndex`保持不变。这个也符合行优先的规则

> 同一个 warp 里面，这些线程在同一条写指令上，要写的地址是不是挨在一起。只要目标地址是连续的，那对于 global memory 来说，他就是最快的

#### tile转置
#### tile内部转置