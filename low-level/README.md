# CUDA Execution Model

本目录记录 CUDA 底层执行模型与性能概念，供其他 kernel 优化实验交叉参考。

`block` 进入 `SM` 后，会被拆成多个 `warp`
 
一个 `block` 里有 `256` 个 `thread`

`warp` 里的 `32` 个线程执行同一条指令

多个 block 可以驻留在同一个 SM，具体数量受线程数、寄存器、shared memory 和硬件上限共同约束。warp 是调度与执行的基本单位；同一 warp 内发生条件分支时，不同路径通常会被串行执行，称为 warp divergence。

优化时需要同时考虑：全局内存访问是否合并、shared memory 是否发生 bank conflict、同步次数、寄存器压力和 occupancy。occupancy 高不等于一定更快，应结合 Nsight Compute 的吞吐与 stall 指标判断实际瓶颈。
