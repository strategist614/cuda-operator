# Parallel Scan

本目录计划实现并行前缀和（scan）。目标是从单 block inclusive/exclusive scan 开始，再扩展到多 block：各 block 局部扫描、扫描 block sums、把偏移加回输出。

实现时重点关注 shared memory bank conflict、非 2 的幂长度、最后一个 block 的边界以及整数溢出。可使用 CPU 串行前缀和作为 reference。

当前目录尚无源码，是待实现的练习占位。
