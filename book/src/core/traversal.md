# 邻域遍历迭代器

> 设计文档：[traversal.pdf](../../../docs/traversal.pdf) | 源码：[traversal.rs](../../../src/traversal.rs)

## 简介

提供 eager/lazy 双策略的邻域遍历迭代器，覆盖顶点环、邻接顶点、面半边与边界环遍历，并支持 k-ring 邻域查询与无向边迭代。

## 关键 API

| 名称 | 功能 |
|------|------|
| `VertexRing` | 顶点环迭代器 |
| `VertexAdjacentVerts` | 邻接顶点迭代器 |
| `FaceHalfEdges` | 面半边迭代器 |
| `BoundaryLoop` | 边界环迭代器 |
| `boundary_loops` | 收集所有边界环 |
| `vertex_k_ring` | k-ring 邻域查询 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/traversal.pdf)。
