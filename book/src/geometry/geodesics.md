# 测地线距离

> 设计文档：[geodesics.pdf](../../../docs/geodesics.pdf) | 源码：[geodesics.rs](../../../src/geodesics.rs)

## 简介

计算网格上的测地线距离。实现 Heat Method（Crane 2013）基于热扩散的快速近似，以及 Dijkstra 单源/多源最短路径与最短路径回溯。

## 关键 API

| 名称 | 功能 |
|------|------|
| `geodesic_distance_from_vertex` | Heat Method 单源测地距离 |
| `dijkstra_geodesic` | Dijkstra 单源测地距离 |
| `dijkstra_multi_source_geodesic` | Dijkstra 多源测地距离 |
| `shortest_path` | 最短路径回溯 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/geodesics.pdf)。
