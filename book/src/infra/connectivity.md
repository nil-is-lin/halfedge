# 连通分量分析

> 设计文档：[connectivity.pdf](../../../docs/connectivity.pdf) | 源码：[connectivity.rs](../../../src/connectivity.rs)

## 简介

提供网格连通分量分析，支持面连通与顶点连通（BFS 遍历）两种方式，以及分量的合并与分割操作。

## 关键 API

| 名称 | 功能 |
|------|------|
| `connected_components` | 面连通分量分析 |
| `component_count` | 连通分量计数 |
| `split_into_components` | 分割为多个网格 |
| `merge_meshes` | 合并多个网格 |
| `vertex_connected_components` | 顶点连通分量分析 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/connectivity.pdf)。
