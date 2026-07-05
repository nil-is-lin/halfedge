# 网格存储 MeshStorage

> 设计文档：[storage.pdf](../../../docs/storage.pdf) | 源码：[storage.rs](../../../src/storage.rs)

## 简介

底层网格存储容器，基于 SlotMap 实现。提供 CRUD、有效性判断、数据迭代器与容量统计，并附带 `max_vertex_valence`/`euler_characteristic`/`genus` 等诊断方法。

## 关键 API

| 名称 | 功能 |
|------|------|
| `MeshStorage` | 网格存储容器 |
| `Vertex` | 顶点数据 |
| `HalfEdge` | 半边数据 |
| `Face` | 面数据 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/storage.pdf)。
