# 链式查询 DSL

> 设计文档：[query.pdf](../../../docs/query.pdf) | 源码：[query.rs](../../../src/query.rs)

## 简介

基于 Builder 模式的延迟链式查询 DSL，支持 `v.halfedge_to(w).cw_rotated().dst_vert().run(&mesh)` 风格的拓扑导航，覆盖 `VertexId`/`HalfEdgeId`/`FaceId` 三类句柄。

## 关键 API

| 名称 | 功能 |
|------|------|
| `MeshQuery` | 链式查询 trait |
| `halfedge_to` | 查询两顶点间的半边 |
| `cw_rotated` | 顺时针旋转半边 |
| `dst_vert` | 目标顶点 |
| `run` | 执行查询 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/query.pdf)。
