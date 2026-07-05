# 句柄与 ID

> 设计文档：[ids.pdf](../../../docs/ids.pdf) | 源码：[ids.rs](../../../src/ids.rs)

## 简介

强类型句柄模块，定义网格元素的标识符。`VertexId`/`HalfEdgeId`/`FaceId`/`EdgeId` 分别标识顶点、半边、面与无向边，其中 `EdgeId` 是无向边的规范代表。

## 关键 API

| 名称 | 功能 |
|------|------|
| `VertexId` | 顶点句柄 |
| `HalfEdgeId` | 半边句柄 |
| `FaceId` | 面句柄 |
| `EdgeId` | 无向边句柄（规范代表） |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/ids.pdf)。
