# GPU 缓冲导出

> 设计文档：[export.pdf](../../../docs/export.pdf) | 源码：[export.rs](../../../src/export.rs)

## 简介

将半边网格导出为 wgpu 兼容的顶点/索引缓冲，便于直接接入实时渲染管线。

## 关键 API

| 名称 | 功能 |
|------|------|
| `mesh_to_vertex_index_buffers` | 导出顶点/索引缓冲 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/export.pdf)。
