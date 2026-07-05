# 内置属性

> 设计文档：[builtin_attrs.pdf](../../../docs/builtin_attrs.pdf) | 源码：[builtin_attrs.rs](../../../src/builtin_attrs.rs)

## 简介

内置几何属性的 Newtype 包装，包括顶点法向、UV、颜色与面法向。提供 OBJ 属性感知的读写 IO，自动解析与导出相应属性。

## 关键 API

| 名称 | 功能 |
|------|------|
| `VertexNormal` | 顶点法向 |
| `VertexUv` | 顶点 UV |
| `VertexColor` | 顶点颜色 |
| `FaceNormal` | 面法向 |
| `add_vertex_normals` | 添加顶点法向属性 |
| `parse_obj_with_attrs` | 解析带属性的 OBJ |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/builtin_attrs.pdf)。
