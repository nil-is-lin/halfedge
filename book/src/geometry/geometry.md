# 几何工具

> 设计文档：[geometry.pdf](../../../docs/geometry.pdf) | 源码：[geometry.rs](../../../src/geometry.rs)

## 简介

提供网格几何度量与查询工具，包括边长、面积、法向、余切拉普拉斯、曲率（高斯/平均/主）、二面角、点-三角距离、AABB 包围盒与射线求交。

## 关键 API

| 名称 | 功能 |
|------|------|
| `edge_length` | 边长 |
| `face_normal` | 面法向 |
| `cotan_laplacian` | 余切拉普拉斯 |
| `vertex_curvature` | 顶点曲率 |
| `ray_mesh_intersection` | 射线-网格求交 |
| `mesh_aabb` | 网格 AABB 包围盒 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/geometry.pdf)。
