# 图元构建

> 设计文档：[primitives.pdf](../../../docs/primitives.pdf) | 源码：[primitives.rs](../../../src/primitives.rs)

## 简介

提供常见几何图元的网格构建器，生成参数化的封闭或带边界网格，作为算法测试与建模的基础形状。

## 关键 API

| 名称 | 功能 |
|------|------|
| `build_cube` | 立方体 |
| `build_uv_sphere` | UV 球 |
| `build_cylinder` | 圆柱 |
| `build_cone` | 圆锥 |
| `build_grid` | 网格平面 |
| `build_torus` | 圆环 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/primitives.pdf)。
