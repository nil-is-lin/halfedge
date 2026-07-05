# 三角剖分

> 设计文档：[triangulation.pdf](../../../docs/triangulation.pdf) | 源码：[triangulation.rs](../../../src/triangulation.rs)

## 简介

提供平面与三维多边形的三角剖分算法，包括耳裁剪（ear clipping）与扇形（fan）三角剖分，适用于将 n-gon 面分解为三角形。

## 关键 API

| 名称 | 功能 |
|------|------|
| `ear_clipping` | 平面耳裁剪三角剖分 |
| `ear_clipping_3d` | 三维耳裁剪三角剖分 |
| `fan_triangulation` | 平面扇形三角剖分 |
| `fan_triangulation_3d` | 三维扇形三角剖分 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/triangulation.pdf)。
