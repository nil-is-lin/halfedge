# 网格变形

> 设计文档：[deformation.pdf](../../../docs/deformation.pdf) | 源码：[deformation.rs](../../../src/deformation.rs)

## 简介

提供网格变形算法。实现 Laplacian Surface Editing（Sorkine 2004）保持局部细节的变形，以及 ARAP（Sorkine & Alexa 2007）尽可能刚性迭代优化变形。

## 关键 API

| 名称 | 功能 |
|------|------|
| `laplacian_deformation` | Laplacian 变形 |
| `arap_deformation` | ARAP 变形 |
| `DeformationConstraint` | 变形约束 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/deformation.pdf)。
