# 内蕴 Delaunay 三角剖分

> 设计文档：[intrinsic.pdf](../../../docs/intrinsic.pdf) | 源码：[intrinsic.rs](../../../src/intrinsic.rs)

## 简介

提供内蕴 Delaunay 三角剖分。基于内蕴边翻转（Fisher 2007）在不改变顶点位置的前提下优化连接，计算内蕴边长与余切权重。

## 关键 API

| 名称 | 功能 |
|------|------|
| `intrinsic_delaunay` | 内蕴 Delaunay 三角剖分 |
| `compute_intrinsic_lengths` | 计算内蕴边长 |
| `is_intrinsic_delaunay_edge` | 判断边是否满足内蕴 Delaunay |
| `intrinsic_cotan_weight` | 内蕴余切权重 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/intrinsic.pdf)。
