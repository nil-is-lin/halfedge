# 共形映射

> 设计文档：[conformal.pdf](../../../docs/conformal.pdf) | 源码：[conformal.rs](../../../src/conformal.rs)

## 简介

提供离散共形映射工具。包含调和映射、Möbius 变换与离散共形比例因子计算，用于角度保持的曲面参数化与变形。

## 关键 API

| 名称 | 功能 |
|------|------|
| `harmonic_map` | 调和映射 |
| `apply_mobius_transform` | 应用 Möbius 变换 |
| `compute_vertex_scale_factors` | 计算顶点共形比例因子 |
| `mobius_to_center` | Möbius 变换归中 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/conformal.pdf)。
