# 曲面参数化

> 设计文档：[parameterization.pdf](../../../docs/parameterization.pdf) | 源码：[parameterization.rs](../../../src/parameterization.rs)

## 简介

提供曲面参数化算法，将三维网格映射到二维参数域。包含 Tutte 重心映射、调和参数化、LSCM（最小二乘共形）与 MVC（均值坐标）参数化。

## 关键 API

| 名称 | 功能 |
|------|------|
| `tutte_embedding` | Tutte 重心映射 |
| `harmonic_parameterization` | 调和参数化 |
| `lscm` | 最小二乘共形参数化 |
| `mvc_parameterization` | 均值坐标参数化 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/parameterization.pdf)。
