# 稀疏线性代数

> 设计文档：[linalg.pdf](../../../docs/linalg.pdf) | 源码：[linalg.rs](../../../src/linalg.rs)

## 简介

提供稀疏线性代数工具，包含对称系统构建器与共轭梯度法（CG）求解器，服务于参数化、变形、测地线等需要求解大型稀疏线性系统的算法。

## 关键 API

| 名称 | 功能 |
|------|------|
| `SparseSystem` | 稀疏对称系统构建器 |
| `conjugate_gradient` | 共轭梯度法求解 |
| `regularize_diagonal` | 对角正则化 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/linalg.pdf)。
