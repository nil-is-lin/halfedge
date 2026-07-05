# N-RoSy 方向场

> 设计文档：[direction_field.pdf](../../../docs/direction_field.pdf) | 源码：[direction_field.rs](../../../src/direction_field.rs)

## 简介

生成网格上的 N-RoSy 方向场，采用协变拉普拉斯特征值方法（Knoppel 2013）。N=1 为向量场、N=2 为交叉场、N=4 为帧场，并支持奇异点检测。

## 关键 API

| 名称 | 功能 |
|------|------|
| `smoothest_nrosy` | 最光滑 N-RoSy 场 |
| `smoothest_cross_field` | 最光滑交叉场 |
| `smoothest_frame_field` | 最光滑帧场 |
| `detect_singularities` | 奇异点检测 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/direction_field.pdf)。
