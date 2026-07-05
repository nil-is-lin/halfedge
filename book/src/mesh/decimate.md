# QEM 简化

> 设计文档：[decimate.pdf](../../../docs/decimate.pdf) | 源码：[decimate.rs](../../../src/decimate.rs)

## 简介

基于二次误差度量（Quadric Error Metric）的网格简化算法。通过迭代塌缩误差最小的边，在控制误差的前提下减少网格面数与顶点数。

## 关键 API

| 名称 | 功能 |
|------|------|
| `decimate_qem` | QEM 简化 |
| `decimate_to_vertices` | 简化至目标顶点数 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/decimate.pdf)。
