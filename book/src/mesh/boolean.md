# 布尔运算

> 设计文档：[boolean.pdf](../../../docs/boolean.pdf) | 源码：[boolean.rs](../../../src/boolean.rs)

## 简介

提供两个网格之间的布尔运算，支持并集、交集、差集与对称差集。基于交线计算与三角面分类实现构造实体几何（CSG）操作。

## 关键 API

| 名称 | 功能 |
|------|------|
| `boolean_union` | 并集 |
| `boolean_intersection` | 交集 |
| `boolean_difference` | 差集 |
| `boolean_operation` | 通用布尔运算 |
| `BoolOp` | 布尔运算类型枚举 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/boolean.pdf)。
