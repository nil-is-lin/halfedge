# 属性系统

> 设计文档：[property.pdf](../../../docs/property.pdf) | 源码：[property.rs](../../../src/property.rs)

## 简介

OpenMesh 风格的动态属性系统，基于 `Any + TypeId` 类型擦除实现。允许在运行时为顶点、半边、面等元素附加任意类型的命名属性。

## 关键 API

| 名称 | 功能 |
|------|------|
| `MeshProperties` | 属性容器 |
| `PropertyHandle` | 属性句柄 |
| `PropertyStore` | 属性存储 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/property.pdf)。
