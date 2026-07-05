# 曲面细分

> 设计文档：[subdiv.pdf](../../../docs/subdiv.pdf) | 源码：[subdiv.rs](../../../src/subdiv.rs)

## 简介

提供曲面细分算法，通过迭代细化提升网格分辨率与光滑度。包含 Loop（三角形网格）、Catmull-Clark（任意网格）与 √3 细分。

## 关键 API

| 名称 | 功能 |
|------|------|
| `loop_subdivide` | Loop 细分 |
| `catmull_clark_subdivide` | Catmull-Clark 细分 |
| `sqrt3_subdivide` | √3 细分 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/subdiv.pdf)。
