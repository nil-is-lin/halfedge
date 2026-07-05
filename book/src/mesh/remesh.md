# 各向同性重网格化

> 设计文档：[remesh.pdf](../../../docs/remesh.pdf) | 源码：[remesh.rs](../../../src/remesh.rs)

## 简介

提供各向同性重网格化算法，按目标边长迭代进行边分裂/塌缩/翻转与切向平滑，使网格三角形单元趋于均匀正则。

## 关键 API

| 名称 | 功能 |
|------|------|
| `isotropic_remesh` | 各向同性重网格化 |
| `remesh_to_length` | 按目标边长重网格化 |
| `quick_remesh` | 快速重网格化 |
| `RemeshStats` | 重网格化统计 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/remesh.pdf)。
