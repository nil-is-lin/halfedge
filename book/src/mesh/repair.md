# 网格修复

> 设计文档：[repair.pdf](../../../docs/repair.pdf) | 源码：[repair.rs](../../../src/repair.rs)

## 简介

提供网格修复功能，包括洞填充、退化面移除与孤立顶点清理，修复常见拓扑缺陷使网格满足流形与封闭要求。

## 关键 API

| 名称 | 功能 |
|------|------|
| `fill_hole` | 填充单个洞 |
| `fill_all_holes` | 填充所有洞 |
| `remove_degenerate_faces` | 移除退化面 |
| `remove_isolated_vertices` | 移除孤立顶点 |
| `repair_mesh` | 综合修复 |
| `RepairStats` | 修复统计 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/repair.pdf)。
