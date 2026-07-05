# 拓扑操作

> 设计文档：[topology_ops.pdf](../../../docs/topology_ops.pdf) | 源码：[topology_ops.rs](../../../src/topology_ops.rs)

## 简介

提供网格拓扑编辑操作，包括 split/flip/collapse/extrude/poke 等。包含 `add_triangle` 构建器与 `validate_mesh` 校验，并检查链接条件以维护流形性质。

## 关键 API

| 名称 | 功能 |
|------|------|
| `add_triangle` | 三角形构建器 |
| `split_edge` | 边分裂 |
| `flip_edge` | 边翻转 |
| `collapse_edge` | 边塌缩 |
| `extrude_face` | 面拉伸 |
| `validate_mesh` | 网格校验 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/topology_ops.pdf)。
