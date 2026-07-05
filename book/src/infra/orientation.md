# 面朝向一致性

> 设计文档：[orientation.pdf](../../../docs/orientation.pdf) | 源码：[orientation.rs](../../../src/orientation.rs)

## 简介

检测与修复网格面朝向的一致性。通过相邻面法向关系判断是否可定向，并修复不一致的面朝向使网格满足一致定向。

## 关键 API

| 名称 | 功能 |
|------|------|
| `are_normals_consistent` | 检测法向一致性 |
| `is_orientable` | 判断可定向性 |
| `fix_orientations` | 修复面朝向 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/orientation.pdf)。
