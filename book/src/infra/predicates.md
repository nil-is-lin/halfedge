# 鲁棒几何谓词

> 设计文档：[predicates.pdf](../../../docs/predicates.pdf) | 源码：[predicates.rs](../../../src/predicates.rs)

## 简介

基于 Shewchuk 自适应精度浮点运算的鲁棒几何谓词，提供 `orient2d`/`orient3d`/`incircle`/`insphere` 等判定，在退化情况下保证符号精确。

## 关键 API

| 名称 | 功能 |
|------|------|
| `orient2d` | 二维方向判定 |
| `orient3d` | 三维方向判定 |
| `incircle` | 圆内判定 |
| `insphere` | 球内判定 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/predicates.pdf)。
