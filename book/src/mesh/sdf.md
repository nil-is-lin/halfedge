# SDF 与 Marching Cubes

> 设计文档：[sdf.pdf](../../../docs/sdf.pdf) | 源码：[sdf.rs](../../../src/sdf.rs)

## 简介

提供有符号距离函数（SDF）图元、CSG 组合操作与 Marching Cubes 等值面提取（Lorensen 1987），可从隐式场生成三角网格。

## 关键 API

| 名称 | 功能 |
|------|------|
| `Sdf` | SDF trait |
| `SdfSphere` / `SdfBox` / `SdfTorus` | SDF 图元 |
| `SdfUnion` / `SdfSmoothUnion` | CSG 组合操作 |
| `march_sdf` | Marching Cubes 等值面提取 |
| `march_field` | 标量场等值面提取 |
| `McParams` | Marching Cubes 参数 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/sdf.pdf)。
