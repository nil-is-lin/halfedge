# I/O

> 设计文档：[io.pdf](../../../docs/io.pdf) | 源码：[io.rs](../../../src/io.rs)

## 简介

提供网格文件读写支持，包括 OBJ（支持 n-gon）、PLY（ASCII）与 STL（ASCII 及二进制）格式的加载与保存，以及从顶点/面索引构建半边网格的 builder。

## 关键 API

| 名称 | 功能 |
|------|------|
| `load_obj` / `save_obj` | OBJ 读写 |
| `load_ply` / `save_ply` | PLY 读写 |
| `load_stl` / `save_stl_ascii` | STL 读写 |
| `parse_obj` | 解析 OBJ 文本 |
| `format_obj` | 格式化为 OBJ 文本 |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/io.pdf)。
