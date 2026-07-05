# halfedge

[![Crates.io](https://img.shields.io/crates/v/halfedge.svg)](https://crates.io/crates/halfedge)
[![Documentation](https://docs.rs/halfedge/badge.svg)](https://docs.rs/halfedge)
[![License](https://img.shields.io/crates/l/halfedge.svg)](https://github.com/nil-is-lin/halfedge)

一个基于半边数据结构的 Rust 三维网格处理库，提供从拓扑遍历到高级几何算法的完整工具链。

## 特性

- **半边数据结构**：基于 SlotMap 的稳定句柄（`VertexId` / `HalfEdgeId` / `FaceId` / `EdgeId`），版本号自动防悬垂
- **双模迭代器**：eager（预收集，可变借用）与 lazy（零分配，不可变借用）双策略
- **链式查询 DSL**：`v.halfedge_to(w).cw_rotated().dst_vert().run(&mesh)`
- **拓扑操作全系列**：split / flip / collapse / extrude / poke，每次操作自动验证
- **几何算法**：参数化、测地线、共形映射、网格变形、方向场、内蕴 Delaunay
- **网格处理**：细分（Loop/Catmull-Clark/√3）、QEM 简化、布尔运算、重网格化、Marching Cubes
- **鲁棒谓词**：Shewchuk orient2d / orient3d / incircle / insphere
- **I/O**：OBJ / PLY / STL 读写
- **并行**：rayon 并行曲率/平滑/面积计算

## 统计

| 指标 | 数量 |
|------|------|
| 源码模块 | 35 |
| LaTeX 设计文档 | 29 |
| 单元测试 | 477 |
| Criterion 基准 | 4 组 |
| 可运行示例 | 15 |

## 设计文档

每个模块都有对应的 LaTeX 设计文档（含算法公式、流程图、复杂度分析），位于 [`docs/`](https://github.com/nil-is-lin/halfedge/tree/main/docs) 目录。本教程各章节均链接到对应 PDF。

## 安装

```toml
[dependencies]
halfedge = "0.1"
```

## 许可证

MIT OR Apache-2.0
