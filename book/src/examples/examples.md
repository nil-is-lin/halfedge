# 示例总览

> 设计文档：[examples.pdf](../../../docs/examples.pdf) | 源码：[examples/](../../../examples/)

## 简介

仓库 `examples/` 目录下提供 15 个可运行示例，覆盖从基础存储、拓扑操作到几何查询、细分、属性系统与 GPU 导出的典型用法。

## 关键 API

| 示例名 | 模块 | 运行命令 |
|--------|------|----------|
| `storage_basic` | storage | `cargo run --example storage_basic` |
| `build_mesh` | topology_ops | `cargo run --example build_mesh` |
| `obj_io` | io | `cargo run --example obj_io` |
| `traversal` | traversal | `cargo run --example traversal` |
| `topology_ops` | topology_ops | `cargo run --example topology_ops` |
| `extrude_face` | topology_ops | `cargo run --example extrude_face` |
| `geometry_query` | geometry | `cargo run --example geometry_query` |
| `laplacian_smooth` | geometry | `cargo run --example laplacian_smooth` |
| `point_triangle_distance` | geometry | `cargo run --example point_triangle_distance` |
| `loop_subdivision` | subdiv | `cargo run --example loop_subdivision` |
| `property` | property | `cargo run --example property` |
| `validate` | validate | `cargo run --example validate` |
| `export_wgpu` | export | `cargo run --example export_wgpu` |
| `icosphere` | test_util | `cargo run --example icosphere` |
| `engvis_viewer` | export | `cargo run --example engvis_viewer` |

## 更多

完整的算法公式、流程图、复杂度分析与测试覆盖详见 [设计文档 PDF](../../../docs/examples.pdf)。
