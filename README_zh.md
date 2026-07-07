# halfedge

[English](README.md) | **中文**

一个 Rust 半边网格数据结构库，提供 3D 网格处理的完整工具链：遍历、拓扑操作、几何计算、细分、抽稀、参数化、测地线、变形、布尔运算等。

## 功能特性

- **半边数据结构**：基于 slotmap 的稳定句柄（`VertexId` / `HalfEdgeId` / `FaceId` / `EdgeId`）
- **遍历**：急/惰性双迭代器（零分配）、边界环、k-ring 邻域、无向边
- **查询 DSL**：链式构建器（`v.halfedge_to(w).cw_rotated().dst_vert().run(&mesh)`）
- **拓扑操作**：边分裂 / 翻转 / 折叠 / 挤出 / poke、`add_triangle` 构建器
- **几何**：边长、面面积/法向、余切拉普拉斯、曲率（高斯 / 平均 / 主曲率）、二面角、点到三角形距离、AABB、射线相交
- **细分**：Loop、Catmull-Clark、√3
- **抽稀**：QEM（二次误差度量）简化，含鲁棒退化检测
- **参数化**：Tutte 嵌入、调和（余切）、LSCM、MVC（均值坐标）
- **测地线**：Heat Method（Crane 2013）、Dijkstra 单/多源、最短路径回溯
- **变形**：拉普拉斯曲面编辑（Sorkine 2004）、ARAP（Sorkine & Alexa 2007）
- **共形映射**：调和映射、Mobius 变换、离散共形缩放因子
- **布尔运算**：并集 / 交集 / 差集 / 对称差
- **重网格化**：各向同性 remesh
- **三角化**：耳切法 & 扇形三角化（平面 / 3D）
- **顶点焊接**：按距离阈值焊接
- **图元**：立方体 / UV 球 / 圆柱 / 圆锥 / 网格 / 圆环
- **I/O**：OBJ（n-gon）、PLY（ASCII + 二进制）、STL（ASCII & 二进制）、OFF、glTF/GLB 读写；统一 `load_mesh` / `save_mesh` 入口
- **属性系统**：OpenMesh 风格动态属性（`Any + TypeId` 类型擦除）；`Vec<Option<T>>` 底层实现 O(1) 访问
- **内建属性**：顶点法向 / UV / 颜色 / 面法向的 newtype 包装
- **MeshCache**：计算属性的惰性缓存层（面法向/面积、顶点法向/度数、边长）
- **验证**：完整拓扑自检（twin / next / 流形 / 退化）
- **连通性**：连通分量（面/顶点 BFS）、合并/拆分
- **方向**：一致性检测与修复
- **BVH**：层次包围盒（AABB 树），用于射线/最近查询
- **稀疏线性代数**：对称系统构建器 + 带雅可比预条件的共轭梯度法
- **内蕴 Delaunay**：内蕴边翻转实现 Delaunay 三角化（Fisher 2007）
- **方向场**：N-RoSy 场，基于协变拉普拉斯特征值（Knoppel 2013）
- **SDF 与 Marching Cubes**：符号距离函数 + 等值面提取（Lorensen 1987）
- **网格修复**：补洞、退化面删除、孤立顶点清理
- **鲁棒谓词**：Shewchuk `orient2d` / `orient3d` / `incircle` / `insphere`（自适应精度）
- **Serde 支持**：feature 门控的 `Serialize` / `Deserialize`（`MeshStorage` 及句柄）
- **并行**：rayon 并行迭代器覆盖几何、重网格化、抽稀、布尔、测地线、参数化、变形
- **`Scalar` 类型别名**：可配置浮点类型（默认 `f64`），为未来泛型支持预留

## 快速开始

```toml
[dependencies]
halfedge = "0.1"
```

```rust
use halfedge::storage::{MeshStorage, Vertex};
use halfedge::topology_ops::add_triangle;

let mut mesh = MeshStorage::new();
let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));
add_triangle(&mut mesh, v0, v1, v2).unwrap();
```

## 可选 feature

- **`serde`**：为 `MeshStorage`、`Vertex`、`HalfEdge`、`Face`、`EdgeId` 启用 `Serialize`/`Deserialize`：

```toml
[dependencies]
halfedge = { version = "0.1", features = ["serde"] }
```

## 示例

[`examples/`](examples/) 目录包含 15 个可独立运行的示例：

```sh
cargo run --example storage_basic       # 基础存储
cargo run --example obj_io              # OBJ 读写
cargo run --example topology_ops        # 拓扑操作
cargo run --example extrude_face        # 面挤出
cargo run --example loop_subdivision    # Loop 细分
cargo run --example laplacian_smooth    # 拉普拉斯平滑
cargo run --example validate            # 拓扑验证
cargo run --example icosphere           # icosphere 生成
cargo run --example engvis_viewer       # 交互式 3D 查看器（wgpu）
```

## 基准测试

```sh
cargo bench
```

结果写入 `target/criterion/report/index.html`。

## 文档

- 每个模块在 [`docs/`](docs/) 中有配套 LaTeX 设计文档（含算法推导、TikZ 流程图、复杂度分析）
- 中文教程网站（[mdbook](https://rust-lang.github.io/mdBook/)）在 [`book/`](book/)：

```sh
cargo install mdbook
mdbook serve book --open
```

- API 文档：<https://docs.rs/halfedge>

## 已知问题

- **`block v0.1.6` 未来不兼容告警**：使用 `viewer` feature 构建（拉入 `wgpu`/`engvis`）时，
  Cargo 会对传递依赖 `block` 打印一条"未来 Rust 版本将被拒绝"的告警。该问题源自上游依赖，
  **不影响默认构建（不启用任何 feature）**。待上游依赖发布修复后自动消失，本仓库无需特殊处理。

## 许可证

双许可证，任选其一：

- Apache License 2.0（[LICENSE-APACHE](LICENSE-APACHE)）
- MIT License（[LICENSE-MIT](LICENSE-MIT)）

## 贡献

欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解开发环境、代码规范和 PR 流程。
