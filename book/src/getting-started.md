# 快速开始

## 安装

在 `Cargo.toml` 中添加：

```toml
[dependencies]
halfedge = "0.1"
```

## 第一个网格

```rust
use halfedge::storage::{MeshStorage, Vertex};
use halfedge::topology_ops::add_triangle;

let mut mesh = MeshStorage::new();
let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));
add_triangle(&mut mesh, v0, v1, v2).unwrap();
```

## 链式查询 DSL

```rust
use halfedge::query::MeshQuery;

// 查询 v0 到 v1 之间半边，CW 旋转后的目标顶点
let dst = v0.halfedge_to(v1)
    .cw_rotated()
    .dst_vert()
    .run(&mesh);
```

## 遍历邻域

```rust
use halfedge::traversal::{VertexRing, VertexAdjacentVerts};

// eager（预收集，迭代期可 &mut mesh）
for he in VertexRing::new(&mesh, v0) {
    println!("halfedge: {:?}", he);
}

// lazy（零分配，迭代期持有 &mesh）
for neighbor in VertexAdjacentVerts::lazy(&mesh, v0) {
    println!("neighbor: {:?}", neighbor);
}
```

## 几何查询

```rust
use halfedge::geometry::{edge_length, face_normal, face_area};

for fid in mesh.face_ids() {
    if let Some(area) = face_area(&mesh, fid) {
        println!("face {:?} area = {}", fid, area);
    }
}
```

## 运行示例

```sh
cargo run --example storage_basic      # 基本存储
cargo run --example traversal          # 遍历演示
cargo run --example loop_subdivision   # Loop 细分
cargo run --example icosphere          # icosphere 生成
cargo run --example engvis_viewer      # 交互式 3D 查看器
```

## 运行基准

```sh
cargo bench --bench traversal          # 遍历迭代器基准
cargo bench --bench topology_ops       # 拓扑操作基准
cargo bench --bench subdiv             # 细分基准
cargo bench --bench geometry_algs      # 几何算法基准
```

## 下一步

- 阅读[句柄与 ID](./core/ids.md)了解数据结构基础
- 阅读[邻域遍历迭代器](./core/traversal.md)了解 eager/lazy 双策略
- 各章节均链接到对应的 LaTeX 设计文档 PDF，包含完整算法公式与流程图
