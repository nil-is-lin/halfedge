//! 拓扑操作模块
//!
//! 在 [`MeshStorage`] 之上叠加拓扑操作：
//! - [`split_edge`]：边分裂（在中点插入新顶点）
//! - [`flip_edge`]：内部边翻转（替换四边形对角线）
//! - [`collapse_edge`] / [`collapse_edge_at`]：边折叠（合并两端顶点）
//! - [`split_face`]：面分裂 / poke（在面中心插入新顶点，1→3 面）
//! - [`extrude_face`] / [`extrude_faces`] / [`extrude_region`]：面挤出
//! - [`add_triangle`]：高级面构建器，自动完成半边拓扑连接与 twin 配对
//!
//! 每个操作内置合法性校验（`validate_mesh`），保证操作后网格仍为流形三角曲面。
//!
//! ## 拓扑约定
//! - `HalfEdge.vertex` 是 tip（目的顶点），origin = `twin.vertex`
//! - 同面 `next/prev` 形成 CCW 闭合环
//! - `twin` 互指，构成无向边
//! - 边界半边 `face = None`
//!
//! ## 旋转规则（CCW 朝向网格，绕 origin）
//! - CCW next: `he.prev.twin`
//! - CW next: `he.twin.next`

mod builders;
mod edit;
mod extrude;
mod helpers;
mod validate;

#[cfg(test)]
mod tests;

pub use builders::add_triangle;
pub use edit::{collapse_edge, collapse_edge_at, flip_edge, split_edge, split_face};
pub use extrude::{extrude_face, extrude_faces, extrude_region};
pub use helpers::TopologyError;
pub use validate::validate_mesh;
