//! 几何工具模块
//!
//! 在 [`MeshStorage`] 之上叠加**几何查询**与**几何处理**能力：
//! - 基本量：边长 / 三角面积 / 最小内角 / 面法向 / 顶点法向（面积加权）
//! - 多边形：`polygon_area` / `polygon_normal`（Newell 方法，支持 n-gon）
//! - 包围盒：`mesh_aabb`（AABB）、`mesh_centroid`（顶点质心）
//! - 平滑：拉普拉斯平滑（统一权重 + 余切权重 `cotan_laplacian`）
//! - 特征边：`dihedral_angle`（二面角）、`is_feature_edge` / `feature_edges`
//! - 距离：点到三角形最近距离（Ericson 算法）
//!
//! ## 模块组织
//! - **query**: 基本几何查询（边长、面积、法向、内角、余切拉普拉斯、二面角、特征边）
//! - **quality**: 网格质量度量（纵横比、半径比、边长统计）
//! - **curvature**: 离散曲率（高斯、平均、主曲率）
//! - **distance**: 点到三角形距离、射线求交
//! - **aabb**: 轴对齐包围盒
//! - **smooth**: 拉普拉斯平滑、Taubin 平滑、双边去噪
//!
//! ## 设计原则
//! 所有查询函数返回 `Option<T>`：若传入的句柄失效或几何退化（零面积、
//! 零长度），返回 `None` 而非 panic。修改型函数（如拉普拉斯平滑）
//! 内部先收集所有新位置到 `Vec`，再批量写回，避免借用冲突。
//!
//! ## 拓扑约定
//! 与 [`crate::traversal`] 一致：`HalfEdge.vertex` 是 tip（目的顶点），
//! `twin.vertex` 是 origin。面边界环按 `next` 顺序遍历，CCW 朝向。
//!
//! [`crate::traversal`]: crate::traversal

// 子模块声明
mod aabb;
mod curvature;
mod distance;
mod quality;
mod query;
mod smooth;

// 重新导出所有公共 API
pub use aabb::{AABB, mesh_aabb, mesh_centroid};
pub use curvature::{
    VertexCurvature, all_gaussian_curvatures_par, all_mean_curvatures_par, gaussian_curvature,
    mean_curvature, principal_curvatures, vertex_curvature,
};
pub use distance::{
    RayHit, closest_point_on_triangle, point_triangle_distance, ray_mesh_intersection,
    ray_mesh_intersection_par, ray_mesh_intersects, ray_triangle_intersection,
};
pub use quality::{
    EdgeLengthStats, MeshQualityStats, edge_length_stats, face_aspect_ratio, face_radius_ratio,
    mesh_quality,
};
pub use query::{
    cotan_edge_weight, cotan_laplacian, dihedral_angle, edge_length, face_area, face_min_angle,
    face_normal, feature_edges, is_feature_edge, mesh_volume, mesh_volume_par, polygon_area,
    polygon_normal, surface_area, surface_area_par, vertex_normal, vertex_normals_par,
};
pub use smooth::{
    bilateral_smooth_mesh, feature_edges_par, laplacian_smooth_mesh, laplacian_smooth_mesh_par,
    laplacian_smooth_vertex, taubin_smooth_mesh,
};
