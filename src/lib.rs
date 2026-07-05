//! 半边网格库（halfedge）
//!
//! 当前模块树：
//! - [`ids`]：强类型句柄（`VertexId`/`HalfEdgeId`/`FaceId`/`EdgeId`），
//!   其中 `EdgeId` 是无向边的规范代表。
//! - [`storage`]：底层 `MeshStorage` 容器，SlotMap 存储，提供
//!   CRUD、有效性判断、数据迭代器（`vertices`/`halfedges`/`faces`）与容量统计。
//! - [`traversal`]：邻域遍历迭代器（eager/lazy）、无向边迭代器（`EdgeIter`）、
//!   边界环遍历（`BoundaryLoop`/`BoundaryLoopLazy`）、k-ring 邻域、边界判定。
//! - [`query`]：链式查询 DSL（Builder 模式），支持
//!   `v.halfedge_to(w).cw_rotated().dst_vert().run(&mesh)` 风格的延迟链式查询，
//!   现已覆盖 `VertexId`/`HalfEdgeId`/`FaceId` 三类句柄。
//! - [`topology_ops`]：拓扑操作（split/flip/collapse/extrude/poke），
//!   高级构建器（`add_triangle`），网格校验（`validate_mesh`）。
//! - [`geometry`]：几何工具（边长、面积、法向、最小内角、余切拉普拉斯、
//!   拉普拉斯平滑、二面角/特征边检测、点到三角距离）；多边形面积/法向
//!   （Newell 方法）；AABB 包围盒与质心。
//! - [`subdiv`]：Loop/Catmull-Clark/√3 曲面细分。
//! - [`decimate`]：QEM 二次误差度量网格简化。
//! - [`property`]：OpenMesh 风格属性系统（`Any + TypeId` 类型擦除）。
//! - [`connectivity`]：连通分量分析（面连通 / 顶点连通，BFS）。
//! - [`orientation`]：面朝向一致性检测（`are_normals_consistent`/
//!   `is_orientable`）与修复（`fix_orientations`）。
//! - [`weld`]：顶点焊接（按距离阈值合并邻近顶点）。
//! - [`validate`]：拓扑自检（twin/next/悬空 ID/退化/流形约束的完整校验）。
//! - [`io`]：OBJ 加载/保存（支持 n-gon）+ PLY ASCII 加载/保存 +
//!   从顶点面索引构建半边网格的 builder。
//! - [`export`]：导出 wgpu 兼容的顶点/索引缓冲。
//! - [`test_util`]：测试夹具（icosphere 球体生成）。
//! - [`linalg`]：稀疏线性代数（稀疏系统构建器 + 共轭梯度法）。
//! - [`parameterization`]：曲面参数化（Tutte 重心映射、调和参数化、LSCM）。
//! - [`geodesics`]：测地线距离（Heat Method + 最短路径回溯）。
//! - [`conformal`]：共形映射（调和映射、Möbius 变换、离散共形比例因子）。
//! - [`builtin_attrs`]：内置属性（顶点法向 / UV / 颜色 / 面法向）的 Newtype 包装与 OBJ 属性感知 IO。
//! - [`bvh`]：BVH（AABB 二叉树）加速结构，加速射线求交与最近点查询。
//! - [`deformation`]：网格变形（Laplacian 变形 + ARAP 变形）。

pub mod boolean;
pub mod builtin_attrs;
pub mod bvh;
pub mod conformal;
pub mod connectivity;
pub mod decimate;
pub mod deformation;
pub mod export;
pub mod geodesics;
pub mod geometry;
pub mod ids;
pub mod io;
pub mod linalg;
pub mod orientation;
pub mod parameterization;
pub mod primitives;
pub mod property;
pub mod query;
pub mod remesh;
pub mod storage;
pub mod subdiv;
pub mod test_util;
pub mod topology_ops;
pub mod traversal;
pub mod triangulation;
pub mod validate;
pub mod weld;

pub use boolean::{
    BoolOp, boolean_difference, boolean_intersection, boolean_operation,
    boolean_symmetric_difference, boolean_union,
};
pub use builtin_attrs::{
    FaceNormal, VertexColor, VertexNormal, VertexUv, add_face_normals, add_vertex_colors,
    add_vertex_normals, add_vertex_uvs, collect_vertex_colors, collect_vertex_normals,
    collect_vertex_uvs, format_obj_with_attrs, install_vertex_attrs, parse_obj_with_attrs,
    populate_face_normals, populate_vertex_normals,
};
pub use bvh::{Bvh, LEAF_MAX_FACES};
pub use conformal::{
    apply_mobius_transform, compute_vertex_scale_factors, harmonic_map, mobius_to_center,
};
pub use connectivity::{
    component_count, component_of_face, connected_components, extract_component, extract_faces,
    merge_meshes, split_into_components, vertex_connected_components,
};
pub use decimate::{decimate_qem, decimate_to_vertices};
pub use deformation::{DeformationConstraint, arap_deformation, laplacian_deformation};
pub use export::mesh_to_vertex_index_buffers;
pub use geodesics::{
    dijkstra_geodesic, dijkstra_multi_source_geodesic, dijkstra_shortest_path,
    dijkstra_with_parent, geodesic_distance_from_vertex, multi_source_geodesic, shortest_path,
};
pub use geometry::{
    AABB, EdgeLengthStats, MeshQualityStats, RayHit, VertexCurvature, all_gaussian_curvatures_par,
    all_mean_curvatures_par, bilateral_smooth_mesh, closest_point_on_triangle, cotan_laplacian,
    dihedral_angle, edge_length, edge_length_stats, face_area, face_aspect_ratio, face_min_angle,
    face_normal, face_radius_ratio, feature_edges, feature_edges_par, gaussian_curvature,
    is_feature_edge, laplacian_smooth_mesh, laplacian_smooth_mesh_par, laplacian_smooth_vertex,
    mean_curvature, mesh_aabb, mesh_centroid, mesh_quality, mesh_volume, mesh_volume_par,
    point_triangle_distance, polygon_area, polygon_normal, principal_curvatures,
    ray_mesh_intersection, ray_mesh_intersection_par, ray_mesh_intersects,
    ray_triangle_intersection, surface_area, surface_area_par, taubin_smooth_mesh,
    vertex_curvature, vertex_normal, vertex_normals_par,
};
pub use ids::{EdgeId, FaceId, HalfEdgeId, VertexId};
pub use io::{
    MeshError, ObjError, PlyError, StlError, build_mesh_from_polygons,
    build_mesh_from_vertices_and_faces, format_obj, format_ply, format_stl_ascii,
    format_stl_binary, load_mesh, load_obj, load_ply, load_stl, parse_obj, parse_ply,
    parse_stl_ascii, parse_stl_binary, parse_stl_bytes, save_mesh, save_obj, save_ply,
    save_stl_ascii, save_stl_binary,
};
pub use linalg::{SparseSystem, conjugate_gradient, regularize_diagonal};
pub use orientation::{are_normals_consistent, fix_orientations, is_orientable};
pub use parameterization::{
    harmonic_parameterization, lscm, mvc_parameterization, tutte_embedding,
};
pub use primitives::{
    build_cone, build_cube, build_cylinder, build_grid, build_torus, build_uv_sphere,
};
pub use property::{MeshProperties, PropertyHandle, PropertyStore};
pub use remesh::{RemeshStats, isotropic_remesh, quick_remesh, remesh_to_length};
pub use storage::{Face, HalfEdge, MeshStorage, Vertex};
pub use subdiv::catmull_clark::catmull_clark_subdivide;
pub use subdiv::loop_subdivide;
pub use subdiv::sqrt3::sqrt3_subdivide;
pub use test_util::build_icosphere;
pub use topology_ops::{
    TopologyError, add_triangle, collapse_edge, collapse_edge_at, extrude_face, extrude_faces,
    extrude_region, flip_edge, split_edge, split_face, validate_mesh,
};
pub use traversal::{
    BoundaryLoop, BoundaryLoopLazy, EdgeIter, VertexAdjacentEdges, boundary_loops,
    is_boundary_edge, is_boundary_vertex, is_closed,
};
pub use triangulation::{ear_clipping, ear_clipping_3d, fan_triangulation, fan_triangulation_3d};
pub use validate::{ValidationError, check_topology, validate_topology};
pub use weld::weld_vertices;
