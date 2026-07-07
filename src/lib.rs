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
//! - [`predicates`]：Shewchuk 鲁棒几何谓词（`orient2d` / `orient3d` /
//!   `incircle` / `insphere`），自适应精度浮点运算，退化情况下保证符号精确。
//! - [`weld`]：顶点焊接（按距离阈值合并邻近顶点）。
//! - [`validate`]：拓扑自检（twin/next/悬空 ID/退化/流形约束的完整校验）。
//! - [`io`]：多格式 I/O（OBJ/PLY/STL/OFF/glTF），
//!   支持 ASCII 与二进制格式，统一 `load_mesh`/`save_mesh` 入口。
//! - [`export`]：导出 wgpu 兼容的顶点/索引缓冲。
//! - [`test_util`]：测试夹具（icosphere 球体生成）。
//! - [`linalg`]：稀疏线性代数（稀疏系统构建器 + 共轭梯度法，
//!   含 Jacobi 预条件子加速收敛）。
//! - [`cache`]：惰性缓存层（`MeshCache`），缓存面法向、面积、
//!   顶点法向、顶点价、边长等常用几何数据，避免重复计算。
//! - [`boolean`]：布尔运算（并集/交集/差集/对称差）。
//! - [`remesh`]：各向同性重网格（Botsch-Kobbelt 风格），
//!   支持并行迭代平滑。
//! - [`repair`]：网格修复工具（补孔、去退化面、去孤立顶点）。
//! - [`sdf`]：有符号距离场（SDF 基本体、CSG、Marching Cubes 等值面提取）。
//! - [`primitives`]：参数化基本体生成（球/立方体/圆柱/圆锥/环面/网格）。
//! - [`parameterization`]：曲面参数化（Tutte 重心映射、调和参数化、LSCM）。
//! - [`geodesics`]：测地线距离（Heat Method + 最短路径回溯）。
//! - [`conformal`]：共形映射（调和映射、Möbius 变换、离散共形比例因子）。
//! - [`builtin_attrs`]：内置属性（顶点法向 / UV / 颜色 / 面法向）的 Newtype 包装与 OBJ 属性感知 IO。
//! - [`bvh`]：BVH（AABB 二叉树）加速结构，加速射线求交与最近点查询。
//! - [`deformation`]：网格变形（Laplacian 变形 + ARAP 变形）。
//!
//! ## 网格假设（Mesh assumptions）
//!
//! 本库针对**有向、流形、三角**网格设计：
//!
//! - 每个面为三角形；非三角形面（如四边形）虽可被底层存储容纳，但多数算法假设三角化输入；
//! - 网格为流形：每条无向边至多被 2 个面共享，每个顶点处的半边环闭合（内部顶点）或为恰有 2 个端点的边界链（边界顶点）；
//! - 面朝向一致（可定向）。
//!
//! 非流形、含退化面 / 悬空 ID、或面朝向不一致的输入，可能在 [`validate`] /
//! [`topology_ops::validate_mesh`] 处被拒绝，或产生不正确的结果。对非流形网格的
//! 支持见 `docs/nonmanifold_design.md`（Roadmap 的 P1 项，尚未实现）。
//!
//! ## 快速开始（Quick start）
//!
//! ```rust
//! use halfedge::{build_cube, face_normal, validate_mesh, FaceId};
//!
//! let mesh = build_cube(1.0);
//! let first_face: FaceId = mesh.face_ids().next().expect("cube has faces");
//! let n = face_normal(&mesh, first_face).expect("face has a normal");
//! assert!(n.iter().all(|x| x.is_finite()));
//! validate_mesh(&mesh).expect("a cube is a valid manifold mesh");
//! ```
//!
//! ## API 稳定性
//!
//! 此库使用三级稳定性标记（详见各模块文档）。在 1.0 之前：
//!
//! - **Stable**：核心数据结构与基础操作，承诺语义稳定。包括 [`ids`]、[`storage`]、
//!   [`traversal`]、[`topology_ops`]、[`geometry`]、[`io`]、[`property`]、[`validate`]、
//!   [`predicates`]、[`weld`]、[`primitives`]。
//! - **Unstable**：高级算法，API 可能在 1.0 之前调整（签名、命名、错误类型）。包括
//!   [`subdiv`]、[`decimate`]、[`remesh`]、[`repair`]、[`orientation`]、[`connectivity`]、
//!   [`cache`]、[`linalg`]、[`bvh`]、[`boolean`]、[`geodesics`]、[`deformation`]、
//!   [`parameterization`]、[`conformal`]、[`builtin_attrs`]、[`export`]。
//! - **Experimental**：研究性或最新添加的功能，可能大幅变动或移除。包括 [`direction_field`]、
//!   [`intrinsic`]、[`sdf`]、[`triangulation`]。
//!
//! 1.0 将冻结 Stable 与 Unstable 等级的公开 API。

/// 库使用的默认浮点标量类型。
///
/// 当前为 `f64`。所有公开 API 中的浮点参数均使用此类型别名，
/// 便于未来迁移至 `f32` 或泛型标量（`S: num_traits::Float`）。
pub type Scalar = f64;

/// 三维向量类型（`[Scalar; 3]`）。
pub type Vec3 = [Scalar; 3];

/// 精确算术标量类型，启用 `exact` feature 时可用。
///
/// 类型别名为 [`num_rational::Rational64`]，为**未来的精确算术内核**预留。
/// 当前（v0.1.x）几何内核仍全部使用 [`Scalar`]（`f64`），此类型**尚未接入任何运算**，
/// 仅作为 early-adopter 的占位与接口锚点。完整接入是 Roadmap 的 P1 项。
#[cfg(feature = "exact")]
pub type ExactScalar = num_rational::Rational64;

/// **Stability:** unstable — boolean operations may change before 1.0.
pub mod boolean;
/// **Stability:** unstable — builtin attribute API may change before 1.0.
pub mod builtin_attrs;
/// **Stability:** unstable — BVH API may change before 1.0.
pub mod bvh;
/// **Stability:** unstable — cache API may change before 1.0.
pub mod cache;
/// **Stability:** unstable — conformal mapping API may change before 1.0.
pub mod conformal;
/// **Stability:** unstable — connectivity API may change before 1.0.
pub mod connectivity;
/// **Stability:** unstable — decimation API may change before 1.0.
pub mod decimate;
/// **Stability:** unstable — deformation API may change before 1.0.
pub mod deformation;

/// 内部模块：方向场计算。公开 API 通过 re-export 访问。
#[doc(hidden)]
pub mod direction_field;

/// 内部模块：wgpu 缓冲导出。公开 API 通过 re-export 访问。
#[doc(hidden)]
pub mod export;

/// **Stability:** unstable — geodesics API may change before 1.0.
pub mod geodesics;
/// **Stability:** stable — core geometry API.
pub mod geometry;
/// **Stability:** stable — handle types.
pub mod ids;

/// 内部模块：内蕴 Delaunay 优化。公开 API 通过 re-export 访问。
#[doc(hidden)]
pub mod intrinsic;

/// **Stability:** stable — I/O API.
pub mod io;
/// **Stability:** unstable — linear algebra API may change before 1.0.
pub mod linalg;
/// **Stability:** unstable — orientation API may change before 1.0.
pub mod orientation;
/// **Stability:** unstable — parameterization API may change before 1.0.
pub mod parameterization;
/// **Stability:** stable — robust geometric predicates.
pub mod predicates;
/// **Stability:** stable — primitive builders.
pub mod primitives;
/// **Stability:** stable — property system.
pub mod property;

/// 内部模块：链式查询 DSL。大多数用户通过 [`traversal`] 模块即可满足需求。
#[doc(hidden)]
pub mod query;

/// **Stability:** unstable — remeshing API may change before 1.0.
pub mod remesh;
/// **Stability:** unstable — repair API may change before 1.0.
pub mod repair;
/// **Stability:** experimental — SDF / Marching Cubes API may change significantly.
pub mod sdf;
/// **Stability:** stable — storage container.
pub mod storage;
/// **Stability:** unstable — subdivision API may change before 1.0.
pub mod subdiv;

/// 内部模块：测试夹具，不属于稳定公开 API。
#[doc(hidden)]
pub mod test_util;

/// **Stability:** stable — topology operations.
pub mod topology_ops;
/// **Stability:** stable — traversal iterators.
pub mod traversal;

/// 内部模块：三角化工具。公开 API 通过 re-export 访问。
#[doc(hidden)]
pub mod triangulation;

/// **Stability:** stable — topology validation.
pub mod validate;
/// **Stability:** stable — vertex welding.
pub mod weld;

pub use boolean::{
    BoolOp, boolean_difference, boolean_intersection, boolean_operation,
    boolean_symmetric_difference, boolean_union,
};
pub use builtin_attrs::{
    FaceColor, FaceNormal, FaceOpacity, FaceSelected, HalfEdgeColor, HalfEdgeSelected,
    HalfEdgeThickness, VertexColor, VertexNormal, VertexSelected, VertexSize, VertexUv,
    add_face_colors, add_face_normals, add_face_opacity, add_face_selection, add_halfedge_colors,
    add_halfedge_selection, add_halfedge_thickness, add_vertex_colors, add_vertex_normals,
    add_vertex_selection, add_vertex_sizes, add_vertex_uvs, clear_face_colors, clear_face_opacity,
    clear_face_selection, clear_halfedge_selection, clear_halfedge_thickness,
    clear_vertex_selection, clear_vertex_sizes, collect_vertex_colors, collect_vertex_normals,
    collect_vertex_uvs, count_selected_faces, count_selected_vertices, deselect_edge,
    deselect_face, deselect_halfedge, deselect_vertex, edge_color, edge_thickness, face_color,
    face_opacity, format_obj_with_attrs, halfedge_thickness, install_vertex_attrs,
    invert_vertex_selection, is_edge_selected, is_face_selected, is_halfedge_selected,
    is_vertex_selected, parse_obj_with_attrs, populate_face_normals, populate_vertex_normals,
    select_all_faces, select_all_vertices, select_edge, select_face, select_halfedge,
    select_vertex, selected_edge_ids, selected_face_ids, selected_halfedge_ids,
    selected_vertex_ids, set_edge_color, set_edge_thickness, set_face_color, set_face_opacity,
    set_halfedge_thickness, set_uniform_edge_thickness, set_uniform_face_opacity,
    set_uniform_vertex_size, set_vertex_size, toggle_edge_selection, toggle_face_selection,
    toggle_halfedge_selection, toggle_vertex_selection, vertex_size,
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
pub use direction_field::{
    FaceLocalFrame, Singularity, build_face_local_frames, compute_transport_angles,
    detect_singularities, smoothest_cross_field, smoothest_frame_field, smoothest_nrosy,
    smoothest_vector_field,
};
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
pub use intrinsic::{
    IntrinsicDelaunayStats, compute_intrinsic_lengths, intrinsic_cotan_weight, intrinsic_delaunay,
    is_intrinsic_delaunay_edge,
};
pub use io::{
    MeshBuildError, MeshError, ObjError, PlyError, StlError, build_mesh_from_polygons,
    build_mesh_from_vertices_and_faces, format_obj, format_ply, format_stl_ascii,
    format_stl_binary, load_mesh, load_obj, load_ply, load_stl, parse_obj, parse_ply,
    parse_stl_ascii, parse_stl_binary, parse_stl_bytes, save_mesh, save_obj, save_ply,
    save_stl_ascii, save_stl_binary,
};
pub use linalg::{
    SparseSystem, build_cotan_laplacian, build_vertex_index, conjugate_gradient,
    regularize_diagonal,
};
pub use orientation::{are_normals_consistent, fix_orientations, is_orientable};
pub use parameterization::{
    harmonic_parameterization, lscm, mvc_parameterization, tutte_embedding,
};
pub use predicates::{
    incircle, insphere, is_ccw2d, is_collinear2d, is_convex_vertex2d, is_coplanar, orient2d,
    orient3d, point_in_triangle_2d, signed_area2d, tet_signed_volume, triangle_area_2d,
};
pub use primitives::{
    build_cone, build_cube, build_cylinder, build_grid, build_torus, build_uv_sphere,
};
pub use property::{MeshProperties, PropertyHandle, PropertyStore};
pub use remesh::{RemeshStats, isotropic_remesh, quick_remesh, remesh_to_length};
pub use repair::{
    RepairStats, detect_nonmanifold_edges, detect_nonmanifold_vertices, fill_all_holes, fill_hole,
    remove_degenerate_faces, remove_face, remove_isolated_vertices, repair_mesh,
};
pub use sdf::{
    McParams, Sdf, SdfBox, SdfCapsule, SdfDifference, SdfIntersection, SdfSmoothUnion, SdfSphere,
    SdfTorus, SdfTranslate, SdfUnion, march_field, march_sdf,
};
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
pub use validate::{ValidationError, check_topology, validate_first_error, validate_topology};
pub use weld::weld_vertices;
