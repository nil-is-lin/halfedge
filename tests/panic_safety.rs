//! 历史 panic 回归测试
//!
//! 集中验证缺陷 #23 修复的关键 panic 路径不会复现。
//! 每个测试对应代码评审报告中曾识别的 unwrap / 除零 / 越界风险。

use halfedge::ids::HalfEdgeId;
use halfedge::{
    build_cube, build_icosphere,
    geometry::{mesh_volume, mesh_volume_par, ray_mesh_intersects, surface_area, surface_area_par},
    io::{build_mesh_from_vertices_and_faces, parse_obj},
    linalg::{SparseSystem, conjugate_gradient},
    storage::MeshStorage,
    topology_ops::{collapse_edge, flip_edge, split_edge},
    traversal::is_closed,
    validate::validate_topology,
};

// ============================================================
// geometry.rs：mesh_volume / mesh_volume_par 不应 panic
// ============================================================

/// 历史 bug：`mesh_volume_par` 在三角面顶点 ID 失效时 unwrap 会 panic。
/// 现已改为 match 守卫，跳过不一致面返回 0 贡献。
#[test]
fn regression_mesh_volume_par_on_empty_mesh_no_panic() {
    let empty = MeshStorage::new();
    let v = mesh_volume_par(&empty);
    assert_eq!(v, 0.0);
}

#[test]
fn regression_mesh_volume_par_consistent_with_serial() {
    let mesh = build_icosphere(1);
    let serial = mesh_volume(&mesh);
    let par = mesh_volume_par(&mesh);
    assert!(
        (serial - par).abs() < 1e-9,
        "串行与并行体积应一致: {} vs {}",
        serial,
        par
    );
}

#[test]
fn regression_surface_area_par_on_empty_mesh_no_panic() {
    let empty = MeshStorage::new();
    let a = surface_area_par(&empty);
    assert_eq!(a, 0.0);
}

// ============================================================
// geometry.rs：ray_mesh_intersects 不应 panic
// ============================================================

/// 历史 bug：空网格上 `ray_mesh_intersects` 遍历面时取顶点 unwrap 会 panic。
#[test]
fn regression_ray_mesh_intersects_empty_mesh_no_panic() {
    let empty = MeshStorage::new();
    let origin = [0.0, 0.0, 0.0];
    let dir = [0.0, 0.0, 1.0];
    let _ = ray_mesh_intersects(origin, dir, &empty);
}

#[test]
fn regression_ray_mesh_intersects_cube_outside_returns_false() {
    // 射线从立方体外部穿过——奇偶校验应判定为 "外部" → false
    let cube = build_cube(1.0);
    let origin = [-5.0, 0.5, 0.5];
    let dir = [1.0, 0.0, 0.0];
    let inside = ray_mesh_intersects(origin, dir, &cube);
    // 2 个交点 = 偶数 = 起点在外部
    assert!(!inside, "起点在网格外应返回 false");
}

// ============================================================
// linalg.rs：conjugate_gradient 不应 panic
// ============================================================

/// 历史 bug：维度不匹配时 `assert_eq!` panic。
#[test]
fn regression_cg_dimension_mismatch_returns_none() {
    let mut sys = SparseSystem::new(3);
    sys.add(0, 0, 1.0);
    sys.add(1, 1, 1.0);
    sys.add(2, 2, 1.0);
    let a = sys.finish();
    let b = vec![1.0, 2.0]; // 长度故意不匹配
    assert!(conjugate_gradient(&a, &b, 10, 1e-6).is_none());
}

/// 历史 bug：空矩阵导致除零。
#[test]
fn regression_cg_empty_system_returns_empty_vec() {
    let mut sys = SparseSystem::new(0);
    sys.add(0, 0, 1.0); // 不会写入（dim=0）
    let a = sys.finish();
    let b: Vec<f64> = vec![];
    let x = conjugate_gradient(&a, &b, 10, 1e-6);
    assert!(x.is_some());
    assert!(x.unwrap().is_empty());
}

/// 历史 bug：右端向量全零时 b_norm 接近零导致除零。
#[test]
fn regression_cg_zero_rhs_returns_zero_solution() {
    let mut sys = SparseSystem::new(3);
    sys.add(0, 0, 1.0);
    sys.add(1, 1, 1.0);
    sys.add(2, 2, 1.0);
    let a = sys.finish();
    let b = vec![0.0, 0.0, 0.0];
    let x = conjugate_gradient(&a, &b, 10, 1e-6).unwrap();
    assert!(x.iter().all(|&v| v.abs() < 1e-30));
}

/// 历史 bug：max_iter=0 立即返回 None，但应不 panic。
#[test]
fn regression_cg_max_iter_zero_no_panic() {
    let mut sys = SparseSystem::new(3);
    sys.add(0, 0, 1.0);
    sys.add(1, 1, 1.0);
    sys.add(2, 2, 1.0);
    let a = sys.finish();
    let b = vec![1.0, 2.0, 3.0];
    assert!(conjugate_gradient(&a, &b, 0, 1e-6).is_none());
}

/// 历史 bug：半正定系统（拉普拉斯）未正则化时 p·Ap=0 导致除零。
#[test]
fn regression_cg_semidefinite_no_panic() {
    // 半正定拉普拉斯：[[1,-1,0],[-1,2,-1],[0,-1,1]]，零空间 = [1,1,1]
    let mut sys = SparseSystem::new(3);
    sys.add(0, 1, -1.0);
    sys.add(1, 2, -1.0);
    sys.add(0, 0, 1.0);
    sys.add(1, 1, 2.0);
    sys.add(2, 2, 1.0);
    let a = sys.finish();
    let b = vec![1.0, 0.0, -1.0];
    // 未正则化时可能返回 None（不收敛或 p·Ap=0），但绝不应 panic
    let _ = conjugate_gradient(&a, &b, 100, 1e-6);
}

// ============================================================
// remesh.rs：tangential_smooth / split / collapse / flip 不应 panic
// （通过 isotropic_remesh 端到端验证）
// ============================================================

#[test]
fn regression_remesh_on_empty_mesh_no_panic() {
    use halfedge::remesh::{quick_remesh, remesh_to_length};
    let mut empty = MeshStorage::new();
    let _ = quick_remesh(&mut empty);
    let _ = remesh_to_length(&mut empty, 0.5);
}

#[test]
fn regression_remesh_on_cube_preserves_topology() {
    use halfedge::isotropic_remesh;
    let mut mesh = build_cube(1.0);
    let _ = isotropic_remesh(&mut mesh, Some(0.5), 3, false);
    assert!(
        validate_topology(&mesh).is_empty(),
        "remesh 后拓扑应保持有效"
    );
    assert!(is_closed(&mesh), "remesh 后应仍闭合");
}

// ============================================================
// topology_ops.rs：split/flip/collapse 在边界条件下不应 panic
// ============================================================

#[test]
fn regression_split_edge_on_invalid_he_returns_err_no_panic() {
    let mut mesh = build_icosphere(0);
    // 使用 slotmap 默认 key（未分配过），保证句柄不存在
    let fake = HalfEdgeId::default();
    let res = split_edge(&mut mesh, fake);
    assert!(res.is_err());
}

#[test]
fn regression_flip_edge_on_invalid_he_returns_err_no_panic() {
    let mut mesh = build_icosphere(0);
    let fake = HalfEdgeId::default();
    let res = flip_edge(&mut mesh, fake);
    assert!(res.is_err());
}

#[test]
fn regression_collapse_edge_on_invalid_he_returns_err_no_panic() {
    let mut mesh = build_icosphere(0);
    let fake = HalfEdgeId::default();
    let res = collapse_edge(&mut mesh, fake);
    assert!(res.is_err());
}

// ============================================================
// io.rs：build_mesh_from_vertices_and_faces 索引越界 panic
// ============================================================

/// 历史 bug：面索引超出顶点数时 unwrap panic。现在返回 MeshBuildError 而非 panic。
#[test]
fn regression_build_mesh_index_out_of_range_returns_error() {
    let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let faces = [[0u32, 1, 99]];
    let res = build_mesh_from_vertices_and_faces(&vertices, &faces);
    assert!(res.is_err());
}

// ============================================================
// io.rs：parse_obj 损坏输入不应 panic
// ============================================================

#[test]
fn regression_parse_obj_empty_input_no_panic() {
    // 空输入：返回空网格（Ok），不应 panic
    let mesh = parse_obj("").expect("空 OBJ 应可解析为空网格");
    assert_eq!(mesh.vertex_count(), 0);
    assert_eq!(mesh.face_count(), 0);
}

#[test]
fn regression_parse_obj_only_vertices_no_faces_no_panic() {
    let input = "v 0 0 0\nv 1 0 0\nv 0 1 0\n";
    let mesh = parse_obj(input).expect("仅有顶点的 OBJ 应可解析");
    assert_eq!(mesh.vertex_count(), 3);
    assert_eq!(mesh.face_count(), 0);
}

#[test]
fn regression_parse_obj_malformed_face_skipped_no_panic() {
    // 损坏的面行应被跳过或返回错误，但不应 panic
    let input = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2\nf 1 2 3\n";
    let _ = parse_obj(input);
}

// ============================================================
// 综合回归：cube 体积/表面积仍正确
// ============================================================

#[test]
fn regression_cube_volume_and_area_correct() {
    let cube = build_cube(2.0); // 边长 2
    let v = mesh_volume(&cube);
    let a = surface_area(&cube);
    assert!(
        (v.abs() - 8.0).abs() < 0.01,
        "边长 2 立方体体积应 ≈ 8.0，实际 {}",
        v
    );
    assert!(
        (a - 24.0).abs() < 0.01,
        "边长 2 立方体表面积应 ≈ 24.0，实际 {}",
        a
    );
}
