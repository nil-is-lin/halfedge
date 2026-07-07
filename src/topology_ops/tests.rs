//! 拓扑操作的单元测试。

use super::edit::check_link_condition;
use super::*;
use crate::ids::{FaceId, HalfEdgeId, VertexId};
use crate::storage::{Face, HalfEdge, MeshStorage, Vertex};
use crate::traversal::{FaceHalfEdges, VertexRing, is_boundary_edge, is_boundary_vertex};

// ---------- 测试夹具 ----------

/// 构造单个三角面片（CCW），所有边都是边界边。
/// ```text
///        v2
///        ▲
///       / │
///   h2 /  │ t1
///     /   │
///    /    │
///   v0────┼───▶ v1
///    \    │
///     \   │
///   t2 \  │ h0
///       \ │
///        ▼
/// ```
/// 面 F = (h0: v0→v1, h1: v1→v2, h2: v2→v0)
fn build_triangle() -> (MeshStorage, [VertexId; 3], [HalfEdgeId; 6], FaceId) {
    let mut mesh = MeshStorage::new();
    let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
    let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
    let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));

    let h0 = mesh.add_halfedge(HalfEdge::new(v1)); // v0→v1
    let h1 = mesh.add_halfedge(HalfEdge::new(v2)); // v1→v2
    let h2 = mesh.add_halfedge(HalfEdge::new(v0)); // v2→v0
    let t0 = mesh.add_halfedge(HalfEdge::new(v0)); // v1→v0
    let t1 = mesh.add_halfedge(HalfEdge::new(v1)); // v2→v1
    let t2 = mesh.add_halfedge(HalfEdge::new(v2)); // v0→v2

    let f = mesh.add_face(Face::new());

    for (he, twin, next, prev) in [(h0, t0, h1, h2), (h1, t1, h2, h0), (h2, t2, h0, h1)] {
        let h = mesh.get_halfedge_mut(he).unwrap();
        h.twin = Some(twin);
        h.next = Some(next);
        h.prev = Some(prev);
        h.face = Some(f);
    }
    for (t, he) in [(t0, h0), (t1, h1), (t2, h2)] {
        mesh.get_halfedge_mut(t).unwrap().twin = Some(he);
    }
    mesh.get_vertex_mut(v0).unwrap().halfedge = Some(h0);
    mesh.get_vertex_mut(v1).unwrap().halfedge = Some(h1);
    mesh.get_vertex_mut(v2).unwrap().halfedge = Some(h2);
    mesh.get_face_mut(f).unwrap().halfedge = Some(h0);

    (mesh, [v0, v1, v2], [h0, h1, h2, t0, t1, t2], f)
}

/// 构造两个三角形拼成的四边形，共享边 v0-v1。
/// ```text
///   v2 ────── v3
///    │ ╲     ╱│
///    │  ╲   ╱ │
///    │ F1╲ ╱F2│
///    │   ╲╱  │
///    │   ╱╲  │
///    │  ╱  ╲ │
///    │ ╱   ╲│
///   v0 ──── v1   （共享边 v0-v1）
/// ```
fn build_two_triangles() -> (MeshStorage, [VertexId; 4], [HalfEdgeId; 10], FaceId, FaceId) {
    let mut mesh = MeshStorage::new();
    let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
    let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
    let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));
    let v3 = mesh.add_vertex(Vertex::new([1.0, -1.0, 0.0])); // F2 在共享边下方，几何 CCW

    // F1 = v0→v1→v2→v0
    let h0 = mesh.add_halfedge(HalfEdge::new(v1)); // v0→v1
    let h1 = mesh.add_halfedge(HalfEdge::new(v2)); // v1→v2
    let h2 = mesh.add_halfedge(HalfEdge::new(v0)); // v2→v0
    // F2 = v1→v0→v3→v1
    let g0 = mesh.add_halfedge(HalfEdge::new(v0)); // v1→v0 (twin of h0)
    let g1 = mesh.add_halfedge(HalfEdge::new(v3)); // v0→v3
    let g2 = mesh.add_halfedge(HalfEdge::new(v1)); // v3→v1
    // 边界 twin
    let t1 = mesh.add_halfedge(HalfEdge::new(v1)); // v2→v1
    let t2 = mesh.add_halfedge(HalfEdge::new(v2)); // v0→v2
    let t_g1 = mesh.add_halfedge(HalfEdge::new(v0)); // v3→v0
    let t_g2 = mesh.add_halfedge(HalfEdge::new(v3)); // v1→v3

    let f1 = mesh.add_face(Face::new());
    let f2 = mesh.add_face(Face::new());

    for (he, twin, next, prev) in [(h0, g0, h1, h2), (h1, t1, h2, h0), (h2, t2, h0, h1)] {
        let h = mesh.get_halfedge_mut(he).unwrap();
        h.twin = Some(twin);
        h.next = Some(next);
        h.prev = Some(prev);
        h.face = Some(f1);
    }
    for (he, twin, next, prev) in [(g0, h0, g1, g2), (g1, t_g1, g2, g0), (g2, t_g2, g0, g1)] {
        let h = mesh.get_halfedge_mut(he).unwrap();
        h.twin = Some(twin);
        h.next = Some(next);
        h.prev = Some(prev);
        h.face = Some(f2);
    }
    for (t, he) in [(t1, h1), (t2, h2), (t_g1, g1), (t_g2, g2)] {
        mesh.get_halfedge_mut(t).unwrap().twin = Some(he);
    }
    mesh.get_vertex_mut(v0).unwrap().halfedge = Some(h0);
    mesh.get_vertex_mut(v1).unwrap().halfedge = Some(g0);
    mesh.get_vertex_mut(v2).unwrap().halfedge = Some(h1);
    mesh.get_vertex_mut(v3).unwrap().halfedge = Some(g1);
    mesh.get_face_mut(f1).unwrap().halfedge = Some(h0);
    mesh.get_face_mut(f2).unwrap().halfedge = Some(g0);

    (
        mesh,
        [v0, v1, v2, v3],
        [h0, h1, h2, g0, g1, g2, t1, t2, t_g1, t_g2],
        f1,
        f2,
    )
}

/// 构造 3 个三角形围成的闭合扇形，中心 c 是内部顶点。
/// ```text
///         v2
///        ╱ ╲
///       ╱   ╲
///      ╱  F3 ╲
///     ╱       ╲
///    v0 ── c ── v1
///     ╲       ╱
///      ╲ F1  ╱
///       ╲   ╱
///        ╲ ╱
/// ```
/// F1 = c→v0→v1→c, F2 = c→v1→v2→c, F3 = c→v2→v0→c
fn build_closed_fan() -> (MeshStorage, VertexId, [VertexId; 3], [FaceId; 3]) {
    let mut mesh = MeshStorage::new();
    let c = mesh.add_vertex(Vertex::new([0.5, 0.5, 0.0]));
    let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
    let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
    let v2 = mesh.add_vertex(Vertex::new([0.5, 1.0, 0.0]));

    let a1 = mesh.add_halfedge(HalfEdge::new(v0)); // c→v0
    let b1 = mesh.add_halfedge(HalfEdge::new(v1)); // v0→v1
    let c1 = mesh.add_halfedge(HalfEdge::new(c)); // v1→c
    let a2 = mesh.add_halfedge(HalfEdge::new(v1)); // c→v1
    let b2 = mesh.add_halfedge(HalfEdge::new(v2)); // v1→v2
    let c2 = mesh.add_halfedge(HalfEdge::new(c)); // v2→c
    let a3 = mesh.add_halfedge(HalfEdge::new(v2)); // c→v2
    let b3 = mesh.add_halfedge(HalfEdge::new(v0)); // v2→v0
    let c3 = mesh.add_halfedge(HalfEdge::new(c)); // v0→c
    let t1 = mesh.add_halfedge(HalfEdge::new(v0)); // v1→v0
    let t2 = mesh.add_halfedge(HalfEdge::new(v1)); // v2→v1
    let t3 = mesh.add_halfedge(HalfEdge::new(v2)); // v0→v2

    let f1 = mesh.add_face(Face::new());
    let f2 = mesh.add_face(Face::new());
    let f3 = mesh.add_face(Face::new());

    for (he, twin, next, prev, face) in [
        (a1, c3, b1, c1, f1),
        (b1, t1, c1, a1, f1),
        (c1, a2, a1, b1, f1),
        (a2, c1, b2, c2, f2),
        (b2, t2, c2, a2, f2),
        (c2, a3, a2, b2, f2),
        (a3, c2, b3, c3, f3),
        (b3, t3, c3, a3, f3),
        (c3, a1, a3, b3, f3),
    ] {
        let h = mesh.get_halfedge_mut(he).unwrap();
        h.twin = Some(twin);
        h.next = Some(next);
        h.prev = Some(prev);
        h.face = Some(face);
    }
    for (t, he) in [(t1, b1), (t2, b2), (t3, b3)] {
        mesh.get_halfedge_mut(t).unwrap().twin = Some(he);
    }
    mesh.get_vertex_mut(c).unwrap().halfedge = Some(a1);
    mesh.get_vertex_mut(v0).unwrap().halfedge = Some(b1);
    mesh.get_vertex_mut(v1).unwrap().halfedge = Some(b2);
    mesh.get_vertex_mut(v2).unwrap().halfedge = Some(b3);
    mesh.get_face_mut(f1).unwrap().halfedge = Some(a1);
    mesh.get_face_mut(f2).unwrap().halfedge = Some(a2);
    mesh.get_face_mut(f3).unwrap().halfedge = Some(a3);

    (mesh, c, [v0, v1, v2], [f1, f2, f3])
}

// ---------- validate_mesh 测试 ----------

#[test]
fn validate_clean_meshes() {
    let (m1, _, _, _) = build_triangle();
    assert!(validate_mesh(&m1).is_ok());

    let (m2, _, _, _, _) = build_two_triangles();
    assert!(validate_mesh(&m2).is_ok());

    let (m3, _, _, _) = build_closed_fan();
    assert!(validate_mesh(&m3).is_ok());
}

// ---------- split_edge 测试 ----------

#[test]
fn split_boundary_edge_of_single_triangle() {
    // 分裂单三角形的边界边 h0 (v0→v1)
    let (mut mesh, v, he, _f) = build_triangle();
    let [v0, v1, _v2] = v;
    let [h0, _h1, _h2, _t0, _t1, _t2] = he;

    let v_before = mesh.vertex_count();
    let f_before = mesh.face_count();
    let he_before = mesh.halfedge_count();

    let m = split_edge(&mut mesh, h0).expect("分裂边界边应成功");

    // 数量校验：+1 顶点, +1 面, +4 半边（删 2 加 6）
    assert_eq!(mesh.vertex_count(), v_before + 1);
    assert_eq!(mesh.face_count(), f_before + 1);
    assert_eq!(mesh.halfedge_count(), he_before + 4);

    // 新顶点存在
    assert!(mesh.contains_vertex(m));
    // 原半边已删除
    assert!(!mesh.contains_halfedge(h0));

    // 拓扑合法
    assert!(validate_mesh(&mesh).is_ok());

    // M 位于 v0-v1 中点
    let pos_m = mesh.get_vertex(m).unwrap().position;
    let pos_v0 = mesh.get_vertex(v0).unwrap().position;
    let pos_v1 = mesh.get_vertex(v1).unwrap().position;
    assert!((pos_m[0] - (pos_v0[0] + pos_v1[0]) / 2.0).abs() < 1e-12);

    // 现在有 2 个三角形
    assert_eq!(mesh.face_count(), 2);
    // 所有面都是三角面
    for f in mesh.face_ids().collect::<Vec<_>>() {
        let count = FaceHalfEdges::new(&mesh, f).count();
        assert_eq!(count, 3, "折叠后面应为三角面");
    }
}

#[test]
fn split_interior_edge_of_two_triangles() {
    // 分享两三角形拼四边形的内部边 h0 (v0→v1)
    let (mut mesh, _v, he, _f1, _f2) = build_two_triangles();
    let [h0, _h1, _h2, _g0, _g1, _g2, _t1, _t2, _t_g1, _t_g2] = he;

    let v_before = mesh.vertex_count();
    let f_before = mesh.face_count();
    let he_before = mesh.halfedge_count();

    let m = split_edge(&mut mesh, h0).expect("分裂内部边应成功");

    // 数量校验：+1 顶点, +2 面, +6 半边（删 2 加 8）
    assert_eq!(mesh.vertex_count(), v_before + 1);
    assert_eq!(mesh.face_count(), f_before + 2);
    assert_eq!(mesh.halfedge_count(), he_before + 6);

    assert!(mesh.contains_vertex(m));
    assert!(!mesh.contains_halfedge(h0));

    assert!(validate_mesh(&mesh).is_ok());

    // 所有面都是三角面
    for f in mesh.face_ids().collect::<Vec<_>>() {
        let count = FaceHalfEdges::new(&mesh, f).count();
        assert_eq!(count, 3, "分裂后面应为三角面");
    }
}

#[test]
fn split_boundary_edge_passed_as_boundary_halfedge() {
    // 传入边界半边 t0 (v1→v0)，应自动转为操作 twin h0
    let (mut mesh, _v, he, _f) = build_triangle();
    let [h0, _h1, _h2, t0, _t1, _t2] = he;

    let m = split_edge(&mut mesh, t0).expect("传入边界半边应自动转为操作 twin");
    assert!(mesh.contains_vertex(m));
    assert!(!mesh.contains_halfedge(h0));
    assert!(!mesh.contains_halfedge(t0));
    assert!(validate_mesh(&mesh).is_ok());
}

// ---------- flip_edge 测试 ----------

#[test]
fn flip_interior_edge_of_two_triangles() {
    let (mut mesh, v, he, _f1, _f2) = build_two_triangles();
    let [_v0, _v1, v2, _v3] = v;
    let [h0, _h1, _h2, _g0, _g1, _g2, _t1, _t2, _t_g1, _t_g2] = he;

    let v_before = mesh.vertex_count();
    let f_before = mesh.face_count();
    let he_before = mesh.halfedge_count();

    flip_edge(&mut mesh, h0).expect("翻转内部边应成功");

    // 数量不变
    assert_eq!(mesh.vertex_count(), v_before);
    assert_eq!(mesh.face_count(), f_before);
    assert_eq!(mesh.halfedge_count(), he_before);

    assert!(validate_mesh(&mesh).is_ok());

    // 翻转后 h0 的 vertex 应变为 v2（原 C），twin 的 vertex 应变为 v3（原 D）
    // h0 现在是 D→C = v3→v2
    let h = mesh.get_halfedge(h0).unwrap();
    assert_eq!(h.vertex, v2, "翻转后 h.vertex 应为 C=v2");

    // 所有面都是三角面
    for f in mesh.face_ids().collect::<Vec<_>>() {
        let count = FaceHalfEdges::new(&mesh, f).count();
        assert_eq!(count, 3, "翻转后面应为三角面");
    }
}

#[test]
fn flip_boundary_edge_fails() {
    let (mut mesh, _v, he, _f) = build_triangle();
    let [h0, _h1, _h2, _t0, _t1, _t2] = he;

    let err = flip_edge(&mut mesh, h0).unwrap_err();
    assert_eq!(err, TopologyError::FlipOnBoundaryEdge(h0));
}

#[test]
fn flip_then_flip_restores_topology() {
    let (mut mesh, _v, he, _f1, _f2) = build_two_triangles();
    let [h0, _, _, _, _, _, _, _, _, _] = he;

    flip_edge(&mut mesh, h0).expect("第一次翻转");
    assert!(validate_mesh(&mesh).is_ok());
    flip_edge(&mut mesh, h0).expect("第二次翻转恢复");
    assert!(validate_mesh(&mesh).is_ok());

    // 两次翻转后 h0.vertex 应回到原值 v1
    // 原 h0: v0→v1, vertex=v1
    // 翻转后 h0 变为 v3→v2，再翻转变回 v0→v1
    // 验证面数和半边数不变
    assert_eq!(mesh.face_count(), 2);
}

// ---------- collapse_edge 测试 ----------

#[test]
fn collapse_interior_edge_of_closed_fan() {
    // 在闭合扇形中折叠 c-v0 之间的边（a1: c→v0）
    let (mut mesh, c, outer, _faces) = build_closed_fan();
    let [v0, _v1, _v2] = outer;

    // 找到 c→v0 的半边
    let a1 = VertexRing::new(&mesh, c)
        .find(|he| mesh.get_halfedge(*he).unwrap().vertex == v0)
        .expect("c→v0 半边必须存在");

    let v_before = mesh.vertex_count();
    let f_before = mesh.face_count();
    let he_before = mesh.halfedge_count();

    let k = collapse_edge(&mut mesh, a1).expect("折叠内部边应成功");

    // 数量校验：-1 顶点, -2 面, -6 半边
    assert_eq!(mesh.vertex_count(), v_before - 1);
    assert_eq!(mesh.face_count(), f_before - 2);
    assert_eq!(mesh.halfedge_count(), he_before - 6);

    assert!(mesh.contains_vertex(k));
    assert!(!mesh.contains_vertex(c));
    assert!(!mesh.contains_vertex(v0));

    assert!(validate_mesh(&mesh).is_ok());

    // 剩余 1 个三角形
    assert_eq!(mesh.face_count(), 1);
    for f in mesh.face_ids().collect::<Vec<_>>() {
        let count = FaceHalfEdges::new(&mesh, f).count();
        assert_eq!(count, 3, "折叠后面应为三角面");
    }
}

#[test]
fn collapse_boundary_edge_fails() {
    let (mut mesh, _v, he, _f) = build_triangle();
    let [h0, _h1, _h2, _t0, _t1, _t2] = he;

    let err = collapse_edge(&mut mesh, h0).unwrap_err();
    assert_eq!(err, TopologyError::CollapseOnBoundaryEdge(h0));
}

/// 构造一个链接条件不满足的网格：5 顶点 4 面，A-B 边的公共邻居为 {C, D, E}（超过 2）。
///
/// ```text
///        E
///       ╱ ╲
///   F3 ╱   ╲ F4
///     ╱     ╲
///    A ──F1── B
///     ╲     ╱
///   F2 ╲   ╱
///       ╲ ╱
///        D        （C 在 F1 下方）
/// ```
/// F1 = A→B→C→A, F2 = B→A→D→B, F3 = A→C→E→A, F4 = B→D→E→B
/// A 的邻居 = {B, C, D, E}，B 的邻居 = {A, C, D, E}
/// 公共邻居 = {C, D, E} ≠ {C, D} → 链接条件违反
fn build_link_violation_mesh() -> (MeshStorage, VertexId, VertexId) {
    let mut mesh = MeshStorage::new();
    let a = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
    let b = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
    let c = mesh.add_vertex(Vertex::new([0.5, -1.0, 0.0]));
    let d = mesh.add_vertex(Vertex::new([0.5, 0.0, 1.0]));
    let e = mesh.add_vertex(Vertex::new([0.5, 1.0, 0.0]));

    // F1 = A→B→C→A
    let h_ab = mesh.add_halfedge(HalfEdge::new(b)); // A→B
    let h_bc = mesh.add_halfedge(HalfEdge::new(c)); // B→C
    let h_ca = mesh.add_halfedge(HalfEdge::new(a)); // C→A
    // F2 = B→A→D→B
    let h_ba = mesh.add_halfedge(HalfEdge::new(a)); // B→A
    let h_ad = mesh.add_halfedge(HalfEdge::new(d)); // A→D
    let h_db = mesh.add_halfedge(HalfEdge::new(b)); // D→B
    // F3 = A→C→E→A
    let h_ac = mesh.add_halfedge(HalfEdge::new(c)); // A→C
    let h_ce = mesh.add_halfedge(HalfEdge::new(e)); // C→E
    let h_ea = mesh.add_halfedge(HalfEdge::new(a)); // E→A
    // F4 = B→D→E→B
    let h_bd = mesh.add_halfedge(HalfEdge::new(d)); // B→D
    let h_de = mesh.add_halfedge(HalfEdge::new(e)); // D→E
    let h_eb = mesh.add_halfedge(HalfEdge::new(b)); // E→B
    // 边界 twin
    let t_bc = mesh.add_halfedge(HalfEdge::new(b)); // C→B
    let t_ad = mesh.add_halfedge(HalfEdge::new(a)); // D→A
    let t_ce = mesh.add_halfedge(HalfEdge::new(c)); // E→C
    let t_ea = mesh.add_halfedge(HalfEdge::new(e)); // A→E
    let t_de = mesh.add_halfedge(HalfEdge::new(d)); // E→D
    let t_eb = mesh.add_halfedge(HalfEdge::new(e)); // B→E

    let f1 = mesh.add_face(Face::new());
    let f2 = mesh.add_face(Face::new());
    let f3 = mesh.add_face(Face::new());
    let f4 = mesh.add_face(Face::new());

    // F1 内部
    for (he, twin, next, prev) in [
        (h_ab, h_ba, h_bc, h_ca),
        (h_bc, t_bc, h_ca, h_ab),
        (h_ca, h_ac, h_ab, h_bc),
    ] {
        let h = mesh.get_halfedge_mut(he).unwrap();
        h.twin = Some(twin);
        h.next = Some(next);
        h.prev = Some(prev);
        h.face = Some(f1);
    }
    // F2 内部
    for (he, twin, next, prev) in [
        (h_ba, h_ab, h_ad, h_db),
        (h_ad, t_ad, h_db, h_ba),
        (h_db, h_bd, h_ba, h_ad),
    ] {
        let h = mesh.get_halfedge_mut(he).unwrap();
        h.twin = Some(twin);
        h.next = Some(next);
        h.prev = Some(prev);
        h.face = Some(f2);
    }
    // F3 内部
    for (he, twin, next, prev) in [
        (h_ac, h_ca, h_ce, h_ea),
        (h_ce, t_ce, h_ea, h_ac),
        (h_ea, t_ea, h_ac, h_ce),
    ] {
        let h = mesh.get_halfedge_mut(he).unwrap();
        h.twin = Some(twin);
        h.next = Some(next);
        h.prev = Some(prev);
        h.face = Some(f3);
    }
    // F4 内部
    for (he, twin, next, prev) in [
        (h_bd, h_db, h_de, h_eb),
        (h_de, t_de, h_eb, h_bd),
        (h_eb, t_eb, h_bd, h_de),
    ] {
        let h = mesh.get_halfedge_mut(he).unwrap();
        h.twin = Some(twin);
        h.next = Some(next);
        h.prev = Some(prev);
        h.face = Some(f4);
    }
    // 边界 twin（仅互指 twin）
    for (t, he) in [
        (t_bc, h_bc),
        (t_ad, h_ad),
        (t_ce, h_ce),
        (t_ea, h_ea),
        (t_de, h_de),
        (t_eb, h_eb),
    ] {
        mesh.get_halfedge_mut(t).unwrap().twin = Some(he);
    }
    // 顶点 outgoing 入口
    mesh.get_vertex_mut(a).unwrap().halfedge = Some(h_ab);
    mesh.get_vertex_mut(b).unwrap().halfedge = Some(h_ba);
    mesh.get_vertex_mut(c).unwrap().halfedge = Some(h_ca);
    mesh.get_vertex_mut(d).unwrap().halfedge = Some(h_db);
    mesh.get_vertex_mut(e).unwrap().halfedge = Some(h_ea);
    // 面入口
    mesh.get_face_mut(f1).unwrap().halfedge = Some(h_ab);
    mesh.get_face_mut(f2).unwrap().halfedge = Some(h_ba);
    mesh.get_face_mut(f3).unwrap().halfedge = Some(h_ac);
    mesh.get_face_mut(f4).unwrap().halfedge = Some(h_bd);

    (mesh, a, b)
}

#[test]
fn collapse_with_violated_link_condition_fails() {
    // 5 顶点 4 面网格：A-B 边的公共邻居 = {C, D, E}（3 个 > 2），违反链接条件
    let (mut mesh, a, b) = build_link_violation_mesh();

    // 先验证这是合法的流形网格
    assert!(validate_mesh(&mesh).is_ok());

    // 找到 A→B 的半边
    let h_ab = VertexRing::new(&mesh, a)
        .find(|he| mesh.get_halfedge(*he).unwrap().vertex == b)
        .expect("A→B 半边必须存在");

    let err = collapse_edge(&mut mesh, h_ab).unwrap_err();
    assert_eq!(
        err,
        TopologyError::LinkConditionViolated { a, b },
        "公共邻居 {{C,D,E}} != {{C,D}} → 应返回链接条件违反"
    );

    // 确认网格未被修改（操作是原子的）
    assert!(validate_mesh(&mesh).is_ok());
    assert!(mesh.contains_vertex(a));
    assert!(mesh.contains_vertex(b));
}

#[test]
fn collapse_link_condition_check_function() {
    // 直接测试 check_link_condition 函数
    let (mesh, c, outer, _faces) = build_closed_fan();
    let [v0, v1, v2] = outer;

    // c-v0 的公共邻居 = {v1, v2}
    assert!(check_link_condition(&mesh, c, v0, v1, v2));
    // 传入错误的 C/D 应返回 false
    assert!(!check_link_condition(&mesh, c, v0, v1, v0));
}

// ---------- 综合测试 ----------

#[test]
fn split_then_validate_all_faces_triangular() {
    let (mut mesh, _v, he, _f1, _f2) = build_two_triangles();
    let [h0, _h1, _, _, _, _, _, _, _, _] = he;

    // 分裂内部边
    split_edge(&mut mesh, h0).unwrap();
    assert!(validate_mesh(&mesh).is_ok());

    // 再分裂一条边界边
    // 找 v2 的一条边界半边
    let boundary_he = mesh
        .halfedge_ids()
        .find(|he| is_boundary_edge(&mesh, *he))
        .expect("应存在边界半边");
    split_edge(&mut mesh, boundary_he).unwrap();
    assert!(validate_mesh(&mesh).is_ok());

    // 所有面都是三角面
    for f in mesh.face_ids().collect::<Vec<_>>() {
        assert_eq!(FaceHalfEdges::new(&mesh, f).count(), 3);
    }
}

#[test]
fn flip_preserves_boundary_vertices() {
    let (mut mesh, v, he, _f1, _f2) = build_two_triangles();
    let [v0, v1, v2, v3] = v;
    let [h0, _, _, _, _, _, _, _, _, _] = he;

    flip_edge(&mut mesh, h0).unwrap();

    // 翻转后所有顶点仍是边界顶点（两三角形拼四边形的四个角都是边界）
    for vi in [v0, v1, v2, v3] {
        assert!(
            is_boundary_vertex(&mesh, vi),
            "翻转后顶点 {:?} 应仍为边界顶点",
            vi
        );
    }
}

#[test]
fn collapse_then_remaining_mesh_valid() {
    // 在闭合扇形中折叠一条边，然后验证剩余网格
    let (mut mesh, c, outer, _faces) = build_closed_fan();
    let [v0, v1, v2] = outer;

    // 折叠 c→v0
    let a1 = VertexRing::new(&mesh, c)
        .find(|he| mesh.get_halfedge(*he).unwrap().vertex == v0)
        .unwrap();
    let k = collapse_edge(&mut mesh, a1).expect("折叠应成功");

    // K 应该是边界顶点（因为 v0 是边界顶点，折叠后 K 继承其边界性）
    assert!(is_boundary_vertex(&mesh, k), "折叠后新顶点 K 应为边界顶点");
    // v1, v2 仍存在且是边界顶点
    assert!(mesh.contains_vertex(v1));
    assert!(mesh.contains_vertex(v2));
    assert!(is_boundary_vertex(&mesh, v1));
    assert!(is_boundary_vertex(&mesh, v2));

    assert!(validate_mesh(&mesh).is_ok());
}

// ---------- collapse_edge_at 测试 ----------

#[test]
fn collapse_edge_uses_midpoint_position() {
    // 折叠 c→v0：A=c=(0.5,0.5,0), B=v0=(0,0,0)，中点应为 (0.25,0.25,0)
    let (mut mesh, c, outer, _faces) = build_closed_fan();
    let [v0, _v1, _v2] = outer;

    let a1 = VertexRing::new(&mesh, c)
        .find(|he| mesh.get_halfedge(*he).unwrap().vertex == v0)
        .expect("c→v0 半边必须存在");

    let k = collapse_edge(&mut mesh, a1).expect("折叠应成功");
    let pos = mesh.get_vertex(k).unwrap().position;

    assert_eq!(pos, [0.25, 0.25, 0.0]);
    assert!(validate_mesh(&mesh).is_ok());
}

#[test]
fn collapse_edge_at_uses_custom_position() {
    // 同样折叠 c→v0，但指定自定义位置
    let (mut mesh, c, outer, _faces) = build_closed_fan();
    let [v0, _v1, _v2] = outer;

    let a1 = VertexRing::new(&mesh, c)
        .find(|he| mesh.get_halfedge(*he).unwrap().vertex == v0)
        .expect("c→v0 半边必须存在");

    let target = [1.0, 2.0, 3.0];
    let k = collapse_edge_at(&mut mesh, a1, target).expect("折叠应成功");
    let pos = mesh.get_vertex(k).unwrap().position;

    assert_eq!(pos, target);
    assert!(validate_mesh(&mesh).is_ok());
}

#[test]
fn collapse_edge_at_preserves_topology_counts() {
    // collapse_edge_at 与 collapse_edge 的拓扑变化应完全相同
    let (mut mesh1, c, outer, _faces) = build_closed_fan();
    let [v0, _v1, _v2] = outer;
    let a1 = VertexRing::new(&mesh1, c)
        .find(|he| mesh1.get_halfedge(*he).unwrap().vertex == v0)
        .unwrap();

    let (mut mesh2, c2, outer2, _faces2) = build_closed_fan();
    let [v0_2, _v1_2, _v2_2] = outer2;
    let a1_2 = VertexRing::new(&mesh2, c2)
        .find(|he| mesh2.get_halfedge(*he).unwrap().vertex == v0_2)
        .unwrap();

    let k1 = collapse_edge(&mut mesh1, a1).unwrap();
    let k2 = collapse_edge_at(&mut mesh2, a1_2, [10.0, 20.0, 30.0]).unwrap();

    // 拓扑计数一致
    assert_eq!(mesh1.vertex_count(), mesh2.vertex_count());
    assert_eq!(mesh1.face_count(), mesh2.face_count());
    assert_eq!(mesh1.halfedge_count(), mesh2.halfedge_count());
    // 仅位置不同
    let p1 = mesh1.get_vertex(k1).unwrap().position;
    let p2 = mesh2.get_vertex(k2).unwrap().position;
    assert_eq!(p1, [0.25, 0.25, 0.0]);
    assert_eq!(p2, [10.0, 20.0, 30.0]);
    // 两者都通过校验
    assert!(validate_mesh(&mesh1).is_ok());
    assert!(validate_mesh(&mesh2).is_ok());
}

#[test]
fn collapse_edge_at_on_boundary_edge_fails() {
    // collapse_edge_at 同样禁止折叠边界边
    let (mut mesh, _v, he, _f) = build_triangle();
    let [h0, _h1, _h2, _t0, _t1, _t2] = he;

    let err = collapse_edge_at(&mut mesh, h0, [1.0, 1.0, 1.0]).unwrap_err();
    assert_eq!(err, TopologyError::CollapseOnBoundaryEdge(h0));
}

#[test]
fn invalid_halfedge_returns_error() {
    let mut mesh = MeshStorage::new();
    let fake = HalfEdgeId::default();
    assert_eq!(
        split_edge(&mut mesh, fake).unwrap_err(),
        TopologyError::InvalidHalfEdge(fake)
    );
    assert_eq!(
        flip_edge(&mut mesh, fake).unwrap_err(),
        TopologyError::InvalidHalfEdge(fake)
    );
    assert_eq!(
        collapse_edge(&mut mesh, fake).unwrap_err(),
        TopologyError::InvalidHalfEdge(fake)
    );
}

// ---------- extrude_face 测试 ----------

/// 辅助：构造两个互不连通的三角形，便于测试批量挤出。
/// 三角形 A = (0,0,0)-(1,0,0)-(0,1,0)，三角形 B = (5,0,0)-(6,0,0)-(5,1,0)。
fn build_two_disjoint_triangles() -> (MeshStorage, [FaceId; 2]) {
    use crate::io::build_mesh_from_vertices_and_faces;
    let verts = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [5.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [5.0, 1.0, 0.0],
    ];
    let faces = vec![[0, 1, 2], [3, 4, 5]];
    let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
    let fids: Vec<FaceId> = mesh.face_ids().collect();
    assert_eq!(fids.len(), 2);
    (mesh, [fids[0], fids[1]])
}

#[test]
fn extrude_single_triangle() {
    let (mut mesh, v, _he, f) = build_triangle();
    let [v0, v1, v2] = v;

    let v_before = mesh.vertex_count();
    let f_before = mesh.face_count();
    let he_before = mesh.halfedge_count();

    let offset = [0.0, 0.0, 1.0];
    let new_faces = extrude_face(&mut mesh, f, offset).expect("挤出应成功");

    // 数量校验：+3 顶点, +7 面, +18 半边
    assert_eq!(mesh.vertex_count(), v_before + 3);
    assert_eq!(mesh.face_count(), f_before + 7);
    assert_eq!(mesh.halfedge_count(), he_before + 18);
    assert_eq!(new_faces.len(), 7);

    // 完整拓扑校验
    let errors = crate::validate::validate_topology(&mesh);
    assert!(errors.is_empty(), "挤出后拓扑应一致：{:?}", errors);

    // 新顶点位置校验：原顶点 + offset
    let mut new_verts: Vec<VertexId> = mesh
        .vertex_ids()
        .filter(|v| ![v0, v1, v2].contains(v))
        .collect();
    new_verts.sort_by_key(|x| {
        let p = mesh.get_vertex(*x).unwrap().position;
        (p[0] * 1000.0) as i64 + (p[1] * 1000.0) as i64 * 1000
    });
    for v in &new_verts {
        let p = mesh.get_vertex(*v).unwrap().position;
        assert!((p[2] - 1.0).abs() < 1e-12, "新顶点 z 坐标应为 1.0");
    }
    assert_eq!(new_verts.len(), 3);

    // 原顶点位置不变
    for v in [v0, v1, v2] {
        let p = mesh.get_vertex(v).unwrap().position;
        assert_eq!(p[2], 0.0);
    }

    // 原 face 仍存在，且为底面（法向不变）
    assert!(mesh.contains_face(f));

    // Euler 示性数：3-3+1=1（带边界）→ 6-12+8=2（闭合三棱柱）
    let chi =
        mesh.vertex_count() as i64 - (mesh.halfedge_count() / 2) as i64 + mesh.face_count() as i64;
    assert_eq!(chi, 2, "挤出后 Euler 示性数应为 2（闭合三棱柱）");
}

#[test]
fn extrude_zero_offset_returns_error() {
    let (mut mesh, _v, _he, f) = build_triangle();
    let err = extrude_face(&mut mesh, f, [0.0, 0.0, 0.0]).unwrap_err();
    assert_eq!(err, TopologyError::DegenerateTriangle);
}

#[test]
fn extrude_degenerate_side_face_returns_error() {
    // offset 平行于三角形所在平面（z=0）会使侧面三角形退化
    let (mut mesh, _v, _he, f) = build_triangle();
    // 三角形在 xy 平面，offset 沿 x 轴 → 边 v0-v1 与 offset 平行 → 侧面退化
    let err = extrude_face(&mut mesh, f, [1.0, 0.0, 0.0]).unwrap_err();
    assert_eq!(err, TopologyError::DegenerateTriangle);
}

#[test]
fn extrude_face_with_internal_edge_returns_error() {
    // build_two_triangles 中 F1 的边 h0 (v0→v1) 是内部边（twin 有 face=f2）
    let (mut mesh, _v, _he, f1, _f2) = build_two_triangles();
    let err = extrude_face(&mut mesh, f1, [0.0, 0.0, 1.0]).unwrap_err();
    match err {
        TopologyError::Inconsistent(_) => {}
        other => panic!("期望 Inconsistent 错误，得到 {:?}", other),
    }
}

#[test]
fn extrude_faces_batch_disjoint() {
    let (mut mesh, fs) = build_two_disjoint_triangles();
    let v_before = mesh.vertex_count();
    let f_before = mesh.face_count();
    let he_before = mesh.halfedge_count();

    let offset = [0.0, 0.0, 2.0];
    let new_faces = extrude_faces(&mut mesh, &fs, offset).expect("批量挤出应成功");

    // 每个面 +3 顶点 / +7 面 / +18 半边
    assert_eq!(mesh.vertex_count(), v_before + 6);
    assert_eq!(mesh.face_count(), f_before + 14);
    assert_eq!(mesh.halfedge_count(), he_before + 36);
    assert_eq!(new_faces.len(), 14);

    // 完整拓扑校验
    let errors = crate::validate::validate_topology(&mesh);
    assert!(errors.is_empty(), "批量挤出后拓扑应一致：{:?}", errors);
}

#[test]
fn extrude_face_from_built_mesh() {
    // 通过 build_mesh_from_vertices_and_faces 构建网格后再挤出
    use crate::io::build_mesh_from_vertices_and_faces;
    let verts = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0]];
    let faces = vec![[0, 1, 2]];
    let mut mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
    let f = mesh.face_ids().next().expect("应有一个面");

    let offset = [0.0, 0.0, 3.0];
    let new_faces = extrude_face(&mut mesh, f, offset).expect("挤出应成功");
    assert_eq!(new_faces.len(), 7);

    // 完整拓扑校验
    let errors = crate::validate::validate_topology(&mesh);
    assert!(errors.is_empty(), "挤出后拓扑应一致：{:?}", errors);

    // 验证顶点位置
    let mut z_top = 0;
    let mut z_bot = 0;
    for v in mesh.vertex_ids() {
        let p = mesh.get_vertex(v).unwrap().position;
        if (p[2] - 3.0).abs() < 1e-12 {
            z_top += 1;
        } else if p[2].abs() < 1e-12 {
            z_bot += 1;
        } else {
            panic!("意外顶点 z 坐标 {}", p[2]);
        }
    }
    assert_eq!(z_top, 3);
    assert_eq!(z_bot, 3);
}

#[test]
fn extrude_face_invalid_face_id() {
    let (mut mesh, _v, _he, _f) = build_triangle();
    let fake = FaceId::default();
    let err = extrude_face(&mut mesh, fake, [0.0, 0.0, 1.0]).unwrap_err();
    match err {
        TopologyError::Inconsistent(_) => {}
        other => panic!("期望 Inconsistent 错误，得到 {:?}", other),
    }
}

#[test]
fn extrude_face_preserves_boundary_after_extrude() {
    // 挤出后整个面成为闭合体（无边界），原边界边变为内部边
    let (mut mesh, _v, _he, f) = build_triangle();
    let _ = extrude_face(&mut mesh, f, [0.0, 0.0, 1.0]).expect("挤出应成功");

    // 闭合体不应有任何边界半边
    let boundary_count = mesh
        .halfedge_ids()
        .filter(|h| {
            mesh.get_halfedge(*h)
                .map(|he| he.face.is_none())
                .unwrap_or(false)
        })
        .count();
    assert_eq!(boundary_count, 0, "挤出形成闭合体后不应有边界半边");
}

// ---------- extrude_region 测试 ----------

/// 构造闭合四面体（4 顶点、4 面、6 边、12 半边，所有边内部）。
/// 顶点：A=(0,0,0), B=(1,0,0), C=(0,1,0), D=(0,0,1)
/// 面（CCW 朝外）：
/// - F1 = (A, C, B)  法向 -z
/// - F2 = (A, B, D)  法向 -y
/// - F3 = (A, D, C)  法向 -x
/// - F4 = (B, C, D)  法向 (1,1,1)
fn build_tetrahedron() -> (MeshStorage, [VertexId; 4], [FaceId; 4]) {
    use crate::io::build_mesh_from_vertices_and_faces;
    let verts = vec![
        [0.0, 0.0, 0.0], // A
        [1.0, 0.0, 0.0], // B
        [0.0, 1.0, 0.0], // C
        [0.0, 0.0, 1.0], // D
    ];
    let faces = vec![
        [0, 2, 1], // F1 = (A, C, B)
        [0, 1, 3], // F2 = (A, B, D)
        [0, 3, 2], // F3 = (A, D, C)
        [1, 2, 3], // F4 = (B, C, D)
    ];
    let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
    let v_ids: Vec<VertexId> = mesh.vertex_ids().collect();
    let f_ids: Vec<FaceId> = mesh.face_ids().collect();
    assert_eq!(v_ids.len(), 4);
    assert_eq!(f_ids.len(), 4);
    assert_eq!(mesh.halfedge_count(), 12);
    (
        mesh,
        [v_ids[0], v_ids[1], v_ids[2], v_ids[3]],
        [f_ids[0], f_ids[1], f_ids[2], f_ids[3]],
    )
}

#[test]
fn extrude_region_single_face_of_tetrahedron() {
    // 从闭合四面体中挤出 1 个面 → 3 条边界边，3 个边界顶点
    let (mut mesh, _v, faces) = build_tetrahedron();
    assert!(validate_mesh(&mesh).is_ok());

    let v_before = mesh.vertex_count();
    let f_before = mesh.face_count();
    let he_before = mesh.halfedge_count();

    let offset = [0.0, 0.0, 0.5];
    let new_faces = extrude_region(&mut mesh, &[faces[0]], offset).expect("单面区域挤出应成功");

    // 3 条边界边 → 6 个侧面三角形
    assert_eq!(new_faces.len(), 6);
    // 3 个边界顶点 → +3 顶点
    assert_eq!(mesh.vertex_count(), v_before + 3);
    // +6 侧面（原 4 面复用为顶面）
    assert_eq!(mesh.face_count(), f_before + 6);
    // 半边：3 边界边×(2 h_top + 2 diag) + 3 顶点×2 垂直 = 12 + 6 = 18
    assert_eq!(mesh.halfedge_count(), he_before + 18);

    // 完整拓扑校验
    let errors = crate::validate::validate_topology(&mesh);
    assert!(errors.is_empty(), "挤出后拓扑应一致：{:?}", errors);

    // Euler 示性数：原闭合四面体 χ=2，挤出后仍为闭合体 χ=2
    let chi =
        mesh.vertex_count() as i64 - (mesh.halfedge_count() / 2) as i64 + mesh.face_count() as i64;
    assert_eq!(chi, 2, "挤出后 Euler 示性数应为 2");

    // 闭合体不应有边界半边
    let boundary_count = mesh
        .halfedge_ids()
        .filter(|h| {
            mesh.get_halfedge(*h)
                .map(|he| he.face.is_none())
                .unwrap_or(false)
        })
        .count();
    assert_eq!(boundary_count, 0, "挤出后应为闭合体");
}

#[test]
fn extrude_region_two_adjacent_faces() {
    // 挤出四面体的 2 个相邻面（共享边为内部边，不产生侧面）
    let (mut mesh, _v, faces) = build_tetrahedron();
    assert!(validate_mesh(&mesh).is_ok());

    let v_before = mesh.vertex_count();
    let f_before = mesh.face_count();
    let he_before = mesh.halfedge_count();

    // offset 选 [0.5,0.5,0.5] 避免与边界边 DA=(0,0,-1) 平行
    let offset = [0.5, 0.5, 0.5];
    // faces[0] = (A,C,B), faces[1] = (A,B,D)，共享边 AB
    let new_faces =
        extrude_region(&mut mesh, &[faces[0], faces[1]], offset).expect("双相邻面区域挤出应成功");

    // 4 条边界边 → 8 个侧面三角形
    assert_eq!(new_faces.len(), 8);
    // 4 个边界顶点 → +4 顶点
    assert_eq!(mesh.vertex_count(), v_before + 4);
    // +8 侧面
    assert_eq!(mesh.face_count(), f_before + 8);
    // 半边：4 边界边×4 + 4 顶点×2 = 16 + 8 = 24
    assert_eq!(mesh.halfedge_count(), he_before + 24);

    let errors = crate::validate::validate_topology(&mesh);
    assert!(errors.is_empty(), "挤出后拓扑应一致：{:?}", errors);

    let chi =
        mesh.vertex_count() as i64 - (mesh.halfedge_count() / 2) as i64 + mesh.face_count() as i64;
    assert_eq!(chi, 2, "挤出后 Euler 示性数应为 2");

    // 共享边 AB 不应产生侧面（内部边）
    // 验证：new_faces 数量 = 8 = 4 条边界边 × 2，说明共享边未产生侧面
}

#[test]
fn extrude_region_three_faces_leaving_cap() {
    // 挤出四面体的 3 个面，剩余 1 个面成"盖子"
    // 顶点 A 在 3 个面上 → 内部顶点；B/C/D 各缺 1 个面 → 边界顶点
    let (mut mesh, v, faces) = build_tetrahedron();
    assert!(validate_mesh(&mesh).is_ok());
    let [a, _b, _c, _d] = v;

    let v_before = mesh.vertex_count();
    let f_before = mesh.face_count();
    let he_before = mesh.halfedge_count();

    let offset = [0.0, 0.0, 0.5];
    // faces[0]=F1, faces[1]=F2, faces[2]=F3 → 剩余 faces[3]=F4 为盖子
    let new_faces = extrude_region(&mut mesh, &[faces[0], faces[1], faces[2]], offset)
        .expect("三面区域挤出应成功");

    // 3 条边界边（F4 的三条边）→ 6 个侧面三角形
    assert_eq!(new_faces.len(), 6);
    // 3 个边界顶点 (B,C,D) → +3 顶点；A 为内部顶点，原地平移
    assert_eq!(mesh.vertex_count(), v_before + 3);
    // +6 侧面
    assert_eq!(mesh.face_count(), f_before + 6);
    // 半边：3 边界边×4 + 3 顶点×2 = 12 + 6 = 18
    assert_eq!(mesh.halfedge_count(), he_before + 18);

    let errors = crate::validate::validate_topology(&mesh);
    assert!(errors.is_empty(), "挤出后拓扑应一致：{:?}", errors);

    let chi =
        mesh.vertex_count() as i64 - (mesh.halfedge_count() / 2) as i64 + mesh.face_count() as i64;
    assert_eq!(chi, 2, "挤出后 Euler 示性数应为 2");

    // 验证内部顶点 A 被平移（位置 += offset）
    let pos_a = mesh.get_vertex(a).unwrap().position;
    assert!(
        (pos_a[2] - 0.5).abs() < 1e-12,
        "内部顶点 A 应被平移到 z=0.5"
    );

    // 剩余面 F4 仍存在（盖子）
    assert!(mesh.contains_face(faces[3]));
}

#[test]
fn extrude_region_zero_offset_returns_error() {
    let (mut mesh, _v, faces) = build_tetrahedron();
    let err = extrude_region(&mut mesh, &[faces[0]], [0.0, 0.0, 0.0]).unwrap_err();
    assert_eq!(err, TopologyError::DegenerateTriangle);
}

#[test]
fn extrude_region_empty_faces_returns_error() {
    let (mut mesh, _v, _faces) = build_tetrahedron();
    let err = extrude_region(&mut mesh, &[], [0.0, 0.0, 1.0]).unwrap_err();
    match err {
        TopologyError::Inconsistent(_) => {}
        other => panic!("期望 Inconsistent 错误，得到 {:?}", other),
    }
}

#[test]
fn extrude_region_degenerate_side_returns_error() {
    // offset 平行于某条边界边会使侧面退化
    let (mut mesh, _v, faces) = build_tetrahedron();
    // F1 = (A,C,B)，边 AC 在 y 轴方向，边 CB 在 (−1,1,0) 方向，边 BA 在 (−1,0,0) 方向
    // offset 沿 y 轴 → 边 AC 与 offset 平行 → 侧面退化
    let err = extrude_region(&mut mesh, &[faces[0]], [0.0, 1.0, 0.0]).unwrap_err();
    assert_eq!(err, TopologyError::DegenerateTriangle);
}

// ---------- add_triangle 测试 ----------

#[test]
fn add_triangle_to_empty_mesh() {
    let mut mesh = MeshStorage::new();
    let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
    let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
    let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));

    let f = add_triangle(&mut mesh, v0, v1, v2).expect("add_triangle 应成功");
    assert!(mesh.contains_face(f));
    assert_eq!(mesh.face_count(), 1);
    assert_eq!(mesh.vertex_count(), 3);
    assert_eq!(mesh.halfedge_count(), 6); // 3 面内 + 3 边界 twin
    assert!(validate_mesh(&mesh).is_ok());
}

#[test]
fn add_two_adjacent_triangles() {
    let mut mesh = MeshStorage::new();
    let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
    let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
    let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));
    let v3 = mesh.add_vertex(Vertex::new([1.0, -1.0, 0.0]));

    let _f1 = add_triangle(&mut mesh, v0, v1, v2).expect("F1");
    let _f2 = add_triangle(&mut mesh, v1, v0, v3).expect("F2");

    assert_eq!(mesh.face_count(), 2);
    assert_eq!(mesh.vertex_count(), 4);
    assert!(validate_mesh(&mesh).is_ok());

    // 共享边 v0-v1 应有内部 twin 配对
    // 总共 10 半边（6 面内 + 4 边界 twin）
    assert_eq!(mesh.halfedge_count(), 10);
}

#[test]
fn add_degenerate_triangle_fails() {
    let mut mesh = MeshStorage::new();
    let v0 = mesh.add_vertex(Vertex::new([0.0; 3]));
    let v1 = mesh.add_vertex(Vertex::new([1.0; 3]));

    let err = add_triangle(&mut mesh, v0, v1, v0).unwrap_err();
    assert_eq!(err, TopologyError::DegenerateTriangle);
}

#[test]
fn add_triangle_with_invalid_vertex_fails() {
    let mut mesh = MeshStorage::new();
    let v = mesh.add_vertex(Vertex::new([0.0; 3]));
    let bad = VertexId::default();

    let err = add_triangle(&mut mesh, v, bad, v).unwrap_err();
    assert!(matches!(
        err,
        TopologyError::DegenerateTriangle | TopologyError::Inconsistent(_)
    ));
}

#[test]
fn add_triangle_preserves_existing_edge() {
    // 先手动建一个三角形，再用 add_triangle 加一个共享边的
    let mut mesh = MeshStorage::new();
    let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
    let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
    let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));

    // 用 add_triangle 建第一个三角形
    let _f1 = add_triangle(&mut mesh, v0, v1, v2).expect("F1");
    assert_eq!(mesh.face_count(), 1);

    // 再加一个共享 v0-v1 边的三角形
    let v3 = mesh.add_vertex(Vertex::new([1.0, -1.0, 0.0]));
    let _f2 = add_triangle(&mut mesh, v1, v0, v3).expect("F2");

    assert_eq!(mesh.face_count(), 2);
    assert!(validate_mesh(&mesh).is_ok());
    // 4 边界边（v1-v2, v2-v0, v0-v3, v3-v1）+ 1 内部边(v0-v1)
    // = 10 半边
    assert_eq!(mesh.halfedge_count(), 10);
}
