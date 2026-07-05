//! 示例：拓扑自检 validate_topology
//!
//! 演示 validate 模块的完整校验：干净网格通过、人为破坏后检测错误。
//! 运行：`cargo run --example validate`

use halfedge::{
    ValidationError, build_icosphere, build_mesh_from_vertices_and_faces, check_topology,
    validate_topology,
};

fn main() {
    // ---------- 1. 干净网格应通过校验 ----------
    let mesh = build_icosphere(1);
    let errors = validate_topology(&mesh);
    println!(
        "[干净 icosphere(1)] 校验错误数 = {}（应为 0）",
        errors.len()
    );
    println!(
        "  check_topology: {}",
        if check_topology(&mesh).is_ok() {
            "OK"
        } else {
            "FAIL"
        }
    );

    // ---------- 2. 检测 twin 不匹配 ----------
    let (mut mesh, hes) = build_two_triangle_mesh();
    let h0 = hes[0];
    // 让 h0.twin 指向自己（破坏 twin 互指）
    mesh.get_halfedge_mut(h0).unwrap().twin = Some(h0);
    let errors = validate_topology(&mesh);
    let has_twin_mismatch = errors
        .iter()
        .any(|e| matches!(e, ValidationError::TwinMismatch { .. }));
    let has_self_loop = errors
        .iter()
        .any(|e| matches!(e, ValidationError::SelfLoopHalfEdge(_)));
    println!(
        "\n[twin 不匹配] 检测到 TwinMismatch = {}, SelfLoop = {}",
        has_twin_mismatch, has_self_loop
    );

    // ---------- 3. 检测悬空顶点引用 ----------
    let (mut mesh, hes) = build_two_triangle_mesh();
    let h0 = hes[0];
    use halfedge::ids::VertexId;
    let bad_v = VertexId::default();
    mesh.get_halfedge_mut(h0).unwrap().vertex = bad_v;
    let errors = validate_topology(&mesh);
    let has_dangling = errors
        .iter()
        .any(|e| matches!(e, ValidationError::HalfEdgeDanglingVertex { .. }));
    println!(
        "\n[悬空顶点] 检测到 HalfEdgeDanglingVertex = {}",
        has_dangling
    );

    // ---------- 4. 检测退化面 ----------
    let vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0], // 与 A、B 共线
    ];
    let faces = [[0, 1, 2]];
    let mesh = build_mesh_from_vertices_and_faces(&vertices, &faces);
    let errors = validate_topology(&mesh);
    let has_degenerate = errors
        .iter()
        .any(|e| matches!(e, ValidationError::DegenerateFace { .. }));
    println!(
        "\n[退化面] 共线三角形检测到 DegenerateFace = {}",
        has_degenerate
    );

    // ---------- 5. 检测顶点入口不一致 ----------
    let (mut mesh, hes) = build_two_triangle_mesh();
    let h1 = hes[1]; // h1 的 tip 是 v2，origin 是 v1
    // 让 v0 的 outgoing 指向 h1（origin 不匹配）
    let v0 = mesh.vertex_ids().next().unwrap();
    mesh.get_vertex_mut(v0).unwrap().halfedge = Some(h1);
    let errors = validate_topology(&mesh);
    let has_inconsistent = errors
        .iter()
        .any(|e| matches!(e, ValidationError::VertexHalfEdgeInconsistent { .. }));
    println!(
        "\n[入口不一致] 检测到 VertexHalfEdgeInconsistent = {}",
        has_inconsistent
    );

    // ---------- 6. 输出所有错误详情 ----------
    println!("\n[完整错误列表示例]");
    let (mut mesh, hes) = build_two_triangle_mesh();
    let h0 = hes[0];
    mesh.get_halfedge_mut(h0).unwrap().twin = Some(h0);
    let errors = validate_topology(&mesh);
    for (i, e) in errors.iter().enumerate() {
        println!("  错误 {}: {}", i + 1, e);
    }
    println!("  共 {} 个错误", errors.len());
}

/// 构造两个三角形拼成的四边形，返回 (mesh, [h0, h1, h2, ...])。
fn build_two_triangle_mesh() -> (halfedge::MeshStorage, Vec<halfedge::HalfEdgeId>) {
    let vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let faces = [[0, 1, 2], [0, 2, 3]];
    let mesh = build_mesh_from_vertices_and_faces(&vertices, &faces);
    let hes: Vec<_> = mesh.halfedge_ids().collect();
    (mesh, hes)
}
