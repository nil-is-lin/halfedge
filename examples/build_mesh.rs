//! 示例：从顶点 + 面索引构建完整半边网格
//!
//! 演示 io::build_mesh_from_vertices_and_faces，自动构建 twin / next / prev / 边界环。
//! 运行：`cargo run --example build_mesh`

use halfedge::traversal::{is_boundary_edge, is_boundary_vertex};
use halfedge::{build_mesh_from_vertices_and_faces, validate_topology};

fn main() {
    // 四边形拆成两个三角形（CCW 朝向 +z）
    let vertices = [
        [0.0, 0.0, 0.0], // v0
        [1.0, 0.0, 0.0], // v1
        [1.0, 1.0, 0.0], // v2
        [0.0, 1.0, 0.0], // v3
    ];
    let faces = [
        [0, 1, 2], // 三角形 1：v0-v1-v2
        [0, 2, 3], // 三角形 2：v0-v2-v3（共享边 v0-v2）
    ];

    let mesh = build_mesh_from_vertices_and_faces(&vertices, &faces);

    println!(
        "构建结果：{} 顶点 / {} 半边 / {} 面",
        mesh.vertex_count(),
        mesh.halfedge_count(),
        mesh.face_count()
    );
    // 4 顶点、2 面、内部边 1 对（2 半边）+ 边界边 4 对（8 半边）= 10 半边

    // 完整校验
    let errors = validate_topology(&mesh);
    println!("拓扑校验：{} 个错误", errors.len());
    for e in &errors {
        println!("  - {}", e);
    }

    // 边界检测
    println!("\n边界检测：");
    for v in mesh.vertex_ids() {
        println!(
            "  顶点 {:?} 是边界顶点：{}",
            v,
            is_boundary_vertex(&mesh, v)
        );
    }

    println!("\n半边列表（标记边界 / 内部）：");
    for he in mesh.halfedge_ids() {
        let h = mesh.get_halfedge(he).unwrap();
        let tip = h.vertex;
        let origin = mesh.get_halfedge(h.twin.unwrap()).unwrap().vertex;
        let kind = if is_boundary_edge(&mesh, he) {
            "边界"
        } else {
            "内部"
        };
        println!("  {:?}: {:?}→{:?} [{}]", he, origin, tip, kind);
    }
}
