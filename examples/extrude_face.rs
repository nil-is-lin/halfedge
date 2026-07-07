//! 示例：面挤出（extrude_face）
//!
//! 演示 topology_ops::extrude_face 将三角面沿给定方向挤出形成棱柱体的效果。
//! 运行：`cargo run --example extrude_face`

use halfedge::traversal::FaceHalfEdges;
use halfedge::validate::check_topology;
use halfedge::{
    build_mesh_from_vertices_and_faces, extrude_face, extrude_faces, validate_topology,
};

fn main() {
    println!("=== extrude_face 面挤出示例 ===\n");

    // ---------- 1. 单个三角形挤出 ----------
    println!("[1] 单个三角形挤出");
    let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
    let faces = vec![[0, 1, 2]];
    let mut mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
    print_mesh_stats("  挤出前", &mesh);

    let f = mesh.face_ids().next().unwrap();
    let offset = [0.0, 0.0, 1.0];
    let new_faces = extrude_face(&mut mesh, f, offset).expect("挤出应成功");
    print_mesh_stats("  挤出后", &mesh);
    println!(
        "  新增面数 = {}（应为 7：1 顶面 + 6 侧面）",
        new_faces.len()
    );
    println!(
        "  拓扑校验：{}",
        if check_topology(&mesh).is_ok() {
            "OK"
        } else {
            "FAIL"
        }
    );
    println!(
        "  Euler 示性数 V-E+F = {}（闭合三棱柱应为 2）",
        mesh.vertex_count() as i64 - (mesh.halfedge_count() / 2) as i64 + mesh.face_count() as i64
    );

    // ---------- 2. 不同 offset 方向 ----------
    println!("\n[2] 不同 offset 方向的挤出结果");
    let offsets: [[f64; 3]; 4] = [
        [0.0, 0.0, 1.0],  // 沿 +Z（沿法向）
        [1.0, 0.0, 0.0],  // 沿 +X（侧面斜向）
        [0.5, 0.5, 1.0],  // 斜向上
        [0.0, 0.0, -2.0], // 沿 -Z 反向
    ];
    for (i, &off) in offsets.iter().enumerate() {
        let mut m = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let f = m.face_ids().next().unwrap();
        match extrude_face(&mut m, f, off) {
            Ok(_) => {
                let ok = check_topology(&m).is_ok();
                println!(
                    "  offset = {:?}: V={} F={} HE={} 校验={}",
                    off,
                    m.vertex_count(),
                    m.face_count(),
                    m.halfedge_count(),
                    if ok { "OK" } else { "FAIL" }
                );
            }
            Err(e) => println!("  offset = {:?}: 失败 - {}", off, e),
        }
        let _ = i;
    }

    // ---------- 3. 边界情况：零向量 / 退化 ----------
    println!("\n[3] 边界情况：零向量 / 退化 offset");
    let mut m = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
    let f = m.face_ids().next().unwrap();
    match extrude_face(&mut m, f, [0.0, 0.0, 0.0]) {
        Ok(_) => println!("  零向量 offset：意外成功"),
        Err(e) => println!("  零向量 offset：{}（预期行为）", e),
    }

    let mut m = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
    let f = m.face_ids().next().unwrap();
    // 三角形在 xy 平面，offset 沿 x 轴 → 与边 v0-v1 平行 → 侧面退化
    match extrude_face(&mut m, f, [1.0, 0.0, 0.0]) {
        Ok(_) => println!("  x 轴 offset：意外成功"),
        Err(e) => println!("  x 轴 offset：{}（预期行为：侧面退化）", e),
    }

    // ---------- 4. 批量挤出多个不相交三角形 ----------
    println!("\n[4] 批量挤出多个不相交三角形");
    let verts = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [3.5, 1.0, 0.0],
        [6.0, 0.0, 0.0],
        [7.0, 0.0, 0.0],
        [6.5, 1.0, 0.0],
    ];
    let faces = vec![[0, 1, 2], [3, 4, 5], [6, 7, 8]];
    let mut mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
    print_mesh_stats("  批量挤出前", &mesh);

    let fids: Vec<_> = mesh.face_ids().collect();
    let offset = [0.0, 0.0, 0.5];
    let new_faces = extrude_faces(&mut mesh, &fids, offset).expect("批量挤出应成功");
    print_mesh_stats("  批量挤出后", &mesh);
    println!("  新增面数 = {}（3 个面 × 7 = 21）", new_faces.len());
    println!(
        "  拓扑校验：{}",
        if check_topology(&mesh).is_ok() {
            "OK"
        } else {
            "FAIL"
        }
    );

    // ---------- 5. 顶点位置验证 ----------
    println!("\n[5] 顶点位置验证（offset 沿法向）");
    let mut m = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
    let f = m.face_ids().next().unwrap();
    let offset = [0.0, 0.0, 5.0];
    let _ = extrude_face(&mut m, f, offset).unwrap();
    let mut bottom = 0;
    let mut top = 0;
    for v in m.vertex_ids() {
        let p = m.get_vertex(v).unwrap().position;
        if p[2].abs() < 1e-12 {
            bottom += 1;
        } else if (p[2] - 5.0).abs() < 1e-12 {
            top += 1;
        }
    }
    // 网格含 3 个不相交三角形（9 顶点），只挤出其中一个 → 底 9 + 顶 3 = 12
    println!("  底面顶点数 = {}（应为 9）", bottom);
    println!("  顶面顶点数 = {}（应为 3）", top);

    // ---------- 6. validate_topology 完整校验 ----------
    println!("\n[6] validate_topology 完整校验");
    let errors = validate_topology(&mesh);
    println!("  批量挤出后错误数 = {}（应为 0）", errors.len());

    // ---------- 7. 面半边遍历验证 ----------
    println!("\n[7] 挤出后面半边遍历验证");
    for f in mesh.face_ids() {
        let he_count = FaceHalfEdges::new(&mesh, f).count();
        assert_eq!(he_count, 3, "面 {:?} 边界环长度应为 3", f);
    }
    println!(
        "  所有 {} 个面的边界环长度均为 3（三角网格）",
        mesh.face_count()
    );
}

fn print_mesh_stats(label: &str, mesh: &halfedge::MeshStorage) {
    println!(
        "{}: V={:>3} F={:>3} E={:>3} HE={:>3}",
        label,
        mesh.vertex_count(),
        mesh.face_count(),
        mesh.halfedge_count() / 2,
        mesh.halfedge_count()
    );
}
