//! 示例：三大核心拓扑操作 split / flip / collapse
//!
//! 演示 topology_ops 模块的边分裂、边翻转、边折叠。
//! 运行：`cargo run --example topology_ops`

use halfedge::validate::check_topology;
use halfedge::{build_icosphere, collapse_edge, flip_edge, split_edge, validate_mesh};

fn main() {
    let mut mesh = build_icosphere(0);
    println!(
        "初始 icosphere(0)：{} 顶点 / {} 面 / {} 半边",
        mesh.vertex_count(),
        mesh.face_count(),
        mesh.halfedge_count()
    );

    // ---------- 1. split_edge：边分裂 ----------
    // 取一条内部半边
    let he = mesh
        .halfedge_ids()
        .find(|h| {
            mesh.get_halfedge(*h)
                .map(|h| h.face.is_some())
                .unwrap_or(false)
        })
        .unwrap();

    let v_count_before = mesh.vertex_count();
    let f_count_before = mesh.face_count();
    let new_v = split_edge(&mut mesh, he).expect("split 应成功");
    println!("\n[split_edge] 在中点插入新顶点 {:?}", new_v);
    println!("  顶点 {} → {}", v_count_before, mesh.vertex_count());
    println!("  面 {} → {}", f_count_before, mesh.face_count());
    println!("  validate_mesh: {:?}", validate_mesh(&mesh));
    println!(
        "  check_topology: {}",
        if check_topology(&mesh).is_ok() {
            "OK"
        } else {
            "FAIL"
        }
    );

    // ---------- 2. flip_edge：内部边翻转 ----------
    // 重新构造一个干净网格做 flip
    let mut mesh = build_icosphere(0);
    // 找一条内部边（两侧都有面）
    let he = mesh
        .halfedge_ids()
        .find(|h| {
            mesh.get_halfedge(*h)
                .map(|h| h.face.is_some())
                .unwrap_or(false)
        })
        .unwrap();

    println!("\n[flip_edge] 翻转内部边 {:?}", he);
    flip_edge(&mut mesh, he).expect("flip 应成功");
    println!(
        "  顶点 / 面 / 半边数不变：{} / {} / {}",
        mesh.vertex_count(),
        mesh.face_count(),
        mesh.halfedge_count()
    );
    println!("  validate_mesh: {:?}", validate_mesh(&mesh));
    println!(
        "  check_topology: {}",
        if check_topology(&mesh).is_ok() {
            "OK"
        } else {
            "FAIL"
        }
    );

    // ---------- 3. collapse_edge：边折叠 ----------
    // 重新构造干净网格做 collapse
    let mut mesh = build_icosphere(0);
    let he = mesh
        .halfedge_ids()
        .find(|h| {
            mesh.get_halfedge(*h)
                .map(|h| h.face.is_some())
                .unwrap_or(false)
        })
        .unwrap();

    let v_before = mesh.vertex_count();
    let f_before = mesh.face_count();
    let he_before = mesh.halfedge_count();
    let new_v = collapse_edge(&mut mesh, he).expect("collapse 应成功");
    println!("\n[collapse_edge] 折叠边，合并为新顶点 {:?}", new_v);
    println!("  顶点 {} → {}", v_before, mesh.vertex_count());
    println!("  面 {} → {}", f_before, mesh.face_count());
    println!("  半边 {} → {}", he_before, mesh.halfedge_count());
    println!("  validate_mesh: {:?}", validate_mesh(&mesh));
    println!(
        "  check_topology: {}",
        if check_topology(&mesh).is_ok() {
            "OK"
        } else {
            "FAIL"
        }
    );

    // ---------- 4. 错误处理：翻转边界边应失败 ----------
    // 构造一个单三角形（全是边界边）
    let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let faces = [[0, 1, 2]];
    let mut mesh = halfedge::build_mesh_from_vertices_and_faces(&vertices, &faces).unwrap();
    let boundary_he = mesh
        .halfedge_ids()
        .find(|h| {
            mesh.get_halfedge(*h)
                .map(|h| h.face.is_some())
                .unwrap_or(false)
        })
        .unwrap();
    let result = flip_edge(&mut mesh, boundary_he);
    println!(
        "\n错误处理：翻转边界边 → {:?}",
        result.err().map(|e| e.to_string())
    );
}
