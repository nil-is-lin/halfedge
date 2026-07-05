//! 示例：storage 模块基础用法
//!
//! 演示 MeshStorage 的顶点 / 半边 / 面 CRUD 与句柄有效性判断。
//! 运行：`cargo run --example storage_basic`

use halfedge::{Face, HalfEdge, MeshStorage, Vertex};

fn main() {
    let mut mesh = MeshStorage::new();

    // ---------- 增：创建 3 顶点 + 3 半边 + 1 面 ----------
    let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
    let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
    let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));

    let h0 = mesh.add_halfedge(HalfEdge::new(v1)); // v0 → v1
    let h1 = mesh.add_halfedge(HalfEdge::new(v2)); // v1 → v2
    let h2 = mesh.add_halfedge(HalfEdge::new(v0)); // v2 → v0
    let f = mesh.add_face(Face::new());

    // 设置 next/prev/face 环
    for (he, next, prev) in [(h0, h1, h2), (h1, h2, h0), (h2, h0, h1)] {
        let h = mesh.get_halfedge_mut(he).unwrap();
        h.next = Some(next);
        h.prev = Some(prev);
        h.face = Some(f);
    }
    mesh.get_face_mut(f).unwrap().halfedge = Some(h0);
    mesh.get_vertex_mut(v0).unwrap().halfedge = Some(h0);

    println!(
        "网格统计：{} 顶点 / {} 半边 / {} 面",
        mesh.vertex_count(),
        mesh.halfedge_count(),
        mesh.face_count()
    );

    // ---------- 查：句柄有效性 + 字段读取 ----------
    println!("\n查询 v0：");
    println!("  contains_vertex(v0) = {}", mesh.contains_vertex(v0));
    println!("  position = {:?}", mesh.get_vertex(v0).unwrap().position);
    println!(
        "  outgoing halfedge = {:?}",
        mesh.get_vertex(v0).unwrap().halfedge
    );

    // ---------- 改：原地修改字段 ----------
    mesh.get_vertex_mut(v0).unwrap().position = [0.5, 0.5, 0.0];
    println!(
        "\n修改后 v0 position = {:?}",
        mesh.get_vertex(v0).unwrap().position
    );

    // ---------- 删：句柄失效 + ABA 安全 ----------
    let removed = mesh.remove_vertex(v2).unwrap();
    println!("\n删除 v2：position = {:?}", removed.position);
    println!(
        "  contains_vertex(v2) = {} (应为 false)",
        mesh.contains_vertex(v2)
    );
    println!("  get_vertex(v2) = {:?} (应为 None)", mesh.get_vertex(v2));
    println!("  vertex_count = {} (应为 2)", mesh.vertex_count());

    // 槽位复用：新插入的顶点不会复活旧句柄
    let v3 = mesh.add_vertex(Vertex::new([2.0, 2.0, 2.0]));
    println!(
        "\n新插入 v3 = {:?}, v2 仍无效 = {}",
        v3,
        !mesh.contains_vertex(v2)
    );

    // ---------- 迭代：遍历所有有效句柄 ----------
    println!("\n所有顶点句柄：");
    for (i, v) in mesh.vertex_ids().enumerate() {
        let p = mesh.get_vertex(v).unwrap().position;
        println!("  [{}] {:?} -> position = {:?}", i, v, p);
    }
}
