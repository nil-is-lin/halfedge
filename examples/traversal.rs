//! 示例：邻域遍历迭代器
//!
//! 演示 traversal 模块的 5 个迭代器 + 2 个边界判定函数。
//! 运行：`cargo run --example traversal`

use halfedge::build_icosphere;
use halfedge::traversal::{
    FaceHalfEdges, FaceVertices, VertexAdjacentFaces, VertexAdjacentVerts, VertexRing,
    is_boundary_edge, is_boundary_vertex,
};

fn main() {
    // 用 icosphere(0) 作为测试网格：12 顶点 / 20 面 / 30 边，闭合流形
    let mesh = build_icosphere(0);
    println!(
        "icosphere(0)：{} 顶点 / {} 面 / {} 半边",
        mesh.vertex_count(),
        mesh.face_count(),
        mesh.halfedge_count()
    );

    // 取第一个顶点和第一个面
    let v = mesh.vertex_ids().next().unwrap();
    let f = mesh.face_ids().next().unwrap();

    // ---------- 1. VertexRing：顶点的所有 outgoing 半边 ----------
    let ring: Vec<_> = VertexRing::new(&mesh, v).collect();
    println!("\n顶点 {:?} 的 outgoing 半边环（{} 条）：", v, ring.len());
    for he in &ring {
        let h = mesh.get_halfedge(*he).unwrap();
        let origin = mesh.get_halfedge(h.twin.unwrap()).unwrap().vertex;
        println!("  {:?}: {:?}→{:?}", he, origin, h.vertex);
    }

    // ---------- 2. VertexAdjacentVerts：邻居顶点 ----------
    let neighbors: Vec<_> = VertexAdjacentVerts::new(&mesh, v).collect();
    println!(
        "\n顶点 {:?} 的邻居（{} 个）：{:?}",
        v,
        neighbors.len(),
        neighbors
    );

    // ---------- 3. VertexAdjacentFaces：邻接面 ----------
    let adj_faces: Vec<_> = VertexAdjacentFaces::new(&mesh, v).collect();
    println!(
        "顶点 {:?} 的邻接面（{} 个）：{:?}",
        v,
        adj_faces.len(),
        adj_faces
    );

    // ---------- 4. FaceHalfEdges：面边界环半边 ----------
    let face_he: Vec<_> = FaceHalfEdges::new(&mesh, f).collect();
    println!("\n面 {:?} 的边界环（{} 条半边）：", f, face_he.len());
    for he in &face_he {
        let h = mesh.get_halfedge(*he).unwrap();
        println!("  {:?}: tip={:?}", he, h.vertex);
    }

    // ---------- 5. FaceVertices：面顶点 ----------
    let face_verts: Vec<_> = FaceVertices::new(&mesh, f).collect();
    println!("面 {:?} 的顶点：{:?}", f, face_verts);

    // ---------- 6. 边界判定 ----------
    // 闭合 icosphere 没有边界
    println!("\n边界判定：");
    let sample_he = mesh.halfedge_ids().next().unwrap();
    println!(
        "  is_boundary_edge({:?}) = {}（闭合球面应无边界）",
        sample_he,
        is_boundary_edge(&mesh, sample_he)
    );
    println!(
        "  is_boundary_vertex({:?}) = {}（闭合球面应无边界）",
        v,
        is_boundary_vertex(&mesh, v)
    );

    // ---------- 7. 迭代期可自由 &mut mesh ----------
    // 由于迭代器构造期已收集所有 ID，迭代期不持有借用，
    // 可以在循环体内修改 mesh（这里仅演示读取，实际可 get_vertex_mut）
    let count = VertexAdjacentVerts::new(&mesh, v).count();
    println!(
        "\n迭代期修改：顶点 {:?} 有 {} 个邻居，循环内可自由修改 mesh",
        v, count
    );
    for n in VertexAdjacentVerts::new(&mesh, v) {
        let p = mesh.get_vertex(n).unwrap().position;
        println!("  邻居 {:?} 位置 {:?}", n, p);
    }
}
