//! 示例：几何查询（边长 / 面积 / 法向 / 最小内角 / 顶点法向）
//!
//! 演示 geometry 模块的 5 个查询函数。
//! 运行：`cargo run --example geometry_query`

use halfedge::{
    build_icosphere, edge_length, face_area, face_min_angle, face_normal, vertex_normal,
};

fn main() {
    let mesh = build_icosphere(0);
    println!(
        "icosphere(0)：{} 顶点 / {} 面",
        mesh.vertex_count(),
        mesh.face_count()
    );

    // ---------- 1. edge_length：边长 ----------
    let he = mesh.halfedge_ids().next().unwrap();
    let len = edge_length(&mesh, he).unwrap();
    println!("\n[edge_length] 半边 {:?} 长度 = {:.6}", he, len);

    // 统计所有边长
    let mut min_len = f64::MAX;
    let mut max_len: f64 = 0.0;
    let mut sum = 0.0;
    let mut count = 0;
    for h in mesh.halfedge_ids() {
        // 只统计每条无向边一次（取 tip < origin 的半边）
        let he_data = mesh.get_halfedge(h).unwrap();
        let origin = mesh.get_halfedge(he_data.twin.unwrap()).unwrap().vertex;
        if format!("{:?}", he_data.vertex) < format!("{:?}", origin)
            && let Some(l) = edge_length(&mesh, h)
        {
            min_len = min_len.min(l);
            max_len = max_len.max(l);
            sum += l;
            count += 1;
        }
    }
    println!(
        "  边长统计：{} 条边，min={:.6}, max={:.6}, avg={:.6}",
        count,
        min_len,
        max_len,
        sum / count as f64
    );

    // ---------- 2. face_area：三角面积 ----------
    let f = mesh.face_ids().next().unwrap();
    let area = face_area(&mesh, f).unwrap();
    println!("\n[face_area] 面 {:?} 面积 = {:.6}", f, area);

    let total_area: f64 = mesh.face_ids().filter_map(|f| face_area(&mesh, f)).sum();
    println!(
        "  总面积 = {:.6}（单位球面面积 4π ≈ {:.6}）",
        total_area,
        4.0 * std::f64::consts::PI
    );

    // ---------- 3. face_normal：面法向 ----------
    let n = face_normal(&mesh, f).unwrap();
    println!(
        "\n[face_normal] 面 {:?} 法向 = [{:.4}, {:.4}, {:.4}]",
        f, n[0], n[1], n[2]
    );
    println!(
        "  |n| = {:.6}（应为单位向量）",
        (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
    );

    // ---------- 4. face_min_angle：最小内角 ----------
    let min_ang = face_min_angle(&mesh, f).unwrap();
    let deg = min_ang.to_degrees();
    println!(
        "\n[face_min_angle] 面 {:?} 最小内角 = {:.4} rad = {:.2}°",
        f, min_ang, deg
    );

    // 统计所有面的最小内角范围
    let mut all_min: Vec<f64> = mesh
        .face_ids()
        .filter_map(|f| face_min_angle(&mesh, f))
        .map(|a| a.to_degrees())
        .collect();
    all_min.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "  全网最小内角范围：{:.2}° ~ {:.2}°",
        all_min[0],
        all_min[all_min.len() - 1]
    );

    // ---------- 5. vertex_normal：顶点法向（面积加权） ----------
    let v = mesh.vertex_ids().next().unwrap();
    let vn = vertex_normal(&mesh, v).unwrap();
    let pos = mesh.get_vertex(v).unwrap().position;
    println!(
        "\n[vertex_normal] 顶点 {:?} 法向 = [{:.4}, {:.4}, {:.4}]",
        v, vn[0], vn[1], vn[2]
    );
    println!(
        "  顶点位置 = [{:.4}, {:.4}, {:.4}]（球面上应与法向平行）",
        pos[0], pos[1], pos[2]
    );
    let dot = vn[0] * pos[0] + vn[1] * pos[1] + vn[2] * pos[2];
    println!("  法向·位置 = {:.6}（正值表示法向朝外）", dot);
}
