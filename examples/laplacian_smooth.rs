//! 示例：拉普拉斯平滑
//!
//! 演示 geometry 模块的拉普拉斯平滑（顶点级 + 整网级）。
//! 运行：`cargo run --example laplacian_smooth`

use halfedge::{build_icosphere, laplacian_smooth_mesh, laplacian_smooth_vertex};

fn main() {
    let mut mesh = build_icosphere(1); // 42 顶点的细分球
    println!("icosphere(1)：{} 顶点", mesh.vertex_count());

    // ---------- 1. 单顶点拉普拉斯位置 ----------
    let v = mesh.vertex_ids().next().unwrap();
    let original = mesh.get_vertex(v).unwrap().position;
    let target = laplacian_smooth_vertex(&mesh, v).unwrap();
    println!("\n[laplacian_smooth_vertex] 顶点 {:?}", v);
    println!(
        "  原位置     = [{:.4}, {:.4}, {:.4}]",
        original[0], original[1], original[2]
    );
    println!(
        "  邻居平均   = [{:.4}, {:.4}, {:.4}]",
        target[0], target[1], target[2]
    );
    let displacement = (
        target[0] - original[0],
        target[1] - original[1],
        target[2] - original[2],
    );
    let disp_len = (displacement.0 * displacement.0
        + displacement.1 * displacement.1
        + displacement.2 * displacement.2)
        .sqrt();
    println!("  位移量     = {:.6}", disp_len);

    // ---------- 2. 整网平滑：重心保留 ----------
    let centroid_before = mesh_centroid(&mesh);
    println!(
        "\n[laplacian_smooth_mesh] 平滑前重心 = [{:.4}, {:.4}, {:.4}]",
        centroid_before.0, centroid_before.1, centroid_before.2
    );

    // λ=0.5, 5 次迭代
    laplacian_smooth_mesh(&mut mesh, 0.5, 5);

    let centroid_after = mesh_centroid(&mesh);
    println!(
        "  λ=0.5, 5 次迭代后重心 = [{:.4}, {:.4}, {:.4}]",
        centroid_after.0, centroid_after.1, centroid_after.2
    );
    println!(
        "  重心偏移 = {:.2e}（应接近 0，拉普拉斯平滑保留重心）",
        ((centroid_after.0 - centroid_before.0).powi(2)
            + (centroid_after.1 - centroid_before.1).powi(2)
            + (centroid_after.2 - centroid_before.2).powi(2))
        .sqrt()
    );

    // ---------- 3. 平滑前后顶点到原点距离变化 ----------
    // icosphere 顶点都在单位球面上（|p|=1），平滑后顶点会向内收缩
    let mut max_deviation = 0.0_f64;
    for v in mesh.vertex_ids() {
        let p = mesh.get_vertex(v).unwrap().position;
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        max_deviation = max_deviation.max((r - 1.0).abs());
    }
    println!(
        "\n  平滑后顶点距单位球面最大偏差 = {:.6}（拉普拉斯平滑使球面收缩）",
        max_deviation
    );

    // ---------- 4. 边界条件：λ=0 或 iterations=0 不做任何修改 ----------
    let mut mesh = build_icosphere(0);
    let p_before: Vec<_> = mesh
        .vertex_ids()
        .map(|v| mesh.get_vertex(v).unwrap().position)
        .collect();
    laplacian_smooth_mesh(&mut mesh, 0.0, 10);
    let p_after: Vec<_> = mesh
        .vertex_ids()
        .map(|v| mesh.get_vertex(v).unwrap().position)
        .collect();
    let unchanged = p_before.iter().zip(p_after.iter()).all(|(a, b)| a == b);
    println!("\n  λ=0 时网格不变：{}", unchanged);
}

fn mesh_centroid(mesh: &halfedge::MeshStorage) -> (f64, f64, f64) {
    let mut sum = [0.0; 3];
    let mut n = 0;
    for v in mesh.vertex_ids() {
        let p = mesh.get_vertex(v).unwrap().position;
        sum[0] += p[0];
        sum[1] += p[1];
        sum[2] += p[2];
        n += 1;
    }
    (sum[0] / n as f64, sum[1] / n as f64, sum[2] / n as f64)
}
