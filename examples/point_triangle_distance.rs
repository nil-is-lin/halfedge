//! 示例：点到三角形最近距离（Ericson 算法）
//!
//! 演示 geometry::point_triangle_distance 的 7 个 Voronoi 区域判别。
//! 运行：`cargo run --example point_triangle_distance`

use halfedge::point_triangle_distance;

fn main() {
    // 三角形：A=(0,0,0), B=(1,0,0), C=(0,1,0)，位于 xy 平面，CCW 朝 +z
    let a = [0.0, 0.0, 0.0];
    let b = [1.0, 0.0, 0.0];
    let c = [0.0, 1.0, 0.0];

    println!("三角形 A={:?} B={:?} C={:?}", a, b, c);
    println!();

    // ---------- 1. 面域：投影点在三角形内部 ----------
    let p = [0.25, 0.25, 0.5]; // 重心附近，上方 0.5
    let d = point_triangle_distance(p, a, b, c);
    println!("[面域] P={:?} 距离 = {:.4}（应为 0.5，垂直高度）", p, d);

    // ---------- 2. 顶点域 A ----------
    let p = [-1.0, -1.0, 0.0]; // 远离 A
    let d = point_triangle_distance(p, a, b, c);
    let expected = (2.0_f64).sqrt();
    println!(
        "[顶点域 A] P={:?} 距离 = {:.4}（应为 √2 ≈ {:.4}）",
        p, d, expected
    );

    // ---------- 3. 顶点域 B ----------
    let p = [3.0, 0.0, 0.0]; // 远离 B
    let d = point_triangle_distance(p, a, b, c);
    println!("[顶点域 B] P={:?} 距离 = {:.4}（应为 2.0）", p, d);

    // ---------- 4. 顶点域 C ----------
    let p = [0.0, 3.0, 0.0]; // 远离 C
    let d = point_triangle_distance(p, a, b, c);
    println!("[顶点域 C] P={:?} 距离 = {:.4}（应为 2.0）", p, d);

    // ---------- 5. 边域 AB ----------
    let p = [0.5, -1.0, 0.0]; // 投影在 AB 延长线外
    let d = point_triangle_distance(p, a, b, c);
    println!("[边域 AB] P={:?} 距离 = {:.4}（应为 1.0）", p, d);

    // ---------- 6. 边域 AC ----------
    let p = [-1.0, 0.5, 0.0]; // 投影在 AC 延长线外
    let d = point_triangle_distance(p, a, b, c);
    println!("[边域 AC] P={:?} 距离 = {:.4}（应为 1.0）", p, d);

    // ---------- 7. 边域 BC ----------
    let p = [1.0, 1.0, 0.0]; // 投影在 BC 边上
    let d = point_triangle_distance(p, a, b, c);
    let expected = 1.0 / (2.0_f64).sqrt();
    println!(
        "[边域 BC] P={:?} 距离 = {:.4}（应为 1/√2 ≈ {:.4}）",
        p, d, expected
    );

    // ---------- 8. 3D 空间任意点 ----------
    let p = [0.3, 0.3, 1.0];
    let d = point_triangle_distance(p, a, b, c);
    println!(
        "\n[3D 点] P={:?} 距离 = {:.4}（上方 1.0，应接近 1.0）",
        p, d
    );

    // ---------- 9. 退化三角形（共线）应不 panic ----------
    let a = [0.0, 0.0, 0.0];
    let b = [1.0, 0.0, 0.0];
    let c = [2.0, 0.0, 0.0]; // 共线
    let p = [0.5, 1.0, 0.0];
    let d = point_triangle_distance(p, a, b, c);
    println!(
        "\n[退化三角形] 共线 A-B-C, P={:?} 距离 = {:.4}（退化为顶点距离）",
        p, d
    );
}
