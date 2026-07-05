//! 示例：Loop 细分
//!
//! 演示 subdiv::loop_subdivide 在 icosphere 上的光滑细分效果。
//! 运行：`cargo run --example loop_subdivision`

use halfedge::traversal::VertexRing;
use halfedge::validate::check_topology;
use halfedge::{build_icosphere, loop_subdivide, validate_topology};

fn main() {
    println!("=== Loop 细分示例 ===\n");

    // ---------- 1. 单次细分：规模变化 ----------
    println!("[1] 单次 Loop 细分规模变化");
    let mesh = build_icosphere(1);
    print_mesh_stats("  细分前 icosphere(1)", &mesh);

    let refined = loop_subdivide(&mesh);
    print_mesh_stats("  细分后              ", &refined);
    println!("  规模验证：V'=V+E (42+120=162)，F'=4F (4*80=320)");
    println!(
        "  拓扑校验：{}",
        if check_topology(&refined).is_ok() {
            "OK"
        } else {
            "FAIL"
        }
    );

    // ---------- 2. 连续多次细分 ----------
    println!("\n[2] 连续多次 Loop 细分（从 icosphere(0) 开始）");
    let mut current = build_icosphere(0);
    println!(
        "  初始   : V={:>5} F={:>5} E={:>5}",
        current.vertex_count(),
        current.face_count(),
        current.halfedge_count() / 2
    );
    for i in 1..=4 {
        current = loop_subdivide(&current);
        let ok = check_topology(&current).is_ok();
        println!(
            "  细分 {}x: V={:>5} F={:>5} E={:>5}  拓扑校验={}",
            i,
            current.vertex_count(),
            current.face_count(),
            current.halfedge_count() / 2,
            if ok { "OK" } else { "FAIL" }
        );
    }

    // ---------- 3. 顶点度数分布 ----------
    println!("\n[3] 细分后顶点度数分布（icosphere(0) → 细分 1 次）");
    let mesh = build_icosphere(0);
    let refined = loop_subdivide(&mesh);
    let mut degree_hist: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for v in refined.vertex_ids() {
        let deg = VertexRing::new(&refined, v).count();
        *degree_hist.entry(deg).or_insert(0) += 1;
    }
    let mut sorted: Vec<_> = degree_hist.iter().collect();
    sorted.sort_by_key(|(d, _)| **d);
    for (deg, cnt) in sorted {
        println!("  度数 {} 的顶点数 = {}", deg, cnt);
    }
    println!("  （icosphere(0) 的 12 个原始顶点度数=5，细分后仍保持度数=5；");
    println!("   新增的边中点度数=6，符合 Loop 细分的正则性质）");

    // ---------- 4. Euler 示性数保持 ----------
    println!("\n[4] Euler 示性数 V - E + F 保持不变");
    let mut mesh = build_icosphere(0);
    for i in 0..=3 {
        if i > 0 {
            mesh = loop_subdivide(&mesh);
        }
        let v = mesh.vertex_count() as i64;
        let e = (mesh.halfedge_count() / 2) as i64;
        let f = mesh.face_count() as i64;
        println!(
            "  细分 {}x: V-E+F = {}-{}+{} = {}（应为 2）",
            i,
            v,
            e,
            f,
            v - e + f
        );
    }

    // ---------- 5. 位置变化：单位球面偏差 ----------
    // Loop 绽分是逼近型细分，会使网格略微收缩
    println!("\n[5] 单位球面偏差（icosphere 越细分越接近球面）");
    let mut mesh = build_icosphere(1);
    for i in 0..=3 {
        if i > 0 {
            mesh = loop_subdivide(&mesh);
        }
        let mut max_dev = 0.0_f64;
        let mut avg_dev = 0.0_f64;
        let n = mesh.vertex_count() as f64;
        for v in mesh.vertex_ids() {
            let p = mesh.get_vertex(v).unwrap().position;
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            let dev = (r - 1.0).abs();
            max_dev = max_dev.max(dev);
            avg_dev += dev;
        }
        println!(
            "  细分 {}x: 顶点数={:>5} 最大偏差={:.2e} 平均偏差={:.2e}",
            i,
            mesh.vertex_count(),
            max_dev,
            avg_dev / n
        );
    }

    // ---------- 6. validate_topology 完整校验 ----------
    println!("\n[6] validate_topology 完整校验");
    let mesh = build_icosphere(2);
    let refined = loop_subdivide(&mesh);
    let errors = validate_topology(&refined);
    println!(
        "  icosphere(2) 细分 1 次后错误数 = {}（应为 0）",
        errors.len()
    );
}

fn print_mesh_stats(label: &str, mesh: &halfedge::MeshStorage) {
    println!(
        "{}: V={:>5} F={:>5} E={:>5} HE={:>5}",
        label,
        mesh.vertex_count(),
        mesh.face_count(),
        mesh.halfedge_count() / 2,
        mesh.halfedge_count()
    );
}
