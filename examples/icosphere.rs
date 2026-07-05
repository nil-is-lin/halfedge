//! 示例：生成 icosphere 球体
//!
//! 演示 test_util::build_icosphere 在不同细分次数下的网格规模与性质。
//! 运行：`cargo run --example icosphere`

use halfedge::traversal::{FaceHalfEdges, VertexRing};
use halfedge::{build_icosphere, face_normal, validate_topology};

fn main() {
    println!("=== icosphere 生成器示例 ===\n");

    // ---------- 不同细分级别的网格规模 ----------
    println!("[细分级别与网格规模]");
    println!(
        "  {:>4} {:>8} {:>8} {:>10} {:>10}",
        "n", "顶点", "面", "半边", "公式验证"
    );
    for n in 0..=4 {
        let mesh = build_icosphere(n);
        let v = mesh.vertex_count();
        let f = mesh.face_count();
        let he = mesh.halfedge_count();
        // 公式：V=10·4^n+2, F=20·4^n, E=30·4^n, 半边=2E=60·4^n
        let v_expected = 10 * 4_usize.pow(n as u32) + 2;
        let f_expected = 20 * 4_usize.pow(n as u32);
        let he_expected = 60 * 4_usize.pow(n as u32);
        let ok = v == v_expected && f == f_expected && he == he_expected;
        println!(
            "  {:>4} {:>8} {:>8} {:>10} {:>10}",
            n,
            v,
            f,
            he,
            if ok { "✓" } else { "✗" }
        );
    }

    // ---------- 性质 1：所有顶点在单位球面上 ----------
    let mesh = build_icosphere(2);
    let mut max_dev = 0.0_f64;
    for v in mesh.vertex_ids() {
        let p = mesh.get_vertex(v).unwrap().position;
        let r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
        max_dev = max_dev.max((r2 - 1.0).abs());
    }
    println!(
        "\n[性质 1: 单位球面] icosphere(2) 顶点距单位球面最大偏差 = {:.2e}",
        max_dev
    );

    // ---------- 性质 2：所有面法向朝外 ----------
    let mesh = build_icosphere(1);
    let mut all_outward = true;
    let mut count = 0;
    for f in mesh.face_ids() {
        let n = face_normal(&mesh, f).unwrap();
        let verts: Vec<_> = FaceHalfEdges::new(&mesh, f)
            .filter_map(|he| mesh.get_halfedge(he))
            .map(|h| h.vertex)
            .filter_map(|v| mesh.get_vertex(v))
            .map(|v| v.position)
            .collect();
        let centroid = [
            (verts[0][0] + verts[1][0] + verts[2][0]) / 3.0,
            (verts[0][1] + verts[1][1] + verts[2][1]) / 3.0,
            (verts[0][2] + verts[1][2] + verts[2][2]) / 3.0,
        ];
        let dot = n[0] * centroid[0] + n[1] * centroid[1] + n[2] * centroid[2];
        if dot <= 0.0 {
            all_outward = false;
            break;
        }
        count += 1;
    }
    println!(
        "[性质 2: 法向朝外] icosphere(1) 所有 {} 个面法向朝外 = {}",
        count, all_outward
    );

    // ---------- 性质 3：每个顶点恰好有 5 或 6 个邻居 ----------
    // icosahedron 的 12 个原始顶点度数为 5，细分产生的新顶点度数为 6
    let mesh = build_icosphere(0);
    let mut degree_hist: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for v in mesh.vertex_ids() {
        let deg = VertexRing::new(&mesh, v).count();
        *degree_hist.entry(deg).or_insert(0) += 1;
    }
    println!("\n[性质 3: 顶点度数] icosphere(0) 度数分布：");
    for (deg, cnt) in degree_hist.iter().collect::<Vec<_>>() {
        println!("  度数 {} 的顶点数 = {}", deg, cnt);
    }

    // ---------- 性质 4：拓扑校验通过 ----------
    let mesh = build_icosphere(3);
    let errors = validate_topology(&mesh);
    println!(
        "\n[性质 4: 拓扑校验] icosphere(3) 错误数 = {}（应为 0）",
        errors.len()
    );

    // ---------- 性质 5：Euler 公式 V - E + F = 2 ----------
    for n in 0..=3 {
        let mesh = build_icosphere(n);
        let v = mesh.vertex_count();
        let e = mesh.halfedge_count() / 2;
        let f = mesh.face_count();
        let euler = v as i64 - e as i64 + f as i64;
        println!(
            "[性质 5: Euler] n={}: V-E+F = {}-{}+{} = {}（应为 2）",
            n, v, e, f, euler
        );
    }
}
