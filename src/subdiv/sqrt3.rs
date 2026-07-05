//! √3（Sqrt3）细分模块
//!
//! 实现 Kobbelt 2000 的 √3 细分算法，适用于三角网格。
//! 每次细分面数增长 3 倍（而非 Loop 的 4 倍），适合渐进式细分。
//!
//! ## 算法概述
//! √3 细分是\textbf{插值型}细分方案。每个原始三角面插入重心面点，
//! 然后翻转所有原始内部边（连接相邻面点），最后松弛原始顶点。
//!
//! ### 步骤
//! 1. **面点（Face Point）**：每个三角面的重心 $\frac{v_0+v_1+v_2}{3}$
//! 2. **边翻转**：每条原始内部边 $(v_0, v_1)$ 翻转为连接两侧面点 $(c_1, c_2)$，
//!    产生 2 个新三角形；原始边界边保留 fan 三角形 $(v_0, v_1, c)$
//! 3. **顶点松弛**：
//!    - 内部顶点（valence=$n$）：$v' = (1-\alpha_n)\,v + \frac{\alpha_n}{n}\sum \text{neighbors}$，
//!      $\alpha_n = \frac{4 - 2\cos(2\pi/n)}{9}$
//!    - 边界顶点：保持原位置
//!
//! ### 规模变化
//! 设原始网格有 $V$ 顶点、$E$ 边、$F$ 面，细分后：
//! $$V' = V + F, \quad F' = 3F$$
//! 对于闭合三角网格 $E' = 3F$，Euler 示性数 $V'-E'+F' = 2$ 保持不变。
//!
//! ## 与 Loop / Catmull-Clark 的对比
//! | 特性 | Loop | Catmull-Clark | √3 |
//! |------|------|--------------|----|
//! | 面增长倍数 | 4× | 4k→2k（三角化） | 3× |
//! | 输入要求 | 三角形 | 任意多边形 | 三角形 |
//! | 顶点更新 | 逼近型 | 逼近型 | 插值型（松弛后近似） |
//!
//! √3 的 3× 面增长（而非 4×）使其在渐进式细分中提供更细粒度的控制。

use std::collections::{HashMap, HashSet};

use crate::ids::{FaceId, VertexId};
use crate::io::build_mesh_from_vertices_and_faces;
use crate::storage::MeshStorage;
use crate::traversal::{FaceHalfEdges, VertexRing, is_boundary_vertex};

// ============================================================
// 公开 API
// ============================================================

/// 对三角网格执行一次 √3 细分，返回细分后的新网格。
///
/// 输入网格不修改，输出为全新构建的 `MeshStorage`。
///
/// # 算法
/// 1. 为每个三角面计算重心面点；
/// 2. 翻转每条原始内部边（连接相邻面点），边界边保留 fan 三角形；
/// 3. 松弛内部顶点（α_n 权重），边界顶点保持原位置。
///
/// # 规模变化
/// $V' = V + F$，$F' = 3F$。
///
/// # 健壮性
/// - 所有查询通过 `mesh.get_*` + `Option` 链式调用，无效 ID 跳过；
/// - 边界顶点保持原位置；
/// - 内部顶点 valence=0 时保持原位置（孤立顶点）；
/// - 非三角面被跳过。
pub fn sqrt3_subdivide(mesh: &MeshStorage) -> MeshStorage {
    // ---------- 1. 收集原始顶点 ----------
    let orig_v_ids: Vec<VertexId> = mesh.vertex_ids().collect();
    let n_orig = orig_v_ids.len();
    if n_orig == 0 {
        return MeshStorage::new();
    }
    let mut v_index: HashMap<VertexId, u32> = HashMap::new();
    for (i, &v) in orig_v_ids.iter().enumerate() {
        v_index.insert(v, i as u32);
    }

    // ---------- 2. 收集所有三角面 ----------
    let mut orig_faces: Vec<[u32; 3]> = Vec::new();
    let mut face_id_to_idx: HashMap<FaceId, usize> = HashMap::new();
    for f_id in mesh.face_ids() {
        let verts: Vec<u32> = FaceHalfEdges::new(mesh, f_id)
            .filter_map(|he| mesh.get_halfedge(he))
            .map(|h| h.vertex)
            .filter_map(|v| v_index.get(&v).copied())
            .collect();
        if verts.len() == 3 {
            face_id_to_idx.insert(f_id, orig_faces.len());
            orig_faces.push([verts[0], verts[1], verts[2]]);
        }
    }
    let n_faces = orig_faces.len();
    if n_faces == 0 {
        // 无面：仅复制顶点，不产生面
        let positions: Vec<[f64; 3]> = orig_v_ids
            .iter()
            .filter_map(|&v| mesh.get_vertex(v))
            .map(|vt| vt.position)
            .collect();
        return build_mesh_from_vertices_and_faces(&positions, &[]);
    }

    // ---------- 3. 计算面点（重心）----------
    let face_points: Vec<[f64; 3]> = orig_faces
        .iter()
        .map(|face| {
            let [a, b, c] = *face;
            let pa = mesh
                .get_vertex(orig_v_ids[a as usize])
                .map(|v| v.position)
                .unwrap_or([0.0; 3]);
            let pb = mesh
                .get_vertex(orig_v_ids[b as usize])
                .map(|v| v.position)
                .unwrap_or([0.0; 3]);
            let pc = mesh
                .get_vertex(orig_v_ids[c as usize])
                .map(|v| v.position)
                .unwrap_or([0.0; 3]);
            [
                (pa[0] + pb[0] + pc[0]) / 3.0,
                (pa[1] + pb[1] + pc[1]) / 3.0,
                (pa[2] + pb[2] + pc[2]) / 3.0,
            ]
        })
        .collect();

    // ---------- 4. 构建新顶点位置数组 ----------
    // 布局：[原始顶点(松弛后), 面点]
    let face_point_start = n_orig;
    let mut new_positions: Vec<[f64; 3]> = Vec::with_capacity(n_orig + n_faces);
    for &v in &orig_v_ids {
        new_positions.push(mesh.get_vertex(v).map(|vt| vt.position).unwrap_or([0.0; 3]));
    }
    for fp in &face_points {
        new_positions.push(*fp);
    }

    // ---------- 5. 松弛原始顶点（使用原始邻居的旧位置）----------
    for (i, &v) in orig_v_ids.iter().enumerate() {
        let p_old = new_positions[i];
        let new_pos = if is_boundary_vertex(mesh, v) {
            // 边界顶点保持原位置
            p_old
        } else {
            // 内部顶点：v' = (1-α_n)*v + (α_n/n)*Σ(neighbors)
            let neighbors: Vec<[f64; 3]> = VertexRing::new(mesh, v)
                .filter_map(|he| mesh.get_halfedge(he))
                .map(|h| h.vertex)
                .filter_map(|nb| mesh.get_vertex(nb))
                .map(|vt| vt.position)
                .collect();
            let n = neighbors.len();
            if n == 0 {
                p_old
            } else {
                let alpha = sqrt3_alpha(n);
                let n_f = n as f64;
                let mut sum = [0.0; 3];
                for nb in &neighbors {
                    sum[0] += nb[0];
                    sum[1] += nb[1];
                    sum[2] += nb[2];
                }
                [
                    (1.0 - alpha) * p_old[0] + (alpha / n_f) * sum[0],
                    (1.0 - alpha) * p_old[1] + (alpha / n_f) * sum[1],
                    (1.0 - alpha) * p_old[2] + (alpha / n_f) * sum[2],
                ]
            }
        };
        new_positions[i] = new_pos;
    }

    // ---------- 6. 构建新面 ----------
    // 对每条边（去重）：
    //   - 内部边（两侧有面）：翻转 → 2 个三角形
    //     设 h: origin(o) → tip(t), face=f1; twin: tip(t) → origin(o), face=f2
    //     翻转后三角形: (o, fp2, fp1) 和 (t, fp1, fp2)，均 CCW
    //   - 边界边（一侧有面）：保留 fan → 1 个三角形
    //     内部半边 h: origin(o) → tip(t), face=f
    //     fan 三角形: (o, t, fp) CCW
    let mut new_faces: Vec<[u32; 3]> = Vec::with_capacity(n_faces * 3);
    let mut processed: HashSet<(u32, u32)> = HashSet::new();

    for he_id in mesh.halfedge_ids() {
        let h = match mesh.get_halfedge(he_id) {
            Some(h) => h,
            None => continue,
        };
        let v_tip = h.vertex;
        let twin_id = match h.twin {
            Some(t) => t,
            None => continue, // 拓扑断开
        };
        let twin = match mesh.get_halfedge(twin_id) {
            Some(t) => t,
            None => continue,
        };
        let v_origin = twin.vertex;
        let i_o = match v_index.get(&v_origin) {
            Some(&i) => i,
            None => continue,
        };
        let i_t = match v_index.get(&v_tip) {
            Some(&i) => i,
            None => continue,
        };
        if i_o == i_t {
            continue; // 退化边
        }
        let key = edge_key(i_o, i_t);
        if !processed.insert(key) {
            continue; // 已处理
        }

        let face_h = h.face;
        let face_twin = twin.face;
        let fp_h = face_h.and_then(|f| face_id_to_idx.get(&f).copied());
        let fp_twin = face_twin.and_then(|f| face_id_to_idx.get(&f).copied());

        match (fp_h, fp_twin) {
            (Some(f1_idx), Some(f2_idx)) => {
                // 内部边：翻转
                let fp1 = (face_point_start + f1_idx) as u32; // h 所在面点
                let fp2 = (face_point_start + f2_idx) as u32; // twin 所在面点
                new_faces.push([i_o, fp2, fp1]);
                new_faces.push([i_t, fp1, fp2]);
            }
            (Some(f1_idx), None) => {
                // 边界边：h 有面（内部半边），twin 无面
                // fan 三角形 (origin, tip, fp) CCW
                let fp1 = (face_point_start + f1_idx) as u32;
                new_faces.push([i_o, i_t, fp1]);
            }
            (None, Some(f2_idx)) => {
                // 边界边：h 无面（边界半边），twin 有面（内部半边）
                // 内部半边 = twin: origin(i_t) → tip(i_o), face=f2
                // fan 三角形 (i_t, i_o, fp2) CCW
                let fp2 = (face_point_start + f2_idx) as u32;
                new_faces.push([i_t, i_o, fp2]);
            }
            (None, None) => {
                // 两侧均无面，跳过（退化）
            }
        }
    }

    // ---------- 7. 构建新网格 ----------
    build_mesh_from_vertices_and_faces(&new_positions, &new_faces)
}

// ============================================================
// 内部辅助
// ============================================================

/// √3 细分 α 权重：
/// $\alpha_n = \frac{4 - 2\cos(2\pi/n)}{9}$
fn sqrt3_alpha(n: usize) -> f64 {
    if n == 0 {
        return 0.0; // 守卫：除零
    }
    let n_f = n as f64;
    (4.0 - 2.0 * (2.0 * std::f64::consts::PI / n_f).cos()) / 9.0
}

/// 边键：(min, max)，保证无向边唯一表示。
#[inline]
fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::build_icosphere;
    use crate::validate::check_topology;

    // ---------- 规模验证 ----------

    #[test]
    fn sqrt3_icosphere0_vertex_face_count() {
        // icosphere(0): V=12, F=20, E=30
        // √3 细分后：V' = V + F = 12 + 20 = 32, F' = 3*F = 60
        let mesh = build_icosphere(0);
        assert_eq!(mesh.vertex_count(), 12);
        assert_eq!(mesh.face_count(), 20);

        let refined = sqrt3_subdivide(&mesh);
        assert_eq!(refined.vertex_count(), 32, "顶点数应为 V+F = 12+20 = 32");
        assert_eq!(refined.face_count(), 60, "面数应为 3*F = 3*20 = 60");
    }

    #[test]
    fn sqrt3_icosphere1_vertex_face_count() {
        // icosphere(1): V=42, F=80, E=120
        // √3 细分后：V' = 42+80 = 122, F' = 3*80 = 240
        let mesh = build_icosphere(1);
        let refined = sqrt3_subdivide(&mesh);
        assert_eq!(refined.vertex_count(), 122, "顶点数应为 42+80=122");
        assert_eq!(refined.face_count(), 240, "面数应为 3*80=240");
    }

    // ---------- 拓扑校验 ----------

    #[test]
    fn sqrt3_icosphere0_passes_validation() {
        let mesh = build_icosphere(0);
        let refined = sqrt3_subdivide(&mesh);
        assert!(
            check_topology(&refined).is_ok(),
            "细分后的网格应通过完整拓扑校验: {:?}",
            check_topology(&refined)
        );
    }

    #[test]
    fn sqrt3_icosphere2_passes_validation() {
        let mesh = build_icosphere(2);
        let refined = sqrt3_subdivide(&mesh);
        assert!(check_topology(&refined).is_ok());
    }

    // ---------- Euler 示性数 ----------

    #[test]
    fn sqrt3_preserves_euler_characteristic() {
        // 闭合三角网格的 Euler 示性数 V - E + F = 2 应保持不变
        for n in 0..=2 {
            let mesh = build_icosphere(n);
            let v = mesh.vertex_count() as i64;
            let e = (mesh.halfedge_count() / 2) as i64;
            let f = mesh.face_count() as i64;
            assert_eq!(v - e + f, 2, "细分前 icosphere({}) Euler 示性数应为 2", n);

            let refined = sqrt3_subdivide(&mesh);
            let v2 = refined.vertex_count() as i64;
            let e2 = (refined.halfedge_count() / 2) as i64;
            let f2 = refined.face_count() as i64;
            assert_eq!(
                v2 - e2 + f2,
                2,
                "细分后 icosphere({}) Euler 示性数应保持 2，实际 {}-{}+{}={}",
                n,
                v2,
                e2,
                f2,
                v2 - e2 + f2
            );
        }
    }

    // ---------- 与 Loop 对比 ----------

    #[test]
    fn sqrt3_vs_loop_face_growth() {
        // 同一 icosphere(1)：Loop 得 F'=320（4×），√3 得 F'=240（3×）
        let mesh = build_icosphere(1);
        let loop_refined = crate::subdiv::loop_subdivide(&mesh);
        let sqrt3_refined = sqrt3_subdivide(&mesh);
        assert_eq!(loop_refined.face_count(), 320, "Loop 应为 4*80=320");
        assert_eq!(sqrt3_refined.face_count(), 240, "√3 应为 3*80=240");
    }

    // ---------- 连续细分 ----------

    #[test]
    fn sqrt3_multiple_iterations_stay_valid() {
        // 连续细分 2 次，每次都应通过拓扑校验
        let mut mesh = build_icosphere(0);
        for i in 0..2 {
            mesh = sqrt3_subdivide(&mesh);
            assert!(
                check_topology(&mesh).is_ok(),
                "第 {} 次细分后拓扑校验失败",
                i + 1
            );
        }
        // icosphere(0) 细分 2 次：
        // 1次: V=32, F=60
        // 2次: V=32+60=92, F=3*60=180
        assert_eq!(mesh.vertex_count(), 92);
        assert_eq!(mesh.face_count(), 180);
    }

    // ---------- 边界网格 ----------

    #[test]
    fn sqrt3_single_triangle() {
        // 单三角面片（全是边界边）
        // V=3, F=1 → V'=3+1=4, F'=3（3 条边界边各 1 个 fan 三角形）
        let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = [[0, 1, 2]];
        let mesh = build_mesh_from_vertices_and_faces(&vertices, &faces);
        let refined = sqrt3_subdivide(&mesh);
        assert_eq!(refined.vertex_count(), 4, "单三角形 V'=3+1=4");
        assert_eq!(refined.face_count(), 3, "单三角形 F'=3");
        assert!(check_topology(&refined).is_ok());
    }

    #[test]
    fn sqrt3_open_quad() {
        // 两个三角形拼成的开四边形
        // V=4, F=2, 内部边 1 条, 边界边 4 条
        // V' = 4+2 = 6
        // F' = 2*1(内部边) + 4*1(边界边) = 6
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let faces = [[0, 1, 2], [0, 2, 3]];
        let mesh = build_mesh_from_vertices_and_faces(&vertices, &faces);
        let refined = sqrt3_subdivide(&mesh);
        assert_eq!(refined.vertex_count(), 6, "开四边形 V'=4+2=6");
        assert_eq!(refined.face_count(), 6, "开四边形 F'=2+4=6");
        assert!(check_topology(&refined).is_ok());
    }

    // ---------- 面点位置正确性 ----------

    #[test]
    fn sqrt3_face_point_at_centroid() {
        // 单三角形面点应在重心 (1/3, 1/3, 0)
        let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = [[0, 1, 2]];
        let mesh = build_mesh_from_vertices_and_faces(&vertices, &faces);
        let refined = sqrt3_subdivide(&mesh);

        let target = [1.0 / 3.0, 1.0 / 3.0, 0.0];
        let eps = 1e-9;
        let found = refined
            .vertex_ids()
            .filter_map(|v| refined.get_vertex(v))
            .any(|vt| {
                let p = vt.position;
                (p[0] - target[0]).abs() < eps
                    && (p[1] - target[1]).abs() < eps
                    && (p[2] - target[2]).abs() < eps
            });
        assert!(found, "面点应在重心 (1/3, 1/3, 0)");
    }

    // ---------- α 权重 ----------

    #[test]
    fn sqrt3_alpha_regular_valence_6() {
        // valence=6 的正则情况：α = (4 - 2*cos(π/3))/9 = (4-1)/9 = 1/3
        let alpha = sqrt3_alpha(6);
        assert!(
            (alpha - 1.0 / 3.0).abs() < 1e-12,
            "α(6) 应为 1/3，实际 {}",
            alpha
        );
    }

    #[test]
    fn sqrt3_alpha_valence_3() {
        // valence=3：α = (4 - 2*cos(2π/3))/9 = (4 - 2*(-1/2))/9 = 5/9
        let alpha = sqrt3_alpha(3);
        assert!(
            (alpha - 5.0 / 9.0).abs() < 1e-12,
            "α(3) 应为 5/9，实际 {}",
            alpha
        );
    }

    #[test]
    fn sqrt3_alpha_zero_valence_returns_zero() {
        // 守卫：n=0 不应 panic
        assert_eq!(sqrt3_alpha(0), 0.0);
    }

    // ---------- 空网格 ----------

    #[test]
    fn sqrt3_empty_mesh() {
        let mesh = MeshStorage::new();
        let refined = sqrt3_subdivide(&mesh);
        assert_eq!(refined.vertex_count(), 0);
        assert_eq!(refined.face_count(), 0);
    }

    // ---------- 内部顶点松弛验证 ----------

    #[test]
    fn sqrt3_internal_vertex_relaxation() {
        // 四面体（闭合，所有顶点 valence=3，所有边内部）
        // α_3 = 5/9, v' = (4/9)*v + (5/27)*Σ(neighbors)
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        let mesh = build_mesh_from_vertices_and_faces(&vertices, &faces);
        let refined = sqrt3_subdivide(&mesh);

        // 顶点 v0=(0,0,0) 的 3 个邻居：v1=(1,0,0), v2=(0,1,0), v3=(0,0,1)
        // Σ = (1,1,1)
        // v' = (4/9)*(0,0,0) + (5/27)*(1,1,1) = (5/27, 5/27, 5/27)
        let eps = 1e-9;
        let target = [5.0 / 27.0; 3];
        let found = refined
            .vertex_ids()
            .filter_map(|v| refined.get_vertex(v))
            .any(|vt| {
                let p = vt.position;
                (p[0] - target[0]).abs() < eps
                    && (p[1] - target[1]).abs() < eps
                    && (p[2] - target[2]).abs() < eps
            });
        assert!(found, "v0 松弛后应在 (5/27, 5/27, 5/27)，未找到该位置");
    }
}
