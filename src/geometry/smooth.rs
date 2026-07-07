//! 网格平滑
//!
//! 提供网格平滑算法：
//! - 拉普拉斯平滑：[`laplacian_smooth_vertex`], [`laplacian_smooth_mesh`]
//! - Taubin 双向平滑：[`taubin_smooth_mesh`]
//! - 双边网格去噪：[`bilateral_smooth_mesh`]
//! - 并行拉普拉斯平滑：[`laplacian_smooth_mesh_par`]
//! - 并行特征边检测：[`feature_edges_par`]

use crate::ids::{HalfEdgeId, VertexId};
use crate::linalg::vec3::{Vec3, add, scale, sub};
use crate::storage::MeshStorage;
use crate::traversal::VertexAdjacentVerts;

// ============================================================
// 拉普拉斯平滑
// ============================================================

/// 计算顶点 `v` 的拉普拉斯平滑位置（不修改网格）。
///
/// 使用**统一权重**：新位置为邻居顶点位置的算术平均。
/// $$
/// \vec{p}_v' = \frac{1}{|N(v)|} \sum_{u \in N(v)} \vec{p}_u
/// $$
///
/// 孤立顶点（无邻居）返回原位置。顶点无效返回 `None`。
pub fn laplacian_smooth_vertex(mesh: &MeshStorage, v: VertexId) -> Option<Vec3> {
    let p = mesh.get_vertex(v)?.position;
    let neighbors: Vec<Vec3> = VertexAdjacentVerts::new(mesh, v)
        .filter_map(|n| mesh.get_vertex(n).map(|vt| vt.position))
        .collect();
    if neighbors.is_empty() {
        return Some(p);
    }
    let mut sum = [0.0f64; 3];
    for q in &neighbors {
        sum = add(sum, *q);
    }
    Some(scale(sum, 1.0 / neighbors.len() as f64))
}

/// 对整个网格执行 `iterations` 次拉普拉斯平滑。
///
/// 每次迭代使用**显式**更新：
/// $$
/// \vec{p}_v \leftarrow (1 - \lambda)\, \vec{p}_v + \lambda\, \vec{p}_v'
/// $$
///
/// 其中 $\vec{p}_v'$ 是邻居平均位置。$\lambda \in (0, 1]$ 控制平滑强度。
///
/// 实现细节：每次迭代先收集所有新位置到 `Vec`，再批量写回，
/// 避免「先更新的顶点影响后更新的顶点」的顺序依赖。
pub fn laplacian_smooth_mesh(mesh: &mut MeshStorage, lambda: f64, iterations: usize) {
    if lambda <= 0.0 || iterations == 0 {
        return;
    }
    let clamped_lambda = lambda.min(1.0);
    for _ in 0..iterations {
        let new_positions: Vec<(VertexId, Vec3)> = mesh
            .vertex_ids()
            .filter_map(|v| {
                let old_p = mesh.get_vertex(v)?.position;
                let target = laplacian_smooth_vertex(mesh, v)?;
                let blended = add(
                    scale(old_p, 1.0 - clamped_lambda),
                    scale(target, clamped_lambda),
                );
                Some((v, blended))
            })
            .collect();

        for (v, p) in new_positions {
            if let Some(vt) = mesh.get_vertex_mut(v) {
                vt.position = p;
            }
        }
    }
}

// ============================================================
// 高级平滑
// ============================================================

/// Taubin 双向平滑（λ/μ 双滤波）。
///
/// 每次迭代执行两步：
/// 1. 正向平滑：$p \gets p + \lambda \,\Delta p$（收缩）；
/// 2. 反向平滑：$p \gets p + \mu \,\Delta p$（膨胀）。
///
/// 经典取值：$\lambda = 0.5, \mu = -0.53$（满足 $\lambda + \mu < 0$ 且
/// $|\mu| > \lambda$）。优点：与 Laplacian 不同，Taubin 不会持续收缩，
/// 可在多次迭代后保持网格体积。
pub fn taubin_smooth_mesh(mesh: &mut MeshStorage, lambda: f64, mu: f64, iterations: usize) {
    if iterations == 0 || lambda <= 0.0 || mu >= 0.0 {
        return;
    }
    let lambda = lambda.min(1.0);
    let mu = mu.max(-1.0);
    for _ in 0..iterations {
        // 正向（收缩）：lambda > 0
        laplacian_step_signed(mesh, lambda);
        // 反向（膨胀）：mu < 0
        laplacian_step_signed(mesh, mu);
    }
}

/// 单步拉普拉斯平滑（允许负 lambda，Taubin 反向步用）。
fn laplacian_step_signed(mesh: &mut MeshStorage, lambda: f64) {
    let new_positions: Vec<(VertexId, Vec3)> = mesh
        .vertex_ids()
        .filter_map(|v| {
            let old_p = mesh.get_vertex(v)?.position;
            let target = laplacian_smooth_vertex(mesh, v)?;
            let blended = add(scale(old_p, 1.0 - lambda), scale(target, lambda));
            Some((v, blended))
        })
        .collect();
    for (v, p) in new_positions {
        if let Some(vt) = mesh.get_vertex_mut(v) {
            vt.position = p;
        }
    }
}

/// 双边网格去噪（bilateral mesh denoising）。
///
/// 顶点更新：
/// $$
///   p_i \gets p_i + \frac{1}{W} \sum_{j \in N(i)}
///      \frac{\|p_j - p_i\|}{\sigma_c}
///      \cdot e^{-\|p_j - p_i\|^2 / (2\sigma_c^2)}
///      \cdot e^{-\|n_i - n_j\|^2 / (2\sigma_s^2)}
///      \cdot (p_j - p_i)
/// $$
/// 其中 $\sigma_c$ 为空间权重（建议取平均边长），$\sigma_s$ 为法向权重（建议 0.1）。
///
/// 与 Laplacian 不同，双边滤波保留特征边（法向差异大的邻居贡献小）。
pub fn bilateral_smooth_mesh(
    mesh: &mut MeshStorage,
    sigma_c: f64,
    sigma_s: f64,
    iterations: usize,
) {
    if iterations == 0 || sigma_c <= 0.0 || sigma_s <= 0.0 {
        return;
    }
    let sigma_c2 = 2.0 * sigma_c * sigma_c;
    let sigma_s2 = 2.0 * sigma_s * sigma_s;
    for _ in 0..iterations {
        // 预计算所有顶点法向（避免循环中重复）
        let normals: std::collections::HashMap<VertexId, Vec3> = mesh
            .vertex_ids()
            .filter_map(|v| super::query::vertex_normal(mesh, v).map(|n| (v, n)))
            .collect();
        let verts: Vec<VertexId> = mesh.vertex_ids().collect();
        let updates: Vec<(VertexId, Vec3)> = verts
            .iter()
            .filter_map(|&v| {
                let p_i = mesh.get_vertex(v)?.position;
                let n_i = normals.get(&v).copied().unwrap_or([0.0; 3]);
                let mut sum_disp = [0.0; 3];
                let mut sum_w = 0.0;
                for n in VertexAdjacentVerts::new(mesh, v) {
                    if n == v {
                        continue;
                    }
                    let p_j = mesh.get_vertex(n)?.position;
                    let n_j = normals.get(&n).copied().unwrap_or([0.0; 3]);
                    let d = sub(p_j, p_i);
                    let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    if dist < 1e-20 {
                        continue;
                    }
                    let w_c = (-dist * dist / sigma_c2).exp();
                    let n_diff = sub(n_i, n_j);
                    let n_dot = n_diff[0].powi(2) + n_diff[1].powi(2) + n_diff[2].powi(2);
                    let w_s = (-n_dot / sigma_s2).exp();
                    let w = w_c * w_s;
                    sum_disp[0] += w * d[0];
                    sum_disp[1] += w * d[1];
                    sum_disp[2] += w * d[2];
                    sum_w += w;
                }
                if sum_w < 1e-20 {
                    return None;
                }
                let delta = scale(sum_disp, 1.0 / sum_w);
                Some((v, add(p_i, delta)))
            })
            .collect();
        for (v, p) in updates {
            if let Some(vt) = mesh.get_vertex_mut(v) {
                vt.position = p;
            }
        }
    }
}

// ============================================================
// 并行变体（rayon）
// ============================================================

/// 并行拉普拉斯平滑（每迭代内 rayon 并行计算新位置）。
pub fn laplacian_smooth_mesh_par(mesh: &mut MeshStorage, lambda: f64, iterations: usize) {
    use rayon::prelude::*;
    if lambda <= 0.0 || iterations == 0 {
        return;
    }
    let clamped_lambda = lambda.min(1.0);
    for _ in 0..iterations {
        let verts: Vec<VertexId> = mesh.vertex_ids().collect();
        let new_positions: Vec<(VertexId, Vec3)> = verts
            .par_iter()
            .filter_map(|&v| {
                let old_p = mesh.get_vertex(v)?.position;
                let target = laplacian_smooth_vertex(mesh, v)?;
                let blended = add(
                    scale(old_p, 1.0 - clamped_lambda),
                    scale(target, clamped_lambda),
                );
                Some((v, blended))
            })
            .collect();
        for (v, p) in new_positions {
            if let Some(vt) = mesh.get_vertex_mut(v) {
                vt.position = p;
            }
        }
    }
}

/// 并行检测所有特征边。
pub fn feature_edges_par(mesh: &MeshStorage, angle_threshold: f64) -> Vec<HalfEdgeId> {
    use rayon::prelude::*;
    let hes: Vec<HalfEdgeId> = mesh.halfedge_ids().collect();
    hes.par_iter()
        .filter(|&&he| super::query::is_feature_edge(mesh, he, angle_threshold).unwrap_or(false))
        .copied()
        .collect()
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::VertexId;
    use crate::storage::{Face, HalfEdge, Vertex};

    /// 构造单位等腰直角三角形（在 xy 平面，CCW 朝向 +z）：
    /// A=(0,0,0), B=(1,0,0), C=(0,1,0)
    fn build_unit_triangle() -> (MeshStorage, [VertexId; 3], crate::ids::FaceId) {
        let mut mesh = MeshStorage::new();
        let a = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let b = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let c = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));

        let h_ab = mesh.add_halfedge(HalfEdge::new(b)); // A→B
        let h_bc = mesh.add_halfedge(HalfEdge::new(c)); // B→C
        let h_ca = mesh.add_halfedge(HalfEdge::new(a)); // C→A
        let t_ab = mesh.add_halfedge(HalfEdge::new(a)); // B→A
        let t_bc = mesh.add_halfedge(HalfEdge::new(b)); // C→B
        let t_ca = mesh.add_halfedge(HalfEdge::new(c)); // A→C

        let f = mesh.add_face(Face::new());
        for (he, twin, next, prev) in [
            (h_ab, t_ab, h_bc, h_ca),
            (h_bc, t_bc, h_ca, h_ab),
            (h_ca, t_ca, h_ab, h_bc),
        ] {
            let h = mesh.get_halfedge_mut(he).unwrap();
            h.twin = Some(twin);
            h.next = Some(next);
            h.prev = Some(prev);
            h.face = Some(f);
        }
        for (t, he) in [(t_ab, h_ab), (t_bc, h_bc), (t_ca, h_ca)] {
            mesh.get_halfedge_mut(t).unwrap().twin = Some(he);
        }
        mesh.get_vertex_mut(a).unwrap().halfedge = Some(h_ab);
        mesh.get_vertex_mut(b).unwrap().halfedge = Some(h_bc);
        mesh.get_vertex_mut(c).unwrap().halfedge = Some(h_ca);
        mesh.get_face_mut(f).unwrap().halfedge = Some(h_ab);

        (mesh, [a, b, c], f)
    }

    // ---------- 拉普拉斯平滑 ----------

    #[test]
    fn laplacian_smooth_vertex_isolated_returns_original() {
        let mut mesh = MeshStorage::new();
        let v = mesh.add_vertex(Vertex::new([1.0, 2.0, 3.0]));
        // 无任何邻接，应返回原位置
        let p = laplacian_smooth_vertex(&mesh, v).unwrap();
        assert_eq!(p, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn laplacian_smooth_vertex_two_neighbors() {
        let (mesh, v, _f) = build_unit_triangle();
        // 顶点 A 的邻居是 B、C，平均位置应为 (0.5, 0.5, 0)
        let p = laplacian_smooth_vertex(&mesh, v[0]).unwrap();
        assert!((p[0] - 0.5).abs() < 1e-9);
        assert!((p[1] - 0.5).abs() < 1e-9);
        assert!((p[2] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn laplacian_smooth_mesh_preserves_centroid() {
        let (mut mesh, _v, _f) = build_unit_triangle();
        // 原始重心 (1/3, 1/3, 0)
        let original_centroid = [1.0 / 3.0, 1.0 / 3.0, 0.0];
        // lambda=1.0 一轮迭代后：每个顶点变为邻居平均
        // A→(B+C)/2=(0.5,0.5,0), B→(A+C)/2=(0,0.5,0), C→(A+B)/2=(0.5,0,0)
        // 三顶点平均 = (1/3, 1/3, 0) = 原重心（重心被保留）
        laplacian_smooth_mesh(&mut mesh, 1.0, 1);
        let positions: Vec<_> = mesh
            .vertex_ids()
            .map(|v| mesh.get_vertex(v).unwrap().position)
            .collect();
        let new_centroid = [
            positions.iter().map(|p| p[0]).sum::<f64>() / positions.len() as f64,
            positions.iter().map(|p| p[1]).sum::<f64>() / positions.len() as f64,
            positions.iter().map(|p| p[2]).sum::<f64>() / positions.len() as f64,
        ];
        for i in 0..3 {
            assert!(
                (new_centroid[i] - original_centroid[i]).abs() < 1e-9,
                "重心分量 {} 应保留: 实际 {} vs 期望 {}",
                i,
                new_centroid[i],
                original_centroid[i]
            );
        }
    }

    // ---------- 高级平滑 ----------

    #[test]
    fn taubin_smooth_preserves_volume_better_than_laplacian() {
        let mesh0 = crate::test_util::build_icosphere(1);
        let area0 = super::super::query::surface_area(&mesh0);

        // Laplacian 20 步：体积显著缩小
        let mut mesh_lap = crate::test_util::build_icosphere(1);
        laplacian_smooth_mesh(&mut mesh_lap, 0.5, 20);
        let area_lap = super::super::query::surface_area(&mesh_lap);
        let laplacian_shrink = (area0 - area_lap).abs() / area0;

        // Taubin 20 步：体积变化应小于 Laplacian
        let mut mesh_tau = crate::test_util::build_icosphere(1);
        taubin_smooth_mesh(&mut mesh_tau, 0.5, -0.53, 20);
        let area_tau = super::super::query::surface_area(&mesh_tau);
        let taubin_shrink = (area0 - area_tau).abs() / area0;

        assert!(
            taubin_shrink < laplacian_shrink,
            "Taubin 收缩 ({}) 应小于 Laplacian ({})",
            taubin_shrink,
            laplacian_shrink
        );
    }

    #[test]
    fn bilateral_smooth_runs_on_icosphere() {
        let mesh0 = crate::test_util::build_icosphere(1);
        let mut mesh = crate::test_util::build_icosphere(1);
        let stats = super::super::quality::edge_length_stats(&mesh);
        bilateral_smooth_mesh(&mut mesh, stats.mean, 0.1, 3);
        // 双边滤波后网格规模不变
        assert_eq!(mesh.vertex_count(), mesh0.vertex_count());
        assert_eq!(mesh.face_count(), mesh0.face_count());
    }

    #[test]
    fn taubin_smooth_zero_iterations_is_noop() {
        let mut mesh = crate::test_util::build_icosphere(1);
        let p0 = mesh.vertex_ids().next().unwrap();
        let pos_before = mesh.get_vertex(p0).unwrap().position;
        taubin_smooth_mesh(&mut mesh, 0.5, -0.53, 0);
        let pos_after = mesh.get_vertex(p0).unwrap().position;
        assert_eq!(pos_before, pos_after);
    }

    #[test]
    fn bilateral_smooth_zero_iterations_is_noop() {
        let mut mesh = crate::test_util::build_icosphere(1);
        let p0 = mesh.vertex_ids().next().unwrap();
        let pos_before = mesh.get_vertex(p0).unwrap().position;
        bilateral_smooth_mesh(&mut mesh, 0.1, 0.1, 0);
        let pos_after = mesh.get_vertex(p0).unwrap().position;
        assert_eq!(pos_before, pos_after);
    }

    // ---------- 并行函数一致性 ----------

    #[test]
    fn feature_edges_par_matches_serial() {
        let mesh = crate::test_util::build_icosphere(1);
        // icosphere(1) 相邻面法向夹角约 20–40°，取 0.3 rad 阈值可得到非空特征边集
        let threshold = 0.3_f64;
        let mut s = super::super::query::feature_edges(&mesh, threshold);
        let mut p = feature_edges_par(&mesh, threshold);
        s.sort();
        p.sort();
        assert_eq!(s, p, "feature_edges 串/并行结果不一致");
    }
}
