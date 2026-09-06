//! 离散曲率（Meyer et al. 2003）
//!
//! 提供顶点曲率计算：
//! - 高斯曲率：[`gaussian_curvature`]
//! - 平均曲率：[`mean_curvature`]
//! - 主曲率：[`principal_curvatures`]
//! - 完整曲率信息：[`vertex_curvature`], [`VertexCurvature`]
//! - 并行计算：[`all_gaussian_curvatures_par`], [`all_mean_curvatures_par`]

use crate::Scalar;
use crate::ids::VertexId;
use crate::storage::MeshStorage;
use crate::traversal::{VertexRing, is_boundary_vertex};

/// 顶点曲率信息。
#[derive(Debug, Clone, Copy)]
pub struct VertexCurvature {
    /// 高斯曲率 $K = \kappa_1 \cdot \kappa_2$
    pub gaussian: Scalar,
    /// 平均曲率 $H = (\kappa_1 + \kappa_2) / 2$
    pub mean: Scalar,
    /// 最大主曲率 $\kappa_1$
    pub k1: Scalar,
    /// 最小主曲率 $\kappa_2$
    pub k2: Scalar,
}

/// 计算混合面积（Voronoi 面积，用于曲率归一化）。
/// 对非钝角三角形用 Voronoi 面积；钝角三角形退化时用重心面积。
fn mixed_area_at_vertex(mesh: &MeshStorage, v: VertexId) -> f64 {
    let mut area = 0.0;
    for he in VertexRing::new(mesh, v) {
        // 拓扑不一致时跳过，避免 panic
        let h = match mesh.get_halfedge(he) {
            Some(h) => h,
            None => continue,
        };
        // 每个入射三角形为 (v, a, b)：he 是 v→a 的出边，b 是该三角形中 v 的另一个邻居。
        let a = h.vertex; // tip = 邻居 a
        let b = match h.next.and_then(|n| mesh.get_halfedge(n)) {
            Some(n) => n.vertex, // 邻居 b（he.next 的 tip）
            None => continue,
        };

        let (pv, pa, pb) = match (mesh.get_vertex(v), mesh.get_vertex(a), mesh.get_vertex(b)) {
            (Some(vv), Some(va), Some(vb)) => (vv.position, va.position, vb.position),
            _ => continue,
        };

        // 三边平方长度：va = |a-v|²，vb = |b-v|²，ab = |a-b|²
        let va2 = (pa[0] - pv[0]).powi(2) + (pa[1] - pv[1]).powi(2) + (pa[2] - pv[2]).powi(2);
        let vb2 = (pb[0] - pv[0]).powi(2) + (pb[1] - pv[1]).powi(2) + (pb[2] - pv[2]).powi(2);
        let ab2 = (pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2);

        // 判断钝角：钝角在 v/a/b 处时，其对边分别为 ab/vb/va
        let obtuse_at_v = ab2 > va2 + vb2;
        let obtuse_at_a = vb2 > va2 + ab2;
        let obtuse_at_b = va2 > vb2 + ab2;

        let tri_area = crate::linalg::vec3::triangle_area(pv, pa, pb);
        if obtuse_at_v {
            // v 处钝角：用三角形面积的 1/2
            area += tri_area / 2.0;
        } else if obtuse_at_a || obtuse_at_b {
            // 其他钝角：用三角形面积的 1/4
            area += tri_area / 4.0;
        } else {
            // Voronoi 面积：A_voronoi = 1/8 Σ (cot α_ij + cot β_ij) · ||v_j - v_i||²
            // 对三角形 (v, a, b) 贡献 1/8 · (cot(∠b)·|a-v|² + cot(∠a)·|b-v|²)
            let cot_a = cotan_from_pos(pa, pv, pb); // 角在 a
            let cot_b = cotan_from_pos(pb, pv, pa); // 角在 b
            area += (va2 * cot_b + vb2 * cot_a) / 8.0;
        }
    }
    if area < 1e-14 { 1e-14 } else { area }
}

fn cotan_from_pos(o: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let oa = [a[0] - o[0], a[1] - o[1], a[2] - o[2]];
    let ob = [b[0] - o[0], b[1] - o[1], b[2] - o[2]];
    let dot = oa[0] * ob[0] + oa[1] * ob[1] + oa[2] * ob[2];
    let cross = [
        oa[1] * ob[2] - oa[2] * ob[1],
        oa[2] * ob[0] - oa[0] * ob[2],
        oa[0] * ob[1] - oa[1] * ob[0],
    ];
    let cross_len = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    if cross_len < 1e-14 {
        0.0
    } else {
        dot / cross_len
    }
}

/// 顶点高斯曲率（离散微分几何，Meyer et al. 2003）。
///
/// $$
/// K(v) = \frac{2\pi - \sum_j \theta_j}{A_{\text{mixed}}(v)}
/// $$
///
/// 其中 $\theta_j$ 是顶点 v 处各三角形内角，
/// $A_{\text{mixed}}$ 为混合面积。边界顶点返回 0。
pub fn gaussian_curvature(mesh: &MeshStorage, v: VertexId) -> Option<f64> {
    if !mesh.contains_vertex(v) {
        return None;
    }
    if is_boundary_vertex(mesh, v) {
        return Some(0.0);
    }

    let mut angle_sum = 0.0;
    for he in VertexRing::new(mesh, v) {
        let h = mesh.get_halfedge(he)?;
        let a = h.vertex; // 邻居 a（he 的 tip）
        let b = h.next.and_then(|n| mesh.get_halfedge(n))?.vertex; // 邻居 b（同三角形中 v 的另一个邻居）

        let pv = mesh.get_vertex(v)?.position;
        let pa = mesh.get_vertex(a)?.position;
        let pb = mesh.get_vertex(b)?.position;

        let oa = [pa[0] - pv[0], pa[1] - pv[1], pa[2] - pv[2]];
        let ob = [pb[0] - pv[0], pb[1] - pv[1], pb[2] - pv[2]];
        let dot = oa[0] * ob[0] + oa[1] * ob[1] + oa[2] * ob[2];
        let oa_len = (oa[0] * oa[0] + oa[1] * oa[1] + oa[2] * oa[2]).sqrt();
        let ob_len = (ob[0] * ob[0] + ob[1] * ob[1] + ob[2] * ob[2]).sqrt();
        if oa_len < 1e-14 || ob_len < 1e-14 {
            continue;
        }
        let cos = (dot / (oa_len * ob_len)).clamp(-1.0, 1.0);
        angle_sum += cos.acos();
    }

    let area = mixed_area_at_vertex(mesh, v);
    Some((std::f64::consts::TAU - angle_sum) / area)
}

/// 顶点平均曲率（离散微分几何，Meyer et al. 2003）。
///
/// $$
/// H(v) = \frac{\| \sum_j (\cot \alpha_j + \cot \beta_j) (v_j - v) \|}{4 \cdot A_{\text{mixed}}(v)}
/// $$
///
/// 边界顶点返回 0。
pub fn mean_curvature(mesh: &MeshStorage, v: VertexId) -> Option<f64> {
    if !mesh.contains_vertex(v) {
        return None;
    }
    if is_boundary_vertex(mesh, v) {
        return Some(0.0);
    }

    let laplacian = super::query::cotan_laplacian(mesh, v)?;
    let len =
        (laplacian[0] * laplacian[0] + laplacian[1] * laplacian[1] + laplacian[2] * laplacian[2])
            .sqrt();
    if len < 1e-14 {
        return Some(0.0);
    }
    let area = mixed_area_at_vertex(mesh, v);
    // H = |Δv| / (4 A_mixed)：|Δv| = Σ(cot α + cot β)(v_j - v) 的模。
    Some(len / (4.0 * area))
}

/// 顶点主曲率 $\kappa_1, \kappa_2$。
///
/// 从高斯曲率 K 和平均曲率 H 推导：
/// $$
/// \kappa_{1,2} = H \pm \sqrt{H^2 - K}
/// $$
///
/// 若 $H^2 - K < 0$（数值误差），取 $\kappa_1 = \kappa_2 = H$。
pub fn principal_curvatures(mesh: &MeshStorage, v: VertexId) -> Option<(f64, f64)> {
    let g = gaussian_curvature(mesh, v)?;
    let h = mean_curvature(mesh, v)?;
    let disc = h * h - g;
    if disc < 0.0 {
        // 数值误差，返回 H, H
        Some((h, h))
    } else {
        let sqrt_disc = disc.sqrt();
        Some((h + sqrt_disc, h - sqrt_disc))
    }
}

/// 返回顶点的完整曲率信息（高斯、平均、主曲率）。
pub fn vertex_curvature(mesh: &MeshStorage, v: VertexId) -> Option<VertexCurvature> {
    let g = gaussian_curvature(mesh, v)?;
    let h = mean_curvature(mesh, v)?;
    let (k1, k2) = principal_curvatures(mesh, v)?;
    Some(VertexCurvature {
        gaussian: g,
        mean: h,
        k1,
        k2,
    })
}

// ============================================================
// 并行变体（rayon）
// ============================================================

/// 并行计算所有顶点的高斯曲率。
pub fn all_gaussian_curvatures_par(mesh: &MeshStorage) -> Vec<Option<f64>> {
    use rayon::prelude::*;
    let verts: Vec<VertexId> = mesh.vertex_ids().collect();
    verts
        .par_iter()
        .map(|&v| gaussian_curvature(mesh, v))
        .collect()
}

/// 并行计算所有顶点的平均曲率。
pub fn all_mean_curvatures_par(mesh: &MeshStorage) -> Vec<Option<f64>> {
    use rayon::prelude::*;
    let verts: Vec<VertexId> = mesh.vertex_ids().collect();
    verts.par_iter().map(|&v| mean_curvature(mesh, v)).collect()
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_curvature_unit_sphere() {
        // 单位 icosphere 的理论高斯曲率 K = 1/r² = 1。
        // 修复前 mixed_area_at_vertex 恒返回 1e-14，导致 K ≈ 2π/1e-14 异常巨大。
        let mesh = crate::test_util::build_icosphere(3);
        for v in mesh.vertex_ids() {
            if let Some(k) = gaussian_curvature(&mesh, v) {
                assert!(k.is_finite(), "高斯曲率非有限");
                assert!(k > 0.1 && k < 10.0, "单位球高斯曲率应≈1，实际 {k}");
            }
        }
    }

    #[test]
    fn mean_curvature_unit_sphere() {
        // 单位 icosphere 的理论平均曲率 H = 1/r = 1。
        // 修复前少了 1/2 因子，返回约 2。
        let mesh = crate::test_util::build_icosphere(3);
        for v in mesh.vertex_ids() {
            if let Some(h) = mean_curvature(&mesh, v) {
                assert!(h.is_finite(), "平均曲率非有限");
                assert!(h > 0.5 && h < 1.5, "单位球平均曲率应≈1，实际 {h}");
            }
        }
    }

    #[test]
    fn principal_curvatures_unit_sphere() {
        // 单位球主曲率 κ1 = κ2 = 1。
        let mesh = crate::test_util::build_icosphere(3);
        for v in mesh.vertex_ids() {
            if let Some((k1, k2)) = principal_curvatures(&mesh, v) {
                assert!(k1.is_finite() && k2.is_finite(), "主曲率非有限");
                assert!(k1 > 0.5 && k1 < 1.5, "单位球 κ1 应≈1，实际 {k1}");
                assert!(k2 > 0.5 && k2 < 1.5, "单位球 κ2 应≈1，实际 {k2}");
            }
        }
    }

    #[test]
    fn gaussian_curvature_icosphere() {
        let mesh = crate::test_util::build_icosphere(2);
        let mut found = false;
        for v in mesh.vertex_ids() {
            if let Some(k) = gaussian_curvature(&mesh, v)
                && k.is_finite()
                && k > 0.0
            {
                found = true;
                break;
            }
        }
        assert!(found, "应有正高斯曲率");
    }

    #[test]
    fn mean_curvature_icosphere() {
        let mesh = crate::test_util::build_icosphere(2);
        let mut found = false;
        for v in mesh.vertex_ids() {
            if let Some(h) = mean_curvature(&mesh, v)
                && h.is_finite()
                && h > 0.0
            {
                found = true;
                break;
            }
        }
        assert!(found, "应有正平均曲率");
    }

    #[test]
    fn principal_curvatures_nonzero() {
        let mesh = crate::test_util::build_icosphere(2);
        let mut found = false;
        for v in mesh.vertex_ids() {
            if let Some((k1, k2)) = principal_curvatures(&mesh, v)
                && k1.is_finite()
                && k2.is_finite()
                && k1 > 0.0
            {
                found = true;
                break;
            }
        }
        assert!(found, "应有正主曲率");
    }

    #[test]
    fn vertex_curvature_struct() {
        let mesh = crate::test_util::build_icosphere(2);
        let mut tested = false;
        for v in mesh.vertex_ids() {
            if let Some(c) = vertex_curvature(&mesh, v)
                && c.k1.is_finite()
                && c.k2.is_finite()
            {
                // 主曲率顺序
                assert!(c.k1 >= c.k2 - 1e-10);
                tested = true;
                break;
            }
        }
        assert!(tested, "至少有一个顶点的有效曲率");
    }

    #[test]
    fn all_gaussian_curvatures_par_matches_serial() {
        let mesh = crate::test_util::build_icosphere(1);
        let par = all_gaussian_curvatures_par(&mesh);
        let serial: Vec<Option<f64>> = mesh
            .vertex_ids()
            .map(|v| gaussian_curvature(&mesh, v))
            .collect();
        assert_eq!(par.len(), serial.len());
        for (i, (a, b)) in par.iter().zip(serial.iter()).enumerate() {
            match (a, b) {
                (Some(x), Some(y)) => assert!(
                    (x - y).abs() < 1e-10,
                    "顶点 {} 高斯曲率不一致: {} vs {}",
                    i,
                    x,
                    y
                ),
                (None, None) => {}
                _ => panic!("顶点 {} 高斯曲率 None 不一致: {:?} vs {:?}", i, a, b),
            }
        }
    }

    #[test]
    fn all_mean_curvatures_par_matches_serial() {
        let mesh = crate::test_util::build_icosphere(1);
        let par = all_mean_curvatures_par(&mesh);
        let serial: Vec<Option<f64>> = mesh
            .vertex_ids()
            .map(|v| mean_curvature(&mesh, v))
            .collect();
        assert_eq!(par.len(), serial.len());
        for (i, (a, b)) in par.iter().zip(serial.iter()).enumerate() {
            match (a, b) {
                (Some(x), Some(y)) => assert!(
                    (x - y).abs() < 1e-10,
                    "顶点 {} 平均曲率不一致: {} vs {}",
                    i,
                    x,
                    y
                ),
                (None, None) => {}
                _ => panic!("顶点 {} 平均曲率 None 不一致: {:?} vs {:?}", i, a, b),
            }
        }
    }
}
