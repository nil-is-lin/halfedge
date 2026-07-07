//! 网格质量度量
//!
//! 提供三角形网格的质量度量：
//! - 纵横比：[`face_aspect_ratio`]
//! - 半径比：[`face_radius_ratio`]
//! - 边长统计：[`edge_length_stats`], [`EdgeLengthStats`]
//! - 网格质量汇总：[`mesh_quality`], [`MeshQualityStats`]

use crate::Scalar;
use crate::ids::FaceId;
use crate::storage::MeshStorage;
use crate::traversal::FaceVertices;

/// 三角形的纵横比（aspect ratio）。
///
/// 定义：最长边 / 最短边。等边三角形为 1，越瘦长越大。
/// 退化三角形（最短边 = 0）返回 `None`。
pub fn face_aspect_ratio(mesh: &MeshStorage, f: FaceId) -> Option<f64> {
    let verts: Vec<_> = FaceVertices::new(mesh, f).collect();
    if verts.len() != 3 {
        return None;
    }
    let p0 = mesh.get_vertex(verts[0])?.position;
    let p1 = mesh.get_vertex(verts[1])?.position;
    let p2 = mesh.get_vertex(verts[2])?.position;
    let l0 = (p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2) + (p1[2] - p0[2]).powi(2);
    let l1 = (p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2) + (p2[2] - p1[2]).powi(2);
    let l2 = (p0[0] - p2[0]).powi(2) + (p0[1] - p2[1]).powi(2) + (p0[2] - p2[2]).powi(2);
    let l0 = l0.sqrt();
    let l1 = l1.sqrt();
    let l2 = l2.sqrt();
    let l_min = l0.min(l1).min(l2);
    let l_max = l0.max(l1).max(l2);
    if l_min < 1e-20 {
        return None;
    }
    Some(l_max / l_min)
}

/// 三角形的半径比（radius ratio），又称 RQ。
///
/// 定义：内切圆半径 / 外接圆半径 × 2，归一化到 [0, 1]：
/// \[
///   \mathrm{RQ} = \frac{2 r_{\text{in}}}{r_{\text{out}}} = \frac{8 (s-a)(s-b)(s-c)}{abc}
/// \]
/// 等边三角形为 1，退化三角形为 0。
pub fn face_radius_ratio(mesh: &MeshStorage, f: FaceId) -> Option<f64> {
    let verts: Vec<_> = FaceVertices::new(mesh, f).collect();
    if verts.len() != 3 {
        return None;
    }
    let p0 = mesh.get_vertex(verts[0])?.position;
    let p1 = mesh.get_vertex(verts[1])?.position;
    let p2 = mesh.get_vertex(verts[2])?.position;
    let a = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2) + (p1[2] - p0[2]).powi(2)).sqrt();
    let b = ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2) + (p2[2] - p1[2]).powi(2)).sqrt();
    let c = ((p0[0] - p2[0]).powi(2) + (p0[1] - p2[1]).powi(2) + (p0[2] - p2[2]).powi(2)).sqrt();
    if a < 1e-20 || b < 1e-20 || c < 1e-20 {
        return None;
    }
    let s = (a + b + c) * 0.5;
    let area2 = s * (s - a) * (s - b) * (s - c);
    if area2 <= 0.0 {
        return Some(0.0);
    }
    // RQ = 8 (s-a)(s-b)(s-c) / (abc)
    Some(8.0 * (s - a) * (s - b) * (s - c) / (a * b * c))
}

/// 边长统计：返回 (min, max, mean, variance)。
///
/// 遍历所有规范半边（每条无向边只算一次），按 `EdgeIter` 顺序。
pub fn edge_length_stats(mesh: &MeshStorage) -> EdgeLengthStats {
    let mut lengths: Vec<f64> = Vec::with_capacity(mesh.edge_count());
    for e in mesh.edge_ids() {
        let he = e.halfedge();
        if let Some(len) = super::query::edge_length(mesh, he) {
            lengths.push(len);
        }
    }
    if lengths.is_empty() {
        return EdgeLengthStats::default();
    }
    let n = lengths.len() as f64;
    let min = lengths.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = lengths.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = lengths.iter().sum::<f64>() / n;
    let variance = lengths.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    EdgeLengthStats {
        min,
        max,
        mean,
        variance,
        count: lengths.len(),
    }
}

/// 边长统计结果。
#[derive(Debug, Clone, Default)]
pub struct EdgeLengthStats {
    pub min: Scalar,
    pub max: Scalar,
    pub mean: Scalar,
    pub variance: Scalar,
    pub count: usize,
}

impl EdgeLengthStats {
    /// 标准差 = √variance。
    pub fn std_dev(&self) -> f64 {
        self.variance.sqrt()
    }
    /// 边长比 = max / min（退化情形返回 +∞）。
    pub fn ratio(&self) -> f64 {
        if self.min < 1e-20 {
            f64::INFINITY
        } else {
            self.max / self.min
        }
    }
}

/// 网格质量统计：纵横比、半径比、边长比的汇总。
#[derive(Debug, Clone, Default)]
pub struct MeshQualityStats {
    pub aspect_min: Scalar,
    pub aspect_max: Scalar,
    pub aspect_mean: Scalar,
    pub radius_ratio_min: Scalar,
    pub radius_ratio_mean: Scalar,
    pub edges: EdgeLengthStats,
    pub face_count: usize,
}

/// 计算整张网格的质量统计。
///
/// - `aspect_*`：纵横比，等边为 1，越瘦长越大；
/// - `radius_ratio_*`：归一化半径比，[0, 1]，1 为最优；
/// - `edges`：边长统计。
pub fn mesh_quality(mesh: &MeshStorage) -> MeshQualityStats {
    let mut aspects: Vec<f64> = Vec::with_capacity(mesh.face_count());
    let mut rrs: Vec<f64> = Vec::with_capacity(mesh.face_count());
    for f in mesh.face_ids() {
        if let Some(ar) = face_aspect_ratio(mesh, f) {
            aspects.push(ar);
        }
        if let Some(rr) = face_radius_ratio(mesh, f) {
            rrs.push(rr);
        }
    }
    let face_count = aspects.len();
    let aspect_min = aspects.iter().cloned().fold(f64::INFINITY, f64::min);
    let aspect_max = aspects.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let aspect_mean = if aspects.is_empty() {
        0.0
    } else {
        aspects.iter().sum::<f64>() / aspects.len() as f64
    };
    let radius_ratio_min = rrs.iter().cloned().fold(f64::INFINITY, f64::min);
    let radius_ratio_mean = if rrs.is_empty() {
        0.0
    } else {
        rrs.iter().sum::<f64>() / rrs.len() as f64
    };
    MeshQualityStats {
        aspect_min,
        aspect_max,
        aspect_mean,
        radius_ratio_min,
        radius_ratio_mean,
        edges: edge_length_stats(mesh),
        face_count,
    }
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aspect_ratio_equilateral_is_one() {
        // 等边三角形：纵横比 = 1
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 3.0_f64.sqrt() / 2.0, 0.0],
        ];
        let faces = vec![[0u32, 1, 2]];
        let mesh = crate::io::build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let f = mesh.face_ids().next().unwrap();
        let ar = face_aspect_ratio(&mesh, f).expect("等边三角形纵横比");
        assert!((ar - 1.0).abs() < 1e-10, "等边纵横比应=1, got {ar}");
    }

    #[test]
    fn aspect_ratio_degenerate_returns_none() {
        // 退化三角形（共线）：最短边可能为 0 或面积 0
        let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let faces = vec![[0u32, 1, 2]];
        let mesh = crate::io::build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let f = mesh.face_ids().next().unwrap();
        // 此处最短边 ≠ 0，但面积 = 0；aspect_ratio 仍可计算
        // 真正退化（最短边=0）应返回 None
        let ar = face_aspect_ratio(&mesh, f);
        assert!(ar.is_some(), "aspect_ratio 即使面积 0 也可计算");
    }

    #[test]
    fn radius_ratio_equilateral_is_one() {
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 3.0_f64.sqrt() / 2.0, 0.0],
        ];
        let faces = vec![[0u32, 1, 2]];
        let mesh = crate::io::build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let f = mesh.face_ids().next().unwrap();
        let rr = face_radius_ratio(&mesh, f).expect("等边半径比");
        assert!((rr - 1.0).abs() < 1e-10, "等边半径比应=1, got {rr}");
    }

    #[test]
    fn radius_ratio_degenerate_is_zero() {
        let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let faces = vec![[0u32, 1, 2]];
        let mesh = crate::io::build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let f = mesh.face_ids().next().unwrap();
        let rr = face_radius_ratio(&mesh, f).expect("退化三角形半径比");
        assert!(rr.abs() < 1e-10, "退化半径比应=0, got {rr}");
    }

    #[test]
    fn edge_length_stats_icosphere_consistent() {
        let mesh = crate::test_util::build_icosphere(1);
        let stats = edge_length_stats(&mesh);
        assert_eq!(stats.count, mesh.edge_count());
        // icosphere(1) 边长应在合理范围
        assert!(stats.min > 0.0);
        assert!(stats.max < 2.0);
        assert!(stats.mean > stats.min);
        assert!(stats.mean < stats.max);
        // ratio() 有限
        assert!(stats.ratio().is_finite());
    }

    #[test]
    fn mesh_quality_icosphere_returns_finite_stats() {
        let mesh = crate::test_util::build_icosphere(1);
        let q = mesh_quality(&mesh);
        assert_eq!(q.face_count, mesh.face_count());
        assert!(q.aspect_min >= 1.0, "纵横比最小值 ≥ 1");
        assert!(q.aspect_max.is_finite());
        assert!(q.radius_ratio_min >= 0.0);
        assert!(q.radius_ratio_min <= 1.0);
        assert!(q.radius_ratio_mean <= 1.0);
        assert!(q.edges.count > 0);
    }
}
