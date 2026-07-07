//! 轴对齐包围盒（AABB）
//!
//! 提供 AABB 计算和相关函数：
//! - [`AABB`] 结构体
//! - 计算网格 AABB：[`mesh_aabb`]
//! - 计算网格质心：[`mesh_centroid`]

use crate::Scalar;
use crate::storage::MeshStorage;

/// 轴对齐包围盒。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min: [Scalar; 3],
    pub max: [Scalar; 3],
}

impl AABB {
    /// 创建空包围盒。
    pub fn new() -> Self {
        Self {
            min: [f64::MAX; 3],
            max: [f64::MIN; 3],
        }
    }

    /// 从点集计算包围盒。
    pub fn from_points(points: &[[f64; 3]]) -> Self {
        let mut aabb = Self::new();
        for p in points {
            aabb.extend(p);
        }
        aabb
    }

    /// 扩展以包含给定点。
    pub fn extend(&mut self, point: &[f64; 3]) {
        for ((&p, min_i), max_i) in point
            .iter()
            .zip(self.min.iter_mut())
            .zip(self.max.iter_mut())
        {
            if p < *min_i {
                *min_i = p;
            }
            if p > *max_i {
                *max_i = p;
            }
        }
    }

    /// 两包围盒的并集。
    pub fn union(&self, other: &AABB) -> AABB {
        let mut result = *self;
        for i in 0..3 {
            if other.min[i] < result.min[i] {
                result.min[i] = other.min[i];
            }
            if other.max[i] > result.max[i] {
                result.max[i] = other.max[i];
            }
        }
        result
    }

    /// 包围盒中心。
    pub fn center(&self) -> [f64; 3] {
        [
            (self.min[0] + self.max[0]) / 2.0,
            (self.min[1] + self.max[1]) / 2.0,
            (self.min[2] + self.max[2]) / 2.0,
        ]
    }

    /// 对角线长度。
    pub fn diagonal(&self) -> f64 {
        let d = [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }

    /// 是否为空（min > max 在任一轴上）。
    pub fn is_empty(&self) -> bool {
        self.min[0] > self.max[0]
    }
}

impl Default for AABB {
    fn default() -> Self {
        Self::new()
    }
}

/// 计算网格的轴对齐包围盒。若无顶点则返回 `None`。
///
/// 使用 SOA 位置缓存（24 字步长连续访问），缓存命中率优于遍历 `Vertex`。
pub fn mesh_aabb(mesh: &MeshStorage) -> Option<AABB> {
    let positions = mesh.positions_dense();
    if positions.is_empty() {
        return None;
    }
    let mut aabb = AABB::new();
    for p in positions {
        aabb.extend(p);
    }
    Some(aabb)
}

/// 计算网格顶点质心（所有顶点位置的算术平均）。若无顶点则返回 `None`。
///
/// 使用 SOA 位置缓存做连续遍历。
pub fn mesh_centroid(mesh: &MeshStorage) -> Option<[f64; 3]> {
    let positions = mesh.positions_dense();
    if positions.is_empty() {
        return None;
    }
    let mut sum = [0.0; 3];
    for p in positions {
        sum[0] += p[0];
        sum[1] += p[1];
        sum[2] += p[2];
    }
    let n = positions.len() as f64;
    Some([sum[0] / n, sum[1] / n, sum[2] / n])
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::VertexId;
    use crate::storage::{Face, HalfEdge, Vertex};

    #[test]
    fn aabb_unit_triangle() {
        let (mesh, _v, _f) = build_unit_triangle();
        let aabb = mesh_aabb(&mesh).unwrap();
        assert!(!aabb.is_empty());
        assert!((aabb.min[0] - 0.0).abs() < 1e-12);
        assert!((aabb.max[0] - 1.0).abs() < 1e-12);
        assert!((aabb.max[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn aabb_empty_mesh() {
        let mesh = MeshStorage::new();
        assert!(mesh_aabb(&mesh).is_none());
    }

    #[test]
    fn aabb_center_and_diagonal() {
        let aabb = AABB {
            min: [0.0, 0.0, 0.0],
            max: [2.0, 2.0, 2.0],
        };
        assert_eq!(aabb.center(), [1.0, 1.0, 1.0]);
        let diag = aabb.diagonal();
        assert!((diag - (12.0_f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn centroid_basic() {
        let (mesh, _v, _f) = build_unit_triangle();
        let c = mesh_centroid(&mesh).unwrap();
        assert!((c[0] - 1.0 / 3.0).abs() < 1e-12);
        assert!((c[1] - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn aabb_on_icosphere() {
        let mesh = crate::test_util::build_icosphere(1);
        let aabb = mesh_aabb(&mesh).unwrap();
        assert!(aabb.min[0] >= -1.0 && aabb.min[0] <= -0.9);
        assert!(aabb.max[0] <= 1.0 && aabb.max[0] >= 0.9);
    }

    // 辅助函数：构造单位等腰直角三角形
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
}
