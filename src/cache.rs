//! 几何缓存模块
//!
//! 在 [`MeshStorage`] 之上叠加**惰性缓存**层，避免重复计算常用的几何导出量
//! （面法向、面面积、顶点法向、顶点度数、边长）。
//!
//! ## 设计
//! [`MeshCache`] **不持有**网格引用，而是在每次查询时将 `&MeshStorage` 作为参数传入。
//! 这样设计的好处是避免了生命周期约束，缓存可以独立于网格存在，
//! 且在网格拓扑不变时可以跨多次查询复用。
//!
//! **该缓存为可选组件**：`geometry` 模块默认**不**自动使用它。只有在需要复用
//! 几何导出量时，才由调用方显式持有 `MeshCache` 实例并在查询时传入 `&MeshStorage`。
//!
//! ## 用法
//! ```
//! # use halfedge::{build_mesh_from_vertices_and_faces, cache::MeshCache};
//! # let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
//! # let faces = vec![[0u32, 1, 2]];
//! # let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
//! let mut cache = MeshCache::new();
//! for f in mesh.face_ids() {
//!     let n = cache.face_normal(&mesh, f);
//!     let a = cache.face_area(&mesh, f);
//!     // ...
//! }
//! ```
//!
//! 当网格拓扑或顶点位置发生变化时，应调用 [`MeshCache::invalidate`] 清空缓存，
//! 或使用粒度更细的 [`invalidate_face`](MeshCache::invalidate_face) /
//! [`invalidate_vertex`](MeshCache::invalidate_vertex) /
//! [`invalidate_edge`](MeshCache::invalidate_edge)。
//!
//! ## 退化处理
//! 底层 [`crate::geometry`] 函数对退化几何（零面积面、孤立顶点等）返回 `None`。
//! 缓存层将其映射为零值（`[0,0,0]` / `0.0` / `0`），并在后续查询中直接返回缓存的零值，
//! 避免重复计算退化情形。

use std::collections::HashMap;

use crate::geometry::{edge_length, face_area, face_normal, vertex_normal};
use crate::ids::{FaceId, HalfEdgeId, VertexId};
use crate::linalg::vec3::Vec3;
use crate::storage::MeshStorage;
use crate::traversal::VertexAdjacentVerts;

/// 几何缓存：惰性计算并缓存常用几何导出量。
///
/// 缓存项：
/// - 面法向（`HashMap<FaceId, Vec3>`）
/// - 面面积（`HashMap<FaceId, f64>`）
/// - 顶点法向（`HashMap<VertexId, Vec3>`）
/// - 顶点度数（`HashMap<VertexId, usize>`）
/// - 边长（`HashMap<HalfEdgeId, f64>`）
///
/// 每个查询方法先检查缓存，命中则直接返回；未命中时调用 [`crate::geometry`]
/// 中的对应函数计算，存入缓存后返回。
///
/// **不持有** `&MeshStorage` 引用，每次查询时将网格引用作为参数传入，
/// 避免生命周期约束。
#[derive(Debug)]
pub struct MeshCache {
    /// 面法向缓存。
    face_normals: HashMap<FaceId, Vec3>,
    /// 面面积缓存。
    face_areas: HashMap<FaceId, f64>,
    /// 顶点法向缓存。
    vertex_normals: HashMap<VertexId, Vec3>,
    /// 顶点度数缓存。
    vertex_valences: HashMap<VertexId, usize>,
    /// 边长缓存（按半边索引）。
    edge_lengths: HashMap<HalfEdgeId, f64>,
}

impl MeshCache {
    /// 创建一个空的几何缓存。
    ///
    /// ```
    /// use halfedge::{build_mesh_from_vertices_and_faces, cache::MeshCache};
    ///
    /// let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    /// let faces = vec![[0u32, 1, 2]];
    /// let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
    /// let mut cache = MeshCache::new();
    /// let f = mesh.face_ids().next().unwrap();
    /// let n = cache.face_normal(&mesh, f);
    /// assert!((n[2] - 1.0).abs() < 1e-9);
    /// ```
    pub fn new() -> Self {
        Self {
            face_normals: HashMap::new(),
            face_areas: HashMap::new(),
            vertex_normals: HashMap::new(),
            vertex_valences: HashMap::new(),
            edge_lengths: HashMap::new(),
        }
    }

    // ============================================================
    // 查询方法
    // ============================================================

    /// 面法向：归一化的 $(\vec{B}-\vec{A}) \times (\vec{C}-\vec{A})$。
    ///
    /// 退化面（零面积）返回 `[0.0, 0.0, 0.0]`。
    pub fn face_normal(&mut self, mesh: &MeshStorage, f: FaceId) -> Vec3 {
        if let Some(&n) = self.face_normals.get(&f) {
            return n;
        }
        let n = face_normal(mesh, f).unwrap_or([0.0; 3]);
        self.face_normals.insert(f, n);
        n
    }

    /// 三角面面积：$\frac{1}{2} |(\vec{B}-\vec{A}) \times (\vec{C}-\vec{A})|$。
    ///
    /// 退化面返回 `0.0`。
    pub fn face_area(&mut self, mesh: &MeshStorage, f: FaceId) -> f64 {
        if let Some(&a) = self.face_areas.get(&f) {
            return a;
        }
        let a = face_area(mesh, f).unwrap_or(0.0);
        self.face_areas.insert(f, a);
        a
    }

    /// 顶点法向（面积加权邻接面法向平均）。
    ///
    /// 孤立顶点（无邻接面）返回 `[0.0, 0.0, 0.0]`。
    pub fn vertex_normal(&mut self, mesh: &MeshStorage, v: VertexId) -> Vec3 {
        if let Some(&n) = self.vertex_normals.get(&v) {
            return n;
        }
        let n = vertex_normal(mesh, v).unwrap_or([0.0; 3]);
        self.vertex_normals.insert(v, n);
        n
    }

    /// 顶点度数（邻接顶点数）。
    pub fn vertex_valence(&mut self, mesh: &MeshStorage, v: VertexId) -> usize {
        if let Some(&val) = self.vertex_valences.get(&v) {
            return val;
        }
        let val = VertexAdjacentVerts::new(mesh, v).count();
        self.vertex_valences.insert(v, val);
        val
    }

    /// 边长：半边两端顶点（origin 与 tip）的欧氏距离。
    ///
    /// 无效半边返回 `0.0`。
    pub fn edge_length(&mut self, mesh: &MeshStorage, he: HalfEdgeId) -> f64 {
        if let Some(&l) = self.edge_lengths.get(&he) {
            return l;
        }
        let l = edge_length(mesh, he).unwrap_or(0.0);
        self.edge_lengths.insert(he, l);
        l
    }

    // ============================================================
    // 失效方法
    // ============================================================

    /// 清空所有缓存。
    pub fn invalidate(&mut self) {
        self.face_normals.clear();
        self.face_areas.clear();
        self.vertex_normals.clear();
        self.vertex_valences.clear();
        self.edge_lengths.clear();
    }

    /// 失效指定面的缓存（面法向 + 面面积）。
    ///
    /// **注意**：面的几何变化会影响其顶点的法向，但本方法不自动级联失效
    /// 顶点法向缓存。如需完整失效，请额外调用
    /// [`invalidate_vertex`](Self::invalidate_vertex)。
    pub fn invalidate_face(&mut self, f: FaceId) {
        self.face_normals.remove(&f);
        self.face_areas.remove(&f);
    }

    /// 失效指定顶点的缓存（顶点法向 + 顶点度数）。
    pub fn invalidate_vertex(&mut self, v: VertexId) {
        self.vertex_normals.remove(&v);
        self.vertex_valences.remove(&v);
    }

    /// 失效指定半边的边长缓存。
    pub fn invalidate_edge(&mut self, he: HalfEdgeId) {
        self.edge_lengths.remove(&he);
    }

    // ============================================================
    // 缓存统计（主要用于测试与调试）
    // ============================================================

    /// 返回当前缓存的面法向数量。
    pub fn face_normal_count(&self) -> usize {
        self.face_normals.len()
    }

    /// 返回当前缓存的面面积数量。
    pub fn face_area_count(&self) -> usize {
        self.face_areas.len()
    }

    /// 返回当前缓存的顶点法向数量。
    pub fn vertex_normal_count(&self) -> usize {
        self.vertex_normals.len()
    }

    /// 返回当前缓存的顶点度数数量。
    pub fn vertex_valence_count(&self) -> usize {
        self.vertex_valences.len()
    }

    /// 返回当前缓存的边长数量。
    pub fn edge_length_count(&self) -> usize {
        self.edge_lengths.len()
    }
}

impl Default for MeshCache {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{
        edge_length as geo_edge_length, face_area as geo_face_area, face_normal as geo_face_normal,
        vertex_normal as geo_vertex_normal,
    };
    use crate::io::build_mesh_from_vertices_and_faces;
    use crate::test_util::build_icosphere;
    use crate::traversal::VertexAdjacentVerts;

    /// 构建单三角形网格：A=(0,0,0), B=(1,0,0), C=(0,1,0)，CCW 朝向 +z。
    fn build_triangle() -> MeshStorage {
        let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![[0u32, 1, 2]];
        build_mesh_from_vertices_and_faces(&verts, &faces).unwrap()
    }

    // ---------- 缓存值与直接计算一致 ----------

    #[test]
    fn face_normal_matches_direct() {
        let mesh = build_triangle();
        let f = mesh.face_ids().next().unwrap();
        let mut cache = MeshCache::new();
        let cached = cache.face_normal(&mesh, f);
        let direct = geo_face_normal(&mesh, f).unwrap();
        for i in 0..3 {
            assert!((cached[i] - direct[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn face_area_matches_direct() {
        let mesh = build_triangle();
        let f = mesh.face_ids().next().unwrap();
        let mut cache = MeshCache::new();
        let cached = cache.face_area(&mesh, f);
        let direct = geo_face_area(&mesh, f).unwrap();
        assert!((cached - direct).abs() < 1e-12);
    }

    #[test]
    fn vertex_normal_matches_direct() {
        let mesh = build_triangle();
        let v = mesh.vertex_ids().next().unwrap();
        let mut cache = MeshCache::new();
        let cached = cache.vertex_normal(&mesh, v);
        let direct = geo_vertex_normal(&mesh, v).unwrap();
        for i in 0..3 {
            assert!((cached[i] - direct[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn vertex_valence_matches_direct() {
        let mesh = build_triangle();
        let v = mesh.vertex_ids().next().unwrap();
        let mut cache = MeshCache::new();
        let cached = cache.vertex_valence(&mesh, v);
        let direct = VertexAdjacentVerts::new(&mesh, v).count();
        assert_eq!(cached, direct);
        // 单三角形每个顶点度数为 2
        assert_eq!(cached, 2);
    }

    #[test]
    fn edge_length_matches_direct() {
        let mesh = build_triangle();
        let he = mesh.halfedge_ids().next().unwrap();
        let mut cache = MeshCache::new();
        let cached = cache.edge_length(&mesh, he);
        let direct = geo_edge_length(&mesh, he).unwrap();
        assert!((cached - direct).abs() < 1e-12);
    }

    // ---------- 缓存避免重复计算 ----------

    #[test]
    fn cache_stores_after_first_call() {
        let mesh = build_triangle();
        let f = mesh.face_ids().next().unwrap();
        let mut cache = MeshCache::new();
        assert_eq!(cache.face_normal_count(), 0);
        assert_eq!(cache.face_area_count(), 0);

        let _ = cache.face_normal(&mesh, f);
        assert_eq!(cache.face_normal_count(), 1);

        let _ = cache.face_area(&mesh, f);
        assert_eq!(cache.face_area_count(), 1);

        // 再次调用，缓存数量不应增加
        let _ = cache.face_normal(&mesh, f);
        let _ = cache.face_area(&mesh, f);
        assert_eq!(cache.face_normal_count(), 1);
        assert_eq!(cache.face_area_count(), 1);
    }

    #[test]
    fn cache_returns_same_value_on_repeat() {
        let mesh = build_triangle();
        let f = mesh.face_ids().next().unwrap();
        let mut cache = MeshCache::new();
        let n1 = cache.face_normal(&mesh, f);
        let n2 = cache.face_normal(&mesh, f);
        assert_eq!(n1, n2);
    }

    #[test]
    fn cache_hit_does_not_increase_count() {
        let mesh = build_icosphere(1);
        let faces: Vec<_> = mesh.face_ids().collect();
        let mut cache = MeshCache::new();

        // 第一轮：填充缓存
        for &f in &faces {
            cache.face_normal(&mesh, f);
        }
        assert_eq!(cache.face_normal_count(), faces.len());

        // 第二轮：全部命中，数量不变
        for &f in &faces {
            cache.face_normal(&mesh, f);
        }
        assert_eq!(cache.face_normal_count(), faces.len());
    }

    // ---------- 全局失效 ----------

    #[test]
    fn invalidate_all_clears_everything() {
        let mesh = build_triangle();
        let f = mesh.face_ids().next().unwrap();
        let v = mesh.vertex_ids().next().unwrap();
        let he = mesh.halfedge_ids().next().unwrap();
        let mut cache = MeshCache::new();

        cache.face_normal(&mesh, f);
        cache.face_area(&mesh, f);
        cache.vertex_normal(&mesh, v);
        cache.vertex_valence(&mesh, v);
        cache.edge_length(&mesh, he);

        assert!(cache.face_normal_count() > 0);
        assert!(cache.face_area_count() > 0);
        assert!(cache.vertex_normal_count() > 0);
        assert!(cache.vertex_valence_count() > 0);
        assert!(cache.edge_length_count() > 0);

        cache.invalidate();

        assert_eq!(cache.face_normal_count(), 0);
        assert_eq!(cache.face_area_count(), 0);
        assert_eq!(cache.vertex_normal_count(), 0);
        assert_eq!(cache.vertex_valence_count(), 0);
        assert_eq!(cache.edge_length_count(), 0);
    }

    // ---------- 粒度失效 ----------

    #[test]
    fn invalidate_face_clears_only_that_face() {
        let mesh = build_icosphere(1);
        let faces: Vec<_> = mesh.face_ids().collect();
        let f0 = faces[0];
        let f1 = faces[1];
        let mut cache = MeshCache::new();

        cache.face_normal(&mesh, f0);
        cache.face_normal(&mesh, f1);
        assert_eq!(cache.face_normal_count(), 2);

        cache.invalidate_face(f0);
        assert_eq!(cache.face_normal_count(), 1);

        // f1 仍在缓存中，再次查询不会增加计数
        let _ = cache.face_normal(&mesh, f1);
        assert_eq!(cache.face_normal_count(), 1);
    }

    #[test]
    fn invalidate_face_clears_area_too() {
        let mesh = build_triangle();
        let f = mesh.face_ids().next().unwrap();
        let mut cache = MeshCache::new();

        cache.face_normal(&mesh, f);
        cache.face_area(&mesh, f);
        assert_eq!(cache.face_normal_count(), 1);
        assert_eq!(cache.face_area_count(), 1);

        cache.invalidate_face(f);
        assert_eq!(cache.face_normal_count(), 0);
        assert_eq!(cache.face_area_count(), 0);
    }

    #[test]
    fn invalidate_vertex_clears_only_that_vertex() {
        let mesh = build_icosphere(1);
        let verts: Vec<_> = mesh.vertex_ids().collect();
        let v0 = verts[0];
        let v1 = verts[1];
        let mut cache = MeshCache::new();

        cache.vertex_normal(&mesh, v0);
        cache.vertex_normal(&mesh, v1);
        cache.vertex_valence(&mesh, v0);
        cache.vertex_valence(&mesh, v1);
        assert_eq!(cache.vertex_normal_count(), 2);
        assert_eq!(cache.vertex_valence_count(), 2);

        cache.invalidate_vertex(v0);
        assert_eq!(cache.vertex_normal_count(), 1);
        assert_eq!(cache.vertex_valence_count(), 1);
    }

    #[test]
    fn invalidate_edge_clears_only_that_edge() {
        let mesh = build_icosphere(1);
        let hes: Vec<_> = mesh.halfedge_ids().collect();
        let he0 = hes[0];
        let he1 = hes[1];
        let mut cache = MeshCache::new();

        cache.edge_length(&mesh, he0);
        cache.edge_length(&mesh, he1);
        assert_eq!(cache.edge_length_count(), 2);

        cache.invalidate_edge(he0);
        assert_eq!(cache.edge_length_count(), 1);
    }

    #[test]
    fn invalidate_then_recompute_returns_same_value() {
        let mesh = build_triangle();
        let f = mesh.face_ids().next().unwrap();
        let mut cache = MeshCache::new();

        let n1 = cache.face_normal(&mesh, f);
        cache.invalidate_face(f);
        let n2 = cache.face_normal(&mesh, f);

        assert_eq!(n1, n2);
    }

    // ---------- 单三角形网格 ----------

    #[test]
    fn triangle_face_normal_points_up() {
        let mesh = build_triangle();
        let f = mesh.face_ids().next().unwrap();
        let mut cache = MeshCache::new();
        let n = cache.face_normal(&mesh, f);
        assert!(n[0].abs() < 1e-12);
        assert!(n[1].abs() < 1e-12);
        assert!((n[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn triangle_face_area_is_half() {
        let mesh = build_triangle();
        let f = mesh.face_ids().next().unwrap();
        let mut cache = MeshCache::new();
        let a = cache.face_area(&mesh, f);
        assert!((a - 0.5).abs() < 1e-12);
    }

    #[test]
    fn triangle_vertex_valence_is_two() {
        let mesh = build_triangle();
        for v in mesh.vertex_ids() {
            let mut cache = MeshCache::new();
            assert_eq!(cache.vertex_valence(&mesh, v), 2);
        }
    }

    #[test]
    fn triangle_vertex_normal_points_up() {
        let mesh = build_triangle();
        for v in mesh.vertex_ids() {
            let mut cache = MeshCache::new();
            let n = cache.vertex_normal(&mesh, v);
            assert!((n[2] - 1.0).abs() < 1e-12);
        }
    }

    // ---------- 较大网格（icosphere）----------

    #[test]
    fn icosphere_all_face_normals_match_direct() {
        let mesh = build_icosphere(2);
        let mut cache = MeshCache::new();
        for f in mesh.face_ids() {
            let cached = cache.face_normal(&mesh, f);
            let direct = geo_face_normal(&mesh, f).unwrap();
            for i in 0..3 {
                assert!(
                    (cached[i] - direct[i]).abs() < 1e-12,
                    "面 {:?} 法向不匹配",
                    f
                );
            }
        }
        assert_eq!(cache.face_normal_count(), mesh.face_count());
    }

    #[test]
    fn icosphere_all_face_areas_match_direct() {
        let mesh = build_icosphere(2);
        let mut cache = MeshCache::new();
        for f in mesh.face_ids() {
            let cached = cache.face_area(&mesh, f);
            let direct = geo_face_area(&mesh, f).unwrap();
            assert!((cached - direct).abs() < 1e-12);
        }
        assert_eq!(cache.face_area_count(), mesh.face_count());
    }

    #[test]
    fn icosphere_all_vertex_normals_match_direct() {
        let mesh = build_icosphere(2);
        let mut cache = MeshCache::new();
        for v in mesh.vertex_ids() {
            let cached = cache.vertex_normal(&mesh, v);
            if let Some(direct) = geo_vertex_normal(&mesh, v) {
                for i in 0..3 {
                    assert!((cached[i] - direct[i]).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn icosphere_all_vertex_valences_match_direct() {
        let mesh = build_icosphere(2);
        let mut cache = MeshCache::new();
        for v in mesh.vertex_ids() {
            let cached = cache.vertex_valence(&mesh, v);
            let direct = VertexAdjacentVerts::new(&mesh, v).count();
            assert_eq!(cached, direct);
        }
        assert_eq!(cache.vertex_valence_count(), mesh.vertex_count());
    }

    #[test]
    fn icosphere_all_edge_lengths_match_direct() {
        let mesh = build_icosphere(2);
        let mut cache = MeshCache::new();
        for he in mesh.halfedge_ids() {
            let cached = cache.edge_length(&mesh, he);
            let direct = geo_edge_length(&mesh, he).unwrap();
            assert!((cached - direct).abs() < 1e-12);
        }
        assert_eq!(cache.edge_length_count(), mesh.halfedge_count());
    }

    #[test]
    fn icosphere_cache_survives_repeated_queries() {
        let mesh = build_icosphere(2);
        let mut cache = MeshCache::new();
        // 第一轮：填充缓存
        for f in mesh.face_ids() {
            cache.face_normal(&mesh, f);
        }
        let count_after_first = cache.face_normal_count();
        // 第二轮：应全部命中缓存
        for f in mesh.face_ids() {
            cache.face_normal(&mesh, f);
        }
        assert_eq!(cache.face_normal_count(), count_after_first);
    }

    // ---------- Default trait ----------

    #[test]
    fn default_creates_empty_cache() {
        let cache = MeshCache::default();
        assert_eq!(cache.face_normal_count(), 0);
        assert_eq!(cache.face_area_count(), 0);
        assert_eq!(cache.vertex_normal_count(), 0);
        assert_eq!(cache.vertex_valence_count(), 0);
        assert_eq!(cache.edge_length_count(), 0);
    }

    // ---------- 退化几何 ----------

    #[test]
    fn degenerate_face_returns_zero_normal() {
        // 共线三点 → 退化三角形
        let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let faces = vec![[0u32, 1, 2]];
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let f = mesh.face_ids().next().unwrap();
        let mut cache = MeshCache::new();
        let n = cache.face_normal(&mesh, f);
        assert_eq!(n, [0.0, 0.0, 0.0]);
        // 面积也应为 0
        let a = cache.face_area(&mesh, f);
        assert!(a.abs() < 1e-12);
    }

    #[test]
    fn degenerate_face_cached_zero_on_repeat() {
        let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let faces = vec![[0u32, 1, 2]];
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let f = mesh.face_ids().next().unwrap();
        let mut cache = MeshCache::new();

        let n1 = cache.face_normal(&mesh, f);
        assert_eq!(n1, [0.0, 0.0, 0.0]);
        assert_eq!(cache.face_normal_count(), 1);

        // 第二次查询应命中缓存，返回同样的零值
        let n2 = cache.face_normal(&mesh, f);
        assert_eq!(n2, [0.0, 0.0, 0.0]);
        assert_eq!(cache.face_normal_count(), 1);
    }
}
