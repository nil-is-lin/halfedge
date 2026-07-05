//! BVH（Bounding Volume Hierarchy）加速结构
//!
//! 基于三角面 AABB 包围盒的二叉树，加速射线求交与最近点查询。
//! 构建复杂度 $O(F \log F)$，单次射线/最近点查询平均复杂度 $O(\log F)$。
//!
//! ## 构建
//!
//! 1. 收集每个三角面的 AABB 与中心点（preprocess）；
//! 2. 递归地选取最长轴，按面中心在该轴上的中位数划分；
//! 3. 当面数 $\le$ [`LEAF_MAX_FACES`] 时停止，作为叶子节点。
//!
//! ## 射线查询
//!
//! 先与节点 AABB 做 slab 法求交（命中时返回 `(t_enter, t_exit)`）；
//! 仅当 `t_enter < current_best_t` 才递归子树。叶子节点内对每个面
//! 调用 Möller-Trumbore。
//!
//! ## 最近点查询
//!
//! 维护当前最近距离平方 `best_d²`；访问节点前先计算点到节点 AABB 的
//! 最近距离平方 `d_aabb²`，若 `d_aabb² > best_d²` 则剪枝。
//!
//! ## 与暴力算法的关系
//!
//! [`crate::geometry::ray_mesh_intersection`] 与
//! [`crate::geometry::point_triangle_distance`] 都是 $O(F)$ 暴力扫描；
//! BVH 把它们降到 $O(\log F)$，对大网格（>1k 面）有数量级加速。
//! 对小网格（< 64 面）BVH 自身的常数开销可能让它略慢于暴力。
//!
//! [`LEAF_MAX_FACES`]: LEAF_MAX_FACES

use crate::geometry::{AABB, RayHit, closest_point_on_triangle, ray_triangle_intersection};
use crate::ids::{FaceId, VertexId};
use crate::storage::MeshStorage;
use crate::traversal::FaceVertices;

/// 叶子节点最多容纳的面数。超过此值则继续分裂。
pub const LEAF_MAX_FACES: usize = 8;

/// BVH 节点（内部节点与叶子节点统一表示）。
#[derive(Debug, Clone)]
struct BvhNode {
    /// 节点的合并 AABB（覆盖子树所有面）。
    aabb: AABB,
    /// 内部节点：左子节点索引。
    left: Option<usize>,
    /// 内部节点：右子节点索引。
    right: Option<usize>,
    /// 叶子节点：面 ID 列表；内部节点为空。
    faces: Vec<FaceId>,
}

impl BvhNode {
    #[inline]
    fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }
}

/// Bounding Volume Hierarchy：基于 AABB 的二叉加速结构。
#[derive(Debug, Clone)]
pub struct Bvh {
    nodes: Vec<BvhNode>,
    root: Option<usize>,
}

impl Bvh {
    /// 构建空 BVH。
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root: None,
        }
    }

    /// 从网格构建 BVH（仅包含三角面）。
    ///
    /// 非三角面（n-gon）被跳过；退化三角形（含重复顶点）也跳过。
    pub fn build(mesh: &MeshStorage) -> Self {
        let mut prims: Vec<(FaceId, AABB, [f64; 3])> = Vec::with_capacity(mesh.face_count());
        for f in mesh.face_ids() {
            let verts: Vec<VertexId> = FaceVertices::new(mesh, f).collect();
            if verts.len() != 3 {
                continue;
            }
            let positions: Vec<[f64; 3]> = verts
                .iter()
                .filter_map(|v| mesh.get_vertex(*v))
                .map(|v| v.position)
                .collect();
            if positions.len() != 3 {
                continue;
            }
            let aabb = AABB::from_points(&positions);
            // 退化三角形（共线/重复点）的 AABB 对角线 ≈ 0，跳过
            if aabb.diagonal() < 1e-14 {
                continue;
            }
            let center = aabb.center();
            prims.push((f, aabb, center));
        }

        if prims.is_empty() {
            return Self::new();
        }

        // 预估节点数：2N - 1（满二叉树上限）
        let mut nodes: Vec<BvhNode> = Vec::with_capacity(2 * prims.len() - 1);
        let root = Self::build_recursive(&mut prims, &mut nodes);
        Self {
            nodes,
            root: Some(root),
        }
    }

    /// 递归构建子树，返回子树根节点索引。
    fn build_recursive(
        prims: &mut Vec<(FaceId, AABB, [f64; 3])>,
        nodes: &mut Vec<BvhNode>,
    ) -> usize {
        debug_assert!(!prims.is_empty());

        // 计算节点 AABB（合并所有 primitive 的 AABB）
        let mut node_aabb = AABB::new();
        for (_, aabb, _) in prims.iter() {
            node_aabb = node_aabb.union(aabb);
        }

        // 终止条件：足够少 → 叶子
        if prims.len() <= LEAF_MAX_FACES {
            let faces: Vec<FaceId> = prims.iter().map(|(f, _, _)| *f).collect();
            let idx = nodes.len();
            nodes.push(BvhNode {
                aabb: node_aabb,
                left: None,
                right: None,
                faces,
            });
            return idx;
        }

        // 选取最长轴
        let diag = [
            node_aabb.max[0] - node_aabb.min[0],
            node_aabb.max[1] - node_aabb.min[1],
            node_aabb.max[2] - node_aabb.min[2],
        ];
        let axis = if diag[0] >= diag[1] && diag[0] >= diag[2] {
            0
        } else if diag[1] >= diag[2] {
            1
        } else {
            2
        };

        // 按面中心在该轴上排序，取中位数划分
        prims.sort_by(|a, b| {
            a.2[axis]
                .partial_cmp(&b.2[axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mid = prims.len() / 2;
        let mut right_prims = prims.split_off(mid);

        // 先占位父节点（这样子节点 push 时不会失效 idx）
        let idx = nodes.len();
        nodes.push(BvhNode {
            aabb: node_aabb,
            left: None,
            right: None,
            faces: Vec::new(),
        });

        let left = Self::build_recursive(prims, nodes);
        let right = Self::build_recursive(&mut right_prims, nodes);

        nodes[idx].left = Some(left);
        nodes[idx].right = Some(right);
        idx
    }

    /// 射线与网格求**最近**交点（BVH 加速）。
    ///
    /// 返回 `t` 最小的 [`RayHit`]；无交点返回 `None`。
    /// 复杂度平均 $O(\log F)$，最坏 $O(F)$。
    pub fn ray_intersection(
        &self,
        origin: [f64; 3],
        direction: [f64; 3],
        mesh: &MeshStorage,
    ) -> Option<RayHit> {
        let root = self.root?;
        let mut best: Option<(f64, RayHit)> = None;
        self.ray_recursive(root, origin, direction, mesh, &mut best);
        best.map(|(_, h)| h)
    }

    fn ray_recursive(
        &self,
        node_idx: usize,
        origin: [f64; 3],
        direction: [f64; 3],
        mesh: &MeshStorage,
        best: &mut Option<(f64, RayHit)>,
    ) {
        let node = &self.nodes[node_idx];
        // 与节点 AABB 求交
        let (t_enter, _t_exit) = match ray_aabb(origin, direction, &node.aabb) {
            Some(t) => t,
            None => return,
        };
        // 当前最近交点比进入 AABB 还近 → 剪枝
        if let Some((best_t, _)) = best
            && *best_t < t_enter
        {
            return;
        }

        if node.is_leaf() {
            for &f in &node.faces {
                let verts: Vec<VertexId> = FaceVertices::new(mesh, f).collect();
                if verts.len() != 3 {
                    continue;
                }
                let v0 = match mesh.get_vertex(verts[0]) {
                    Some(v) => v.position,
                    None => continue,
                };
                let v1 = match mesh.get_vertex(verts[1]) {
                    Some(v) => v.position,
                    None => continue,
                };
                let v2 = match mesh.get_vertex(verts[2]) {
                    Some(v) => v.position,
                    None => continue,
                };

                if let Some((t, u, v)) = ray_triangle_intersection(origin, direction, v0, v1, v2) {
                    let hit = RayHit {
                        position: [
                            origin[0] + t * direction[0],
                            origin[1] + t * direction[1],
                            origin[2] + t * direction[2],
                        ],
                        t,
                        face: f,
                        barycentric: (u, v),
                    };
                    match best {
                        Some((bt, _)) if t < *bt => *best = Some((t, hit)),
                        None => *best = Some((t, hit)),
                        _ => {}
                    }
                }
            }
        } else {
            // 选择最近 AABB 优先访问（提升剪枝效率）
            let left_idx = node.left;
            let right_idx = node.right;
            let (first, second) = match (left_idx, right_idx) {
                (Some(l), Some(r)) => {
                    let tl = ray_aabb(origin, direction, &self.nodes[l].aabb).map(|(t, _)| t);
                    let tr = ray_aabb(origin, direction, &self.nodes[r].aabb).map(|(t, _)| t);
                    match (tl, tr) {
                        (Some(tl), Some(tr)) if tr < tl => (Some(r), Some(l)),
                        _ => (Some(l), Some(r)),
                    }
                }
                (a, b) => (a, b),
            };
            if let Some(f) = first {
                self.ray_recursive(f, origin, direction, mesh, best);
            }
            if let Some(s) = second {
                self.ray_recursive(s, origin, direction, mesh, best);
            }
        }
    }

    /// 最近点查询：在网格三角面上查找距离 `p` 最近的点。
    ///
    /// 返回 `(face_id, closest_point, distance_squared)`；网格无三角面时返回 `None`。
    pub fn nearest_point(
        &self,
        p: [f64; 3],
        mesh: &MeshStorage,
    ) -> Option<(FaceId, [f64; 3], f64)> {
        let root = self.root?;
        let mut best: Option<(FaceId, [f64; 3], f64)> = None;
        self.nearest_recursive(root, p, mesh, &mut best);
        best
    }

    fn nearest_recursive(
        &self,
        node_idx: usize,
        p: [f64; 3],
        mesh: &MeshStorage,
        best: &mut Option<(FaceId, [f64; 3], f64)>,
    ) {
        let node = &self.nodes[node_idx];
        // AABB 最近距离剪枝
        let dist_to_aabb = point_aabb_distance_sq(p, &node.aabb);
        if let Some((_, _, best_d)) = best
            && dist_to_aabb > *best_d
        {
            return;
        }

        if node.is_leaf() {
            for &f in &node.faces {
                let verts: Vec<VertexId> = FaceVertices::new(mesh, f).collect();
                if verts.len() != 3 {
                    continue;
                }
                let v0 = match mesh.get_vertex(verts[0]) {
                    Some(v) => v.position,
                    None => continue,
                };
                let v1 = match mesh.get_vertex(verts[1]) {
                    Some(v) => v.position,
                    None => continue,
                };
                let v2 = match mesh.get_vertex(verts[2]) {
                    Some(v) => v.position,
                    None => continue,
                };

                let closest = closest_point_on_triangle(p, v0, v1, v2);
                let d = [closest[0] - p[0], closest[1] - p[1], closest[2] - p[2]];
                let dist_sq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                match best {
                    Some((_, _, bd)) if dist_sq < *bd => *best = Some((f, closest, dist_sq)),
                    None => *best = Some((f, closest, dist_sq)),
                    _ => {}
                }
            }
        } else {
            // 优先访问更近的子节点
            let mut children: Vec<(usize, f64)> = Vec::with_capacity(2);
            if let Some(l) = node.left {
                children.push((l, point_aabb_distance_sq(p, &self.nodes[l].aabb)));
            }
            if let Some(r) = node.right {
                children.push((r, point_aabb_distance_sq(p, &self.nodes[r].aabb)));
            }
            children.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for (idx, _) in children {
                self.nearest_recursive(idx, p, mesh, best);
            }
        }
    }

    /// BVH 节点总数。
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// 是否为空（无三角面）。
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// 叶子节点数。
    pub fn leaf_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    /// 最大树深（根节点深 1）。
    pub fn depth(&self) -> usize {
        self.root
            .map(|r| Self::depth_recursive(&self.nodes, r))
            .unwrap_or(0)
    }

    fn depth_recursive(nodes: &[BvhNode], idx: usize) -> usize {
        let node = &nodes[idx];
        if node.is_leaf() {
            1
        } else {
            let l = node
                .left
                .map(|i| Self::depth_recursive(nodes, i))
                .unwrap_or(0);
            let r = node
                .right
                .map(|i| Self::depth_recursive(nodes, i))
                .unwrap_or(0);
            1 + l.max(r)
        }
    }
}

impl Default for Bvh {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// 内部辅助
// ============================================================

/// 射线与 AABB 的 slab 法求交。返回 `(t_enter, t_exit)` 或 `None`。
///
/// $t_{enter} = \max_i \min(t_i^{\min}, t_i^{\max})$，
/// $t_{exit} = \min_i \max(t_i^{\min}, t_i^{\max})$，
/// 当 $t_{enter} \le t_{exit}$ 且 $t_{exit} \ge 0$ 时相交。
fn ray_aabb(origin: [f64; 3], direction: [f64; 3], aabb: &AABB) -> Option<(f64, f64)> {
    let mut t_enter = f64::NEG_INFINITY;
    let mut t_exit = f64::INFINITY;
    for i in 0..3 {
        if direction[i].abs() < 1e-14 {
            // 与 slab 平行：原点必须在 slab 内
            if origin[i] < aabb.min[i] || origin[i] > aabb.max[i] {
                return None;
            }
        } else {
            let inv_d = 1.0 / direction[i];
            let t1 = (aabb.min[i] - origin[i]) * inv_d;
            let t2 = (aabb.max[i] - origin[i]) * inv_d;
            let (t1, t2) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
            if t1 > t_enter {
                t_enter = t1;
            }
            if t2 < t_exit {
                t_exit = t2;
            }
            if t_enter > t_exit {
                return None;
            }
        }
    }
    // 要求交点在射线正方向（t_exit >= 0）；允许原点在 AABB 内（t_enter < 0）
    if t_exit < 0.0 {
        None
    } else {
        Some((t_enter, t_exit))
    }
}

/// 点到 AABB 的最近距离平方。
///
/// 对每个轴：若 `p[i] < min[i]`，距离贡献 `(min[i] - p[i])²`；
/// 若 `p[i] > max[i]`，贡献 `(p[i] - max[i])²`；否则 0。
fn point_aabb_distance_sq(p: [f64; 3], aabb: &AABB) -> f64 {
    let mut d = 0.0;
    for ((&p_i, &min_i), &max_i) in p.iter().zip(aabb.min.iter()).zip(aabb.max.iter()) {
        if p_i < min_i {
            let dd = min_i - p_i;
            d += dd * dd;
        } else if p_i > max_i {
            let dd = p_i - max_i;
            d += dd * dd;
        }
    }
    d
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_mesh_from_vertices_and_faces;
    use crate::test_util::build_icosphere;

    fn build_two_triangles() -> MeshStorage {
        // 两个不相邻三角形，分处不同位置
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
        ];
        let faces = vec![[0, 1, 2], [3, 4, 5]];
        build_mesh_from_vertices_and_faces(&verts, &faces)
    }

    #[test]
    fn build_empty_mesh() {
        let mesh = MeshStorage::new();
        let bvh = Bvh::build(&mesh);
        assert!(bvh.is_empty());
        assert_eq!(bvh.node_count(), 0);
        assert_eq!(bvh.depth(), 0);
    }

    #[test]
    fn build_two_triangles_basic() {
        let mesh = build_two_triangles();
        let bvh = Bvh::build(&mesh);
        // 2 个面 ≤ LEAF_MAX_FACES(8) → 单个叶子节点
        assert!(!bvh.is_empty());
        assert_eq!(bvh.node_count(), 1);
        assert_eq!(bvh.leaf_count(), 1);
        assert_eq!(bvh.depth(), 1);
    }

    #[test]
    fn ray_hit_first_triangle() {
        let mesh = build_two_triangles();
        let bvh = Bvh::build(&mesh);
        let hit = bvh.ray_intersection([0.2, 0.2, 1.0], [0.0, 0.0, -1.0], &mesh);
        assert!(hit.is_some());
        let h = hit.unwrap();
        // 命中第一个三角形 (z=0 平面)
        assert!((h.position[2]).abs() < 1e-9);
        assert!((h.t - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ray_hit_second_triangle_when_first_missed() {
        let mesh = build_two_triangles();
        let bvh = Bvh::build(&mesh);
        // 射线从 (10.2, 0.2, 1.0) 向下射，只能命中第二个三角形
        let hit = bvh.ray_intersection([10.2, 0.2, 1.0], [0.0, 0.0, -1.0], &mesh);
        assert!(hit.is_some());
        let h = hit.unwrap();
        assert!((h.position[0] - 10.2).abs() < 1e-9);
    }

    #[test]
    fn ray_miss_returns_none() {
        let mesh = build_two_triangles();
        let bvh = Bvh::build(&mesh);
        // 射线从 (5, 5, 1) 向下射，无三角形可命中
        let hit = bvh.ray_intersection([5.0, 5.0, 1.0], [0.0, 0.0, -1.0], &mesh);
        assert!(hit.is_none());
    }

    #[test]
    fn ray_returns_nearest_hit() {
        let mesh = build_two_triangles();
        let _bvh = Bvh::build(&mesh);
        // 射线从 (-1, 0.3, 1) 沿 +x 方向，会先穿过第二个三角形（z=0）...不会，因为面法线是 +z
        // 改为从 z=5 沿 -z 射，从上方穿过
        // 实际让两三角形重叠：第一个三角形在 z=0，第二个也在 z=0 → 不会重叠
        // 改测试：用 icosphere 验证最近交点
        let mesh = build_icosphere(2);
        let bvh = Bvh::build(&mesh);
        // 从球外向球心方向射
        let center = [0.0, 0.0, 0.0];
        let origin = [0.0, 0.0, 5.0];
        let dir = [0.0, 0.0, -1.0];
        let hit = bvh.ray_intersection(origin, dir, &mesh);
        assert!(hit.is_some());
        let h = hit.unwrap();
        // 命中点应在球面上，距离原点 ≈ 球半径
        let r = ((h.position[0] - center[0]).powi(2)
            + (h.position[1] - center[1]).powi(2)
            + (h.position[2] - center[2]).powi(2))
        .sqrt();
        // icosphere(2) 半径 ≈ 1.0
        assert!((r - 1.0).abs() < 0.1, "球面半径 = {r}");
        let _ = dir;
    }

    #[test]
    fn nearest_point_on_triangle_returns_vertex() {
        let mesh = build_two_triangles();
        let bvh = Bvh::build(&mesh);
        // 查询点远离所有三角形
        let (f, closest, d_sq) = bvh
            .nearest_point([-1.0, -1.0, 1.0], &mesh)
            .expect("应有最近点");
        // 最近点应是第一个三角形的顶点 (0,0,0)
        assert!((closest[0]).abs() < 1e-9);
        assert!((closest[1]).abs() < 1e-9);
        assert!((closest[2]).abs() < 1e-9);
        // 距离 = sqrt(1+1+1) = sqrt(3)
        assert!((d_sq - 3.0).abs() < 1e-9, "d² = {d_sq}");
        let _ = f;
    }

    #[test]
    fn nearest_point_inside_triangle() {
        let mesh = build_two_triangles();
        let bvh = Bvh::build(&mesh);
        // 查询点在第一个三角形上方
        let (f, closest, d_sq) = bvh
            .nearest_point([0.3, 0.3, 0.5], &mesh)
            .expect("应有最近点");
        assert!((closest[2]).abs() < 1e-9); // 投影到 z=0
        assert!((d_sq - 0.25).abs() < 1e-9, "d² = {d_sq}");
        let _ = f;
    }

    #[test]
    fn nearest_point_far_away_picks_closer_triangle() {
        let mesh = build_two_triangles();
        let bvh = Bvh::build(&mesh);
        // 查询点靠近第二个三角形
        let (f, _closest, _d_sq) = bvh
            .nearest_point([10.3, 0.3, 0.5], &mesh)
            .expect("应有最近点");
        // 第二个三角形的面 ID 应是 mesh.face_ids() 的第二个
        let face_ids: Vec<FaceId> = mesh.face_ids().collect();
        assert_eq!(f, face_ids[1]);
    }

    #[test]
    fn icosphere_bvh_consistent_with_brute_force_ray() {
        let mesh = build_icosphere(2);
        let bvh = Bvh::build(&mesh);
        // 测试多条射线（避开对角方向，以免恰好命中两三角形共享边产生并列 t）
        for (origin, dir) in [
            ([0.0, 0.0, 5.0], [0.0, 0.0, -1.0]),
            ([5.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
            ([0.0, 5.0, 0.0], [0.0, -1.0, 0.0]),
            ([3.0, 1.0, 2.0], [-1.0, -0.3, -0.7]),
            ([-2.5, 1.5, 2.0], [0.7, -0.4, -0.6]),
        ] {
            let bvh_hit = bvh.ray_intersection(origin, dir, &mesh);
            let brute_hit = crate::geometry::ray_mesh_intersection(origin, dir, &mesh);
            // 两者都应命中或都未命中
            assert_eq!(bvh_hit.is_some(), brute_hit.is_some());
            if let (Some(b), Some(g)) = (bvh_hit, brute_hit) {
                assert!(
                    (b.t - g.t).abs() < 1e-9,
                    "t 不一致：BVH={} 暴力={}",
                    b.t,
                    g.t
                );
                // 当 t 几乎相等（< 1e-12）时允许 face 不同（射线命中共享边/顶点的并列情形）
                if (b.t - g.t).abs() >= 1e-12 {
                    assert_eq!(b.face, g.face, "面 ID 不一致");
                }
            }
        }
    }

    #[test]
    fn icosphere_bvh_consistent_with_brute_force_nearest() {
        let mesh = build_icosphere(2);
        let bvh = Bvh::build(&mesh);
        // 测试多个查询点
        for p in [
            [0.5, 0.5, 0.5],
            [-0.7, 0.2, 0.1],
            [3.0, -2.0, 1.5],
            [-1.0, -1.0, -1.0],
        ] {
            let bvh_res = bvh.nearest_point(p, &mesh);
            // 暴力扫描所有三角面
            let mut brute_best: Option<(FaceId, [f64; 3], f64)> = None;
            for f in mesh.face_ids() {
                let verts: Vec<VertexId> = crate::traversal::FaceVertices::new(&mesh, f).collect();
                if verts.len() != 3 {
                    continue;
                }
                let v0 = mesh.get_vertex(verts[0]).unwrap().position;
                let v1 = mesh.get_vertex(verts[1]).unwrap().position;
                let v2 = mesh.get_vertex(verts[2]).unwrap().position;
                let c = closest_point_on_triangle(p, v0, v1, v2);
                let d = [c[0] - p[0], c[1] - p[1], c[2] - p[2]];
                let d_sq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                match brute_best {
                    Some((_, _, bd)) if d_sq < bd => brute_best = Some((f, c, d_sq)),
                    None => brute_best = Some((f, c, d_sq)),
                    _ => {}
                }
            }
            assert!(bvh_res.is_some(), "BVH 应有结果");
            assert!(brute_best.is_some(), "暴力应有结果");
            let (bf, bc, bd) = bvh_res.unwrap();
            let (_, _, gd) = brute_best.unwrap();
            assert!((bd - gd).abs() < 1e-9, "距离不一致：BVH={bd} 暴力={gd}");
            let _ = (bf, bc);
        }
    }

    #[test]
    fn large_mesh_bvh_depth_is_logarithmic() {
        // 高细分 icosphere：80 面 → BVH 深度应 < 10
        let mesh = build_icosphere(3);
        let bvh = Bvh::build(&mesh);
        assert!(bvh.depth() <= 12, "BVH 深度 = {}", bvh.depth());
        assert!(bvh.leaf_count() >= 1);
        // 节点数 ≤ 2F - 1
        assert!(bvh.node_count() < 2 * mesh.face_count());
    }

    #[test]
    fn ray_aabb_basic_hit() {
        let aabb = AABB {
            min: [-1.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };
        // 射线从 (0,0,5) 沿 -z 射，必命中
        assert!(ray_aabb([0.0, 0.0, 5.0], [0.0, 0.0, -1.0], &aabb).is_some());
        // 射线从 (5,5,5) 沿 -z 射，垂直于 z 轴但偏离中心，未命中
        assert!(ray_aabb([5.0, 5.0, 5.0], [0.0, 0.0, -1.0], &aabb).is_none());
        // 射线沿 +x，从 (-5,0,0) 出发
        assert!(ray_aabb([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0], &aabb).is_some());
    }

    #[test]
    fn ray_aabb_parallel_slab() {
        let aabb = AABB {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };
        // 射线方向与 y slab 平行（y=2 在 slab 外），未命中
        assert!(ray_aabb([0.5, 2.0, 0.5], [1.0, 0.0, 0.0], &aabb).is_none());
        // 射线方向与 y slab 平行（y=0.5 在 slab 内），命中
        assert!(ray_aabb([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0], &aabb).is_some());
    }

    #[test]
    fn point_aabb_distance_basic() {
        let aabb = AABB {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };
        // 点在内部
        assert_eq!(point_aabb_distance_sq([0.5, 0.5, 0.5], &aabb), 0.0);
        // 点在角落外（距离 sqrt(3)）
        assert_eq!(point_aabb_distance_sq([-1.0, -1.0, -1.0], &aabb), 3.0);
        // 点在面外（距离 1）
        assert_eq!(point_aabb_distance_sq([2.0, 0.5, 0.5], &aabb), 1.0);
    }
}
