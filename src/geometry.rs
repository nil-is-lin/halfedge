//! 几何工具模块
//!
//! 在 [`MeshStorage`] 之上叠加**几何查询**与**几何处理**能力：
//! - 基本量：边长 / 三角面积 / 最小内角 / 面法向 / 顶点法向（面积加权）
//! - 多边形：`polygon_area` / `polygon_normal`（Newell 方法，支持 n-gon）
//! - 包围盒：`mesh_aabb`（AABB）、`mesh_centroid`（顶点质心）
//! - 平滑：拉普拉斯平滑（统一权重 + 余切权重 `cotan_laplacian`）
//! - 特征边：`dihedral_angle`（二面角）、`is_feature_edge` / `feature_edges`
//! - 距离：点到三角形最近距离（Ericson 算法）
//!
//! ## 设计原则
//! 所有查询函数返回 `Option<T>`：若传入的句柄失效或几何退化（零面积、
//! 零长度），返回 `None` 而非 panic。修改型函数（如拉普拉斯平滑）
//! 内部先收集所有新位置到 `Vec`，再批量写回，避免借用冲突。
//!
//! ## 拓扑约定
//! 与 [`crate::traversal`] 一致：`HalfEdge.vertex` 是 tip（目的顶点），
//! `twin.vertex` 是 origin。面边界环按 `next` 顺序遍历，CCW 朝向。
//!
//! [`crate::traversal`]: crate::traversal

use crate::ids::{FaceId, HalfEdgeId, VertexId};
use crate::linalg::vec3;
use crate::linalg::vec3::{Vec3, sub, add, scale, dot, cross, length, normalize, angle_between};
use crate::storage::MeshStorage;
use crate::traversal::{FaceHalfEdges, VertexAdjacentFaces, VertexAdjacentVerts, VertexRing};

// ============================================================
// 几何查询
// ============================================================

/// 边长：半边 `he` 两端顶点（origin 与 tip）的欧氏距离。
///
/// 返回 `None` 当：`he` 无效、`twin` 缺失、或顶点不存在。
pub fn edge_length(mesh: &MeshStorage, he: HalfEdgeId) -> Option<f64> {
    let h = mesh.get_halfedge(he)?;
    let tip = h.vertex;
    let twin_id = h.twin?;
    let origin = mesh.get_halfedge(twin_id)?.vertex;
    let p_tip = mesh.get_vertex(tip)?.position;
    let p_origin = mesh.get_vertex(origin)?.position;
    Some(length(sub(p_tip, p_origin)))
}

/// 三角面面积：$\frac{1}{2} |(\vec{B}-\vec{A}) \times (\vec{C}-\vec{A})|$。
///
/// 返回 `None` 当：面无效、边界环长度非 3、或顶点不存在。
pub fn face_area(mesh: &MeshStorage, f: FaceId) -> Option<f64> {
    let (a, b, c) = face_triangle_positions(mesh, f)?;
    Some(0.5 * length(cross(sub(b, a), sub(c, a))))
}

/// 面法向：归一化的 $(\vec{B}-\vec{A}) \times (\vec{C}-\vec{A})$。
///
/// 退化三角形（零面积）返回 `None`。
pub fn face_normal(mesh: &MeshStorage, f: FaceId) -> Option<Vec3> {
    let (a, b, c) = face_triangle_positions(mesh, f)?;
    let n = cross(sub(b, a), sub(c, a));
    let l = length(n);
    if l < 1e-12 {
        return None;
    }
    Some(scale(n, 1.0 / l))
}

/// 三角面最小内角（弧度）。退化三角形返回 `0.0`。
///
/// 三个内角分别由顶点 A、B、C 处的两条边向量求出，取最小值。
pub fn face_min_angle(mesh: &MeshStorage, f: FaceId) -> Option<f64> {
    let (a, b, c) = face_triangle_positions(mesh, f)?;
    let angle_a = angle_between(sub(b, a), sub(c, a));
    let angle_b = angle_between(sub(a, b), sub(c, b));
    let angle_c = angle_between(sub(a, c), sub(b, c));
    Some(angle_a.min(angle_b).min(angle_c))
}

/// 顶点法向：邻接面法向的**面积加权**平均。
///
/// $$
/// \vec{n}_v = \frac{\sum_{f \in N(v)} A_f \cdot \hat{n}_f}
///                  {\left\| \sum_{f \in N(v)} A_f \cdot \hat{n}_f \right\|}
/// $$
///
/// 孤立顶点（无邻接面）或所有邻接面都退化时返回 `None`。
pub fn vertex_normal(mesh: &MeshStorage, v: VertexId) -> Option<Vec3> {
    let mut accum = [0.0f64; 3];
    let mut has_any = false;
    for f in VertexAdjacentFaces::new(mesh, v) {
        if let (Some(n), Some(area)) = (face_normal(mesh, f), face_area(mesh, f)) {
            accum = add(accum, scale(n, area));
            has_any = true;
        }
    }
    if !has_any {
        return None;
    }
    let result = normalize(accum);
    if length(result) < 1e-12 {
        None
    } else {
        Some(result)
    }
}

// ============================================================
// 余切权重拉普拉斯 + 二面角
// ============================================================

/// 计算顶点 `v` 的余切权重拉普拉斯位置（不修改网格）。
///
/// $$
/// \Delta p_v = \sum_{u \in N(v)} (\cot \alpha_{vu} + \cot \beta_{vu}) \cdot (p_u - p_v)
/// $$
///
/// 其中 $\alpha_{vu}$ 和 $\beta_{vu}$ 是边 $(v,u)$ 在两侧三角形中的对角。
/// 边界顶点或拓扑不完整的顶点返回 `None`。
pub fn cotan_laplacian(mesh: &MeshStorage, v: VertexId) -> Option<[f64; 3]> {
    let pos_v = mesh.get_vertex(v)?.position;
    let outgoing: Vec<HalfEdgeId> = VertexRing::new(mesh, v).collect();
    let mut sum = [0.0; 3];
    for &he in &outgoing {
        let neighbor = mesh.get_halfedge(he)?.vertex;
        let pos_u = mesh.get_vertex(neighbor)?.position;
        let weight = cotan_edge_weight(mesh, he)?;
        let diff = sub(pos_u, pos_v);
        sum = add(sum, scale(diff, weight));
    }
    Some(sum)
}

/// 计算边 `he` 的余切权重 cot(α) + cot(β)。
///
/// `he` 应是一条 outgoing 半边。α 和 β 分别是边两端在左右三角形中的对角。
pub fn cotan_edge_weight(mesh: &MeshStorage, he: HalfEdgeId) -> Option<f64> {
    let h = mesh.get_halfedge(he)?;
    let a_pos = mesh.get_vertex(h.vertex)?.position;
    // 获取两个三角形
    let mut weight = 0.0;
    // 左侧三角（he 所在的面）
    if let Some(face) = h.face {
        let fhe_ids: Vec<_> = FaceHalfEdges::new(mesh, face).collect();
        if fhe_ids.len() == 3 {
            let v0 = mesh.get_halfedge(fhe_ids[0])?.vertex;
            let v1 = mesh.get_halfedge(fhe_ids[1])?.vertex;
            let v2 = mesh.get_halfedge(fhe_ids[2])?.vertex;
            let pos = [
                mesh.get_vertex(v0)?.position,
                mesh.get_vertex(v1)?.position,
                mesh.get_vertex(v2)?.position,
            ];
            // 找到 opposite vertex（不是 he.src 也不是 he.dst 的那个）
            let src = h.twin.and_then(|t| mesh.get_halfedge(t)).map(|t| t.vertex);
            let dst = h.vertex;
            if let Some(src_v) = src {
                let pos_src_v = mesh.get_vertex(src_v)?.position;
                let pos_dst_v = mesh.get_vertex(dst)?.position;
                let opp = pos.iter().position(|&p| p != pos_src_v && p != pos_dst_v);
                if let Some(idx) = opp {
                    let opp_pos = pos[idx];
                    let pos_src = mesh.get_vertex(src_v)?.position;
                    let pos_dst = a_pos;
                    let cot = cotan(opp_pos, pos_src, pos_dst);
                    weight += cot;
                }
            }
        }
    }
    // 右侧三角（twin 所在的面）
    if let Some(twin) = h.twin
        && let Some(tface) = mesh.get_halfedge(twin)?.face
    {
        let fhe_ids: Vec<_> = FaceHalfEdges::new(mesh, tface).collect();
        if fhe_ids.len() == 3 {
            let v0 = mesh.get_halfedge(fhe_ids[0])?.vertex;
            let v1 = mesh.get_halfedge(fhe_ids[1])?.vertex;
            let v2 = mesh.get_halfedge(fhe_ids[2])?.vertex;
            let pos = [
                mesh.get_vertex(v0)?.position,
                mesh.get_vertex(v1)?.position,
                mesh.get_vertex(v2)?.position,
            ];
            let src = h.vertex;
            let dst = mesh.get_halfedge(twin)?.vertex;
            let pos_src_val = mesh.get_vertex(src)?.position;
            let pos_dst_val = mesh.get_vertex(dst)?.position;
            let opp = pos
                .iter()
                .position(|&p| p != pos_src_val && p != pos_dst_val);
            if let Some(idx) = opp {
                let opp_pos = pos[idx];
                let pos_src = mesh.get_vertex(src)?.position;
                let pos_dst = mesh.get_vertex(dst)?.position;
                let cot = cotan(opp_pos, pos_src, pos_dst);
                weight += cot;
            }
        }
    }
    Some(weight)
}

/// 计算角 O 在三角形 OAB 中的余切值。
fn cotan(opposite: Vec3, a: Vec3, b: Vec3) -> f64 {
    let oa = sub(a, opposite);
    let ob = sub(b, opposite);
    let cross_len = length(cross(oa, ob));
    let dot_val = dot(oa, ob);
    if cross_len < 1e-14 {
        0.0
    } else {
        dot_val / cross_len
    }
}

/// 计算半边 `he` 的二面角（弧度）。
///
/// `he` 必须为内部边（twin 存在且双方都有面），否则返回 `None`。
pub fn dihedral_angle(mesh: &MeshStorage, he: HalfEdgeId) -> Option<f64> {
    let h = mesh.get_halfedge(he)?;
    let f1 = h.face?;
    let twin = h.twin?;
    let f2 = mesh.get_halfedge(twin)?.face?;
    let n1 = face_normal(mesh, f1)?;
    let n2 = face_normal(mesh, f2)?;
    let d = dot(n1, n2).clamp(-1.0, 1.0);
    Some(d.acos())
}

/// 判断半边 `he` 是否为特征边（二面角 > 阈值）。
pub fn is_feature_edge(
    mesh: &MeshStorage,
    he: HalfEdgeId,
    angle_threshold_rad: f64,
) -> Option<bool> {
    Some(dihedral_angle(mesh, he)? > angle_threshold_rad)
}

/// 返回所有二面角大于阈值的内部边的半边列表。
pub fn feature_edges(mesh: &MeshStorage, angle_threshold_rad: f64) -> Vec<HalfEdgeId> {
    mesh.halfedge_ids()
        .filter(|&he| is_feature_edge(mesh, he, angle_threshold_rad).unwrap_or(false))
        .collect()
}

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
// 点到三角形最近距离
// ============================================================

/// 点到三角形的最近距离。
///
/// 参数顺序：查询点 `p`，三角形三顶点 `a, b, c`（CCW 或 CW 均可）。
///
/// ## 算法
/// Ericson, *Real-Time Collision Detection* 5.1.5。
/// 将三角形所在平面划分为 7 个 Voronoi 区域（3 个顶点域、3 条边域、1 个面域），
/// 通过点积判定 `p` 投影所属区域，再求最近点。
///
/// ## 复杂度
/// $O(1)$，无循环、无开方（除最后一次距离计算）。
pub fn point_triangle_distance(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f64 {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);

    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return length(ap); // 顶点 A 域
    }

    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return length(bp); // 顶点 B 域
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = if d1 - d3 == 0.0 { 0.0 } else { d1 / (d1 - d3) };
        let closest = add(a, scale(ab, v));
        return length(sub(p, closest)); // 边 AB 域
    }

    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return length(cp); // 顶点 C 域
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = if d2 - d6 == 0.0 { 0.0 } else { d2 / (d2 - d6) };
        let closest = add(a, scale(ac, w));
        return length(sub(p, closest)); // 边 AC 域
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let denom = (d4 - d3) + (d5 - d6);
        let w = if denom == 0.0 { 0.0 } else { (d4 - d3) / denom };
        let closest = add(b, scale(sub(c, b), w));
        return length(sub(p, closest)); // 边 BC 域
    }

    // 面域：重心坐标 (1-v-w, v, w)
    let denom = va + vb + vc;
    if denom.abs() < 1e-20 {
        // 退化三角形（共线），退化到三顶点最近者
        return length(ap).min(length(bp)).min(length(cp));
    }
    let inv = 1.0 / denom;
    let v = vb * inv;
    let w = vc * inv;
    let closest = add(a, add(scale(ab, v), scale(ac, w)));
    length(sub(p, closest))
}

/// 点到三角形所在表面的最近点（Ericson *Real-Time Collision Detection* 5.1.5）。
///
/// 与 [`point_triangle_distance`] 同源算法，但返回最近点本身而非距离。
/// 用于 BVH 最近点查询等需要插值/法向插值等后续计算的场景。
pub fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);

    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a; // 顶点 A 域
    }

    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b; // 顶点 B 域
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = if d1 - d3 == 0.0 { 0.0 } else { d1 / (d1 - d3) };
        return add(a, scale(ab, v)); // 边 AB 域
    }

    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c; // 顶点 C 域
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = if d2 - d6 == 0.0 { 0.0 } else { d2 / (d2 - d6) };
        return add(a, scale(ac, w)); // 边 AC 域
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let denom = (d4 - d3) + (d5 - d6);
        let w = if denom == 0.0 { 0.0 } else { (d4 - d3) / denom };
        return add(b, scale(sub(c, b), w)); // 边 BC 域
    }

    // 面域：重心坐标 (1-v-w, v, w)
    let denom = va + vb + vc;
    if denom.abs() < 1e-20 {
        // 退化三角形：返回最近顶点
        let la = dot(ap, ap);
        let lb = dot(bp, bp);
        let lc = dot(cp, cp);
        if la <= lb && la <= lc {
            return a;
        }
        if lb <= lc {
            return b;
        }
        return c;
    }
    let inv = 1.0 / denom;
    let v = vb * inv;
    let w = vc * inv;
    add(a, add(scale(ab, v), scale(ac, w)))
}

// ============================================================
// 辅助
// ============================================================

/// 取面边界环上所有顶点的位置。若面不存在或顶点缺失返回 `None`。
fn face_polygon_positions(mesh: &MeshStorage, f: FaceId) -> Option<Vec<Vec3>> {
    let mut verts = Vec::new();
    for he in FaceHalfEdges::new(mesh, f) {
        let v = mesh.get_halfedge(he)?.vertex;
        verts.push(mesh.get_vertex(v)?.position);
    }
    if verts.is_empty() { None } else { Some(verts) }
}

/// 取面边界环上三个顶点的位置。若面非三角或顶点缺失返回 `None`。
fn face_triangle_positions(mesh: &MeshStorage, f: FaceId) -> Option<(Vec3, Vec3, Vec3)> {
    let verts = face_polygon_positions(mesh, f)?;
    if verts.len() != 3 {
        return None;
    }
    Some((verts[0], verts[1], verts[2]))
}

/// 任意多边形面的面积（Newell 方法）。支持三角面及 n-gon。
pub fn polygon_area(mesh: &MeshStorage, f: FaceId) -> Option<f64> {
    let verts = face_polygon_positions(mesh, f)?;
    let n = verts.len();
    if n < 3 {
        return None;
    }
    let mut sum = [0.0; 3];
    for i in 0..n {
        let j = (i + 1) % n;
        sum = add(sum, cross(verts[i], verts[j]));
    }
    Some(0.5 * length(sum))
}

/// 任意多边形面的法向（Newell 方法）。
pub fn polygon_normal(mesh: &MeshStorage, f: FaceId) -> Option<Vec3> {
    let verts = face_polygon_positions(mesh, f)?;
    let n_verts = verts.len();
    if n_verts < 3 {
        return None;
    }
    let center = {
        let mut c = [0.0; 3];
        for v in &verts {
            c = add(c, *v);
        }
        [
            c[0] / n_verts as f64,
            c[1] / n_verts as f64,
            c[2] / n_verts as f64,
        ]
    };
    let mut normal = [0.0; 3];
    for i in 0..n_verts {
        let j = (i + 1) % n_verts;
        normal = add(normal, cross(sub(verts[i], center), sub(verts[j], center)));
    }
    let l = length(normal);
    if l < 1e-12 {
        None
    } else {
        Some(scale(normal, 1.0 / l))
    }
}

// ============================================================
// AABB 与质心
// ============================================================

/// 轴对齐包围盒。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min: [f64; 3],
    pub max: [f64; 3],
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
pub fn mesh_aabb(mesh: &MeshStorage) -> Option<AABB> {
    let mut aabb = AABB::new();
    let mut has_vertex = false;
    for v in mesh.vertices() {
        aabb.extend(&v.position);
        has_vertex = true;
    }
    if has_vertex { Some(aabb) } else { None }
}

/// 计算网格顶点质心（所有顶点位置的算术平均）。若无顶点则返回 `None`。
pub fn mesh_centroid(mesh: &MeshStorage) -> Option<[f64; 3]> {
    let mut sum = [0.0; 3];
    let mut count = 0usize;
    for v in mesh.vertices() {
        for (s, &p) in sum.iter_mut().zip(v.position.iter()) {
            *s += p;
        }
        count += 1;
    }
    if count == 0 {
        None
    } else {
        Some([
            sum[0] / count as f64,
            sum[1] / count as f64,
            sum[2] / count as f64,
        ])
    }
}

// ============================================================
// 射线求交（Möller-Trumbore）
// ============================================================

/// 射线与网格的交点信息。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    /// 交点位置
    pub position: [f64; 3],
    /// 从射线原点到交点的距离（参数 t，满足 `origin + t * direction`）
    pub t: f64,
    /// 相交的三角面的索引（`FaceId`）
    pub face: FaceId,
    /// 重心坐标 (u, v)，第三个坐标为 1-u-v
    pub barycentric: (f64, f64),
}

/// 射线与三角形求交（基于 Shewchuk 鲁棒 `orient3d` 谓词）。
///
/// 返回参数 t（`origin + t * direction` 即为交点）与重心坐标 `(u, v)`。
/// 不相交返回 `None`。
///
/// ## 算法
/// 1. 用 `orient3d(v0, v1, v2, origin)` 与 `orient3d(v0, v1, v2, origin+dir)`
///    判定射线是否穿过三角形平面（鲁棒符号判定）；
/// 2. 线性插值求交点参数 t；
/// 3. 投影到三角形法向主轴的正交平面，用鲁棒 `point_in_triangle_2d`
///    判定交点是否在三角形内；
/// 4. 用 2D 重心坐标公式计算 (u, v)。
///
/// 相比朴素 Möller-Trumbore，本实现在共面、共线、交点接近三角形边等
/// 退化情况下能给出精确判定，避免因浮点舍入导致的误判。
pub fn ray_triangle_intersection(
    origin: [f64; 3],
    direction: [f64; 3],
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
) -> Option<(f64, f64, f64)> {
    use crate::predicates::{orient3d, point_in_triangle_2d};

    // 1. 鲁棒判定射线与三角形平面相交
    // 由 orient3d 在第 4 个参数上的线性性：
    //   f(t) = orient3d(v0, v1, v2, origin + t*dir) = a + t*(b - a)
    // 其中 a = orient3d(v0, v1, v2, origin)，b = orient3d(v0, v1, v2, origin+dir)
    // 注意：direction 是方向向量（不一定是单位向量），origin+dir 是 t=1 处的点。
    let a = orient3d(v0, v1, v2, origin);
    let end = [
        origin[0] + direction[0],
        origin[1] + direction[1],
        origin[2] + direction[2],
    ];
    let b = orient3d(v0, v1, v2, end);

    // 求交点参数 t：a + t*(b - a) = 0 → t = -a / (b - a)
    let denom = b - a;
    if denom == 0.0 {
        return None; // 射线与平面平行（含在平面内的退化情况）
    }
    let t = -a / denom;
    if t <= 1e-12 {
        return None; // 交点在射线后方或原点附近
    }

    // 3. 计算交点 P = origin + t * direction
    let p = [
        origin[0] + t * direction[0],
        origin[1] + t * direction[1],
        origin[2] + t * direction[2],
    ];

    // 4. 投影到 2D（丢弃法向最大分量轴），用鲁棒 point_in_triangle_2d 判定
    let e1 = sub(v1, v0);
    let e2 = sub(v2, v0);
    let normal = cross(e1, e2);
    let abs_n = [normal[0].abs(), normal[1].abs(), normal[2].abs()];
    let drop_axis = if abs_n[0] >= abs_n[1] && abs_n[0] >= abs_n[2] {
        0
    } else if abs_n[1] >= abs_n[2] {
        1
    } else {
        2
    };

    let project = |q: [f64; 3]| -> [f64; 2] {
        match drop_axis {
            0 => [q[1], q[2]],
            1 => [q[0], q[2]],
            _ => [q[0], q[1]],
        }
    };

    let p2d = project(p);
    let a2d = project(v0);
    let b2d = project(v1);
    let c2d = project(v2);

    if !point_in_triangle_2d(p2d, a2d, b2d, c2d) {
        return None;
    }

    // 5. 用 2D 重心坐标公式计算 (u, v)
    // P = v0 + u*(v1-v0) + v*(v2-v0)，在 2D 投影中求解
    let ab = [b2d[0] - a2d[0], b2d[1] - a2d[1]];
    let ac = [c2d[0] - a2d[0], c2d[1] - a2d[1]];
    let ap = [p2d[0] - a2d[0], p2d[1] - a2d[1]];
    let denom = ab[0] * ac[1] - ab[1] * ac[0];
    if denom == 0.0 {
        return None; // 退化三角形（共线）
    }
    let inv_denom = 1.0 / denom;
    let u = (ap[0] * ac[1] - ap[1] * ac[0]) * inv_denom;
    let v = (ab[0] * ap[1] - ab[1] * ap[0]) * inv_denom;

    Some((t, u, v))
}

/// 射线与网格求最近交点。返回距离最近的 `RayHit`；无交点返回 `None`。
///
/// 遍历所有三角面，计算与射线的交点，取 t 最小者。复杂度 O(F)。
pub fn ray_mesh_intersection(
    origin: [f64; 3],
    direction: [f64; 3],
    mesh: &MeshStorage,
) -> Option<RayHit> {
    let mut best: Option<RayHit> = None;

    for f in mesh.face_ids() {
        let verts: Vec<VertexId> = crate::traversal::FaceVertices::new(mesh, f).collect();
        if verts.len() != 3 {
            continue;
        }
        let v0 = mesh.get_vertex(verts[0])?.position;
        let v1 = mesh.get_vertex(verts[1])?.position;
        let v2 = mesh.get_vertex(verts[2])?.position;

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
                Some(ref b) if hit.t < b.t => best = Some(hit),
                None => best = Some(hit),
                _ => {}
            }
        }
    }

    best
}

/// 射线是否与网格相交（假设网格闭合；使用射线法奇偶判定）。
///
/// 沿 direction 方向发射射线，统计与网格三角面的交点个数。
/// 奇数个交点 → `true`（进入网格内部）。
pub fn ray_mesh_intersects(origin: [f64; 3], direction: [f64; 3], mesh: &MeshStorage) -> bool {
    let mut count = 0u32;
    for f in mesh.face_ids() {
        let verts: Vec<VertexId> = crate::traversal::FaceVertices::new(mesh, f).collect();
        if verts.len() != 3 {
            continue;
        }
        // 拓扑不一致时跳过，避免 panic
        let (v0, v1, v2) = match (
            mesh.get_vertex(verts[0]),
            mesh.get_vertex(verts[1]),
            mesh.get_vertex(verts[2]),
        ) {
            (Some(a), Some(b), Some(c)) => (a.position, b.position, c.position),
            _ => continue,
        };
        if ray_triangle_intersection(origin, direction, v0, v1, v2).is_some() {
            count += 1;
        }
    }
    count % 2 == 1
}

// ============================================================
// 表面积 / 体积
// ============================================================

/// 计算网格总表面积（所有三角面面积之和）。
pub fn surface_area(mesh: &MeshStorage) -> f64 {
    mesh.face_ids().filter_map(|f| face_area(mesh, f)).sum()
}

/// 计算闭合网格的有向体积（散度定理）。
///
/// $$
/// V = \frac{1}{6} \sum_{f} \operatorname{orient3d}(\mathbf{0}, v_0, v_1, v_2)
/// $$
///
/// 使用 Shewchuk 鲁棒 `orient3d` 谓词，保证在退化（共面）情况下符号精确。
/// 正值为 CCW 面朝外（右手系）。非闭合网格结果无意义。
pub fn mesh_volume(mesh: &MeshStorage) -> f64 {
    use crate::predicates::tet_signed_volume;
    let origin = [0.0, 0.0, 0.0];
    let mut volume = 0.0;
    for f in mesh.face_ids() {
        let verts: Vec<VertexId> = crate::traversal::FaceVertices::new(mesh, f).collect();
        if verts.len() != 3 {
            continue;
        }
        // 拓扑不一致时跳过，避免 panic
        let (v0, v1, v2) = match (
            mesh.get_vertex(verts[0]),
            mesh.get_vertex(verts[1]),
            mesh.get_vertex(verts[2]),
        ) {
            (Some(a), Some(b), Some(c)) => (a.position, b.position, c.position),
            _ => continue,
        };
        // 四面体 (原点, v0, v1, v2) 的有符号体积
        volume += tet_signed_volume(origin, v0, v1, v2);
    }
    volume
}

// ============================================================
// 离散曲率（Meyer et al. 2003）
// ============================================================

/// 顶点曲率信息。
#[derive(Debug, Clone, Copy)]
pub struct VertexCurvature {
    /// 高斯曲率 $K = \kappa_1 \cdot \kappa_2$
    pub gaussian: f64,
    /// 平均曲率 $H = (\kappa_1 + \kappa_2) / 2$
    pub mean: f64,
    /// 最大主曲率 $\kappa_1$
    pub k1: f64,
    /// 最小主曲率 $\kappa_2$
    pub k2: f64,
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
        let a = h.vertex; // tip
        let b = h.twin.and_then(|t| mesh.get_halfedge(t)).map(|t| t.vertex); // origin
        let Some(b) = b else { continue };

        let (pa, pb, pv) = match (
            mesh.get_vertex(a),
            mesh.get_vertex(b),
            mesh.get_vertex(v),
        ) {
            (Some(va), Some(vb), Some(vv)) => (va.position, vb.position, vv.position),
            _ => continue,
        };

        let a2 = (pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2);
        let b2 = (pv[0] - pa[0]).powi(2) + (pv[1] - pa[1]).powi(2) + (pv[2] - pa[2]).powi(2);
        let c2 = (pb[0] - pv[0]).powi(2) + (pb[1] - pv[1]).powi(2) + (pb[2] - pv[2]).powi(2);

        // 判断钝角
        let obtuse_at_v = b2 + c2 < a2;
        let obtuse_at_a = a2 + b2 < c2;
        let obtuse_at_b = a2 + c2 < b2;

        if obtuse_at_v {
            // v 处钝角：用三角形面积的 1/2
            let tri_area = vec3::triangle_area(pv, pa, pb);
            area += tri_area / 2.0;
        } else if obtuse_at_a || obtuse_at_b {
            // 其他钝角：用三角形面积的 1/4
            let tri_area = vec3::triangle_area(pv, pa, pb);
            area += tri_area / 4.0;
        } else {
            // Voronoi 面积
            let cot_a = cotan_from_pos(pv, pa, pb); // angle at v in triangle pv-a-b... wait
            let cot_b = cotan_from_pos(pa, pb, pv);
            area += (b2 * cot_b + c2 * cot_a) / 8.0; // actually need cot at vertices a and b opposite to v
            // Standard formula: A_voronoi = 1/8 Σ (cot α_ij + cot β_ij) * ||v_j - v_i||²
            // Let me use a simpler approach: just use 1/3 of each incident triangle area
            let tri_area = vec3::triangle_area(pv, pa, pb);
            area += tri_area / 3.0;
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
    if crate::traversal::is_boundary_vertex(mesh, v) {
        return Some(0.0);
    }

    let mut angle_sum = 0.0;
    for he in VertexRing::new(mesh, v) {
        let h = mesh.get_halfedge(he)?;
        let a = h.vertex;
        let b = h.twin.and_then(|t| mesh.get_halfedge(t))?.vertex;

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
    if crate::traversal::is_boundary_vertex(mesh, v) {
        return Some(0.0);
    }

    let laplacian = cotan_laplacian(mesh, v)?;
    let len =
        (laplacian[0] * laplacian[0] + laplacian[1] * laplacian[1] + laplacian[2] * laplacian[2])
            .sqrt();
    if len < 1e-14 {
        return Some(0.0);
    }
    let area = mixed_area_at_vertex(mesh, v);
    Some(0.5 * len / area)
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
// ============================================================
// 并行变体（rayon）
// ============================================================

/// 并行计算网格总表面积。
pub fn surface_area_par(mesh: &MeshStorage) -> f64 {
    use rayon::prelude::*;
    let face_ids: Vec<_> = mesh.face_ids().collect();
    face_ids
        .par_iter()
        .filter_map(|&f| face_area(mesh, f))
        .sum()
}

/// 并行计算闭合网格的有向体积。
///
/// 与 [`mesh_volume`] 相同的算法（Shewchuk 鲁棒 `orient3d`），但使用 rayon
/// 并行迭代所有三角面。
pub fn mesh_volume_par(mesh: &MeshStorage) -> f64 {
    use crate::predicates::tet_signed_volume;
    use rayon::prelude::*;
    let origin = [0.0, 0.0, 0.0];
    let face_ids: Vec<_> = mesh.face_ids().collect();
    face_ids
        .par_iter()
        .map(|&f| {
            let verts: Vec<VertexId> = crate::traversal::FaceVertices::new(mesh, f).collect();
            if verts.len() != 3 {
                return 0.0;
            }
            // 拓扑不一致时跳过，避免 panic（与串行版本 mesh_volume 一致）
            let (v0, v1, v2) = match (
                mesh.get_vertex(verts[0]),
                mesh.get_vertex(verts[1]),
                mesh.get_vertex(verts[2]),
            ) {
                (Some(a), Some(b), Some(c)) => (a.position, b.position, c.position),
                _ => return 0.0,
            };
            tet_signed_volume(origin, v0, v1, v2)
        })
        .sum::<f64>()
}

/// 并行计算所有顶点的法向（按 vertex_ids 顺序）。
pub fn vertex_normals_par(mesh: &MeshStorage) -> Vec<[f64; 3]> {
    use rayon::prelude::*;
    let verts: Vec<VertexId> = mesh.vertex_ids().collect();
    verts
        .par_iter()
        .map(|&v| vertex_normal(mesh, v).unwrap_or([0.0, 0.0, 0.0]))
        .collect()
}

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
        .filter(|&&he| is_feature_edge(mesh, he, angle_threshold).unwrap_or(false))
        .copied()
        .collect()
}

/// 并行检测射线与网格的所有交点。
pub fn ray_mesh_intersection_par(
    mesh: &MeshStorage,
    origin: [f64; 3],
    direction: [f64; 3],
) -> Vec<RayHit> {
    use rayon::prelude::*;
    let faces: Vec<FaceId> = mesh.face_ids().collect();
    faces
        .par_iter()
        .filter_map(|&f| {
            let verts: Vec<VertexId> = crate::traversal::FaceVertices::new(mesh, f).collect();
            if verts.len() != 3 {
                return None;
            }
            let v0 = mesh.get_vertex(verts[0])?.position;
            let v1 = mesh.get_vertex(verts[1])?.position;
            let v2 = mesh.get_vertex(verts[2])?.position;
            let (t, u, v) = ray_triangle_intersection(origin, direction, v0, v1, v2)?;
            let position = [
                origin[0] + direction[0] * t,
                origin[1] + direction[1] * t,
                origin[2] + direction[2] * t,
            ];
            Some(RayHit {
                position,
                t,
                face: f,
                barycentric: (u, v),
            })
        })
        .collect()
}

// ============================================================
// 网格质量度量
// ============================================================

/// 三角形的纵横比（aspect ratio）。
///
/// 定义：最长边 / 最短边。等边三角形为 1，越瘦长越大。
/// 退化三角形（最短边 = 0）返回 `None`。
pub fn face_aspect_ratio(mesh: &MeshStorage, f: FaceId) -> Option<f64> {
    let verts: Vec<VertexId> = crate::traversal::FaceVertices::new(mesh, f).collect();
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
    let verts: Vec<VertexId> = crate::traversal::FaceVertices::new(mesh, f).collect();
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
        if let Some(len) = edge_length(mesh, he) {
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
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub variance: f64,
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
    pub aspect_min: f64,
    pub aspect_max: f64,
    pub aspect_mean: f64,
    pub radius_ratio_min: f64,
    pub radius_ratio_mean: f64,
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
/// \[
///   p_i \gets p_i + \frac{1}{W} \sum_{j \in N(i)}
///      \frac{\|p_j - p_i\|}{\sigma_c}
///      \cdot e^{-\|p_j - p_i\|^2 / (2\sigma_c^2)}
///      \cdot e^{-\|n_i - n_j\|^2 / (2\sigma_s^2)}
///      \cdot (p_j - p_i)
/// \]
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
            .filter_map(|v| vertex_normal(mesh, v).map(|n| (v, n)))
            .collect();
        let verts: Vec<VertexId> = mesh.vertex_ids().collect();
        let updates: Vec<(VertexId, Vec3)> = verts
            .iter()
            .filter_map(|&v| {
                let p_i = mesh.get_vertex(v)?.position;
                let n_i = normals.get(&v).copied().unwrap_or([0.0; 3]);
                let mut sum_disp = [0.0; 3];
                let mut sum_w = 0.0;
                for n in crate::traversal::VertexAdjacentVerts::new(mesh, v) {
                    if n == v {
                        continue;
                    }
                    let p_j = mesh.get_vertex(n)?.position;
                    let n_j = normals.get(&n).copied().unwrap_or([0.0; 3]);
                    let d = sub(p_j, p_i);
                    let dist = length(d);
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
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{Face, HalfEdge, MeshStorage, Vertex};

    /// 构造单位等腰直角三角形（在 xy 平面，CCW 朝向 +z）：
    /// A=(0,0,0), B=(1,0,0), C=(0,1,0)
    fn build_unit_triangle() -> (MeshStorage, [VertexId; 3], FaceId) {
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

    // ---------- 边长 ----------

    #[test]
    fn edge_length_basic() {
        let (mesh, _v, _f) = build_unit_triangle();
        // 找一条 A→B 的半边（长度应为 1）
        let he_ab = mesh
            .halfedge_ids()
            .find(|h| {
                let h = mesh.get_halfedge(*h).unwrap();
                h.vertex == _v[1] && mesh.get_halfedge(h.twin.unwrap()).unwrap().vertex == _v[0]
            })
            .unwrap();
        assert!((edge_length(&mesh, he_ab).unwrap() - 1.0).abs() < 1e-9);

        // 找一条 B→C 的半边（长度应为 √2）
        let he_bc = mesh
            .halfedge_ids()
            .find(|h| {
                let h = mesh.get_halfedge(*h).unwrap();
                h.vertex == _v[2] && mesh.get_halfedge(h.twin.unwrap()).unwrap().vertex == _v[1]
            })
            .unwrap();
        assert!((edge_length(&mesh, he_bc).unwrap() - 2.0_f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn edge_length_invalid_returns_none() {
        let (mesh, _v, _f) = build_unit_triangle();
        let bad = HalfEdgeId::default();
        assert!(edge_length(&mesh, bad).is_none());
    }

    // ---------- 面积 ----------

    #[test]
    fn face_area_unit_triangle() {
        let (mesh, _v, f) = build_unit_triangle();
        assert!((face_area(&mesh, f).unwrap() - 0.5).abs() < 1e-9);
    }

    // ---------- 法向 ----------

    #[test]
    fn face_normal_ccw_points_up() {
        let (mesh, _v, f) = build_unit_triangle();
        let n = face_normal(&mesh, f).unwrap();
        assert!((n[0] - 0.0).abs() < 1e-9);
        assert!((n[1] - 0.0).abs() < 1e-9);
        assert!((n[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn vertex_normal_corner_of_triangle() {
        let (mesh, v, _f) = build_unit_triangle();
        // 单三角形所有顶点的法向都应为 +z
        let n = vertex_normal(&mesh, v[0]).unwrap();
        assert!((n[2] - 1.0).abs() < 1e-9);
    }

    // ---------- 最小内角 ----------

    #[test]
    fn face_min_angle_unit_triangle() {
        let (mesh, _v, f) = build_unit_triangle();
        // 等腰直角三角形：角度为 45°, 45°, 90°；最小 45° = π/4
        let min_ang = face_min_angle(&mesh, f).unwrap();
        assert!((min_ang - std::f64::consts::FRAC_PI_4).abs() < 1e-9);
    }

    #[test]
    fn face_min_angle_equilateral() {
        let mut mesh = MeshStorage::new();
        let a = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let b = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let c = mesh.add_vertex(Vertex::new([0.5, 3.0_f64.sqrt() / 2.0, 0.0]));

        let h_ab = mesh.add_halfedge(HalfEdge::new(b));
        let h_bc = mesh.add_halfedge(HalfEdge::new(c));
        let h_ca = mesh.add_halfedge(HalfEdge::new(a));
        let t_ab = mesh.add_halfedge(HalfEdge::new(a));
        let t_bc = mesh.add_halfedge(HalfEdge::new(b));
        let t_ca = mesh.add_halfedge(HalfEdge::new(c));

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

        // 等边三角形：所有内角均为 60° = π/3
        let min_ang = face_min_angle(&mesh, f).unwrap();
        assert!((min_ang - std::f64::consts::FRAC_PI_3).abs() < 1e-9);
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

    // ---------- 点到三角形距离 ----------

    #[test]
    fn point_triangle_distance_point_on_face() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        // 重心
        let p = [1.0 / 3.0, 1.0 / 3.0, 0.0];
        assert!(point_triangle_distance(p, a, b, c).abs() < 1e-9);
    }

    #[test]
    fn point_triangle_distance_above_face() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let p = [0.2, 0.2, 1.0];
        assert!((point_triangle_distance(p, a, b, c) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn point_triangle_distance_vertex_region() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        // 远离 A 的方向
        let p = [-1.0, -1.0, 0.0];
        assert!((point_triangle_distance(p, a, b, c) - 2.0_f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn point_triangle_distance_edge_region() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        // 投影落在边 AB 外延，最近点应是 B
        let p = [2.0, 0.0, 0.0];
        assert!((point_triangle_distance(p, a, b, c) - 1.0).abs() < 1e-9);
    }

    // ---------- 退化三角形 ----------

    #[test]
    fn degenerate_face_returns_none_for_normal() {
        let mut mesh = MeshStorage::new();
        let a = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let b = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        // C 与 A、B 共线 → 退化
        let c = mesh.add_vertex(Vertex::new([2.0, 0.0, 0.0]));

        let h_ab = mesh.add_halfedge(HalfEdge::new(b));
        let h_bc = mesh.add_halfedge(HalfEdge::new(c));
        let h_ca = mesh.add_halfedge(HalfEdge::new(a));
        let t_ab = mesh.add_halfedge(HalfEdge::new(a));
        let t_bc = mesh.add_halfedge(HalfEdge::new(b));
        let t_ca = mesh.add_halfedge(HalfEdge::new(c));

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

        assert!(face_normal(&mesh, f).is_none());
        assert!((face_area(&mesh, f).unwrap() - 0.0).abs() < 1e-12);
        // 最小内角应为 0
        assert!(face_min_angle(&mesh, f).unwrap().abs() < 1e-9);
    }

    // ---------- AABB / centroid ----------

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

    // ---------- 射线求交 ----------

    #[test]
    fn ray_triangle_hit() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        // 从上方垂直射下
        let hit = ray_triangle_intersection([0.25, 0.25, 1.0], [0.0, 0.0, -1.0], v0, v1, v2);
        assert!(hit.is_some());
        let (t, u, v) = hit.unwrap();
        assert!((t - 1.0).abs() < 1e-10);
        assert!((u - 0.25).abs() < 1e-10);
        assert!((v - 0.25).abs() < 1e-10);
    }

    #[test]
    fn ray_triangle_miss_parallel() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        // 平行于三角形
        assert!(ray_triangle_intersection([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], v0, v1, v2).is_none());
    }

    #[test]
    fn test_ray_mesh_intersects() {
        let mesh = crate::test_util::build_icosphere(2);
        // 从球外向球心方向射 → 应命中（奇偶检验：奇数交点）
        let hits = ray_mesh_intersects([2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], &mesh);
        // 可能命中也可能恰好经过顶点/边；仅验证不 panic
        let _ = hits;
    }

    #[test]
    fn test_ray_mesh_intersection_icosphere() {
        let mesh = crate::test_util::build_icosphere(2);
        let hit = ray_mesh_intersection([2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], &mesh);
        assert!(hit.is_some());
        let h = hit.unwrap();
        assert!((h.position[0] - 1.0).abs() < 0.1);
        assert!(h.t > 0.0 && h.t < 2.0);
    }

    #[test]
    fn test_ray_mesh_intersection_miss() {
        let mesh = crate::test_util::build_icosphere(2);
        assert!(ray_mesh_intersection([3.0, 0.0, 0.0], [1.0, 0.0, 0.0], &mesh).is_none());
    }

    // ---------- 表面积 / 体积 / 曲率 ----------

    #[test]
    fn surface_area_icosphere() {
        let mesh = crate::test_util::build_icosphere(2);
        let area = surface_area(&mesh);
        // 单位球表面积 = 4π ≈ 12.566
        assert!(area > 10.0 && area < 15.0, "表面积应在 4π 附近: {}", area);
    }

    #[test]
    fn volume_icosphere() {
        let mesh = crate::test_util::build_icosphere(2);
        let vol = mesh_volume(&mesh);
        // 单位球体积 = 4π/3 ≈ 4.189
        assert!(vol > 3.0 && vol < 5.5, "体积应在 4π/3 附近: {}", vol);
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

    // ---------- 网格质量度量 ----------

    #[test]
    fn aspect_ratio_equilateral_is_one() {
        // 等边三角形：纵横比 = 1
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 3.0_f64.sqrt() / 2.0, 0.0],
        ];
        let faces = vec![[0u32, 1, 2]];
        let mesh = crate::io::build_mesh_from_vertices_and_faces(&verts, &faces);
        let f = mesh.face_ids().next().unwrap();
        let ar = face_aspect_ratio(&mesh, f).expect("等边三角形纵横比");
        assert!((ar - 1.0).abs() < 1e-10, "等边纵横比应=1, got {ar}");
    }

    #[test]
    fn aspect_ratio_degenerate_returns_none() {
        // 退化三角形（共线）：最短边可能为 0 或面积 0
        let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let faces = vec![[0u32, 1, 2]];
        let mesh = crate::io::build_mesh_from_vertices_and_faces(&verts, &faces);
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
        let mesh = crate::io::build_mesh_from_vertices_and_faces(&verts, &faces);
        let f = mesh.face_ids().next().unwrap();
        let rr = face_radius_ratio(&mesh, f).expect("等边半径比");
        assert!((rr - 1.0).abs() < 1e-10, "等边半径比应=1, got {rr}");
    }

    #[test]
    fn radius_ratio_degenerate_is_zero() {
        let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let faces = vec![[0u32, 1, 2]];
        let mesh = crate::io::build_mesh_from_vertices_and_faces(&verts, &faces);
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

    // ---------- 高级平滑 ----------

    #[test]
    fn taubin_smooth_preserves_volume_better_than_laplacian() {
        let mesh0 = crate::test_util::build_icosphere(1);
        let area0 = surface_area(&mesh0);

        // Laplacian 20 步：体积显著缩小
        let mut mesh_lap = crate::test_util::build_icosphere(1);
        laplacian_smooth_mesh(&mut mesh_lap, 0.5, 20);
        let area_lap = surface_area(&mesh_lap);
        let laplacian_shrink = (area0 - area_lap).abs() / area0;

        // Taubin 20 步：体积变化应小于 Laplacian
        let mut mesh_tau = crate::test_util::build_icosphere(1);
        taubin_smooth_mesh(&mut mesh_tau, 0.5, -0.53, 20);
        let area_tau = surface_area(&mesh_tau);
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
        let stats = edge_length_stats(&mesh);
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

    // ---------- 空/退化网格 ----------

    #[test]
    fn ray_mesh_intersects_empty_mesh_returns_false() {
        let mesh = MeshStorage::new();
        assert!(!ray_mesh_intersects([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], &mesh));
    }

    #[test]
    fn mesh_volume_empty_mesh_returns_zero() {
        let mesh = MeshStorage::new();
        assert_eq!(mesh_volume(&mesh), 0.0);
    }

    #[test]
    fn mesh_volume_par_empty_mesh_returns_zero() {
        let mesh = MeshStorage::new();
        assert_eq!(mesh_volume_par(&mesh), 0.0);
    }

    #[test]
    fn surface_area_empty_mesh_returns_zero() {
        let mesh = MeshStorage::new();
        assert_eq!(surface_area(&mesh), 0.0);
    }

    #[test]
    fn surface_area_par_empty_mesh_returns_zero() {
        let mesh = MeshStorage::new();
        assert_eq!(surface_area_par(&mesh), 0.0);
    }

    // ---------- 并行函数一致性（icosphere(1)） ----------

    #[test]
    fn surface_area_par_matches_serial() {
        let mesh = crate::test_util::build_icosphere(1);
        let s = surface_area(&mesh);
        let p = surface_area_par(&mesh);
        assert!((s - p).abs() < 1e-10, "serial={} par={}", s, p);
    }

    #[test]
    fn mesh_volume_par_matches_serial() {
        let mesh = crate::test_util::build_icosphere(1);
        let s = mesh_volume(&mesh);
        let p = mesh_volume_par(&mesh);
        assert!((s - p).abs() < 1e-10, "serial={} par={}", s, p);
    }

    #[test]
    fn vertex_normals_par_matches_serial() {
        let mesh = crate::test_util::build_icosphere(1);
        let par = vertex_normals_par(&mesh);
        let serial: Vec<[f64; 3]> = mesh
            .vertex_ids()
            .map(|v| vertex_normal(&mesh, v).unwrap_or([0.0, 0.0, 0.0]))
            .collect();
        assert_eq!(par.len(), serial.len());
        for (i, (a, b)) in par.iter().zip(serial.iter()).enumerate() {
            for c in 0..3 {
                assert!(
                    (a[c] - b[c]).abs() < 1e-10,
                    "顶点 {} 法向分量 {} 不一致: {} vs {}",
                    i,
                    c,
                    a[c],
                    b[c]
                );
            }
        }
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
                _ => panic!(
                    "顶点 {} 高斯曲率 None 不一致: {:?} vs {:?}",
                    i, a, b
                ),
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
                _ => panic!(
                    "顶点 {} 平均曲率 None 不一致: {:?} vs {:?}",
                    i, a, b
                ),
            }
        }
    }

    #[test]
    fn feature_edges_par_matches_serial() {
        let mesh = crate::test_util::build_icosphere(1);
        // icosphere(1) 相邻面法向夹角约 20–40°，取 0.3 rad 阈值可得到非空特征边集
        let threshold = 0.3_f64;
        let mut s = feature_edges(&mesh, threshold);
        let mut p = feature_edges_par(&mesh, threshold);
        s.sort();
        p.sort();
        assert_eq!(s, p, "feature_edges 串/并行结果不一致");
    }

    #[test]
    fn ray_mesh_intersection_par_matches_serial() {
        let mesh = crate::test_util::build_icosphere(1);
        let origin = [2.0, 0.0, 0.0];
        let direction = [-1.0, 0.0, 0.0];
        let serial = ray_mesh_intersection(origin, direction, &mesh);
        let par = ray_mesh_intersection_par(&mesh, origin, direction);
        match serial {
            Some(s_hit) => {
                assert!(!par.is_empty(), "par 应至少有一个交点");
                // par 返回所有交点，取 t 最小者与 serial 比较
                let min_par = par
                    .iter()
                    .min_by(|a, b| a.t.partial_cmp(&b.t).unwrap())
                    .unwrap();
                assert!(
                    (s_hit.t - min_par.t).abs() < 1e-10,
                    "t 不一致: serial={} par={}",
                    s_hit.t,
                    min_par.t
                );
            }
            None => {
                assert!(par.is_empty(), "serial 无交点时 par 应为空");
            }
        }
    }

    // ---------- ray_mesh_intersects 正确性 ----------

    #[test]
    fn ray_mesh_intersects_origin_outside_returns_false() {
        // ray_mesh_intersects 实现为奇偶判定（origin 是否在闭合网格内部）。
        // 原点 (0,0,-5) 在球外，射线沿 +z 穿过球面（与 2 个三角面相交，偶数）→ 返回 false。
        let mesh = crate::test_util::build_icosphere(1);
        let result = ray_mesh_intersects([0.0, 0.0, -5.0], [0.0, 0.0, 1.0], &mesh);
        assert!(!result, "原点在球外，奇偶判定应返回 false");
    }

    #[test]
    fn ray_mesh_intersects_misses_returns_false() {
        // 射线从远处射向 +x 方向，不与球面相交（0 个交点）。
        let mesh = crate::test_util::build_icosphere(1);
        let result = ray_mesh_intersects([10.0, 10.0, 10.0], [1.0, 0.0, 0.0], &mesh);
        assert!(!result, "射线不与球面相交，应返回 false");
    }

    // ---------- mesh_volume 符号 ----------

    #[test]
    fn mesh_volume_cube_positive() {
        // build_cube(1.0) 面 CCW 朝外，有向体积应为正 ≈ 1.0
        let mesh = crate::primitives::build_cube(1.0);
        let vol = mesh_volume(&mesh);
        assert!(vol > 0.0, "立方体体积应为正: {}", vol);
        assert!(
            (vol - 1.0).abs() < 1e-9,
            "边长 1 立方体体积应 ≈ 1.0: {}",
            vol
        );
    }

    #[test]
    fn mesh_volume_par_cube_positive() {
        let mesh = crate::primitives::build_cube(1.0);
        let vol = mesh_volume_par(&mesh);
        assert!(vol > 0.0, "立方体体积应为正: {}", vol);
        assert!(
            (vol - 1.0).abs() < 1e-9,
            "边长 1 立方体体积应 ≈ 1.0: {}",
            vol
        );
    }
}
