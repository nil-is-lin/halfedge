//! 基本几何查询
//!
//! 在 [`MeshStorage`] 之上叠加基本几何查询能力：
//! - 边长：[`edge_length`]
//! - 三角面面积：[`face_area`]
//! - 面法向：[`face_normal`]
//! - 三角面最小内角：[`face_min_angle`]
//! - 顶点法向：[`vertex_normal`]（面积加权平均）
//! - 多边形面积和法向：[`polygon_area`], [`polygon_normal`]（Newell 方法）
//! - 表面积：[`surface_area`]
//! - 有向体积：[`mesh_volume`]
//! - 余切权重拉普拉斯：[`cotan_laplacian`], [`cotan_edge_weight`]
//! - 二面角：[`dihedral_angle`]
//! - 特征边：[`is_feature_edge`], [`feature_edges`]

use crate::ids::{FaceId, HalfEdgeId, VertexId};
use crate::linalg::vec3::{Vec3, add, angle_between, cross, dot, length, normalize, scale, sub};
use crate::storage::MeshStorage;
use crate::traversal::{FaceHalfEdges, VertexAdjacentFaces, VertexRing};

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
///
/// ```
/// use halfedge::{build_mesh_from_vertices_and_faces, face_area};
///
/// let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
/// let faces = vec![[0u32, 1, 2]];
/// let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
/// let f = mesh.face_ids().next().unwrap();
/// let area = face_area(&mesh, f).unwrap();
/// assert!((area - 0.5).abs() < 1e-9);
/// ```
pub fn face_area(mesh: &MeshStorage, f: FaceId) -> Option<f64> {
    let (a, b, c) = face_triangle_positions(mesh, f)?;
    Some(0.5 * length(cross(sub(b, a), sub(c, a))))
}

/// 面法向：归一化的 $(\vec{B}-\vec{A}) \times (\vec{C}-\vec{A})$。
///
/// 退化三角形（零面积）返回 `None`。
///
/// ```
/// use halfedge::{build_mesh_from_vertices_and_faces, face_normal};
///
/// let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
/// let faces = vec![[0u32, 1, 2]];
/// let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
/// let f = mesh.face_ids().next().unwrap();
/// let n = face_normal(&mesh, f).unwrap();
/// assert!((n[2] - 1.0).abs() < 1e-9);
/// ```
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
///
/// ```
/// use halfedge::{build_mesh_from_vertices_and_faces, vertex_normal};
///
/// let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
/// let faces = vec![[0u32, 1, 2]];
/// let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
/// let v = mesh.vertex_ids().next().unwrap();
/// let n = vertex_normal(&mesh, v).unwrap();
/// assert!((n[2] - 1.0).abs() < 1e-9);
/// ```
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
    let positions = mesh.positions_dense();
    let mut volume = 0.0;
    for f in mesh.face_ids() {
        let verts: Vec<VertexId> = crate::traversal::FaceVertices::new(mesh, f).collect();
        if verts.len() != 3 {
            continue;
        }
        // 通过 SOA 位置缓存读取顶点位置（24 字节步长连续访问）
        let (v0, v1, v2) = match (
            mesh.position_index(verts[0]),
            mesh.position_index(verts[1]),
            mesh.position_index(verts[2]),
        ) {
            (Some(i0), Some(i1), Some(i2)) => (
                positions[i0 as usize],
                positions[i1 as usize],
                positions[i2 as usize],
            ),
            _ => continue,
        };
        volume += tet_signed_volume(origin, v0, v1, v2);
    }
    volume
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
    let positions = mesh.positions_dense();
    let face_ids: Vec<_> = mesh.face_ids().collect();
    face_ids
        .par_iter()
        .map(|&f| {
            let verts: Vec<VertexId> = crate::traversal::FaceVertices::new(mesh, f).collect();
            if verts.len() != 3 {
                return 0.0;
            }
            // 通过 SOA 位置缓存读取（与串行版本一致）
            let (v0, v1, v2) = match (
                mesh.position_index(verts[0]),
                mesh.position_index(verts[1]),
                mesh.position_index(verts[2]),
            ) {
                (Some(i0), Some(i1), Some(i2)) => (
                    positions[i0 as usize],
                    positions[i1 as usize],
                    positions[i2 as usize],
                ),
                _ => return 0.0,
            };
            tet_signed_volume(origin, v0, v1, v2)
        })
        .sum()
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

    // ---------- 表面积 / 体积 ----------

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

    // ---------- 空/退化网格 ----------

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
}
