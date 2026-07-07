//! 细分模块
//!
//! 提供三种网格细分算法：
//! - [`loop_subdivide`]：Loop 细分（三角网格，每个面分裂为 4 个三角形）
//! - [`catmull_clark::catmull_clark_subdivide`]：Catmull-Clark 细分（任意多边形网格，
//!   每个面分裂为 k 个四边形并三角化输出）
//! - [`sqrt3::sqrt3_subdivide`]：√3 细分（三角网格，每个面分裂为 3 个三角形，
//!   适合渐进式细分）
//!
//! ## Loop 细分
//! Loop 细分是一种\textbf{逼近型}细分方案，每次细分将每个三角面分裂为 4 个小
//! 三角形，同时按加权平均更新所有顶点位置，使网格趋于光滑。
//!
//! ### 步骤
//! 1. **边点计算**：在每条边中点插入新顶点
//!    - 内部边：$\frac{3}{8}(v_0+v_1) + \frac{1}{8}(v_2+v_3)$，其中 $v_2, v_3$
//!      是两侧相对顶点
//!    - 边界边：$\frac{1}{2}(v_0+v_1)$
//! 2. **顶点更新**：更新原始顶点位置
//!    - 内部顶点（valence=$n$）：$(1-n\beta)v + \beta\sum \text{neighbors}$，
//!      $\beta = \frac{1}{n}\left(\frac{5}{8} - \left(\frac{3}{8}+\frac{1}{4}\cos\frac{2\pi}{n}\right)^2\right)$
//!    - 边界顶点：$\frac{1}{8}v_{\text{prev}} + \frac{3}{4}v + \frac{1}{8}v_{\text{next}}$
//! 3. **面分裂**：每个原始三角形分裂为 4 个小三角形。
//!
//! ### 规模变化
//! 设原始网格有 $V$ 顶点、$E$ 边、$F$ 面，细分后：
//! $$V' = V + E, \quad F' = 4F, \quad E' = 2E$$
//!
//! ## √3 细分
//! √3 细分是\textbf{插值型}细分方案，面数增长 3 倍（而非 Loop 的 4 倍），
//! 在渐进式细分中提供更细粒度的控制。
//! $$V' = V + F, \quad F' = 3F$$
//!
//! ## 示例
//! ```
//! use halfedge::{build_icosphere, loop_subdivide};
//!
//! let mesh = build_icosphere(1);      // V=42, F=80
//! let refined = loop_subdivide(&mesh); // V=162, F=320
//! ```

pub mod catmull_clark;
pub mod sqrt3;

use std::collections::HashMap;

use crate::ids::VertexId;
use crate::io::build_mesh_from_vertices_and_faces;
use crate::storage::MeshStorage;
use crate::traversal::{FaceHalfEdges, VertexRing, is_boundary_edge, is_boundary_vertex};

// ============================================================
// 公开 API
// ============================================================

/// 对三角网格执行一次 Loop 细分，返回细分后的新网格。
///
/// 输入网格不修改，输出为全新构建的 `MeshStorage`。
///
/// # 算法
/// 1. 在每条边中点插入新顶点（内部边用 3/8+1/8 权重，边界边用 1/2 权重）；
/// 2. 更新原始顶点位置（内部顶点用 Loop β 权重，边界顶点用 1/8-3/4-1/8 权重）；
/// 3. 每个原始三角形分裂为 4 个小三角形。
///
/// # 规模变化
/// $V' = V + E$，$F' = 4F$。
///
/// # 健壮性
/// - 所有查询通过 `mesh.get_*` + `Option` 链式调用，无效 ID 返回 `None` 时跳过；
/// - 边界顶点的邻居数 ≠ 2 时保持原位置（退化情况）；
/// - 内部顶点 valence=0 时保持原位置（孤立顶点）。
///
/// # 非三角面
/// Loop 细分仅支持三角形。若输入网格包含非三角面（如四边形），
/// 这些面会被跳过并通过 `log::warn!` 输出警告。
/// 若需处理任意多边形网格，请使用
/// [`catmull_clark::catmull_clark_subdivide`]。
///
/// ```
/// use halfedge::{build_icosphere, loop_subdivide};
///
/// let mesh = build_icosphere(1); // V=42, F=80
/// let refined = loop_subdivide(&mesh);
/// assert_eq!(refined.face_count(), mesh.face_count() * 4);
/// ```
pub fn loop_subdivide(mesh: &MeshStorage) -> MeshStorage {
    // 1. 收集原始顶点 ID 并建立 VertexId → u32 索引映射
    let orig_v_ids: Vec<VertexId> = mesh.vertex_ids().collect();
    let n_orig_verts = orig_v_ids.len();
    let mut v_index: HashMap<VertexId, u32> = HashMap::new();
    for (i, &v) in orig_v_ids.iter().enumerate() {
        v_index.insert(v, i as u32);
    }

    // 2. 收集所有面，转为顶点索引三元组（CCW 顺序，由 FaceHalfEdges 保证）
    //    非三角面被跳过并发出警告（Loop 细分仅支持三角形）
    let mut orig_faces: Vec<[u32; 3]> = Vec::new();
    let mut skipped_non_triangle: u32 = 0;
    for f_id in mesh.face_ids() {
        let verts: Vec<u32> = FaceHalfEdges::new(mesh, f_id)
            .filter_map(|he| mesh.get_halfedge(he))
            .map(|h| h.vertex)
            .filter_map(|v| v_index.get(&v).copied())
            .collect();
        match verts.len() {
            3 => orig_faces.push([verts[0], verts[1], verts[2]]),
            _ => skipped_non_triangle += 1,
        }
    }
    if skipped_non_triangle > 0 {
        log::warn!(
            "[halfedge::loop_subdivide] 警告：输入网格含 {} 个非三角面，已跳过（Loop 细分仅支持三角形）。\
             若需处理任意多边形，请使用 catmull_clark_subdivide。",
            skipped_non_triangle
        );
    }

    // 3. 计算每条边的中点
    // new_positions[0..n_orig_verts] = 原始顶点位置（暂存旧值）
    // new_positions[n_orig_verts..]  = 新边中点位置
    let mut new_positions: Vec<[f64; 3]> = orig_v_ids
        .iter()
        .filter_map(|&v| mesh.get_vertex(v))
        .map(|vt| vt.position)
        .collect();
    // 防御：若某些顶点查询失败导致长度不一致，补齐
    while new_positions.len() < n_orig_verts {
        new_positions.push([0.0; 3]);
    }

    // 边键 = (min(u32), max(u32))，去重每条无向边
    let mut edge_midpoint: HashMap<(u32, u32), u32> = HashMap::new();

    for he_id in mesh.halfedge_ids() {
        let h = match mesh.get_halfedge(he_id) {
            Some(h) => h,
            None => continue,
        };
        let v1 = h.vertex; // tip
        let v0 = match h.twin.and_then(|t| mesh.get_halfedge(t)) {
            Some(t) => t.vertex, // origin = twin.vertex
            None => continue,    // 拓扑断开，跳过
        };
        let i0 = match v_index.get(&v0) {
            Some(&i) => i,
            None => continue,
        };
        let i1 = match v_index.get(&v1) {
            Some(&i) => i,
            None => continue,
        };
        if i0 == i1 {
            continue; // 退化边
        }
        let key = edge_key(i0, i1);
        if edge_midpoint.contains_key(&key) {
            continue; // 已处理
        }

        let p0 = new_positions[i0 as usize];
        let p1 = new_positions[i1 as usize];

        let midpoint = if is_boundary_edge(mesh, he_id) {
            // 边界边：1/2*(v0+v1)
            [
                0.5 * (p0[0] + p1[0]),
                0.5 * (p0[1] + p1[1]),
                0.5 * (p0[2] + p1[2]),
            ]
        } else {
            // 内部边：3/8*(v0+v1) + 1/8*(v2+v3)
            // v2 = he.next.vertex（he 所在面的对边顶点）
            // v3 = twin.next.vertex（twin 所在面的对边顶点）
            let v2 = h.next.and_then(|n| mesh.get_halfedge(n)).map(|n| n.vertex);
            let v3 = h
                .twin
                .and_then(|t| mesh.get_halfedge(t))
                .and_then(|t| t.next)
                .and_then(|n| mesh.get_halfedge(n))
                .map(|n| n.vertex);

            match (v2, v3) {
                (Some(v2), Some(v3)) => {
                    let p2 = mesh.get_vertex(v2).map(|v| v.position).unwrap_or([0.0; 3]);
                    let p3 = mesh.get_vertex(v3).map(|v| v.position).unwrap_or([0.0; 3]);
                    [
                        3.0 / 8.0 * (p0[0] + p1[0]) + 1.0 / 8.0 * (p2[0] + p3[0]),
                        3.0 / 8.0 * (p0[1] + p1[1]) + 1.0 / 8.0 * (p2[1] + p3[1]),
                        3.0 / 8.0 * (p0[2] + p1[2]) + 1.0 / 8.0 * (p2[2] + p3[2]),
                    ]
                }
                _ => {
                    // 拓扑不完整，退化为边界处理
                    [
                        0.5 * (p0[0] + p1[0]),
                        0.5 * (p0[1] + p1[1]),
                        0.5 * (p0[2] + p1[2]),
                    ]
                }
            }
        };

        let new_idx = new_positions.len() as u32;
        new_positions.push(midpoint);
        edge_midpoint.insert(key, new_idx);
    }

    // 4. 计算原始顶点的新位置（使用旧位置，与边中点计算独立）
    //    先存到独立 Vec，最后覆盖，保证边中点用旧位置算
    let mut updated_orig: Vec<[f64; 3]> = Vec::with_capacity(n_orig_verts);
    for (i, &v) in orig_v_ids.iter().enumerate() {
        let p_old = new_positions[i]; // 旧位置

        let new_pos = if is_boundary_vertex(mesh, v) {
            // 边界顶点：1/8*v_prev + 3/4*v + 1/8*v_next
            // 找到恰好 2 个边界邻居（通过 is_boundary_edge 判定）
            let boundary_neighbors: Vec<VertexId> = VertexRing::new(mesh, v)
                .filter(|&he| is_boundary_edge(mesh, he))
                .filter_map(|he| mesh.get_halfedge(he))
                .map(|h| h.vertex)
                .collect();
            if boundary_neighbors.len() == 2 {
                let p_prev = mesh
                    .get_vertex(boundary_neighbors[0])
                    .map(|v| v.position)
                    .unwrap_or([0.0; 3]);
                let p_next = mesh
                    .get_vertex(boundary_neighbors[1])
                    .map(|v| v.position)
                    .unwrap_or([0.0; 3]);
                [
                    0.125 * p_prev[0] + 0.75 * p_old[0] + 0.125 * p_next[0],
                    0.125 * p_prev[1] + 0.75 * p_old[1] + 0.125 * p_next[1],
                    0.125 * p_prev[2] + 0.75 * p_old[2] + 0.125 * p_next[2],
                ]
            } else {
                p_old // 退化：保持原位置
            }
        } else {
            // 内部顶点：(1 - n*β)*v + β*Σ(neighbors)
            let neighbors: Vec<[f64; 3]> = VertexRing::new(mesh, v)
                .filter_map(|he| mesh.get_halfedge(he))
                .map(|h| h.vertex)
                .filter_map(|nb| mesh.get_vertex(nb))
                .map(|vt| vt.position)
                .collect();
            let n = neighbors.len();
            if n == 0 {
                p_old // 孤立顶点
            } else {
                let beta = loop_beta(n);
                let mut sum = [0.0; 3];
                for nb in &neighbors {
                    sum[0] += nb[0];
                    sum[1] += nb[1];
                    sum[2] += nb[2];
                }
                let weight = 1.0 - (n as f64) * beta;
                [
                    weight * p_old[0] + beta * sum[0],
                    weight * p_old[1] + beta * sum[1],
                    weight * p_old[2] + beta * sum[2],
                ]
            }
        };
        updated_orig.push(new_pos);
    }
    // 覆盖原始顶点位置
    for (i, pos) in updated_orig.into_iter().enumerate() {
        new_positions[i] = pos;
    }

    // 5. 构建新面索引：每个三角形分裂为 4 个
    let mut new_faces: Vec<[u32; 3]> = Vec::with_capacity(orig_faces.len() * 4);
    for face in &orig_faces {
        let [a, b, c] = *face;
        let ab = match edge_midpoint.get(&edge_key(a, b)) {
            Some(&idx) => idx,
            None => continue, // 缺失边中点，跳过该面（防御）
        };
        let bc = match edge_midpoint.get(&edge_key(b, c)) {
            Some(&idx) => idx,
            None => continue,
        };
        let ca = match edge_midpoint.get(&edge_key(c, a)) {
            Some(&idx) => idx,
            None => continue,
        };
        new_faces.push([a, ab, ca]);
        new_faces.push([b, bc, ab]);
        new_faces.push([c, ca, bc]);
        new_faces.push([ab, bc, ca]);
    }

    // 6. 构建新网格
    build_mesh_from_vertices_and_faces(&new_positions, &new_faces)
        .expect("Loop subdivision output is always valid")
}

// ============================================================
// 内部辅助
// ============================================================

/// Loop 细分 β 权重：
/// $\beta = \frac{1}{n}\left(\frac{5}{8} - \left(\frac{3}{8}+\frac{1}{4}\cos\frac{2\pi}{n}\right)^2\right)$
fn loop_beta(n: usize) -> f64 {
    if n == 0 {
        return 0.0; // 守卫：除零
    }
    let n_f = n as f64;
    let cos_term = (2.0 * std::f64::consts::PI / n_f).cos();
    let inner = 3.0 / 8.0 + 1.0 / 4.0 * cos_term;
    (1.0 / n_f) * (5.0 / 8.0 - inner * inner)
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
    use crate::storage::{Face, HalfEdge, Vertex};
    use crate::test_util::build_icosphere;
    use crate::validate::check_topology;

    /// 构造仅含 1 个四边形面的网格（无 twin，仅面环）。
    /// 用于测试 Loop/√3 细分对非三角面的跳过 + 警告行为。
    fn build_single_quad_mesh() -> MeshStorage {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let v2 = mesh.add_vertex(Vertex::new([1.0, 1.0, 0.0]));
        let v3 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));

        let h0 = mesh.add_halfedge(HalfEdge::new(v1)); // v0→v1
        let h1 = mesh.add_halfedge(HalfEdge::new(v2)); // v1→v2
        let h2 = mesh.add_halfedge(HalfEdge::new(v3)); // v2→v3
        let h3 = mesh.add_halfedge(HalfEdge::new(v0)); // v3→v0

        for (he, next, prev) in [(h0, h1, h3), (h1, h2, h0), (h2, h3, h1), (h3, h0, h2)] {
            let h = mesh.get_halfedge_mut(he).unwrap();
            h.next = Some(next);
            h.prev = Some(prev);
        }

        let face = mesh.add_face(Face::new());
        mesh.get_face_mut(face).unwrap().halfedge = Some(h0);
        for he in [h0, h1, h2, h3] {
            mesh.get_halfedge_mut(he).unwrap().face = Some(face);
        }

        // 顶点入口：任一 outgoing 半边
        mesh.get_vertex_mut(v0).unwrap().halfedge = Some(h0);
        mesh.get_vertex_mut(v1).unwrap().halfedge = Some(h1);
        mesh.get_vertex_mut(v2).unwrap().halfedge = Some(h2);
        mesh.get_vertex_mut(v3).unwrap().halfedge = Some(h3);

        mesh
    }

    // ---------- 非三角面跳过 + 警告 ----------

    #[test]
    fn loop_subdivide_skips_non_triangle_with_warning() {
        // 输入：1 个四边形面（无三角面）
        // 期望：跳过四边形，输出为 0 面（顶点保留）；
        //      警告通过 log::warn! 输出，测试只验证行为不变化。
        let mesh = build_single_quad_mesh();
        assert_eq!(mesh.face_count(), 1, "输入应含 1 个四边形面");

        let refined = loop_subdivide(&mesh);
        assert_eq!(refined.face_count(), 0, "非三角面应被跳过，输出 0 面");
        assert_eq!(
            refined.vertex_count(),
            4,
            "原始 4 个顶点应保留（无新边中点）"
        );
    }

    #[test]
    fn loop_subdivide_mixed_mesh_keeps_only_triangles() {
        // 输入：1 个三角形（含完整 twin）+ 1 个独立四边形（无 twin）
        // 期望：仅三角形被细分 → 4 个三角面；四边形被跳过
        let tri_verts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let tri_faces = [[0, 1, 2]];
        let mut mesh = build_mesh_from_vertices_and_faces(&tri_verts, &tri_faces).unwrap();

        // 追加独立四边形面（无 twin）
        let qv0 = mesh.add_vertex(Vertex::new([10.0, 0.0, 0.0]));
        let qv1 = mesh.add_vertex(Vertex::new([11.0, 0.0, 0.0]));
        let qv2 = mesh.add_vertex(Vertex::new([11.0, 1.0, 0.0]));
        let qv3 = mesh.add_vertex(Vertex::new([10.0, 1.0, 0.0]));
        let qh0 = mesh.add_halfedge(HalfEdge::new(qv1));
        let qh1 = mesh.add_halfedge(HalfEdge::new(qv2));
        let qh2 = mesh.add_halfedge(HalfEdge::new(qv3));
        let qh3 = mesh.add_halfedge(HalfEdge::new(qv0));
        for (he, next, prev) in [
            (qh0, qh1, qh3),
            (qh1, qh2, qh0),
            (qh2, qh3, qh1),
            (qh3, qh0, qh2),
        ] {
            let h = mesh.get_halfedge_mut(he).unwrap();
            h.next = Some(next);
            h.prev = Some(prev);
        }
        let qf = mesh.add_face(Face::new());
        mesh.get_face_mut(qf).unwrap().halfedge = Some(qh0);
        for he in [qh0, qh1, qh2, qh3] {
            mesh.get_halfedge_mut(he).unwrap().face = Some(qf);
        }
        mesh.get_vertex_mut(qv0).unwrap().halfedge = Some(qh0);
        mesh.get_vertex_mut(qv1).unwrap().halfedge = Some(qh1);
        mesh.get_vertex_mut(qv2).unwrap().halfedge = Some(qh2);
        mesh.get_vertex_mut(qv3).unwrap().halfedge = Some(qh3);

        assert_eq!(mesh.face_count(), 2);

        let refined = loop_subdivide(&mesh);
        // 仅三角形被细分（1 三角 → 4 三角），四边形被跳过
        assert_eq!(refined.face_count(), 4, "仅三角面被细分为 4 面");
    }

    // ---------- 规模验证 ----------

    #[test]
    fn loop_subdivide_icosphere1_vertex_face_count() {
        // icosphere(1): V=42, F=80, E=120
        // 细分后：V' = 42 + 120 = 162, F' = 4 * 80 = 320
        let mesh = build_icosphere(1);
        assert_eq!(mesh.vertex_count(), 42);
        assert_eq!(mesh.face_count(), 80);

        let refined = loop_subdivide(&mesh);
        assert_eq!(refined.vertex_count(), 162, "顶点数应为 42+120=162");
        assert_eq!(refined.face_count(), 320, "面数应为 4*80=320");
    }

    #[test]
    fn loop_subdivide_icosphere0_vertex_face_count() {
        // icosphere(0): V=12, F=20, E=30
        // 细分后：V' = 12 + 30 = 42, F' = 4 * 20 = 80
        let mesh = build_icosphere(0);
        let refined = loop_subdivide(&mesh);
        assert_eq!(refined.vertex_count(), 42);
        assert_eq!(refined.face_count(), 80);
    }

    #[test]
    fn loop_subdivide_icosphere2_vertex_face_count() {
        // icosphere(2): V=162, F=320, E=480
        // 细分后：V' = 162 + 480 = 642, F' = 4 * 320 = 1280
        let mesh = build_icosphere(2);
        let refined = loop_subdivide(&mesh);
        assert_eq!(refined.vertex_count(), 642);
        assert_eq!(refined.face_count(), 1280);
    }

    // ---------- 拓扑校验 ----------

    #[test]
    fn loop_subdivide_passes_topology_validation() {
        let mesh = build_icosphere(1);
        let refined = loop_subdivide(&mesh);
        assert!(
            check_topology(&refined).is_ok(),
            "细分后的网格应通过完整拓扑校验: {:?}",
            check_topology(&refined)
        );
    }

    #[test]
    fn loop_subdivide_icosphere2_passes_validation() {
        let mesh = build_icosphere(2);
        let refined = loop_subdivide(&mesh);
        assert!(check_topology(&refined).is_ok());
    }

    // ---------- Euler 公式 ----------

    #[test]
    fn loop_subdivide_preserves_euler_characteristic() {
        // 闭合三角网格的 Euler 示性数 V - E + F = 2 应保持不变
        for n in 0..=2 {
            let mesh = build_icosphere(n);
            let v = mesh.vertex_count() as i64;
            let e = (mesh.halfedge_count() / 2) as i64;
            let f = mesh.face_count() as i64;
            assert_eq!(v - e + f, 2, "细分前 icosphere({}) Euler 示性数应为 2", n);

            let refined = loop_subdivide(&mesh);
            let v2 = refined.vertex_count() as i64;
            let e2 = (refined.halfedge_count() / 2) as i64;
            let f2 = refined.face_count() as i64;
            assert_eq!(
                v2 - e2 + f2,
                2,
                "细分后 icosphere({}) Euler 示性数应保持 2",
                n
            );
        }
    }

    // ---------- 边界网格 ----------

    #[test]
    fn loop_subdivide_single_triangle() {
        // 单三角面片（全是边界边）
        // V=3, F=1, E=3 → V'=6, F'=4
        let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = [[0, 1, 2]];
        let mesh = build_mesh_from_vertices_and_faces(&vertices, &faces).unwrap();
        let refined = loop_subdivide(&mesh);
        assert_eq!(refined.vertex_count(), 6);
        assert_eq!(refined.face_count(), 4);
        assert!(check_topology(&refined).is_ok());
    }

    #[test]
    fn loop_subdivide_open_quad() {
        // 两个三角形拼成的开四边形
        // V=4, F=2, E=5（4 边界 + 1 内部）→ V'=9, F'=8
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let faces = [[0, 1, 2], [0, 2, 3]];
        let mesh = build_mesh_from_vertices_and_faces(&vertices, &faces).unwrap();
        let refined = loop_subdivide(&mesh);
        assert_eq!(refined.vertex_count(), 9); // 4 + 5
        assert_eq!(refined.face_count(), 8); // 4 * 2
        assert!(check_topology(&refined).is_ok());
    }

    // ---------- 位置正确性 ----------

    #[test]
    fn loop_subdivide_boundary_edge_midpoint() {
        // 单三角形：所有边都是边界边，中点 = 1/2*(v0+v1)
        let vertices = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let faces = [[0, 1, 2]];
        let mesh = build_mesh_from_vertices_and_faces(&vertices, &faces).unwrap();
        let refined = loop_subdivide(&mesh);

        // 细分后应有 6 个顶点：3 原始 + 3 边中点
        // 边 (0,1) 中点应为 (1,0,0)
        // 边 (1,2) 中点应为 (1,1,0)
        // 边 (2,0) 中点应为 (0,1,0)
        let mut positions: Vec<[f64; 3]> = refined
            .vertex_ids()
            .filter_map(|v| refined.get_vertex(v))
            .map(|vt| vt.position)
            .collect();
        positions.sort_by(|a, b| {
            a[0].partial_cmp(&b[0])
                .unwrap()
                .then(a[1].partial_cmp(&b[1]).unwrap())
                .then(a[2].partial_cmp(&b[2]).unwrap())
        });

        // 期望位置（共 6 个顶点：3 原始更新 + 3 边中点）：
        // 原始顶点更新后（边界顶点公式 1/8*prev + 3/4*v + 1/8*next）：
        //   v0=(0,0,0) → 1/8*(2,0,0) + 3/4*(0,0,0) + 1/8*(0,2,0) = (0.25, 0.25, 0)
        //   v1=(2,0,0) → 1/8*(0,2,0) + 3/4*(2,0,0) + 1/8*(0,0,0) = (1.5, 0.25, 0)
        //   v2=(0,2,0) → 1/8*(0,0,0) + 3/4*(0,2,0) + 1/8*(2,0,0) = (0.25, 1.5, 0)
        // 边中点（边界边公式 1/2*(v0+v1)）：
        //   (0,1): (1, 0, 0)
        //   (1,2): (1, 1, 0)
        //   (2,0): (0, 1, 0)
        let expected = [
            [0.25, 0.25, 0.0],
            [0.25, 1.5, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.5, 0.25, 0.0],
            [0.0, 1.0, 0.0],
        ];
        // 由于排序顺序可能变化，用集合近似比较
        let eps = 1e-9;
        for e in &expected {
            let found = positions.iter().any(|p| {
                (p[0] - e[0]).abs() < eps && (p[1] - e[1]).abs() < eps && (p[2] - e[2]).abs() < eps
            });
            assert!(found, "未找到期望位置 {:?}，实际位置 = {:?}", e, positions);
        }
    }

    #[test]
    fn loop_subdivide_interior_edge_midpoint() {
        // 两个三角形拼成四边形：内部边 (0,2)
        // 内部边中点 = 3/8*(v0+v2) + 1/8*(v1+v3)
        // v0=(0,0,0), v2=(1,1,0), v1=(1,0,0), v3=(0,1,0)
        // 中点 = 3/8*((0,0,0)+(1,1,0)) + 1/8*((1,0,0)+(0,1,0))
        //      = 3/8*(1,1,0) + 1/8*(1,1,0) = 4/8*(1,1,0) = (0.5, 0.5, 0)
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let faces = [[0, 1, 2], [0, 2, 3]];
        let mesh = build_mesh_from_vertices_and_faces(&vertices, &faces).unwrap();
        let refined = loop_subdivide(&mesh);

        // 找到位置最接近 (0.5, 0.5, 0) 的顶点
        let target = [0.5, 0.5, 0.0];
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
        assert!(found, "内部边中点应位于 (0.5, 0.5, 0)");
    }

    // ---------- β 权重 ----------

    #[test]
    fn loop_beta_regular_valence_6() {
        // valence=6 的正则情况：β = 1/16
        let beta = loop_beta(6);
        assert!(
            (beta - 1.0 / 16.0).abs() < 1e-12,
            "β(6) 应为 1/16，实际 {}",
            beta
        );
    }

    #[test]
    fn loop_beta_valence_3() {
        // valence=3：β = 3/16
        let beta = loop_beta(3);
        assert!(
            (beta - 3.0 / 16.0).abs() < 1e-12,
            "β(3) 应为 3/16，实际 {}",
            beta
        );
    }

    #[test]
    fn loop_beta_zero_valence_returns_zero() {
        // 守卫：n=0 不应 panic
        assert_eq!(loop_beta(0), 0.0);
    }

    // ---------- 收敛性 ----------

    #[test]
    fn loop_subdivide_multiple_iterations_stay_valid() {
        // 连续细分 3 次，每次都应通过拓扑校验
        let mut mesh = build_icosphere(0);
        for i in 0..3 {
            mesh = loop_subdivide(&mesh);
            assert!(
                check_topology(&mesh).is_ok(),
                "第 {} 次细分后拓扑校验失败",
                i + 1
            );
        }
        // icosphere(0) 细分 3 次：V = 12 → 42 → 162 → 642
        assert_eq!(mesh.vertex_count(), 642);
        assert_eq!(mesh.face_count(), 1280);
    }

    // ---------- 空网格 ----------

    #[test]
    fn loop_subdivide_empty_mesh() {
        let mesh = MeshStorage::new();
        let refined = loop_subdivide(&mesh);
        assert_eq!(refined.vertex_count(), 0);
        assert_eq!(refined.face_count(), 0);
    }
}
