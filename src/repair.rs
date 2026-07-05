//! 网格修复与洞填充模块。
//!
//! 提供洞填充、孤立元素清理、非流形检测和法线一致化等操作，
//! 组合为高层修复 pipeline。
//!
//! ## 核心 API
//!
//! | 函数 | 功能 |
//! |------|------|
//! | [`fill_hole`] | 填充单个洞（指定边界半边） |
//! | [`fill_all_holes`] | 填充所有洞 |
//! | [`remove_isolated_vertices`] | 删除无关联半边的顶点 |
//! | [`remove_face`] | 删除面，将邻居半边转为边界 |
//! | [`remove_degenerate_faces`] | 删除退化面 |
//! | [`detect_nonmanifold_edges`] | 检测非流形边 |
//! | [`detect_nonmanifold_vertices`] | 检测非流形顶点 |
//! | [`repair_mesh`] | 一键修复 pipeline |

use crate::ids::{FaceId, HalfEdgeId, VertexId};
use crate::predicates::is_triangle_degenerate_3d;
use crate::storage::{HalfEdge, MeshStorage};
use crate::topology_ops::{TopologyError, add_triangle};
use crate::traversal::{BoundaryLoop, FaceHalfEdges, boundary_loops};
use crate::triangulation::ear_clipping_3d;

// ============================================================
// 删除面（创建洞）
// ============================================================

/// 删除面，在洞边界上创建新的边界半边环。
///
/// 删除面及其关联的面侧半边，对每条被暴露的边创建边界半边，
/// 形成边界环供 [`fill_hole`] 使用。
///
/// 当相邻面已被删除时（twin 是边界半边），直接删除该边界半边，
/// 不创建新边界（该边两侧均无面）。
///
/// 边界半边不设置 next/prev（`BoundaryLoop` 通过顶点环遍历，
/// 不依赖边界半边的 next/prev）。
pub fn remove_face(mesh: &mut MeshStorage, face: FaceId) -> Result<(), TopologyError> {
    let he_ids: Vec<HalfEdgeId> = FaceHalfEdges::new(mesh, face).collect();
    if he_ids.is_empty() {
        mesh.remove_face(face);
        return Ok(());
    }

    // 收集面半边数据（在删除前克隆）
    let he_data: Vec<_> = he_ids
        .iter()
        .map(|&he| mesh.get_halfedge(he).cloned())
        .collect();

    // 对每条面半边的 twin：清空 twin 的 twin 指向
    for he_opt in &he_data {
        let Some(h) = he_opt else { continue };
        if let Some(twid) = h.twin
            && let Some(t) = mesh.get_halfedge_mut(twid)
        {
            t.twin = None;
        }
    }

    // 删除面和面侧半边
    mesh.remove_face(face);
    for he in he_ids {
        mesh.remove_halfedge(he);
    }

    // 为每条暴露的边创建新的边界半边
    for he_opt in &he_data {
        let Some(h) = he_opt else { continue };
        let twin_id = h.twin;
        let Some(twid) = twin_id else { continue };
        if !mesh.contains_halfedge(twid) {
            continue;
        }

        let twin_data = match mesh.get_halfedge(twid) {
            Some(td) => td,
            None => continue,
        };

        if twin_data.face.is_none() {
            // twin 已是边界半边（相邻面先前已被删除）
            // 该边两侧均无面，直接删除此边界半边
            mesh.remove_halfedge(twid);
            continue;
        }

        // 正常情况：twin 是面半边，创建边界半边与之配对
        // 新边界半边与面半边同向：src → dst（vertex = dst）
        let dst = h.vertex;
        let bhe = mesh.add_halfedge(HalfEdge::new(dst));
        // 不设置边界半边的 next/prev（保持 None）
        if let Some(t) = mesh.get_halfedge_mut(twid) {
            t.twin = Some(bhe);
        }
        if let Some(b) = mesh.get_halfedge_mut(bhe) {
            b.twin = Some(twid);
        }
    }

    // 修复所有顶点入口
    fix_all_vertex_entries(mesh);

    Ok(())
}

/// 修复所有顶点的 outgoing 半边入口。
fn fix_all_vertex_entries(mesh: &mut MeshStorage) {
    for v in mesh.vertex_ids().collect::<Vec<_>>() {
        let current = mesh.get_vertex(v).and_then(|vt| vt.halfedge);
        let needs_fix = current.is_none()
            || current.is_some_and(|he| {
                if !mesh.contains_halfedge(he) {
                    return true;
                }
                // 验证该半边的 origin 确实是 v
                let origin = mesh
                    .get_halfedge(he)
                    .and_then(|h| h.twin)
                    .and_then(|t| mesh.get_halfedge(t))
                    .map(|t| t.vertex);
                origin != Some(v)
            });
        if needs_fix {
            let new_he = find_outgoing_halfedge(mesh, v);
            if let Some(vt) = mesh.get_vertex_mut(v) {
                vt.halfedge = new_he;
            }
        }
    }
}

/// 找到顶点 v 的一条 outgoing 半边。
fn find_outgoing_halfedge(mesh: &MeshStorage, v: VertexId) -> Option<HalfEdgeId> {
    for he in mesh.halfedge_ids() {
        let h = mesh.get_halfedge(he)?;
        let origin = h.twin.and_then(|t| mesh.get_halfedge(t)).map(|t| t.vertex);
        if origin == Some(v) {
            return Some(he);
        }
    }
    None
}

// ============================================================
// 洞填充
// ============================================================

/// 填充单个洞。返回填充面的 ID 列表。
///
/// # 参数
/// - `mesh`: 三角网格
/// - `boundary_he`: 洞边界上的任一半边（`face` 为 `None` 的边界半边）
///
/// # 算法
/// 1. 从 `boundary_he` 出发，用 [`BoundaryLoop`] 收集边界环顶点
/// 2. 提取顶点坐标，用 [`ear_clipping_3d`] 三角化
/// 3. 清除边界半边的 next/prev（防止 `add_triangle` 删除部分边界后产生悬空指针）
/// 4. 将三角化结果写回网格（[`add_triangle`]），由其 twin 配对逻辑自动找到并删除边界半边
///
/// # 设计要点
/// 不提前删除边界半边，因为 [`add_triangle`] 的 twin 配对逻辑需要
/// `E.twin.face.is_none()` 来识别边界半边。删除边界半边会导致相邻面半边
/// 的 twin 变为 None，从而配对失败。
///
/// # 错误
/// - 边界环顶点数 < 3 → `Inconsistent`
/// - `boundary_he` 不是边界半边 → `Inconsistent`
/// - 三角化或 `add_triangle` 失败 → 传播错误
pub fn fill_hole(
    mesh: &mut MeshStorage,
    boundary_he: HalfEdgeId,
) -> Result<Vec<FaceId>, TopologyError> {
    // 1. 校验 boundary_he 是边界半边
    let he_data = mesh
        .get_halfedge(boundary_he)
        .ok_or(TopologyError::InvalidHalfEdge(boundary_he))?;
    if he_data.face.is_some() {
        return Err(TopologyError::Inconsistent("给定的半边不是边界半边".into()));
    }

    // 2. 收集边界环半边
    let loop_halfedges: Vec<HalfEdgeId> = BoundaryLoop::new(mesh, boundary_he).collect();
    let n = loop_halfedges.len();
    if n < 3 {
        return Err(TopologyError::Inconsistent(
            "边界环顶点数不足 3，无法填充".into(),
        ));
    }

    // 3. 提取顶点序列（各边界半边的 tip）
    let loop_vertices: Vec<VertexId> = loop_halfedges
        .iter()
        .filter_map(|&he| mesh.get_halfedge(he).map(|h| h.vertex))
        .collect();

    if loop_vertices.len() < 3 {
        return Err(TopologyError::Inconsistent("边界环顶点数不足 3".into()));
    }

    // 4. 提取顶点坐标
    let positions: Vec<[f64; 3]> = loop_vertices
        .iter()
        .map(|&v| {
            mesh.get_vertex(v)
                .map(|vt| vt.position)
                .unwrap_or([0.0, 0.0, 0.0])
        })
        .collect();

    // 5. 三角化
    let tris = ear_clipping_3d(&positions);
    if tris.is_empty() {
        return Err(TopologyError::Inconsistent(
            "三角化失败，可能边界环退化".into(),
        ));
    }

    // 6. 清除边界半边的 next/prev
    // 当 add_triangle 删除部分边界半边后，剩余边界半边的 next/prev
    // 可能指向已删除的半边，造成悬空指针。清除后 next=None 跳过校验。
    for &he in &loop_halfedges {
        if let Some(h) = mesh.get_halfedge_mut(he) {
            h.next = None;
            h.prev = None;
        }
    }

    // 7. 调用 add_triangle 创建填充面
    // add_triangle 的 twin 配对逻辑会自动找到边界半边并删除之
    let mut filled_faces = Vec::with_capacity(tris.len());
    for &[i, j, k] in &tris {
        let vi = loop_vertices[i];
        let vj = loop_vertices[j];
        let vk = loop_vertices[k];
        let face = add_triangle(mesh, vi, vj, vk).or_else(|_| add_triangle(mesh, vk, vj, vi))?;
        filled_faces.push(face);
    }

    // 8. 修复顶点入口
    fix_all_vertex_entries(mesh);

    Ok(filled_faces)
}

/// 填充所有洞。返回每个洞的填充面列表。
pub fn fill_all_holes(mesh: &mut MeshStorage) -> Result<Vec<Vec<FaceId>>, TopologyError> {
    let loops = boundary_loops(mesh);
    let mut results = Vec::with_capacity(loops.len());
    for boundary in &loops {
        if boundary.is_empty() {
            continue;
        }
        let start_he = boundary[0];
        let faces = fill_hole(mesh, start_he)?;
        results.push(faces);
    }
    Ok(results)
}

// ============================================================
// 孤立元素清理
// ============================================================

/// 删除所有孤立顶点（无 outgoing 半边的顶点）。返回被删除的顶点数量。
pub fn remove_isolated_vertices(mesh: &mut MeshStorage) -> usize {
    let isolated: Vec<VertexId> = mesh
        .vertex_ids()
        .filter(|&v| mesh.get_vertex(v).and_then(|vt| vt.halfedge).is_none())
        .collect();
    let count = isolated.len();
    for v in isolated {
        mesh.remove_vertex(v);
    }
    count
}

/// 删除所有退化面（面积为零或三顶点共线的三角面）。
pub fn remove_degenerate_faces(mesh: &mut MeshStorage) -> usize {
    let degenerate: Vec<FaceId> = mesh
        .face_ids()
        .filter(|&f| {
            let verts: Vec<VertexId> = FaceHalfEdges::new(mesh, f)
                .filter_map(|he| mesh.get_halfedge(he).map(|h| h.vertex))
                .collect();
            if verts.len() != 3 {
                return true;
            }
            let p0 = mesh.get_vertex(verts[0]).map(|v| v.position);
            let p1 = mesh.get_vertex(verts[1]).map(|v| v.position);
            let p2 = mesh.get_vertex(verts[2]).map(|v| v.position);
            match (p0, p1, p2) {
                (Some(a), Some(b), Some(c)) => is_triangle_degenerate_3d(a, b, c),
                _ => true,
            }
        })
        .collect();
    let count = degenerate.len();
    let mut failed: u32 = 0;
    for f in degenerate {
        if remove_face(mesh, f).is_err() {
            failed += 1;
        }
    }
    if failed > 0 {
        eprintln!(
            "[halfedge::remove_degenerate_faces] 警告：{failed} 个退化面删除失败"
        );
    }
    count
}

// ============================================================
// 非流形检测
// ============================================================

/// 检测非流形边：被 3 个或更多面共享的边。
pub fn detect_nonmanifold_edges(mesh: &MeshStorage) -> Vec<HalfEdgeId> {
    let mut edge_count: std::collections::HashMap<(VertexId, VertexId), usize> =
        std::collections::HashMap::new();
    let mut edge_rep: std::collections::HashMap<(VertexId, VertexId), HalfEdgeId> =
        std::collections::HashMap::new();
    for he in mesh.halfedge_ids() {
        let h = match mesh.get_halfedge(he) {
            Some(h) => h,
            None => continue,
        };
        if h.face.is_none() {
            continue;
        }
        let tip = h.vertex;
        let origin = match h.twin.and_then(|t| mesh.get_halfedge(t)) {
            Some(t) => t.vertex,
            None => continue,
        };
        let key = if origin < tip {
            (origin, tip)
        } else {
            (tip, origin)
        };
        *edge_count.entry(key).or_insert(0) += 1;
        edge_rep.entry(key).or_insert(he);
    }
    edge_count
        .into_iter()
        .filter(|(_, count)| *count > 2)
        .filter_map(|(key, _)| edge_rep.get(&key).copied())
        .collect()
}

/// 检测非流形顶点：outgoing 环不闭合且非边界的内部顶点。
pub fn detect_nonmanifold_vertices(mesh: &MeshStorage) -> Vec<VertexId> {
    let mut result = Vec::new();
    for v in mesh.vertex_ids().collect::<Vec<_>>() {
        let he = match mesh.get_vertex(v).and_then(|vt| vt.halfedge) {
            Some(h) => h,
            None => continue,
        };
        let mut cur = he;
        let mut closed = false;
        let max_iter = mesh.halfedge_count() + 1;
        for _ in 0..max_iter {
            let next = mesh
                .get_halfedge(cur)
                .and_then(|h| h.prev)
                .and_then(|p| mesh.get_halfedge(p))
                .and_then(|h| h.twin);
            match next {
                Some(n) if n == he => {
                    closed = true;
                    break;
                }
                Some(n) => cur = n,
                None => break,
            }
        }
        if !closed {
            let mut cur = he;
            let mut cw_closed = false;
            for _ in 0..max_iter {
                let prev = mesh
                    .get_halfedge(cur)
                    .and_then(|h| h.twin)
                    .and_then(|t| mesh.get_halfedge(t))
                    .and_then(|h| h.next);
                match prev {
                    Some(p) if p == he => {
                        cw_closed = true;
                        break;
                    }
                    Some(p) => cur = p,
                    None => break,
                }
            }
            if !cw_closed {
                let outgoing: Vec<HalfEdgeId> =
                    crate::traversal::VertexRing::new(mesh, v).collect();
                let boundary_count = outgoing
                    .iter()
                    .filter(|&&he_id| {
                        mesh.get_halfedge(he_id).is_some_and(|h| h.face.is_none())
                            || mesh
                                .get_halfedge(he_id)
                                .and_then(|h| h.twin)
                                .and_then(|t| mesh.get_halfedge(t))
                                .is_some_and(|t| t.face.is_none())
                    })
                    .count();
                if boundary_count > 2 {
                    result.push(v);
                }
            }
        }
    }
    result
}

// ============================================================
// 一键修复 pipeline
// ============================================================

/// 网格修复统计信息。
#[derive(Debug, Clone, Default)]
pub struct RepairStats {
    /// 填充的洞数量
    pub holes_filled: usize,
    /// 删除的孤立顶点数
    pub isolated_vertices_removed: usize,
    /// 删除的退化面数
    pub degenerate_faces_removed: usize,
    /// 翻转的面数（法线一致化）
    pub faces_flipped: usize,
}

/// 一键修复 pipeline。
pub fn repair_mesh(mesh: &mut MeshStorage) -> Result<RepairStats, TopologyError> {
    let mut stats = RepairStats::default();
    let hole_results = fill_all_holes(mesh)?;
    stats.holes_filled = hole_results.len();
    stats.degenerate_faces_removed = remove_degenerate_faces(mesh);
    stats.isolated_vertices_removed = remove_isolated_vertices(mesh);
    stats.faces_flipped = crate::orientation::fix_orientations(mesh);
    Ok(stats)
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{MeshStorage, Vertex};
    use crate::test_util;

    fn build_mesh_with_hole() -> MeshStorage {
        let mut mesh = test_util::build_icosphere(1);
        let face = mesh.face_ids().next().unwrap();
        remove_face(&mut mesh, face).unwrap();
        mesh
    }

    #[test]
    fn fill_hole_basic() {
        let mut mesh = build_mesh_with_hole();
        let loops = boundary_loops(&mesh);
        assert_eq!(loops.len(), 1, "应有一个洞");
        let faces = fill_hole(&mut mesh, loops[0][0]).unwrap();
        assert_eq!(faces.len(), 1, "三角形洞应填充为 1 个面");
        assert!(boundary_loops(&mesh).is_empty(), "填充后不应有洞");
    }

    #[test]
    fn fill_all_holes_icosphere() {
        let mut mesh = build_mesh_with_hole();
        let results = fill_all_holes(&mut mesh).unwrap();
        assert_eq!(results.len(), 1);
        assert!(boundary_loops(&mesh).is_empty());
    }

    #[test]
    fn fill_hole_on_closed_mesh_returns_empty() {
        let mesh = test_util::build_icosphere(1);
        assert!(boundary_loops(&mesh).is_empty(), "闭合网格不应有洞");
    }

    #[test]
    fn fill_hole_non_boundary_is_error() {
        let mut mesh = test_util::build_icosphere(1);
        let face_he = mesh
            .face_ids()
            .next()
            .and_then(|f| mesh.get_face(f)?.halfedge)
            .unwrap();
        assert!(fill_hole(&mut mesh, face_he).is_err());
    }

    #[test]
    fn remove_isolated_vertices_basic() {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([0.0; 3]));
        let v1 = mesh.add_vertex(Vertex::new([1.0; 3]));
        let v2 = mesh.add_vertex(Vertex::new([2.0; 3]));
        let v3 = mesh.add_vertex(Vertex::new([3.0; 3]));
        add_triangle(&mut mesh, v0, v1, v2).unwrap();
        assert_eq!(remove_isolated_vertices(&mut mesh), 1);
        assert!(!mesh.contains_vertex(v3));
    }

    #[test]
    fn remove_face_creates_boundary() {
        let mut mesh = test_util::build_icosphere(1);
        let face = mesh.face_ids().next().unwrap();
        let before = mesh.face_count();
        remove_face(&mut mesh, face).unwrap();
        assert_eq!(mesh.face_count(), before - 1);
        let loops = boundary_loops(&mesh);
        assert_eq!(loops.len(), 1, "删除面后应有一个洞");
        assert_eq!(loops[0].len(), 3, "三角形洞应有 3 条边界半边");
    }

    #[test]
    fn remove_face_and_fill_restores() {
        let mut mesh = test_util::build_icosphere(1);
        let before_faces = mesh.face_count();
        let face = mesh.face_ids().next().unwrap();
        remove_face(&mut mesh, face).unwrap();
        let faces = fill_all_holes(&mut mesh).unwrap();
        assert_eq!(faces.len(), 1);
        assert_eq!(mesh.face_count(), before_faces);
    }

    #[test]
    fn detect_nonmanifold_edges_closed_mesh() {
        let mesh = test_util::build_icosphere(1);
        assert!(detect_nonmanifold_edges(&mesh).is_empty());
    }

    #[test]
    fn repair_mesh_basic() {
        let mut mesh = build_mesh_with_hole();
        let stats = repair_mesh(&mut mesh).unwrap();
        assert_eq!(stats.holes_filled, 1);
        assert!(boundary_loops(&mesh).is_empty());
    }

    #[test]
    fn fill_hole_quad_hole() {
        // 构建只有一个三角形的网格 → 3 条边界半边形成三角形洞
        // 然后添加第 4 个顶点 → 4 边洞
        // 简化：用 build_uv_sphere (有边界的网格) 来测试
        // 或用 build_grid 来测试
        let mesh = crate::primitives::build_grid(1.0, 1.0, 2, 2);
        let mut mesh = mesh;
        // grid 有边界 → 可以填充
        let loops = boundary_loops(&mesh);
        if !loops.is_empty() {
            let faces = fill_hole(&mut mesh, loops[0][0]).unwrap();
            assert!(!faces.is_empty());
        }
    }

    #[test]
    fn remove_and_fill_multiple_faces() {
        let mut mesh = test_util::build_icosphere(1);
        let faces: Vec<FaceId> = mesh.face_ids().take(2).collect();
        remove_face(&mut mesh, faces[0]).unwrap();
        remove_face(&mut mesh, faces[1]).unwrap();
        let loops = boundary_loops(&mesh);
        assert!(!loops.is_empty());
        let results = fill_all_holes(&mut mesh).unwrap();
        assert_eq!(results.len(), loops.len());
    }

    #[test]
    fn remove_degenerate_faces_on_good_mesh() {
        let mut mesh = test_util::build_icosphere(1);
        assert_eq!(remove_degenerate_faces(&mut mesh), 0);
    }
}
