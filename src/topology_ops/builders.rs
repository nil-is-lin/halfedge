//! 简单面构建器：add_triangle。

use std::collections::HashMap;

use crate::ids::{FaceId, HalfEdgeId, VertexId};
use crate::storage::{Face, HalfEdge, MeshStorage};

use super::helpers::TopologyError;
use super::validate::validate_mesh;

/// 添加一个三角形面 `(v0, v1, v2)`（CCW 顺序），自动完成半边拓扑连接。
///
/// 自动处理：
/// - 创建 3 条新半边及其 `next`/`prev` 环
/// - 创建新面
/// - 查找已有半边中与每条边反向的 twin 并配对
/// - 对新边创建边界半边（若对应 twin 不存在）
/// - 为 `halfedge` 为 `None` 的顶点设置 outgoing 半边
///
/// # 错误
/// - 若任意两个顶点索引相同（退化三角形），返回 `TopologyError::DegenerateTriangle`
/// - 若任何顶点不存在，返回 `TopologyError::Inconsistent`
/// - 若构建后 `validate_mesh` 失败，返回对应错误
pub fn add_triangle(
    mesh: &mut MeshStorage,
    v0: VertexId,
    v1: VertexId,
    v2: VertexId,
) -> Result<FaceId, TopologyError> {
    // ---------- 1. 校验 ----------
    if v0 == v1 || v1 == v2 || v0 == v2 {
        return Err(TopologyError::DegenerateTriangle);
    }
    if !mesh.contains_vertex(v0) || !mesh.contains_vertex(v1) || !mesh.contains_vertex(v2) {
        return Err(TopologyError::Inconsistent("顶点不存在".into()));
    }

    // ---------- 2. 创建 3 条半边 ----------
    let h0 = mesh.add_halfedge(HalfEdge::new(v1)); // v0→v1
    let h1 = mesh.add_halfedge(HalfEdge::new(v2)); // v1→v2
    let h2 = mesh.add_halfedge(HalfEdge::new(v0)); // v2→v0

    // 设置 next/prev 环
    for (he, next, prev) in [(h0, h1, h2), (h1, h2, h0), (h2, h0, h1)] {
        let h = mesh
            .get_halfedge_mut(he)
            .expect("he just created by add_halfedge");
        h.next = Some(next);
        h.prev = Some(prev);
    }

    // ---------- 3. 创建面 ----------
    let face = mesh.add_face(Face::new());
    mesh.get_face_mut(face).expect("face just created").halfedge = Some(h0);
    for he in [h0, h1, h2] {
        mesh.get_halfedge_mut(he)
            .expect("he just created by add_halfedge")
            .face = Some(face);
    }

    // ---------- 4. 为每条边找或创建 twin ----------
    // 一次性建立边索引：key = (origin, tip) = (twin.vertex, he.vertex)，
    // 仅收录 twin 为边界半边的内部半边。后续 3 次查找 O(1)。
    let mut boundary_twin_map: HashMap<(VertexId, VertexId), HalfEdgeId> = HashMap::new();
    for ehe in mesh.halfedge_ids() {
        if ehe == h0 || ehe == h1 || ehe == h2 {
            continue;
        }
        let h = match mesh.get_halfedge(ehe) {
            Some(h) => h,
            None => continue,
        };
        if let Some(twin_id) = h.twin
            && let Some(twin_data) = mesh.get_halfedge(twin_id)
            && twin_data.face.is_none()
        {
            // h 从 twin_data.vertex → h.vertex，且 twin 是边界半边
            boundary_twin_map.insert((twin_data.vertex, h.vertex), ehe);
        }
    }

    // 对每条新边 he: src→dst，查找已有半边中方向为 dst→src 且 twin 为边界的。
    let edges = [(h0, v0, v1), (h1, v1, v2), (h2, v2, v0)];
    for (he, src, dst) in edges {
        // 查找 key = (dst, src)：即从 dst→src 的内部半边，其 twin（src→dst）为边界
        let existing: Option<HalfEdgeId> = boundary_twin_map.get(&(dst, src)).copied();

        match existing {
            Some(ex) => {
                // 已有半边 E: dst→src，其 twin 是边界半边。
                // 将新边 he 与 E 配对，删除旧边界 twin。
                let old_twin = mesh
                    .get_halfedge(ex)
                    .expect("ex from boundary_twin_map, validated")
                    .twin;
                if let Some(old) = old_twin {
                    mesh.remove_halfedge(old);
                }
                mesh.get_halfedge_mut(he).expect("he just created").twin = Some(ex);
                mesh.get_halfedge_mut(ex)
                    .expect("ex from boundary_twin_map, validated")
                    .twin = Some(he);
            }
            None => {
                // 无匹配：创建新的边界半边 dst→src
                let twin = mesh.add_halfedge(HalfEdge::new(src));
                mesh.get_halfedge_mut(he).expect("he just created").twin = Some(twin);
                mesh.get_halfedge_mut(twin).expect("twin just created").twin = Some(he);
            }
        }
    }

    // ---------- 6. 设置顶点 outgoing 半边入口 ----------
    for (v, he) in [(v0, h0), (v1, h1), (v2, h2)] {
        if mesh
            .get_vertex(v)
            .expect("v validated by contains_vertex")
            .halfedge
            .is_none()
        {
            mesh.get_vertex_mut(v)
                .expect("v validated by contains_vertex")
                .halfedge = Some(he);
        }
    }

    // ---------- 7. 最终校验 ----------
    validate_mesh(mesh)?;

    Ok(face)
}
