//! 拓扑自检模块
//!
//! 提供 [`validate_topology`]：对 [`MeshStorage`] 执行**完整**的流形三角曲面
//! 不变量校验，输出所有违例的列表（而非首个错误）。
//!
//! ## 与 [`crate::topology_ops::validate_mesh`] 的区别
//! - `validate_mesh`：**轻量级**，仅校验 twin 互指 / next-prev 一致 / 面边界环长度，
//!   用于拓扑操作前后快速断言；遇到首个错误即返回。
//! - `validate_topology`：**完整**，额外校验悬空 ID、退化面、流形约束、
//!   顶点/面入口字段一致性，收集所有错误后返回。
//!
//! ## 校验项
//! 1. **句柄有效性**：所有 `vertex/face/halfedge` 引用必须指向存活元素；
//! 2. **twin 双向匹配**：`A.twin = B` ⟹ `B.twin = A`，且 `A.vertex ≠ B.vertex`（无自环）；
//! 3. **next/prev 一致性**：`A.next = B` ⟺ `B.prev = A`；
//! 4. **面边界环**：每个面的 `next` 链闭合，长度恰为 3；
//! 5. **入口字段一致**：`Vertex.halfedge` 的 origin 必须是该顶点；
//!    `Face.halfedge` 的 `face` 字段必须指向该面；
//! 6. **退化面**：三角面三顶点不共线，面积大于阈值；
//! 7. **流形约束**：每条无向边至多被 2 个面共享；每个顶点 outgoing 环
//!    或闭合（内部顶点）或开链且两端为边界半边（边界顶点）。

use std::fmt;

use crate::ids::{FaceId, HalfEdgeId, VertexId};
use crate::storage::MeshStorage;
use crate::traversal::{FaceHalfEdges, VertexRing};

/// 退化面面积阈值（小于此值视为退化）。
const DEGENERATE_AREA_EPS: f64 = 1e-12;

/// 拓扑校验错误。
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    /// 半边的 `vertex` 字段指向已删除的顶点。
    HalfEdgeDanglingVertex { he: HalfEdgeId, vertex: VertexId },
    /// 半边的 `face` 字段指向已删除的面。
    HalfEdgeDanglingFace { he: HalfEdgeId, face: FaceId },
    /// 半边的 `twin` 字段指向已删除的半边。
    HalfEdgeDanglingTwin { he: HalfEdgeId, twin: HalfEdgeId },
    /// 半边的 `next` 字段指向已删除的半边。
    HalfEdgeDanglingNext { he: HalfEdgeId, next: HalfEdgeId },
    /// 半边的 `prev` 字段指向已删除的半边。
    HalfEdgeDanglingPrev { he: HalfEdgeId, prev: HalfEdgeId },
    /// twin 不互指：`A.twin = B` 但 `B.twin ≠ A`。
    TwinMismatch {
        a: HalfEdgeId,
        b: HalfEdgeId,
        b_twin: Option<HalfEdgeId>,
    },
    /// 自环半边：`he.vertex == he.twin.vertex`。
    SelfLoopHalfEdge(HalfEdgeId),
    /// next/prev 不一致：`A.next = B` 但 `B.prev ≠ A`。
    NextPrevMismatch {
        he: HalfEdgeId,
        next: HalfEdgeId,
        next_prev: Option<HalfEdgeId>,
    },
    /// 面边界环不闭合或长度非 3。
    FaceNotTriangular { face: FaceId, boundary_len: usize },
    /// 顶点的 `halfedge` 入口指向无效半边，或半边的 origin 不是该顶点。
    VertexHalfEdgeInconsistent { v: VertexId, he: Option<HalfEdgeId> },
    /// 面的 `halfedge` 入口指向无效半边，或半边的 `face` 不是该面。
    FaceHalfEdgeInconsistent { f: FaceId, he: Option<HalfEdgeId> },
    /// 退化面（三顶点共线或重合）。
    DegenerateFace { face: FaceId, area: f64 },
    /// 非流形边：同一条无向边被超过 2 个面共享（无法用 twin 表达）。
    /// 注意：在标准半边结构中，每条半边只有一个 twin，因此非流形边通常表现为
    /// 多条独立的 twin 对共端点。此处通过「同端点的不同半边对」计数检测。
    NonManifoldEdge {
        endpoint_a: VertexId,
        endpoint_b: VertexId,
        face_count: usize,
    },
    /// 顶点 outgoing 环异常：内部顶点未闭合，或边界顶点开链端点数 ≠ 2。
    NonManifoldVertex { v: VertexId, boundary_count: usize },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HalfEdgeDanglingVertex { he, vertex } => {
                write!(f, "半边 {:?} 的 vertex {:?} 已删除", he, vertex)
            }
            Self::HalfEdgeDanglingFace { he, face } => {
                write!(f, "半边 {:?} 的 face {:?} 已删除", he, face)
            }
            Self::HalfEdgeDanglingTwin { he, twin } => {
                write!(f, "半边 {:?} 的 twin {:?} 已删除", he, twin)
            }
            Self::HalfEdgeDanglingNext { he, next } => {
                write!(f, "半边 {:?} 的 next {:?} 已删除", he, next)
            }
            Self::HalfEdgeDanglingPrev { he, prev } => {
                write!(f, "半边 {:?} 的 prev {:?} 已删除", he, prev)
            }
            Self::TwinMismatch { a, b, b_twin } => {
                write!(
                    f,
                    "twin 不互指：{:?}.twin={:?}, 但 {:?}.twin={:?}",
                    a, b, b, b_twin
                )
            }
            Self::SelfLoopHalfEdge(he) => write!(f, "半边 {:?} 是自环（origin==tip）", he),
            Self::NextPrevMismatch {
                he,
                next,
                next_prev,
            } => {
                write!(
                    f,
                    "next/prev 不一致：{:?}.next={:?}, 但 {:?}.prev={:?}",
                    he, next, next, next_prev
                )
            }
            Self::FaceNotTriangular { face, boundary_len } => {
                write!(f, "面 {:?} 边界环长度={}, 非三角面", face, boundary_len)
            }
            Self::VertexHalfEdgeInconsistent { v, he } => {
                write!(f, "顶点 {:?} 的 halfedge 入口 {:?} 不一致", v, he)
            }
            Self::FaceHalfEdgeInconsistent { f: fid, he } => {
                write!(f, "面 {:?} 的 halfedge 入口 {:?} 不一致", fid, he)
            }
            Self::DegenerateFace { face, area } => {
                write!(f, "面 {:?} 退化，面积={:.2e}", face, area)
            }
            Self::NonManifoldEdge {
                endpoint_a,
                endpoint_b,
                face_count,
            } => {
                write!(
                    f,
                    "非流形边：{:?}-{:?} 被 {} 个面共享",
                    endpoint_a, endpoint_b, face_count
                )
            }
            Self::NonManifoldVertex { v, boundary_count } => {
                write!(
                    f,
                    "非流形顶点 {:?}: outgoing 环边界端点数={}",
                    v, boundary_count
                )
            }
        }
    }
}

impl std::error::Error for ValidationError {}

// ============================================================
// 主校验函数
// ============================================================

/// 完整校验网格拓扑。
///
/// 收集所有违例返回。若返回空 `Vec` 表示网格通过校验。
///
/// # 复杂度
/// $O(V + E + F)$：每个元素遍历常数次。
pub fn validate_topology(mesh: &MeshStorage) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    validate_halfedge_references(mesh, &mut errors);
    validate_twin_relations(mesh, &mut errors);
    validate_next_prev(mesh, &mut errors);
    validate_face_boundaries(mesh, &mut errors);
    validate_entry_fields(mesh, &mut errors);
    validate_degenerate_faces(mesh, &mut errors);
    validate_manifold_edges(mesh, &mut errors);
    validate_manifold_vertices(mesh, &mut errors);

    errors
}

/// 便捷形式：通过校验返回 `Ok(())`，否则返回 `Err(Vec)`。
pub fn check_topology(mesh: &MeshStorage) -> Result<(), Vec<ValidationError>> {
    let errors = validate_topology(mesh);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// ============================================================
// 各项校验
// ============================================================

/// 1. 句柄有效性：检查每条半边的 vertex/face/twin/next/prev 引用是否指向存活元素。
fn validate_halfedge_references(mesh: &MeshStorage, errors: &mut Vec<ValidationError>) {
    for he_id in mesh.halfedge_ids().collect::<Vec<_>>() {
        let he = match mesh.get_halfedge(he_id) {
            Some(h) => h,
            None => continue,
        };
        if !mesh.contains_vertex(he.vertex) {
            errors.push(ValidationError::HalfEdgeDanglingVertex {
                he: he_id,
                vertex: he.vertex,
            });
        }
        if let Some(f) = he.face
            && !mesh.contains_face(f)
        {
            errors.push(ValidationError::HalfEdgeDanglingFace { he: he_id, face: f });
        }
        if let Some(t) = he.twin
            && !mesh.contains_halfedge(t)
        {
            errors.push(ValidationError::HalfEdgeDanglingTwin { he: he_id, twin: t });
        }
        if let Some(n) = he.next
            && !mesh.contains_halfedge(n)
        {
            errors.push(ValidationError::HalfEdgeDanglingNext { he: he_id, next: n });
        }
        if let Some(p) = he.prev
            && !mesh.contains_halfedge(p)
        {
            errors.push(ValidationError::HalfEdgeDanglingPrev { he: he_id, prev: p });
        }
    }
}

/// 2. twin 双向匹配 + 自环检查。
fn validate_twin_relations(mesh: &MeshStorage, errors: &mut Vec<ValidationError>) {
    for he_id in mesh.halfedge_ids().collect::<Vec<_>>() {
        let he = match mesh.get_halfedge(he_id) {
            Some(h) => h,
            None => continue,
        };
        if let Some(twin_id) = he.twin {
            let twin = match mesh.get_halfedge(twin_id) {
                Some(t) => t,
                None => continue, // 已由 references 报告
            };
            if twin.twin != Some(he_id) {
                errors.push(ValidationError::TwinMismatch {
                    a: he_id,
                    b: twin_id,
                    b_twin: twin.twin,
                });
            }
            // 自环：twin.vertex 是 origin，he.vertex 是 tip；二者相同即自环
            if twin.vertex == he.vertex {
                errors.push(ValidationError::SelfLoopHalfEdge(he_id));
            }
        }
    }
}

/// 3. next/prev 一致性。
fn validate_next_prev(mesh: &MeshStorage, errors: &mut Vec<ValidationError>) {
    for he_id in mesh.halfedge_ids().collect::<Vec<_>>() {
        let he = match mesh.get_halfedge(he_id) {
            Some(h) => h,
            None => continue,
        };
        if let Some(next_id) = he.next {
            let next = match mesh.get_halfedge(next_id) {
                Some(n) => n,
                None => continue,
            };
            if next.prev != Some(he_id) {
                errors.push(ValidationError::NextPrevMismatch {
                    he: he_id,
                    next: next_id,
                    next_prev: next.prev,
                });
            }
        }
    }
}

/// 4. 面边界环长度为 3。
fn validate_face_boundaries(mesh: &MeshStorage, errors: &mut Vec<ValidationError>) {
    for f_id in mesh.face_ids().collect::<Vec<_>>() {
        let f = match mesh.get_face(f_id) {
            Some(f) => f,
            None => continue,
        };
        let Some(start) = f.halfedge else {
            errors.push(ValidationError::FaceHalfEdgeInconsistent { f: f_id, he: None });
            continue;
        };
        if !mesh.contains_halfedge(start) {
            errors.push(ValidationError::FaceHalfEdgeInconsistent {
                f: f_id,
                he: Some(start),
            });
            continue;
        }
        let mut count = 0usize;
        let mut cur = start;
        let max_iter = mesh.halfedge_count() + 1;
        for _ in 0..max_iter {
            count += 1;
            match mesh.get_halfedge(cur).and_then(|h| h.next) {
                Some(n) if n != start => cur = n,
                _ => break,
            }
        }
        if count != 3 {
            errors.push(ValidationError::FaceNotTriangular {
                face: f_id,
                boundary_len: count,
            });
        }
    }
}

/// 5. 入口字段一致性：Vertex.halfedge / Face.halfedge。
fn validate_entry_fields(mesh: &MeshStorage, errors: &mut Vec<ValidationError>) {
    // 顶点入口
    for v_id in mesh.vertex_ids().collect::<Vec<_>>() {
        let v = match mesh.get_vertex(v_id) {
            Some(v) => v,
            None => continue,
        };
        match v.halfedge {
            None => {
                // 无 outgoing 入口：若该顶点确实无邻接半边则合法（孤立顶点），
                // 否则报告。检查方式：扫描是否有半边的 twin.vertex == v_id。
                let has_incoming = mesh.halfedge_ids().any(|h| {
                    mesh.get_halfedge(h)
                        .map(|he| he.vertex == v_id)
                        .unwrap_or(false)
                });
                if has_incoming {
                    errors.push(ValidationError::VertexHalfEdgeInconsistent { v: v_id, he: None });
                }
            }
            Some(he) => {
                if !mesh.contains_halfedge(he) {
                    errors.push(ValidationError::VertexHalfEdgeInconsistent {
                        v: v_id,
                        he: Some(he),
                    });
                } else {
                    // 校验 origin == v_id：he.twin.vertex 应为 v_id
                    let origin_ok = mesh
                        .get_halfedge(he)
                        .and_then(|h| h.twin)
                        .and_then(|t| mesh.get_halfedge(t))
                        .map(|t| t.vertex == v_id)
                        .unwrap_or(false);
                    if !origin_ok {
                        errors.push(ValidationError::VertexHalfEdgeInconsistent {
                            v: v_id,
                            he: Some(he),
                        });
                    }
                }
            }
        }
    }

    // 面入口
    for f_id in mesh.face_ids().collect::<Vec<_>>() {
        let f = match mesh.get_face(f_id) {
            Some(f) => f,
            None => continue,
        };
        match f.halfedge {
            None => {
                errors.push(ValidationError::FaceHalfEdgeInconsistent { f: f_id, he: None });
            }
            Some(he) => {
                if !mesh.contains_halfedge(he) {
                    errors.push(ValidationError::FaceHalfEdgeInconsistent {
                        f: f_id,
                        he: Some(he),
                    });
                } else {
                    let face_ok = mesh
                        .get_halfedge(he)
                        .map(|h| h.face == Some(f_id))
                        .unwrap_or(false);
                    if !face_ok {
                        errors.push(ValidationError::FaceHalfEdgeInconsistent {
                            f: f_id,
                            he: Some(he),
                        });
                    }
                }
            }
        }
    }
}

/// 6. 退化面：面积接近零。
fn validate_degenerate_faces(mesh: &MeshStorage, errors: &mut Vec<ValidationError>) {
    for f_id in mesh.face_ids().collect::<Vec<_>>() {
        let verts: Vec<_> = FaceHalfEdges::new(mesh, f_id)
            .filter_map(|he| mesh.get_halfedge(he))
            .filter_map(|h| mesh.get_vertex(h.vertex))
            .map(|v| v.position)
            .collect();
        if verts.len() != 3 {
            continue; // 已由 face_boundaries 报告
        }
        let area = triangle_area(verts[0], verts[1], verts[2]);
        if area < DEGENERATE_AREA_EPS {
            errors.push(ValidationError::DegenerateFace { face: f_id, area });
        }
    }
}

/// 7. 流形边检查：每条无向边（端点对）至多被 2 个面共享。
fn validate_manifold_edges(mesh: &MeshStorage, errors: &mut Vec<ValidationError>) {
    use std::collections::HashMap;
    // 端点对（小到大排序作为无向边键）→ 关联面数
    let mut edge_face_count: HashMap<(VertexId, VertexId), usize> = HashMap::new();

    for f_id in mesh.face_ids().collect::<Vec<_>>() {
        let verts: Vec<_> = FaceHalfEdges::new(mesh, f_id)
            .filter_map(|he| mesh.get_halfedge(he))
            .map(|h| h.vertex)
            .collect();
        for w in verts.windows(2) {
            let (a, b) = (w[0], w[1]);
            let key = if a < b { (a, b) } else { (b, a) };
            *edge_face_count.entry(key).or_insert(0) += 1;
        }
        // 闭合环：最后一条边
        if verts.len() >= 3 {
            let (a, b) = (verts[verts.len() - 1], verts[0]);
            let key = if a < b { (a, b) } else { (b, a) };
            *edge_face_count.entry(key).or_insert(0) += 1;
        }
    }

    for ((a, b), count) in edge_face_count {
        if count > 2 {
            errors.push(ValidationError::NonManifoldEdge {
                endpoint_a: a,
                endpoint_b: b,
                face_count: count,
            });
        }
    }
}

/// 8. 流形顶点检查：outgoing 环或闭合（内部顶点），或开链恰有 2 个边界端点（边界顶点）。
fn validate_manifold_vertices(mesh: &MeshStorage, errors: &mut Vec<ValidationError>) {
    for v_id in mesh.vertex_ids().collect::<Vec<_>>() {
        let out_he: Vec<HalfEdgeId> = VertexRing::new(mesh, v_id).collect();
        if out_he.is_empty() {
            continue; // 孤立顶点
        }
        // 统计 outgoing 半边中边界半边的数量
        // 边界半边定义：face=None 或 twin.face=None
        let mut boundary_count = 0;
        for he in &out_he {
            let is_boundary = mesh
                .get_halfedge(*he)
                .map(|h| {
                    if h.face.is_none() {
                        return true;
                    }
                    h.twin
                        .map(|t| {
                            mesh.get_halfedge(t)
                                .map(|th| th.face.is_none())
                                .unwrap_or(true)
                        })
                        .unwrap_or(true)
                })
                .unwrap_or(true);
            if is_boundary {
                boundary_count += 1;
            }
        }
        // 流形顶点：boundary_count ∈ {0, 2}
        // 0 = 内部顶点（闭合环）；2 = 边界顶点（开链两端）
        if boundary_count != 0 && boundary_count != 2 {
            errors.push(ValidationError::NonManifoldVertex {
                v: v_id,
                boundary_count,
            });
        }
    }
}

// ============================================================
// 辅助
// ============================================================

#[inline]
fn triangle_area(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{Face, HalfEdge, MeshStorage, Vertex};
    use crate::topology_ops::{flip_edge, split_edge};

    /// 构造一个干净的单三角面片。
    fn build_clean_triangle() -> (MeshStorage, [VertexId; 3], FaceId) {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));

        let h0 = mesh.add_halfedge(HalfEdge::new(v1));
        let h1 = mesh.add_halfedge(HalfEdge::new(v2));
        let h2 = mesh.add_halfedge(HalfEdge::new(v0));
        let t0 = mesh.add_halfedge(HalfEdge::new(v0));
        let t1 = mesh.add_halfedge(HalfEdge::new(v1));
        let t2 = mesh.add_halfedge(HalfEdge::new(v2));

        let f = mesh.add_face(Face::new());
        for (he, twin, next, prev) in [(h0, t0, h1, h2), (h1, t1, h2, h0), (h2, t2, h0, h1)] {
            let h = mesh.get_halfedge_mut(he).unwrap();
            h.twin = Some(twin);
            h.next = Some(next);
            h.prev = Some(prev);
            h.face = Some(f);
        }
        for (t, he) in [(t0, h0), (t1, h1), (t2, h2)] {
            mesh.get_halfedge_mut(t).unwrap().twin = Some(he);
        }
        mesh.get_vertex_mut(v0).unwrap().halfedge = Some(h0);
        mesh.get_vertex_mut(v1).unwrap().halfedge = Some(h1);
        mesh.get_vertex_mut(v2).unwrap().halfedge = Some(h2);
        mesh.get_face_mut(f).unwrap().halfedge = Some(h0);

        (mesh, [v0, v1, v2], f)
    }

    #[test]
    fn clean_triangle_passes_validation() {
        let (mesh, _v, _f) = build_clean_triangle();
        let errors = validate_topology(&mesh);
        assert!(errors.is_empty(), "应有 0 个错误，实际: {:?}", errors);
    }

    #[test]
    fn clean_closed_fan_passes_validation() {
        // 3 个三角形围成闭合扇形（中心顶点为内部）
        let mut mesh = MeshStorage::new();
        let c = mesh.add_vertex(Vertex::new([0.5, 0.5, 0.0]));
        let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let v2 = mesh.add_vertex(Vertex::new([0.5, 1.0, 0.0]));

        let a1 = mesh.add_halfedge(HalfEdge::new(v0));
        let b1 = mesh.add_halfedge(HalfEdge::new(v1));
        let c1 = mesh.add_halfedge(HalfEdge::new(c));
        let a2 = mesh.add_halfedge(HalfEdge::new(v1));
        let b2 = mesh.add_halfedge(HalfEdge::new(v2));
        let c2 = mesh.add_halfedge(HalfEdge::new(c));
        let a3 = mesh.add_halfedge(HalfEdge::new(v2));
        let b3 = mesh.add_halfedge(HalfEdge::new(v0));
        let c3 = mesh.add_halfedge(HalfEdge::new(c));
        let t1 = mesh.add_halfedge(HalfEdge::new(v0));
        let t2 = mesh.add_halfedge(HalfEdge::new(v1));
        let t3 = mesh.add_halfedge(HalfEdge::new(v2));

        let f1 = mesh.add_face(Face::new());
        let f2 = mesh.add_face(Face::new());
        let f3 = mesh.add_face(Face::new());

        for (he, twin, next, prev, face) in [
            (a1, c3, b1, c1, f1),
            (b1, t1, c1, a1, f1),
            (c1, a2, a1, b1, f1),
            (a2, c1, b2, c2, f2),
            (b2, t2, c2, a2, f2),
            (c2, a3, a2, b2, f2),
            (a3, c2, b3, c3, f3),
            (b3, t3, c3, a3, f3),
            (c3, a1, a3, b3, f3),
        ] {
            let h = mesh.get_halfedge_mut(he).unwrap();
            h.twin = Some(twin);
            h.next = Some(next);
            h.prev = Some(prev);
            h.face = Some(face);
        }
        for (t, he) in [(t1, b1), (t2, b2), (t3, b3)] {
            mesh.get_halfedge_mut(t).unwrap().twin = Some(he);
        }
        mesh.get_vertex_mut(c).unwrap().halfedge = Some(a1);
        mesh.get_vertex_mut(v0).unwrap().halfedge = Some(b1);
        mesh.get_vertex_mut(v1).unwrap().halfedge = Some(b2);
        mesh.get_vertex_mut(v2).unwrap().halfedge = Some(b3);
        mesh.get_face_mut(f1).unwrap().halfedge = Some(a1);
        mesh.get_face_mut(f2).unwrap().halfedge = Some(a2);
        mesh.get_face_mut(f3).unwrap().halfedge = Some(a3);

        let errors = validate_topology(&mesh);
        assert!(errors.is_empty(), "应有 0 个错误，实际: {:?}", errors);
    }

    #[test]
    fn detects_twin_mismatch() {
        let (mut mesh, _v, _f) = build_clean_triangle();
        // 故意破坏 twin：让 t0.twin 指向 t1（错误）
        let t0 = mesh
            .halfedge_ids()
            .find(|h| {
                mesh.get_halfedge(*h)
                    .map(|he| he.face.is_none() && he.vertex == _v[0])
                    .unwrap_or(false)
            })
            .unwrap();
        let t1 = mesh
            .halfedge_ids()
            .find(|h| {
                mesh.get_halfedge(*h)
                    .map(|he| he.face.is_none() && he.vertex == _v[1])
                    .unwrap_or(false)
            })
            .unwrap();
        mesh.get_halfedge_mut(t0).unwrap().twin = Some(t1);
        let errors = validate_topology(&mesh);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::TwinMismatch { .. })),
            "应检测到 twin 不匹配, 实际: {:?}",
            errors
        );
    }

    #[test]
    fn detects_dangling_vertex_reference() {
        let (mut mesh, v, _f) = build_clean_triangle();
        // 故意让某条半边指向已删除的顶点
        let h0 = mesh
            .halfedge_ids()
            .find(|h| {
                mesh.get_halfedge(*h)
                    .map(|he| he.vertex == v[1] && he.face.is_some())
                    .unwrap_or(false)
            })
            .unwrap();
        let bad_v = VertexId::default();
        mesh.get_halfedge_mut(h0).unwrap().vertex = bad_v;
        let errors = validate_topology(&mesh);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::HalfEdgeDanglingVertex { .. })),
            "应检测到悬空顶点引用"
        );
    }

    #[test]
    fn detects_degenerate_face() {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        // 共线
        let v2 = mesh.add_vertex(Vertex::new([2.0, 0.0, 0.0]));

        let h0 = mesh.add_halfedge(HalfEdge::new(v1));
        let h1 = mesh.add_halfedge(HalfEdge::new(v2));
        let h2 = mesh.add_halfedge(HalfEdge::new(v0));
        let t0 = mesh.add_halfedge(HalfEdge::new(v0));
        let t1 = mesh.add_halfedge(HalfEdge::new(v1));
        let t2 = mesh.add_halfedge(HalfEdge::new(v2));

        let f = mesh.add_face(Face::new());
        for (he, twin, next, prev) in [(h0, t0, h1, h2), (h1, t1, h2, h0), (h2, t2, h0, h1)] {
            let h = mesh.get_halfedge_mut(he).unwrap();
            h.twin = Some(twin);
            h.next = Some(next);
            h.prev = Some(prev);
            h.face = Some(f);
        }
        for (t, he) in [(t0, h0), (t1, h1), (t2, h2)] {
            mesh.get_halfedge_mut(t).unwrap().twin = Some(he);
        }
        mesh.get_vertex_mut(v0).unwrap().halfedge = Some(h0);
        mesh.get_vertex_mut(v1).unwrap().halfedge = Some(h1);
        mesh.get_vertex_mut(v2).unwrap().halfedge = Some(h2);
        mesh.get_face_mut(f).unwrap().halfedge = Some(h0);

        let errors = validate_topology(&mesh);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::DegenerateFace { .. })),
            "应检测到退化面, 实际: {:?}",
            errors
        );
    }

    #[test]
    fn detects_non_triangular_face() {
        let (mut mesh, v, _f) = build_clean_triangle();
        // 在面中插入第 4 条半边（构造四边形）
        let v3 = mesh.add_vertex(Vertex::new([1.0, 1.0, 0.0]));
        let h_extra = mesh.add_halfedge(HalfEdge::new(v3));
        // 让 h0.next = h_extra, h_extra.next = h1, h_extra.prev = h0, h1.prev = h_extra
        let h0 = mesh
            .halfedge_ids()
            .find(|h| {
                mesh.get_halfedge(*h)
                    .map(|he| he.vertex == v[1] && he.face.is_some())
                    .unwrap_or(false)
            })
            .unwrap();
        let h1 = mesh
            .halfedge_ids()
            .find(|h| {
                mesh.get_halfedge(*h)
                    .map(|he| he.vertex == v[2] && he.face.is_some())
                    .unwrap_or(false)
            })
            .unwrap();
        mesh.get_halfedge_mut(h0).unwrap().next = Some(h_extra);
        mesh.get_halfedge_mut(h1).unwrap().prev = Some(h_extra);
        mesh.get_halfedge_mut(h_extra).unwrap().next = Some(h1);
        mesh.get_halfedge_mut(h_extra).unwrap().prev = Some(h0);
        mesh.get_halfedge_mut(h_extra).unwrap().face = Some(_f);

        let errors = validate_topology(&mesh);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::FaceNotTriangular { boundary_len, .. } if *boundary_len >= 4)),
            "应检测到非三角面, 实际: {:?}",
            errors
        );
    }

    #[test]
    fn detects_vertex_halfedge_inconsistency() {
        let (mut mesh, v, _f) = build_clean_triangle();
        // 让 v0.halfedge 指向 v1 的 outgoing（origin 不匹配）
        let h1 = mesh
            .halfedge_ids()
            .find(|h| {
                mesh.get_halfedge(*h)
                    .map(|he| he.vertex == v[2] && he.face.is_some())
                    .unwrap_or(false)
            })
            .unwrap();
        mesh.get_vertex_mut(v[0]).unwrap().halfedge = Some(h1);
        let errors = validate_topology(&mesh);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::VertexHalfEdgeInconsistent { .. })),
            "应检测到顶点入口不一致"
        );
    }

    #[test]
    fn topology_ops_preserve_validity() {
        // 经过 split / flip / collapse 后仍应通过完整校验
        let (mut mesh, _v, _f) = build_clean_triangle();
        // 边分裂（边界边）
        let he_boundary = mesh
            .halfedge_ids()
            .find(|h| {
                mesh.get_halfedge(*h)
                    .map(|he| he.face.is_some())
                    .unwrap_or(false)
            })
            .unwrap();
        let _ = split_edge(&mut mesh, he_boundary).unwrap();
        assert!(
            check_topology(&mesh).is_ok(),
            "split 后应通过校验: {:?}",
            check_topology(&mesh)
        );
    }

    #[test]
    fn flip_preserves_full_validity() {
        // 构造两个三角形拼成的四边形（v3 在共享边下方，F2 几何 CCW）
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));
        let v3 = mesh.add_vertex(Vertex::new([1.0, -1.0, 0.0]));

        let h0 = mesh.add_halfedge(HalfEdge::new(v1));
        let h1 = mesh.add_halfedge(HalfEdge::new(v2));
        let h2 = mesh.add_halfedge(HalfEdge::new(v0));
        let g0 = mesh.add_halfedge(HalfEdge::new(v0));
        let g1 = mesh.add_halfedge(HalfEdge::new(v3));
        let g2 = mesh.add_halfedge(HalfEdge::new(v1));
        let t1 = mesh.add_halfedge(HalfEdge::new(v1));
        let t2 = mesh.add_halfedge(HalfEdge::new(v2));
        let t_g1 = mesh.add_halfedge(HalfEdge::new(v0));
        let t_g2 = mesh.add_halfedge(HalfEdge::new(v3));

        let f1 = mesh.add_face(Face::new());
        let f2 = mesh.add_face(Face::new());

        for (he, twin, next, prev) in [(h0, g0, h1, h2), (h1, t1, h2, h0), (h2, t2, h0, h1)] {
            let h = mesh.get_halfedge_mut(he).unwrap();
            h.twin = Some(twin);
            h.next = Some(next);
            h.prev = Some(prev);
            h.face = Some(f1);
        }
        for (he, twin, next, prev) in [(g0, h0, g1, g2), (g1, t_g1, g2, g0), (g2, t_g2, g0, g1)] {
            let h = mesh.get_halfedge_mut(he).unwrap();
            h.twin = Some(twin);
            h.next = Some(next);
            h.prev = Some(prev);
            h.face = Some(f2);
        }
        for (t, he) in [(t1, h1), (t2, h2), (t_g1, g1), (t_g2, g2)] {
            mesh.get_halfedge_mut(t).unwrap().twin = Some(he);
        }
        mesh.get_vertex_mut(v0).unwrap().halfedge = Some(h0);
        mesh.get_vertex_mut(v1).unwrap().halfedge = Some(g0);
        mesh.get_vertex_mut(v2).unwrap().halfedge = Some(h2);
        mesh.get_vertex_mut(v3).unwrap().halfedge = Some(g2);
        mesh.get_face_mut(f1).unwrap().halfedge = Some(h0);
        mesh.get_face_mut(f2).unwrap().halfedge = Some(g0);

        assert!(check_topology(&mesh).is_ok());
        flip_edge(&mut mesh, h0).unwrap();
        assert!(
            check_topology(&mesh).is_ok(),
            "flip 后应通过校验: {:?}",
            check_topology(&mesh)
        );
    }
}
