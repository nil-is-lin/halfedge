//! 拓扑操作模块
//!
//! 在 [`MeshStorage`] 之上叠加拓扑操作：
//! - [`split_edge`]：边分裂（在中点插入新顶点）
//! - [`flip_edge`]：内部边翻转（替换四边形对角线）
//! - [`collapse_edge`] / [`collapse_edge_at`]：边折叠（合并两端顶点）
//! - [`split_face`]：面分裂 / poke（在面中心插入新顶点，1→3 面）
//! - [`extrude_face`] / [`extrude_faces`] / [`extrude_region`]：面挤出
//! - [`add_triangle`]：高级面构建器，自动完成半边拓扑连接与 twin 配对
//!
//! 每个操作内置合法性校验（`validate_mesh`），保证操作后网格仍为流形三角曲面。
//!
//! ## 拓扑约定
//! - `HalfEdge.vertex` 是 tip（目的顶点），origin = `twin.vertex`
//! - 同面 `next/prev` 形成 CCW 闭合环
//! - `twin` 互指，构成无向边
//! - 边界半边 `face = None`
//!
//! ## 旋转规则（CCW 朝向网格，绕 origin）
//! - CCW next: `he.prev.twin`
//! - CW next: `he.twin.next`

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::ids::{FaceId, HalfEdgeId, VertexId};
use crate::storage::{Face, HalfEdge, MeshStorage, Vertex};
use crate::traversal::{FaceHalfEdges, VertexAdjacentFaces, VertexAdjacentVerts, VertexRing};

// ============================================================
// 错误类型
// ============================================================

/// 拓扑操作失败原因。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyError {
    /// 传入的半边句柄无效或已被删除。
    InvalidHalfEdge(HalfEdgeId),
    /// 试图翻转边界边（禁止）。
    FlipOnBoundaryEdge(HalfEdgeId),
    /// 试图折叠边界边（禁止）。
    CollapseOnBoundaryEdge(HalfEdgeId),
    /// 半边没有 twin（拓扑未完整缝合）。
    NoTwin(HalfEdgeId),
    /// 半边没有 face（两侧均无面，无法操作）。
    NoFace(HalfEdgeId),
    /// 操作会产生退化三角形（三个顶点共线或重合）。
    DegenerateTriangle,
    /// 链接条件不满足，折叠会产生非流形。
    LinkConditionViolated { a: VertexId, b: VertexId },
    /// 网格拓扑不一致（twin 不互指、next/prev 不闭合等）。
    Inconsistent(String),
}

impl fmt::Display for TopologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHalfEdge(h) => write!(f, "无效半边句柄 {:?}", h),
            Self::FlipOnBoundaryEdge(h) => write!(f, "禁止翻转边界边 {:?}", h),
            Self::CollapseOnBoundaryEdge(h) => write!(f, "禁止折叠边界边 {:?}", h),
            Self::NoTwin(h) => write!(f, "半边 {:?} 没有 twin", h),
            Self::NoFace(h) => write!(f, "半边 {:?} 两侧均无面", h),
            Self::DegenerateTriangle => write!(f, "操作会产生退化三角形"),
            Self::LinkConditionViolated { a, b } => {
                write!(f, "链接条件不满足：折叠 {:?}-{:?} 会产生非流形", a, b)
            }
            Self::Inconsistent(msg) => write!(f, "网格拓扑不一致：{}", msg),
        }
    }
}

impl std::error::Error for TopologyError {}

// ============================================================
// 校验
// ============================================================

/// 校验网格是否满足流形三角曲面不变量（**轻量级，首错返回**）。
///
/// 检查项：
/// 1. 每条半边的 `twin` 互指；
/// 2. `twin.vertex` 与自身 `vertex` 不同（无自环）；
/// 3. `next/prev` 互为反问（若 `next = X`，则 `X.prev = self`）；
/// 4. 每个面的边界环长度为 3（三角网格）。
///
/// # 何时使用此函数
///
/// **专用于拓扑操作（split/flip/collapse 等）内部的前后置断言**：
/// - 复用 `TopologyError` 类型，便于在操作函数中 `?` 传播；
/// - 仅校验 4 项核心不变量，速度快；
/// - 遇到首个错误即返回，不收集全部违例。
///
/// # 何时使用其他验证函数
///
/// - 需要**结构化错误类型**（`ValidationError`）或**全部违例**：
///   见 [`crate::validate::validate_topology`] / [`crate::validate::check_topology`]。
/// - 需要**快速失败 + 结构化错误**：见 [`crate::validate::validate_first_error`]。
/// - 三者的对比表与决策树见 `validate` 模块文档。
pub fn validate_mesh(mesh: &MeshStorage) -> Result<(), TopologyError> {
    let all_he: Vec<HalfEdgeId> = mesh.halfedge_ids().collect();

    for he_id in &all_he {
        let he = match mesh.get_halfedge(*he_id) {
            Some(h) => h,
            None => continue,
        };

        // twin 互指
        if let Some(twin_id) = he.twin {
            let twin = match mesh.get_halfedge(twin_id) {
                Some(t) => t,
                None => {
                    return Err(TopologyError::Inconsistent(format!(
                        "半边 {:?} 的 twin {:?} 不存在",
                        he_id, twin_id
                    )));
                }
            };
            if twin.twin != Some(*he_id) {
                return Err(TopologyError::Inconsistent(format!(
                    "twin 不互指：{:?}.twin={:?}, 但 {:?}.twin={:?}",
                    he_id, twin_id, twin_id, twin.twin
                )));
            }
            if twin.vertex == he.vertex {
                return Err(TopologyError::Inconsistent(format!(
                    "半边 {:?} 与其 twin 顶点相同（自环）",
                    he_id
                )));
            }
        }

        // next/prev 一致性
        if let Some(next_id) = he.next {
            match mesh.get_halfedge(next_id) {
                Some(next) if next.prev == Some(*he_id) => {}
                Some(next) => {
                    return Err(TopologyError::Inconsistent(format!(
                        "next/prev 不一致：{:?}.next={:?}, 但 {:?}.prev={:?}",
                        he_id, next_id, next_id, next.prev
                    )));
                }
                None => {
                    return Err(TopologyError::Inconsistent(format!(
                        "半边 {:?} 的 next {:?} 不存在",
                        he_id, next_id
                    )));
                }
            }
        }
        if let Some(prev_id) = he.prev {
            match mesh.get_halfedge(prev_id) {
                Some(prev) if prev.next == Some(*he_id) => {}
                Some(prev) => {
                    return Err(TopologyError::Inconsistent(format!(
                        "prev/next 不一致：{:?}.prev={:?}, 但 {:?}.next={:?}",
                        he_id, prev_id, prev_id, prev.next
                    )));
                }
                None => {
                    return Err(TopologyError::Inconsistent(format!(
                        "半边 {:?} 的 prev {:?} 不存在",
                        he_id, prev_id
                    )));
                }
            }
        }
    }

    // 每个面的边界环长度为 3
    for f_id in mesh.face_ids() {
        let f = match mesh.get_face(f_id) {
            Some(f) => f,
            None => continue,
        };
        if let Some(start) = f.halfedge {
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
                return Err(TopologyError::Inconsistent(format!(
                    "面 {:?} 的边界环长度为 {}，非三角面",
                    f_id, count
                )));
            }
        }
    }

    Ok(())
}

// ============================================================
// split_edge：边分裂
// ============================================================

/// 分裂半边 `he` 所在的边，在两端顶点中点处插入新顶点 `M`，重构相邻面的拓扑。
///
/// ## 拓扑修改逻辑
///
/// 设 `h = he`（若 `h.face = None` 但 `twin.face = Some`，自动改为操作 `twin`），
/// `A = h.twin.vertex`（origin），`B = h.vertex`（tip），`F1 = h.face`。
///
/// `F1 = (A→B→C→A)`，其中 `n1 = h.next (B→C)`，`p1 = h.prev (C→A)`。
///
/// 若 `twin.face = Some(F2)`（内部边），`F2 = (B→A→D→B)`，
/// `n2 = twin.next (A→D)`，`p2 = twin.prev (D→B)`。
///
/// ### 内部边（两侧均有面）
/// 删除 `h, twin`。新建 8 条半边与 2 个面：
/// - `F1` 重用为 `(A→M→C→A)`，新建 `F_new1 = (M→B→C→M)`；
/// - `F2` 重用为 `(B→M→D→B)`，新建 `F_new2 = (M→A→D→M)`；
/// - 复用 `n1, p1, n2, p2`，仅修改其 `next/prev/face`。
///
/// ### 边界边（仅一侧有面）
/// 删除 `h, twin`。新建 6 条半边与 1 个面：
/// - `F1` 重用为 `(A→M→C→A)`，新建 `F_new1 = (M→B→C→M)`；
/// - `twin (B→A, 边界)` 分裂为 `b_to_m (B→M, 边界)` + `m_to_a (M→A, 边界)`。
///
/// # 返回
/// 新顶点 `M` 的 ID。
pub fn split_edge(mesh: &mut MeshStorage, he: HalfEdgeId) -> Result<VertexId, TopologyError> {
    // ---------- 1. 校验与数据收集 ----------

    let h_data = mesh
        .get_halfedge(he)
        .ok_or(TopologyError::InvalidHalfEdge(he))?
        .clone();

    // 若 h 是边界半边，改操作 twin（确保 h.face = Some）
    let (h_id, h) = if h_data.face.is_none() {
        let twin_id = h_data.twin.ok_or(TopologyError::NoTwin(he))?;
        let twin_data = mesh
            .get_halfedge(twin_id)
            .ok_or(TopologyError::InvalidHalfEdge(twin_id))?
            .clone();
        if twin_data.face.is_none() {
            return Err(TopologyError::NoFace(he));
        }
        (twin_id, twin_data)
    } else {
        (he, h_data)
    };

    let twin_id = h.twin.ok_or(TopologyError::NoTwin(h_id))?;
    let twin = mesh
        .get_halfedge(twin_id)
        .ok_or(TopologyError::InvalidHalfEdge(twin_id))?
        .clone();

    let a = twin.vertex; // origin of h
    let b = h.vertex; // tip of h
    let f1 = h.face.ok_or(TopologyError::NoFace(h_id))?;
    let n1 = h
        .next
        .ok_or_else(|| TopologyError::Inconsistent("h.next 为 None".into()))?;
    let p1 = h
        .prev
        .ok_or_else(|| TopologyError::Inconsistent("h.prev 为 None".into()))?;
    let c = mesh
        .get_halfedge(n1)
        .ok_or(TopologyError::InvalidHalfEdge(n1))?
        .vertex;

    let has_f2 = twin.face.is_some();
    let (f2, n2, p2, d) = if has_f2 {
        let f2 = twin.face.unwrap();
        let n2 = twin
            .next
            .ok_or_else(|| TopologyError::Inconsistent("twin.next 为 None".into()))?;
        let p2 = twin
            .prev
            .ok_or_else(|| TopologyError::Inconsistent("twin.prev 为 None".into()))?;
        let d = mesh
            .get_halfedge(n2)
            .ok_or(TopologyError::InvalidHalfEdge(n2))?
            .vertex;
        (Some(f2), Some(n2), Some(p2), Some(d))
    } else {
        (None, None, None, None)
    };

    // ---------- 2. 创建中点 M ----------

    let pos_a = mesh
        .get_vertex(a)
        .ok_or_else(|| TopologyError::Inconsistent("顶点 A 不存在".into()))?
        .position;
    let pos_b = mesh
        .get_vertex(b)
        .ok_or_else(|| TopologyError::Inconsistent("顶点 B 不存在".into()))?
        .position;
    let mid = [
        (pos_a[0] + pos_b[0]) * 0.5,
        (pos_a[1] + pos_b[1]) * 0.5,
        (pos_a[2] + pos_b[2]) * 0.5,
    ];
    let m = mesh.add_vertex(Vertex::new(mid));

    // ---------- 3. 创建新半边 ----------

    let a_to_m = mesh.add_halfedge(HalfEdge::new(m)); // A→M
    let m_to_a = mesh.add_halfedge(HalfEdge::new(a)); // M→A
    let m_to_b = mesh.add_halfedge(HalfEdge::new(b)); // M→B
    let b_to_m = mesh.add_halfedge(HalfEdge::new(m)); // B→M
    let m_to_c = mesh.add_halfedge(HalfEdge::new(c)); // M→C
    let c_to_m = mesh.add_halfedge(HalfEdge::new(m)); // C→M

    let (m_to_d, d_to_m) = if has_f2 {
        let d = d.unwrap();
        (
            Some(mesh.add_halfedge(HalfEdge::new(d))), // M→D
            Some(mesh.add_halfedge(HalfEdge::new(m))), // D→M
        )
    } else {
        (None, None)
    };

    // ---------- 4. 重连 F1 = (A→M→C→A) ----------

    {
        let am = mesh.get_halfedge_mut(a_to_m).unwrap();
        am.twin = Some(m_to_a);
        am.next = Some(m_to_c);
        am.prev = Some(p1);
        am.face = Some(f1);

        let mc = mesh.get_halfedge_mut(m_to_c).unwrap();
        mc.twin = Some(c_to_m);
        mc.next = Some(p1);
        mc.prev = Some(a_to_m);
        mc.face = Some(f1);

        // p1 (C→A): 原 next=h, prev=n1 → 现 next=a_to_m, prev=m_to_c
        let p1h = mesh.get_halfedge_mut(p1).unwrap();
        p1h.next = Some(a_to_m);
        p1h.prev = Some(m_to_c);
    }

    // ---------- 5. 新建 F_new1 = (M→B→C→M) ----------

    let f_new1 = mesh.add_face(Face::new());
    {
        let mb = mesh.get_halfedge_mut(m_to_b).unwrap();
        mb.twin = Some(b_to_m);
        mb.next = Some(n1);
        mb.prev = Some(c_to_m);
        mb.face = Some(f_new1);

        // n1 (B→C): 原 next=p1, prev=h → 现 next=c_to_m, prev=m_to_b, face=F_new1
        let n1h = mesh.get_halfedge_mut(n1).unwrap();
        n1h.next = Some(c_to_m);
        n1h.prev = Some(m_to_b);
        n1h.face = Some(f_new1);

        let cm = mesh.get_halfedge_mut(c_to_m).unwrap();
        cm.twin = Some(m_to_c);
        cm.next = Some(m_to_b);
        cm.prev = Some(n1);
        cm.face = Some(f_new1);
    }
    mesh.get_face_mut(f_new1).unwrap().halfedge = Some(m_to_b);

    // ---------- 6. 处理 twin 侧 ----------

    if has_f2 {
        let f2 = f2.unwrap();
        let n2 = n2.unwrap();
        let p2 = p2.unwrap();
        let m_to_d = m_to_d.unwrap();
        let d_to_m = d_to_m.unwrap();

        // F2 重用为 (B→M→D→B)
        {
            let bm = mesh.get_halfedge_mut(b_to_m).unwrap();
            bm.twin = Some(m_to_b);
            bm.next = Some(m_to_d);
            bm.prev = Some(p2);
            bm.face = Some(f2);

            let md = mesh.get_halfedge_mut(m_to_d).unwrap();
            md.twin = Some(d_to_m);
            md.next = Some(p2);
            md.prev = Some(b_to_m);
            md.face = Some(f2);

            // p2 (D→B): 原 next=twin, prev=n2 → 现 next=b_to_m, prev=m_to_d
            let p2h = mesh.get_halfedge_mut(p2).unwrap();
            p2h.next = Some(b_to_m);
            p2h.prev = Some(m_to_d);
        }

        // 新建 F_new2 = (M→A→D→M)
        let f_new2 = mesh.add_face(Face::new());
        {
            let ma = mesh.get_halfedge_mut(m_to_a).unwrap();
            ma.twin = Some(a_to_m);
            ma.next = Some(n2);
            ma.prev = Some(d_to_m);
            ma.face = Some(f_new2);

            // n2 (A→D): 原 next=p2, prev=twin → 现 next=d_to_m, prev=m_to_a, face=F_new2
            let n2h = mesh.get_halfedge_mut(n2).unwrap();
            n2h.next = Some(d_to_m);
            n2h.prev = Some(m_to_a);
            n2h.face = Some(f_new2);

            let dm = mesh.get_halfedge_mut(d_to_m).unwrap();
            dm.twin = Some(m_to_d);
            dm.next = Some(m_to_a);
            dm.prev = Some(n2);
            dm.face = Some(f_new2);
        }
        mesh.get_face_mut(f_new2).unwrap().halfedge = Some(m_to_a);

        mesh.get_face_mut(f2).unwrap().halfedge = Some(b_to_m);
    } else {
        // 边界情形：twin (B→A) 分裂为 b_to_m (B→M) + m_to_a (M→A)，均 face=None
        let bm = mesh.get_halfedge_mut(b_to_m).unwrap();
        bm.twin = Some(m_to_b);
        bm.face = None;

        let ma = mesh.get_halfedge_mut(m_to_a).unwrap();
        ma.twin = Some(a_to_m);
        ma.face = None;
    }

    // ---------- 7. 面入口更新 ----------

    mesh.get_face_mut(f1).unwrap().halfedge = Some(a_to_m);

    // ---------- 8. 顶点 outgoing 更新 ----------

    if mesh.get_vertex(a).unwrap().halfedge == Some(h_id) {
        mesh.get_vertex_mut(a).unwrap().halfedge = Some(a_to_m);
    }
    if mesh.get_vertex(b).unwrap().halfedge == Some(twin_id) {
        mesh.get_vertex_mut(b).unwrap().halfedge = Some(b_to_m);
    }
    // M 的 outgoing 入口必须是 origin=M 的半边（即 twin.vertex=M）。
    // a_to_m 是 A→M（vertex=M, origin=A），是 A 的 outgoing；
    // m_to_a 是 M→A（vertex=A, origin=M），是 M 的 outgoing。
    mesh.get_vertex_mut(m).unwrap().halfedge = Some(m_to_a);

    // ---------- 9. 删除原 h, twin ----------

    mesh.remove_halfedge(h_id);
    mesh.remove_halfedge(twin_id);

    Ok(m)
}

// ============================================================
// flip_edge：内部边翻转
// ============================================================

/// 翻转内部边 `he` 所在的对角线，将四边形 `A-B-C-D` 的对角线从 `A-B` 替换为 `C-D`。
///
/// ## 拓扑修改逻辑
///
/// 设 `h = he`，`A = h.twin.vertex`（origin），`B = h.vertex`（tip）。
/// `F1 = h.face = (A→B→C→A)`，`F2 = twin.face = (B→A→D→B)`。
/// `n1 = h.next (B→C)`，`p1 = h.prev (C→A)`，
/// `n2 = twin.next (A→D)`，`p2 = twin.prev (D→B)`。
///
/// 翻转后：
/// - `h` 从 `A→B` 变为 `D→C`（`vertex = C`）；
/// - `twin` 从 `B→A` 变为 `C→D`（`vertex = D`）；
/// - `F1` 重用为 `(D→C→A→D) = h → p1 → n2`；
/// - `F2` 重用为 `(C→D→B→C) = twin → p2 → n1`。
///
/// # 校验
/// - `h.face` 与 `twin.face` 均须为 `Some`（内部边，禁止翻转边界边）；
/// - `C != D`（防退化）。
pub fn flip_edge(mesh: &mut MeshStorage, he: HalfEdgeId) -> Result<(), TopologyError> {
    // ---------- 1. 校验与数据收集 ----------

    let h = mesh
        .get_halfedge(he)
        .ok_or(TopologyError::InvalidHalfEdge(he))?
        .clone();
    let twin_id = h.twin.ok_or(TopologyError::NoTwin(he))?;
    let twin = mesh
        .get_halfedge(twin_id)
        .ok_or(TopologyError::InvalidHalfEdge(twin_id))?
        .clone();

    let f1 = h.face.ok_or(TopologyError::FlipOnBoundaryEdge(he))?;
    let f2 = twin.face.ok_or(TopologyError::FlipOnBoundaryEdge(he))?;

    let a = twin.vertex; // origin of h（翻转后 A 不再是 h 的端点）
    let b = h.vertex; // tip of h（翻转后 B 不再是 h 的端点）

    let n1 = h
        .next
        .ok_or_else(|| TopologyError::Inconsistent("h.next 为 None".into()))?;
    let p1 = h
        .prev
        .ok_or_else(|| TopologyError::Inconsistent("h.prev 为 None".into()))?;
    let n2 = twin
        .next
        .ok_or_else(|| TopologyError::Inconsistent("twin.next 为 None".into()))?;
    let p2 = twin
        .prev
        .ok_or_else(|| TopologyError::Inconsistent("twin.prev 为 None".into()))?;

    let c = mesh
        .get_halfedge(n1)
        .ok_or(TopologyError::InvalidHalfEdge(n1))?
        .vertex;
    let d = mesh
        .get_halfedge(n2)
        .ok_or(TopologyError::InvalidHalfEdge(n2))?
        .vertex;

    if c == d {
        return Err(TopologyError::DegenerateTriangle);
    }

    // ---------- 2. 翻转 h / twin ----------

    // h: A→B → D→C (vertex=C)
    // twin: B→A → C→D (vertex=D)
    {
        let h_mut = mesh.get_halfedge_mut(he).unwrap();
        h_mut.vertex = c;
        h_mut.next = Some(p1);
        h_mut.prev = Some(n2);
        h_mut.face = Some(f1);

        let twin_mut = mesh.get_halfedge_mut(twin_id).unwrap();
        twin_mut.vertex = d;
        twin_mut.next = Some(p2);
        twin_mut.prev = Some(n1);
        twin_mut.face = Some(f2);
    }

    // ---------- 3. 重连四条邻接半边 ----------

    // F1 = (D→C→A→D) = h → p1 → n2 → h
    // p1 (C→A): 原 next=h, prev=n1 → 现 next=n2, prev=h
    // n2 (A→D): 原 next=p2, prev=twin → 现 next=h, prev=p1, face=F1
    {
        let p1_mut = mesh.get_halfedge_mut(p1).unwrap();
        p1_mut.next = Some(n2);
        p1_mut.prev = Some(he);

        let n2_mut = mesh.get_halfedge_mut(n2).unwrap();
        n2_mut.next = Some(he);
        n2_mut.prev = Some(p1);
        n2_mut.face = Some(f1);
    }

    // F2 = (C→D→B→C) = twin → p2 → n1 → twin
    // p2 (D→B): 原 next=twin, prev=n2 → 现 next=n1, prev=twin
    // n1 (B→C): 原 next=p1, prev=h → 现 next=twin, prev=p2, face=F2
    {
        let p2_mut = mesh.get_halfedge_mut(p2).unwrap();
        p2_mut.next = Some(n1);
        p2_mut.prev = Some(twin_id);

        let n1_mut = mesh.get_halfedge_mut(n1).unwrap();
        n1_mut.next = Some(twin_id);
        n1_mut.prev = Some(p2);
        n1_mut.face = Some(f2);
    }

    // ---------- 4. 面入口 ----------

    mesh.get_face_mut(f1).unwrap().halfedge = Some(he);
    mesh.get_face_mut(f2).unwrap().halfedge = Some(twin_id);

    // ---------- 5. 顶点 outgoing 更新 ----------

    // A 原本 outgoing 可能是 he（A→B），翻转后 he 起点变为 D，需更新为 n2（A→D）
    if mesh.get_vertex(a).unwrap().halfedge == Some(he) {
        mesh.get_vertex_mut(a).unwrap().halfedge = Some(n2);
    }
    // B 原本 outgoing 可能是 twin（B→A），翻转后 twin 起点变为 C，需更新为 n1（B→C）
    if mesh.get_vertex(b).unwrap().halfedge == Some(twin_id) {
        mesh.get_vertex_mut(b).unwrap().halfedge = Some(n1);
    }
    // C、D 的 outgoing 不需要改（它们仍指向 n1/p1/n2/p2，这些半边存活）

    Ok(())
}

// ============================================================
// collapse_edge：边折叠
// ============================================================

/// 折叠半边 `he` 两端顶点为一个新顶点 `K`，删除两个相邻三角形。
///
/// 新顶点 `K` 的位置为 `A`、`B` 中点。若需指定最优位置（如 QEM 减面），
/// 请使用 [`collapse_edge_at`]。
///
/// ## 拓扑修改逻辑
///
/// 设 `h = he`，`A = h.twin.vertex`（origin），`B = h.vertex`（tip）。
/// `F1 = h.face = (A→B→C→A)`，`F2 = twin.face = (B→A→D→B)`。
/// `n1 = h.next (B→C)`，`p1 = h.prev (C→A)`，
/// `n2 = twin.next (A→D)`，`p2 = twin.prev (D→B)`。
///
/// # 校验
/// - `h.face` 与 `twin.face` 均须为 `Some`（禁止折叠边界边）；
/// - `C != D`（防退化三角形）；
/// - **链接条件**：`A` 与 `B` 的公共邻居集合恰好为 `{C, D}`，否则折叠会产生非流形。
///
/// # 操作步骤
/// 1. 创建新顶点 `K`（位置为 `A`、`B` 中点，或由 `target_pos` 指定）；
/// 2. 缝合孔洞：`p1.twin ↔ n1.twin`，`n2.twin ↔ p2.twin`；
/// 3. 将所有 `vertex = A` 或 `vertex = B` 的存活半边的 `vertex` 更新为 `K`；
/// 4. 更新 `K`、`C`、`D` 的 `outgoing` 入口；
/// 5. 删除 `h, twin, n1, p1, n2, p2, F1, F2, A, B`。
///
/// # 返回
/// 新顶点 `K` 的 ID。
pub fn collapse_edge(mesh: &mut MeshStorage, he: HalfEdgeId) -> Result<VertexId, TopologyError> {
    collapse_edge_impl(mesh, he, None)
}

/// 折叠半边 `he`，新顶点位置为 `target_pos`（而非中点）。
///
/// 用于 QEM 减面等需要最优折叠位置的场景。拓扑修改逻辑与
/// [`collapse_edge`] 完全相同，仅新顶点 `K` 的位置不同。
///
/// # 返回
/// 新顶点 `K` 的 ID。
pub fn collapse_edge_at(
    mesh: &mut MeshStorage,
    he: HalfEdgeId,
    target_pos: [f64; 3],
) -> Result<VertexId, TopologyError> {
    collapse_edge_impl(mesh, he, Some(target_pos))
}

/// 边折叠核心实现。
///
/// `target_pos = None` 时新顶点取 `A`、`B` 中点；
/// `target_pos = Some(p)` 时新顶点取 `p`（用于 QEM 减面等）。
fn collapse_edge_impl(
    mesh: &mut MeshStorage,
    he: HalfEdgeId,
    target_pos: Option<[f64; 3]>,
) -> Result<VertexId, TopologyError> {
    // ---------- 1. 校验与数据收集 ----------

    let h = mesh
        .get_halfedge(he)
        .ok_or(TopologyError::InvalidHalfEdge(he))?
        .clone();
    let twin_id = h.twin.ok_or(TopologyError::NoTwin(he))?;
    let twin = mesh
        .get_halfedge(twin_id)
        .ok_or(TopologyError::InvalidHalfEdge(twin_id))?
        .clone();

    let f1 = h.face.ok_or(TopologyError::CollapseOnBoundaryEdge(he))?;
    let f2 = twin.face.ok_or(TopologyError::CollapseOnBoundaryEdge(he))?;

    let a = twin.vertex;
    let b = h.vertex;

    let n1 = h
        .next
        .ok_or_else(|| TopologyError::Inconsistent("h.next 为 None".into()))?;
    let p1 = h
        .prev
        .ok_or_else(|| TopologyError::Inconsistent("h.prev 为 None".into()))?;
    let n2 = twin
        .next
        .ok_or_else(|| TopologyError::Inconsistent("twin.next 为 None".into()))?;
    let p2 = twin
        .prev
        .ok_or_else(|| TopologyError::Inconsistent("twin.prev 为 None".into()))?;

    let c = mesh
        .get_halfedge(n1)
        .ok_or(TopologyError::InvalidHalfEdge(n1))?
        .vertex;
    let d = mesh
        .get_halfedge(n2)
        .ok_or(TopologyError::InvalidHalfEdge(n2))?
        .vertex;

    if c == d {
        return Err(TopologyError::DegenerateTriangle);
    }

    // 链接条件：A、B 的公共邻居恰好为 {C, D}
    if !check_link_condition(mesh, a, b, c, d) {
        return Err(TopologyError::LinkConditionViolated { a, b });
    }

    // ---------- 2. 收集需要更新 vertex 的半边 ----------

    let deleted = [he, twin_id, n1, p1, n2, p2];
    let deleted_set: HashSet<HalfEdgeId> = deleted.iter().copied().collect();

    // 收集 A、B 的所有 incoming 半边（即 outgoing 的 twin），排除被删除的
    let mut to_update: Vec<HalfEdgeId> = Vec::new();
    for out_he in VertexRing::new(mesh, a).collect::<Vec<_>>() {
        if let Some(t_id) = mesh.get_halfedge(out_he).and_then(|h| h.twin)
            && !deleted_set.contains(&t_id)
        {
            to_update.push(t_id);
        }
    }
    for out_he in VertexRing::new(mesh, b).collect::<Vec<_>>() {
        if let Some(t_id) = mesh.get_halfedge(out_he).and_then(|h| h.twin)
            && !deleted_set.contains(&t_id)
        {
            to_update.push(t_id);
        }
    }

    // 获取 twin 的 twin ID（缝合用），注意这些 twin 不在 deleted_set 中
    let p1_twin = mesh
        .get_halfedge(p1)
        .ok_or(TopologyError::InvalidHalfEdge(p1))?
        .twin
        .ok_or_else(|| TopologyError::Inconsistent("p1.twin 为 None".into()))?;
    let n1_twin = mesh
        .get_halfedge(n1)
        .ok_or(TopologyError::InvalidHalfEdge(n1))?
        .twin
        .ok_or_else(|| TopologyError::Inconsistent("n1.twin 为 None".into()))?;
    let n2_twin = mesh
        .get_halfedge(n2)
        .ok_or(TopologyError::InvalidHalfEdge(n2))?
        .twin
        .ok_or_else(|| TopologyError::Inconsistent("n2.twin 为 None".into()))?;
    let p2_twin = mesh
        .get_halfedge(p2)
        .ok_or(TopologyError::InvalidHalfEdge(p2))?
        .twin
        .ok_or_else(|| TopologyError::Inconsistent("p2.twin 为 None".into()))?;

    // ---------- 3. 创建新顶点 K ----------

    let new_pos = match target_pos {
        Some(p) => p,
        None => {
            let pos_a = mesh
                .get_vertex(a)
                .ok_or_else(|| TopologyError::Inconsistent("顶点 A 不存在".into()))?
                .position;
            let pos_b = mesh
                .get_vertex(b)
                .ok_or_else(|| TopologyError::Inconsistent("顶点 B 不存在".into()))?
                .position;
            [
                (pos_a[0] + pos_b[0]) * 0.5,
                (pos_a[1] + pos_b[1]) * 0.5,
                (pos_a[2] + pos_b[2]) * 0.5,
            ]
        }
    };
    let k = mesh.add_vertex(Vertex::new(new_pos));

    // ---------- 4. 缝合 twin ----------

    // p1.twin (A→C) 与 n1.twin (C→B) 互为 twin（原 twin p1/n1 被删除）
    {
        let p1t = mesh.get_halfedge_mut(p1_twin).unwrap();
        p1t.twin = Some(n1_twin);
        let n1t = mesh.get_halfedge_mut(n1_twin).unwrap();
        n1t.twin = Some(p1_twin);
    }
    // n2.twin (D→A) 与 p2.twin (B→D) 互为 twin（原 twin n2/p2 被删除）
    {
        let n2t = mesh.get_halfedge_mut(n2_twin).unwrap();
        n2t.twin = Some(p2_twin);
        let p2t = mesh.get_halfedge_mut(p2_twin).unwrap();
        p2t.twin = Some(n2_twin);
    }

    // ---------- 5. 批量更新 vertex = A/B → K ----------

    for he_id in &to_update {
        let h_mut = mesh.get_halfedge_mut(*he_id).unwrap();
        if h_mut.vertex == a || h_mut.vertex == b {
            h_mut.vertex = k;
        }
    }

    // ---------- 6. 更新顶点 outgoing ----------

    // K 的 outgoing：选一条存活的 outgoing 半边
    // 优先选 p1_twin（原 A→C，现 K→C），它一定存活
    mesh.get_vertex_mut(k).unwrap().halfedge = Some(p1_twin);

    // C 的 outgoing 若指向被删除的半边，更新为 n1_twin（C→K）
    let c_out = mesh.get_vertex(c).unwrap().halfedge;
    if c_out == Some(n1) || c_out == Some(p1) || c_out.is_none() {
        mesh.get_vertex_mut(c).unwrap().halfedge = Some(n1_twin);
    }
    // D 的 outgoing 若指向被删除的半边，更新为 n2_twin（D→K）
    let d_out = mesh.get_vertex(d).unwrap().halfedge;
    if d_out == Some(n2) || d_out == Some(p2) || d_out.is_none() {
        mesh.get_vertex_mut(d).unwrap().halfedge = Some(n2_twin);
    }

    // ---------- 7. 删除废弃元素 ----------

    mesh.remove_halfedge(he);
    mesh.remove_halfedge(twin_id);
    mesh.remove_halfedge(n1);
    mesh.remove_halfedge(p1);
    mesh.remove_halfedge(n2);
    mesh.remove_halfedge(p2);
    mesh.remove_face(f1);
    mesh.remove_face(f2);
    mesh.remove_vertex(a);
    mesh.remove_vertex(b);

    Ok(k)
}

/// 检查链接条件：`A` 与 `B` 的公共邻居恰好为 `{C, D}`。
fn check_link_condition(
    mesh: &MeshStorage,
    a: VertexId,
    b: VertexId,
    c: VertexId,
    d: VertexId,
) -> bool {
    let neighbors_a: HashSet<VertexId> = VertexAdjacentVerts::new(mesh, a).collect();
    let neighbors_b: HashSet<VertexId> = VertexAdjacentVerts::new(mesh, b).collect();
    let common: HashSet<&VertexId> = neighbors_a.intersection(&neighbors_b).collect();
    common.len() == 2 && common.contains(&c) && common.contains(&d)
}

// ============================================================
// extrude_face：面挤出
// ============================================================

/// 将三角面 `face` 沿 `offset` 方向挤出，形成棱柱体。
///
/// ## 拓扑修改逻辑
///
/// 设 `F = (v0, v1, v2)` CCW，半边 `h0(v0→v1), h1(v1→v2), h2(v2→v0)`，
/// 对应边界 twin `t0(v1→v0), t1(v2→v1), t2(v0→v2)`（均 `face = None`）。
///
/// ### 限制
/// F 的三条边必须均为边界边（twin 的 `face = None`），否则返回
/// [`TopologyError::Inconsistent`]。
///
/// ### 创建元素
/// - 3 个新顶点 `v0', v1', v2'` = 原位置 + offset
/// - 7 个新面：1 顶面 `F'` + 6 侧面三角形（每条侧边 2 个）
/// - 18 条新半边（顶面 3 + 侧面 15），重用 3 条原边界 twin
///
/// ### 顶面 F'
/// `F' = (v2', v1', v0')` CCW（反向），半边
/// `tp_0(v2'→v1'), tp_1(v1'→v0'), tp_2(v0'→v2')`。
///
/// ### 侧面三角形（边 i: `hi(vi→vi+1)`）
/// - `T1_i = (vi+1, vi, vi')`：重用 `ti(vi+1→vi)` + 新建 `a_i(vi→vi')` + 新建 `b_i(vi'→vi+1)`
/// - `T2_i = (vi+1, vi', vi+1')`：新建 `c_i(vi+1→vi')` + 新建 `d_i(vi'→vi+1')` + 新建 `e_i(vi+1'→vi+1)`
///
/// ### twin 对应（12 对）
/// - `ti ↔ hi`（已存在，保留）
/// - `a_i ↔ e_{i-1}`（垂直边，相邻侧面共享）
/// - `b_i ↔ c_i`（侧面内对角线）
/// - `d_i ↔ tp_{map(i)}`（顶面与侧面共享边）：
///   `d_0↔tp_1`, `d_1↔tp_0`, `d_2↔tp_2`
///
/// # 返回
/// 新创建的面 ID 列表（顺序：`[F', T1_0, T2_0, T1_1, T2_1, T1_2, T2_2]`）。
///
/// # 退化检查
/// - offset 为零向量（长度 < 1e-12）返回 [`TopologyError::DegenerateTriangle`]
/// - 任一侧面三角形面积 < 1e-12（offset 与边平行）返回 [`TopologyError::DegenerateTriangle`]
///
/// # 校验
/// 操作完成后调用 [`crate::validate::validate_topology`] 完整校验网格。
pub fn extrude_face(
    mesh: &mut MeshStorage,
    face: FaceId,
    offset: [f64; 3],
) -> Result<Vec<FaceId>, TopologyError> {
    // ---------- 1. 退化检查：offset 非零 ----------
    let offset_len = (offset[0].powi(2) + offset[1].powi(2) + offset[2].powi(2)).sqrt();
    if offset_len < 1e-12 {
        return Err(TopologyError::DegenerateTriangle);
    }

    // ---------- 2. 收集面 F 的三条半边与顶点 ----------
    if !mesh.contains_face(face) {
        return Err(TopologyError::Inconsistent(format!("面 {:?} 不存在", face)));
    }
    let hs: Vec<HalfEdgeId> = FaceHalfEdges::new(mesh, face).collect();
    if hs.len() != 3 {
        return Err(TopologyError::Inconsistent(format!(
            "面 {:?} 边界环长度={}, 非三角面",
            face,
            hs.len()
        )));
    }
    let h_arr: [HalfEdgeId; 3] = [hs[0], hs[1], hs[2]];

    let h0d = mesh
        .get_halfedge(h_arr[0])
        .ok_or(TopologyError::InvalidHalfEdge(h_arr[0]))?
        .clone();
    let h1d = mesh
        .get_halfedge(h_arr[1])
        .ok_or(TopologyError::InvalidHalfEdge(h_arr[1]))?
        .clone();
    let h2d = mesh
        .get_halfedge(h_arr[2])
        .ok_or(TopologyError::InvalidHalfEdge(h_arr[2]))?
        .clone();

    // h0: v0→v1 (vertex=v1), h1: v1→v2 (vertex=v2), h2: v2→v0 (vertex=v0)
    let v: [VertexId; 3] = [h2d.vertex, h0d.vertex, h1d.vertex];

    let pos: [[f64; 3]; 3] = {
        let mut arr = [[0.0f64; 3]; 3];
        for i in 0..3 {
            arr[i] = mesh
                .get_vertex(v[i])
                .ok_or_else(|| TopologyError::Inconsistent(format!("顶点 {:?} 不存在", v[i])))?
                .position;
        }
        arr
    };

    // ---------- 3. 校验所有边为边界边 ----------
    let t: [HalfEdgeId; 3] = [
        h0d.twin.ok_or(TopologyError::NoTwin(h_arr[0]))?,
        h1d.twin.ok_or(TopologyError::NoTwin(h_arr[1]))?,
        h2d.twin.ok_or(TopologyError::NoTwin(h_arr[2]))?,
    ];

    for &item in &t {
        let td = mesh
            .get_halfedge(item)
            .ok_or(TopologyError::InvalidHalfEdge(item))?;
        if td.face.is_some() {
            return Err(TopologyError::Inconsistent(
                "extrude_face 仅支持所有边为边界边的面".into(),
            ));
        }
    }

    // ---------- 4. 退化侧面检查 ----------
    // 侧面三角形 T1_i = (v_{i+1}, v_i, v_i')，退化 ⟺ 三顶点共线
    // ⟺ (v_i - v_{i+1}) 与 offset 共线 ⟺ |edge_i × offset| = 0
    // 用 Shewchuk 鲁棒谓词精确判定（无浮点阈值）
    for i in 0..3 {
        let a = pos[(i + 1) % 3];
        let b = pos[i];
        let c = [
            pos[i][0] + offset[0],
            pos[i][1] + offset[1],
            pos[i][2] + offset[2],
        ];
        if crate::predicates::is_triangle_degenerate_3d(a, b, c) {
            return Err(TopologyError::DegenerateTriangle);
        }
    }

    // ---------- 5. 创建新顶点 ----------
    let vp: [VertexId; 3] = [
        mesh.add_vertex(Vertex::new([
            pos[0][0] + offset[0],
            pos[0][1] + offset[1],
            pos[0][2] + offset[2],
        ])),
        mesh.add_vertex(Vertex::new([
            pos[1][0] + offset[0],
            pos[1][1] + offset[1],
            pos[1][2] + offset[2],
        ])),
        mesh.add_vertex(Vertex::new([
            pos[2][0] + offset[0],
            pos[2][1] + offset[1],
            pos[2][2] + offset[2],
        ])),
    ];

    // ---------- 6. 创建新半边 ----------
    // 顶面 3 条: tp[0]=v2'→v1', tp[1]=v1'→v0', tp[2]=v0'→v2'
    let tp: [HalfEdgeId; 3] = [
        mesh.add_halfedge(HalfEdge::new(vp[1])), // v2'→v1'
        mesh.add_halfedge(HalfEdge::new(vp[0])), // v1'→v0'
        mesh.add_halfedge(HalfEdge::new(vp[2])), // v0'→v2'
    ];

    // 侧面 15 条 (每条侧边 5 条)
    let a: [HalfEdgeId; 3] = [
        mesh.add_halfedge(HalfEdge::new(vp[0])), // a[0]: v0→v0'
        mesh.add_halfedge(HalfEdge::new(vp[1])), // a[1]: v1→v1'
        mesh.add_halfedge(HalfEdge::new(vp[2])), // a[2]: v2→v2'
    ];
    let b: [HalfEdgeId; 3] = [
        mesh.add_halfedge(HalfEdge::new(v[1])), // b[0]: v0'→v1
        mesh.add_halfedge(HalfEdge::new(v[2])), // b[1]: v1'→v2
        mesh.add_halfedge(HalfEdge::new(v[0])), // b[2]: v2'→v0
    ];
    let c: [HalfEdgeId; 3] = [
        mesh.add_halfedge(HalfEdge::new(vp[0])), // c[0]: v1→v0'
        mesh.add_halfedge(HalfEdge::new(vp[1])), // c[1]: v2→v1'
        mesh.add_halfedge(HalfEdge::new(vp[2])), // c[2]: v0→v2'
    ];
    let d: [HalfEdgeId; 3] = [
        mesh.add_halfedge(HalfEdge::new(vp[1])), // d[0]: v0'→v1'
        mesh.add_halfedge(HalfEdge::new(vp[2])), // d[1]: v1'→v2'
        mesh.add_halfedge(HalfEdge::new(vp[0])), // d[2]: v2'→v0'
    ];
    let e: [HalfEdgeId; 3] = [
        mesh.add_halfedge(HalfEdge::new(v[1])), // e[0]: v1'→v1
        mesh.add_halfedge(HalfEdge::new(v[2])), // e[1]: v2'→v2
        mesh.add_halfedge(HalfEdge::new(v[0])), // e[2]: v0'→v0
    ];

    // ---------- 7. 创建新面 ----------
    let f_top = mesh.add_face(Face::new());
    let f_t1: [FaceId; 3] = [
        mesh.add_face(Face::new()),
        mesh.add_face(Face::new()),
        mesh.add_face(Face::new()),
    ];
    let f_t2: [FaceId; 3] = [
        mesh.add_face(Face::new()),
        mesh.add_face(Face::new()),
        mesh.add_face(Face::new()),
    ];

    // ---------- 8. twin 映射表 ----------
    // tp[i] ↔ tp_twin[i]: tp[0]↔d[1], tp[1]↔d[0], tp[2]↔d[2]
    // d[i]  ↔ d_twin[i]:  d[0]↔tp[1], d[1]↔tp[0], d[2]↔tp[2]
    let tp_twin: [HalfEdgeId; 3] = [d[1], d[0], d[2]];
    let d_twin: [HalfEdgeId; 3] = [tp[1], tp[0], tp[2]];

    // ---------- 9. 顶面 F' 半边设置 ----------
    // tp[0] → tp[1] → tp[2] → tp[0]
    for i in 0..3 {
        let he = mesh.get_halfedge_mut(tp[i]).unwrap();
        he.twin = Some(tp_twin[i]);
        he.next = Some(tp[(i + 1) % 3]);
        he.prev = Some(tp[(i + 2) % 3]);
        he.face = Some(f_top);
    }
    mesh.get_face_mut(f_top).unwrap().halfedge = Some(tp[0]);

    // ---------- 10. 侧面 T1_i, T2_i 设置 ----------
    for i in 0..3 {
        // T1_i = (vi+1, vi, vi'): t[i] → a[i] → b[i] → t[i]
        {
            let he = mesh.get_halfedge_mut(t[i]).unwrap();
            he.twin = Some(h_arr[i]); // 保持原 twin
            he.next = Some(a[i]);
            he.prev = Some(b[i]);
            he.face = Some(f_t1[i]);

            let he = mesh.get_halfedge_mut(a[i]).unwrap();
            he.twin = Some(e[(i + 2) % 3]); // a_i ↔ e_{i-1}
            he.next = Some(b[i]);
            he.prev = Some(t[i]);
            he.face = Some(f_t1[i]);

            let he = mesh.get_halfedge_mut(b[i]).unwrap();
            he.twin = Some(c[i]); // b_i ↔ c_i
            he.next = Some(t[i]);
            he.prev = Some(a[i]);
            he.face = Some(f_t1[i]);
        }
        mesh.get_face_mut(f_t1[i]).unwrap().halfedge = Some(t[i]);

        // T2_i = (vi+1, vi', vi+1'): c[i] → d[i] → e[i] → c[i]
        {
            let he = mesh.get_halfedge_mut(c[i]).unwrap();
            he.twin = Some(b[i]); // c_i ↔ b_i
            he.next = Some(d[i]);
            he.prev = Some(e[i]);
            he.face = Some(f_t2[i]);

            let he = mesh.get_halfedge_mut(d[i]).unwrap();
            he.twin = Some(d_twin[i]); // d_i ↔ tp_?
            he.next = Some(e[i]);
            he.prev = Some(c[i]);
            he.face = Some(f_t2[i]);

            let he = mesh.get_halfedge_mut(e[i]).unwrap();
            he.twin = Some(a[(i + 1) % 3]); // e_i ↔ a_{i+1}
            he.next = Some(c[i]);
            he.prev = Some(d[i]);
            he.face = Some(f_t2[i]);
        }
        mesh.get_face_mut(f_t2[i]).unwrap().halfedge = Some(c[i]);
    }

    // ---------- 11. 更新新顶点 outgoing 入口 ----------
    // v_i'.halfedge = d[i]（outgoing，origin = v_i'）
    for i in 0..3 {
        mesh.get_vertex_mut(vp[i]).unwrap().halfedge = Some(d[i]);
    }

    // ---------- 12. validate_topology 校验 ----------
    let errors = crate::validate::validate_topology(mesh);
    if !errors.is_empty() {
        return Err(TopologyError::Inconsistent(format!(
            "extrude_face 后校验失败：{:?}",
            errors
        )));
    }

    // ---------- 13. 返回新面列表 ----------
    Ok(vec![
        f_top, f_t1[0], f_t2[0], f_t1[1], f_t2[1], f_t1[2], f_t2[2],
    ])
}

/// 批量挤出多个面。
///
/// 逐个调用 [`extrude_face`]。如果两个面共享边，先挤出的面会使共享边变为内部边，
/// 后续挤出会失败并返回 [`TopologyError::Inconsistent`]（此时网格已部分修改）。
///
/// # 返回
/// 所有新创建的面 ID 列表（每个面 7 个，顺序同 [`extrude_face`]）。
pub fn extrude_faces(
    mesh: &mut MeshStorage,
    faces: &[FaceId],
    offset: [f64; 3],
) -> Result<Vec<FaceId>, TopologyError> {
    let mut all_new = Vec::new();
    for &f in faces {
        let new = extrude_face(mesh, f, offset)?;
        all_new.extend(new);
    }
    Ok(all_new)
}

// ============================================================
// extrude_region：区域挤出
// ============================================================

/// 区域挤出：一次性挤出多个相邻面，自动处理内部边。
///
/// 模仿 Blender/Maya 中 "Extrude Region" 的行为：选中区域内所有面共同移动
/// `offset`，区域内部边不产生侧面，仅在区域边界生成侧面四边形（三角化）。
///
/// ## 算法概览
///
/// 1. **顶点分类**：对每个区域顶点 `v`，若所有邻接面均在区域内 → 内部顶点
///    （原地平移 `pos += offset`，`vert_map[v] = v`）；否则 → 边界顶点
///    （创建新顶点 `v' = pos + offset`，`vert_map[v] = v'`）。
/// 2. **半边分类**：对每条区域半边 `h`，若 `h.twin.face` 也在区域内 → 内部半边
///    （直接更新 `h.vertex = vert_map[v_t]`，twin 同理）；否则 → 边界半边
///    （创建 `h_top` 替换 `h` 在面环中的位置，`h` 释放供侧面使用）。
/// 3. **侧面创建**：每条边界半边 `h(v_o→v_t)` 生成四边形 `(v_o, v_t, v_t', v_o')`，
///    三角化为 `T1=(v_o,v_t,v_t')` 与 `T2=(v_o,v_t',v_o')`。
/// 4. **垂直半边**：每个边界顶点 `v` 创建一对 `(up: v→v', down: v'→v)`，
///    相邻侧面共享。
///
/// ## 边界情况
/// - `offset` 长度 < 1e-12 → [`TopologyError::DegenerateTriangle`]
/// - 任一侧面面积 < 1e-12 → [`TopologyError::DegenerateTriangle`]
/// - 区域包含整个网格 → 所有顶点为内部顶点，仅平移，无侧面
///
/// ## 返回
/// 新创建的面 ID 列表（仅侧面三角形，顺序为每条边界边的 `[T1, T2]`）。
/// 原区域面被复用为顶面（朝向不变），不在返回列表中。
pub fn extrude_region(
    mesh: &mut MeshStorage,
    faces: &[FaceId],
    offset: [f64; 3],
) -> Result<Vec<FaceId>, TopologyError> {
    // ---------- 1. 退化检查 ----------
    let offset_len = (offset[0].powi(2) + offset[1].powi(2) + offset[2].powi(2)).sqrt();
    if offset_len < 1e-12 {
        return Err(TopologyError::DegenerateTriangle);
    }

    // ---------- 2. 验证输入面 ----------
    if faces.is_empty() {
        return Err(TopologyError::Inconsistent("挤出区域为空".into()));
    }
    let region_set: HashSet<FaceId> = faces.iter().copied().collect();
    if region_set.len() != faces.len() {
        return Err(TopologyError::Inconsistent("挤出区域包含重复面".into()));
    }
    for &f in faces {
        if !mesh.contains_face(f) {
            return Err(TopologyError::Inconsistent(format!("面 {:?} 不存在", f)));
        }
    }

    // ---------- 3. 收集区域半边与顶点 ----------
    // region_face_hes: 每个区域面的三条半边（按面组织）
    let mut region_face_hes: Vec<(FaceId, [HalfEdgeId; 3])> = Vec::new();
    let mut region_verts_set: HashSet<VertexId> = HashSet::new();
    for &f in faces {
        let hes: Vec<HalfEdgeId> = FaceHalfEdges::new(mesh, f).collect();
        if hes.len() != 3 {
            return Err(TopologyError::Inconsistent(format!(
                "面 {:?} 边界环长度={}, 非三角面",
                f,
                hes.len()
            )));
        }
        for &h in &hes {
            let v_t = mesh
                .get_halfedge(h)
                .ok_or(TopologyError::InvalidHalfEdge(h))?
                .vertex;
            region_verts_set.insert(v_t);
        }
        region_face_hes.push((f, [hes[0], hes[1], hes[2]]));
    }

    // ---------- 4. 顶点分类 ----------
    // vert_map: 原顶点 → 挤出后顶点（内部顶点映射到自身，边界顶点映射到新顶点）
    let mut vert_map: HashMap<VertexId, VertexId> = HashMap::new();
    for &v in &region_verts_set {
        let adj_faces: Vec<FaceId> = VertexAdjacentFaces::new(mesh, v).collect();
        let all_in_region =
            !adj_faces.is_empty() && adj_faces.iter().all(|f| region_set.contains(f));
        if all_in_region {
            // 内部顶点：原地平移
            let pos = mesh
                .get_vertex(v)
                .ok_or_else(|| TopologyError::Inconsistent(format!("顶点 {:?} 不存在", v)))?
                .position;
            mesh.get_vertex_mut(v).unwrap().position =
                [pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2]];
            vert_map.insert(v, v);
        } else {
            // 边界顶点：创建新顶点
            let pos = mesh
                .get_vertex(v)
                .ok_or_else(|| TopologyError::Inconsistent(format!("顶点 {:?} 不存在", v)))?
                .position;
            let v_new = mesh.add_vertex(Vertex::new([
                pos[0] + offset[0],
                pos[1] + offset[1],
                pos[2] + offset[2],
            ]));
            vert_map.insert(v, v_new);
        }
    }

    // ---------- 5. 半边分类 ----------
    // boundary_hes: 边界半边列表 (h, v_o, v_t, F)
    // interior_hes: 内部半边集合
    let mut boundary_hes: Vec<(HalfEdgeId, VertexId, VertexId, FaceId)> = Vec::new();
    let mut interior_hes: HashSet<HalfEdgeId> = HashSet::new();

    for &(_f, hes) in &region_face_hes {
        for &h in &hes {
            let h_data = mesh
                .get_halfedge(h)
                .ok_or(TopologyError::InvalidHalfEdge(h))?;
            let twin_id = h_data
                .twin
                .ok_or_else(|| TopologyError::Inconsistent(format!("半边 {:?} 无 twin", h)))?;
            let twin_data = mesh
                .get_halfedge(twin_id)
                .ok_or(TopologyError::InvalidHalfEdge(twin_id))?;
            let v_t = h_data.vertex;
            let v_o = twin_data.vertex;
            match twin_data.face {
                Some(twin_face) if region_set.contains(&twin_face) => {
                    interior_hes.insert(h);
                }
                _ => {
                    boundary_hes.push((h, v_o, v_t, _f));
                }
            }
        }
    }

    // ---------- 6. 退化侧面检查 ----------
    // 侧面三角形 (v_o, v_t, v_o + offset)，退化 ⟺ 三顶点共线
    // ⟺ (pos_t - pos_o) 与 offset 共线。用 Shewchuk 鲁棒谓词精确判定
    for &(_h, v_o, v_t, _f) in &boundary_hes {
        let pos_o = mesh
            .get_vertex(v_o)
            .ok_or_else(|| TopologyError::Inconsistent(format!("顶点 {:?} 不存在", v_o)))?
            .position;
        let pos_t = mesh
            .get_vertex(v_t)
            .ok_or_else(|| TopologyError::Inconsistent(format!("顶点 {:?} 不存在", v_t)))?
            .position;
        let pos_o_up = [
            pos_o[0] + offset[0],
            pos_o[1] + offset[1],
            pos_o[2] + offset[2],
        ];
        if crate::predicates::is_triangle_degenerate_3d(pos_o, pos_t, pos_o_up) {
            return Err(TopologyError::DegenerateTriangle);
        }
    }

    // ---------- 7. 为每个边界顶点创建垂直半边对 ----------
    // vert_he: 边界顶点 → (up: v→v', down: v'→v)
    let mut vert_he: HashMap<VertexId, (HalfEdgeId, HalfEdgeId)> = HashMap::new();
    for &(_h, v_o, v_t, _f) in &boundary_hes {
        for &v in &[v_o, v_t] {
            vert_he.entry(v).or_insert_with(|| {
                let v_new = vert_map[&v];
                let up = mesh.add_halfedge(HalfEdge::new(v_new)); // v → v'
                let down = mesh.add_halfedge(HalfEdge::new(v)); // v' → v
                mesh.get_halfedge_mut(up).unwrap().twin = Some(down);
                mesh.get_halfedge_mut(down).unwrap().twin = Some(up);
                (up, down)
            });
        }
    }

    // ---------- 8. 处理内部半边：更新 vertex ----------
    for &h in &interior_hes {
        let v_t = mesh.get_halfedge(h).unwrap().vertex;
        let new_v = vert_map[&v_t];
        mesh.get_halfedge_mut(h).unwrap().vertex = new_v;
    }

    // ---------- 9. 处理边界半边：创建 h_top 并替换 h 在面环中的位置 ----------
    // top_he_map: 原边界半边 h → 新顶半边 h_top
    // (h_top.vertex = vert_map[v_t], h_top.twin = s_top, s_top.vertex = vert_map[v_o])
    let mut top_he_map: HashMap<HalfEdgeId, HalfEdgeId> = HashMap::new();
    let mut s_top_map: HashMap<HalfEdgeId, HalfEdgeId> = HashMap::new(); // h → s_top
    for &(h, v_o, v_t, _f) in &boundary_hes {
        let v_t_new = vert_map[&v_t];
        let v_o_new = vert_map[&v_o];
        let h_top = mesh.add_halfedge(HalfEdge::new(v_t_new)); // v_o' → v_t'
        let s_top = mesh.add_halfedge(HalfEdge::new(v_o_new)); // v_t' → v_o'
        mesh.get_halfedge_mut(h_top).unwrap().twin = Some(s_top);
        mesh.get_halfedge_mut(s_top).unwrap().twin = Some(h_top);
        top_he_map.insert(h, h_top);
        s_top_map.insert(h, s_top);
    }

    // ---------- 10. 重建区域面的半边环 ----------
    // 每个区域面 F 的环：对每条半边 h，若 h 是边界则替换为 h_top，否则保留 h
    for &(f, hes) in &region_face_hes {
        let [h0, h1, h2] = hes;
        // 计算替换后的半边
        let r0 = *top_he_map.get(&h0).unwrap_or(&h0);
        let r1 = *top_he_map.get(&h1).unwrap_or(&h1);
        let r2 = *top_he_map.get(&h2).unwrap_or(&h2);

        // 设置环
        let r0_he = mesh.get_halfedge_mut(r0).unwrap();
        r0_he.next = Some(r1);
        r0_he.prev = Some(r2);
        r0_he.face = Some(f);

        let r1_he = mesh.get_halfedge_mut(r1).unwrap();
        r1_he.next = Some(r2);
        r1_he.prev = Some(r0);
        r1_he.face = Some(f);

        let r2_he = mesh.get_halfedge_mut(r2).unwrap();
        r2_he.next = Some(r0);
        r2_he.prev = Some(r1);
        r2_he.face = Some(f);

        mesh.get_face_mut(f).unwrap().halfedge = Some(r0);

        // 清除原边界半边的 next/prev/face（供侧面复用）
        for &h in &hes {
            if top_he_map.contains_key(&h) {
                let h_he = mesh.get_halfedge_mut(h).unwrap();
                h_he.next = None;
                h_he.prev = None;
                h_he.face = None;
            }
        }
    }

    // ---------- 11. 创建侧面 ----------
    let mut new_faces = Vec::new();
    for &(h, v_o, v_t, _f) in &boundary_hes {
        let v_t_new = vert_map[&v_t];
        let s_top = s_top_map[&h];
        let (up_t, _down_t) = vert_he[&v_t]; // v_t → v_t_new
        let (_up_o, down_o) = vert_he[&v_o]; // v_o_new → v_o

        // 对角线
        let diag = mesh.add_halfedge(HalfEdge::new(v_t_new)); // v_o → v_t_new
        let diag_rev = mesh.add_halfedge(HalfEdge::new(v_o)); // v_t_new → v_o
        mesh.get_halfedge_mut(diag).unwrap().twin = Some(diag_rev);
        mesh.get_halfedge_mut(diag_rev).unwrap().twin = Some(diag);

        // T1 = (v_o, v_t, v_t_new): h → up_t → diag_rev
        let f_t1 = mesh.add_face(Face::new());
        {
            let he = mesh.get_halfedge_mut(h).unwrap();
            he.next = Some(up_t);
            he.prev = Some(diag_rev);
            he.face = Some(f_t1);

            let he = mesh.get_halfedge_mut(up_t).unwrap();
            he.next = Some(diag_rev);
            he.prev = Some(h);
            he.face = Some(f_t1);

            let he = mesh.get_halfedge_mut(diag_rev).unwrap();
            he.next = Some(h);
            he.prev = Some(up_t);
            he.face = Some(f_t1);
        }
        mesh.get_face_mut(f_t1).unwrap().halfedge = Some(h);

        // T2 = (v_o, v_t_new, v_o_new): diag → s_top → down_o
        let f_t2 = mesh.add_face(Face::new());
        {
            let he = mesh.get_halfedge_mut(diag).unwrap();
            he.next = Some(s_top);
            he.prev = Some(down_o);
            he.face = Some(f_t2);

            let he = mesh.get_halfedge_mut(s_top).unwrap();
            he.next = Some(down_o);
            he.prev = Some(diag);
            he.face = Some(f_t2);

            let he = mesh.get_halfedge_mut(down_o).unwrap();
            he.next = Some(diag);
            he.prev = Some(s_top);
            he.face = Some(f_t2);
        }
        mesh.get_face_mut(f_t2).unwrap().halfedge = Some(diag);

        new_faces.push(f_t1);
        new_faces.push(f_t2);
    }

    // ---------- 12. 修复顶点 outgoing 入口 ----------
    // 边界顶点 v → up_v (v→v')；新顶点 v' → down_v (v'→v)
    for &(_h, v_o, v_t, _f) in &boundary_hes {
        for &v in &[v_o, v_t] {
            let (up, down) = vert_he[&v];
            // 原顶点 v 的 outgoing 指向 up（v→v'）
            mesh.get_vertex_mut(v).unwrap().halfedge = Some(up);
            // 新顶点 v' 的 outgoing 指向 down（v'→v）
            let v_new = vert_map[&v];
            mesh.get_vertex_mut(v_new).unwrap().halfedge = Some(down);
        }
    }
    // 内部顶点 v：检查 v.halfedge 是否仍指向有效半边（原边界半边已被释放，但
    // 内部顶点不接触边界半边，所以 v.halfedge 应仍指向内部半边）。
    // 此处无需额外处理。

    // ---------- 13. validate_topology 校验 ----------
    let errors = crate::validate::validate_topology(mesh);
    if !errors.is_empty() {
        return Err(TopologyError::Inconsistent(format!(
            "extrude_region 后校验失败：{:?}",
            errors
        )));
    }

    // ---------- 14. 返回新侧面列表 ----------
    Ok(new_faces)
}

// ============================================================
// add_triangle: 高级面构建器
// ============================================================

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
        let h = mesh.get_halfedge_mut(he).unwrap();
        h.next = Some(next);
        h.prev = Some(prev);
    }

    // ---------- 3. 创建面 ----------
    let face = mesh.add_face(Face::new());
    mesh.get_face_mut(face).unwrap().halfedge = Some(h0);
    for he in [h0, h1, h2] {
        mesh.get_halfedge_mut(he).unwrap().face = Some(face);
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
                let old_twin = mesh.get_halfedge(ex).unwrap().twin;
                if let Some(old) = old_twin {
                    mesh.remove_halfedge(old);
                }
                mesh.get_halfedge_mut(he).unwrap().twin = Some(ex);
                mesh.get_halfedge_mut(ex).unwrap().twin = Some(he);
            }
            None => {
                // 无匹配：创建新的边界半边 dst→src
                let twin = mesh.add_halfedge(HalfEdge::new(src));
                mesh.get_halfedge_mut(he).unwrap().twin = Some(twin);
                mesh.get_halfedge_mut(twin).unwrap().twin = Some(he);
            }
        }
    }

    // ---------- 6. 设置顶点 outgoing 半边入口 ----------
    for (v, he) in [(v0, h0), (v1, h1), (v2, h2)] {
        if mesh.get_vertex(v).unwrap().halfedge.is_none() {
            mesh.get_vertex_mut(v).unwrap().halfedge = Some(he);
        }
    }

    // ---------- 7. 最终校验 ----------
    validate_mesh(mesh)?;

    Ok(face)
}

/// 面分裂：在面中心插入新顶点，将 1 个三角面分裂为 3 个。
///
/// 新顶点位于原面三个顶点的几何中心。返回新顶点 ID。
pub fn split_face(mesh: &mut MeshStorage, face: FaceId) -> Result<VertexId, TopologyError> {
    // 1. 获取面信息
    let face_data = mesh
        .get_face(face)
        .ok_or_else(|| TopologyError::Inconsistent("面不存在".into()))?;
    let _start_he = face_data
        .halfedge
        .ok_or_else(|| TopologyError::Inconsistent("面无半边".into()))?;

    // 2. 收集面顶点
    let he_ids: Vec<HalfEdgeId> = FaceHalfEdges::new(mesh, face).collect();
    if he_ids.len() != 3 {
        return Err(TopologyError::Inconsistent("仅支持三角面".into()));
    }
    let he_data: Vec<_> = he_ids
        .iter()
        .map(|&h| mesh.get_halfedge(h).unwrap().clone())
        .collect();
    let verts: Vec<VertexId> = he_data.iter().map(|h| h.vertex).collect();
    let [v0, v1, v2] = [verts[0], verts[1], verts[2]];
    let [_he0, _he1, _he2] = [he_ids[0], he_ids[1], he_ids[2]];
    let old_twins = [he_data[0].twin, he_data[1].twin, he_data[2].twin];

    // 3. 计算中心点
    let pos: Vec<[f64; 3]> = verts
        .iter()
        .map(|&v| mesh.get_vertex(v).unwrap().position)
        .collect();
    let center = [
        (pos[0][0] + pos[1][0] + pos[2][0]) / 3.0,
        (pos[0][1] + pos[1][1] + pos[2][1]) / 3.0,
        (pos[0][2] + pos[1][2] + pos[2][2]) / 3.0,
    ];

    // 4. 插入新顶点
    let m = mesh.add_vertex(Vertex::new(center));

    // 5. 删除旧面及其半边
    mesh.remove_face(face);
    for &he in &he_ids {
        mesh.remove_halfedge(he);
    }

    // 6. 创建 3 个新面，每个面两条"辐条"边 + 一条外边界边
    // F0: m→v1→v2, F1: m→v2→v0, F2: m→v0→v1
    let mut new_faces = Vec::new();
    for (i, &(_, src, dst)) in [(0, v0, v1), (1, v1, v2), (2, v2, v0)].iter().enumerate() {
        // 辐条边: m→src, dst→m
        let spoke_out = mesh.add_halfedge(HalfEdge::new(src)); // m→src
        let spoke_in = mesh.add_halfedge(HalfEdge::new(m)); // dst→m

        // 辐条互为 twin
        mesh.get_halfedge_mut(spoke_out).unwrap().twin = Some(spoke_in);
        mesh.get_halfedge_mut(spoke_in).unwrap().twin = Some(spoke_out);

        // 外边界边：从 src→dst
        let outer = mesh.add_halfedge(HalfEdge::new(dst));
        // 恢复旧 twin 关系
        if let Some(old_twin) = old_twins[(i + 1) % 3] {
            mesh.get_halfedge_mut(outer).unwrap().twin = Some(old_twin);
            if let Some(t) = mesh.get_halfedge_mut(old_twin) {
                t.twin = Some(outer);
            }
        }

        // 设置 next/prev 环: m→src→dst→m (spoke_out → outer → spoke_in)
        mesh.get_halfedge_mut(spoke_out).unwrap().next = Some(outer);
        mesh.get_halfedge_mut(spoke_out).unwrap().prev = Some(spoke_in);
        mesh.get_halfedge_mut(outer).unwrap().next = Some(spoke_in);
        mesh.get_halfedge_mut(outer).unwrap().prev = Some(spoke_out);
        mesh.get_halfedge_mut(spoke_in).unwrap().next = Some(spoke_out);
        mesh.get_halfedge_mut(spoke_in).unwrap().prev = Some(outer);

        // 创建面
        let new_f = mesh.add_face(Face::new());
        mesh.get_face_mut(new_f).unwrap().halfedge = Some(spoke_out);
        for &he in &[spoke_out, outer, spoke_in] {
            mesh.get_halfedge_mut(he).unwrap().face = Some(new_f);
        }
        new_faces.push(new_f);
    }

    // 7. 设置中心顶点的 outgoing 半边
    mesh.get_vertex_mut(m).unwrap().halfedge = Some(
        mesh.get_face(new_faces[0])
            .and_then(|f| f.halfedge)
            .unwrap(),
    );

    // 8. 校验
    validate_mesh(mesh)?;

    Ok(m)
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::FaceId;
    use crate::storage::{Face, HalfEdge, MeshStorage, Vertex};
    use crate::traversal::{FaceHalfEdges, VertexRing, is_boundary_edge, is_boundary_vertex};

    // ---------- 测试夹具 ----------

    /// 构造单个三角面片（CCW），所有边都是边界边。
    /// ```text
    ///        v2
    ///        ▲
    ///       / │
    ///   h2 /  │ t1
    ///     /   │
    ///    /    │
    ///   v0────┼───▶ v1
    ///    \    │
    ///     \   │
    ///   t2 \  │ h0
    ///       \ │
    ///        ▼
    /// ```
    /// 面 F = (h0: v0→v1, h1: v1→v2, h2: v2→v0)
    fn build_triangle() -> (MeshStorage, [VertexId; 3], [HalfEdgeId; 6], FaceId) {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));

        let h0 = mesh.add_halfedge(HalfEdge::new(v1)); // v0→v1
        let h1 = mesh.add_halfedge(HalfEdge::new(v2)); // v1→v2
        let h2 = mesh.add_halfedge(HalfEdge::new(v0)); // v2→v0
        let t0 = mesh.add_halfedge(HalfEdge::new(v0)); // v1→v0
        let t1 = mesh.add_halfedge(HalfEdge::new(v1)); // v2→v1
        let t2 = mesh.add_halfedge(HalfEdge::new(v2)); // v0→v2

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

        (mesh, [v0, v1, v2], [h0, h1, h2, t0, t1, t2], f)
    }

    /// 构造两个三角形拼成的四边形，共享边 v0-v1。
    /// ```text
    ///   v2 ────── v3
    ///    │ ╲     ╱│
    ///    │  ╲   ╱ │
    ///    │ F1╲ ╱F2│
    ///    │   ╲╱  │
    ///    │   ╱╲  │
    ///    │  ╱  ╲ │
    ///    │ ╱   ╲│
    ///   v0 ──── v1   （共享边 v0-v1）
    /// ```
    fn build_two_triangles() -> (MeshStorage, [VertexId; 4], [HalfEdgeId; 10], FaceId, FaceId) {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));
        let v3 = mesh.add_vertex(Vertex::new([1.0, -1.0, 0.0])); // F2 在共享边下方，几何 CCW

        // F1 = v0→v1→v2→v0
        let h0 = mesh.add_halfedge(HalfEdge::new(v1)); // v0→v1
        let h1 = mesh.add_halfedge(HalfEdge::new(v2)); // v1→v2
        let h2 = mesh.add_halfedge(HalfEdge::new(v0)); // v2→v0
        // F2 = v1→v0→v3→v1
        let g0 = mesh.add_halfedge(HalfEdge::new(v0)); // v1→v0 (twin of h0)
        let g1 = mesh.add_halfedge(HalfEdge::new(v3)); // v0→v3
        let g2 = mesh.add_halfedge(HalfEdge::new(v1)); // v3→v1
        // 边界 twin
        let t1 = mesh.add_halfedge(HalfEdge::new(v1)); // v2→v1
        let t2 = mesh.add_halfedge(HalfEdge::new(v2)); // v0→v2
        let t_g1 = mesh.add_halfedge(HalfEdge::new(v0)); // v3→v0
        let t_g2 = mesh.add_halfedge(HalfEdge::new(v3)); // v1→v3

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
        mesh.get_vertex_mut(v2).unwrap().halfedge = Some(h1);
        mesh.get_vertex_mut(v3).unwrap().halfedge = Some(g1);
        mesh.get_face_mut(f1).unwrap().halfedge = Some(h0);
        mesh.get_face_mut(f2).unwrap().halfedge = Some(g0);

        (
            mesh,
            [v0, v1, v2, v3],
            [h0, h1, h2, g0, g1, g2, t1, t2, t_g1, t_g2],
            f1,
            f2,
        )
    }

    /// 构造 3 个三角形围成的闭合扇形，中心 c 是内部顶点。
    /// ```text
    ///         v2
    ///        ╱ ╲
    ///       ╱   ╲
    ///      ╱  F3 ╲
    ///     ╱       ╲
    ///    v0 ── c ── v1
    ///     ╲       ╱
    ///      ╲ F1  ╱
    ///       ╲   ╱
    ///        ╲ ╱
    /// ```
    /// F1 = c→v0→v1→c, F2 = c→v1→v2→c, F3 = c→v2→v0→c
    fn build_closed_fan() -> (MeshStorage, VertexId, [VertexId; 3], [FaceId; 3]) {
        let mut mesh = MeshStorage::new();
        let c = mesh.add_vertex(Vertex::new([0.5, 0.5, 0.0]));
        let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let v2 = mesh.add_vertex(Vertex::new([0.5, 1.0, 0.0]));

        let a1 = mesh.add_halfedge(HalfEdge::new(v0)); // c→v0
        let b1 = mesh.add_halfedge(HalfEdge::new(v1)); // v0→v1
        let c1 = mesh.add_halfedge(HalfEdge::new(c)); // v1→c
        let a2 = mesh.add_halfedge(HalfEdge::new(v1)); // c→v1
        let b2 = mesh.add_halfedge(HalfEdge::new(v2)); // v1→v2
        let c2 = mesh.add_halfedge(HalfEdge::new(c)); // v2→c
        let a3 = mesh.add_halfedge(HalfEdge::new(v2)); // c→v2
        let b3 = mesh.add_halfedge(HalfEdge::new(v0)); // v2→v0
        let c3 = mesh.add_halfedge(HalfEdge::new(c)); // v0→c
        let t1 = mesh.add_halfedge(HalfEdge::new(v0)); // v1→v0
        let t2 = mesh.add_halfedge(HalfEdge::new(v1)); // v2→v1
        let t3 = mesh.add_halfedge(HalfEdge::new(v2)); // v0→v2

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

        (mesh, c, [v0, v1, v2], [f1, f2, f3])
    }

    // ---------- validate_mesh 测试 ----------

    #[test]
    fn validate_clean_meshes() {
        let (m1, _, _, _) = build_triangle();
        assert!(validate_mesh(&m1).is_ok());

        let (m2, _, _, _, _) = build_two_triangles();
        assert!(validate_mesh(&m2).is_ok());

        let (m3, _, _, _) = build_closed_fan();
        assert!(validate_mesh(&m3).is_ok());
    }

    // ---------- split_edge 测试 ----------

    #[test]
    fn split_boundary_edge_of_single_triangle() {
        // 分裂单三角形的边界边 h0 (v0→v1)
        let (mut mesh, v, he, _f) = build_triangle();
        let [v0, v1, _v2] = v;
        let [h0, _h1, _h2, _t0, _t1, _t2] = he;

        let v_before = mesh.vertex_count();
        let f_before = mesh.face_count();
        let he_before = mesh.halfedge_count();

        let m = split_edge(&mut mesh, h0).expect("分裂边界边应成功");

        // 数量校验：+1 顶点, +1 面, +4 半边（删 2 加 6）
        assert_eq!(mesh.vertex_count(), v_before + 1);
        assert_eq!(mesh.face_count(), f_before + 1);
        assert_eq!(mesh.halfedge_count(), he_before + 4);

        // 新顶点存在
        assert!(mesh.contains_vertex(m));
        // 原半边已删除
        assert!(!mesh.contains_halfedge(h0));

        // 拓扑合法
        assert!(validate_mesh(&mesh).is_ok());

        // M 位于 v0-v1 中点
        let pos_m = mesh.get_vertex(m).unwrap().position;
        let pos_v0 = mesh.get_vertex(v0).unwrap().position;
        let pos_v1 = mesh.get_vertex(v1).unwrap().position;
        assert!((pos_m[0] - (pos_v0[0] + pos_v1[0]) / 2.0).abs() < 1e-12);

        // 现在有 2 个三角形
        assert_eq!(mesh.face_count(), 2);
        // 所有面都是三角面
        for f in mesh.face_ids().collect::<Vec<_>>() {
            let count = FaceHalfEdges::new(&mesh, f).count();
            assert_eq!(count, 3, "折叠后面应为三角面");
        }
    }

    #[test]
    fn split_interior_edge_of_two_triangles() {
        // 分享两三角形拼四边形的内部边 h0 (v0→v1)
        let (mut mesh, _v, he, _f1, _f2) = build_two_triangles();
        let [h0, _h1, _h2, _g0, _g1, _g2, _t1, _t2, _t_g1, _t_g2] = he;

        let v_before = mesh.vertex_count();
        let f_before = mesh.face_count();
        let he_before = mesh.halfedge_count();

        let m = split_edge(&mut mesh, h0).expect("分裂内部边应成功");

        // 数量校验：+1 顶点, +2 面, +6 半边（删 2 加 8）
        assert_eq!(mesh.vertex_count(), v_before + 1);
        assert_eq!(mesh.face_count(), f_before + 2);
        assert_eq!(mesh.halfedge_count(), he_before + 6);

        assert!(mesh.contains_vertex(m));
        assert!(!mesh.contains_halfedge(h0));

        assert!(validate_mesh(&mesh).is_ok());

        // 所有面都是三角面
        for f in mesh.face_ids().collect::<Vec<_>>() {
            let count = FaceHalfEdges::new(&mesh, f).count();
            assert_eq!(count, 3, "分裂后面应为三角面");
        }
    }

    #[test]
    fn split_boundary_edge_passed_as_boundary_halfedge() {
        // 传入边界半边 t0 (v1→v0)，应自动转为操作 twin h0
        let (mut mesh, _v, he, _f) = build_triangle();
        let [h0, _h1, _h2, t0, _t1, _t2] = he;

        let m = split_edge(&mut mesh, t0).expect("传入边界半边应自动转为操作 twin");
        assert!(mesh.contains_vertex(m));
        assert!(!mesh.contains_halfedge(h0));
        assert!(!mesh.contains_halfedge(t0));
        assert!(validate_mesh(&mesh).is_ok());
    }

    // ---------- flip_edge 测试 ----------

    #[test]
    fn flip_interior_edge_of_two_triangles() {
        let (mut mesh, v, he, _f1, _f2) = build_two_triangles();
        let [_v0, _v1, v2, _v3] = v;
        let [h0, _h1, _h2, _g0, _g1, _g2, _t1, _t2, _t_g1, _t_g2] = he;

        let v_before = mesh.vertex_count();
        let f_before = mesh.face_count();
        let he_before = mesh.halfedge_count();

        flip_edge(&mut mesh, h0).expect("翻转内部边应成功");

        // 数量不变
        assert_eq!(mesh.vertex_count(), v_before);
        assert_eq!(mesh.face_count(), f_before);
        assert_eq!(mesh.halfedge_count(), he_before);

        assert!(validate_mesh(&mesh).is_ok());

        // 翻转后 h0 的 vertex 应变为 v2（原 C），twin 的 vertex 应变为 v3（原 D）
        // h0 现在是 D→C = v3→v2
        let h = mesh.get_halfedge(h0).unwrap();
        assert_eq!(h.vertex, v2, "翻转后 h.vertex 应为 C=v2");

        // 所有面都是三角面
        for f in mesh.face_ids().collect::<Vec<_>>() {
            let count = FaceHalfEdges::new(&mesh, f).count();
            assert_eq!(count, 3, "翻转后面应为三角面");
        }
    }

    #[test]
    fn flip_boundary_edge_fails() {
        let (mut mesh, _v, he, _f) = build_triangle();
        let [h0, _h1, _h2, _t0, _t1, _t2] = he;

        let err = flip_edge(&mut mesh, h0).unwrap_err();
        assert_eq!(err, TopologyError::FlipOnBoundaryEdge(h0));
    }

    #[test]
    fn flip_then_flip_restores_topology() {
        let (mut mesh, _v, he, _f1, _f2) = build_two_triangles();
        let [h0, _, _, _, _, _, _, _, _, _] = he;

        flip_edge(&mut mesh, h0).expect("第一次翻转");
        assert!(validate_mesh(&mesh).is_ok());
        flip_edge(&mut mesh, h0).expect("第二次翻转恢复");
        assert!(validate_mesh(&mesh).is_ok());

        // 两次翻转后 h0.vertex 应回到原值 v1
        // 原 h0: v0→v1, vertex=v1
        // 翻转后 h0 变为 v3→v2，再翻转变回 v0→v1
        // 验证面数和半边数不变
        assert_eq!(mesh.face_count(), 2);
    }

    // ---------- collapse_edge 测试 ----------

    #[test]
    fn collapse_interior_edge_of_closed_fan() {
        // 在闭合扇形中折叠 c-v0 之间的边（a1: c→v0）
        let (mut mesh, c, outer, _faces) = build_closed_fan();
        let [v0, _v1, _v2] = outer;

        // 找到 c→v0 的半边
        let a1 = VertexRing::new(&mesh, c)
            .find(|he| mesh.get_halfedge(*he).unwrap().vertex == v0)
            .expect("c→v0 半边必须存在");

        let v_before = mesh.vertex_count();
        let f_before = mesh.face_count();
        let he_before = mesh.halfedge_count();

        let k = collapse_edge(&mut mesh, a1).expect("折叠内部边应成功");

        // 数量校验：-1 顶点, -2 面, -6 半边
        assert_eq!(mesh.vertex_count(), v_before - 1);
        assert_eq!(mesh.face_count(), f_before - 2);
        assert_eq!(mesh.halfedge_count(), he_before - 6);

        assert!(mesh.contains_vertex(k));
        assert!(!mesh.contains_vertex(c));
        assert!(!mesh.contains_vertex(v0));

        assert!(validate_mesh(&mesh).is_ok());

        // 剩余 1 个三角形
        assert_eq!(mesh.face_count(), 1);
        for f in mesh.face_ids().collect::<Vec<_>>() {
            let count = FaceHalfEdges::new(&mesh, f).count();
            assert_eq!(count, 3, "折叠后面应为三角面");
        }
    }

    #[test]
    fn collapse_boundary_edge_fails() {
        let (mut mesh, _v, he, _f) = build_triangle();
        let [h0, _h1, _h2, _t0, _t1, _t2] = he;

        let err = collapse_edge(&mut mesh, h0).unwrap_err();
        assert_eq!(err, TopologyError::CollapseOnBoundaryEdge(h0));
    }

    /// 构造一个链接条件不满足的网格：5 顶点 4 面，A-B 边的公共邻居为 {C, D, E}（超过 2）。
    ///
    /// ```text
    ///        E
    ///       ╱ ╲
    ///   F3 ╱   ╲ F4
    ///     ╱     ╲
    ///    A ──F1── B
    ///     ╲     ╱
    ///   F2 ╲   ╱
    ///       ╲ ╱
    ///        D        （C 在 F1 下方）
    /// ```
    /// F1 = A→B→C→A, F2 = B→A→D→B, F3 = A→C→E→A, F4 = B→D→E→B
    /// A 的邻居 = {B, C, D, E}，B 的邻居 = {A, C, D, E}
    /// 公共邻居 = {C, D, E} ≠ {C, D} → 链接条件违反
    fn build_link_violation_mesh() -> (MeshStorage, VertexId, VertexId) {
        let mut mesh = MeshStorage::new();
        let a = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let b = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let c = mesh.add_vertex(Vertex::new([0.5, -1.0, 0.0]));
        let d = mesh.add_vertex(Vertex::new([0.5, 0.0, 1.0]));
        let e = mesh.add_vertex(Vertex::new([0.5, 1.0, 0.0]));

        // F1 = A→B→C→A
        let h_ab = mesh.add_halfedge(HalfEdge::new(b)); // A→B
        let h_bc = mesh.add_halfedge(HalfEdge::new(c)); // B→C
        let h_ca = mesh.add_halfedge(HalfEdge::new(a)); // C→A
        // F2 = B→A→D→B
        let h_ba = mesh.add_halfedge(HalfEdge::new(a)); // B→A
        let h_ad = mesh.add_halfedge(HalfEdge::new(d)); // A→D
        let h_db = mesh.add_halfedge(HalfEdge::new(b)); // D→B
        // F3 = A→C→E→A
        let h_ac = mesh.add_halfedge(HalfEdge::new(c)); // A→C
        let h_ce = mesh.add_halfedge(HalfEdge::new(e)); // C→E
        let h_ea = mesh.add_halfedge(HalfEdge::new(a)); // E→A
        // F4 = B→D→E→B
        let h_bd = mesh.add_halfedge(HalfEdge::new(d)); // B→D
        let h_de = mesh.add_halfedge(HalfEdge::new(e)); // D→E
        let h_eb = mesh.add_halfedge(HalfEdge::new(b)); // E→B
        // 边界 twin
        let t_bc = mesh.add_halfedge(HalfEdge::new(b)); // C→B
        let t_ad = mesh.add_halfedge(HalfEdge::new(a)); // D→A
        let t_ce = mesh.add_halfedge(HalfEdge::new(c)); // E→C
        let t_ea = mesh.add_halfedge(HalfEdge::new(e)); // A→E
        let t_de = mesh.add_halfedge(HalfEdge::new(d)); // E→D
        let t_eb = mesh.add_halfedge(HalfEdge::new(e)); // B→E

        let f1 = mesh.add_face(Face::new());
        let f2 = mesh.add_face(Face::new());
        let f3 = mesh.add_face(Face::new());
        let f4 = mesh.add_face(Face::new());

        // F1 内部
        for (he, twin, next, prev) in [
            (h_ab, h_ba, h_bc, h_ca),
            (h_bc, t_bc, h_ca, h_ab),
            (h_ca, h_ac, h_ab, h_bc),
        ] {
            let h = mesh.get_halfedge_mut(he).unwrap();
            h.twin = Some(twin);
            h.next = Some(next);
            h.prev = Some(prev);
            h.face = Some(f1);
        }
        // F2 内部
        for (he, twin, next, prev) in [
            (h_ba, h_ab, h_ad, h_db),
            (h_ad, t_ad, h_db, h_ba),
            (h_db, h_bd, h_ba, h_ad),
        ] {
            let h = mesh.get_halfedge_mut(he).unwrap();
            h.twin = Some(twin);
            h.next = Some(next);
            h.prev = Some(prev);
            h.face = Some(f2);
        }
        // F3 内部
        for (he, twin, next, prev) in [
            (h_ac, h_ca, h_ce, h_ea),
            (h_ce, t_ce, h_ea, h_ac),
            (h_ea, t_ea, h_ac, h_ce),
        ] {
            let h = mesh.get_halfedge_mut(he).unwrap();
            h.twin = Some(twin);
            h.next = Some(next);
            h.prev = Some(prev);
            h.face = Some(f3);
        }
        // F4 内部
        for (he, twin, next, prev) in [
            (h_bd, h_db, h_de, h_eb),
            (h_de, t_de, h_eb, h_bd),
            (h_eb, t_eb, h_bd, h_de),
        ] {
            let h = mesh.get_halfedge_mut(he).unwrap();
            h.twin = Some(twin);
            h.next = Some(next);
            h.prev = Some(prev);
            h.face = Some(f4);
        }
        // 边界 twin（仅互指 twin）
        for (t, he) in [
            (t_bc, h_bc),
            (t_ad, h_ad),
            (t_ce, h_ce),
            (t_ea, h_ea),
            (t_de, h_de),
            (t_eb, h_eb),
        ] {
            mesh.get_halfedge_mut(t).unwrap().twin = Some(he);
        }
        // 顶点 outgoing 入口
        mesh.get_vertex_mut(a).unwrap().halfedge = Some(h_ab);
        mesh.get_vertex_mut(b).unwrap().halfedge = Some(h_ba);
        mesh.get_vertex_mut(c).unwrap().halfedge = Some(h_ca);
        mesh.get_vertex_mut(d).unwrap().halfedge = Some(h_db);
        mesh.get_vertex_mut(e).unwrap().halfedge = Some(h_ea);
        // 面入口
        mesh.get_face_mut(f1).unwrap().halfedge = Some(h_ab);
        mesh.get_face_mut(f2).unwrap().halfedge = Some(h_ba);
        mesh.get_face_mut(f3).unwrap().halfedge = Some(h_ac);
        mesh.get_face_mut(f4).unwrap().halfedge = Some(h_bd);

        (mesh, a, b)
    }

    #[test]
    fn collapse_with_violated_link_condition_fails() {
        // 5 顶点 4 面网格：A-B 边的公共邻居 = {C, D, E}（3 个 > 2），违反链接条件
        let (mut mesh, a, b) = build_link_violation_mesh();

        // 先验证这是合法的流形网格
        assert!(validate_mesh(&mesh).is_ok());

        // 找到 A→B 的半边
        let h_ab = VertexRing::new(&mesh, a)
            .find(|he| mesh.get_halfedge(*he).unwrap().vertex == b)
            .expect("A→B 半边必须存在");

        let err = collapse_edge(&mut mesh, h_ab).unwrap_err();
        assert_eq!(
            err,
            TopologyError::LinkConditionViolated { a, b },
            "公共邻居 {{C,D,E}} != {{C,D}} → 应返回链接条件违反"
        );

        // 确认网格未被修改（操作是原子的）
        assert!(validate_mesh(&mesh).is_ok());
        assert!(mesh.contains_vertex(a));
        assert!(mesh.contains_vertex(b));
    }

    #[test]
    fn collapse_link_condition_check_function() {
        // 直接测试 check_link_condition 函数
        let (mesh, c, outer, _faces) = build_closed_fan();
        let [v0, v1, v2] = outer;

        // c-v0 的公共邻居 = {v1, v2}
        assert!(check_link_condition(&mesh, c, v0, v1, v2));
        // 传入错误的 C/D 应返回 false
        assert!(!check_link_condition(&mesh, c, v0, v1, v0));
    }

    // ---------- 综合测试 ----------

    #[test]
    fn split_then_validate_all_faces_triangular() {
        let (mut mesh, _v, he, _f1, _f2) = build_two_triangles();
        let [h0, _h1, _, _, _, _, _, _, _, _] = he;

        // 分裂内部边
        split_edge(&mut mesh, h0).unwrap();
        assert!(validate_mesh(&mesh).is_ok());

        // 再分裂一条边界边
        // 找 v2 的一条边界半边
        let boundary_he = mesh
            .halfedge_ids()
            .find(|he| is_boundary_edge(&mesh, *he))
            .expect("应存在边界半边");
        split_edge(&mut mesh, boundary_he).unwrap();
        assert!(validate_mesh(&mesh).is_ok());

        // 所有面都是三角面
        for f in mesh.face_ids().collect::<Vec<_>>() {
            assert_eq!(FaceHalfEdges::new(&mesh, f).count(), 3);
        }
    }

    #[test]
    fn flip_preserves_boundary_vertices() {
        let (mut mesh, v, he, _f1, _f2) = build_two_triangles();
        let [v0, v1, v2, v3] = v;
        let [h0, _, _, _, _, _, _, _, _, _] = he;

        flip_edge(&mut mesh, h0).unwrap();

        // 翻转后所有顶点仍是边界顶点（两三角形拼四边形的四个角都是边界）
        for vi in [v0, v1, v2, v3] {
            assert!(
                is_boundary_vertex(&mesh, vi),
                "翻转后顶点 {:?} 应仍为边界顶点",
                vi
            );
        }
    }

    #[test]
    fn collapse_then_remaining_mesh_valid() {
        // 在闭合扇形中折叠一条边，然后验证剩余网格
        let (mut mesh, c, outer, _faces) = build_closed_fan();
        let [v0, v1, v2] = outer;

        // 折叠 c→v0
        let a1 = VertexRing::new(&mesh, c)
            .find(|he| mesh.get_halfedge(*he).unwrap().vertex == v0)
            .unwrap();
        let k = collapse_edge(&mut mesh, a1).expect("折叠应成功");

        // K 应该是边界顶点（因为 v0 是边界顶点，折叠后 K 继承其边界性）
        assert!(is_boundary_vertex(&mesh, k), "折叠后新顶点 K 应为边界顶点");
        // v1, v2 仍存在且是边界顶点
        assert!(mesh.contains_vertex(v1));
        assert!(mesh.contains_vertex(v2));
        assert!(is_boundary_vertex(&mesh, v1));
        assert!(is_boundary_vertex(&mesh, v2));

        assert!(validate_mesh(&mesh).is_ok());
    }

    // ---------- collapse_edge_at 测试 ----------

    #[test]
    fn collapse_edge_uses_midpoint_position() {
        // 折叠 c→v0：A=c=(0.5,0.5,0), B=v0=(0,0,0)，中点应为 (0.25,0.25,0)
        let (mut mesh, c, outer, _faces) = build_closed_fan();
        let [v0, _v1, _v2] = outer;

        let a1 = VertexRing::new(&mesh, c)
            .find(|he| mesh.get_halfedge(*he).unwrap().vertex == v0)
            .expect("c→v0 半边必须存在");

        let k = collapse_edge(&mut mesh, a1).expect("折叠应成功");
        let pos = mesh.get_vertex(k).unwrap().position;

        assert_eq!(pos, [0.25, 0.25, 0.0]);
        assert!(validate_mesh(&mesh).is_ok());
    }

    #[test]
    fn collapse_edge_at_uses_custom_position() {
        // 同样折叠 c→v0，但指定自定义位置
        let (mut mesh, c, outer, _faces) = build_closed_fan();
        let [v0, _v1, _v2] = outer;

        let a1 = VertexRing::new(&mesh, c)
            .find(|he| mesh.get_halfedge(*he).unwrap().vertex == v0)
            .expect("c→v0 半边必须存在");

        let target = [1.0, 2.0, 3.0];
        let k = collapse_edge_at(&mut mesh, a1, target).expect("折叠应成功");
        let pos = mesh.get_vertex(k).unwrap().position;

        assert_eq!(pos, target);
        assert!(validate_mesh(&mesh).is_ok());
    }

    #[test]
    fn collapse_edge_at_preserves_topology_counts() {
        // collapse_edge_at 与 collapse_edge 的拓扑变化应完全相同
        let (mut mesh1, c, outer, _faces) = build_closed_fan();
        let [v0, _v1, _v2] = outer;
        let a1 = VertexRing::new(&mesh1, c)
            .find(|he| mesh1.get_halfedge(*he).unwrap().vertex == v0)
            .unwrap();

        let (mut mesh2, c2, outer2, _faces2) = build_closed_fan();
        let [v0_2, _v1_2, _v2_2] = outer2;
        let a1_2 = VertexRing::new(&mesh2, c2)
            .find(|he| mesh2.get_halfedge(*he).unwrap().vertex == v0_2)
            .unwrap();

        let k1 = collapse_edge(&mut mesh1, a1).unwrap();
        let k2 = collapse_edge_at(&mut mesh2, a1_2, [10.0, 20.0, 30.0]).unwrap();

        // 拓扑计数一致
        assert_eq!(mesh1.vertex_count(), mesh2.vertex_count());
        assert_eq!(mesh1.face_count(), mesh2.face_count());
        assert_eq!(mesh1.halfedge_count(), mesh2.halfedge_count());
        // 仅位置不同
        let p1 = mesh1.get_vertex(k1).unwrap().position;
        let p2 = mesh2.get_vertex(k2).unwrap().position;
        assert_eq!(p1, [0.25, 0.25, 0.0]);
        assert_eq!(p2, [10.0, 20.0, 30.0]);
        // 两者都通过校验
        assert!(validate_mesh(&mesh1).is_ok());
        assert!(validate_mesh(&mesh2).is_ok());
    }

    #[test]
    fn collapse_edge_at_on_boundary_edge_fails() {
        // collapse_edge_at 同样禁止折叠边界边
        let (mut mesh, _v, he, _f) = build_triangle();
        let [h0, _h1, _h2, _t0, _t1, _t2] = he;

        let err = collapse_edge_at(&mut mesh, h0, [1.0, 1.0, 1.0]).unwrap_err();
        assert_eq!(err, TopologyError::CollapseOnBoundaryEdge(h0));
    }

    #[test]
    fn invalid_halfedge_returns_error() {
        let mut mesh = MeshStorage::new();
        let fake = HalfEdgeId::default();
        assert_eq!(
            split_edge(&mut mesh, fake).unwrap_err(),
            TopologyError::InvalidHalfEdge(fake)
        );
        assert_eq!(
            flip_edge(&mut mesh, fake).unwrap_err(),
            TopologyError::InvalidHalfEdge(fake)
        );
        assert_eq!(
            collapse_edge(&mut mesh, fake).unwrap_err(),
            TopologyError::InvalidHalfEdge(fake)
        );
    }

    // ---------- extrude_face 测试 ----------

    /// 辅助：构造两个互不连通的三角形，便于测试批量挤出。
    /// 三角形 A = (0,0,0)-(1,0,0)-(0,1,0)，三角形 B = (5,0,0)-(6,0,0)-(5,1,0)。
    fn build_two_disjoint_triangles() -> (MeshStorage, [FaceId; 2]) {
        use crate::io::build_mesh_from_vertices_and_faces;
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [5.0, 1.0, 0.0],
        ];
        let faces = vec![[0, 1, 2], [3, 4, 5]];
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces);
        let fids: Vec<FaceId> = mesh.face_ids().collect();
        assert_eq!(fids.len(), 2);
        (mesh, [fids[0], fids[1]])
    }

    #[test]
    fn extrude_single_triangle() {
        let (mut mesh, v, _he, f) = build_triangle();
        let [v0, v1, v2] = v;

        let v_before = mesh.vertex_count();
        let f_before = mesh.face_count();
        let he_before = mesh.halfedge_count();

        let offset = [0.0, 0.0, 1.0];
        let new_faces = extrude_face(&mut mesh, f, offset).expect("挤出应成功");

        // 数量校验：+3 顶点, +7 面, +18 半边
        assert_eq!(mesh.vertex_count(), v_before + 3);
        assert_eq!(mesh.face_count(), f_before + 7);
        assert_eq!(mesh.halfedge_count(), he_before + 18);
        assert_eq!(new_faces.len(), 7);

        // 完整拓扑校验
        let errors = crate::validate::validate_topology(&mesh);
        assert!(errors.is_empty(), "挤出后拓扑应一致：{:?}", errors);

        // 新顶点位置校验：原顶点 + offset
        let mut new_verts: Vec<VertexId> = mesh
            .vertex_ids()
            .filter(|v| ![v0, v1, v2].contains(v))
            .collect();
        new_verts.sort_by_key(|x| {
            let p = mesh.get_vertex(*x).unwrap().position;
            (p[0] * 1000.0) as i64 + (p[1] * 1000.0) as i64 * 1000
        });
        for v in &new_verts {
            let p = mesh.get_vertex(*v).unwrap().position;
            assert!((p[2] - 1.0).abs() < 1e-12, "新顶点 z 坐标应为 1.0");
        }
        assert_eq!(new_verts.len(), 3);

        // 原顶点位置不变
        for v in [v0, v1, v2] {
            let p = mesh.get_vertex(v).unwrap().position;
            assert_eq!(p[2], 0.0);
        }

        // 原 face 仍存在，且为底面（法向不变）
        assert!(mesh.contains_face(f));

        // Euler 示性数：3-3+1=1（带边界）→ 6-12+8=2（闭合三棱柱）
        let chi = mesh.vertex_count() as i64 - (mesh.halfedge_count() / 2) as i64
            + mesh.face_count() as i64;
        assert_eq!(chi, 2, "挤出后 Euler 示性数应为 2（闭合三棱柱）");
    }

    #[test]
    fn extrude_zero_offset_returns_error() {
        let (mut mesh, _v, _he, f) = build_triangle();
        let err = extrude_face(&mut mesh, f, [0.0, 0.0, 0.0]).unwrap_err();
        assert_eq!(err, TopologyError::DegenerateTriangle);
    }

    #[test]
    fn extrude_degenerate_side_face_returns_error() {
        // offset 平行于三角形所在平面（z=0）会使侧面三角形退化
        let (mut mesh, _v, _he, f) = build_triangle();
        // 三角形在 xy 平面，offset 沿 x 轴 → 边 v0-v1 与 offset 平行 → 侧面退化
        let err = extrude_face(&mut mesh, f, [1.0, 0.0, 0.0]).unwrap_err();
        assert_eq!(err, TopologyError::DegenerateTriangle);
    }

    #[test]
    fn extrude_face_with_internal_edge_returns_error() {
        // build_two_triangles 中 F1 的边 h0 (v0→v1) 是内部边（twin 有 face=f2）
        let (mut mesh, _v, _he, f1, _f2) = build_two_triangles();
        let err = extrude_face(&mut mesh, f1, [0.0, 0.0, 1.0]).unwrap_err();
        match err {
            TopologyError::Inconsistent(_) => {}
            other => panic!("期望 Inconsistent 错误，得到 {:?}", other),
        }
    }

    #[test]
    fn extrude_faces_batch_disjoint() {
        let (mut mesh, fs) = build_two_disjoint_triangles();
        let v_before = mesh.vertex_count();
        let f_before = mesh.face_count();
        let he_before = mesh.halfedge_count();

        let offset = [0.0, 0.0, 2.0];
        let new_faces = extrude_faces(&mut mesh, &fs, offset).expect("批量挤出应成功");

        // 每个面 +3 顶点 / +7 面 / +18 半边
        assert_eq!(mesh.vertex_count(), v_before + 6);
        assert_eq!(mesh.face_count(), f_before + 14);
        assert_eq!(mesh.halfedge_count(), he_before + 36);
        assert_eq!(new_faces.len(), 14);

        // 完整拓扑校验
        let errors = crate::validate::validate_topology(&mesh);
        assert!(errors.is_empty(), "批量挤出后拓扑应一致：{:?}", errors);
    }

    #[test]
    fn extrude_face_from_built_mesh() {
        // 通过 build_mesh_from_vertices_and_faces 构建网格后再挤出
        use crate::io::build_mesh_from_vertices_and_faces;
        let verts = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0]];
        let faces = vec![[0, 1, 2]];
        let mut mesh = build_mesh_from_vertices_and_faces(&verts, &faces);
        let f = mesh.face_ids().next().expect("应有一个面");

        let offset = [0.0, 0.0, 3.0];
        let new_faces = extrude_face(&mut mesh, f, offset).expect("挤出应成功");
        assert_eq!(new_faces.len(), 7);

        // 完整拓扑校验
        let errors = crate::validate::validate_topology(&mesh);
        assert!(errors.is_empty(), "挤出后拓扑应一致：{:?}", errors);

        // 验证顶点位置
        let mut z_top = 0;
        let mut z_bot = 0;
        for v in mesh.vertex_ids() {
            let p = mesh.get_vertex(v).unwrap().position;
            if (p[2] - 3.0).abs() < 1e-12 {
                z_top += 1;
            } else if p[2].abs() < 1e-12 {
                z_bot += 1;
            } else {
                panic!("意外顶点 z 坐标 {}", p[2]);
            }
        }
        assert_eq!(z_top, 3);
        assert_eq!(z_bot, 3);
    }

    #[test]
    fn extrude_face_invalid_face_id() {
        let (mut mesh, _v, _he, _f) = build_triangle();
        let fake = FaceId::default();
        let err = extrude_face(&mut mesh, fake, [0.0, 0.0, 1.0]).unwrap_err();
        match err {
            TopologyError::Inconsistent(_) => {}
            other => panic!("期望 Inconsistent 错误，得到 {:?}", other),
        }
    }

    #[test]
    fn extrude_face_preserves_boundary_after_extrude() {
        // 挤出后整个面成为闭合体（无边界），原边界边变为内部边
        let (mut mesh, _v, _he, f) = build_triangle();
        let _ = extrude_face(&mut mesh, f, [0.0, 0.0, 1.0]).expect("挤出应成功");

        // 闭合体不应有任何边界半边
        let boundary_count = mesh
            .halfedge_ids()
            .filter(|h| {
                mesh.get_halfedge(*h)
                    .map(|he| he.face.is_none())
                    .unwrap_or(false)
            })
            .count();
        assert_eq!(boundary_count, 0, "挤出形成闭合体后不应有边界半边");
    }

    // ---------- extrude_region 测试 ----------

    /// 构造闭合四面体（4 顶点、4 面、6 边、12 半边，所有边内部）。
    /// 顶点：A=(0,0,0), B=(1,0,0), C=(0,1,0), D=(0,0,1)
    /// 面（CCW 朝外）：
    /// - F1 = (A, C, B)  法向 -z
    /// - F2 = (A, B, D)  法向 -y
    /// - F3 = (A, D, C)  法向 -x
    /// - F4 = (B, C, D)  法向 (1,1,1)
    fn build_tetrahedron() -> (MeshStorage, [VertexId; 4], [FaceId; 4]) {
        use crate::io::build_mesh_from_vertices_and_faces;
        let verts = vec![
            [0.0, 0.0, 0.0], // A
            [1.0, 0.0, 0.0], // B
            [0.0, 1.0, 0.0], // C
            [0.0, 0.0, 1.0], // D
        ];
        let faces = vec![
            [0, 2, 1], // F1 = (A, C, B)
            [0, 1, 3], // F2 = (A, B, D)
            [0, 3, 2], // F3 = (A, D, C)
            [1, 2, 3], // F4 = (B, C, D)
        ];
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces);
        let v_ids: Vec<VertexId> = mesh.vertex_ids().collect();
        let f_ids: Vec<FaceId> = mesh.face_ids().collect();
        assert_eq!(v_ids.len(), 4);
        assert_eq!(f_ids.len(), 4);
        assert_eq!(mesh.halfedge_count(), 12);
        (
            mesh,
            [v_ids[0], v_ids[1], v_ids[2], v_ids[3]],
            [f_ids[0], f_ids[1], f_ids[2], f_ids[3]],
        )
    }

    #[test]
    fn extrude_region_single_face_of_tetrahedron() {
        // 从闭合四面体中挤出 1 个面 → 3 条边界边，3 个边界顶点
        let (mut mesh, _v, faces) = build_tetrahedron();
        assert!(validate_mesh(&mesh).is_ok());

        let v_before = mesh.vertex_count();
        let f_before = mesh.face_count();
        let he_before = mesh.halfedge_count();

        let offset = [0.0, 0.0, 0.5];
        let new_faces = extrude_region(&mut mesh, &[faces[0]], offset).expect("单面区域挤出应成功");

        // 3 条边界边 → 6 个侧面三角形
        assert_eq!(new_faces.len(), 6);
        // 3 个边界顶点 → +3 顶点
        assert_eq!(mesh.vertex_count(), v_before + 3);
        // +6 侧面（原 4 面复用为顶面）
        assert_eq!(mesh.face_count(), f_before + 6);
        // 半边：3 边界边×(2 h_top + 2 diag) + 3 顶点×2 垂直 = 12 + 6 = 18
        assert_eq!(mesh.halfedge_count(), he_before + 18);

        // 完整拓扑校验
        let errors = crate::validate::validate_topology(&mesh);
        assert!(errors.is_empty(), "挤出后拓扑应一致：{:?}", errors);

        // Euler 示性数：原闭合四面体 χ=2，挤出后仍为闭合体 χ=2
        let chi = mesh.vertex_count() as i64 - (mesh.halfedge_count() / 2) as i64
            + mesh.face_count() as i64;
        assert_eq!(chi, 2, "挤出后 Euler 示性数应为 2");

        // 闭合体不应有边界半边
        let boundary_count = mesh
            .halfedge_ids()
            .filter(|h| {
                mesh.get_halfedge(*h)
                    .map(|he| he.face.is_none())
                    .unwrap_or(false)
            })
            .count();
        assert_eq!(boundary_count, 0, "挤出后应为闭合体");
    }

    #[test]
    fn extrude_region_two_adjacent_faces() {
        // 挤出四面体的 2 个相邻面（共享边为内部边，不产生侧面）
        let (mut mesh, _v, faces) = build_tetrahedron();
        assert!(validate_mesh(&mesh).is_ok());

        let v_before = mesh.vertex_count();
        let f_before = mesh.face_count();
        let he_before = mesh.halfedge_count();

        // offset 选 [0.5,0.5,0.5] 避免与边界边 DA=(0,0,-1) 平行
        let offset = [0.5, 0.5, 0.5];
        // faces[0] = (A,C,B), faces[1] = (A,B,D)，共享边 AB
        let new_faces = extrude_region(&mut mesh, &[faces[0], faces[1]], offset)
            .expect("双相邻面区域挤出应成功");

        // 4 条边界边 → 8 个侧面三角形
        assert_eq!(new_faces.len(), 8);
        // 4 个边界顶点 → +4 顶点
        assert_eq!(mesh.vertex_count(), v_before + 4);
        // +8 侧面
        assert_eq!(mesh.face_count(), f_before + 8);
        // 半边：4 边界边×4 + 4 顶点×2 = 16 + 8 = 24
        assert_eq!(mesh.halfedge_count(), he_before + 24);

        let errors = crate::validate::validate_topology(&mesh);
        assert!(errors.is_empty(), "挤出后拓扑应一致：{:?}", errors);

        let chi = mesh.vertex_count() as i64 - (mesh.halfedge_count() / 2) as i64
            + mesh.face_count() as i64;
        assert_eq!(chi, 2, "挤出后 Euler 示性数应为 2");

        // 共享边 AB 不应产生侧面（内部边）
        // 验证：new_faces 数量 = 8 = 4 条边界边 × 2，说明共享边未产生侧面
    }

    #[test]
    fn extrude_region_three_faces_leaving_cap() {
        // 挤出四面体的 3 个面，剩余 1 个面成"盖子"
        // 顶点 A 在 3 个面上 → 内部顶点；B/C/D 各缺 1 个面 → 边界顶点
        let (mut mesh, v, faces) = build_tetrahedron();
        assert!(validate_mesh(&mesh).is_ok());
        let [a, _b, _c, _d] = v;

        let v_before = mesh.vertex_count();
        let f_before = mesh.face_count();
        let he_before = mesh.halfedge_count();

        let offset = [0.0, 0.0, 0.5];
        // faces[0]=F1, faces[1]=F2, faces[2]=F3 → 剩余 faces[3]=F4 为盖子
        let new_faces = extrude_region(&mut mesh, &[faces[0], faces[1], faces[2]], offset)
            .expect("三面区域挤出应成功");

        // 3 条边界边（F4 的三条边）→ 6 个侧面三角形
        assert_eq!(new_faces.len(), 6);
        // 3 个边界顶点 (B,C,D) → +3 顶点；A 为内部顶点，原地平移
        assert_eq!(mesh.vertex_count(), v_before + 3);
        // +6 侧面
        assert_eq!(mesh.face_count(), f_before + 6);
        // 半边：3 边界边×4 + 3 顶点×2 = 12 + 6 = 18
        assert_eq!(mesh.halfedge_count(), he_before + 18);

        let errors = crate::validate::validate_topology(&mesh);
        assert!(errors.is_empty(), "挤出后拓扑应一致：{:?}", errors);

        let chi = mesh.vertex_count() as i64 - (mesh.halfedge_count() / 2) as i64
            + mesh.face_count() as i64;
        assert_eq!(chi, 2, "挤出后 Euler 示性数应为 2");

        // 验证内部顶点 A 被平移（位置 += offset）
        let pos_a = mesh.get_vertex(a).unwrap().position;
        assert!(
            (pos_a[2] - 0.5).abs() < 1e-12,
            "内部顶点 A 应被平移到 z=0.5"
        );

        // 剩余面 F4 仍存在（盖子）
        assert!(mesh.contains_face(faces[3]));
    }

    #[test]
    fn extrude_region_zero_offset_returns_error() {
        let (mut mesh, _v, faces) = build_tetrahedron();
        let err = extrude_region(&mut mesh, &[faces[0]], [0.0, 0.0, 0.0]).unwrap_err();
        assert_eq!(err, TopologyError::DegenerateTriangle);
    }

    #[test]
    fn extrude_region_empty_faces_returns_error() {
        let (mut mesh, _v, _faces) = build_tetrahedron();
        let err = extrude_region(&mut mesh, &[], [0.0, 0.0, 1.0]).unwrap_err();
        match err {
            TopologyError::Inconsistent(_) => {}
            other => panic!("期望 Inconsistent 错误，得到 {:?}", other),
        }
    }

    #[test]
    fn extrude_region_degenerate_side_returns_error() {
        // offset 平行于某条边界边会使侧面退化
        let (mut mesh, _v, faces) = build_tetrahedron();
        // F1 = (A,C,B)，边 AC 在 y 轴方向，边 CB 在 (−1,1,0) 方向，边 BA 在 (−1,0,0) 方向
        // offset 沿 y 轴 → 边 AC 与 offset 平行 → 侧面退化
        let err = extrude_region(&mut mesh, &[faces[0]], [0.0, 1.0, 0.0]).unwrap_err();
        assert_eq!(err, TopologyError::DegenerateTriangle);
    }

    // ---------- add_triangle 测试 ----------

    #[test]
    fn add_triangle_to_empty_mesh() {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));

        let f = add_triangle(&mut mesh, v0, v1, v2).expect("add_triangle 应成功");
        assert!(mesh.contains_face(f));
        assert_eq!(mesh.face_count(), 1);
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.halfedge_count(), 6); // 3 面内 + 3 边界 twin
        assert!(validate_mesh(&mesh).is_ok());
    }

    #[test]
    fn add_two_adjacent_triangles() {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));
        let v3 = mesh.add_vertex(Vertex::new([1.0, -1.0, 0.0]));

        let _f1 = add_triangle(&mut mesh, v0, v1, v2).expect("F1");
        let _f2 = add_triangle(&mut mesh, v1, v0, v3).expect("F2");

        assert_eq!(mesh.face_count(), 2);
        assert_eq!(mesh.vertex_count(), 4);
        assert!(validate_mesh(&mesh).is_ok());

        // 共享边 v0-v1 应有内部 twin 配对
        // 总共 10 半边（6 面内 + 4 边界 twin）
        assert_eq!(mesh.halfedge_count(), 10);
    }

    #[test]
    fn add_degenerate_triangle_fails() {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([0.0; 3]));
        let v1 = mesh.add_vertex(Vertex::new([1.0; 3]));

        let err = add_triangle(&mut mesh, v0, v1, v0).unwrap_err();
        assert_eq!(err, TopologyError::DegenerateTriangle);
    }

    #[test]
    fn add_triangle_with_invalid_vertex_fails() {
        let mut mesh = MeshStorage::new();
        let v = mesh.add_vertex(Vertex::new([0.0; 3]));
        let bad = VertexId::default();

        let err = add_triangle(&mut mesh, v, bad, v).unwrap_err();
        assert!(matches!(
            err,
            TopologyError::DegenerateTriangle | TopologyError::Inconsistent(_)
        ));
    }

    #[test]
    fn add_triangle_preserves_existing_edge() {
        // 先手动建一个三角形，再用 add_triangle 加一个共享边的
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));

        // 用 add_triangle 建第一个三角形
        let _f1 = add_triangle(&mut mesh, v0, v1, v2).expect("F1");
        assert_eq!(mesh.face_count(), 1);

        // 再加一个共享 v0-v1 边的三角形
        let v3 = mesh.add_vertex(Vertex::new([1.0, -1.0, 0.0]));
        let _f2 = add_triangle(&mut mesh, v1, v0, v3).expect("F2");

        assert_eq!(mesh.face_count(), 2);
        assert!(validate_mesh(&mesh).is_ok());
        // 4 边界边（v1-v2, v2-v0, v0-v3, v3-v1）+ 1 内部边(v0-v1)
        // = 10 半边
        assert_eq!(mesh.halfedge_count(), 10);
    }
}
