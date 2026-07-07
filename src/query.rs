//! 链式查询 DSL 模块
//!
//! 灵感来自 SMesh 和 OpenMesh 的链式查询接口。为 [`VertexId`]、[`HalfEdgeId`]
//! 和 [`FaceId`] 提供一组返回 [`MeshQuery<T>`] 的方法，支持\textbf{延迟链式调用}：
//! 每个方法返回一个新的 `MeshQuery`，捕获前序查询的闭包，在 `.run(&mesh)` 时
//! 一次性求值整条链。
//!
//! ## 设计要点
//! - `MeshQuery<T>` 内部存储 `Box<dyn FnOnce(&MeshStorage) -> Option<T>>`，
//!   构造期零查询、零分配（仅一次 `Box` 分配）；求值期按链顺序逐跳执行。
//! - 链中任一环节返回 `None` 则\textbf{短路}，后续环节不执行，最终返回 `None`。
//! - 所有方法对无效 / 已删除 ID 返回 `None`，不 panic。
//!
//! ## 示例
//! ```ignore
//! use halfedge::query::MeshQuery;
//!
//! // 找到从 v0 到 v1 的半边，绕 v0 CW 旋转到相邻面，取目标顶点
//! let neighbor = v0.halfedge_to(v1).cw_rotated().dst_vert().run(&mesh);
//! ```

use crate::ids::{FaceId, HalfEdgeId, VertexId};
use crate::storage::MeshStorage;

// ============================================================
// MeshQuery<T> 核心结构
// ============================================================

/// 延迟查询容器：存储一个将在 `.run()` 时执行的闭包。
///
/// 每个链式方法消费 `self`，将前序闭包包装进新闭包，返回新的 `MeshQuery`。
/// 链中任一环节返回 `None` 则短路。
///
/// `#[must_use]` 提醒调用者必须消费查询结果（如通过 `.run(&mesh)`），
/// 避免静默丢弃查询链导致无操作。
#[must_use]
pub struct MeshQuery<T> {
    #[allow(clippy::type_complexity)]
    f: Box<dyn FnOnce(&MeshStorage) -> Option<T>>,
}

impl<T: 'static> MeshQuery<T> {
    /// 从闭包构造查询。通常不直接调用，而是通过 `VertexId` / `HalfEdgeId` 的方法间接构造。
    pub fn new<F>(f: F) -> Self
    where
        F: FnOnce(&MeshStorage) -> Option<T> + 'static,
    {
        Self { f: Box::new(f) }
    }

    /// 执行查询链，返回结果。消费 `self`。
    pub fn run(self, mesh: &MeshStorage) -> Option<T> {
        (self.f)(mesh)
    }

    /// 内部组合子：在前序查询结果上应用进一步操作。
    fn then<F, U>(self, f: F) -> MeshQuery<U>
    where
        F: FnOnce(&MeshStorage, T) -> Option<U> + 'static,
        U: 'static,
    {
        let prev = self.f;
        MeshQuery::new(move |mesh| {
            let val = prev(mesh)?;
            f(mesh, val)
        })
    }
}

// ============================================================
// 内部辅助函数：半边操作（&MeshStorage, HalfEdgeId）-> Option<T>
// ============================================================

fn he_twin(mesh: &MeshStorage, he: HalfEdgeId) -> Option<HalfEdgeId> {
    mesh.get_halfedge(he).and_then(|h| h.twin)
}

fn he_next(mesh: &MeshStorage, he: HalfEdgeId) -> Option<HalfEdgeId> {
    mesh.get_halfedge(he).and_then(|h| h.next)
}

fn he_prev(mesh: &MeshStorage, he: HalfEdgeId) -> Option<HalfEdgeId> {
    mesh.get_halfedge(he).and_then(|h| h.prev)
}

fn he_face(mesh: &MeshStorage, he: HalfEdgeId) -> Option<FaceId> {
    mesh.get_halfedge(he).and_then(|h| h.face)
}

/// 源顶点（origin）= twin.vertex
fn he_src_vert(mesh: &MeshStorage, he: HalfEdgeId) -> Option<VertexId> {
    mesh.get_halfedge(he)
        .and_then(|h| h.twin)
        .and_then(|t| mesh.get_halfedge(t))
        .map(|t| t.vertex)
}

/// 目标顶点（tip）= he.vertex
fn he_dst_vert(mesh: &MeshStorage, he: HalfEdgeId) -> Option<VertexId> {
    mesh.get_halfedge(he).map(|h| h.vertex)
}

/// CW 旋转（绕 origin 顺时针）= twin.next
fn he_cw_rotated(mesh: &MeshStorage, he: HalfEdgeId) -> Option<HalfEdgeId> {
    mesh.get_halfedge(he)
        .and_then(|h| h.twin)
        .and_then(|t| mesh.get_halfedge(t))
        .and_then(|h| h.next)
}

/// CCW 旋转（绕 origin 逆时针）= prev.twin
fn he_ccw_rotated(mesh: &MeshStorage, he: HalfEdgeId) -> Option<HalfEdgeId> {
    mesh.get_halfedge(he)
        .and_then(|h| h.prev)
        .and_then(|p| mesh.get_halfedge(p))
        .and_then(|h| h.twin)
}

// ============================================================
// 内部辅助函数：顶点操作
// ============================================================

fn vertex_halfedge(mesh: &MeshStorage, v: VertexId) -> Option<HalfEdgeId> {
    mesh.get_vertex(v).and_then(|vt| vt.halfedge)
}

/// 遍历 v 的 outgoing 环，找到 tip == target 的那条半边。
/// 复用 traversal::VertexRingLazy，正确处理闭合环与开链。
fn vertex_halfedge_to(mesh: &MeshStorage, v: VertexId, target: VertexId) -> Option<HalfEdgeId> {
    crate::traversal::VertexRingLazy::new(mesh, v).find(|&he| {
        mesh.get_halfedge(he)
            .map(|h| h.vertex == target)
            .unwrap_or(false)
    })
}

// ============================================================
// HalfEdgeId 直接方法：每个返回 MeshQuery<T>
// ============================================================

impl HalfEdgeId {
    /// 取 twin 半边。
    pub fn twin(self) -> MeshQuery<HalfEdgeId> {
        MeshQuery::new(move |mesh| he_twin(mesh, self))
    }

    /// 取同面 next 半边。
    pub fn next(self) -> MeshQuery<HalfEdgeId> {
        MeshQuery::new(move |mesh| he_next(mesh, self))
    }

    /// 取同面 prev 半边。
    pub fn prev(self) -> MeshQuery<HalfEdgeId> {
        MeshQuery::new(move |mesh| he_prev(mesh, self))
    }

    /// 取所属面。边界半边返回 `None`。
    pub fn face(self) -> MeshQuery<FaceId> {
        MeshQuery::new(move |mesh| he_face(mesh, self))
    }

    /// 取源顶点（origin = twin.vertex）。
    pub fn src_vert(self) -> MeshQuery<VertexId> {
        MeshQuery::new(move |mesh| he_src_vert(mesh, self))
    }

    /// 取目标顶点（tip = he.vertex）。
    pub fn dst_vert(self) -> MeshQuery<VertexId> {
        MeshQuery::new(move |mesh| he_dst_vert(mesh, self))
    }

    /// 绕 origin 顺时针旋转到相邻面的 outgoing 半边（twin.next）。
    pub fn cw_rotated(self) -> MeshQuery<HalfEdgeId> {
        MeshQuery::new(move |mesh| he_cw_rotated(mesh, self))
    }

    /// 绕 origin 逆时针旋转到相邻面的 outgoing 半边（prev.twin）。
    pub fn ccw_rotated(self) -> MeshQuery<HalfEdgeId> {
        MeshQuery::new(move |mesh| he_ccw_rotated(mesh, self))
    }
}

// ============================================================
// VertexId 直接方法
// ============================================================

impl VertexId {
    /// 取 v 的 outgoing 半边入口。
    pub fn halfedge(self) -> MeshQuery<HalfEdgeId> {
        MeshQuery::new(move |mesh| vertex_halfedge(mesh, self))
    }

    /// 遍历 outgoing 环，找到 tip == target 的那条半边。
    pub fn halfedge_to(self, target: VertexId) -> MeshQuery<HalfEdgeId> {
        MeshQuery::new(move |mesh| vertex_halfedge_to(mesh, self, target))
    }
}

// ============================================================
// FaceId 直接方法
// ============================================================

/// FaceId::halfedge 用内部辅助
fn face_halfedge(mesh: &MeshStorage, f: FaceId) -> Option<HalfEdgeId> {
    mesh.get_face(f).and_then(|ft| ft.halfedge)
}

impl FaceId {
    /// 取面的入口半边。
    pub fn halfedge(self) -> MeshQuery<HalfEdgeId> {
        MeshQuery::new(move |mesh| face_halfedge(mesh, self))
    }
}

// ============================================================
// MeshQuery<HalfEdgeId> 链式方法
// ============================================================

impl MeshQuery<HalfEdgeId> {
    /// 取当前半边的 twin 半边（反向边）。
    pub fn twin(self) -> MeshQuery<HalfEdgeId> {
        self.then(he_twin)
    }

    /// 取当前半边的 next 半边（沿面顺次）。
    pub fn next(self) -> MeshQuery<HalfEdgeId> {
        self.then(he_next)
    }

    /// 取当前半边的 prev 半边（沿面逆顺）。
    pub fn prev(self) -> MeshQuery<HalfEdgeId> {
        self.then(he_prev)
    }

    /// 取当前半边所属的面。
    pub fn face(self) -> MeshQuery<FaceId> {
        self.then(he_face)
    }

    /// 取当前半边的源顶点（origin）。
    pub fn src_vert(self) -> MeshQuery<VertexId> {
        self.then(he_src_vert)
    }

    /// 取当前半边的目标顶点（destination / tip）。
    pub fn dst_vert(self) -> MeshQuery<VertexId> {
        self.then(he_dst_vert)
    }

    /// 取当前半边绕源顶点的 CW 旋转（twin.prev）。
    pub fn cw_rotated(self) -> MeshQuery<HalfEdgeId> {
        self.then(he_cw_rotated)
    }

    /// 取当前半边绕源顶点的 CCW 旋转（prev.twin），等效 `HalfEdgeId::ccw_rotated`。
    pub fn ccw_rotated(self) -> MeshQuery<HalfEdgeId> {
        self.then(he_ccw_rotated)
    }
}

// ============================================================
// MeshQuery<VertexId> 链式方法
// ============================================================

impl MeshQuery<VertexId> {
    /// 取顶点出发的任一 outgoing 半边。
    pub fn halfedge(self) -> MeshQuery<HalfEdgeId> {
        self.then(vertex_halfedge)
    }

    /// 取从当前顶点到目标顶点 `target` 的半边。
    pub fn halfedge_to(self, target: VertexId) -> MeshQuery<HalfEdgeId> {
        self.then(move |mesh, v| vertex_halfedge_to(mesh, v, target))
    }
}

// ============================================================
// MeshQuery<FaceId> 链式方法
// ============================================================

impl MeshQuery<FaceId> {
    /// 取面的入口半边。
    pub fn halfedge(self) -> MeshQuery<HalfEdgeId> {
        self.then(face_halfedge)
    }
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{Face, HalfEdge, MeshStorage, Vertex};

    // ---------- 测试夹具（与 traversal.rs 一致） ----------

    /// 单个三角面片：
    /// ```text
    ///        v2
    ///        ▲
    ///       /│
    ///   h2 / │ t1
    ///     /  │
    ///   v0───┼───▶ v1
    ///     \  │
    ///   t2 \ │ h0
    ///        ▼
    /// ```
    /// h0: v0→v1, h1: v1→v2, h2: v2→v0（面 F，CCW）
    /// t0..t2: 对应 twin（边界，无 next/prev/face）
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

    /// 3 三角形闭合扇形：中心 c 被 3 个面环绕（内部顶点）。
    /// ```
    /// F1 = c→v0→v1→c, F2 = c→v1→v2→c, F3 = c→v2→v0→c
    /// a_i: c→v_{i}, b_i: v_{i}→v_{i+1} (外边界), c_i: v_{i+1}→c
    /// ```
    fn build_closed_fan() -> (
        MeshStorage,
        VertexId,        // center
        [VertexId; 3],   // outer verts
        [HalfEdgeId; 9], // [a1,a2,a3, b1,b2,b3, c1,c2,c3]
        [FaceId; 3],
    ) {
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

        (
            mesh,
            c,
            [v0, v1, v2],
            [a1, a2, a3, b1, b2, b3, c1, c2, c3],
            [f1, f2, f3],
        )
    }

    /// 两三角形拼成四边形，共享边 v0-v1。
    /// F1 = v0→v1→v2→v0, F2 = v1→v0→v3→v1
    fn build_two_triangles() -> (
        MeshStorage,
        [VertexId; 4],
        [HalfEdgeId; 10], // [h0,h1,h2, g0,g1,g2, t1,t2,t_g1,t_g2]
        [FaceId; 2],
    ) {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));
        let v3 = mesh.add_vertex(Vertex::new([1.0, -1.0, 0.0])); // F2 在共享边下方，几何 CCW

        let h0 = mesh.add_halfedge(HalfEdge::new(v1)); // v0→v1
        let h1 = mesh.add_halfedge(HalfEdge::new(v2)); // v1→v2
        let h2 = mesh.add_halfedge(HalfEdge::new(v0)); // v2→v0
        let g0 = mesh.add_halfedge(HalfEdge::new(v0)); // v1→v0 (twin of h0)
        let g1 = mesh.add_halfedge(HalfEdge::new(v3)); // v0→v3
        let g2 = mesh.add_halfedge(HalfEdge::new(v1)); // v3→v1
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
            [f1, f2],
        )
    }

    // ---------- 基本查询：VertexId ----------

    #[test]
    fn vertex_halfedge_query() {
        let (mesh, v, he, _f) = build_triangle();
        let [v0, _v1, _v2] = v;
        let [h0, _h1, _h2, _t0, _t1, _t2] = he;
        // v0.halfedge = h0
        assert_eq!(v0.halfedge().run(&mesh), Some(h0));
    }

    #[test]
    fn vertex_halfedge_to_finds_correct_edge() {
        let (mesh, v, he, _f) = build_triangle();
        let [v0, v1, v2] = v;
        let [h0, _h1, _h2, _t0, _t1, t2] = he;
        // v0→v1 = h0, v0→v2 = t2
        assert_eq!(v0.halfedge_to(v1).run(&mesh), Some(h0));
        assert_eq!(v0.halfedge_to(v2).run(&mesh), Some(t2));
        // 不存在的邻居
        let fake_v = VertexId::default();
        assert_eq!(v0.halfedge_to(fake_v).run(&mesh), None);
    }

    // ---------- 基本查询：HalfEdgeId ----------

    #[test]
    fn halfedge_twin_query() {
        let (mesh, _v, he, _f) = build_triangle();
        let [h0, h1, h2, t0, t1, t2] = he;
        assert_eq!(h0.twin().run(&mesh), Some(t0));
        assert_eq!(t0.twin().run(&mesh), Some(h0));
        assert_eq!(h1.twin().run(&mesh), Some(t1));
        assert_eq!(h2.twin().run(&mesh), Some(t2));
    }

    #[test]
    fn halfedge_next_prev_query() {
        let (mesh, _v, he, _f) = build_triangle();
        let [h0, h1, h2, _t0, _t1, _t2] = he;
        // 面内 CCW 环：h0→h1→h2→h0
        assert_eq!(h0.next().run(&mesh), Some(h1));
        assert_eq!(h1.next().run(&mesh), Some(h2));
        assert_eq!(h2.next().run(&mesh), Some(h0));
        assert_eq!(h0.prev().run(&mesh), Some(h2));
        assert_eq!(h1.prev().run(&mesh), Some(h0));
        assert_eq!(h2.prev().run(&mesh), Some(h1));
    }

    #[test]
    fn halfedge_face_query() {
        let (mesh, _v, he, f) = build_triangle();
        let [h0, h1, h2, t0, t1, t2] = he;
        // 面内半边 → Some(f)
        assert_eq!(h0.face().run(&mesh), Some(f));
        assert_eq!(h1.face().run(&mesh), Some(f));
        assert_eq!(h2.face().run(&mesh), Some(f));
        // 边界 twin → None
        assert_eq!(t0.face().run(&mesh), None);
        assert_eq!(t1.face().run(&mesh), None);
        assert_eq!(t2.face().run(&mesh), None);
    }

    #[test]
    fn halfedge_src_dst_vert_query() {
        let (mesh, v, he, _f) = build_triangle();
        let [v0, v1, v2] = v;
        let [h0, h1, _h2, t0, _t1, _t2] = he;
        // h0: v0→v1, src=v0, dst=v1
        assert_eq!(h0.src_vert().run(&mesh), Some(v0));
        assert_eq!(h0.dst_vert().run(&mesh), Some(v1));
        // t0: v1→v0, src=v1, dst=v0
        assert_eq!(t0.src_vert().run(&mesh), Some(v1));
        assert_eq!(t0.dst_vert().run(&mesh), Some(v0));
        // h1: v1→v2
        assert_eq!(h1.src_vert().run(&mesh), Some(v1));
        assert_eq!(h1.dst_vert().run(&mesh), Some(v2));
    }

    #[test]
    fn halfedge_rotation_on_closed_fan() {
        let (mesh, c, outer, he, _faces) = build_closed_fan();
        let [_v0, v1, v2] = outer;
        let [a1, a2, a3, _b1, _b2, _b3, _c1, _c2, _c3] = he;
        // a1: c→v0
        // cw_rotated(a1) = twin(a1).next = c3.next = a3 (c→v2)
        assert_eq!(a1.cw_rotated().run(&mesh), Some(a3));
        // ccw_rotated(a1) = prev(a1).twin = c1.twin = a2 (c→v1)
        assert_eq!(a1.ccw_rotated().run(&mesh), Some(a2));
        // 验证旋转后目标顶点
        assert_eq!(a1.cw_rotated().dst_vert().run(&mesh), Some(v2));
        assert_eq!(a1.ccw_rotated().dst_vert().run(&mesh), Some(v1));
        // 连续 CW 旋转应回到起点（闭合环）
        assert_eq!(
            a1.cw_rotated().cw_rotated().cw_rotated().run(&mesh),
            Some(a1)
        );
        // 连续 CCW 旋转
        assert_eq!(
            a1.ccw_rotated().ccw_rotated().ccw_rotated().run(&mesh),
            Some(a1)
        );
        // 中心顶点 c 的 halfedge
        assert_eq!(c.halfedge().run(&mesh), Some(a1));
    }

    #[test]
    fn halfedge_cw_rotated_boundary_returns_none() {
        let (mesh, _v, he, _f) = build_triangle();
        let [h0, _h1, _h2, _t0, _t1, _t2] = he;
        // h0: v0→v1, twin=t0 (边界), t0.next=None → cw_rotated=None
        assert_eq!(h0.cw_rotated().run(&mesh), None);
        // ccw_rotated(h0) = prev(h0).twin = h2.twin = t2
        // t2 是边界半边，但返回值是 t2 本身（Some），不是 None
        // 因为 ccw_rotated = prev.twin，prev(h0)=h2, twin(h2)=t2 → Some(t2)
    }

    // ---------- 链式查询 ----------

    #[test]
    fn chain_user_example_on_two_triangles() {
        let (mesh, v, he, _f) = build_two_triangles();
        let [v0, v1, _v2, v3] = v;
        let [h0, _h1, _h2, _g0, _g1, _g2, _t1, _t2, _t_g1, _t_g2] = he;
        // v0.halfedge_to(v1) = h0 (v0→v1, 共享边)
        assert_eq!(v0.halfedge_to(v1).run(&mesh), Some(h0));
        // h0.cw_rotated() = twin(h0).next = g0.next = g1 (v0→v3)
        // g1.dst_vert() = v3
        let result = v0.halfedge_to(v1).cw_rotated().dst_vert().run(&mesh);
        assert_eq!(result, Some(v3));
    }

    #[test]
    fn chain_closed_fan_rotations() {
        let (mesh, c, outer, _he, _faces) = build_closed_fan();
        let [v0, v1, v2] = outer;
        // c.halfedge_to(v0) = a1 (c→v0)
        // a1.ccw_rotated() = a2 (c→v1)
        // a2.dst_vert() = v1
        assert_eq!(
            c.halfedge_to(v0).ccw_rotated().dst_vert().run(&mesh),
            Some(v1)
        );
        // c.halfedge_to(v0) = a1
        // a1.cw_rotated() = a3 (c→v2)
        // a3.dst_vert() = v2
        assert_eq!(
            c.halfedge_to(v0).cw_rotated().dst_vert().run(&mesh),
            Some(v2)
        );
    }

    #[test]
    fn chain_traverses_face_boundary() {
        let (mesh, v, he, _f) = build_triangle();
        let [v0, _v1, v2] = v;
        let [h0, _h1, _h2, _t0, _t1, _t2] = he;
        // h0.next().next().next() 应回到 h0（闭合面环）
        assert_eq!(h0.next().next().next().run(&mesh), Some(h0));
        // h0.next().dst_vert() = h1.vertex = v2
        assert_eq!(h0.next().dst_vert().run(&mesh), Some(v2));
        // h0.next().next().dst_vert() = h2.vertex = v0
        assert_eq!(h0.next().next().dst_vert().run(&mesh), Some(v0));
    }

    #[test]
    fn chain_src_dst_roundtrip() {
        let (mesh, v, he, _f) = build_triangle();
        let [_v0, v1, _v2] = v;
        let [h0, _h1, _h2, _t0, _t1, _t2] = he;
        // h0.src_vert().halfedge_to(v1) 应找回 h0
        // （src_vert = v0, v0.halfedge_to(v1) = h0）
        assert_eq!(h0.src_vert().halfedge_to(v1).run(&mesh), Some(h0));
        // h0.dst_vert().halfedge() = v1.halfedge = h1
        // 这验证了从 tip 出发取 outgoing 半边
        let result = h0.dst_vert().halfedge().run(&mesh);
        assert!(result.is_some(), "dst_vert().halfedge() 应返回 Some");
    }

    #[test]
    fn chain_short_circuits_on_none() {
        let (mesh, _v, he, _f) = build_triangle();
        let [h0, _h1, _h2, _t0, _t1, _t2] = he;
        // h0.cw_rotated() = None (边界)
        // .cw_rotated() 再次调用 → None（短路）
        // .dst_vert() → None（短路）
        assert_eq!(h0.cw_rotated().cw_rotated().dst_vert().run(&mesh), None);
    }

    // ---------- 无效 / 已删除 ID ----------

    #[test]
    fn invalid_ids_return_none() {
        let mut mesh = MeshStorage::new();
        let v = mesh.add_vertex(Vertex::new([0.0; 3])); // 孤立顶点，无 halfedge
        let he = mesh.add_halfedge(HalfEdge::new(v)); // 无 twin/next/prev/face

        // 孤立顶点
        assert_eq!(v.halfedge().run(&mesh), None);
        assert_eq!(v.halfedge_to(v).run(&mesh), None);

        // 未链接半边
        assert_eq!(he.twin().run(&mesh), None);
        assert_eq!(he.next().run(&mesh), None);
        assert_eq!(he.prev().run(&mesh), None);
        assert_eq!(he.face().run(&mesh), None);
        // he.vertex = v, 但 twin = None → src_vert = None
        assert_eq!(he.src_vert().run(&mesh), None);
        // he.vertex = v → dst_vert = Some(v)
        assert_eq!(he.dst_vert().run(&mesh), Some(v));
        assert_eq!(he.cw_rotated().run(&mesh), None);
        assert_eq!(he.ccw_rotated().run(&mesh), None);

        // 已删除的 ID
        let removed_v = mesh.add_vertex(Vertex::new([1.0; 3]));
        mesh.remove_vertex(removed_v);
        assert_eq!(removed_v.halfedge().run(&mesh), None);
        assert_eq!(removed_v.halfedge_to(removed_v).run(&mesh), None);

        let removed_he = mesh.add_halfedge(HalfEdge::new(v));
        mesh.remove_halfedge(removed_he);
        assert_eq!(removed_he.twin().run(&mesh), None);
        assert_eq!(removed_he.next().run(&mesh), None);
        assert_eq!(removed_he.dst_vert().run(&mesh), None);

        // 默认 ID（未初始化）
        let default_v = VertexId::default();
        let default_he = HalfEdgeId::default();
        assert_eq!(default_v.halfedge().run(&mesh), None);
        assert_eq!(default_he.twin().run(&mesh), None);
    }

    #[test]
    fn chain_on_invalid_ids_short_circuits() {
        let mesh = MeshStorage::new();
        let default_v = VertexId::default();
        // 整条链都应短路返回 None
        assert_eq!(
            default_v
                .halfedge_to(default_v)
                .cw_rotated()
                .dst_vert()
                .run(&mesh),
            None
        );
    }

    // ---------- FaceId 查询 ----------

    #[test]
    fn face_halfedge_query() {
        let (mesh, _v, _he, f) = build_triangle();
        assert!(f.halfedge().run(&mesh).is_some());
        assert!(mesh.contains_halfedge(f.halfedge().run(&mesh).unwrap()));
    }

    #[test]
    fn face_chained_query() {
        let (mesh, v, _he, f) = build_triangle();
        let [_v0, v1, v2] = v;
        // f.halfedge().dst_vert() → v1
        assert_eq!(f.halfedge().dst_vert().run(&mesh), Some(v1));
        // f.halfedge().next().dst_vert() → v2
        assert_eq!(f.halfedge().next().dst_vert().run(&mesh), Some(v2));
    }

    #[test]
    fn face_query_on_invalid_face_returns_none() {
        let mesh = MeshStorage::new();
        let f = FaceId::default();
        assert_eq!(f.halfedge().run(&mesh), None);
        assert_eq!(f.halfedge().dst_vert().run(&mesh), None);
    }

    #[test]
    fn meshquery_face_chain_halfedge() {
        let (mesh, _v, _he, f) = build_triangle();
        let q: MeshQuery<FaceId> =
            MeshQuery::new(move |m| if m.contains_face(f) { Some(f) } else { None });
        assert!(q.halfedge().run(&mesh).is_some());
    }
}
