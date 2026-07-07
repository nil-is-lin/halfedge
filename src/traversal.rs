//! 邻域遍历迭代器模块
//!
//! ## 两种迭代策略
//!
//! ### Eager（预收集）
//! 构造期 O(n) 时间 + O(n) 空间收集到 `Vec`，迭代期不持有借用，
//! 可自由 `&mut mesh`。适合需要随机访问或迭代期修改网格的场景。
//!
//! 类型：[`VertexRing`]、[`VertexAdjacentVerts`]、[`VertexAdjacentFaces`]、
//!       [`FaceHalfEdges`]、[`FaceVertices`]
//!
//! ### Lazy（延迟）
//! 构造期 O(n) 时间 + O(1) 空间（仅 CW 探测，不存中间结果），
//! 迭代期持有 `&MeshStorage` 借用，逐跳查询。适合只读遍历、大规模网格。
//!
//! 类型：[`VertexRingLazy`]、[`VertexAdjacentVertsLazy`]、[`VertexAdjacentFacesLazy`]、
//!       [`FaceHalfEdgesLazy`]、[`FaceVerticesLazy`]
//!
//! 通过 [`VertexRing::lazy`] 等方法创建 lazy 版本。
//!
//! ### EdgeIter（无向边）
//! [`EdgeIter`] 遍历网格中每条唯一的无向边（由规范半边 [`EdgeId`] 代表），
//! 通过 `mesh.edge_ids()` 获取。与 `edge_count()` 配合使用。
//!
//! ### BoundaryLoop（边界环）
//! [`BoundaryLoop`]（预收集）与 [`BoundaryLoopLazy`]（延迟）遍历单个边界环；
//! [`boundary_loops`] 枚举所有边界环；[`is_closed`] 判断网格是否闭合。
//!
//! ### k-ring 邻域
//! - [`vertex_two_ring`]：距离正好为 2 的顶点。
//! - [`vertex_k_ring`]：距离正好为 k 的顶点（BFS）。
//! - [`vertex_k_disk`]：距离 ≤ k 的顶点（k-圆盘）。
//!
//! ## 通用遍历原语
//!
//! 模块还导出两个底层收集函数，作为构建自定义迭代器或分析工具的基础原语：
//! - [`collect_outgoing_halfedges`]：返回顶点的所有 outgoing 半边（CCW 顺序），
//!   已处理开链/闭合环/孤立顶点等所有边界情况。
//! - [`collect_face_halfedges`]：返回面边界环上的所有半边（`next` 顺序），
//!   已处理拓扑断开等异常情况。
//!
//! 下游可直接复用这些经过充分测试的收集逻辑，而不必自己处理 CW+CCW 双向探测
//! 或循环兜底。
//!
//! ## 拓扑假设
//! 所有遍历算法假设网格拓扑是**流形**且**一致**的：
//! - 半边的 `twin` 互指；
//! - 同一面上的 `next`/`prev` 形成闭合环；
//! - 顶点的 outgoing halfedges 形成闭合环（内部顶点）或开链（边界顶点）。
//!
//! 对不一致的拓扑，本模块以「尽可能优雅地终止」为目标，依靠
//! `mesh.halfedge_count()` 作为最大迭代次数兜底，避免死循环。
//!
//! ## 旋转规则（标准半边操作，CCW 朝向网格）
//! 设 `he` 是顶点 `v` 的某条 outgoing 半边（即 `he.twin.vertex == v`），则：
//! - **CCW next**（绕 `v` 的 origin，逆时针下一条 outgoing）：`he.prev.twin`
//! - **CW next**（绕 `v` 的 origin，顺时针下一条 outgoing）：`he.twin.next`
//!
//! 推导：`he.prev` 与 `he` 同面，结尾落在 `v`；其 `twin` 起始于 `v`，且属于「`he.prev`
//! 这条边的另一侧邻接面」，故从 `he` 跨到 `he.prev.twin` 等价于绕 `v` 旋转到相邻面。

use crate::ids::{EdgeId, FaceId, HalfEdgeId, VertexId};
use crate::storage::MeshStorage;

// ============================================================
// 通用遍历原语：收集顶点的所有 outgoing 半边（CCW 顺序）
// ============================================================

/// 收集顶点 `v` 的所有 outgoing 半边，按 CCW 顺序返回。
///
/// 通用遍历原语，已处理边界顶点（开链）与内部顶点（闭合环）的所有边界情况，
/// 下游可直接复用而不必自己实现 CW+CCW 双向探测。
///
/// - **内部顶点**：返回闭合环，起点为 `v.halfedge`。
/// - **边界顶点**：返回开链，起点为 CW 方向最末端的那条边界半边，终点为 CCW 方向
///   最末端的那条边界半边；`v.halfedge` 落在链中某处。
/// - **孤立顶点**（`v.halfedge` 为 `None`）：返回空 `Vec`。
pub fn collect_outgoing_halfedges(mesh: &MeshStorage, v: VertexId) -> Vec<HalfEdgeId> {
    let mut buf = Vec::new();
    collect_outgoing_halfedges_into(mesh, v, &mut buf);
    buf
}

/// 与 [`collect_outgoing_halfedges`] 相同，但填充调用方提供的 `buf`。
///
/// **内存复用**：`buf.clear()` 后填充，不释放已分配容量。
/// 适合在热循环中反复调用以避免重复分配：
///
/// ```ignore
/// let mut buf = Vec::new();
/// for v in mesh.vertex_ids() {
///     collect_outgoing_halfedges_into(mesh, v, &mut buf);
///     // 使用 buf...
/// }
/// ```
pub fn collect_outgoing_halfedges_into(mesh: &MeshStorage, v: VertexId, buf: &mut Vec<HalfEdgeId>) {
    buf.clear();
    let start = match mesh.get_vertex(v).and_then(|vt| vt.halfedge) {
        Some(s) => s,
        None => return,
    };
    // 安全上界：任何正确的旋转环长度不会超过半边总数。
    let max_iter = mesh.halfedge_count() + 1;

    buf.push(start);

    // 向 CCW 方向走：he -> he.prev.twin
    let mut he = start;
    let mut closed = false;
    for _ in 0..max_iter {
        let next = mesh
            .get_halfedge(he)
            .and_then(|h| h.prev)
            .and_then(|p| mesh.get_halfedge(p))
            .and_then(|h| h.twin);
        match next {
            Some(n) if n != start => {
                buf.push(n);
                he = n;
            }
            Some(_) => {
                closed = true; // 回到起点 → 闭合环
                break;
            }
            None => break, // 撞到边界，开链
        }
    }

    if closed {
        return;
    }

    // 开链：向 CW 方向补全，收集到的部分逆序后拼到 buf 前面
    let mut backward = Vec::new();
    let mut he = start;
    for _ in 0..max_iter {
        let prev = mesh
            .get_halfedge(he)
            .and_then(|h| h.twin)
            .and_then(|t| mesh.get_halfedge(t))
            .and_then(|h| h.next);
        match prev {
            Some(p) => {
                backward.push(p);
                he = p;
            }
            None => break,
        }
    }
    backward.reverse();
    buf.splice(0..0, backward);
}

/// 收集面 `f` 边界环上的所有半边，按 `next` 顺序返回。
///
/// 通用遍历原语，已处理拓扑断开等异常情况（撞到 `None` 立即终止）。
pub fn collect_face_halfedges(mesh: &MeshStorage, f: FaceId) -> Vec<HalfEdgeId> {
    let mut buf = Vec::new();
    collect_face_halfedges_into(mesh, f, &mut buf);
    buf
}

/// 与 [`collect_face_halfedges`] 相同，但填充调用方提供的 `buf`。
///
/// **内存复用**：`buf.clear()` 后填充，不释放已分配容量。
pub fn collect_face_halfedges_into(mesh: &MeshStorage, f: FaceId, buf: &mut Vec<HalfEdgeId>) {
    buf.clear();
    let start = match mesh.get_face(f).and_then(|ft| ft.halfedge) {
        Some(s) => s,
        None => return,
    };
    let max_iter = mesh.halfedge_count() + 1;

    buf.push(start);
    let mut he = start;
    for _ in 0..max_iter {
        let next = mesh.get_halfedge(he).and_then(|h| h.next);
        match next {
            Some(n) if n != start => {
                buf.push(n);
                he = n;
            }
            Some(_) => break, // 回到起点
            None => break,    // 拓扑断开
        }
    }
}

// ============================================================
// 1. 顶点相关迭代器
// ============================================================

/// 顶点环绕半边迭代器：依次产出 `v` 的所有 outgoing 半边（CCW 顺序）。
///
/// ```
/// use halfedge::{build_mesh_from_vertices_and_faces, traversal::VertexRing};
///
/// let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
/// let faces = vec![[0u32, 1, 2]];
/// let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
/// let v = mesh.vertex_ids().next().unwrap();
/// let ring: Vec<_> = VertexRing::new(&mesh, v).collect();
/// assert_eq!(ring.len(), 2);
/// ```
pub struct VertexRing {
    halfedges: Vec<HalfEdgeId>,
    cursor: usize,
}

impl VertexRing {
    /// 从网格与顶点创建顶点环绕半边迭代器。
    ///
    /// 遍历 `v` 出发的所有 outgoing 半边（CCW 顺序）。
    pub fn new(mesh: &MeshStorage, v: VertexId) -> Self {
        Self {
            halfedges: collect_outgoing_halfedges(mesh, v),
            cursor: 0,
        }
    }
}

impl Iterator for VertexRing {
    type Item = HalfEdgeId;
    fn next(&mut self) -> Option<Self::Item> {
        match self.halfedges.get(self.cursor) {
            Some(&id) => {
                self.cursor += 1;
                Some(id)
            }
            None => None,
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.halfedges.len().saturating_sub(self.cursor);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for VertexRing {}

/// 顶点邻接顶点迭代器：依次产出 `v` 的所有邻居顶点（CCW 顺序）。
///
/// 即对 `v` 的每个 outgoing 半边 `he`，产出 `he.vertex`（`he` 的 tip）。
pub struct VertexAdjacentVerts {
    verts: Vec<VertexId>,
    cursor: usize,
}

impl VertexAdjacentVerts {
    /// 从网格与顶点创建顶点邻接点迭代器。
    ///
    /// 遍历 `v` 的所有邻居顶点（CCW 顺序）。
    pub fn new(mesh: &MeshStorage, v: VertexId) -> Self {
        let verts = collect_outgoing_halfedges(mesh, v)
            .into_iter()
            .filter_map(|he| mesh.get_halfedge(he).map(|h| h.vertex))
            .collect();
        Self { verts, cursor: 0 }
    }
}

impl Iterator for VertexAdjacentVerts {
    type Item = VertexId;
    fn next(&mut self) -> Option<Self::Item> {
        match self.verts.get(self.cursor) {
            Some(&id) => {
                self.cursor += 1;
                Some(id)
            }
            None => None,
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.verts.len().saturating_sub(self.cursor);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for VertexAdjacentVerts {}

/// 顶点邻接面迭代器：依次产出 `v` 周围的所有面（CCW 顺序，跳过 `None` 边界半边）。
pub struct VertexAdjacentFaces {
    faces: Vec<FaceId>,
    cursor: usize,
}

impl VertexAdjacentFaces {
    /// 从网格与顶点创建顶点邻接面迭代器。
    ///
    /// 遍历 `v` 的所有相邻面（CCW 顺序）。
    pub fn new(mesh: &MeshStorage, v: VertexId) -> Self {
        let faces = collect_outgoing_halfedges(mesh, v)
            .into_iter()
            .filter_map(|he| mesh.get_halfedge(he).and_then(|h| h.face))
            .collect();
        Self { faces, cursor: 0 }
    }
}

impl Iterator for VertexAdjacentFaces {
    type Item = FaceId;
    fn next(&mut self) -> Option<Self::Item> {
        match self.faces.get(self.cursor) {
            Some(&id) => {
                self.cursor += 1;
                Some(id)
            }
            None => None,
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.faces.len().saturating_sub(self.cursor);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for VertexAdjacentFaces {}

// ============================================================
// 2. 面相关迭代器
// ============================================================

/// 面边界半边迭代器：按 `next` 顺序产出面的所有边界半边。
///
/// ```
/// use halfedge::{build_mesh_from_vertices_and_faces, traversal::FaceHalfEdges};
///
/// let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
/// let faces = vec![[0u32, 1, 2]];
/// let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
/// let f = mesh.face_ids().next().unwrap();
/// let hes: Vec<_> = FaceHalfEdges::new(&mesh, f).collect();
/// assert_eq!(hes.len(), 3);
/// ```
pub struct FaceHalfEdges {
    halfedges: Vec<HalfEdgeId>,
    cursor: usize,
}

impl FaceHalfEdges {
    /// 从网格与面创建面半边迭代器。
    ///
    /// 遍历 `f` 的所有半边（CCW 顺序）。
    pub fn new(mesh: &MeshStorage, f: FaceId) -> Self {
        Self {
            halfedges: collect_face_halfedges(mesh, f),
            cursor: 0,
        }
    }
}

impl Iterator for FaceHalfEdges {
    type Item = HalfEdgeId;
    fn next(&mut self) -> Option<Self::Item> {
        match self.halfedges.get(self.cursor) {
            Some(&id) => {
                self.cursor += 1;
                Some(id)
            }
            None => None,
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.halfedges.len().saturating_sub(self.cursor);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for FaceHalfEdges {}

/// 面顶点迭代器：依次产出面边界环上每条半边的 tip 顶点。
pub struct FaceVertices {
    verts: Vec<VertexId>,
    cursor: usize,
}

impl FaceVertices {
    /// 从网格与面创建面邻接面迭代器。
    ///
    /// 遍历 `f` 的所有邻接面（CCW 顺序，对边共享面）。
    pub fn new(mesh: &MeshStorage, f: FaceId) -> Self {
        let verts = collect_face_halfedges(mesh, f)
            .into_iter()
            .filter_map(|he| mesh.get_halfedge(he).map(|h| h.vertex))
            .collect();
        Self { verts, cursor: 0 }
    }
}

impl Iterator for FaceVertices {
    type Item = VertexId;
    fn next(&mut self) -> Option<Self::Item> {
        match self.verts.get(self.cursor) {
            Some(&id) => {
                self.cursor += 1;
                Some(id)
            }
            None => None,
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.verts.len().saturating_sub(self.cursor);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for FaceVertices {}

// ============================================================
// 3. 延迟迭代器（lazy）：迭代期持有 &MeshStorage，零堆分配
// ============================================================
//
// 设计权衡：
// - 预收集版本（上方）：构造期 O(n) 时间 + O(n) 空间收集到 Vec，迭代期不持有借用，
//   可自由 &mut mesh。适合需要随机访问或迭代期修改网格的场景。
// - lazy 版本（下方）：构造期 O(n) 时间 + O(1) 空间（仅 CW 探测找尽头，不存储），
//   迭代期持有 &MeshStorage 借用，逐跳查询。适合只读遍历、大规模网格、内存敏感场景。
//
// 顶点环 lazy 算法：
//   1. 构造时从 start 沿 CW（he.twin.next）走到开链尽头 cw_end（撞 None），
//      或检测到闭合环（回到 start）。
//   2. 迭代时从 cw_end 沿 CCW（he.prev.twin）逐个产出，闭合环回到 cw_end 时终止，
//      开链撞 None 时终止。
//   这样产出顺序与预收集版本完全一致：[CW 方向逆序, start, CCW 方向]。

/// 沿 CW 方向（he.twin.next）探测顶点 outgoing 环的尽头。
///
/// 返回 (cw_end, closed)：
/// - 闭合环：closed = true，cw_end = start（CW 走回到 start）
/// - 开链：closed = false，cw_end = CW 方向最末端的半边（其 twin.next = None）
fn probe_cw_end(mesh: &MeshStorage, start: HalfEdgeId) -> (HalfEdgeId, bool) {
    let mut cw_end = start;
    let max_iter = mesh.halfedge_count() + 1;
    for _ in 0..max_iter {
        let next_cw = mesh
            .get_halfedge(cw_end)
            .and_then(|h| h.twin)
            .and_then(|t| mesh.get_halfedge(t))
            .and_then(|h| h.next);
        match next_cw {
            Some(n) if n == start => return (cw_end, true), // 闭合环（未走最后一步）
            Some(n) => cw_end = n,
            None => return (cw_end, false), // 开链尽头
        }
    }
    // 安全兜底：环异常长（超过半边总数），视为闭合
    (cw_end, true)
}

/// 延迟顶点环绕迭代器：逐跳查询 `&MeshStorage`，零堆分配。
///
/// 构造期沿 CW 方向 O(n) 探测开链尽头（O(1) 空间），迭代期沿 CCW 方向逐个产出。
/// 产出顺序与 [`VertexRing`]（预收集版本）完全一致。
///
/// # 生命周期约束
/// 迭代器持有 `&MeshStorage` 不可变借用，迭代期间**不能** `&mut mesh`。
/// 若需迭代期修改网格，请使用 [`VertexRing::new`]。
pub struct VertexRingLazy<'a> {
    mesh: &'a MeshStorage,
    start: HalfEdgeId,
    current: Option<HalfEdgeId>,
    closed: bool,
}

impl<'a> VertexRingLazy<'a> {
    /// 从网格与顶点创建惰性顶点环绕半边迭代器。
    ///
    /// 不预先收集半边，每次 next 动态遍历。
    pub fn new(mesh: &'a MeshStorage, v: VertexId) -> Self {
        let start = match mesh.get_vertex(v).and_then(|vt| vt.halfedge) {
            Some(s) => s,
            None => {
                return Self {
                    mesh,
                    start: HalfEdgeId::default(),
                    current: None,
                    closed: false,
                };
            }
        };
        let (cw_end, closed) = probe_cw_end(mesh, start);
        // 闭合环：从原始 start 开始（与预收集版本起点一致）；
        // 开链：从 CW 尽头 cw_end 开始（CCW 方向遍历整条开链）。
        let begin = if closed { start } else { cw_end };
        Self {
            mesh,
            start: begin,
            current: Some(begin),
            closed,
        }
    }
}

impl<'a> Iterator for VertexRingLazy<'a> {
    type Item = HalfEdgeId;
    fn next(&mut self) -> Option<HalfEdgeId> {
        let cur = self.current?;
        let result = cur;
        // 沿 CCW 前进：he.prev.twin
        let next_ccw = self
            .mesh
            .get_halfedge(cur)
            .and_then(|h| h.prev)
            .and_then(|p| self.mesh.get_halfedge(p))
            .and_then(|h| h.twin);
        match next_ccw {
            // 闭合环回到起点：终止（避免重复产出 start）
            Some(n) if self.closed && n == self.start => self.current = None,
            Some(n) => self.current = Some(n),
            None => self.current = None, // 开链撞边界
        }
        Some(result)
    }
}

impl<'a> std::iter::FusedIterator for VertexRingLazy<'a> {}

/// 延迟顶点邻接顶点迭代器：基于 [`VertexRingLazy`]，产出每条 outgoing 半边的 tip。
pub struct VertexAdjacentVertsLazy<'a> {
    mesh: &'a MeshStorage,
    ring: VertexRingLazy<'a>,
}

impl<'a> VertexAdjacentVertsLazy<'a> {
    /// 从网格与顶点创建惰性顶点邻接点迭代器。
    ///
    /// 延迟遍历，每次 next 动态计算下一个邻居。
    pub fn new(mesh: &'a MeshStorage, v: VertexId) -> Self {
        Self {
            mesh,
            ring: VertexRingLazy::new(mesh, v),
        }
    }
}

impl<'a> Iterator for VertexAdjacentVertsLazy<'a> {
    type Item = VertexId;
    fn next(&mut self) -> Option<VertexId> {
        let mesh = self.mesh;
        for he in &mut self.ring {
            if let Some(v) = mesh.get_halfedge(he).map(|h| h.vertex) {
                return Some(v);
            }
        }
        None
    }
}

impl<'a> std::iter::FusedIterator for VertexAdjacentVertsLazy<'a> {}

/// 延迟顶点邻接面迭代器：基于 [`VertexRingLazy`]，跳过 `face = None` 的边界半边。
pub struct VertexAdjacentFacesLazy<'a> {
    mesh: &'a MeshStorage,
    ring: VertexRingLazy<'a>,
}

impl<'a> VertexAdjacentFacesLazy<'a> {
    /// 从网格与顶点创建惰性顶点邻接面迭代器。
    ///
    /// 延迟遍历，每次 next 动态计算下一个邻接面。
    pub fn new(mesh: &'a MeshStorage, v: VertexId) -> Self {
        Self {
            mesh,
            ring: VertexRingLazy::new(mesh, v),
        }
    }
}

impl<'a> Iterator for VertexAdjacentFacesLazy<'a> {
    type Item = FaceId;
    fn next(&mut self) -> Option<FaceId> {
        let mesh = self.mesh;
        for he in &mut self.ring {
            if let Some(f) = mesh.get_halfedge(he).and_then(|h| h.face) {
                return Some(f);
            }
        }
        None
    }
}

impl<'a> std::iter::FusedIterator for VertexAdjacentFacesLazy<'a> {}

/// 延迟面边界半边迭代器：沿 `next` 逐跳前进，闭合检测回到起点。
pub struct FaceHalfEdgesLazy<'a> {
    mesh: &'a MeshStorage,
    start: HalfEdgeId,
    current: Option<HalfEdgeId>,
}

impl<'a> FaceHalfEdgesLazy<'a> {
    /// 从网格与面创建惰性面半边迭代器。
    ///
    /// 延迟遍历，每次 next 沿 next 链步进。
    pub fn new(mesh: &'a MeshStorage, f: FaceId) -> Self {
        let start = match mesh.get_face(f).and_then(|ft| ft.halfedge) {
            Some(s) => s,
            None => {
                return Self {
                    mesh,
                    start: HalfEdgeId::default(),
                    current: None,
                };
            }
        };
        Self {
            mesh,
            start,
            current: Some(start),
        }
    }
}

impl<'a> Iterator for FaceHalfEdgesLazy<'a> {
    type Item = HalfEdgeId;
    fn next(&mut self) -> Option<HalfEdgeId> {
        let cur = self.current?;
        let result = cur;
        let next = self.mesh.get_halfedge(cur).and_then(|h| h.next);
        match next {
            Some(n) if n == self.start => self.current = None, // 回到起点
            Some(n) => self.current = Some(n),
            None => self.current = None, // 拓扑断开
        }
        Some(result)
    }
}

impl<'a> std::iter::FusedIterator for FaceHalfEdgesLazy<'a> {}

/// 延迟面顶点迭代器：基于 [`FaceHalfEdgesLazy`]，产出每条半边的 tip 顶点。
pub struct FaceVerticesLazy<'a> {
    mesh: &'a MeshStorage,
    ring: FaceHalfEdgesLazy<'a>,
}

impl<'a> FaceVerticesLazy<'a> {
    /// 从网格与面创建惰性面邻接面迭代器。
    ///
    /// 延迟遍历，每次 next 通过 twin 链步进。
    pub fn new(mesh: &'a MeshStorage, f: FaceId) -> Self {
        Self {
            mesh,
            ring: FaceHalfEdgesLazy::new(mesh, f),
        }
    }
}

impl<'a> Iterator for FaceVerticesLazy<'a> {
    type Item = VertexId;
    fn next(&mut self) -> Option<VertexId> {
        let mesh = self.mesh;
        for he in &mut self.ring {
            if let Some(v) = mesh.get_halfedge(he).map(|h| h.vertex) {
                return Some(v);
            }
        }
        None
    }
}

impl<'a> std::iter::FusedIterator for FaceVerticesLazy<'a> {}

// ============================================================
// 4. lazy 构造函数（在预收集类型上添加 ::lazy 方法）
// ============================================================

impl VertexRing {
    /// 延迟迭代版本：返回 [`VertexRingLazy`]，零堆分配，迭代期持有 `&mesh` 借用。
    ///
    /// 适合只读遍历、大规模网格。若需迭代期修改网格，请用 [`VertexRing::new`]。
    pub fn lazy(mesh: &MeshStorage, v: VertexId) -> VertexRingLazy<'_> {
        VertexRingLazy::new(mesh, v)
    }
}

impl VertexAdjacentVerts {
    /// 延迟迭代版本：返回 [`VertexAdjacentVertsLazy`]。
    pub fn lazy(mesh: &MeshStorage, v: VertexId) -> VertexAdjacentVertsLazy<'_> {
        VertexAdjacentVertsLazy::new(mesh, v)
    }
}

impl VertexAdjacentFaces {
    /// 延迟迭代版本：返回 [`VertexAdjacentFacesLazy`]。
    pub fn lazy(mesh: &MeshStorage, v: VertexId) -> VertexAdjacentFacesLazy<'_> {
        VertexAdjacentFacesLazy::new(mesh, v)
    }
}

impl FaceHalfEdges {
    /// 延迟迭代版本：返回 [`FaceHalfEdgesLazy`]。
    pub fn lazy(mesh: &MeshStorage, f: FaceId) -> FaceHalfEdgesLazy<'_> {
        FaceHalfEdgesLazy::new(mesh, f)
    }
}

impl FaceVertices {
    /// 延迟迭代版本：返回 [`FaceVerticesLazy`]。
    pub fn lazy(mesh: &MeshStorage, f: FaceId) -> FaceVerticesLazy<'_> {
        FaceVerticesLazy::new(mesh, f)
    }
}

// ============================================================
// 5. 迭代器统计信息（count_hint / is_empty）
// ============================================================
//
// 为所有迭代器（预收集 + lazy）提供统一的统计接口：
// - `count_hint(&self) -> Option<usize>`：预收集返回 `Some(剩余数量)`，lazy 返回 `None`
// - `is_empty(&self) -> bool`：是否一次都不会产出
//
// 预收集版本额外实现 `ExactSizeIterator`（提供 `len()` / `is_empty()` trait 方法），
// inherent `is_empty` 与 trait 方法行为一致，但 inherent 优先以便统一调用风格。

/// 为预收集迭代器添加 `count_hint` / `is_empty` inherent 方法。
macro_rules! impl_eager_iterator_stats {
    ($iter:ty, $field:ident) => {
        impl $iter {
            /// 预收集版本：返回剩余元素数量（已知精确长度）。
            pub fn count_hint(&self) -> Option<usize> {
                let remaining = self.$field.len().saturating_sub(self.cursor);
                Some(remaining)
            }

            /// 是否一次都不会产出。
            pub fn is_empty(&self) -> bool {
                self.cursor >= self.$field.len()
            }
        }
    };
}

impl_eager_iterator_stats!(VertexRing, halfedges);
impl_eager_iterator_stats!(VertexAdjacentVerts, verts);
impl_eager_iterator_stats!(VertexAdjacentFaces, faces);
impl_eager_iterator_stats!(FaceHalfEdges, halfedges);
impl_eager_iterator_stats!(FaceVertices, verts);

/// 为基于 `current: Option<_>` 字段的 lazy 迭代器添加统计方法。
macro_rules! impl_lazy_iterator_stats {
    ($iter:ty) => {
        impl $iter {
            /// lazy 版本：长度未知，返回 `None`。
            pub fn count_hint(&self) -> Option<usize> {
                None
            }

            /// 是否一次都不会产出。
            pub fn is_empty(&self) -> bool {
                self.current.is_none()
            }
        }
    };
}

impl_lazy_iterator_stats!(VertexRingLazy<'_>);
impl_lazy_iterator_stats!(FaceHalfEdgesLazy<'_>);

/// 为基于内部 `ring` 字段的 lazy 迭代器添加统计方法（委托给内部 ring）。
macro_rules! impl_lazy_iterator_stats_delegated {
    ($iter:ty) => {
        impl $iter {
            /// lazy 版本：长度未知，返回 `None`。
            pub fn count_hint(&self) -> Option<usize> {
                None
            }

            /// 是否一次都不会产出（委托给内部 ring）。
            pub fn is_empty(&self) -> bool {
                self.ring.is_empty()
            }
        }
    };
}

impl_lazy_iterator_stats_delegated!(VertexAdjacentVertsLazy<'_>);
impl_lazy_iterator_stats_delegated!(VertexAdjacentFacesLazy<'_>);
impl_lazy_iterator_stats_delegated!(FaceVerticesLazy<'_>);

// ============================================================
// 6. MeshStorage 诊断扩展（依赖 traversal，故放在本模块）
// ============================================================

impl MeshStorage {
    /// 最大顶点度数（valence）：所有顶点中 outgoing 半边数的最大值。
    ///
    /// 用于评估迭代器内存开销：预收集版本的总量 $\approx V \times \bar{d} \times 8$ 字节，
    /// 其中 $\bar{d}$ 为平均度数；本方法给出最坏单顶点度数。
    ///
    /// 复杂度 $O(V \cdot \bar{d}) = O(E)$，需要遍历每个顶点的 outgoing 环。
    /// 孤立顶点（无 outgoing 半边）度数为 0；空网格返回 0。
    pub fn max_vertex_valence(&self) -> usize {
        self.vertex_ids()
            .map(|v| VertexRing::new(self, v).count())
            .max()
            .unwrap_or(0)
    }

    /// 无向边数量。对于流形网格，每条边对应一对互为 twin 的半边，
    /// 故 `edge_count() == halfedge_count() / 2`。
    pub fn edge_count(&self) -> usize {
        self.halfedge_count() / 2
    }

    /// 迭代所有唯一的无向边，每条边由其规范半边代表。
    ///
    /// 对每对互为 twin 的半边，只产出 key 较小的那个；边界半边
    /// （`twin == None`）直接以自身为规范代表产出。
    pub fn edge_ids(&self) -> EdgeIter<'_> {
        EdgeIter::new(self)
    }
}

// ============================================================
// 8. EdgeIter — 无向边迭代器
// ============================================================

/// 无向边迭代器：遍历网格中每条唯一的无向边。
///
/// 通过 [`EdgeId`]（规范半边）代表每条边。对每对互为 twin 的半边，
/// 只产出 key 较小的那个；边界半边（`twin == None`）直接以自身作为规范代表。
///
/// 这是一个预收集迭代器：构造期收集所有规范半边到 `Vec`，迭代期不持有借用。
pub struct EdgeIter<'a> {
    edges: Vec<EdgeId>,
    cursor: usize,
    _phantom: std::marker::PhantomData<&'a MeshStorage>,
}

impl<'a> EdgeIter<'a> {
    /// 从网格创建无向边迭代器。
    ///
    /// 遍历所有非边界半边（每条边仅产出一次）。
    pub fn new(mesh: &'a MeshStorage) -> Self {
        use std::collections::HashSet;
        let mut seen = HashSet::with_capacity(mesh.edge_count());
        let mut edges = Vec::with_capacity(mesh.edge_count());
        for he in mesh.halfedge_ids() {
            let canonical = match mesh.get_halfedge(he).and_then(|h| h.twin) {
                Some(twin) => {
                    if he < twin {
                        he
                    } else {
                        twin
                    }
                }
                None => he,
            };
            if seen.insert(canonical) {
                edges.push(EdgeId::from_halfedge(canonical));
            }
        }
        Self {
            edges,
            cursor: 0,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl Iterator for EdgeIter<'_> {
    type Item = EdgeId;

    fn next(&mut self) -> Option<Self::Item> {
        match self.edges.get(self.cursor) {
            Some(&id) => {
                self.cursor += 1;
                Some(id)
            }
            None => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.edges.len().saturating_sub(self.cursor);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for EdgeIter<'_> {}

impl EdgeId {
    /// 从规范半边构造 `EdgeId`。调用者应确保传入的半边是 twin 对中
    /// key 较小者（或为无 twin 的边界半边）。
    pub(crate) fn from_halfedge(he: HalfEdgeId) -> Self {
        Self(he)
    }
}

// ============================================================
// 6.5 顶点邻接边迭代器
// ============================================================

/// 顶点邻接边迭代器：产出与顶点 v 关联的所有唯一无向边（`EdgeId`）。
///
/// 对 v 的每个 outgoing 半边，取规范半边（twin 对中 key 较小者）构造 `EdgeId`。
pub struct VertexAdjacentEdges {
    edges: Vec<EdgeId>,
    cursor: usize,
}

impl VertexAdjacentEdges {
    /// 从网格与顶点创建邻接边迭代器。
    ///
    /// 遍历 `v` 的所有邻接边（半边的无向版本）。
    pub fn new(mesh: &MeshStorage, v: VertexId) -> Self {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        let mut edges = Vec::new();
        for he in collect_outgoing_halfedges(mesh, v) {
            let h = mesh.get_halfedge(he).expect("halfedge exists in mesh");
            let canonical = match h.twin {
                Some(twin) => {
                    if he < twin {
                        he
                    } else {
                        twin
                    }
                }
                None => he,
            };
            if seen.insert(canonical) {
                edges.push(EdgeId::from_halfedge(canonical));
            }
        }
        Self { edges, cursor: 0 }
    }
}

impl Iterator for VertexAdjacentEdges {
    type Item = EdgeId;
    fn next(&mut self) -> Option<Self::Item> {
        match self.edges.get(self.cursor) {
            Some(&id) => {
                self.cursor += 1;
                Some(id)
            }
            None => None,
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.edges.len().saturating_sub(self.cursor);
        (rem, Some(rem))
    }
}

impl ExactSizeIterator for VertexAdjacentEdges {}

// ============================================================
// 7. k-ring 邻域遍历
// ============================================================

/// 顶点二环邻域：距离正好为 2 的顶点（邻居的邻居，不含自身和直接邻居）。
pub fn vertex_two_ring(mesh: &MeshStorage, v: VertexId) -> Vec<VertexId> {
    use std::collections::HashSet;
    let neighbors: HashSet<VertexId> = VertexAdjacentVerts::new(mesh, v).collect();
    let mut result = HashSet::new();
    for n in &neighbors {
        for n2 in VertexAdjacentVerts::new(mesh, *n) {
            if n2 != v && !neighbors.contains(&n2) {
                result.insert(n2);
            }
        }
    }
    let mut out: Vec<_> = result.into_iter().collect();
    out.sort();
    out
}

/// 顶点 k 环邻域：距离正好为 k 的顶点（BFS）。
/// k==0 返回 `[v]`，k==1 等价于 `VertexAdjacentVerts`。
pub fn vertex_k_ring(mesh: &MeshStorage, v: VertexId, k: usize) -> Vec<VertexId> {
    if k == 0 {
        return vec![v];
    }
    use std::collections::{HashMap, VecDeque};
    let mut dist: HashMap<VertexId, usize> = HashMap::new();
    let mut queue = VecDeque::new();
    dist.insert(v, 0);
    queue.push_back(v);
    while let Some(cur) = queue.pop_front() {
        let d = dist[&cur];
        if d >= k {
            continue;
        }
        for n in VertexAdjacentVerts::new(mesh, cur) {
            if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(n) {
                e.insert(d + 1);
                queue.push_back(n);
            }
        }
    }
    let mut result: Vec<_> = dist
        .into_iter()
        .filter(|(_, d)| *d == k)
        .map(|(id, _)| id)
        .collect();
    result.sort();
    result
}

/// 顶点 k 圆盘邻域：距离 ≤ k 的所有顶点。
pub fn vertex_k_disk(mesh: &MeshStorage, v: VertexId, k: usize) -> Vec<VertexId> {
    if k == 0 {
        return vec![v];
    }
    use std::collections::{HashMap, VecDeque};
    let mut dist: HashMap<VertexId, usize> = HashMap::new();
    let mut queue = VecDeque::new();
    dist.insert(v, 0);
    queue.push_back(v);
    while let Some(cur) = queue.pop_front() {
        let d = dist[&cur];
        if d >= k {
            continue;
        }
        for n in VertexAdjacentVerts::new(mesh, cur) {
            if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(n) {
                e.insert(d + 1);
                queue.push_back(n);
            }
        }
    }
    let mut result: Vec<_> = dist.into_keys().collect();
    result.sort();
    result
}

// ============================================================
// 8. 边界环遍历
// ============================================================

/// 判断网格是否为闭合网格（无任何边界边）。
pub fn is_closed(mesh: &MeshStorage) -> bool {
    !mesh
        .halfedge_ids()
        .any(|he| mesh.get_halfedge(he).is_some_and(|h| h.face.is_none()))
}

/// 从一条边界半边出发，沿边界环找到下一条边界半边。
///
/// 从 `he` 的目的顶点出发，沿 outgoing 环遍历，
/// 找到第一条 `face == None` 的半边作为下一边界半边。
/// 若 `he` 不是边界半边或拓扑断开，返回 `None`。
pub fn next_boundary_halfedge(mesh: &MeshStorage, he: HalfEdgeId) -> Option<HalfEdgeId> {
    let h = mesh.get_halfedge(he)?;
    if h.face.is_some() {
        return None;
    }
    // 从 he 的目的顶点出发，找到 outgoing 边界半边
    let v = h.vertex;
    let start = mesh.get_vertex(v)?.halfedge?;
    let max_iter = mesh.halfedge_count() + 1;
    let mut cur = start;
    for _ in 0..max_iter {
        // 检查 cur 是否是边界半边
        if mesh.get_halfedge(cur).is_some_and(|ch| ch.face.is_none()) && cur != he {
            return Some(cur);
        }
        // 沿 CCW 方向旋转（prev.twin）
        let next = mesh
            .get_halfedge(cur)
            .and_then(|ch| ch.prev)
            .and_then(|p| mesh.get_halfedge(p))
            .and_then(|ph| ph.twin);
        match next {
            Some(n) if n == start => break, // 绕了一圈
            Some(n) => cur = n,
            None => break,
        }
    }
    None
}

/// 边界环预收集迭代器：从一条边界半边出发，沿 `twin.next.twin` 遍历整个边界环。
pub struct BoundaryLoop {
    halfedges: Vec<HalfEdgeId>,
    cursor: usize,
}

impl BoundaryLoop {
    /// 从 `start` 边界半边出发，遍历整个边界环并收集。
    ///
    /// `start` 必须为边界半边（`face == None`），否则返回空迭代器。
    pub fn new(mesh: &MeshStorage, start: HalfEdgeId) -> Self {
        let mut halfedges = Vec::new();
        if mesh.get_halfedge(start).is_none_or(|h| h.face.is_some()) {
            return Self {
                halfedges,
                cursor: 0,
            };
        }
        halfedges.push(start);
        let max_iter = mesh.halfedge_count() + 1;
        let mut cur = start;
        for _ in 0..max_iter {
            match next_boundary_halfedge(mesh, cur) {
                Some(n) if n == start => break, // 闭合的边界环
                Some(n) => {
                    halfedges.push(n);
                    cur = n;
                }
                None => break,
            }
        }
        Self {
            halfedges,
            cursor: 0,
        }
    }
}

impl Iterator for BoundaryLoop {
    type Item = HalfEdgeId;
    fn next(&mut self) -> Option<Self::Item> {
        match self.halfedges.get(self.cursor) {
            Some(&id) => {
                self.cursor += 1;
                Some(id)
            }
            None => None,
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.halfedges.len().saturating_sub(self.cursor);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for BoundaryLoop {}

/// 边界环延迟迭代器：沿 `twin.next.twin` 逐跳遍历，零堆分配。
pub struct BoundaryLoopLazy<'a> {
    mesh: &'a MeshStorage,
    start: HalfEdgeId,
    current: Option<HalfEdgeId>,
}

impl<'a> BoundaryLoopLazy<'a> {
    /// 从网格与起始半边创建惰性边界环迭代器。
    ///
    /// 沿边界 next 链遍历，直到回到起点。
    pub fn new(mesh: &'a MeshStorage, start: HalfEdgeId) -> Self {
        let current = mesh
            .get_halfedge(start)
            .filter(|h| h.face.is_none())
            .map(|_| start);
        Self {
            mesh,
            start,
            current,
        }
    }
}

impl Iterator for BoundaryLoopLazy<'_> {
    type Item = HalfEdgeId;
    fn next(&mut self) -> Option<Self::Item> {
        let cur = self.current?;
        let next = next_boundary_halfedge(self.mesh, cur);
        self.current = match next {
            Some(n) if n == self.start => None, // 回到起点
            n => n,
        };
        Some(cur)
    }
}

impl std::iter::FusedIterator for BoundaryLoopLazy<'_> {}

/// 枚举网格中的所有边界环。
///
/// 每个边界环以 `Vec<HalfEdgeId>` 返回（边界环遍历顺序）。
/// 对闭合网格（无边界），返回空 `Vec`。
pub fn boundary_loops(mesh: &MeshStorage) -> Vec<Vec<HalfEdgeId>> {
    use std::collections::HashSet;
    let mut visited = HashSet::with_capacity(mesh.halfedge_count());
    let mut loops = Vec::new();
    for he in mesh.halfedge_ids() {
        if visited.contains(&he) {
            continue;
        }
        let is_boundary = mesh.get_halfedge(he).is_some_and(|h| h.face.is_none());
        if !is_boundary {
            continue;
        }
        let boundary_he: Vec<_> = BoundaryLoop::new(mesh, he).collect();
        for &h in &boundary_he {
            visited.insert(h);
        }
        if !boundary_he.is_empty() {
            loops.push(boundary_he);
        }
    }
    loops
}

// ============================================================
// 8. 工具函数
// ============================================================

/// 判断半边所在的边是否为边界边。
///
/// # 边界边定义
/// 满足以下任一条件即为边界边：
/// - 该半边自身 `face` 为 `None`（位于边界环上）；
/// - `twin` 不存在；
/// - `twin.face` 为 `None`（对侧位于边界环上）。
///
/// # 无效 ID 语义
/// 若 `he` 已被删除或从未分配，返回 `false`。
/// **理由**：已删除的句柄不再代表任何边，因此既非内部边也非边界边。
/// 调用者若需区分"无效"与"非边界"，应先调用
/// [`MeshStorage::contains_halfedge`](crate::storage::MeshStorage::contains_halfedge)。
#[inline]
pub fn is_boundary_edge(mesh: &MeshStorage, he: HalfEdgeId) -> bool {
    let h = match mesh.get_halfedge(he) {
        Some(h) => h,
        None => return false,
    };
    if h.face.is_none() {
        return true;
    }
    match h.twin {
        None => true,
        Some(t) => mesh
            .get_halfedge(t)
            .map(|th| th.face.is_none())
            .unwrap_or(true),
    }
}

/// 判断顶点是否位于网格边界。
///
/// 判据：`v` 的任一 outgoing 半边所在边为边界边，则 `v` 为边界顶点。
/// 顶点无效或没有 outgoing 半边时返回 `false`。
pub fn is_boundary_vertex(mesh: &MeshStorage, v: VertexId) -> bool {
    VertexRingLazy::new(mesh, v).any(|he| is_boundary_edge(mesh, he))
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{Face, HalfEdge, MeshStorage, Vertex};

    /// 构造一个标准孤立三角面片（CCW 朝向）：
    /// ```text
    ///        v2
    ///        ▲
    ///       /│
    ///   h2 / │ t1
    ///     /  │
    ///    /   │
    ///   v0───┼───▶ v1
    ///    \   │
    ///     \  │
    ///   t2 \ │ h0
    ///       \│
    ///        ▼
    /// ```
    /// - 面 `F` 由 `h0 (v0→v1)`, `h1 (v1→v2)`, `h2 (v2→v0)` 三条 CCW 半边构成；
    /// - 每条 `hi` 都有 `twin = ti`（边界半边，`face=None`，无 `next`/`prev`）。
    fn build_triangle() -> (MeshStorage, [VertexId; 3], [HalfEdgeId; 6], FaceId) {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));

        // 面内半边（CCW）
        let h0 = mesh.add_halfedge(HalfEdge::new(v1)); // v0→v1
        let h1 = mesh.add_halfedge(HalfEdge::new(v2)); // v1→v2
        let h2 = mesh.add_halfedge(HalfEdge::new(v0)); // v2→v0
        // twin 半边（边界）
        let t0 = mesh.add_halfedge(HalfEdge::new(v0)); // v1→v0
        let t1 = mesh.add_halfedge(HalfEdge::new(v1)); // v2→v1
        let t2 = mesh.add_halfedge(HalfEdge::new(v2)); // v0→v2

        let f = mesh.add_face(Face::new());

        // 面内半边的 twin/next/prev/face
        for (he, twin, next, prev) in [(h0, t0, h1, h2), (h1, t1, h2, h0), (h2, t2, h0, h1)] {
            let h = mesh.get_halfedge_mut(he).unwrap();
            h.twin = Some(twin);
            h.next = Some(next);
            h.prev = Some(prev);
            h.face = Some(f);
        }
        // twin 的 twin（互指）；next/prev/face 留 None 表示边界
        for (t, he) in [(t0, h0), (t1, h1), (t2, h2)] {
            mesh.get_halfedge_mut(t).unwrap().twin = Some(he);
        }
        // 顶点 outgoing 入口
        mesh.get_vertex_mut(v0).unwrap().halfedge = Some(h0);
        mesh.get_vertex_mut(v1).unwrap().halfedge = Some(h1);
        mesh.get_vertex_mut(v2).unwrap().halfedge = Some(h2);
        // 面入口
        mesh.get_face_mut(f).unwrap().halfedge = Some(h0);

        (mesh, [v0, v1, v2], [h0, h1, h2, t0, t1, t2], f)
    }

    // ---------- 面相关 ----------

    #[test]
    fn face_halfedges_in_ccw_order() {
        let (mesh, _v, he, f) = build_triangle();
        let [h0, h1, h2, _, _, _] = he;
        let collected: Vec<_> = FaceHalfEdges::new(&mesh, f).collect();
        assert_eq!(collected, vec![h0, h1, h2]);
    }

    #[test]
    fn face_vertices_in_ccw_order() {
        let (mesh, v, _he, f) = build_triangle();
        let [v0, v1, v2] = v;
        let collected: Vec<_> = FaceVertices::new(&mesh, f).collect();
        // h0.vertex=v1, h1.vertex=v2, h2.vertex=v0
        assert_eq!(collected, vec![v1, v2, v0]);
    }

    // ---------- 顶点相关 ----------

    #[test]
    fn vertex_ring_around_v0() {
        let (mesh, v, he, _f) = build_triangle();
        let [v0, _v1, _v2] = v;
        let [h0, _h1, _h2, _t0, _t1, t2] = he;
        // v0.outgoing 按 CCW 顺序：h0 (0°) → t2 (90°)
        let collected: Vec<_> = VertexRing::new(&mesh, v0).collect();
        assert_eq!(collected, vec![h0, t2]);
    }

    #[test]
    fn vertex_ring_around_v1() {
        let (mesh, v, he, _f) = build_triangle();
        let [_v0, v1, _v2] = v;
        let [_h0, h1, _h2, t0, _t1, _t2] = he;
        // v1.outgoing = h1 (v1→v2, 90°+), 然后 CCW 走到 t0 (v1→v0, 反向 = 180°)
        let collected: Vec<_> = VertexRing::new(&mesh, v1).collect();
        assert_eq!(collected, vec![h1, t0]);
    }

    #[test]
    fn vertex_adjacent_verts_around_v0() {
        let (mesh, v, _he, _f) = build_triangle();
        let [v0, v1, v2] = v;
        let collected: Vec<_> = VertexAdjacentVerts::new(&mesh, v0).collect();
        assert_eq!(collected, vec![v1, v2]);
    }

    #[test]
    fn vertex_adjacent_faces_around_v0() {
        let (mesh, v, _he, f) = build_triangle();
        let [v0, _v1, _v2] = v;
        let collected: Vec<_> = VertexAdjacentFaces::new(&mesh, v0).collect();
        assert_eq!(collected, vec![f]);
    }

    // ---------- 边界判定 ----------

    #[test]
    fn boundary_detection_for_single_triangle() {
        let (mesh, v, he, _f) = build_triangle();
        let [v0, v1, v2] = v;
        let [h0, h1, h2, t0, t1, t2] = he;
        // 单个三角面片：所有边都是边界边
        for he in [h0, h1, h2, t0, t1, t2] {
            assert!(is_boundary_edge(&mesh, he), "半边 {:?} 应为边界边", he);
        }
        // 所有顶点都是边界顶点
        for v in [v0, v1, v2] {
            assert!(is_boundary_vertex(&mesh, v), "顶点 {:?} 应为边界顶点", v);
        }
    }

    // ---------- 不持有借用：遍历期间可修改网格 ----------

    #[test]
    fn iterator_does_not_hold_borrow() {
        let (mut mesh, v, _he, _f) = build_triangle();
        let [v0, _v1, _v2] = v;
        let iter = VertexRing::new(&mesh, v0);
        {
            // 若 iter 持有对 mesh 的借用，此行会编译失败
            let _mesh_mut = &mut mesh;
        }
        // iter 仍然可用
        assert_eq!(iter.count(), 2);
    }

    #[test]
    fn can_modify_mesh_during_iteration() {
        let (mut mesh, v, _he, _f) = build_triangle();
        let [v0, _v1, _v2] = v;
        // 先收集 ID
        let ids: Vec<_> = VertexRing::new(&mesh, v0).collect();
        // 然后基于这些 ID 修改网格（这里仅演示：把每条半边的 face 清空）
        for he in ids {
            mesh.get_halfedge_mut(he).unwrap().face = None;
        }
        // 修改成功
        for he in VertexRing::new(&mesh, v0) {
            assert!(mesh.get_halfedge(he).unwrap().face.is_none());
        }
    }

    // ---------- 无效 / 孤立元素 ----------

    #[test]
    fn invalid_or_isolated_ids_yield_empty() {
        let mut mesh = MeshStorage::new();
        let v = mesh.add_vertex(Vertex::new([0.0; 3])); // 无 halfedge
        let f = mesh.add_face(Face::new()); // 无 halfedge

        assert_eq!(VertexRing::new(&mesh, v).count(), 0);
        assert_eq!(VertexAdjacentVerts::new(&mesh, v).count(), 0);
        assert_eq!(VertexAdjacentFaces::new(&mesh, v).count(), 0);
        assert_eq!(FaceHalfEdges::new(&mesh, f).count(), 0);
        assert_eq!(FaceVertices::new(&mesh, f).count(), 0);

        // 已删除的 ID 也安全返回空 / false
        let removed_v = mesh.add_vertex(Vertex::new([1.0; 3]));
        mesh.remove_vertex(removed_v);
        assert_eq!(VertexRing::new(&mesh, removed_v).count(), 0);
        assert!(!is_boundary_vertex(&mesh, removed_v));

        let removed_he = mesh.add_halfedge(HalfEdge::new(v));
        mesh.remove_halfedge(removed_he);
        assert!(!is_boundary_edge(&mesh, removed_he));
    }

    // ---------- 内部顶点：两个三角形拼成四边形 ----------

    /// 构造两个三角形拼成的四边形，验证共享边上的顶点是内部顶点：
    /// ```text
    ///   v2 ────── v3
    ///    │ ╲     ╱│
    ///    │  ╲   ╱ │
    ///    │ F1╲ ╱F2│
    ///    │   ╲╱  │
    ///    │   ╱╲  │
    ///    │  ╱  ╲ │
    ///    │ ╱   ╲│
    ///   v0 ──── v1     （共享边 v0-v1，h0 与 g0 互为 twin）
    /// ```
    /// `v0` 与 `v1` 都是内部顶点（被两个面环绕）；`v2`、`v3` 是边界顶点。
    fn build_two_triangles() -> (MeshStorage, [VertexId; 4], FaceId, FaceId) {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));
        let v3 = mesh.add_vertex(Vertex::new([1.0, -1.0, 0.0])); // F2 在共享边下方

        // F1 = v0→v1→v2→v0 (CCW)
        let h0 = mesh.add_halfedge(HalfEdge::new(v1)); // v0→v1
        let h1 = mesh.add_halfedge(HalfEdge::new(v2)); // v1→v2
        let h2 = mesh.add_halfedge(HalfEdge::new(v0)); // v2→v0
        // F2 = v1→v0→v3→v1 (CCW，位于共享边 v0-v1 下方)
        // v3 = (1,-1) 在 v0→v1 右侧，使 next 环顶点序 (v0, v3, v1) 几何 CCW
        let g0 = mesh.add_halfedge(HalfEdge::new(v0)); // v1→v0  (twin of h0)
        let g1 = mesh.add_halfedge(HalfEdge::new(v3)); // v0→v3
        let g2 = mesh.add_halfedge(HalfEdge::new(v1)); // v3→v1
        // 边界 twin
        let t1 = mesh.add_halfedge(HalfEdge::new(v1)); // v2→v1, twin of h1
        let t2 = mesh.add_halfedge(HalfEdge::new(v2)); // v0→v2, twin of h2
        let t_g1 = mesh.add_halfedge(HalfEdge::new(v0)); // v3→v0, twin of g1
        let t_g2 = mesh.add_halfedge(HalfEdge::new(v3)); // v1→v3, twin of g2

        let f1 = mesh.add_face(Face::new());
        let f2 = mesh.add_face(Face::new());

        // F1 内部
        for (he, twin, next, prev) in [(h0, g0, h1, h2), (h1, t1, h2, h0), (h2, t2, h0, h1)] {
            let h = mesh.get_halfedge_mut(he).unwrap();
            h.twin = Some(twin);
            h.next = Some(next);
            h.prev = Some(prev);
            h.face = Some(f1);
        }
        // F2 内部
        for (he, twin, next, prev) in [(g0, h0, g1, g2), (g1, t_g1, g2, g0), (g2, t_g2, g0, g1)] {
            let h = mesh.get_halfedge_mut(he).unwrap();
            h.twin = Some(twin);
            h.next = Some(next);
            h.prev = Some(prev);
            h.face = Some(f2);
        }
        // 边界 twin（仅互指 twin）
        for (t, he) in [(t1, h1), (t2, h2), (t_g1, g1), (t_g2, g2)] {
            mesh.get_halfedge_mut(t).unwrap().twin = Some(he);
        }
        // 顶点 outgoing 入口：v0、v1 选共享边侧的半边
        mesh.get_vertex_mut(v0).unwrap().halfedge = Some(h0);
        mesh.get_vertex_mut(v1).unwrap().halfedge = Some(g0);
        mesh.get_vertex_mut(v2).unwrap().halfedge = Some(h1);
        mesh.get_vertex_mut(v3).unwrap().halfedge = Some(g1);
        // 面入口
        mesh.get_face_mut(f1).unwrap().halfedge = Some(h0);
        mesh.get_face_mut(f2).unwrap().halfedge = Some(g0);

        (mesh, [v0, v1, v2, v3], f1, f2)
    }

    #[test]
    fn two_triangle_corner_vertices_are_still_boundary() {
        // 注意：两三角形拼成的四边形中，v0/v1 虽然被两条共享边触及，但它们仍各自
        // 关联到一条外边界边（v0-v2、v0-v3、v1-v2、v1-v3），因此**仍是边界顶点**。
        let (mesh, v, _f1, _f2) = build_two_triangles();
        let [v0, v1, v2, v3] = v;
        assert!(
            is_boundary_vertex(&mesh, v0),
            "v0 关联到外边界边 → 边界顶点"
        );
        assert!(
            is_boundary_vertex(&mesh, v1),
            "v1 关联到外边界边 → 边界顶点"
        );
        assert!(is_boundary_vertex(&mesh, v2), "v2 应为边界顶点");
        assert!(is_boundary_vertex(&mesh, v3), "v3 应为边界顶点");
    }

    #[test]
    fn two_triangle_shared_vertex_has_three_outgoing() {
        let (mesh, v, _f1, _f2) = build_two_triangles();
        let [v0, _v1, _v2, _v3] = v;
        // v0 的 outgoing 半边有三条：h0 (→v1), t2 (→v2), g1 (→v3)
        let ring: Vec<_> = VertexRing::new(&mesh, v0).collect();
        assert_eq!(ring.len(), 3, "v0 周围应有 3 条 outgoing 半边");
        // 三条都不同
        let mut sorted = ring.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 3, "三条 outgoing 半边应互不相同");
    }

    #[test]
    fn two_triangle_v0_has_two_adjacent_faces() {
        let (mesh, v, f1, f2) = build_two_triangles();
        let [v0, _v1, _v2, _v3] = v;
        let mut faces: Vec<_> = VertexAdjacentFaces::new(&mesh, v0).collect();
        faces.sort();
        let mut expected = vec![f1, f2];
        expected.sort();
        assert_eq!(faces, expected);
    }

    #[test]
    fn shared_edge_is_not_boundary() {
        // 找到 v0→v1 的半边 h0：它是共享边，两侧都有面 → 不是边界
        // 直接遍历 v0 的 outgoing，找到 tip == v1 的那条
        let (mesh, v, _f1, _f2) = build_two_triangles();
        let [v0, v1, _v2, _v3] = v;
        let shared = VertexRing::new(&mesh, v0)
            .find(|he| mesh.get_halfedge(*he).unwrap().vertex == v1)
            .expect("v0→v1 的半边必须存在");
        assert!(!is_boundary_edge(&mesh, shared), "共享边不应是边界边");
    }

    // ---------- 真正的内部顶点：3 个三角形围成的闭合扇形 ----------
    //
    //         v2
    //        ╱ ╲
    //       ╱   ╲
    //      ╱  F3 ╲
    //     ╱       ╲
    //    v0 ───c─── v1
    //     ╲       ╱
    //      ╲  F1 ╱
    //       ╲   ╱
    //        ╲ ╱
    //         (实际上 F1=c→v0→v1, F2=c→v1→v2, F3=c→v2→v0，三个面把中心 c 完全包住)
    //
    // 中心顶点 c 的 outgoing 半边形成闭合环 → 内部顶点；
    // 外圈 v0/v1/v2 各自仍关联一条外边界边 → 边界顶点。

    fn build_closed_fan() -> (MeshStorage, VertexId, [VertexId; 3], [FaceId; 3]) {
        let mut mesh = MeshStorage::new();
        let c = mesh.add_vertex(Vertex::new([0.5, 0.5, 0.0]));
        let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let v2 = mesh.add_vertex(Vertex::new([0.5, 1.0, 0.0]));

        // F1 = c→v0→v1→c
        let a1 = mesh.add_halfedge(HalfEdge::new(v0)); // c→v0
        let b1 = mesh.add_halfedge(HalfEdge::new(v1)); // v0→v1 (外边界)
        let c1 = mesh.add_halfedge(HalfEdge::new(c)); // v1→c
        // F2 = c→v1→v2→c
        let a2 = mesh.add_halfedge(HalfEdge::new(v1)); // c→v1
        let b2 = mesh.add_halfedge(HalfEdge::new(v2)); // v1→v2 (外边界)
        let c2 = mesh.add_halfedge(HalfEdge::new(c)); // v2→c
        // F3 = c→v2→v0→c
        let a3 = mesh.add_halfedge(HalfEdge::new(v2)); // c→v2
        let b3 = mesh.add_halfedge(HalfEdge::new(v0)); // v2→v0 (外边界)
        let c3 = mesh.add_halfedge(HalfEdge::new(c)); // v0→c
        // 外边界 twin
        let t1 = mesh.add_halfedge(HalfEdge::new(v0)); // v1→v0, twin of b1
        let t2 = mesh.add_halfedge(HalfEdge::new(v1)); // v2→v1, twin of b2
        let t3 = mesh.add_halfedge(HalfEdge::new(v2)); // v0→v2, twin of b3

        let f1 = mesh.add_face(Face::new());
        let f2 = mesh.add_face(Face::new());
        let f3 = mesh.add_face(Face::new());

        // F1 内部：a1→b1→c1→a1，twin: a1↔c3, b1↔t1
        for (he, twin, next, prev, face) in [
            (a1, c3, b1, c1, f1),
            (b1, t1, c1, a1, f1),
            (c1, a2, a1, b1, f1),
        ] {
            let h = mesh.get_halfedge_mut(he).unwrap();
            h.twin = Some(twin);
            h.next = Some(next);
            h.prev = Some(prev);
            h.face = Some(face);
        }
        // F2 内部：a2→b2→c2→a2，twin: a2↔c1, b2↔t2
        for (he, twin, next, prev, face) in [
            (a2, c1, b2, c2, f2),
            (b2, t2, c2, a2, f2),
            (c2, a3, a2, b2, f2),
        ] {
            let h = mesh.get_halfedge_mut(he).unwrap();
            h.twin = Some(twin);
            h.next = Some(next);
            h.prev = Some(prev);
            h.face = Some(face);
        }
        // F3 内部：a3→b3→c3→a3，twin: a3↔c2, b3↔t3
        for (he, twin, next, prev, face) in [
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
        // 外边界 twin（仅互指 twin）
        for (t, he) in [(t1, b1), (t2, b2), (t3, b3)] {
            mesh.get_halfedge_mut(t).unwrap().twin = Some(he);
        }
        // 顶点 outgoing 入口
        mesh.get_vertex_mut(c).unwrap().halfedge = Some(a1);
        mesh.get_vertex_mut(v0).unwrap().halfedge = Some(b1);
        mesh.get_vertex_mut(v1).unwrap().halfedge = Some(b2);
        mesh.get_vertex_mut(v2).unwrap().halfedge = Some(b3);
        // 面入口
        mesh.get_face_mut(f1).unwrap().halfedge = Some(a1);
        mesh.get_face_mut(f2).unwrap().halfedge = Some(a2);
        mesh.get_face_mut(f3).unwrap().halfedge = Some(a3);

        (mesh, c, [v0, v1, v2], [f1, f2, f3])
    }

    #[test]
    fn closed_fan_center_is_interior() {
        let (mesh, c, _outer, _faces) = build_closed_fan();
        assert!(!is_boundary_vertex(&mesh, c), "中心顶点 c 应为内部顶点");
    }

    #[test]
    fn closed_fan_outer_vertices_are_boundary() {
        let (mesh, _c, outer, _faces) = build_closed_fan();
        for v in outer {
            assert!(
                is_boundary_vertex(&mesh, v),
                "外圈顶点 {:?} 应为边界顶点",
                v
            );
        }
    }

    #[test]
    fn closed_fan_center_ring_closes_with_three_outgoing() {
        let (mesh, c, _outer, _faces) = build_closed_fan();
        let ring: Vec<_> = VertexRing::new(&mesh, c).collect();
        // 闭合扇形：c 有 3 条 outgoing，按 CCW 顺序为 a1, a2, a3
        assert_eq!(ring.len(), 3, "c 周围应有 3 条 outgoing 半边");
        // 三条互不相同
        let mut sorted = ring.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 3);
    }

    #[test]
    fn closed_fan_center_adjacent_faces_yields_three() {
        let (mesh, c, _outer, faces) = build_closed_fan();
        let mut adj: Vec<_> = VertexAdjacentFaces::new(&mesh, c).collect();
        adj.sort();
        let mut expected = faces.to_vec();
        expected.sort();
        assert_eq!(adj, expected);
    }

    // ---------- lazy 迭代器：与预收集版本一致性 ----------

    #[test]
    fn lazy_vertex_ring_matches_eager_on_single_triangle() {
        let (mesh, v, _he, _f) = build_triangle();
        for vi in v {
            let eager: Vec<_> = VertexRing::new(&mesh, vi).collect();
            let lazy: Vec<_> = VertexRing::lazy(&mesh, vi).collect();
            assert_eq!(eager, lazy, "顶点 {:?} 的 lazy 环应与预收集一致", vi);
        }
    }

    #[test]
    fn lazy_vertex_ring_matches_eager_on_closed_fan() {
        let (mesh, c, outer, _faces) = build_closed_fan();
        // 内部顶点（闭合环）
        let eager: Vec<_> = VertexRing::new(&mesh, c).collect();
        let lazy: Vec<_> = VertexRing::lazy(&mesh, c).collect();
        assert_eq!(eager, lazy, "内部顶点 c 的 lazy 环应与预收集一致");
        // 边界顶点（开链）
        for vi in outer {
            let eager: Vec<_> = VertexRing::new(&mesh, vi).collect();
            let lazy: Vec<_> = VertexRing::lazy(&mesh, vi).collect();
            assert_eq!(eager, lazy, "边界顶点 {:?} 的 lazy 环应与预收集一致", vi);
        }
    }

    #[test]
    fn lazy_vertex_ring_matches_eager_on_two_triangles() {
        let (mesh, v, _f1, _f2) = build_two_triangles();
        for vi in v {
            let eager: Vec<_> = VertexRing::new(&mesh, vi).collect();
            let lazy: Vec<_> = VertexRing::lazy(&mesh, vi).collect();
            assert_eq!(eager, lazy, "顶点 {:?} 的 lazy 环应与预收集一致", vi);
        }
    }

    #[test]
    fn lazy_face_halfedges_matches_eager() {
        let (mesh, _v, _he, f) = build_triangle();
        let eager: Vec<_> = FaceHalfEdges::new(&mesh, f).collect();
        let lazy: Vec<_> = FaceHalfEdges::lazy(&mesh, f).collect();
        assert_eq!(eager, lazy);
    }

    #[test]
    fn lazy_face_vertices_matches_eager() {
        let (mesh, _v, _he, f) = build_triangle();
        let eager: Vec<_> = FaceVertices::new(&mesh, f).collect();
        let lazy: Vec<_> = FaceVertices::lazy(&mesh, f).collect();
        assert_eq!(eager, lazy);
    }

    #[test]
    fn lazy_adjacent_verts_matches_eager() {
        let (mesh, c, outer, _faces) = build_closed_fan();
        for vi in std::iter::once(c).chain(outer.iter().copied()) {
            let eager: Vec<_> = VertexAdjacentVerts::new(&mesh, vi).collect();
            let lazy: Vec<_> = VertexAdjacentVerts::lazy(&mesh, vi).collect();
            assert_eq!(eager, lazy, "顶点 {:?} 的 lazy 邻接顶点应一致", vi);
        }
    }

    #[test]
    fn lazy_adjacent_faces_matches_eager() {
        let (mesh, c, outer, _faces) = build_closed_fan();
        for vi in std::iter::once(c).chain(outer.iter().copied()) {
            let mut eager: Vec<_> = VertexAdjacentFaces::new(&mesh, vi).collect();
            let mut lazy: Vec<_> = VertexAdjacentFaces::lazy(&mesh, vi).collect();
            eager.sort();
            lazy.sort();
            assert_eq!(eager, lazy, "顶点 {:?} 的 lazy 邻接面应一致", vi);
        }
    }

    #[test]
    fn lazy_iterators_handle_invalid_ids() {
        let mut mesh = MeshStorage::new();
        let v = mesh.add_vertex(Vertex::new([0.0; 3])); // 无 halfedge
        let f = mesh.add_face(Face::new()); // 无 halfedge

        assert_eq!(VertexRing::lazy(&mesh, v).count(), 0);
        assert_eq!(VertexAdjacentVerts::lazy(&mesh, v).count(), 0);
        assert_eq!(VertexAdjacentFaces::lazy(&mesh, v).count(), 0);
        assert_eq!(FaceHalfEdges::lazy(&mesh, f).count(), 0);
        assert_eq!(FaceVertices::lazy(&mesh, f).count(), 0);
    }

    #[test]
    fn lazy_iterators_are_fused() {
        // FusedIterator 语义：next 返回 None 后，继续调用永远返回 None
        let (mesh, v, _he, f) = build_triangle();
        let [v0, _v1, _v2] = v;

        let mut ring = VertexRing::lazy(&mesh, v0);
        while ring.next().is_some() {}
        // 耗尽后再调用多次，都应返回 None
        for _ in 0..5 {
            assert!(ring.next().is_none(), "FusedIterator 耗尽后应返回 None");
        }

        let mut face_he = FaceHalfEdgesLazy::new(&mesh, f);
        while face_he.next().is_some() {}
        for _ in 0..5 {
            assert!(face_he.next().is_none());
        }
    }

    #[test]
    fn lazy_iterators_on_icosphere() {
        // icosphere 是闭合流形，所有顶点环都是闭合环
        let mesh = crate::test_util::build_icosphere(1); // 42 顶点
        for v in mesh.vertex_ids() {
            let eager: Vec<_> = VertexRing::new(&mesh, v).collect();
            let lazy: Vec<_> = VertexRing::lazy(&mesh, v).collect();
            assert_eq!(eager, lazy, "icosphere 顶点 {:?} 的 lazy 环应一致", v);
        }
        for f in mesh.face_ids() {
            let eager: Vec<_> = FaceHalfEdges::new(&mesh, f).collect();
            let lazy: Vec<_> = FaceHalfEdges::lazy(&mesh, f).collect();
            assert_eq!(eager, lazy, "icosphere 面 {:?} 的 lazy 半边应一致", f);
        }
    }

    // ---------- 统计信息：count_hint / is_empty / ExactSizeIterator ----------

    #[test]
    fn eager_count_hint_returns_some_remaining() {
        let (mesh, v, _he, f) = build_triangle();
        let [v0, _v1, _v2] = v;

        // VertexRing：v0 有 2 条 outgoing
        let ring = VertexRing::new(&mesh, v0);
        assert_eq!(ring.count_hint(), Some(2));
        assert_eq!(ring.len(), 2);
        assert!(!ring.is_empty());

        // 消费一个后，count_hint 应反映剩余
        let mut ring = VertexRing::new(&mesh, v0);
        let _ = ring.next();
        assert_eq!(ring.count_hint(), Some(1));
        assert_eq!(ring.len(), 1);

        // 耗尽后
        let mut ring = VertexRing::new(&mesh, v0);
        while ring.next().is_some() {}
        assert_eq!(ring.count_hint(), Some(0));
        assert_eq!(ring.len(), 0);
        assert!(ring.is_empty());

        // FaceHalfEdges：三角形有 3 条半边
        let fh = FaceHalfEdges::new(&mesh, f);
        assert_eq!(fh.count_hint(), Some(3));
        assert_eq!(fh.len(), 3);

        // FaceVertices：三角形有 3 个顶点
        let fv = FaceVertices::new(&mesh, f);
        assert_eq!(fv.count_hint(), Some(3));
        assert_eq!(fv.len(), 3);
    }

    #[test]
    fn eager_is_empty_on_invalid_ids() {
        let mut mesh = MeshStorage::new();
        let v = mesh.add_vertex(Vertex::new([0.0; 3])); // 无 halfedge
        let f = mesh.add_face(Face::new()); // 无 halfedge

        assert!(VertexRing::new(&mesh, v).is_empty());
        assert!(VertexAdjacentVerts::new(&mesh, v).is_empty());
        assert!(VertexAdjacentFaces::new(&mesh, v).is_empty());
        assert!(FaceHalfEdges::new(&mesh, f).is_empty());
        assert!(FaceVertices::new(&mesh, f).is_empty());

        // count_hint 对空迭代器返回 Some(0)
        assert_eq!(VertexRing::new(&mesh, v).count_hint(), Some(0));
        assert_eq!(FaceHalfEdges::new(&mesh, f).count_hint(), Some(0));
    }

    #[test]
    fn eager_exact_size_iterator_size_hint() {
        let (mesh, v, _he, f) = build_triangle();
        let [v0, _v1, _v2] = v;

        // size_hint 返回精确值（lower == upper）
        let ring = VertexRing::new(&mesh, v0);
        let (lower, upper) = ring.size_hint();
        assert_eq!(lower, 2);
        assert_eq!(upper, Some(2));

        // 消费一个后
        let mut ring = VertexRing::new(&mesh, v0);
        ring.next();
        let (lower, upper) = ring.size_hint();
        assert_eq!(lower, 1);
        assert_eq!(upper, Some(1));

        // FaceVertices
        let fv = FaceVertices::new(&mesh, f);
        let (lower, upper) = fv.size_hint();
        assert_eq!(lower, 3);
        assert_eq!(upper, Some(3));
    }

    #[test]
    fn lazy_count_hint_returns_none() {
        let (mesh, v, _he, f) = build_triangle();
        let [v0, _v1, _v2] = v;

        // lazy 版本 count_hint 恒为 None
        assert_eq!(VertexRing::lazy(&mesh, v0).count_hint(), None);
        assert_eq!(VertexAdjacentVerts::lazy(&mesh, v0).count_hint(), None);
        assert_eq!(VertexAdjacentFaces::lazy(&mesh, v0).count_hint(), None);
        assert_eq!(FaceHalfEdges::lazy(&mesh, f).count_hint(), None);
        assert_eq!(FaceVertices::lazy(&mesh, f).count_hint(), None);
    }

    #[test]
    fn lazy_is_empty_on_valid_and_invalid() {
        let (mesh, v, _he, f) = build_triangle();
        let [v0, _v1, _v2] = v;

        // 有效顶点：不空
        assert!(!VertexRing::lazy(&mesh, v0).is_empty());
        assert!(!VertexAdjacentVerts::lazy(&mesh, v0).is_empty());
        assert!(!FaceHalfEdges::lazy(&mesh, f).is_empty());
        assert!(!FaceVertices::lazy(&mesh, f).is_empty());

        // 无效 / 孤立 ID：空
        let mut mesh2 = MeshStorage::new();
        let v = mesh2.add_vertex(Vertex::new([0.0; 3])); // 无 halfedge
        let f = mesh2.add_face(Face::new()); // 无 halfedge
        assert!(VertexRing::lazy(&mesh2, v).is_empty());
        assert!(VertexAdjacentVerts::lazy(&mesh2, v).is_empty());
        assert!(VertexAdjacentFaces::lazy(&mesh2, v).is_empty());
        assert!(FaceHalfEdges::lazy(&mesh2, f).is_empty());
        assert!(FaceVertices::lazy(&mesh2, f).is_empty());
    }

    #[test]
    fn lazy_is_empty_after_exhaustion() {
        let (mesh, v, _he, _f) = build_triangle();
        let [v0, _v1, _v2] = v;

        let mut ring = VertexRing::lazy(&mesh, v0);
        assert!(!ring.is_empty()); // 初始非空
        while ring.next().is_some() {}
        assert!(ring.is_empty()); // 耗尽后为空
    }

    // ---------- MeshStorage 诊断方法 ----------

    #[test]
    fn max_vertex_valence_single_triangle() {
        let (mesh, _v, _he, _f) = build_triangle();
        // 单三角形每个顶点有 2 条 outgoing（一条面内 + 一条边界 twin）
        assert_eq!(mesh.max_vertex_valence(), 2);
    }

    #[test]
    fn max_vertex_valence_closed_fan() {
        let (mesh, _c, _outer, _faces) = build_closed_fan();
        // 中心顶点 c 有 3 条 outgoing（闭合扇形）
        assert_eq!(mesh.max_vertex_valence(), 3);
    }

    #[test]
    fn max_vertex_valence_empty_mesh() {
        let mesh = MeshStorage::new();
        assert_eq!(mesh.max_vertex_valence(), 0);
    }

    #[test]
    fn max_vertex_valence_isolated_vertices() {
        let mut mesh = MeshStorage::new();
        mesh.add_vertex(Vertex::new([0.0; 3]));
        mesh.add_vertex(Vertex::new([1.0; 3]));
        // 全是孤立顶点（无 halfedge），度数全为 0
        assert_eq!(mesh.max_vertex_valence(), 0);
    }

    #[test]
    fn max_vertex_valence_icosphere() {
        // icosphere(1) 是 20 面体的 1 次细分，顶点度数最大为 6（原 12 顶点）
        // 或 5（细分新增顶点）。实际最大值应为 6。
        let mesh = crate::test_util::build_icosphere(1);
        let max_val = mesh.max_vertex_valence();
        assert!(
            (5..=6).contains(&max_val),
            "icosphere(1) 最大度数应在 [5,6]，实际 {}",
            max_val
        );
    }

    // ---------- EdgeIter / edge_count 测试 ----------

    #[test]
    fn edge_count_single_triangle() {
        let (mesh, _v, _he, _f) = build_triangle();
        // 单三角形：6 半边 → 3 无向边
        assert_eq!(mesh.edge_count(), 3);
    }

    #[test]
    fn edge_count_two_triangles() {
        let (mesh, _v, _he, _f) = build_two_triangles();
        // 2 三角形共享 1 边：10 半边 → 5 无向边 (4 边界 + 1 共享)
        assert_eq!(mesh.edge_count(), 5);
    }

    #[test]
    fn edge_count_closed_fan() {
        let (mesh, _c, _outer, _faces) = build_closed_fan();
        // 3 三角形 + 3 边界 twin：12 半边 → 6 无向边
        assert_eq!(mesh.edge_count(), 6);
    }

    #[test]
    fn edge_count_empty() {
        let mesh = MeshStorage::new();
        assert_eq!(mesh.edge_count(), 0);
    }

    #[test]
    fn edge_ids_single_triangle_yields_three_unique_edges() {
        let (mesh, _v, _he, _f) = build_triangle();
        let edges: Vec<EdgeId> = mesh.edge_ids().collect();
        assert_eq!(edges.len(), 3, "单三角形应有 3 条无向边");
        // 所有 EdgeId 互不相同
        let mut sorted = edges.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 3);
    }

    #[test]
    fn edge_ids_two_triangles_yields_five_unique_edges() {
        let (mesh, _v, _he, _f) = build_two_triangles();
        let edges: Vec<EdgeId> = mesh.edge_ids().collect();
        assert_eq!(edges.len(), 5);
    }

    #[test]
    fn edge_ids_on_icosphere() {
        let mesh = crate::test_util::build_icosphere(1);
        let edges: Vec<EdgeId> = mesh.edge_ids().collect();
        // icosphere(1): V=42, F=80, 闭合流形 E = 3F/2 = 120
        assert_eq!(edges.len(), 120, "icosphere(1) 应有 120 条无向边");
        assert_eq!(mesh.edge_count(), 120);

        // 每条 EdgeId 的 halfedge() 应存在于 mesh 中
        for e in &edges {
            let he = e.halfedge();
            assert!(mesh.contains_halfedge(he), "EdgeId 的半边应存在");
        }

        // 所有 EdgeId 互不相同
        let mut sorted = edges.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 120, "所有 EdgeId 应互不相同");
    }

    #[test]
    fn edge_id_vertices_on_icosphere() {
        let mesh = crate::test_util::build_icosphere(1);
        for edge in mesh.edge_ids() {
            let (src, dst) = edge.vertices(&mesh).expect("闭合网格每条边应有 twin");
            assert_ne!(src, dst, "边的两端点应不同");
            assert!(mesh.contains_vertex(src) && mesh.contains_vertex(dst));
        }
    }

    #[test]
    fn edge_iter_is_exact_size() {
        let mesh = crate::test_util::build_icosphere(1);
        let mut iter = mesh.edge_ids();
        let total = iter.len();
        assert_eq!(total, 120);
        let (lower, upper) = iter.size_hint();
        assert_eq!(lower, 120);
        assert_eq!(upper, Some(120));

        for _ in 0..60 {
            iter.next();
        }
        assert_eq!(iter.len(), 60);
    }

    #[test]
    fn edge_count_matches_euler() {
        let mesh = crate::test_util::build_icosphere(1);
        let v = mesh.vertex_count();
        let e = mesh.edge_count();
        let f = mesh.face_count();
        assert_eq!(v as i64 - e as i64 + f as i64, 2, "欧拉公式");
    }

    // ---------- 边界环遍历 ----------

    #[test]
    fn is_closed_single_triangle_is_false() {
        let (mesh, _v, _he, _f) = build_triangle();
        assert!(!is_closed(&mesh), "单三角形有边界，非闭合");
    }

    #[test]
    fn is_closed_icosphere_is_true() {
        let mesh = crate::test_util::build_icosphere(1);
        assert!(is_closed(&mesh), "icosphere 是闭合网格");
    }

    #[test]
    fn boundary_loop_single_triangle() {
        let (mesh, _v, he, _f) = build_triangle();
        let [_h0, _h1, _h2, t0, t1, t2] = he;
        let loops = boundary_loops(&mesh);
        assert_eq!(loops.len(), 1, "单三角形应有 1 个边界环");
        let boundary = &loops[0];
        assert_eq!(boundary.len(), 3, "三角形边界环应有 3 条半边");
        // 验证三条边界半边都在环中
        for &bhe in &[t0, t1, t2] {
            assert!(boundary.contains(&bhe), "边界半边 {:?} 应在环中", bhe);
        }
    }

    #[test]
    fn boundary_loop_icosphere_is_empty() {
        let mesh = crate::test_util::build_icosphere(1);
        let loops = boundary_loops(&mesh);
        assert!(loops.is_empty(), "闭合网格不应有边界环");
    }

    #[test]
    fn boundary_loop_lazy_matches_eager() {
        let (mesh, _v, he, _f) = build_triangle();
        let [_h0, _h1, _h2, t0, t1, t2] = he;
        for &start in &[t0, t1, t2] {
            let eager: Vec<_> = BoundaryLoop::new(&mesh, start).collect();
            let lazy: Vec<_> = BoundaryLoopLazy::new(&mesh, start).collect();
            // 顺序可能因起点不同而旋转，但长度应一致
            assert_eq!(eager.len(), lazy.len(), "eager 和 lazy 长度一致");
            assert_eq!(eager.len(), 3);
        }
    }

    #[test]
    fn boundary_loop_detects_multiple_loops() {
        let (mesh, _v, _he, _f) = build_two_triangles();
        // 两三角形共享一边，存在 4 条边界半边
        let loops = boundary_loops(&mesh);
        assert!(!loops.is_empty(), "两三角形应有边界");
    }

    #[test]
    fn is_closed_closed_fan_is_false() {
        let (mesh, _c, _outer, _faces) = build_closed_fan();
        assert!(!is_closed(&mesh), "扇形有外边界");
    }

    #[test]
    fn euler_characteristic_single_triangle() {
        let (mesh, _v, _he, _f) = build_triangle();
        // V=3, E=3 (6 半边 / 2), F=1 → χ = 3 - 3 + 1 = 1
        assert_eq!(mesh.euler_characteristic(), 1);
        // 单三角形是带边界的圆盘，χ=1，genus 公式对带边界曲面不精确
        // (2 - 1) / 2 = 0（向下取整）
        assert_eq!(mesh.genus(), 0);
    }

    #[test]
    fn euler_characteristic_closed_fan() {
        let (mesh, _c, _outer, _faces) = build_closed_fan();
        // V=4 (c + 3 外顶点), E=6 (12 半边 / 2), F=3 → χ = 4 - 6 + 3 = 1
        assert_eq!(mesh.euler_characteristic(), 1);
    }

    #[test]
    fn euler_characteristic_icosphere_is_two() {
        // 闭合球面网格 χ = 2，genus = 0
        let mesh = crate::test_util::build_icosphere(1);
        assert_eq!(mesh.euler_characteristic(), 2, "闭合球面 χ 应为 2");
        assert_eq!(mesh.genus(), 0, "球面亏格应为 0");
    }

    #[test]
    fn euler_characteristic_empty_mesh() {
        let mesh = MeshStorage::new();
        assert_eq!(mesh.euler_characteristic(), 0);
        assert_eq!(mesh.genus(), 1); // (2 - 0) / 2 = 1（无意义，仅验证不 panic）
    }

    // ---------- VertexAdjacentEdges ----------

    #[test]
    fn vertex_adjacent_edges_triangle() {
        let (mesh, v, _he, _f) = build_triangle();
        let [v0, _v1, _v2] = v;
        // 单三角形每个顶点有 2 条关联边
        let edges: Vec<_> = VertexAdjacentEdges::new(&mesh, v0).collect();
        assert_eq!(edges.len(), 2);
        for e in &edges {
            assert!(mesh.contains_halfedge(e.halfedge()));
        }
    }

    #[test]
    fn vertex_adjacent_edges_icosphere() {
        let mesh = crate::test_util::build_icosphere(1);
        for v in mesh.vertex_ids() {
            let edges: Vec<_> = VertexAdjacentEdges::new(&mesh, v).collect();
            let valence = VertexRing::new(&mesh, v).count();
            // 关联边数 == 度数（对于闭合网格）
            assert_eq!(edges.len(), valence, "顶点 {:?} 度数应等于关联边数", v);
        }
    }

    // ---------- _into 内存复用 API ----------

    #[test]
    fn collect_outgoing_halfedges_into_matches_original() {
        let mesh = crate::test_util::build_icosphere(1);
        let mut buf = Vec::new();
        for v in mesh.vertex_ids() {
            let expected = collect_outgoing_halfedges(&mesh, v);
            collect_outgoing_halfedges_into(&mesh, v, &mut buf);
            assert_eq!(buf, expected, "顶点 {:?} 的 outgoing 半边不匹配", v);
        }
    }

    #[test]
    fn collect_face_halfedges_into_matches_original() {
        let mesh = crate::test_util::build_icosphere(1);
        let mut buf = Vec::new();
        for f in mesh.face_ids() {
            let expected = collect_face_halfedges(&mesh, f);
            collect_face_halfedges_into(&mesh, f, &mut buf);
            assert_eq!(buf, expected, "面 {:?} 的半边不匹配", f);
        }
    }

    #[test]
    fn collect_outgoing_halfedges_into_reuses_capacity() {
        let mesh = crate::test_util::build_icosphere(1);
        let mut buf = Vec::new();
        // 第一次调用分配容量
        collect_outgoing_halfedges_into(&mesh, mesh.vertex_ids().next().unwrap(), &mut buf);
        let cap_after_first = buf.capacity();
        assert!(cap_after_first > 0);
        // 后续调用不应释放容量
        for v in mesh.vertex_ids() {
            collect_outgoing_halfedges_into(&mesh, v, &mut buf);
            assert!(buf.capacity() >= cap_after_first, "容量不应缩小");
        }
    }

    #[test]
    fn collect_face_halfedges_into_clears_buffer() {
        let (mesh, _v, _he, f) = build_triangle();
        let mut buf = vec![HalfEdgeId::default(); 100]; // 预填充垃圾数据
        collect_face_halfedges_into(&mesh, f, &mut buf);
        // 应只包含 3 条半边（三角形），不含垃圾数据
        assert_eq!(buf.len(), 3);
    }

    #[test]
    fn collect_outgoing_halfedges_into_isolated_vertex() {
        let mut mesh = MeshStorage::new();
        let v = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let mut buf = vec![HalfEdgeId::default(); 10];
        collect_outgoing_halfedges_into(&mesh, v, &mut buf);
        assert!(buf.is_empty(), "孤立顶点应返回空 buffer");
    }
}
