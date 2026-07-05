//! 半边网格底层存储模块
//!
//! `MeshStorage` 仅承担「存储 + 句柄有效性」职责，不维护任何拓扑一致性约束
//! （twin/next/prev 等关系由调用方或上层 builder 维护）。这样可以让底层尽可能
//! 通用，便于后续叠加不同策略的构建器。
//!
//! ## 存储选型
//! 选用 [`slotmap::SlotMap`] 而非 `HashMap<Id, T>` 或 `Vec<Option<T>>`，原因：
//! 1. **O(1) 插入/删除/查找**，且删除后槽位可被回收复用；
//! 2. **版本号机制天然解决 ABA 问题**——删除一个元素后，其旧句柄即使槽位被
//!    新元素复用，也仍然无效，`get` 会返回 `None`，满足需求 3 中的「自动标记失效」；
//! 3. 配合 [`ids`](crate::ids) 模块的 `new_key_type!`，key 在编译期就是强类型，杜绝跨类型误用。
//!
//! ## 字段约定（半边拓扑）
//! - `HalfEdge::vertex`：该半边指向的「目的顶点」（tip / destination）；
//! - `HalfEdge::twin`：反向半边，二者构成一条无向边；
//! - `HalfEdge::next` / `prev`：同一面边界环上的后继 / 前驱半边；
//! - `HalfEdge::face`：该半边所属的面；边界半边的 `face` 通常为 `None`；
//! - `Vertex::halfedge`：从该顶点出发的任一 outgoing 半边；
//! - `Face::halfedge`：该面边界环上的任一半边。
//!
//! ## 迭代
//! 提供两类迭代器：
//! - **句柄迭代**：`vertex_ids()`/`halfedge_ids()`/`face_ids()` 返回 `VertexId` 等；
//! - **数据迭代**：`vertices()`/`halfedges()`/`faces()` 返回 `&Vertex` 等引用，
//!   以及对应的 `_mut` 可变版本。
//!
//! 此外 `is_empty()` 判断网格是否为空，`edge_count()` 返回无向边数。

use slotmap::SlotMap;

use crate::ids::{FaceId, HalfEdgeId, VertexId};

/// 顶点数据：3D 位置 + 任一从该顶点出发的半边句柄。
#[derive(Debug, Clone)]
pub struct Vertex {
    pub position: [f64; 3],
    pub halfedge: Option<HalfEdgeId>,
}

/// 半边数据：四向邻接（twin/next/prev/face）+ 目的顶点。
#[derive(Debug, Clone)]
pub struct HalfEdge {
    /// 半边指向的顶点（tip）。
    pub vertex: VertexId,
    /// 反向半边。`None` 表示该半边目前还没有配对（例如尚未缝合的边界）。
    pub twin: Option<HalfEdgeId>,
    /// 同一面边界环上的下一条半边。
    pub next: Option<HalfEdgeId>,
    /// 同一面边界环上的上一条半边。
    pub prev: Option<HalfEdgeId>,
    /// 该半边所属的面。`None` 表示这是边界半边。
    pub face: Option<FaceId>,
}

/// 面数据：任一围绕该面的半边句柄，作为访问面边界环的入口。
#[derive(Debug, Clone)]
pub struct Face {
    pub halfedge: Option<HalfEdgeId>,
}

impl Vertex {
    pub fn new(position: [f64; 3]) -> Self {
        Self {
            position,
            halfedge: None,
        }
    }
}

impl Default for Vertex {
    fn default() -> Self {
        Self::new([0.0; 3])
    }
}

impl HalfEdge {
    pub fn new(vertex: VertexId) -> Self {
        Self {
            vertex,
            twin: None,
            next: None,
            prev: None,
            face: None,
        }
    }
}

impl Face {
    pub fn new() -> Self {
        Self { halfedge: None }
    }
}

impl Default for Face {
    fn default() -> Self {
        Self::new()
    }
}

/// 半边网格底层存储容器。
///
/// 三类元素分别放在独立的 `SlotMap` 中，互不干扰。所有公开接口都返回 `Option`
/// 或 ID 类型，访问已删除元素只会得到 `None`，绝不 panic。
#[derive(Debug, Clone)]
pub struct MeshStorage {
    vertices: SlotMap<VertexId, Vertex>,
    halfedges: SlotMap<HalfEdgeId, HalfEdge>,
    faces: SlotMap<FaceId, Face>,
}

impl Default for MeshStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl MeshStorage {
    /// 创建一个空网格。
    pub fn new() -> Self {
        Self {
            vertices: SlotMap::with_key(),
            halfedges: SlotMap::with_key(),
            faces: SlotMap::with_key(),
        }
    }

    // ---------- 增 ----------

    /// 插入一个顶点，返回新分配的句柄。
    pub fn add_vertex(&mut self, vertex: Vertex) -> VertexId {
        self.vertices.insert(vertex)
    }

    /// 插入一条半边，返回新分配的句柄。
    pub fn add_halfedge(&mut self, halfedge: HalfEdge) -> HalfEdgeId {
        self.halfedges.insert(halfedge)
    }

    /// 插入一个面，返回新分配的句柄。
    pub fn add_face(&mut self, face: Face) -> FaceId {
        self.faces.insert(face)
    }

    // ---------- 删 ----------

    /// 删除指定顶点。返回被删除的数据；若句柄已失效则返回 `None`。
    ///
    /// 删除后该句柄永久失效（即使槽位被复用，旧句柄版本号不匹配）。
    /// 注意：本方法不清理指向该顶点的半边引用，拓扑一致性由上层维护。
    pub fn remove_vertex(&mut self, id: VertexId) -> Option<Vertex> {
        self.vertices.remove(id)
    }

    /// 删除指定半边。语义同 [`remove_vertex`](Self::remove_vertex)。
    pub fn remove_halfedge(&mut self, id: HalfEdgeId) -> Option<HalfEdge> {
        self.halfedges.remove(id)
    }

    /// 删除指定面。语义同 [`remove_vertex`](Self::remove_vertex)。
    pub fn remove_face(&mut self, id: FaceId) -> Option<Face> {
        self.faces.remove(id)
    }

    // ---------- 查（不可变） ----------

    pub fn get_vertex(&self, id: VertexId) -> Option<&Vertex> {
        self.vertices.get(id)
    }

    pub fn get_halfedge(&self, id: HalfEdgeId) -> Option<&HalfEdge> {
        self.halfedges.get(id)
    }

    pub fn get_face(&self, id: FaceId) -> Option<&Face> {
        self.faces.get(id)
    }

    // ---------- 查（可变） ----------

    pub fn get_vertex_mut(&mut self, id: VertexId) -> Option<&mut Vertex> {
        self.vertices.get_mut(id)
    }

    pub fn get_halfedge_mut(&mut self, id: HalfEdgeId) -> Option<&mut HalfEdge> {
        self.halfedges.get_mut(id)
    }

    pub fn get_face_mut(&mut self, id: FaceId) -> Option<&mut Face> {
        self.faces.get_mut(id)
    }

    // ---------- 有效性判断 ----------

    /// 句柄是否指向一个仍然存在的顶点。
    pub fn contains_vertex(&self, id: VertexId) -> bool {
        self.vertices.contains_key(id)
    }

    /// 句柄是否指向一条仍然存在的半边。
    pub fn contains_halfedge(&self, id: HalfEdgeId) -> bool {
        self.halfedges.contains_key(id)
    }

    /// 句柄是否指向一个仍然存在的面。
    pub fn contains_face(&self, id: FaceId) -> bool {
        self.faces.contains_key(id)
    }

    // ---------- 容量统计 ----------

    /// 当前顶点数量（不含已删除槽位）。
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// 当前半边数量。
    pub fn halfedge_count(&self) -> usize {
        self.halfedges.len()
    }

    /// 当前面数量。
    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    // ---------- 迭代 ----------

    /// 迭代所有仍有效的顶点句柄。
    pub fn vertex_ids(&self) -> impl Iterator<Item = VertexId> + '_ {
        self.vertices.keys()
    }

    /// 迭代所有仍有效的半边句柄。
    pub fn halfedge_ids(&self) -> impl Iterator<Item = HalfEdgeId> + '_ {
        self.halfedges.keys()
    }

    /// 迭代所有仍有效的面句柄。
    pub fn face_ids(&self) -> impl Iterator<Item = FaceId> + '_ {
        self.faces.keys()
    }

    /// 迭代所有仍有效的顶点数据（不可变引用）。
    pub fn vertices(&self) -> impl Iterator<Item = &Vertex> + '_ {
        self.vertices.values()
    }

    /// 迭代所有仍有效的顶点数据（可变引用）。
    pub fn vertices_mut(&mut self) -> impl Iterator<Item = &mut Vertex> + '_ {
        self.vertices.values_mut()
    }

    /// 迭代所有仍有效的半边数据（不可变引用）。
    pub fn halfedges(&self) -> impl Iterator<Item = &HalfEdge> + '_ {
        self.halfedges.values()
    }

    /// 迭代所有仍有效的半边数据（可变引用）。
    pub fn halfedges_mut(&mut self) -> impl Iterator<Item = &mut HalfEdge> + '_ {
        self.halfedges.values_mut()
    }

    /// 迭代所有仍有效的面数据（不可变引用）。
    pub fn faces(&self) -> impl Iterator<Item = &Face> + '_ {
        self.faces.values()
    }

    /// 迭代所有仍有效的面数据（可变引用）。
    pub fn faces_mut(&mut self) -> impl Iterator<Item = &mut Face> + '_ {
        self.faces.values_mut()
    }

    /// 网格是否为空（无顶点且无面）。
    ///
    /// 注意：半边数量不参与判断，因为有效网格的半边数量始终由面决定。
    pub fn is_empty(&self) -> bool {
        self.vertex_count() == 0 && self.face_count() == 0
    }

    // ---------- 拓扑诊断 ----------

    /// 欧拉示性数 $\chi = V - E + F$。
    ///
    /// 其中 $E$ 为无向边数。假设网格是流形且每条边都有 twin，
    /// 则 $E = \text{halfedge\_count} / 2$。若存在孤立半边（无 twin），
    /// 结果会偏大（$E$ 被低估），但对流形网格结果精确。
    ///
    /// 闭合球面网格 $\chi = 2$；亏格 $g$ 的闭合曲面 $\chi = 2 - 2g$。
    pub fn euler_characteristic(&self) -> i64 {
        let v = self.vertex_count() as i64;
        let e = (self.halfedge_count() / 2) as i64;
        let f = self.face_count() as i64;
        v - e + f
    }

    /// 亏格 $g = (2 - \chi) / 2$。
    ///
    /// 对于闭合定向曲面，$\chi = 2 - 2g$，故 $g = (2 - \chi) / 2$。
    /// 带边界的曲面公式更复杂，此处仍按闭合曲面公式计算，
    /// 结果可能为负或非整数（向下取整），仅作诊断参考。
    pub fn genus(&self) -> i64 {
        (2 - self.euler_characteristic()) / 2
    }

    // ---------- 容量管理 ----------

    /// 清空网格，删除所有顶点、半边、面。等效于 `*self = MeshStorage::new()`。
    ///
    /// `SlotMap::clear()` 是 $O(n)$ 时间，但会释放内部内存。
    /// 对频繁重建网格的场景有用。
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.halfedges.clear();
        self.faces.clear();
    }

    /// 为大约 `vertex_cap` 个顶点、`halfedge_cap` 条半边、`face_cap` 个面预分配内存。
    ///
    /// 适合已知最终规模的批量构建，减少 rehash / 重分配次数。
    /// 三参数均为下限提示，实际分配可能更多。
    pub fn reserve(&mut self, vertex_cap: usize, halfedge_cap: usize, face_cap: usize) {
        self.vertices.reserve(vertex_cap);
        self.halfedges.reserve(halfedge_cap);
        self.faces.reserve(face_cap);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- 新增 / 查询 ----------

    #[test]
    fn add_and_get_vertex() {
        let mut mesh = MeshStorage::new();
        let id = mesh.add_vertex(Vertex::new([1.0, 2.0, 3.0]));

        assert!(mesh.contains_vertex(id));
        let v = mesh.get_vertex(id).expect("刚插入的顶点必须可查");
        assert_eq!(v.position, [1.0, 2.0, 3.0]);
        assert!(v.halfedge.is_none());
        assert_eq!(mesh.vertex_count(), 1);
    }

    #[test]
    fn add_and_get_halfedge() {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([0.0; 3]));
        let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));

        let he = mesh.add_halfedge(HalfEdge::new(v1));
        // 半边从 v0 指向 v1，故同时记录 origin（v0）的 outgoing 半边
        mesh.get_vertex_mut(v0).unwrap().halfedge = Some(he);

        assert!(mesh.contains_halfedge(he));
        let h = mesh.get_halfedge(he).unwrap();
        assert_eq!(h.vertex, v1);
        assert!(h.twin.is_none());
        assert_eq!(mesh.get_vertex(v0).unwrap().halfedge, Some(he));
        assert_eq!(mesh.halfedge_count(), 1);
    }

    #[test]
    fn add_and_get_face() {
        let mut mesh = MeshStorage::new();
        let f = mesh.add_face(Face::new());

        assert!(mesh.contains_face(f));
        assert!(mesh.get_face(f).unwrap().halfedge.is_none());
        assert_eq!(mesh.face_count(), 1);
    }

    // ---------- 删除 / 失效 ----------

    #[test]
    fn remove_vertex_invalidates_id() {
        let mut mesh = MeshStorage::new();
        let id = mesh.add_vertex(Vertex::new([0.0; 3]));

        assert!(mesh.contains_vertex(id));
        let removed = mesh.remove_vertex(id);
        assert!(removed.is_some(), "删除已存在的顶点应返回 Some");
        assert_eq!(removed.unwrap().position, [0.0; 3]);

        // 句柄已失效：contains 与 get 都返回 false / None
        assert!(!mesh.contains_vertex(id));
        assert!(mesh.get_vertex(id).is_none());
        assert_eq!(mesh.vertex_count(), 0);

        // 重复删除返回 None
        assert!(mesh.remove_vertex(id).is_none());
    }

    #[test]
    fn remove_halfedge_invalidates_id() {
        let mut mesh = MeshStorage::new();
        let v = mesh.add_vertex(Vertex::new([0.0; 3]));
        let id = mesh.add_halfedge(HalfEdge::new(v));

        assert!(mesh.remove_halfedge(id).is_some());
        assert!(!mesh.contains_halfedge(id));
        assert!(mesh.get_halfedge(id).is_none());
        assert!(mesh.remove_halfedge(id).is_none());
    }

    #[test]
    fn remove_face_invalidates_id() {
        let mut mesh = MeshStorage::new();
        let id = mesh.add_face(Face::new());

        assert!(mesh.remove_face(id).is_some());
        assert!(!mesh.contains_face(id));
        assert!(mesh.get_face(id).is_none());
        assert!(mesh.remove_face(id).is_none());
    }

    // ---------- 槽位复用不会复活旧句柄（ABA 安全） ----------

    #[test]
    fn slot_reuse_does_not_resurrect_old_id() {
        let mut mesh = MeshStorage::new();
        let old_id = mesh.add_vertex(Vertex::new([1.0, 1.0, 1.0]));
        mesh.remove_vertex(old_id);

        // 再插入一个新顶点，slotmap 很可能复用同一槽位但版本号 +1
        let new_id = mesh.add_vertex(Vertex::new([2.0, 2.0, 2.0]));

        // 新句柄有效
        assert!(mesh.contains_vertex(new_id));
        // 旧句柄依然无效（版本号不匹配）—— 这是 slotmap 的核心保证
        assert!(!mesh.contains_vertex(old_id));
        assert!(mesh.get_vertex(old_id).is_none());

        // 两个句柄不相等
        assert_ne!(old_id, new_id);
    }

    // ---------- 可变访问 ----------

    #[test]
    fn get_mut_allows_in_place_update() {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([0.0; 3]));
        let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let he = mesh.add_halfedge(HalfEdge::new(v1));

        // 通过 get_mut 设置 twin / next / prev / face 之间的链接
        let h = mesh.get_halfedge_mut(he).unwrap();
        h.twin = Some(he); // 自指仅用于测试
        h.next = Some(he);
        h.prev = Some(he);

        let v = mesh.get_vertex_mut(v0).unwrap();
        v.halfedge = Some(he);

        assert_eq!(mesh.get_vertex(v0).unwrap().halfedge, Some(he));
        let h2 = mesh.get_halfedge(he).unwrap();
        assert_eq!(h2.twin, Some(he));
        assert_eq!(h2.next, Some(he));
        assert_eq!(h2.prev, Some(he));
    }

    // ---------- 访问已删除元素不 panic ----------

    #[test]
    fn accessing_invalid_id_never_panics() {
        let mut mesh = MeshStorage::new();
        let id = mesh.add_face(Face::new());
        mesh.remove_face(id);

        // 这些调用都应当平稳返回 None，绝不 panic
        assert!(mesh.get_face(id).is_none());
        assert!(mesh.get_face_mut(id).is_none());
        assert!(!mesh.contains_face(id));
    }

    // ---------- 容量管理 ----------

    #[test]
    fn clear_empties_all_three_slotmaps() {
        let mut mesh = MeshStorage::new();
        mesh.add_vertex(Vertex::new([1.0, 2.0, 3.0]));
        mesh.add_vertex(Vertex::new([4.0, 5.0, 6.0]));
        let v = mesh.add_vertex(Vertex::new([0.0; 3]));
        mesh.add_halfedge(HalfEdge::new(v));
        mesh.add_halfedge(HalfEdge::new(v));
        mesh.add_face(Face::new());

        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.halfedge_count(), 2);
        assert_eq!(mesh.face_count(), 1);

        mesh.clear();

        assert_eq!(mesh.vertex_count(), 0);
        assert_eq!(mesh.halfedge_count(), 0);
        assert_eq!(mesh.face_count(), 0);
        // 迭代器为空
        assert_eq!(mesh.vertex_ids().count(), 0);
        assert_eq!(mesh.halfedge_ids().count(), 0);
        assert_eq!(mesh.face_ids().count(), 0);
        // Euler 示性数也应为 0
        assert_eq!(mesh.euler_characteristic(), 0);
    }

    #[test]
    fn clear_equivalent_to_new() {
        let mut mesh = MeshStorage::new();
        mesh.add_vertex(Vertex::new([1.0; 3]));
        mesh.add_face(Face::new());

        mesh.clear();
        let fresh = MeshStorage::new();

        assert_eq!(mesh.vertex_count(), fresh.vertex_count());
        assert_eq!(mesh.halfedge_count(), fresh.halfedge_count());
        assert_eq!(mesh.face_count(), fresh.face_count());
    }

    #[test]
    fn clear_allows_reuse_after() {
        let mut mesh = MeshStorage::new();
        mesh.add_vertex(Vertex::new([1.0; 3]));
        mesh.clear();

        // clear 后仍可继续添加
        let v = mesh.add_vertex(Vertex::new([2.0; 3]));
        assert!(mesh.contains_vertex(v));
        assert_eq!(mesh.vertex_count(), 1);
    }

    #[test]
    fn reserve_does_not_change_counts() {
        let mut mesh = MeshStorage::new();
        mesh.add_vertex(Vertex::new([0.0; 3]));

        mesh.reserve(100, 200, 50);

        // reserve 仅预分配内存，不改变元素数量
        assert_eq!(mesh.vertex_count(), 1);
        assert_eq!(mesh.halfedge_count(), 0);
        assert_eq!(mesh.face_count(), 0);
    }

    #[test]
    fn reserve_zero_is_noop() {
        let mut mesh = MeshStorage::new();
        mesh.reserve(0, 0, 0);
        // 不 panic 即可
        assert_eq!(mesh.vertex_count(), 0);
    }

    // ---------- 数据迭代器 / is_empty ----------

    #[test]
    fn is_empty_on_new_mesh() {
        let mesh = MeshStorage::new();
        assert!(mesh.is_empty());
    }

    #[test]
    fn is_empty_after_clear() {
        let mut mesh = MeshStorage::new();
        mesh.add_vertex(Vertex::new([0.0; 3]));
        mesh.add_face(Face::new());
        assert!(!mesh.is_empty());
        mesh.clear();
        assert!(mesh.is_empty());
    }

    #[test]
    fn vertices_iter_yields_all_data() {
        let mut mesh = MeshStorage::new();
        let _v0 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let _v1 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));
        let _v2 = mesh.add_vertex(Vertex::new([0.0, 0.0, 1.0]));

        let verts: Vec<&Vertex> = mesh.vertices().collect();
        assert_eq!(verts.len(), 3);
        // 检查所有顶点都找得到
        let positions: Vec<[f64; 3]> = verts.iter().map(|v| v.position).collect();
        assert!(positions.contains(&[1.0, 0.0, 0.0]));
        assert!(positions.contains(&[0.0, 1.0, 0.0]));
        assert!(positions.contains(&[0.0, 0.0, 1.0]));

        // vertices() 和 vertex_ids() 数量一致
        assert_eq!(mesh.vertices().count(), mesh.vertex_ids().count());
    }

    #[test]
    fn vertices_mut_allows_modification() {
        let mut mesh = MeshStorage::new();
        mesh.add_vertex(Vertex::new([1.0; 3]));
        mesh.add_vertex(Vertex::new([2.0; 3]));

        for v in mesh.vertices_mut() {
            v.position = [0.0; 3];
        }

        for v in mesh.vertices() {
            assert_eq!(v.position, [0.0; 3]);
        }
    }

    #[test]
    fn halfedges_faces_iters_match_id_counts() {
        use crate::test_util::build_icosphere;
        let mesh = build_icosphere(1);
        assert_eq!(mesh.halfedges().count(), mesh.halfedge_ids().count());
        assert_eq!(mesh.faces().count(), mesh.face_ids().count());
        assert!(!mesh.is_empty());
    }

    #[test]
    fn empty_iterators_yield_nothing() {
        let mesh = MeshStorage::new();
        assert_eq!(mesh.vertices().count(), 0);
        assert_eq!(mesh.halfedges().count(), 0);
        assert_eq!(mesh.faces().count(), 0);
    }
}
