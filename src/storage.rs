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

use slotmap::{SecondaryMap, SlotMap};

use crate::Scalar;
use crate::ids::{FaceId, HalfEdgeId, VertexId};

/// 顶点数据：3D 位置 + 任一从该顶点出发的半边句柄。
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Vertex {
    pub position: [Scalar; 3],
    pub halfedge: Option<HalfEdgeId>,
}

/// 半边数据：四向邻接（twin/next/prev/face）+ 目的顶点。
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Face {
    pub halfedge: Option<HalfEdgeId>,
}

impl Vertex {
    /// 创建新顶点，指定 3D 位置。
    ///
    /// 初始状态下 `halfedge` 为 `None`。
    pub fn new(position: [Scalar; 3]) -> Self {
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
    /// 创建新半边，指向指定顶点。
    ///
    /// 初始状态下 `next`/`prev`/`twin`/`face` 均为 `None`。
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
    /// 创建新面。
    ///
    /// 初始状态下 `halfedge` 为 `None`。
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
///
/// ## SOA 位置缓存
/// 为提升几何热路径的缓存命中率，`MeshStorage` 内部维护一份稠密的
/// `positions: Vec<[Scalar; 3]>` 缓存，与 `vertices` SlotMap 并行。
/// 通过 [`position_index`](Self::position_index) 可取得 `VertexId → u32`
/// 的稠密索引，再用 [`positions_dense`](Self::positions_dense) 做
/// 24 字节步长的连续访问，避免 SlotMap 槽位元数据造成的缓存浪费。
///
/// 缓存由 [`add_vertex`](Self::add_vertex) / [`remove_vertex`](Self::remove_vertex)
/// / [`set_position`](Self::set_position) 自动同步。若直接通过
/// [`get_vertex_mut`](Self::get_vertex_mut) 修改 `position` 字段，缓存会
/// 暂时失效，需调用 [`sync_position`](Self::sync_position) 或
/// [`rebuild_position_cache`](Self::rebuild_position_cache) 修复。
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MeshStorage {
    vertices: SlotMap<VertexId, Vertex>,
    halfedges: SlotMap<HalfEdgeId, HalfEdge>,
    faces: SlotMap<FaceId, Face>,
    /// SOA 位置缓存：稠密连续的顶点坐标，索引由 `pos_index` 提供。
    #[cfg_attr(feature = "serde", serde(skip))]
    positions: Vec<[Scalar; 3]>,
    /// VertexId → `positions` 索引。SecondaryMap 由 Vec 支持，O(1) 查找。
    #[cfg_attr(feature = "serde", serde(skip))]
    pos_index: SecondaryMap<VertexId, u32>,
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
            positions: Vec::new(),
            pos_index: SecondaryMap::new(),
        }
    }

    // ---------- 增 ----------

    /// 插入一个顶点，返回新分配的句柄。
    ///
    /// 同时将 `vertex.position` 推入 SOA 位置缓存，保持缓存与主存同步。
    pub fn add_vertex(&mut self, vertex: Vertex) -> VertexId {
        let pos = vertex.position;
        let id = self.vertices.insert(vertex);
        let idx = self.positions.len() as u32;
        self.positions.push(pos);
        self.pos_index.insert(id, idx);
        id
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
    ///
    /// SOA 位置缓存通过 swap-remove 同步：被删除顶点的位置由末尾元素填补，
    /// 填补元素的稠密索引被同步更新。
    pub fn remove_vertex(&mut self, id: VertexId) -> Option<Vertex> {
        let removed = self.vertices.remove(id);
        if removed.is_some()
            && let Some(idx) = self.pos_index.remove(id)
        {
            let idx = idx as usize;
            // swap-remove：末尾元素填补 idx 槽位
            self.positions.swap_remove(idx);
            // 若填补元素非自身（idx 仍在范围内），更新填补元素的稠密索引
            if idx < self.positions.len() {
                let old_last_idx = self.positions.len() as u32;
                // 在 pos_index 中找到值为 old_last_idx 的项，改为 idx
                for (_k, v) in self.pos_index.iter_mut() {
                    if *v == old_last_idx {
                        *v = idx as u32;
                        break;
                    }
                }
            }
        }
        removed
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

    /// 按 `VertexId` 获取顶点引用。
    pub fn get_vertex(&self, id: VertexId) -> Option<&Vertex> {
        self.vertices.get(id)
    }

    /// 按 `HalfEdgeId` 获取半边引用。
    pub fn get_halfedge(&self, id: HalfEdgeId) -> Option<&HalfEdge> {
        self.halfedges.get(id)
    }

    /// 按 `FaceId` 获取面引用。
    pub fn get_face(&self, id: FaceId) -> Option<&Face> {
        self.faces.get(id)
    }

    // ---------- 查（可变） ----------

    /// 按 `VertexId` 获取可变顶点引用。
    ///
    /// **注意**：若通过此方法修改 `vertex.position`，SOA 位置缓存会暂时失效。
    /// 建议改用 [`set_position`](Self::set_position) 更新顶点位置，
    /// 或在修改后调用 [`sync_position`](Self::sync_position)。
    pub fn get_vertex_mut(&mut self, id: VertexId) -> Option<&mut Vertex> {
        self.vertices.get_mut(id)
    }

    /// 按 `HalfEdgeId` 获取可变半边引用。
    pub fn get_halfedge_mut(&mut self, id: HalfEdgeId) -> Option<&mut HalfEdge> {
        self.halfedges.get_mut(id)
    }

    /// 按 `FaceId` 获取可变面引用。
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

    // ---------- SOA 位置缓存 ----------

    /// 按 `VertexId` 取顶点位置（读 SOA 缓存，24 字节步长连续访问）。
    ///
    /// 与 `get_vertex(id).position` 等价，但走稠密 `Vec` 缓存，
    /// 批量遍历时缓存命中率更高。
    #[inline]
    pub fn get_position(&self, id: VertexId) -> Option<[Scalar; 3]> {
        let idx = self.pos_index.get(id)?;
        self.positions.get(*idx as usize).copied()
    }

    /// 返回稠密位置切片 `&[[Scalar; 3]]`，用于批量遍历。
    ///
    /// 配合 [`position_index`](Self::position_index) 将 `VertexId` 映射到
    /// 切片下标，可在热路径中以 24 字节步长连续访问所有顶点位置，
    /// 避免 SlotMap 槽位元数据的缓存浪费。
    #[inline]
    pub fn positions_dense(&self) -> &[[Scalar; 3]] {
        &self.positions
    }

    /// `VertexId → u32` 稠密索引，配合 [`positions_dense`](Self::positions_dense) 使用。
    #[inline]
    pub fn position_index(&self, id: VertexId) -> Option<u32> {
        self.pos_index.get(id).copied()
    }

    /// 更新顶点位置，同时同步主存与 SOA 缓存。
    ///
    /// 与 `get_vertex_mut(id).position = pos` 的区别：本方法保持缓存一致性，
    /// 推荐在所有需要修改顶点位置的场景使用。
    pub fn set_position(&mut self, id: VertexId, pos: [Scalar; 3]) -> Option<()> {
        let vertex = self.vertices.get_mut(id)?;
        vertex.position = pos;
        if let Some(idx) = self.pos_index.get(id) {
            self.positions[*idx as usize] = pos;
        }
        Some(())
    }

    /// 将主存中的 `vertex.position` 同步到 SOA 缓存。
    ///
    /// 当通过 [`get_vertex_mut`](Self::get_vertex_mut) 修改 `position` 字段后，
    /// 缓存会暂时失效。调用本方法以单点同步。
    pub fn sync_position(&mut self, id: VertexId) {
        if let (Some(v), Some(idx)) = (self.vertices.get(id), self.pos_index.get(id)) {
            self.positions[*idx as usize] = v.position;
        }
    }

    /// 从主存重建整个 SOA 位置缓存。
    ///
    /// 适用于反序列化后或大量通过 `get_vertex_mut` 修改位置后的批量修复。
    /// 时间复杂度 $O(V)$。
    pub fn rebuild_position_cache(&mut self) {
        self.positions.clear();
        self.pos_index.clear();
        for (id, v) in self.vertices.iter() {
            let idx = self.positions.len() as u32;
            self.positions.push(v.position);
            self.pos_index.insert(id, idx);
        }
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
        self.positions.clear();
        self.pos_index.clear();
    }

    /// 为大约 `vertex_cap` 个顶点、`halfedge_cap` 条半边、`face_cap` 个面预分配内存。
    ///
    /// 适合已知最终规模的批量构建，减少 rehash / 重分配次数。
    /// 三参数均为下限提示，实际分配可能更多。
    pub fn reserve(&mut self, vertex_cap: usize, halfedge_cap: usize, face_cap: usize) {
        self.vertices.reserve(vertex_cap);
        self.halfedges.reserve(halfedge_cap);
        self.faces.reserve(face_cap);
        self.positions.reserve(vertex_cap);
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

    // ---------- 槽位版本号 / 多轮复用 ----------

    #[test]
    fn slot_reuse_multiple_cycles() {
        // 多轮删除/插入循环，验证版本号持续递增、旧句柄始终无效
        let mut mesh = MeshStorage::new();
        let mut old_ids = Vec::new();

        for _ in 0..5 {
            let id = mesh.add_vertex(Vertex::new([1.0; 3]));
            old_ids.push(id);
            mesh.remove_vertex(id);
        }

        // 所有旧句柄都应无效
        for id in &old_ids {
            assert!(!mesh.contains_vertex(*id));
            assert!(mesh.get_vertex(*id).is_none());
        }
        assert_eq!(mesh.vertex_count(), 0);

        // 再插入一个新顶点
        let new_id = mesh.add_vertex(Vertex::new([2.0; 3]));
        assert!(mesh.contains_vertex(new_id));
        for id in &old_ids {
            assert_ne!(*id, new_id);
        }
    }

    #[test]
    fn slot_reuse_halfedge_and_face() {
        // 半边和面同样有版本号机制
        let mut mesh = MeshStorage::new();

        let v = mesh.add_vertex(Vertex::new([0.0; 3]));
        let old_he = mesh.add_halfedge(HalfEdge::new(v));
        let old_face = mesh.add_face(Face::new());

        mesh.remove_halfedge(old_he);
        mesh.remove_face(old_face);

        let new_he = mesh.add_halfedge(HalfEdge::new(v));
        let new_face = mesh.add_face(Face::new());

        assert!(!mesh.contains_halfedge(old_he));
        assert!(!mesh.contains_face(old_face));
        assert!(mesh.contains_halfedge(new_he));
        assert!(mesh.contains_face(new_face));
        assert_ne!(old_he, new_he);
        assert_ne!(old_face, new_face);
    }

    // ---------- 默认值 / 构造 ----------

    #[test]
    fn vertex_default_is_origin() {
        let v = Vertex::default();
        assert_eq!(v.position, [0.0; 3]);
        assert!(v.halfedge.is_none());
    }

    #[test]
    fn face_default_has_no_halfedge() {
        let f = Face::default();
        assert!(f.halfedge.is_none());
    }

    #[test]
    fn halfedge_new_has_no_links() {
        let v = VertexId::default();
        let he = HalfEdge::new(v);
        assert_eq!(he.vertex, v);
        assert!(he.twin.is_none());
        assert!(he.next.is_none());
        assert!(he.prev.is_none());
        assert!(he.face.is_none());
    }

    // ---------- 迭代器一致性 ----------

    #[test]
    fn iteration_count_consistency_after_remove() {
        // 删除部分元素后，句柄迭代与数据迭代数量一致
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([0.0; 3]));
        let _v1 = mesh.add_vertex(Vertex::new([1.0; 3]));
        let _v2 = mesh.add_vertex(Vertex::new([2.0; 3]));

        mesh.remove_vertex(v0);

        assert_eq!(mesh.vertex_count(), 2);
        assert_eq!(mesh.vertex_ids().count(), 2);
        assert_eq!(mesh.vertices().count(), 2);
        // 删除的 v0 不出现在迭代中
        assert!(!mesh.vertex_ids().any(|id| id == v0));
    }

    #[test]
    fn halfedges_mut_allows_modification() {
        let mut mesh = MeshStorage::new();
        let v = mesh.add_vertex(Vertex::new([0.0; 3]));
        mesh.add_halfedge(HalfEdge::new(v));
        mesh.add_halfedge(HalfEdge::new(v));

        for he in mesh.halfedges_mut() {
            he.face = Some(FaceId::default());
        }
        for he in mesh.halfedges() {
            assert_eq!(he.face, Some(FaceId::default()));
        }
    }

    #[test]
    fn faces_mut_allows_modification() {
        let mut mesh = MeshStorage::new();
        let he = mesh.add_halfedge(HalfEdge::new(VertexId::default()));
        mesh.add_face(Face::new());
        mesh.add_face(Face::new());

        for f in mesh.faces_mut() {
            f.halfedge = Some(he);
        }
        for f in mesh.faces() {
            assert_eq!(f.halfedge, Some(he));
        }
    }

    // ---------- 欧拉示性数 / 亏格 ----------

    #[test]
    fn euler_characteristic_of_icosphere_is_two() {
        // 闭合球面网格 χ = 2
        let mesh = crate::test_util::build_icosphere(0);
        assert_eq!(mesh.euler_characteristic(), 2);
        assert_eq!(mesh.genus(), 0);
    }

    #[test]
    fn euler_characteristic_of_tetrahedron_is_two() {
        // 四面体：V=4, E=6, F=4 → χ = 4-6+4 = 2
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let faces = vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]];
        let mesh = crate::io::build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.face_count(), 4);
        assert_eq!(mesh.halfedge_count(), 12);
        assert_eq!(mesh.euler_characteristic(), 2);
        assert_eq!(mesh.genus(), 0);
    }

    #[test]
    fn euler_characteristic_empty_mesh() {
        let mesh = MeshStorage::new();
        assert_eq!(mesh.euler_characteristic(), 0);
    }

    // ---------- 边界检测 ----------

    #[test]
    fn closed_mesh_has_no_boundary() {
        // icosphere 是闭合网格
        let mesh = crate::test_util::build_icosphere(0);
        assert!(crate::traversal::is_closed(&mesh));
    }

    #[test]
    fn single_triangle_has_boundary() {
        // 单个三角形（有边界边）不是闭合网格
        let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![[0, 1, 2]];
        let mesh = crate::io::build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        assert!(!crate::traversal::is_closed(&mesh));
    }

    #[test]
    fn boundary_halfedges_have_no_face() {
        // 单三角形的 3 条边界半边 face 字段为 None
        let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![[0, 1, 2]];
        let mesh = crate::io::build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();

        let boundary_count = mesh
            .halfedge_ids()
            .filter(|he| mesh.get_halfedge(*he).unwrap().face.is_none())
            .count();
        assert_eq!(boundary_count, 3, "单三角形应有 3 条边界半边");
    }

    #[test]
    fn closed_mesh_has_no_boundary_halfedges() {
        let mesh = crate::test_util::build_icosphere(0);
        let boundary_count = mesh
            .halfedge_ids()
            .filter(|he| mesh.get_halfedge(*he).unwrap().face.is_none())
            .count();
        assert_eq!(boundary_count, 0);
    }

    // ---------- 校验集成 ----------

    #[test]
    fn icosphere_passes_full_validation() {
        let mesh = crate::test_util::build_icosphere(1);
        assert!(crate::validate::check_topology(&mesh).is_ok());
    }

    #[test]
    fn single_triangle_passes_validation() {
        let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![[0, 1, 2]];
        let mesh = crate::io::build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        assert!(crate::validate::check_topology(&mesh).is_ok());
    }

    #[test]
    fn two_disconnected_triangles_pass_validation() {
        // 两个不相连的三角形
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ];
        let faces = vec![[0, 1, 2], [3, 4, 5]];
        let mesh = crate::io::build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        assert_eq!(mesh.vertex_count(), 6);
        assert_eq!(mesh.face_count(), 2);
        assert_eq!(mesh.halfedge_count(), 12);
        assert!(crate::validate::check_topology(&mesh).is_ok());
    }

    // ---------- Twin / next / prev 链正确性 ----------

    #[test]
    fn twin_relationship_is_symmetric() {
        // 验证 twin 双向互指
        let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![[0, 1, 2]];
        let mesh = crate::io::build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();

        for he_id in mesh.halfedge_ids() {
            let he = mesh.get_halfedge(he_id).unwrap();
            if let Some(twin_id) = he.twin {
                let twin = mesh.get_halfedge(twin_id).unwrap();
                assert_eq!(twin.twin, Some(he_id), "twin 应双向互指");
            }
        }
    }

    #[test]
    fn next_prev_chain_is_consistent() {
        // 验证 next/prev 双向一致
        let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![[0, 1, 2]];
        let mesh = crate::io::build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();

        for he_id in mesh.halfedge_ids() {
            let he = mesh.get_halfedge(he_id).unwrap();
            if let Some(next_id) = he.next {
                let next = mesh.get_halfedge(next_id).unwrap();
                assert_eq!(next.prev, Some(he_id), "next.prev 应互指");
            }
            if let Some(prev_id) = he.prev {
                let prev = mesh.get_halfedge(prev_id).unwrap();
                assert_eq!(prev.next, Some(he_id), "prev.next 应互指");
            }
        }
    }

    #[test]
    fn face_boundary_next_chain_closes() {
        // 面的 next 链应闭合，长度为 3
        let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![[0, 1, 2]];
        let mesh = crate::io::build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();

        for f_id in mesh.face_ids() {
            let face = mesh.get_face(f_id).unwrap();
            let start = face.halfedge.unwrap();
            let he1 = start;
            let he2 = mesh.get_halfedge(he1).unwrap().next.unwrap();
            let he3 = mesh.get_halfedge(he2).unwrap().next.unwrap();
            let back = mesh.get_halfedge(he3).unwrap().next.unwrap();
            assert_eq!(back, he1, "next 链应回到起点");
            assert_ne!(he1, he2);
            assert_ne!(he2, he3);
            assert_ne!(he1, he3);
        }
    }

    // ---------- 大网格 ----------

    #[test]
    fn large_icosphere_counts_correct() {
        // subdiv=2: V=162, F=320, E=480 → 半边 960
        let mesh = crate::test_util::build_icosphere(2);
        assert_eq!(mesh.vertex_count(), 162);
        assert_eq!(mesh.face_count(), 320);
        assert_eq!(mesh.halfedge_count(), 960);
        assert_eq!(mesh.euler_characteristic(), 2);
        assert!(crate::validate::check_topology(&mesh).is_ok());
    }

    // ---------- MeshStorage Clone / Debug ----------

    #[test]
    fn mesh_storage_clone_is_equal() {
        let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![[0, 1, 2]];
        let mesh = crate::io::build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();

        let cloned = mesh.clone();
        assert_eq!(cloned.vertex_count(), mesh.vertex_count());
        assert_eq!(cloned.halfedge_count(), mesh.halfedge_count());
        assert_eq!(cloned.face_count(), mesh.face_count());
        assert_eq!(cloned.euler_characteristic(), mesh.euler_characteristic());
    }

    #[test]
    fn mesh_storage_debug_formats() {
        let mut mesh = MeshStorage::new();
        mesh.add_vertex(Vertex::new([1.0; 3]));
        let debug = format!("{:?}", mesh);
        assert!(debug.contains("MeshStorage"));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_roundtrip_preserves_topology() {
        // Build a simple triangle mesh
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let faces = vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]];
        let mesh =
            crate::io::build_mesh_from_vertices_and_faces(&verts, &faces).expect("valid mesh");

        let json = serde_json::to_string(&mesh).expect("serialize");
        let deserialized: MeshStorage = serde_json::from_str(&json).expect("deserialize");

        // Topology preserved
        assert_eq!(deserialized.vertex_count(), mesh.vertex_count());
        assert_eq!(deserialized.halfedge_count(), mesh.halfedge_count());
        assert_eq!(deserialized.face_count(), mesh.face_count());
        assert_eq!(
            deserialized.euler_characteristic(),
            mesh.euler_characteristic()
        );

        // Positions preserved
        let orig_pos: Vec<[f64; 3]> = mesh.vertices().map(|v| v.position).collect();
        let new_pos: Vec<[f64; 3]> = deserialized.vertices().map(|v| v.position).collect();
        assert_eq!(orig_pos.len(), new_pos.len());
        for (a, b) in orig_pos.iter().zip(new_pos.iter()) {
            assert_eq!(a, b);
        }
    }

    // ---------- SOA 位置缓存 ----------

    #[test]
    fn soa_cache_get_position_matches_vertex() {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([1.0, 2.0, 3.0]));
        let v1 = mesh.add_vertex(Vertex::new([4.0, 5.0, 6.0]));

        // get_position 应与 get_vertex().position 一致
        assert_eq!(mesh.get_position(v0), Some([1.0, 2.0, 3.0]));
        assert_eq!(mesh.get_position(v1), Some([4.0, 5.0, 6.0]));
        // 无效句柄返回 None
        let fake = VertexId::default();
        assert_eq!(mesh.get_position(fake), None);
    }

    #[test]
    fn soa_cache_positions_dense_length_matches() {
        let mut mesh = MeshStorage::new();
        mesh.add_vertex(Vertex::new([1.0; 3]));
        mesh.add_vertex(Vertex::new([2.0; 3]));
        mesh.add_vertex(Vertex::new([3.0; 3]));

        let dense = mesh.positions_dense();
        assert_eq!(dense.len(), 3);
        assert_eq!(dense[0], [1.0; 3]);
        assert_eq!(dense[1], [2.0; 3]);
        assert_eq!(dense[2], [3.0; 3]);
    }

    #[test]
    fn soa_cache_position_index_round_trip() {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([10.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(Vertex::new([0.0, 20.0, 0.0]));

        let dense = mesh.positions_dense();
        let idx0 = mesh.position_index(v0).expect("v0 应有索引");
        let idx1 = mesh.position_index(v1).expect("v1 应有索引");

        assert_eq!(dense[idx0 as usize], [10.0, 0.0, 0.0]);
        assert_eq!(dense[idx1 as usize], [0.0, 20.0, 0.0]);
    }

    #[test]
    fn soa_cache_set_position_syncs_both() {
        let mut mesh = MeshStorage::new();
        let v = mesh.add_vertex(Vertex::new([0.0; 3]));

        mesh.set_position(v, [7.0, 8.0, 9.0]);

        // 主存与缓存都应更新
        assert_eq!(mesh.get_vertex(v).unwrap().position, [7.0, 8.0, 9.0]);
        assert_eq!(mesh.get_position(v), Some([7.0, 8.0, 9.0]));
        let idx = mesh.position_index(v).unwrap();
        assert_eq!(mesh.positions_dense()[idx as usize], [7.0, 8.0, 9.0]);
    }

    #[test]
    fn soa_cache_remove_preserves_dense_layout() {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([1.0; 3]));
        let _v1 = mesh.add_vertex(Vertex::new([2.0; 3]));
        let v2 = mesh.add_vertex(Vertex::new([3.0; 3]));

        // 删除中间顶点 v1，末尾 v2 的位置会填补到 v1 的索引
        mesh.remove_vertex(_v1);

        // 剩余顶点的位置应仍可通过 get_position 正确读取
        assert_eq!(mesh.get_position(v0), Some([1.0; 3]));
        assert_eq!(mesh.get_position(v2), Some([3.0; 3]));
        // 稠密切片长度应为 2
        assert_eq!(mesh.positions_dense().len(), 2);
    }

    #[test]
    fn soa_cache_sync_position_after_get_vertex_mut() {
        let mut mesh = MeshStorage::new();
        let v = mesh.add_vertex(Vertex::new([0.0; 3]));

        // 通过 get_vertex_mut 直接修改 position（绕过 set_position）
        mesh.get_vertex_mut(v).unwrap().position = [5.0, 5.0, 5.0];
        // 此时缓存已过期
        assert_eq!(mesh.get_position(v), Some([0.0; 3])); // 仍是旧值

        // 调用 sync_position 修复
        mesh.sync_position(v);
        assert_eq!(mesh.get_position(v), Some([5.0, 5.0, 5.0]));
    }

    #[test]
    fn soa_cache_rebuild_from_master() {
        let mut mesh = MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([1.0; 3]));
        let v1 = mesh.add_vertex(Vertex::new([2.0; 3]));

        // 模拟缓存损坏：直接修改 Vertex.position 但不同步缓存
        mesh.get_vertex_mut(v0).unwrap().position = [9.0; 3];
        mesh.get_vertex_mut(v1).unwrap().position = [8.0; 3];

        // 重建缓存
        mesh.rebuild_position_cache();

        assert_eq!(mesh.get_position(v0), Some([9.0; 3]));
        assert_eq!(mesh.get_position(v1), Some([8.0; 3]));
        assert_eq!(mesh.positions_dense().len(), 2);
    }

    #[test]
    fn soa_cache_clear_empties_cache() {
        let mut mesh = MeshStorage::new();
        mesh.add_vertex(Vertex::new([1.0; 3]));
        mesh.add_vertex(Vertex::new([2.0; 3]));
        assert_eq!(mesh.positions_dense().len(), 2);

        mesh.clear();
        assert_eq!(mesh.positions_dense().len(), 0);
    }

    #[test]
    fn soa_cache_empty_mesh_no_panic() {
        let mesh = MeshStorage::new();
        assert_eq!(mesh.positions_dense().len(), 0);
        assert_eq!(mesh.positions_dense(), &[] as &[[f64; 3]]);
    }

    #[test]
    fn soa_cache_icosphere_consistency() {
        use crate::test_util::build_icosphere;
        let mesh = build_icosphere(1);
        // 对所有顶点验证 get_position 与 get_vertex().position 一致
        for v_id in mesh.vertex_ids() {
            let from_vertex = mesh.get_vertex(v_id).unwrap().position;
            let from_cache = mesh.get_position(v_id).unwrap();
            assert_eq!(from_vertex, from_cache, "顶点 {:?} 缓存不一致", v_id);
        }
    }
}
