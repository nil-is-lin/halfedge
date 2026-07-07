//! 属性系统模块
//!
//! 提供 OpenMesh 风格的属性系统，允许为顶点/半边/面附加任意类型的自定义数据。
//! 使用 `Any + TypeId` 实现类型擦除，不依赖 trait 泛型。
//!
//! ## 设计
//!
//! - [`PropertyStore<T>`]：单个属性的存储容器，按 slotmap key 的 index 部分索引。
//! - [`PropertyHandle<T>`]：类型化句柄，用于类型安全的属性访问。
//! - [`MeshProperties`]：管理所有属性的容器，按 `TypeId` 区分不同类型。
//!
//! ## 类型擦除机制
//!
//! 每个属性类型 `T: 'static` 对应一个唯一的 `TypeId`。`MeshProperties` 内部用
//! `HashMap<TypeId, Box<dyn ErasedProperty>>` 存储，通过 `TypeId::of::<T>()` 查找。
//! 私有 trait `ErasedProperty` 提供 `as_any()` 方法用于 downcast 回具体类型，
//! 同时提供 `remove_by_index()` 用于在不知道具体类型的情况下删除属性。
//!
//! ## 存储策略
//!
//! `PropertyStore<T>` 内部使用 `Vec<Option<T>>` 按 slotmap key 的 32 位 index
//! 直接索引，提供：
//! - **O(1) 直接索引**（无哈希计算）
//! - **连续内存布局**（更好的缓存局部性）
//! - **自动稀疏扩展**（按需扩容，未设置的槽位为 `None`）
//!
//! ## 限制
//!
//! - 每个类型 `T` 在每类（vertex/halfedge/face）中只能注册一个属性。
//!   如需多个同类型属性，可用 newtype 包装：`struct Weight(f64); struct Temp(f64);`
//! - 属性键为 slotmap key 的 **32 位 index** 部分。
//!   slot 被复用时旧属性会残留在同一 index，**务必在删除元素时调用**
//!   [`remove_vertex_all_props`](MeshProperties::remove_vertex_all_props)
//!   等方法清理，否则新元素可能继承旧属性值。
//!
//! ## 使用示例
//!
//! ```
//! use halfedge::property::MeshProperties;
//! use halfedge::{build_mesh_from_vertices_and_faces};
//!
//! let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
//! let faces = vec![[0, 1, 2]];
//! let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
//!
//! let mut props = MeshProperties::new();
//! let w = props.add_vertex_prop::<f64>();
//! for v in mesh.vertex_ids() {
//!     props.set_vertex_prop(w, v, 1.0);
//! }
//! ```

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::marker::PhantomData;

use slotmap::Key;

use crate::ids::{FaceId, HalfEdgeId, VertexId};
use crate::storage::{Face, HalfEdge, MeshStorage, Vertex};

// ============================================================
// 内部辅助：从 slotmap key 提取索引
// ============================================================

/// 从 slotmap key 提取 32 位 index 部分作为 `usize`。
///
/// slotmap 的 `as_ffi()` 编码为 `((version) << 32) | idx`，
/// 此函数仅取低 32 位 idx，用于 `Vec` 直接索引。
///
/// **注意**：使用 index 而非完整 64 位键意味着 slot 被复用时
/// 旧属性会残留在同一 index。务必在删除元素时调用
/// `remove_*_all_props` 清理属性。
#[inline]
fn idx_of<K: Key>(id: K) -> usize {
    id.data().as_ffi() as u32 as usize
}

// ============================================================
// PropertyStore：单个属性的存储容器
// ============================================================

/// 单个属性的存储容器，按 slotmap key 的 32 位 index 直接索引。
///
/// 使用 `Vec<Option<T>>` 而非 `HashMap<usize, T>`，优势：
/// 1. **O(1) 直接索引**：无需哈希计算，`data[idx]` 直接访问；
/// 2. **缓存友好**：连续内存布局，遍历和随机访问都受益于 CPU 缓存；
/// 3. **低开销**：`Option<T>` 的 None 变体仅占少量空间（取决于 T 的对齐）。
///
/// 稀疏场景（大部分元素未设置属性）下，Vec 会浪费一些空间存储 `None`，
/// 但在现代 CPU 上连续访问的速度优势通常超过空间开销。
pub struct PropertyStore<T> {
    data: Vec<Option<T>>,
}

impl<T> PropertyStore<T> {
    /// 创建空的属性存储。
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// 确保 Vec 容量足够容纳 `idx`，不足则扩容并填充 `None`。
    #[inline]
    fn ensure_capacity(&mut self, idx: usize) {
        if idx >= self.data.len() {
            self.data.resize_with(idx + 1, || None);
        }
    }

    /// 获取指定索引的属性引用。
    pub fn get(&self, idx: usize) -> Option<&T> {
        self.data.get(idx).and_then(|opt| opt.as_ref())
    }

    /// 获取指定索引的属性可变引用。
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut T> {
        self.data.get_mut(idx).and_then(|opt| opt.as_mut())
    }

    /// 设置指定索引的属性值。若已存在则覆盖。
    pub fn set(&mut self, idx: usize, val: T) {
        self.ensure_capacity(idx);
        self.data[idx] = Some(val);
    }

    /// 删除指定索引的属性值，返回被删除的值。
    pub fn remove(&mut self, idx: usize) -> Option<T> {
        if idx < self.data.len() {
            self.data[idx].take()
        } else {
            None
        }
    }

    /// 是否包含指定索引的属性。
    pub fn contains(&self, idx: usize) -> bool {
        self.get(idx).is_some()
    }

    /// 当前存储的属性数量（非 None 条目数）。
    pub fn len(&self) -> usize {
        self.data.iter().filter(|opt| opt.is_some()).count()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 清空所有属性。
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// 迭代所有 `(index, &value)` 对。
    pub fn iter(&self) -> impl Iterator<Item = (usize, &T)> {
        self.data
            .iter()
            .enumerate()
            .filter_map(|(i, opt)| opt.as_ref().map(|v| (i, v)))
    }
}

impl<T> Default for PropertyStore<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// ErasedProperty：类型擦除 trait（私有）
// ============================================================

/// 类型擦除的属性存储 trait。
///
/// 提供 `as_any()` 用于 downcast 回具体类型，以及 `remove_by_index()` 用于
/// 在不知道具体类型的情况下删除属性（用于 `remove_*_all_props`）。
trait ErasedProperty {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn remove_by_index(&mut self, idx: usize);
    fn clear_all(&mut self);
}

impl<T: 'static> ErasedProperty for PropertyStore<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn remove_by_index(&mut self, idx: usize) {
        if idx < self.data.len() {
            self.data[idx] = None;
        }
    }

    fn clear_all(&mut self) {
        self.data.clear();
    }
}

// ============================================================
// PropertyHandle：类型化句柄
// ============================================================

/// 类型化属性句柄，用于类型安全的属性访问。
///
/// 由于属性按 `TypeId` 索引，句柄本身不携带数据，仅用于在编译期关联
/// 属性类型 `T`，使 `get/set/remove` 方法无需显式标注泛型参数。
///
/// 句柄是 `Copy` 的，可以自由复制。
pub struct PropertyHandle<T> {
    _marker: PhantomData<T>,
}

impl<T> PropertyHandle<T> {
    /// 创建新句柄。通常通过 `add_vertex_prop` 等方法获得。
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Default for PropertyHandle<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for PropertyHandle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for PropertyHandle<T> {}

impl<T> std::fmt::Debug for PropertyHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PropertyHandle")
            .field("type", &std::any::type_name::<T>())
            .finish()
    }
}

// ============================================================
// MeshProperties：属性容器
// ============================================================

/// 属性容器，管理顶点/半边/面的所有自定义属性。
///
/// 三类属性分别存储在独立的 `HashMap` 中，按 `TypeId` 区分不同类型。
/// 每个类型 `T: 'static` 在每类中只能注册一个属性。
///
/// # 使用流程
///
/// 1. 调用 [`add_vertex_prop`](Self::add_vertex_prop)`::<T>()` 注册属性，获得 `PropertyHandle<T>`；
/// 2. 使用 [`set_vertex_prop`](Self::set_vertex_prop)`(handle, id, val)` 设置属性值；
/// 3. 使用 [`get_vertex_prop`](Self::get_vertex_prop)`(handle, id)` 读取属性值；
/// 4. 删除元素时调用 [`remove_vertex_all_props`](Self::remove_vertex_all_props)`(id)` 清理属性。
pub struct MeshProperties {
    vertex_props: HashMap<TypeId, Box<dyn ErasedProperty>>,
    halfedge_props: HashMap<TypeId, Box<dyn ErasedProperty>>,
    face_props: HashMap<TypeId, Box<dyn ErasedProperty>>,
}

impl MeshProperties {
    /// 创建空的属性容器。
    pub fn new() -> Self {
        Self {
            vertex_props: HashMap::new(),
            halfedge_props: HashMap::new(),
            face_props: HashMap::new(),
        }
    }

    // ---------- 统计查询 ----------

    /// 已注册的顶点属性类型数。
    pub fn vertex_prop_type_count(&self) -> usize {
        self.vertex_props.len()
    }

    /// 已注册的半边属性类型数。
    pub fn halfedge_prop_type_count(&self) -> usize {
        self.halfedge_props.len()
    }

    /// 已注册的面属性类型数。
    pub fn face_prop_type_count(&self) -> usize {
        self.face_props.len()
    }

    /// 是否注册了类型为 `T` 的顶点属性。
    pub fn has_vertex_prop<T: 'static>(&self) -> bool {
        self.vertex_props.contains_key(&TypeId::of::<T>())
    }

    /// 是否注册了类型为 `T` 的半边属性。
    pub fn has_halfedge_prop<T: 'static>(&self) -> bool {
        self.halfedge_props.contains_key(&TypeId::of::<T>())
    }

    /// 是否注册了类型为 `T` 的面属性。
    pub fn has_face_prop<T: 'static>(&self) -> bool {
        self.face_props.contains_key(&TypeId::of::<T>())
    }

    // ---------- Vertex 属性 ----------

    /// 注册顶点属性类型 `T`，返回类型化句柄。
    ///
    /// 若该类型已注册，覆盖旧存储（已有数据丢失）。
    pub fn add_vertex_prop<T: 'static>(&mut self) -> PropertyHandle<T> {
        self.vertex_props
            .insert(TypeId::of::<T>(), Box::new(PropertyStore::<T>::new()));
        PropertyHandle::new()
    }

    /// 读取顶点 `id` 的属性值。
    ///
    /// 若属性未注册或该顶点未设置属性，返回 `None`。
    pub fn get_vertex_prop<T: 'static>(
        &self,
        _handle: PropertyHandle<T>,
        id: VertexId,
    ) -> Option<&T> {
        let entry = self.vertex_props.get(&TypeId::of::<T>())?;
        let store = entry.as_any().downcast_ref::<PropertyStore<T>>()?;
        store.get(idx_of(id))
    }

    /// 读取顶点 `id` 的属性值（可变引用）。
    pub fn get_vertex_prop_mut<T: 'static>(
        &mut self,
        _handle: PropertyHandle<T>,
        id: VertexId,
    ) -> Option<&mut T> {
        let entry = self.vertex_props.get_mut(&TypeId::of::<T>())?;
        let store = entry.as_any_mut().downcast_mut::<PropertyStore<T>>()?;
        store.get_mut(idx_of(id))
    }

    /// 设置顶点 `id` 的属性值。若属性未注册，自动注册。
    pub fn set_vertex_prop<T: 'static>(
        &mut self,
        _handle: PropertyHandle<T>,
        id: VertexId,
        val: T,
    ) {
        let type_id = TypeId::of::<T>();
        let entry = self
            .vertex_props
            .get_mut(&type_id)
            .and_then(|e| e.as_any_mut().downcast_mut::<PropertyStore<T>>());
        match entry {
            Some(store) => store.set(idx_of(id), val),
            None => {
                let mut store = PropertyStore::<T>::new();
                store.set(idx_of(id), val);
                self.vertex_props.insert(type_id, Box::new(store));
            }
        }
    }

    /// 删除顶点 `id` 的指定类型属性，返回被删除的值。
    pub fn remove_vertex_prop<T: 'static>(
        &mut self,
        _handle: PropertyHandle<T>,
        id: VertexId,
    ) -> Option<T> {
        let entry = self.vertex_props.get_mut(&TypeId::of::<T>())?;
        let store = entry.as_any_mut().downcast_mut::<PropertyStore<T>>()?;
        store.remove(idx_of(id))
    }

    /// 删除顶点 `id` 的**所有类型**属性。
    ///
    /// 应在 `mesh.remove_vertex(id)` 后调用以保持属性与网格一致。
    pub fn remove_vertex_all_props(&mut self, id: VertexId) {
        let idx = idx_of(id);
        for store in self.vertex_props.values_mut() {
            store.remove_by_index(idx);
        }
    }

    // ---------- HalfEdge 属性 ----------

    /// 注册半边属性类型 `T`，返回类型化句柄。
    pub fn add_halfedge_prop<T: 'static>(&mut self) -> PropertyHandle<T> {
        self.halfedge_props
            .insert(TypeId::of::<T>(), Box::new(PropertyStore::<T>::new()));
        PropertyHandle::new()
    }

    /// 读取半边 `id` 的属性值。
    pub fn get_halfedge_prop<T: 'static>(
        &self,
        _handle: PropertyHandle<T>,
        id: HalfEdgeId,
    ) -> Option<&T> {
        let entry = self.halfedge_props.get(&TypeId::of::<T>())?;
        let store = entry.as_any().downcast_ref::<PropertyStore<T>>()?;
        store.get(idx_of(id))
    }

    /// 读取半边 `id` 的属性值（可变引用）。
    pub fn get_halfedge_prop_mut<T: 'static>(
        &mut self,
        _handle: PropertyHandle<T>,
        id: HalfEdgeId,
    ) -> Option<&mut T> {
        let entry = self.halfedge_props.get_mut(&TypeId::of::<T>())?;
        let store = entry.as_any_mut().downcast_mut::<PropertyStore<T>>()?;
        store.get_mut(idx_of(id))
    }

    /// 设置半边 `id` 的属性值。若属性未注册，自动注册。
    pub fn set_halfedge_prop<T: 'static>(
        &mut self,
        _handle: PropertyHandle<T>,
        id: HalfEdgeId,
        val: T,
    ) {
        let type_id = TypeId::of::<T>();
        let entry = self
            .halfedge_props
            .get_mut(&type_id)
            .and_then(|e| e.as_any_mut().downcast_mut::<PropertyStore<T>>());
        match entry {
            Some(store) => store.set(idx_of(id), val),
            None => {
                let mut store = PropertyStore::<T>::new();
                store.set(idx_of(id), val);
                self.halfedge_props.insert(type_id, Box::new(store));
            }
        }
    }

    /// 删除半边 `id` 的指定类型属性。
    pub fn remove_halfedge_prop<T: 'static>(
        &mut self,
        _handle: PropertyHandle<T>,
        id: HalfEdgeId,
    ) -> Option<T> {
        let entry = self.halfedge_props.get_mut(&TypeId::of::<T>())?;
        let store = entry.as_any_mut().downcast_mut::<PropertyStore<T>>()?;
        store.remove(idx_of(id))
    }

    /// 删除半边 `id` 的**所有类型**属性。
    pub fn remove_halfedge_all_props(&mut self, id: HalfEdgeId) {
        let idx = idx_of(id);
        for store in self.halfedge_props.values_mut() {
            store.remove_by_index(idx);
        }
    }

    // ---------- Face 属性 ----------

    /// 注册面属性类型 `T`，返回类型化句柄。
    pub fn add_face_prop<T: 'static>(&mut self) -> PropertyHandle<T> {
        self.face_props
            .insert(TypeId::of::<T>(), Box::new(PropertyStore::<T>::new()));
        PropertyHandle::new()
    }

    /// 读取面 `id` 的属性值。
    pub fn get_face_prop<T: 'static>(&self, _handle: PropertyHandle<T>, id: FaceId) -> Option<&T> {
        let entry = self.face_props.get(&TypeId::of::<T>())?;
        let store = entry.as_any().downcast_ref::<PropertyStore<T>>()?;
        store.get(idx_of(id))
    }

    /// 读取面 `id` 的属性值（可变引用）。
    pub fn get_face_prop_mut<T: 'static>(
        &mut self,
        _handle: PropertyHandle<T>,
        id: FaceId,
    ) -> Option<&mut T> {
        let entry = self.face_props.get_mut(&TypeId::of::<T>())?;
        let store = entry.as_any_mut().downcast_mut::<PropertyStore<T>>()?;
        store.get_mut(idx_of(id))
    }

    /// 设置面 `id` 的属性值。若属性未注册，自动注册。
    pub fn set_face_prop<T: 'static>(&mut self, _handle: PropertyHandle<T>, id: FaceId, val: T) {
        let type_id = TypeId::of::<T>();
        let entry = self
            .face_props
            .get_mut(&type_id)
            .and_then(|e| e.as_any_mut().downcast_mut::<PropertyStore<T>>());
        match entry {
            Some(store) => store.set(idx_of(id), val),
            None => {
                let mut store = PropertyStore::<T>::new();
                store.set(idx_of(id), val);
                self.face_props.insert(type_id, Box::new(store));
            }
        }
    }

    /// 删除面 `id` 的指定类型属性。
    pub fn remove_face_prop<T: 'static>(
        &mut self,
        _handle: PropertyHandle<T>,
        id: FaceId,
    ) -> Option<T> {
        let entry = self.face_props.get_mut(&TypeId::of::<T>())?;
        let store = entry.as_any_mut().downcast_mut::<PropertyStore<T>>()?;
        store.remove(idx_of(id))
    }

    /// 删除面 `id` 的**所有类型**属性。
    pub fn remove_face_all_props(&mut self, id: FaceId) {
        let idx = idx_of(id);
        for store in self.face_props.values_mut() {
            store.remove_by_index(idx);
        }
    }

    // ---------- 清理 ----------

    /// 清空所有顶点属性存储（保留已注册的类型槽位）。
    pub fn clear_vertex_props(&mut self) {
        for store in self.vertex_props.values_mut() {
            store.clear_all();
        }
    }

    /// 清空所有半边属性存储。
    pub fn clear_halfedge_props(&mut self) {
        for store in self.halfedge_props.values_mut() {
            store.clear_all();
        }
    }

    /// 清空所有面属性存储。
    pub fn clear_face_props(&mut self) {
        for store in self.face_props.values_mut() {
            store.clear_all();
        }
    }
}

impl Default for MeshProperties {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MeshProperties {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshProperties")
            .field("vertex_prop_types", &self.vertex_props.len())
            .field("halfedge_prop_types", &self.halfedge_props.len())
            .field("face_prop_types", &self.face_props.len())
            .finish()
    }
}

// ============================================================
// 便捷包装函数：删除元素时同步清理属性
// ============================================================

/// 删除顶点并同步清理其在 `props` 中的所有属性。
///
/// 等价于先 `mesh.remove_vertex(id)` 再 `props.remove_vertex_all_props(id)`。
pub fn remove_vertex_with_props(
    mesh: &mut MeshStorage,
    props: &mut MeshProperties,
    id: VertexId,
) -> Option<Vertex> {
    let result = mesh.remove_vertex(id);
    props.remove_vertex_all_props(id);
    result
}

/// 删除半边并同步清理其在 `props` 中的所有属性。
pub fn remove_halfedge_with_props(
    mesh: &mut MeshStorage,
    props: &mut MeshProperties,
    id: HalfEdgeId,
) -> Option<HalfEdge> {
    let result = mesh.remove_halfedge(id);
    props.remove_halfedge_all_props(id);
    result
}

/// 删除面并同步清理其在 `props` 中的所有属性。
pub fn remove_face_with_props(
    mesh: &mut MeshStorage,
    props: &mut MeshProperties,
    id: FaceId,
) -> Option<Face> {
    let result = mesh.remove_face(id);
    props.remove_face_all_props(id);
    result
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::build_mesh_from_vertices_and_faces;

    // ---------- PropertyStore 基础测试 ----------

    #[test]
    fn property_store_basic_crud() {
        let mut store = PropertyStore::<f64>::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        store.set(0, 1.5);
        store.set(1, 2.5);
        store.set(2, 3.5);

        assert_eq!(store.len(), 3);
        assert!(!store.is_empty());
        assert!(store.contains(1));
        assert!(!store.contains(99));

        assert_eq!(store.get(0), Some(&1.5));
        assert_eq!(store.get(1), Some(&2.5));
        assert_eq!(store.get(99), None);

        *store.get_mut(1).unwrap() = 9.9;
        assert_eq!(store.get(1), Some(&9.9));

        assert_eq!(store.remove(1), Some(9.9));
        assert_eq!(store.remove(1), None);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn property_store_iter() {
        let mut store = PropertyStore::<String>::new();
        store.set(0, "a".to_string());
        store.set(5, "b".to_string());
        store.set(10, "c".to_string());

        let mut entries: Vec<(usize, String)> = store.iter().map(|(i, v)| (i, v.clone())).collect();
        entries.sort_by_key(|(i, _)| *i);
        assert_eq!(
            entries,
            vec![
                (0, "a".to_string()),
                (5, "b".to_string()),
                (10, "c".to_string())
            ]
        );
    }

    #[test]
    fn property_store_clear() {
        let mut store = PropertyStore::<i32>::new();
        store.set(0, 1);
        store.set(1, 2);
        store.clear();
        assert!(store.is_empty());
        assert_eq!(store.get(0), None);
    }

    // ---------- MeshProperties 类型擦除测试 ----------

    #[test]
    fn add_and_get_vertex_prop() {
        let (_, vids) = build_triangle_mesh();
        let mut props = MeshProperties::new();

        let h = props.add_vertex_prop::<f64>();
        assert!(props.has_vertex_prop::<f64>());
        assert_eq!(props.vertex_prop_type_count(), 1);

        props.set_vertex_prop(h, vids[0], 1.5);
        props.set_vertex_prop(h, vids[1], 2.5);
        props.set_vertex_prop(h, vids[2], 3.5);

        assert_eq!(props.get_vertex_prop(h, vids[0]), Some(&1.5));
        assert_eq!(props.get_vertex_prop(h, vids[1]), Some(&2.5));
        assert_eq!(props.get_vertex_prop(h, vids[2]), Some(&3.5));
    }

    #[test]
    fn get_vertex_prop_mut_works() {
        let (_, vids) = build_triangle_mesh();
        let mut props = MeshProperties::new();

        let h = props.add_vertex_prop::<f64>();
        props.set_vertex_prop(h, vids[0], 1.0);

        if let Some(val) = props.get_vertex_prop_mut(h, vids[0]) {
            *val = 10.0;
        }
        assert_eq!(props.get_vertex_prop(h, vids[0]), Some(&10.0));
    }

    #[test]
    fn remove_vertex_prop() {
        let (_, vids) = build_triangle_mesh();
        let mut props = MeshProperties::new();

        let h = props.add_vertex_prop::<f64>();
        props.set_vertex_prop(h, vids[0], 1.5);

        let removed = props.remove_vertex_prop(h, vids[0]);
        assert_eq!(removed, Some(1.5));
        assert_eq!(props.get_vertex_prop(h, vids[0]), None);

        // 再次删除返回 None
        assert_eq!(props.remove_vertex_prop(h, vids[0]), None);
    }

    #[test]
    fn remove_vertex_all_props_clears_all_types() {
        let (_, vids) = build_triangle_mesh();
        let mut props = MeshProperties::new();

        let h1 = props.add_vertex_prop::<f64>();
        let h2 = props.add_vertex_prop::<i32>();
        let h3 = props.add_vertex_prop::<String>();

        props.set_vertex_prop(h1, vids[0], 1.5);
        props.set_vertex_prop(h2, vids[0], 42);
        props.set_vertex_prop(h3, vids[0], "hello".to_string());

        props.set_vertex_prop(h1, vids[1], 2.5);
        props.set_vertex_prop(h2, vids[1], 99);

        // 删除 vids[0] 的所有属性
        props.remove_vertex_all_props(vids[0]);

        // vids[0] 的所有属性都应被清除
        assert_eq!(props.get_vertex_prop(h1, vids[0]), None);
        assert_eq!(props.get_vertex_prop(h2, vids[0]), None);
        assert_eq!(props.get_vertex_prop(h3, vids[0]), None);

        // vids[1] 的属性应保留
        assert_eq!(props.get_vertex_prop(h1, vids[1]), Some(&2.5));
        assert_eq!(props.get_vertex_prop(h2, vids[1]), Some(&99));
    }

    #[test]
    fn different_types_do_not_interfere() {
        let (_, vids) = build_triangle_mesh();
        let mut props = MeshProperties::new();

        let h_f64 = props.add_vertex_prop::<f64>();
        let h_i32 = props.add_vertex_prop::<i32>();
        let h_string = props.add_vertex_prop::<String>();

        props.set_vertex_prop(h_f64, vids[0], 1.5);
        props.set_vertex_prop(h_i32, vids[0], 42);
        props.set_vertex_prop(h_string, vids[0], "abc".to_string());

        assert_eq!(props.get_vertex_prop(h_f64, vids[0]), Some(&1.5));
        assert_eq!(props.get_vertex_prop(h_i32, vids[0]), Some(&42));
        assert_eq!(
            props.get_vertex_prop(h_string, vids[0]),
            Some(&"abc".to_string())
        );

        // 删除 f64 不影响 i32
        props.remove_vertex_prop(h_f64, vids[0]);
        assert_eq!(props.get_vertex_prop(h_f64, vids[0]), None);
        assert_eq!(props.get_vertex_prop(h_i32, vids[0]), Some(&42));
        assert_eq!(
            props.get_vertex_prop(h_string, vids[0]),
            Some(&"abc".to_string())
        );
    }

    #[test]
    fn vertex_halfedge_face_props_are_independent() {
        let (mesh, vids) = build_triangle_mesh();
        let fids: Vec<FaceId> = mesh.face_ids().collect();
        let mut props = MeshProperties::new();

        // f64 同时注册到 vertex 和 face
        let hv = props.add_vertex_prop::<f64>();
        let hf = props.add_face_prop::<f64>();

        props.set_vertex_prop(hv, vids[0], 1.0);
        props.set_face_prop(hf, fids[0], 99.0);

        // 互不干扰
        assert_eq!(props.get_vertex_prop(hv, vids[0]), Some(&1.0));
        assert_eq!(props.get_face_prop(hf, fids[0]), Some(&99.0));

        // vertex 属性不应出现在 face 中
        assert_eq!(props.vertex_prop_type_count(), 1);
        assert_eq!(props.face_prop_type_count(), 1);
    }

    #[test]
    fn get_unregistered_prop_returns_none() {
        let (_, vids) = build_triangle_mesh();
        let props = MeshProperties::new();

        let h: PropertyHandle<f64> = PropertyHandle::new();
        assert_eq!(props.get_vertex_prop(h, vids[0]), None);
    }

    #[test]
    fn set_auto_registers_prop() {
        let (_, vids) = build_triangle_mesh();
        let mut props = MeshProperties::new();

        // 不调用 add_vertex_prop，直接 set
        let h: PropertyHandle<f64> = PropertyHandle::new();
        props.set_vertex_prop(h, vids[0], 42.0);

        assert!(props.has_vertex_prop::<f64>());
        assert_eq!(props.get_vertex_prop(h, vids[0]), Some(&42.0));
    }

    #[test]
    fn halfedge_and_face_props_work() {
        let (mesh, _vids) = build_triangle_mesh();
        let fids: Vec<FaceId> = mesh.face_ids().collect();
        let heids: Vec<HalfEdgeId> = mesh.halfedge_ids().collect();

        let mut props = MeshProperties::new();

        let hh = props.add_halfedge_prop::<bool>();
        let hf = props.add_face_prop::<u32>();

        props.set_halfedge_prop(hh, heids[0], true);
        props.set_halfedge_prop(hh, heids[1], false);
        props.set_face_prop(hf, fids[0], 7);

        assert_eq!(props.get_halfedge_prop(hh, heids[0]), Some(&true));
        assert_eq!(props.get_halfedge_prop(hh, heids[1]), Some(&false));
        assert_eq!(props.get_face_prop(hf, fids[0]), Some(&7));

        // 批量删除
        props.remove_halfedge_all_props(heids[0]);
        assert_eq!(props.get_halfedge_prop(hh, heids[0]), None);
        assert_eq!(props.get_halfedge_prop(hh, heids[1]), Some(&false));

        props.remove_face_all_props(fids[0]);
        assert_eq!(props.get_face_prop(hf, fids[0]), None);
    }

    #[test]
    fn remove_vertex_with_props_wrapper() {
        let (mut mesh, vids) = build_triangle_mesh();
        let mut props = MeshProperties::new();

        let h = props.add_vertex_prop::<f64>();
        props.set_vertex_prop(h, vids[0], 1.0);
        props.set_vertex_prop(h, vids[1], 2.0);

        // 删除 vids[0] 并同步清理属性
        let removed = remove_vertex_with_props(&mut mesh, &mut props, vids[0]);
        assert!(removed.is_some());
        assert!(!mesh.contains_vertex(vids[0]));

        // vids[0] 的属性已清除
        assert_eq!(props.get_vertex_prop(h, vids[0]), None);
        // vids[1] 的属性保留
        assert_eq!(props.get_vertex_prop(h, vids[1]), Some(&2.0));
    }

    #[test]
    fn property_handle_is_copy_and_debug() {
        let h1: PropertyHandle<f64> = PropertyHandle::new();
        let _h2 = h1; // Copy
        let _h3 = h1; // 再次 Copy

        let debug_str = format!("{:?}", h1);
        assert!(debug_str.contains("PropertyHandle"));
        assert!(debug_str.contains("f64"));
    }

    #[test]
    fn newtype_wrapper_for_multiple_same_type_props() {
        // 使用 newtype 模式为同类型创建多个属性
        #[derive(Debug, PartialEq)]
        struct Weight(f64);
        #[derive(Debug, PartialEq)]
        struct Temperature(f64);

        let (_, vids) = build_triangle_mesh();
        let mut props = MeshProperties::new();

        let hw = props.add_vertex_prop::<Weight>();
        let ht = props.add_vertex_prop::<Temperature>();

        props.set_vertex_prop(hw, vids[0], Weight(1.0));
        props.set_vertex_prop(ht, vids[0], Temperature(25.5));

        assert_eq!(props.get_vertex_prop(hw, vids[0]), Some(&Weight(1.0)));
        assert_eq!(props.get_vertex_prop(ht, vids[0]), Some(&Temperature(25.5)));
        assert_eq!(props.vertex_prop_type_count(), 2);
    }

    #[test]
    fn clear_all_vertex_props() {
        let (_, vids) = build_triangle_mesh();
        let mut props = MeshProperties::new();

        let h1 = props.add_vertex_prop::<f64>();
        let h2 = props.add_vertex_prop::<i32>();

        props.set_vertex_prop(h1, vids[0], 1.0);
        props.set_vertex_prop(h2, vids[0], 42);

        props.clear_vertex_props();

        assert_eq!(props.get_vertex_prop(h1, vids[0]), None);
        assert_eq!(props.get_vertex_prop(h2, vids[0]), None);
        // 类型注册仍保留
        assert!(props.has_vertex_prop::<f64>());
        assert!(props.has_vertex_prop::<i32>());
    }

    #[test]
    fn remove_then_reuse_slot_cleans_up_with_remove_all() {
        // 验证 Vec 存储下，remove_*_all_props 正确清理属性
        let (mut mesh, vids) = build_triangle_mesh();
        let mut props = MeshProperties::new();

        let h = props.add_vertex_prop::<f64>();
        props.set_vertex_prop(h, vids[0], 42.0);
        assert_eq!(props.get_vertex_prop(h, vids[0]), Some(&42.0));

        // 删除顶点并清理属性
        remove_vertex_with_props(&mut mesh, &mut props, vids[0]);

        // 属性应已清除
        assert_eq!(props.get_vertex_prop(h, vids[0]), None);

        // 插入新顶点（可能复用 slot）
        let new_v = mesh.add_vertex(crate::storage::Vertex::new([5.0, 5.0, 5.0]));

        // 新顶点不应继承旧属性
        assert_eq!(props.get_vertex_prop(h, new_v), None);

        // 设置新属性应正常工作
        props.set_vertex_prop(h, new_v, 99.0);
        assert_eq!(props.get_vertex_prop(h, new_v), Some(&99.0));
    }

    // ---------- PropertyStore 进阶测试 ----------

    #[test]
    fn property_store_default_is_empty() {
        let store = PropertyStore::<f64>::default();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn property_store_sparse_indices() {
        // 稀疏索引：设置 idx 0 和 idx 100，中间全为 None
        let mut store = PropertyStore::<i32>::new();
        store.set(0, 1);
        store.set(100, 2);

        assert_eq!(store.len(), 2);
        assert_eq!(store.get(0), Some(&1));
        assert_eq!(store.get(100), Some(&2));
        assert_eq!(store.get(50), None);
        assert!(store.contains(0));
        assert!(store.contains(100));
        assert!(!store.contains(50));
    }

    #[test]
    fn property_store_overwrite_value() {
        let mut store = PropertyStore::<f64>::new();
        store.set(3, 1.0);
        store.set(3, 2.0);
        assert_eq!(store.len(), 1);
        assert_eq!(store.get(3), Some(&2.0));
    }

    #[test]
    fn property_store_remove_out_of_bounds_returns_none() {
        let mut store = PropertyStore::<f64>::new();
        store.set(0, 1.0);
        // 删除超出范围的索引不 panic，返回 None
        assert_eq!(store.remove(999), None);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn property_store_get_mut_none_for_unset() {
        let mut store = PropertyStore::<f64>::new();
        store.set(0, 1.0);
        assert!(store.get_mut(1).is_none());
        assert!(store.get_mut(99).is_none());
    }

    #[test]
    fn property_store_array_type() {
        let mut store = PropertyStore::<[f64; 3]>::new();
        store.set(0, [1.0, 2.0, 3.0]);
        store.set(1, [4.0, 5.0, 6.0]);

        assert_eq!(store.get(0), Some(&[1.0, 2.0, 3.0]));
        assert_eq!(store.get(1), Some(&[4.0, 5.0, 6.0]));

        // 可变修改数组元素
        if let Some(v) = store.get_mut(0) {
            v[1] = 99.0;
        }
        assert_eq!(store.get(0), Some(&[1.0, 99.0, 3.0]));
    }

    #[test]
    fn property_store_bool_type() {
        let mut store = PropertyStore::<bool>::new();
        store.set(0, true);
        store.set(1, false);
        assert_eq!(store.get(0), Some(&true));
        assert_eq!(store.get(1), Some(&false));
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn property_store_iter_empty() {
        let store = PropertyStore::<f64>::new();
        assert_eq!(store.iter().count(), 0);
    }

    #[test]
    fn property_store_iter_after_remove() {
        let mut store = PropertyStore::<i32>::new();
        store.set(0, 10);
        store.set(1, 20);
        store.set(2, 30);

        store.remove(1);

        let entries: Vec<(usize, i32)> = store.iter().map(|(i, &v)| (i, v)).collect();
        assert_eq!(entries, vec![(0, 10), (2, 30)]);
    }

    #[test]
    fn property_store_large_sparse_index() {
        let mut store = PropertyStore::<u32>::new();
        store.set(10_000, 42);
        assert_eq!(store.len(), 1);
        assert_eq!(store.get(10_000), Some(&42));
        assert!(store.contains(10_000));
        assert_eq!(store.get(9_999), None);
    }

    // ---------- MeshProperties 多类型 / 多类别测试 ----------

    #[test]
    fn multiple_property_types_on_same_mesh() {
        // 同一网格上注册 f64 / i32 / bool / [f64;3] 四种顶点属性
        let (_, vids) = build_triangle_mesh();
        let mut props = MeshProperties::new();

        let h_f64 = props.add_vertex_prop::<f64>();
        let h_i32 = props.add_vertex_prop::<i32>();
        let h_bool = props.add_vertex_prop::<bool>();
        let h_arr = props.add_vertex_prop::<[f64; 3]>();

        props.set_vertex_prop(h_f64, vids[0], 1.5);
        props.set_vertex_prop(h_i32, vids[0], -7);
        props.set_vertex_prop(h_bool, vids[0], true);
        props.set_vertex_prop(h_arr, vids[0], [1.0, 2.0, 3.0]);

        assert_eq!(props.get_vertex_prop(h_f64, vids[0]), Some(&1.5));
        assert_eq!(props.get_vertex_prop(h_i32, vids[0]), Some(&-7));
        assert_eq!(props.get_vertex_prop(h_bool, vids[0]), Some(&true));
        assert_eq!(
            props.get_vertex_prop(h_arr, vids[0]),
            Some(&[1.0, 2.0, 3.0])
        );
        assert_eq!(props.vertex_prop_type_count(), 4);
    }

    #[test]
    fn property_detach_and_reattach() {
        // 注册 → 清空 → 重新设置，验证数据独立
        let (_, vids) = build_triangle_mesh();
        let mut props = MeshProperties::new();

        let h = props.add_vertex_prop::<f64>();
        props.set_vertex_prop(h, vids[0], 1.0);
        props.set_vertex_prop(h, vids[1], 2.0);
        assert_eq!(props.vertex_prop_type_count(), 1);

        // 清空数据（保留类型注册）
        props.clear_vertex_props();
        assert_eq!(props.get_vertex_prop(h, vids[0]), None);
        assert_eq!(props.get_vertex_prop(h, vids[1]), None);
        assert!(props.has_vertex_prop::<f64>());

        // 重新设置数据
        props.set_vertex_prop(h, vids[0], 99.0);
        assert_eq!(props.get_vertex_prop(h, vids[0]), Some(&99.0));
        assert_eq!(props.get_vertex_prop(h, vids[1]), None);
    }

    #[test]
    fn clear_halfedge_and_face_props() {
        let (mesh, _vids) = build_triangle_mesh();
        let heids: Vec<HalfEdgeId> = mesh.halfedge_ids().collect();
        let fids: Vec<FaceId> = mesh.face_ids().collect();

        let mut props = MeshProperties::new();
        let hh = props.add_halfedge_prop::<i32>();
        let hf = props.add_face_prop::<f64>();

        for (i, &he) in heids.iter().enumerate() {
            props.set_halfedge_prop(hh, he, i as i32);
        }
        props.set_face_prop(hf, fids[0], 2.71);

        // 清空
        props.clear_halfedge_props();
        props.clear_face_props();

        for he in &heids {
            assert_eq!(props.get_halfedge_prop(hh, *he), None);
        }
        assert_eq!(props.get_face_prop(hf, fids[0]), None);

        // 类型注册仍保留
        assert!(props.has_halfedge_prop::<i32>());
        assert!(props.has_face_prop::<f64>());
    }

    #[test]
    fn remove_halfedge_with_props_wrapper() {
        let (mesh, _vids) = build_triangle_mesh();
        let heids: Vec<HalfEdgeId> = mesh.halfedge_ids().collect();
        let mut mesh = mesh;
        let mut props = MeshProperties::new();

        let hh = props.add_halfedge_prop::<f64>();
        props.set_halfedge_prop(hh, heids[0], 1.0);
        props.set_halfedge_prop(hh, heids[1], 2.0);

        let removed = remove_halfedge_with_props(&mut mesh, &mut props, heids[0]);
        assert!(removed.is_some());
        assert!(!mesh.contains_halfedge(heids[0]));
        assert_eq!(props.get_halfedge_prop(hh, heids[0]), None);
        assert_eq!(props.get_halfedge_prop(hh, heids[1]), Some(&2.0));
    }

    #[test]
    fn remove_face_with_props_wrapper() {
        let (mesh, _vids) = build_triangle_mesh();
        let fids: Vec<FaceId> = mesh.face_ids().collect();
        let mut mesh = mesh;
        let mut props = MeshProperties::new();

        let hf = props.add_face_prop::<u32>();
        props.set_face_prop(hf, fids[0], 42);

        let removed = remove_face_with_props(&mut mesh, &mut props, fids[0]);
        assert!(removed.is_some());
        assert!(!mesh.contains_face(fids[0]));
        assert_eq!(props.get_face_prop(hf, fids[0]), None);
    }

    #[test]
    fn halfedge_prop_mut_works() {
        let (mesh, _) = build_triangle_mesh();
        let heids: Vec<HalfEdgeId> = mesh.halfedge_ids().collect();
        let mut props = MeshProperties::new();

        let hh = props.add_halfedge_prop::<f64>();
        props.set_halfedge_prop(hh, heids[0], 1.0);

        if let Some(val) = props.get_halfedge_prop_mut(hh, heids[0]) {
            *val = 10.0;
        }
        assert_eq!(props.get_halfedge_prop(hh, heids[0]), Some(&10.0));
    }

    #[test]
    fn face_prop_mut_works() {
        let (mesh, _) = build_triangle_mesh();
        let fids: Vec<FaceId> = mesh.face_ids().collect();
        let mut props = MeshProperties::new();

        let hf = props.add_face_prop::<f64>();
        props.set_face_prop(hf, fids[0], 1.0);

        if let Some(val) = props.get_face_prop_mut(hf, fids[0]) {
            *val = 20.0;
        }
        assert_eq!(props.get_face_prop(hf, fids[0]), Some(&20.0));
    }

    #[test]
    fn has_halfedge_and_face_prop_queries() {
        let mut props = MeshProperties::new();
        assert!(!props.has_halfedge_prop::<f64>());
        assert!(!props.has_face_prop::<i32>());
        assert_eq!(props.halfedge_prop_type_count(), 0);
        assert_eq!(props.face_prop_type_count(), 0);

        let _ = props.add_halfedge_prop::<f64>();
        let _ = props.add_face_prop::<i32>();

        assert!(props.has_halfedge_prop::<f64>());
        assert!(props.has_face_prop::<i32>());
        assert_eq!(props.halfedge_prop_type_count(), 1);
        assert_eq!(props.face_prop_type_count(), 1);
    }

    #[test]
    fn unset_property_returns_none_as_default() {
        // 已注册但未设置的属性返回 None（无隐式默认值）
        let (_, vids) = build_triangle_mesh();
        let mut props = MeshProperties::new();
        let h = props.add_vertex_prop::<f64>();

        assert_eq!(props.get_vertex_prop(h, vids[0]), None);
        assert_eq!(props.get_vertex_prop(h, vids[1]), None);
    }

    #[test]
    fn add_prop_overwrites_existing_store() {
        // 同类型重复 add 会覆盖旧存储（已有数据丢失）
        let (_, vids) = build_triangle_mesh();
        let mut props = MeshProperties::new();

        let h = props.add_vertex_prop::<f64>();
        props.set_vertex_prop(h, vids[0], 1.0);
        assert_eq!(props.get_vertex_prop(h, vids[0]), Some(&1.0));

        // 重新注册，数据丢失
        let _ = props.add_vertex_prop::<f64>();
        assert_eq!(props.get_vertex_prop(h, vids[0]), None);
        assert_eq!(props.vertex_prop_type_count(), 1);
    }

    #[test]
    fn property_handle_default() {
        let h: PropertyHandle<i32> = PropertyHandle::default();
        let debug = format!("{:?}", h);
        assert!(debug.contains("i32"));
    }

    #[test]
    fn mesh_properties_debug() {
        let mut props = MeshProperties::new();
        let _ = props.add_vertex_prop::<f64>();
        let _ = props.add_face_prop::<i32>();

        let debug = format!("{:?}", props);
        assert!(debug.contains("vertex_prop_types"));
        assert!(debug.contains("face_prop_types"));
    }

    #[test]
    fn empty_mesh_properties() {
        let props = MeshProperties::new();
        assert_eq!(props.vertex_prop_type_count(), 0);
        assert_eq!(props.halfedge_prop_type_count(), 0);
        assert_eq!(props.face_prop_type_count(), 0);
        assert!(!props.has_vertex_prop::<f64>());
        assert!(!props.has_halfedge_prop::<bool>());
        assert!(!props.has_face_prop::<i32>());
    }

    #[test]
    fn single_vertex_property() {
        // 单顶点网格（无面）的属性操作
        let mut mesh = MeshStorage::new();
        let v = mesh.add_vertex(Vertex::new([1.0, 2.0, 3.0]));

        let mut props = MeshProperties::new();
        let h = props.add_vertex_prop::<f64>();
        props.set_vertex_prop(h, v, 42.0);

        assert_eq!(props.get_vertex_prop(h, v), Some(&42.0));
        assert!(props.get_vertex_prop_mut(h, v).is_some());
    }

    #[test]
    fn all_vertices_deleted_props_cleared() {
        let (mut mesh, vids) = build_triangle_mesh();
        let mut props = MeshProperties::new();

        let h = props.add_vertex_prop::<f64>();
        for v in &vids {
            props.set_vertex_prop(h, *v, 1.0);
        }

        // 删除所有顶点并清理属性
        for v in &vids {
            remove_vertex_with_props(&mut mesh, &mut props, *v);
        }

        for v in &vids {
            assert_eq!(props.get_vertex_prop(h, *v), None);
        }
        assert_eq!(mesh.vertex_count(), 0);
    }

    #[test]
    fn large_mesh_property_correctness() {
        // 在较大网格上验证属性正确性（非性能测试）
        let mesh = crate::test_util::build_icosphere(1);
        let vids: Vec<VertexId> = mesh.vertex_ids().collect();
        let fids: Vec<FaceId> = mesh.face_ids().collect();

        let mut props = MeshProperties::new();
        let hv = props.add_vertex_prop::<f64>();
        let hf = props.add_face_prop::<u32>();

        // 为每个顶点/面设置属性（值为索引）
        for (i, &v) in vids.iter().enumerate() {
            props.set_vertex_prop(hv, v, i as f64);
        }
        for (i, &f) in fids.iter().enumerate() {
            props.set_face_prop(hf, f, i as u32);
        }

        // 验证所有值正确
        for (i, &v) in vids.iter().enumerate() {
            assert_eq!(props.get_vertex_prop(hv, v), Some(&(i as f64)));
        }
        for (i, &f) in fids.iter().enumerate() {
            assert_eq!(props.get_face_prop(hf, f), Some(&(i as u32)));
        }

        // 批量删除部分顶点属性
        for &v in vids.iter().step_by(3) {
            props.remove_vertex_all_props(v);
            assert_eq!(props.get_vertex_prop(hv, v), None);
        }
    }

    // ---------- 测试夹具 ----------

    fn build_triangle_mesh() -> (MeshStorage, [VertexId; 3]) {
        let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![[0, 1, 2]];
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let vids: Vec<VertexId> = mesh.vertex_ids().collect();
        (mesh, [vids[0], vids[1], vids[2]])
    }
}
