//! 强类型 ID 模块
//!
//! 利用 slotmap 的 `new_key_type!` 宏为三类拓扑元素生成各自独立的句柄类型
//! （`VertexId`/`HalfEdgeId`/`FaceId`），使其在编译期就被区分开：把 `VertexId`
//! 误传给期望 `HalfEdgeId` 的接口会直接编译失败，从根源杜绝「用错 ID 类型」这一类 bug。
//!
//! 三者同时兼任 slotmap 的 `Key`，与 `storage` 模块无缝衔接。
//!
//! `new_key_type!` 宏自动派生：`Debug`、`Copy`、`Clone`、`Eq`/`PartialEq`、
//! `Ord`/`PartialOrd`、`Hash`、`Default`。
//!
//! ## EdgeId
//! 本模块还提供 [`EdgeId`]，代表一条无向边（一对互为 twin 的半边）。
//! 内部以"规范半边"（twin 对中 key 较小者）作为代表元，保证每条无向边
//! 有唯一的 `EdgeId`，可用于迭代、属性键等场景。通过 `halfedge()` 取出
//! 内部 `HalfEdgeId`，通过 `vertices()` 查询两端点。

use slotmap::new_key_type;

new_key_type! {
    /// 顶点句柄。删除对应顶点后，该句柄在 slotmap 的版本号机制下自动失效，
    /// 后续 `get_vertex` 调用会返回 `None`。
    pub struct VertexId;

    /// 半边句柄。半边是有向的，成对出现（twin）。
    pub struct HalfEdgeId;

    /// 面句柄。本实现面向三角网格，但面结构本身不限边数。
    pub struct FaceId;
}

/// 一条无向边，代表一对互为 twin 的半边。
///
/// `EdgeId` 包装了 twin 对中 slotmap key 较小的那个半边（"规范半边"），
/// 确保每条无向边有唯一的标识。对于无边界的边界半边（`twin == None`），
/// 直接使用该半边自身作为规范代表。
///
/// `EdgeId` 不是 slotmap 的 `Key`，不能直接用于 `MeshStorage` 的 CRUD
/// 方法；使用 `halfedge()` 取出内部的 `HalfEdgeId` 后再操作。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EdgeId(pub(crate) HalfEdgeId);

impl EdgeId {
    /// 返回该无向边的规范半边句柄。
    pub fn halfedge(self) -> HalfEdgeId {
        self.0
    }

    /// 返回该边的两个端点 `(src, dst)`，其中 `src` 是规范半边的源顶点，
    /// `dst` 为规范半边的目标顶点。
    ///
    /// 要求规范半边必须有 twin（即该边不是边界边），否则返回 `None`。
    /// 边界边的 twin 为 `None`，无法确定 src。
    pub fn vertices(&self, mesh: &crate::storage::MeshStorage) -> Option<(VertexId, VertexId)> {
        let he = mesh.get_halfedge(self.0)?;
        let dst = he.vertex;
        let src = mesh.get_halfedge(he.twin?)?.vertex;
        Some((src, dst))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `new_key_type!` 生成的默认 key 是「未初始化」的（index=0, version=0）。
    /// 它仍然是一个合法的 key 对象，只是不会出现在任何 SlotMap 中。
    #[test]
    fn ids_are_copy_and_defaultable() {
        let v: VertexId = VertexId::default();
        let h: HalfEdgeId = HalfEdgeId::default();
        let f: FaceId = FaceId::default();

        // Copy 语义：可以直接按值复制而不需要 clone。
        let v2 = v;
        let h2 = h;
        let f2 = f;

        // Eq / Ord：默认 key 之间可比较。
        assert_eq!(v, v2);
        assert_eq!(h, h2);
        assert_eq!(f, f2);
        assert!(v <= v2);
    }

    /// 不同 ID 类型在编译期不可互换——这一条由类型系统保证，
    /// 此处仅以「它们都是不同类型」的隐式事实作为说明，不需要运行期断言。
    #[test]
    fn ids_have_distinct_types() {
        // 这两行如果取消注释会编译失败，证明类型不可互换：
        // let _: VertexId = HalfEdgeId::default();
        // let _: HalfEdgeId = FaceId::default();
        let _v: VertexId = VertexId::default();
        let _h: HalfEdgeId = HalfEdgeId::default();
        let _f: FaceId = FaceId::default();
    }
}
