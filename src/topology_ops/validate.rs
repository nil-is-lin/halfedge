//! 网格校验（轻量级，首错返回）。

use crate::ids::HalfEdgeId;
use crate::storage::MeshStorage;

use super::helpers::TopologyError;

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
///
/// ```
/// use halfedge::{build_mesh_from_vertices_and_faces, validate_mesh};
///
/// let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
/// let faces = vec![[0u32, 1, 2]];
/// let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
/// assert!(validate_mesh(&mesh).is_ok());
/// ```
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
