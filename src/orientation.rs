//! 面朝向一致性检测与修复模块。

use std::collections::{HashSet, VecDeque};

use crate::ids::{FaceId, HalfEdgeId};
use crate::storage::MeshStorage;
use crate::traversal::FaceHalfEdges;

/// 检查所有相邻面的朝向是否一致。
/// 对于每条内部边（共享于两面之间），两面的半边走相反方向（互成 twin）。
/// 如果某条边的两面半边走相同方向，则朝向不一致。
pub fn are_normals_consistent(mesh: &MeshStorage) -> bool {
    for he in mesh.halfedge_ids() {
        let h = match mesh.get_halfedge(he) {
            Some(h) => h,
            None => continue,
        };
        let twin = match h.twin {
            Some(t) => t,
            None => continue,
        };
        // he 和 twin 应有相同的两个端点但方向相反
        let twin_data = match mesh.get_halfedge(twin) {
            Some(t) => t,
            None => continue,
        };
        // he: src→dst, twin: dst→src
        // 如果 he.vertex == twin.vertex，则是同向 → 不一致
        if h.vertex == twin_data.vertex {
            return false;
        }
    }
    true
}

/// 判断网格是否可定向。
///
/// BFS 遍历所有面，检查相邻面间的朝向是否可一致分配。
/// 若存在矛盾（不可定向），返回 `false`。
pub fn is_orientable(mesh: &MeshStorage) -> bool {
    let mut visited: HashSet<FaceId> = HashSet::new();
    for seed in mesh.face_ids() {
        if visited.contains(&seed) {
            continue;
        }
        let mut queue = VecDeque::new();
        visited.insert(seed);
        queue.push_back((seed, true)); // true = 保持原朝向
        while let Some((f, _orientation)) = queue.pop_front() {
            for he in FaceHalfEdges::new(mesh, f) {
                let h = match mesh.get_halfedge(he) {
                    Some(h) => h,
                    None => continue,
                };
                let twin = match h.twin {
                    Some(t) => t,
                    None => continue,
                };
                let adj_f = match mesh.get_halfedge(twin).and_then(|t| t.face) {
                    Some(f) => f,
                    None => continue,
                };
                if visited.contains(&adj_f) {
                    continue;
                }
                // 检查方向一致性：he 与 twin 应方向相反
                let twin_data = mesh.get_halfedge(twin);
                if let Some(td) = twin_data
                    && h.vertex == td.vertex
                {
                    return false; // 同向，不可定向
                }
                visited.insert(adj_f);
                queue.push_back((adj_f, true));
            }
        }
    }
    true
}

/// 翻转指定面（反转半边环方向）。
fn flip_face_orientation(mesh: &mut MeshStorage, face: FaceId) {
    let he_ids: Vec<HalfEdgeId> = FaceHalfEdges::new(mesh, face).collect();
    if he_ids.is_empty() {
        return;
    }
    // 收集所有半边的当前状态
    let he_data: Vec<_> = he_ids
        .iter()
        .map(|&id| mesh.get_halfedge(id).unwrap().clone())
        .collect();

    // 反转：对于每条半边，交换 next/prev，交换 vertex 引用
    for &id in &he_ids {
        let h = mesh.get_halfedge_mut(id).unwrap();
        // next ↔ prev
        let old_next = h.next;
        let old_prev = h.prev;
        h.next = old_prev;
        h.prev = old_next;
        // vertex 改为原 prev 的 src（即对面顶点的顶点）
        // 原环顺序: he[i]→he[i+1]→he[i+2]
        // 翻转后: 每条半边的 vertex 变为原前驱(prev)半边的 vertex 的对侧
        // 简化：每条半边 vertex 设为原 next 半边的 twin vertex
    }

    // 实际上，对三角面而言，翻转 = 交换 vertex 为原 next 半边
    if he_ids.len() == 3 {
        let v0 = he_data[0].vertex;
        let v1 = he_data[1].vertex;
        let v2 = he_data[2].vertex;
        // he0: 原 v0→v1，翻转为 ?→？
        // 翻转后面半边环序不变，但每条半边指向不同顶点
        // he0.vertex = v2 (原 he1 起点, 即 he0 的原下一个)
        mesh.get_halfedge_mut(he_ids[0]).unwrap().vertex = v2;
        mesh.get_halfedge_mut(he_ids[1]).unwrap().vertex = v0;
        mesh.get_halfedge_mut(he_ids[2]).unwrap().vertex = v1;
        // 交换 next/prev: 原 next→prev, prev→next
        for &id in &he_ids {
            let h = mesh.get_halfedge_mut(id).unwrap();
            let old_next = h.next;
            let old_prev = h.prev;
            h.next = old_prev;
            h.prev = old_next;
        }
    }
}

/// 修复所有面的朝向一致性。返回翻转的面数量。
///
/// 对每个连通分量选择一个基准面，BFS 遍历相邻面，
/// 翻转与基准不一致的面。
pub fn fix_orientations(mesh: &mut MeshStorage) -> usize {
    let mut visited: HashSet<FaceId> = HashSet::new();
    let mut flipped = 0usize;

    for seed in mesh.face_ids().collect::<Vec<_>>() {
        if visited.contains(&seed) {
            continue;
        }
        let mut queue = VecDeque::new();
        visited.insert(seed);
        queue.push_back(seed);

        while let Some(f) = queue.pop_front() {
            for he in FaceHalfEdges::new(mesh, f).collect::<Vec<_>>() {
                let h = match mesh.get_halfedge(he) {
                    Some(h) => h,
                    None => continue,
                };
                let twin = match h.twin {
                    Some(t) => t,
                    None => continue,
                };
                let adj_f = match mesh.get_halfedge(twin).and_then(|t| t.face) {
                    Some(f) => f,
                    None => continue,
                };
                if visited.contains(&adj_f) {
                    continue;
                }
                let twin_data = mesh.get_halfedge(twin);
                // 检查方向：he 和 twin 应方向相反（vertex 不同）
                // 如果相同，需要翻转 adj_f
                let need_flip = twin_data.is_some_and(|td| h.vertex == td.vertex);
                if need_flip {
                    flip_face_orientation(mesh, adj_f);
                    flipped += 1;
                }
                visited.insert(adj_f);
                queue.push_back(adj_f);
            }
        }
    }
    flipped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icosphere_has_consistent_normals() {
        let mesh = crate::test_util::build_icosphere(1);
        assert!(are_normals_consistent(&mesh));
        assert!(is_orientable(&mesh));
    }

    #[test]
    fn fix_orientations_noop_on_icosphere() {
        let mut mesh = crate::test_util::build_icosphere(1);
        let flipped = fix_orientations(&mut mesh);
        assert_eq!(flipped, 0);
    }
}
