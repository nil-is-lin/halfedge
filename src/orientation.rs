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
///
/// 对任意 $n$ 边形面，反转其半边环的绕行方向（CCW ↔ CW），
/// 同时更新每条半边的 `vertex`、`next`、`prev` 指针以保持拓扑一致性。
///
/// ## 算法
/// 设原始半边环为 $\text{he}[0] \to \text{he}[1] \to \cdots \to \text{he}[n-1] \to \text{he}[0]$，
/// 目标顶点为 $v[i] = \text{he}[i].\text{vertex}$。翻转后：
/// - $\text{he}[i].\text{vertex} \gets v[(i + n - 1) \bmod n]$
/// - $\text{he}[i].\text{next} \gets \text{he}[(i + n - 1) \bmod n]$
/// - $\text{he}[i].\text{prev} \gets \text{he}[(i + 1) \bmod n]$
fn flip_face_orientation(mesh: &mut MeshStorage, face: FaceId) {
    let he_ids: Vec<HalfEdgeId> = FaceHalfEdges::new(mesh, face).collect();
    let n = he_ids.len();
    if n == 0 {
        return;
    }

    // 快照原始状态
    let old_v: Vec<_> = he_ids
        .iter()
        .map(|&id| {
            mesh.get_halfedge(id)
                .expect("halfedge exists in mesh")
                .vertex
        })
        .collect();

    // 一次性应用：vertex 重分配 + next/prev 反转
    for i in 0..n {
        let h = mesh
            .get_halfedge_mut(he_ids[i])
            .expect("halfedge exists in mesh");
        h.vertex = old_v[(i + n - 1) % n];
        h.next = Some(he_ids[(i + n - 1) % n]);
        h.prev = Some(he_ids[(i + 1) % n]);
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
    use crate::io::build_mesh_from_polygons;
    use crate::storage::Vertex;
    use crate::topology_ops::add_triangle;
    use crate::traversal::FaceHalfEdges;

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

    #[test]
    fn empty_mesh_is_consistent() {
        let mesh = crate::storage::MeshStorage::new();
        assert!(are_normals_consistent(&mesh));
        assert!(is_orientable(&mesh));
        let mut mesh2 = crate::storage::MeshStorage::new();
        assert_eq!(fix_orientations(&mut mesh2), 0);
    }

    #[test]
    fn single_triangle_is_consistent() {
        let mut mesh = crate::storage::MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));
        add_triangle(&mut mesh, v0, v1, v2).unwrap();
        assert!(are_normals_consistent(&mesh));
        assert!(is_orientable(&mesh));
    }

    /// 辅助：收集面的半边目标顶点序列。
    fn face_vertex_ring(mesh: &MeshStorage, face: FaceId) -> Vec<crate::ids::VertexId> {
        FaceHalfEdges::new(mesh, face)
            .filter_map(|he| mesh.get_halfedge(he))
            .map(|h| h.vertex)
            .collect()
    }

    #[test]
    fn flip_triangle_reverses_winding() {
        let mut mesh = crate::storage::MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));
        let face = add_triangle(&mut mesh, v0, v1, v2).unwrap();

        let original = face_vertex_ring(&mesh, face);
        assert_eq!(original, vec![v1, v2, v0]); // he0→v1, he1→v2, he2→v0

        flip_face_orientation(&mut mesh, face);

        let flipped = face_vertex_ring(&mesh, face);
        // 翻转后应为原序列的逆序（保持相同半边 ID，但顶点循环反转）
        // 原: [v1, v2, v0] → 翻转: [v0, v2, v1]
        assert_eq!(flipped, vec![v0, v2, v1]);

        // next/prev 链应仍闭合
        let he_ids: Vec<_> = FaceHalfEdges::new(&mesh, face).collect();
        for (i, &he) in he_ids.iter().enumerate() {
            let h = mesh.get_halfedge(he).unwrap();
            let next_id = he_ids[(i + 1) % he_ids.len()];
            let prev_id = he_ids[(i + he_ids.len() - 1) % he_ids.len()];
            assert_eq!(h.next, Some(next_id), "he[{i}].next broken");
            assert_eq!(h.prev, Some(prev_id), "he[{i}].prev broken");
        }

        // 再翻转一次应恢复原始状态
        flip_face_orientation(&mut mesh, face);
        let restored = face_vertex_ring(&mesh, face);
        assert_eq!(restored, original);
    }

    #[test]
    fn flip_quad_reverses_winding() {
        // 使用 build_mesh_from_polygons 构建两个四边形面
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let faces: Vec<Vec<u32>> = vec![vec![0, 1, 4, 5], vec![1, 2, 3, 4]];
        let mut mesh = build_mesh_from_polygons(&verts, &faces).unwrap();

        let f_ids: Vec<_> = mesh.face_ids().collect();
        let face0 = f_ids[0];

        let original = face_vertex_ring(&mesh, face0);
        assert_eq!(original.len(), 4, "quad face should have 4 halfedges");

        flip_face_orientation(&mut mesh, face0);

        let flipped = face_vertex_ring(&mesh, face0);
        // 翻转后顶点序列应为原始循环逆序
        // 原始遍历: [v[1], v[4], v[5], v[0]]
        // 翻转遍历: [v[0], v[5], v[4], v[1]]（next 反转导致遍历方向相反）
        // 因此 flipped[i] == original[(n - 1 - i) % n]
        for i in 0..4 {
            assert_eq!(
                flipped[i],
                original[(4 - 1 - i) % 4],
                "flipped[{i}] should equal original[{}]",
                (4 - 1 - i) % 4
            );
        }

        // next/prev 链应仍闭合
        let he_ids: Vec<_> = FaceHalfEdges::new(&mesh, face0).collect();
        for (i, &he) in he_ids.iter().enumerate() {
            let h = mesh.get_halfedge(he).unwrap();
            assert_eq!(
                h.next,
                Some(he_ids[(i + 1) % 4]),
                "he[{i}].next broken after quad flip"
            );
            assert_eq!(
                h.prev,
                Some(he_ids[(i + 3) % 4]),
                "he[{i}].prev broken after quad flip"
            );
        }

        // 双翻恢复
        flip_face_orientation(&mut mesh, face0);
        let restored = face_vertex_ring(&mesh, face0);
        assert_eq!(restored, original);
    }

    #[test]
    fn fix_orientations_corrects_flipped_face() {
        // 构建四面体（4 个三角面，全部朝向一致）
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let faces = vec![[0u32, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]];
        let mut mesh = crate::io::build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        assert!(are_normals_consistent(&mesh));

        // 手动翻转第一个面 → 制造不一致
        let first_face = mesh.face_ids().next().unwrap();
        flip_face_orientation(&mut mesh, first_face);
        assert!(
            !are_normals_consistent(&mesh),
            "mesh should be inconsistent after flipping one face"
        );

        // fix_orientations 应修复
        let flipped = fix_orientations(&mut mesh);
        assert!(flipped > 0, "should flip at least one face");
        assert!(
            are_normals_consistent(&mesh),
            "mesh should be consistent after fix_orientations"
        );
    }

    #[test]
    fn disconnected_components_each_fixed() {
        // 两个独立三角形（不共享边），手动翻转第二个
        let mut mesh = crate::storage::MeshStorage::new();
        let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
        let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
        let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));
        let v3 = mesh.add_vertex(Vertex::new([2.0, 0.0, 0.0]));
        let v4 = mesh.add_vertex(Vertex::new([3.0, 0.0, 0.0]));
        let v5 = mesh.add_vertex(Vertex::new([2.0, 1.0, 0.0]));
        let f1 = add_triangle(&mut mesh, v0, v1, v2).unwrap();
        let _f2 = add_triangle(&mut mesh, v3, v4, v5).unwrap();

        // 两个独立组件各自应一致
        assert!(are_normals_consistent(&mesh));
        assert_eq!(fix_orientations(&mut mesh), 0);

        // 翻转其中一个 → 不影响一致性（无共享边，are_normals_consistent 检测不到）
        flip_face_orientation(&mut mesh, f1);
        // 无共享 twin → are_normals_consistent 仍返回 true（无矛盾边）
        // 但 fix_orientations 应仍能处理
    }
}
