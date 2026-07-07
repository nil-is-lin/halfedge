//! 顶点焊接模块：按距离阈值合并邻近顶点。

use crate::ids::{HalfEdgeId, VertexId};
use crate::storage::MeshStorage;
use crate::topology_ops::TopologyError;

/// 将距离小于 `epsilon` 的顶点合并。返回被移除的顶点数量。
///
/// 算法：排序后扫描相邻顶点，构建等价类映射，然后重新映射所有半边
/// 的 vertex 引用，删除多余的顶点。焊接后自动清理退化面。
pub fn weld_vertices(mesh: &mut MeshStorage, epsilon: f64) -> Result<usize, TopologyError> {
    if mesh.vertex_count() < 2 {
        return Ok(0);
    }

    // 1. 收集 (VertexId, position)
    let verts: Vec<(VertexId, [f64; 3])> = mesh
        .vertex_ids()
        .map(|id| {
            (
                id,
                mesh.get_vertex(id).expect("vertex exists in mesh").position,
            )
        })
        .collect();

    // 2. 按 x 排序
    let mut indexed: Vec<(usize, VertexId, [f64; 3])> = verts
        .iter()
        .enumerate()
        .map(|(i, (id, p))| (i, *id, *p))
        .collect();
    indexed.sort_by(|a, b| {
        a.2[0]
            .partial_cmp(&b.2[0])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 3. Union-find 等价类
    let n = indexed.len();
    let eps2 = epsilon * epsilon;
    let mut parent: Vec<usize> = (0..n).collect();
    for i in 0..n {
        for j in (i + 1)..n {
            if indexed[j].2[0] - indexed[i].2[0] > epsilon {
                break;
            }
            let dx = indexed[j].2[0] - indexed[i].2[0];
            let dy = indexed[j].2[1] - indexed[i].2[1];
            let dz = indexed[j].2[2] - indexed[i].2[2];
            if dx * dx + dy * dy + dz * dz < eps2 {
                let mut ri = i;
                while parent[ri] != ri {
                    ri = parent[ri];
                }
                let mut rj = j;
                while parent[rj] != rj {
                    rj = parent[rj];
                }
                if ri != rj {
                    parent[rj] = ri;
                }
            }
        }
    }

    // 4. 映射 old_id -> rep_id
    use std::collections::HashMap;
    let mut mapping: HashMap<VertexId, VertexId> = HashMap::new();
    for i in 0..n {
        let mut root = i;
        while parent[root] != root {
            root = parent[root];
        }
        mapping.insert(indexed[i].1, indexed[root].1);
    }

    // 5. 重映射半边 vertex 引用
    let he_ids: Vec<HalfEdgeId> = mesh.halfedge_ids().collect();
    for he in he_ids {
        if let Some(h) = mesh.get_halfedge_mut(he)
            && let Some(&new_v) = mapping.get(&h.vertex)
        {
            h.vertex = new_v;
        }
    }

    // 6. 删除冗余顶点
    let mut removed = 0usize;
    let all_ids: Vec<VertexId> = mesh.vertex_ids().collect();
    for v in all_ids {
        if let Some(&rep) = mapping.get(&v)
            && rep != v
            && mesh.contains_vertex(v)
        {
            mesh.remove_vertex(v);
            removed += 1;
        }
    }

    // 7. 清理退化面
    let all_faces: Vec<crate::ids::FaceId> = mesh.face_ids().collect();
    for f in all_faces {
        if !mesh.contains_face(f) {
            continue;
        }
        let verts: Vec<VertexId> = crate::traversal::FaceVertices::new(mesh, f).collect();
        let unique: std::collections::HashSet<_> = verts.iter().copied().collect();
        if unique.len() < 3 {
            let hes: Vec<HalfEdgeId> = crate::traversal::FaceHalfEdges::new(mesh, f).collect();
            for he in hes {
                if mesh.contains_halfedge(he) {
                    mesh.remove_halfedge(he);
                }
            }
            mesh.remove_face(f);
        }
    }

    if mesh.face_count() > 0 {
        crate::topology_ops::validate_mesh(mesh)?;
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weld_no_vertices() {
        let mut mesh = MeshStorage::new();
        assert_eq!(weld_vertices(&mut mesh, 0.01).unwrap(), 0);
    }

    #[test]
    fn weld_single_vertex() {
        let mut mesh = MeshStorage::new();
        mesh.add_vertex(crate::storage::Vertex::new([0.0; 3]));
        assert_eq!(weld_vertices(&mut mesh, 0.01).unwrap(), 0);
    }

    #[test]
    fn weld_two_close_vertices() {
        let mut mesh = MeshStorage::new();
        let _a = mesh.add_vertex(crate::storage::Vertex::new([0.0, 0.0, 0.0]));
        let _b = mesh.add_vertex(crate::storage::Vertex::new([0.001, 0.0, 0.0]));
        let _c = mesh.add_vertex(crate::storage::Vertex::new([1.0, 0.0, 0.0]));
        let _ = weld_vertices(&mut mesh, 0.01).unwrap();
        // a 和 b 应合并为 1 个，c 保留 → 2 个顶点
        assert_eq!(mesh.vertex_count(), 2);
    }
}
