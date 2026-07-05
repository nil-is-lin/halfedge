//! 连通分量分析模块
//!
//! 提供基于面邻接和顶点邻接的连通分量检测，以及按分量拆分/
//! 提取和多网格合并等操作。

use std::collections::{HashMap, HashSet, VecDeque};

use crate::ids::{FaceId, HalfEdgeId, VertexId};
use crate::storage::{Face, HalfEdge, MeshStorage, Vertex};
use crate::traversal::{VertexAdjacentFaces, VertexAdjacentVerts};

/// 返回所有面连通分量。两个面连通当且仅当它们共享一条边（共享两个顶点）。
pub fn connected_components(mesh: &MeshStorage) -> Vec<Vec<FaceId>> {
    let mut visited: HashSet<FaceId> = HashSet::new();
    let mut components = Vec::new();
    for seed in mesh.face_ids() {
        if visited.contains(&seed) {
            continue;
        }
        let comp = component_of_face(mesh, seed).unwrap_or_default();
        for &f in &comp {
            visited.insert(f);
        }
        if !comp.is_empty() {
            components.push(comp);
        }
    }
    components
}

/// 返回包含 `seed` 的面连通分量。BFS 遍历。
pub fn component_of_face(mesh: &MeshStorage, seed: FaceId) -> Option<Vec<FaceId>> {
    if !mesh.contains_face(seed) {
        return None;
    }
    let mut visited: HashSet<FaceId> = HashSet::new();
    let mut queue = VecDeque::new();
    let mut result = Vec::new();
    visited.insert(seed);
    queue.push_back(seed);
    while let Some(f) = queue.pop_front() {
        result.push(f);
        // 遍历 f 的所有顶点，再找每个顶点的邻接面
        for v in crate::traversal::FaceVertices::new(mesh, f) {
            for adj_f in VertexAdjacentFaces::new(mesh, v) {
                if !visited.contains(&adj_f) {
                    visited.insert(adj_f);
                    queue.push_back(adj_f);
                }
            }
        }
    }
    Some(result)
}

/// 返回面连通分量数量（不分配分量列表）。
pub fn component_count(mesh: &MeshStorage) -> usize {
    connected_components(mesh).len()
}

/// 返回所有顶点连通分量（两个顶点连通当且仅当它们是一条边的两端点）。
pub fn vertex_connected_components(mesh: &MeshStorage) -> Vec<Vec<VertexId>> {
    let mut visited: HashSet<VertexId> = HashSet::new();
    let mut components = Vec::new();
    for seed in mesh.vertex_ids() {
        if visited.contains(&seed) {
            continue;
        }
        let mut queue = VecDeque::new();
        let mut comp = Vec::new();
        visited.insert(seed);
        queue.push_back(seed);
        while let Some(v) = queue.pop_front() {
            comp.push(v);
            for neighbor in VertexAdjacentVerts::new(mesh, v) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }
        if !comp.is_empty() {
            components.push(comp);
        }
    }
    components
}

// ============================================================
// 网格级拆分与提取
// ============================================================

/// 提取包含 `seed` 面的连通分量为独立的 `MeshStorage`。
///
/// 保留所有拓扑关系；与其余分量相邻的边界边在新网格中成为边界边。
pub fn extract_component(mesh: &MeshStorage, seed: FaceId) -> Option<MeshStorage> {
    let faces = component_of_face(mesh, seed)?;
    extract_faces(mesh, &faces)
}

/// 提取指定面集合为独立的 `MeshStorage`。
///
/// `faces` 应来自 `connected_components` 的某一项。保留内部拓扑，
/// 与外部相邻的边在新网格中成为边界边。
pub fn extract_faces(mesh: &MeshStorage, faces: &[FaceId]) -> Option<MeshStorage> {
    if faces.is_empty() {
        return None;
    }

    let _face_set: HashSet<FaceId> = faces.iter().copied().collect();

    // 收集该分量中的所有半边
    let mut he_set: HashSet<HalfEdgeId> = HashSet::new();
    let mut vert_set: HashSet<VertexId> = HashSet::new();
    for &f in faces {
        for he in crate::traversal::FaceHalfEdges::new(mesh, f) {
            he_set.insert(he);
            if let Some(h) = mesh.get_halfedge(he) {
                vert_set.insert(h.vertex);
            }
        }
    }

    // 创建新网格，复制顶点
    let mut new_mesh = MeshStorage::new();
    new_mesh.reserve(vert_set.len(), he_set.len() * 2, faces.len());

    let mut v_map: HashMap<VertexId, VertexId> = HashMap::with_capacity(vert_set.len());
    for &old_v in &vert_set {
        let pos = mesh.get_vertex(old_v)?.position;
        let new_v = new_mesh.add_vertex(Vertex::new(pos));
        v_map.insert(old_v, new_v);
    }

    // 复制半边（保留拓扑，twin 若不在分量内则设为 None）
    let mut he_map: HashMap<HalfEdgeId, HalfEdgeId> = HashMap::with_capacity(he_set.len());
    for &old_he in &he_set {
        let h = mesh.get_halfedge(old_he)?;
        let new_vertex = v_map.get(&h.vertex).copied()?;
        let new_he = new_mesh.add_halfedge(HalfEdge::new(new_vertex));
        he_map.insert(old_he, new_he);
    }

    // 补全半边字段
    for &old_he in &he_set {
        let old = mesh.get_halfedge(old_he).unwrap();
        let new_he = he_map[&old_he];
        let new_h = new_mesh.get_halfedge_mut(new_he).unwrap();
        new_h.twin = old.twin.and_then(|t| he_map.get(&t).copied());
        new_h.next = old.next.and_then(|n| he_map.get(&n).copied());
        new_h.prev = old.prev.and_then(|p| he_map.get(&p).copied());
    }

    // 复制面
    let mut f_map: HashMap<FaceId, FaceId> = HashMap::with_capacity(faces.len());
    for &old_f in faces {
        let new_f = new_mesh.add_face(Face::new());
        f_map.insert(old_f, new_f);
    }

    // 设置面的 halfedge 和半边的 face
    for &old_f in faces {
        let new_f = f_map[&old_f];
        let old_f_he = mesh.get_face(old_f)?.halfedge;
        if let Some(mapped_he) = old_f_he.and_then(|he| he_map.get(&he).copied()) {
            new_mesh.get_face_mut(new_f).unwrap().halfedge = Some(mapped_he);
        }
    }
    for &old_he in &he_set {
        let new_he = he_map[&old_he];
        let old_face = mesh.get_halfedge(old_he)?.face;
        if let Some(mapped_f) = old_face.and_then(|f| f_map.get(&f).copied()) {
            new_mesh.get_halfedge_mut(new_he).unwrap().face = Some(mapped_f);
        }
    }

    // 设置顶点 outgoing 半边
    for &old_v in &vert_set {
        let new_v = v_map[&old_v];
        let old_he = mesh.get_vertex(old_v)?.halfedge;
        if let Some(mapped) = old_he.and_then(|he| he_map.get(&he).copied()) {
            new_mesh.get_vertex_mut(new_v).unwrap().halfedge = Some(mapped);
        }
    }

    Some(new_mesh)
}

/// 将网格按面连通分量拆分为多个独立的 `MeshStorage`。
///
/// 每个分量保留完整的内部拓扑；分量间的边界边在各独立网格中
/// 成为边界边。对闭合连通网格返回单元素 `Vec`。
pub fn split_into_components(mesh: &MeshStorage) -> Vec<MeshStorage> {
    connected_components(mesh)
        .iter()
        .filter_map(|comp| extract_faces(mesh, comp))
        .collect()
}

// ============================================================
// 网格合并
// ============================================================

/// 将两个网格合并为一个。
///
/// 两个网格的拓扑保持独立（各自内部的边仍为内部/边界边；
/// 两网格间不建立新的连接）。顶点位置直接复制。
pub fn merge_meshes(a: &MeshStorage, b: &MeshStorage) -> MeshStorage {
    let total_v = a.vertex_count() + b.vertex_count();
    let total_he = a.halfedge_count() + b.halfedge_count();
    let total_f = a.face_count() + b.face_count();

    let mut mesh = MeshStorage::new();
    mesh.reserve(total_v, total_he, total_f);

    let mut v_map_a: HashMap<VertexId, VertexId> = HashMap::with_capacity(a.vertex_count());
    let mut he_map_a: HashMap<HalfEdgeId, HalfEdgeId> = HashMap::with_capacity(a.halfedge_count());
    let mut f_map_a: HashMap<FaceId, FaceId> = HashMap::with_capacity(a.face_count());
    copy_mesh_into(a, &mut mesh, &mut v_map_a, &mut he_map_a, &mut f_map_a);

    let mut v_map_b: HashMap<VertexId, VertexId> = HashMap::with_capacity(b.vertex_count());
    let mut he_map_b: HashMap<HalfEdgeId, HalfEdgeId> = HashMap::with_capacity(b.halfedge_count());
    let mut f_map_b: HashMap<FaceId, FaceId> = HashMap::with_capacity(b.face_count());
    copy_mesh_into(b, &mut mesh, &mut v_map_b, &mut he_map_b, &mut f_map_b);

    mesh
}

/// 内部：将源网格的所有元素复制到目标网格，记录 ID 映射。
fn copy_mesh_into(
    src: &MeshStorage,
    dst: &mut MeshStorage,
    v_map: &mut HashMap<VertexId, VertexId>,
    he_map: &mut HashMap<HalfEdgeId, HalfEdgeId>,
    f_map: &mut HashMap<FaceId, FaceId>,
) {
    for v_id in src.vertex_ids() {
        let pos = src.get_vertex(v_id).unwrap().position;
        let new_v = dst.add_vertex(Vertex::new(pos));
        v_map.insert(v_id, new_v);
    }
    for he_id in src.halfedge_ids() {
        let h = src.get_halfedge(he_id).unwrap();
        let new_vertex = v_map[&h.vertex];
        let new_he = dst.add_halfedge(HalfEdge::new(new_vertex));
        he_map.insert(he_id, new_he);
    }
    for he_id in src.halfedge_ids() {
        let old = src.get_halfedge(he_id).unwrap();
        let new_he = he_map[&he_id];
        let new_h = dst.get_halfedge_mut(new_he).unwrap();
        new_h.twin = old.twin.and_then(|t| he_map.get(&t).copied());
        new_h.next = old.next.and_then(|n| he_map.get(&n).copied());
        new_h.prev = old.prev.and_then(|p| he_map.get(&p).copied());
    }
    for f_id in src.face_ids() {
        let new_f = dst.add_face(Face::new());
        f_map.insert(f_id, new_f);
    }
    for f_id in src.face_ids() {
        let new_f = f_map[&f_id];
        let old_f_he = src.get_face(f_id).unwrap().halfedge;
        if let Some(mapped) = old_f_he.and_then(|he| he_map.get(&he).copied()) {
            dst.get_face_mut(new_f).unwrap().halfedge = Some(mapped);
        }
    }
    for he_id in src.halfedge_ids() {
        let new_he = he_map[&he_id];
        let old_face = src.get_halfedge(he_id).unwrap().face;
        if let Some(mapped) = old_face.and_then(|f| f_map.get(&f).copied()) {
            dst.get_halfedge_mut(new_he).unwrap().face = Some(mapped);
        }
    }
    for v_id in src.vertex_ids() {
        let new_v = v_map[&v_id];
        let old_he = src.get_vertex(v_id).unwrap().halfedge;
        if let Some(mapped) = old_he.and_then(|he| he_map.get(&he).copied()) {
            dst.get_vertex_mut(new_v).unwrap().halfedge = Some(mapped);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_triangle_one_component() {
        let mesh = crate::test_util::build_icosphere(0); // 20 faces, connected
        let comps = connected_components(&mesh);
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].len(), 20);
        assert_eq!(component_count(&mesh), 1);
    }

    #[test]
    fn two_disconnected_triangles() {
        use crate::topology_ops::add_triangle;
        let mut mesh = MeshStorage::new();
        // Triangle 1
        let a = mesh.add_vertex(crate::storage::Vertex::new([0.0, 0.0, 0.0]));
        let b = mesh.add_vertex(crate::storage::Vertex::new([1.0, 0.0, 0.0]));
        let c = mesh.add_vertex(crate::storage::Vertex::new([0.0, 1.0, 0.0]));
        add_triangle(&mut mesh, a, b, c).unwrap();
        // Triangle 2 (disconnected)
        let d = mesh.add_vertex(crate::storage::Vertex::new([10.0, 0.0, 0.0]));
        let e = mesh.add_vertex(crate::storage::Vertex::new([11.0, 0.0, 0.0]));
        let f = mesh.add_vertex(crate::storage::Vertex::new([10.0, 1.0, 0.0]));
        add_triangle(&mut mesh, d, e, f).unwrap();

        let comps = connected_components(&mesh);
        assert_eq!(comps.len(), 2);
        assert_eq!(component_count(&mesh), 2);
    }

    #[test]
    fn component_of_face_invalid() {
        let mesh = MeshStorage::new();
        assert!(component_of_face(&mesh, FaceId::default()).is_none());
    }

    #[test]
    fn empty_mesh_no_components() {
        let mesh = MeshStorage::new();
        assert_eq!(connected_components(&mesh).len(), 0);
        assert_eq!(vertex_connected_components(&mesh).len(), 0);
    }

    // ---------- 拆分 / 提取 / 合并 ----------

    #[test]
    fn split_two_disconnected_triangles() {
        use crate::topology_ops::add_triangle;
        let mut mesh = MeshStorage::new();
        let a = mesh.add_vertex(crate::storage::Vertex::new([0.0, 0.0, 0.0]));
        let b = mesh.add_vertex(crate::storage::Vertex::new([1.0, 0.0, 0.0]));
        let c = mesh.add_vertex(crate::storage::Vertex::new([0.0, 1.0, 0.0]));
        add_triangle(&mut mesh, a, b, c).unwrap();
        let d = mesh.add_vertex(crate::storage::Vertex::new([10.0, 0.0, 0.0]));
        let e = mesh.add_vertex(crate::storage::Vertex::new([11.0, 0.0, 0.0]));
        let f = mesh.add_vertex(crate::storage::Vertex::new([10.0, 1.0, 0.0]));
        add_triangle(&mut mesh, d, e, f).unwrap();

        let parts = split_into_components(&mesh);
        assert_eq!(parts.len(), 2);
        // 每个分量应有 1 个面、3 个顶点
        for part in &parts {
            assert_eq!(part.face_count(), 1);
            assert_eq!(part.vertex_count(), 3);
            crate::topology_ops::validate_mesh(part).unwrap();
        }
    }

    #[test]
    fn extract_component_basic() {
        use crate::topology_ops::add_triangle;
        let mut mesh = MeshStorage::new();
        let a = mesh.add_vertex(crate::storage::Vertex::new([0.0, 0.0, 0.0]));
        let b = mesh.add_vertex(crate::storage::Vertex::new([1.0, 0.0, 0.0]));
        let c = mesh.add_vertex(crate::storage::Vertex::new([0.0, 1.0, 0.0]));
        let f1 = add_triangle(&mut mesh, a, b, c).unwrap();
        let d = mesh.add_vertex(crate::storage::Vertex::new([10.0, 0.0, 0.0]));
        let e = mesh.add_vertex(crate::storage::Vertex::new([11.0, 0.0, 0.0]));
        let f = mesh.add_vertex(crate::storage::Vertex::new([10.0, 1.0, 0.0]));
        add_triangle(&mut mesh, d, e, f).unwrap();

        let part = extract_component(&mesh, f1).unwrap();
        assert_eq!(part.face_count(), 1);
        assert_eq!(part.vertex_count(), 3);
        crate::topology_ops::validate_mesh(&part).unwrap();
    }

    #[test]
    fn merge_two_icospheres() {
        let a = crate::test_util::build_icosphere(0); // 20 faces
        let b = crate::test_util::build_icosphere(0);
        let merged = merge_meshes(&a, &b);
        assert_eq!(merged.face_count(), 40);
        assert_eq!(merged.vertex_count(), 24); // 12 * 2
        crate::topology_ops::validate_mesh(&merged).unwrap();
    }

    #[test]
    fn merge_preserves_boundaries() {
        use crate::topology_ops::add_triangle;
        let mut m1 = MeshStorage::new();
        let a = m1.add_vertex(crate::storage::Vertex::new([0.0, 0.0, 0.0]));
        let b = m1.add_vertex(crate::storage::Vertex::new([1.0, 0.0, 0.0]));
        let c = m1.add_vertex(crate::storage::Vertex::new([0.0, 1.0, 0.0]));
        add_triangle(&mut m1, a, b, c).unwrap();

        let mut m2 = MeshStorage::new();
        let d = m2.add_vertex(crate::storage::Vertex::new([10.0, 0.0, 0.0]));
        let e = m2.add_vertex(crate::storage::Vertex::new([11.0, 0.0, 0.0]));
        let f = m2.add_vertex(crate::storage::Vertex::new([10.0, 1.0, 0.0]));
        add_triangle(&mut m2, d, e, f).unwrap();

        let merged = merge_meshes(&m1, &m2);
        assert_eq!(merged.face_count(), 2);
        // 每个三角形各自有 3 条边界边
        assert!(!crate::traversal::is_closed(&merged));
        crate::topology_ops::validate_mesh(&merged).unwrap();
    }
}
