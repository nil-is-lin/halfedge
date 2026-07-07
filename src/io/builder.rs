//! Half-edge mesh building from vertex/face index arrays + shared helpers.

use std::collections::HashMap;

use crate::Scalar;
use crate::ids::{HalfEdgeId, VertexId};
use crate::storage::{Face, HalfEdge, MeshStorage, Vertex};

use super::MeshBuildError;

// ============================================================
// Shared mesh-building helpers
// ============================================================

/// Establish twin relationships and set boundary half-edge next/prev links.
///
/// Shared logic for [`build_mesh_from_vertices_and_faces`] and [`build_mesh_from_polygons`]:
///
/// 1. Iterate all directed edges, look up reverse edge:
///    - Found: set twins mutually (interior edge);
///    - Not found: create a boundary half-edge (`face = None`), set as twin;
/// 2. Set `next/prev` for each boundary half-edge to form closed boundary loops.
fn link_twins_and_boundary_loops(
    mesh: &mut MeshStorage,
    edge_map: &HashMap<(u32, u32), HalfEdgeId>,
    v_ids: &[VertexId],
) {
    // --- Step 3: establish twin relationships ---
    let directed_edges: Vec<(u32, u32, HalfEdgeId)> =
        edge_map.iter().map(|((a, b), h)| (*a, *b, *h)).collect();

    let mut boundary_twins: Vec<HalfEdgeId> = Vec::new();
    for (a, b, h) in &directed_edges {
        if mesh
            .get_halfedge(*h)
            .expect("halfedge exists in mesh")
            .twin
            .is_some()
        {
            continue;
        }
        if let Some(reverse_h) = edge_map.get(&(*b, *a)) {
            mesh.get_halfedge_mut(*h)
                .expect("halfedge just created or known to exist")
                .twin = Some(*reverse_h);
            mesh.get_halfedge_mut(*reverse_h)
                .expect("halfedge just created or known to exist")
                .twin = Some(*h);
        } else {
            let origin_v = v_ids[*a as usize];
            let twin_id = mesh.add_halfedge(HalfEdge::new(origin_v));
            mesh.get_halfedge_mut(*h)
                .expect("halfedge just created or known to exist")
                .twin = Some(twin_id);
            mesh.get_halfedge_mut(twin_id)
                .expect("halfedge just created or known to exist")
                .twin = Some(*h);
            boundary_twins.push(twin_id);
        }
    }

    // --- Step 4: set boundary half-edge next/prev ---
    set_boundary_next_prev(mesh, &boundary_twins);
}

/// Set boundary half-edge next/prev links to form closed boundary loops.
///
/// Algorithm:
/// - `bh.next`: rotate CCW around `bh.tip` from `bh.twin`, find boundary outgoing;
/// - `bh.prev`: rotate CW around `bh.origin` from `bh`, find twin that is boundary outgoing.
fn set_boundary_next_prev(mesh: &mut MeshStorage, boundary_twins: &[HalfEdgeId]) {
    for bh in boundary_twins {
        // bh.next: rotate CCW around bh.tip, find boundary outgoing
        let mut cur = mesh
            .get_halfedge(*bh)
            .expect("halfedge exists in mesh")
            .twin
            .expect("twin must be set at this point");
        let max_iter = mesh.halfedge_count() + 1;
        let mut next_bh = None;
        for _ in 0..max_iter {
            let prev = match mesh.get_halfedge(cur).and_then(|h| h.prev) {
                Some(p) => p,
                None => break,
            };
            let prev_twin = match mesh.get_halfedge(prev).and_then(|h| h.twin) {
                Some(t) => t,
                None => break,
            };
            if mesh
                .get_halfedge(prev_twin)
                .map(|h| h.face.is_none())
                .unwrap_or(false)
            {
                next_bh = Some(prev_twin);
                break;
            }
            cur = prev_twin;
        }
        if let Some(n) = next_bh {
            mesh.get_halfedge_mut(*bh)
                .expect("halfedge exists in mesh")
                .next = Some(n);
        }

        // bh.prev: rotate CW around bh.origin, find twin that is boundary outgoing
        let mut cur = *bh;
        let mut prev_bh = None;
        for _ in 0..max_iter {
            let twin = match mesh.get_halfedge(cur).and_then(|h| h.twin) {
                Some(t) => t,
                None => break,
            };
            let twin_next = match mesh.get_halfedge(twin).and_then(|h| h.next) {
                Some(n) => n,
                None => break,
            };
            let twin_next_twin = match mesh.get_halfedge(twin_next).and_then(|h| h.twin) {
                Some(t) => t,
                None => break,
            };
            if mesh
                .get_halfedge(twin_next_twin)
                .map(|h| h.face.is_none())
                .unwrap_or(false)
            {
                prev_bh = Some(twin_next_twin);
                break;
            }
            cur = twin_next;
        }
        if let Some(p) = prev_bh {
            mesh.get_halfedge_mut(*bh)
                .expect("halfedge exists in mesh")
                .prev = Some(p);
        }
    }
}

// ============================================================
// Build half-edge mesh from vertices + face indices
// ============================================================

/// Build a complete half-edge mesh from vertex positions and triangle face indices.
///
/// - `vertices`: vertex positions `[[x, y, z], ...]`;
/// - `faces`: triangle indices `[[v0, v1, v2], ...]`, 0-based, CCW winding.
///
/// Automatically builds twin / next / prev / boundary loops / vertex entry / face entry.
///
/// # Errors
/// Returns [`MeshBuildError::IndexOutOfRange`] when a face index is out of bounds.
///
/// ```
/// use halfedge::build_mesh_from_vertices_and_faces;
///
/// let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
/// let faces = vec![[0u32, 1, 2], [0, 2, 3]];
/// let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
/// assert_eq!(mesh.vertex_count(), 4);
/// assert_eq!(mesh.face_count(), 2);
/// ```
pub fn build_mesh_from_vertices_and_faces(
    vertices: &[[Scalar; 3]],
    faces: &[[u32; 3]],
) -> Result<MeshStorage, MeshBuildError> {
    let mut mesh = MeshStorage::new();
    // Pre-allocate: max half-edges = 3 * F (interior) + 3 * F (boundary twins) = 6 * F
    mesh.reserve(vertices.len(), faces.len() * 6, faces.len());

    // 1. Create all vertices
    let v_ids: Vec<VertexId> = vertices
        .iter()
        .map(|p| mesh.add_vertex(Vertex::new(*p)))
        .collect();

    // 2. Create 3 interior half-edges per face
    let mut edge_map: HashMap<(u32, u32), HalfEdgeId> = HashMap::new();
    let n_verts = v_ids.len();
    for face_idx in faces {
        let [i0, i1, i2] = *face_idx;
        // Bounds check: public builder returns Err rather than panic
        if (i0 as usize) >= n_verts || (i1 as usize) >= n_verts || (i2 as usize) >= n_verts {
            let bad_idx = if (i0 as usize) >= n_verts {
                i0
            } else if (i1 as usize) >= n_verts {
                i1
            } else {
                i2
            };
            return Err(MeshBuildError::IndexOutOfRange {
                idx: bad_idx,
                vertex_count: n_verts,
            });
        }
        let v0 = v_ids[i0 as usize];
        let v1 = v_ids[i1 as usize];
        let v2 = v_ids[i2 as usize];

        let h0 = mesh.add_halfedge(HalfEdge::new(v1)); // v0 -> v1
        let h1 = mesh.add_halfedge(HalfEdge::new(v2)); // v1 -> v2
        let h2 = mesh.add_halfedge(HalfEdge::new(v0)); // v2 -> v0

        let f_id = mesh.add_face(Face::new());
        for (he, next, prev) in [(h0, h1, h2), (h1, h2, h0), (h2, h0, h1)] {
            let h = mesh
                .get_halfedge_mut(he)
                .expect("halfedge just created or known to exist");
            h.next = Some(next);
            h.prev = Some(prev);
            h.face = Some(f_id);
        }
        mesh.get_face_mut(f_id).expect("face just created").halfedge = Some(h0);

        edge_map.insert((i0, i1), h0);
        edge_map.insert((i1, i2), h1);
        edge_map.insert((i2, i0), h2);

        // Vertex outgoing entry (if not yet set)
        if mesh
            .get_vertex(v0)
            .expect("vertex exists in mesh")
            .halfedge
            .is_none()
        {
            mesh.get_vertex_mut(v0)
                .expect("vertex exists in mesh")
                .halfedge = Some(h0);
        }
        if mesh
            .get_vertex(v1)
            .expect("vertex exists in mesh")
            .halfedge
            .is_none()
        {
            mesh.get_vertex_mut(v1)
                .expect("vertex exists in mesh")
                .halfedge = Some(h1);
        }
        if mesh
            .get_vertex(v2)
            .expect("vertex exists in mesh")
            .halfedge
            .is_none()
        {
            mesh.get_vertex_mut(v2)
                .expect("vertex exists in mesh")
                .halfedge = Some(h2);
        }
    }

    // 3-4. Establish twins + set boundary half-edge next/prev (shared logic)
    link_twins_and_boundary_loops(&mut mesh, &edge_map, &v_ids);

    Ok(mesh)
}

/// Build a complete half-edge mesh from vertex positions and arbitrary polygon face indices.
///
/// Unlike [`build_mesh_from_vertices_and_faces`], this function accepts polygons of any
/// vertex count (triangles, quads, pentagons, etc.), suitable for Catmull-Clark subdivision
/// and other polygon-input scenarios.
///
/// # Conventions
/// - Each face's vertex indices are arranged CCW (counter-clockwise from outside);
/// - Twins / next / prev / boundary loops are automatically established;
/// - **Note**: output polygon faces do not pass the triangular face check
///   (`FaceNotTriangular`) in [`crate::validate::validate_topology`], but other checks still pass.
///
/// # Errors
/// Returns [`MeshBuildError::IndexOutOfRange`] when a face index is out of bounds.
pub fn build_mesh_from_polygons(
    vertices: &[[Scalar; 3]],
    faces: &[Vec<u32>],
) -> Result<MeshStorage, MeshBuildError> {
    let mut mesh = MeshStorage::new();
    // Pre-allocate: max half-edges = sum(k_i) (interior) + sum(k_i) (boundary twins) = 2 * sum(k_i)
    let total_he: usize = faces
        .iter()
        .map(|f| f.len())
        .sum::<usize>()
        .saturating_mul(2);
    mesh.reserve(vertices.len(), total_he, faces.len());

    // 1. Create all vertices
    let v_ids: Vec<VertexId> = vertices
        .iter()
        .map(|p| mesh.add_vertex(Vertex::new(*p)))
        .collect();

    // 2. Create k interior half-edges per face
    let mut edge_map: HashMap<(u32, u32), HalfEdgeId> = HashMap::new();
    let n_verts = v_ids.len();
    let mut skipped_degenerate: u32 = 0;
    for face_idx in faces {
        let k = face_idx.len();
        if k < 3 {
            skipped_degenerate += 1;
            continue; // degenerate face, skip
        }
        // Bounds check: public builder returns Err rather than panic
        for idx in face_idx {
            if (*idx as usize) >= n_verts {
                return Err(MeshBuildError::IndexOutOfRange {
                    idx: *idx,
                    vertex_count: n_verts,
                });
            }
        }
        // Create k half-edges
        let mut he_ids: Vec<HalfEdgeId> = Vec::with_capacity(k);
        for i in 0..k {
            let v_from = v_ids[face_idx[i] as usize];
            let v_to = v_ids[face_idx[(i + 1) % k] as usize];
            let h = mesh.add_halfedge(HalfEdge::new(v_to)); // v_from -> v_to
            he_ids.push(h);
            // Vertex outgoing entry (if not yet set)
            if mesh
                .get_vertex(v_from)
                .expect("vertex exists in mesh")
                .halfedge
                .is_none()
            {
                mesh.get_vertex_mut(v_from)
                    .expect("vertex exists in mesh")
                    .halfedge = Some(h);
            }
        }
        // Create face and set next/prev/face
        let f_id = mesh.add_face(Face::new());
        for i in 0..k {
            let next = he_ids[(i + 1) % k];
            let prev = he_ids[(i + k - 1) % k];
            let h = mesh
                .get_halfedge_mut(he_ids[i])
                .expect("halfedge just created or known to exist");
            h.next = Some(next);
            h.prev = Some(prev);
            h.face = Some(f_id);
        }
        mesh.get_face_mut(f_id).expect("face just created").halfedge = Some(he_ids[0]);
        // Register directed edges
        for i in 0..k {
            let a = face_idx[i];
            let b = face_idx[(i + 1) % k];
            edge_map.insert((a, b), he_ids[i]);
        }
    }

    // 3-4. Establish twins + set boundary half-edge next/prev (shared logic)
    link_twins_and_boundary_loops(&mut mesh, &edge_map, &v_ids);

    if skipped_degenerate > 0 {
        log::warn!(
            "[halfedge::build_mesh_from_polygons] warning: skipped {skipped_degenerate} degenerate face(s) (vertex count < 3)"
        );
    }

    Ok(mesh)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::check_topology;

    /// Two triangles forming a quad:
    /// v0-v1-v2 triangle + v0-v2-v3 triangle (CCW, facing +z)
    fn make_quad_data() -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let faces = vec![[0, 1, 2], [0, 2, 3]];
        (vertices, faces)
    }

    #[test]
    fn build_mesh_basic_quad() {
        let (verts, faces) = make_quad_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.face_count(), 2);
        assert_eq!(mesh.halfedge_count(), 10);
    }

    #[test]
    fn build_mesh_passes_full_validation() {
        let (verts, faces) = make_quad_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        assert!(
            check_topology(&mesh).is_ok(),
            "built mesh should pass full validation: {:?}",
            check_topology(&mesh)
        );
    }

    #[test]
    fn build_mesh_closed_tetrahedron() {
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let faces = vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]];
        let mesh = build_mesh_from_vertices_and_faces(&vertices, &faces).unwrap();
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.face_count(), 4);
        assert_eq!(mesh.halfedge_count(), 12);
        assert!(check_topology(&mesh).is_ok());
    }

    #[test]
    fn build_mesh_empty_inputs_returns_empty_mesh() {
        let mesh = build_mesh_from_vertices_and_faces(&[], &[]).unwrap();
        assert_eq!(mesh.vertex_count(), 0);
        assert_eq!(mesh.face_count(), 0);
    }

    #[test]
    fn build_mesh_vertices_no_faces() {
        let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mesh = build_mesh_from_vertices_and_faces(&vertices, &[]).unwrap();
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.face_count(), 0);
    }

    #[test]
    fn build_polygons_empty_inputs_returns_empty_mesh() {
        let mesh = build_mesh_from_polygons(&[], &[]).unwrap();
        assert_eq!(mesh.vertex_count(), 0);
        assert_eq!(mesh.face_count(), 0);
    }

    #[test]
    fn build_mesh_face_index_out_of_range_returns_err() {
        let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = [[0u32, 1, 5]];
        let result = build_mesh_from_vertices_and_faces(&vertices, &faces);
        assert!(result.is_err());
        match result {
            Err(MeshBuildError::IndexOutOfRange { idx, vertex_count }) => {
                assert_eq!(idx, 5);
                assert_eq!(vertex_count, 3);
            }
            _ => panic!("expected IndexOutOfRange error"),
        }
    }

    #[test]
    fn build_polygons_face_index_out_of_range_returns_err() {
        let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = [vec![0u32, 1, 5]];
        let result = build_mesh_from_polygons(&vertices, &faces);
        assert!(result.is_err());
        match result {
            Err(MeshBuildError::IndexOutOfRange { idx, vertex_count }) => {
                assert_eq!(idx, 5);
                assert_eq!(vertex_count, 3);
            }
            _ => panic!("expected IndexOutOfRange error"),
        }
    }

    #[test]
    fn build_polygons_skips_degenerate_face_2_verts() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let faces = [vec![0u32, 1]];
        let mesh = build_mesh_from_polygons(&vertices, &faces).unwrap();
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.face_count(), 0);
    }

    #[test]
    fn build_polygons_mixed_degenerate_and_valid() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let faces = [vec![], vec![0u32, 1, 2]];
        let mesh = build_mesh_from_polygons(&vertices, &faces).unwrap();
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.face_count(), 1);
    }
}
