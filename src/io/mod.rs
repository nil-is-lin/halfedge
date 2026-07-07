//! IO module: OBJ / PLY / STL / OFF / glTF read/write + half-edge mesh building from vertex/face indices.
//!
//! ## Features
//! - [`build_mesh_from_vertices_and_faces`]: build a **complete** half-edge mesh
//!   (with twin / next / prev / boundary loops) from vertex positions and triangle indices.
//! - [`build_mesh_from_polygons`]: build a half-edge mesh from arbitrary polygon faces.
//! - OBJ: `load_obj` / `parse_obj` (triangles and n-gons),
//!   `save_obj` / `format_obj` (serialize to OBJ text, any face vertex count).
//! - PLY: `load_ply` / `parse_ply` / `save_ply` / `format_ply` (ASCII + binary little-endian).
//! - STL: `load_stl` / `parse_stl_ascii` / `parse_stl_binary` / `parse_stl_bytes`
//!   (auto-detect ASCII / binary), `save_stl_ascii` / `save_stl_binary` /
//!   `format_stl_ascii` / `format_stl_binary`.
//! - OFF: `load_off` / `parse_off` / `save_off` / `format_off` (ASCII).
//! - glTF: `load_glb` / `parse_glb` / `save_glb` / `format_glb` (GLB binary, minimal subset).
//! - Unified entry: `load_mesh` / `save_mesh`, dispatches by extension `.obj` / `.ply` / `.stl` / `.off` / `.glb`.
//!
//! ## OBJ format conventions
//! ```text
//! v x y z          # vertex (1-based index)
//! f i j k ...      # face (vertex indices, supports v/vt/vn form, only v is used; supports 3+ vertices)
//! ```
//! - Vertex indices are 1-based; negative indices count from the end (OBJ standard).
//! - Non-`v` / `f` lines (e.g. `vt`, `vn`, `#`, blank lines) are ignored.
//! - Faces with fewer than 3 vertices return an error.
//!
//! ## Boundary half-edge construction
//! 1. Each triangle face creates 3 interior half-edges, with `next/prev/face` set CCW;
//! 2. `HashMap<(u32, u32), HalfEdgeId>` records each directed edge;
//! 3. For each directed edge `(a, b)`, look up `(b, a)`:
//!    - Hit: set twins mutually (interior edge);
//!    - Miss: create a boundary half-edge (`face = None`), set as twin;
//! 4. Boundary half-edge `next/prev` given by:
//!    $$
//!    \text{bh.next} = \text{bh.twin.prev.twin}, \quad
//!    \text{bh.prev} = \text{bh.twin.next.twin}
//!    $$

mod builder;
mod gltf;
mod obj;
mod off;
mod ply;
mod stl;

pub use builder::{build_mesh_from_polygons, build_mesh_from_vertices_and_faces};
pub use gltf::*;
pub use obj::*;
pub use off::*;
pub use ply::*;
pub use stl::*;

use std::fmt;
use std::path::Path;

use crate::storage::MeshStorage;

// ============================================================
// MeshBuildError
// ============================================================

/// Mesh build error.
#[derive(Debug)]
pub enum MeshBuildError {
    /// Face index out of range: `(index, vertex count)`.
    IndexOutOfRange { idx: u32, vertex_count: usize },
}

impl fmt::Display for MeshBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IndexOutOfRange { idx, vertex_count } => {
                write!(
                    f,
                    "face index {} out of range (vertex count {})",
                    idx, vertex_count
                )
            }
        }
    }
}

impl std::error::Error for MeshBuildError {}

// ============================================================
// Unified error type
// ============================================================

/// Unified IO error type.
#[derive(Debug)]
pub enum MeshError {
    Obj(ObjError),
    Ply(PlyError),
    Stl(StlError),
    Off(OffError),
    Gltf(GltfError),
    Build(MeshBuildError),
    UnsupportedFormat(String),
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Obj(e) => write!(f, "OBJ error: {e}"),
            Self::Ply(e) => write!(f, "PLY error: {e}"),
            Self::Stl(e) => write!(f, "STL error: {e}"),
            Self::Off(e) => write!(f, "OFF error: {e}"),
            Self::Gltf(e) => write!(f, "glTF error: {e}"),
            Self::Build(e) => write!(f, "mesh build error: {e}"),
            Self::UnsupportedFormat(s) => write!(f, "unsupported format: {s}"),
        }
    }
}

impl std::error::Error for MeshError {}

impl From<ObjError> for MeshError {
    fn from(e: ObjError) -> Self {
        Self::Obj(e)
    }
}

impl From<PlyError> for MeshError {
    fn from(e: PlyError) -> Self {
        Self::Ply(e)
    }
}

impl From<StlError> for MeshError {
    fn from(e: StlError) -> Self {
        Self::Stl(e)
    }
}

impl From<OffError> for MeshError {
    fn from(e: OffError) -> Self {
        Self::Off(e)
    }
}

impl From<GltfError> for MeshError {
    fn from(e: GltfError) -> Self {
        Self::Gltf(e)
    }
}

impl From<MeshBuildError> for MeshError {
    fn from(e: MeshBuildError) -> Self {
        Self::Build(e)
    }
}

// ============================================================
// Unified load/save
// ============================================================

/// Auto-detect file format and load a mesh.
///
/// Selects parser by file extension:
/// - `.obj` -> OBJ (triangles and n-gons)
/// - `.ply` -> PLY (auto-detect ASCII / binary little-endian)
/// - `.stl` -> STL (auto-detect ASCII / binary)
/// - `.off` -> OFF (ASCII)
/// - `.glb` / `.gltf` -> glTF GLB (minimal subset)
pub fn load_mesh<P: AsRef<Path>>(path: P) -> Result<MeshStorage, MeshError> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "obj" => Ok(load_obj(path)?),
        "ply" => Ok(load_ply(path)?),
        "stl" => Ok(load_stl(path)?),
        "off" => Ok(load_off(path)?),
        "glb" | "gltf" => Ok(load_glb(path)?),
        other => Err(MeshError::UnsupportedFormat(other.into())),
    }
}

/// Auto-detect file format and save a mesh.
///
/// Selects serializer by file extension:
/// - `.obj` -> OBJ (ASCII)
/// - `.ply` -> PLY (ASCII; for binary call `save_ply_binary` directly)
/// - `.stl` -> STL (ASCII; for binary call `save_stl_binary` directly)
/// - `.off` -> OFF (ASCII)
/// - `.glb` / `.gltf` -> glTF GLB (binary)
pub fn save_mesh<P: AsRef<Path>>(mesh: &MeshStorage, path: P) -> Result<(), MeshError> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "obj" => {
            save_obj(mesh, path)?;
            Ok(())
        }
        "ply" => {
            save_ply(mesh, path)?;
            Ok(())
        }
        "stl" => {
            save_stl_ascii(mesh, path)?;
            Ok(())
        }
        "off" => {
            save_off(mesh, path)?;
            Ok(())
        }
        "glb" | "gltf" => {
            save_glb(mesh, path)?;
            Ok(())
        }
        other => Err(MeshError::UnsupportedFormat(other.into())),
    }
}

// ============================================================
// Tests (auto-detect format)
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tetra_data() -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let faces = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        (vertices, faces)
    }

    #[test]
    fn auto_detect_obj() {
        let mesh = crate::test_util::build_icosphere(0);
        let path = std::env::temp_dir().join("halfedge_autodetect.obj");
        save_mesh(&mesh, &path).unwrap();
        let loaded = load_mesh(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.face_count(), mesh.face_count());
    }

    #[test]
    fn auto_detect_ply() {
        let mesh = crate::test_util::build_icosphere(0);
        let path = std::env::temp_dir().join("halfedge_autodetect.ply");
        save_mesh(&mesh, &path).unwrap();
        let loaded = load_mesh(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.face_count(), mesh.face_count());
    }

    #[test]
    fn auto_detect_unsupported() {
        let path = std::env::temp_dir().join("halfedge_autodetect.unknown");
        let err = load_mesh(&path).unwrap_err();
        assert!(matches!(err, MeshError::UnsupportedFormat(_)));
    }

    #[test]
    fn auto_detect_stl() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let path = std::env::temp_dir().join("halfedge_autodetect.stl");
        save_mesh(&mesh, &path).expect("save_mesh(.stl) failed");
        let loaded = load_mesh(&path).expect("load_mesh(.stl) failed");
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.face_count(), mesh.face_count());
    }

    #[test]
    fn auto_detect_off() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let path = std::env::temp_dir().join("halfedge_autodetect.off");
        save_mesh(&mesh, &path).expect("save_mesh(.off) failed");
        let loaded = load_mesh(&path).expect("load_mesh(.off) failed");
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.face_count(), mesh.face_count());
    }

    #[test]
    fn auto_detect_glb() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let path = std::env::temp_dir().join("halfedge_autodetect.glb");
        save_mesh(&mesh, &path).expect("save_mesh(.glb) failed");
        let loaded = load_mesh(&path).expect("load_mesh(.glb) failed");
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.face_count(), mesh.face_count());
    }
}
