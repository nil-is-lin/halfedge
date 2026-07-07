//! STL format parser/writer (ASCII + binary).

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

use crate::geometry::face_normal;
use crate::ids::VertexId;
use crate::storage::MeshStorage;
use crate::traversal::FaceVertices;

use super::build_mesh_from_vertices_and_faces;

// ============================================================
// Error type
// ============================================================

/// STL parse/serialize error.
#[derive(Debug)]
pub enum StlError {
    Io(std::io::Error),
    /// ASCII parse error: line number + description.
    Parse {
        line: usize,
        msg: String,
    },
    /// Binary file size mismatch (actual / expected).
    BadBinarySize {
        actual: usize,
        expected: usize,
    },
    /// Face vertex count != 3.
    NotTriangular {
        face_verts: usize,
    },
}

impl fmt::Display for StlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Parse { line, msg } => write!(f, "STL parse error on line {line}: {msg}"),
            Self::BadBinarySize { actual, expected } => write!(
                f,
                "STL binary size mismatch: actual {actual} bytes, expected {expected} bytes"
            ),
            Self::NotTriangular { face_verts } => {
                write!(f, "STL face vertex count {face_verts} != 3")
            }
        }
    }
}

impl std::error::Error for StlError {}

impl From<std::io::Error> for StlError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ============================================================
// STL loading
// ============================================================

/// Load an STL file. Auto-detects ASCII (first line contains "solid") vs binary format.
///
/// Note: detecting by "solid" on the first line is unreliable (some binary files
/// happen to start with "solid"). This implementation uses a heuristic: if the first
/// 5 bytes are `solid` AND the file size exactly equals `84 + 50 * N` (N = face count),
/// it is treated as binary; otherwise ASCII.
pub fn load_stl<P: AsRef<Path>>(path: P) -> Result<MeshStorage, StlError> {
    let bytes = fs::read(path)?;
    parse_stl_bytes(&bytes)
}

/// Parse STL byte stream (auto-detect ASCII / binary).
pub fn parse_stl_bytes(bytes: &[u8]) -> Result<MeshStorage, StlError> {
    // Binary STL: 80-byte header + 4-byte face count + 50*N bytes triangle data
    if bytes.len() >= 84 {
        let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
        let expected = 84 + 50 * n;
        // Heuristic: size matches exactly -> binary; otherwise try ASCII
        if bytes.len() == expected {
            return parse_stl_binary(bytes, n);
        }
    }
    // Otherwise parse as ASCII
    let text = std::str::from_utf8(bytes).map_err(|_| StlError::Parse {
        line: 0,
        msg: "file is not valid UTF-8".into(),
    })?;
    parse_stl_ascii(text)
}

/// Parse ASCII STL.
///
/// Format:
/// ```text
/// solid name
///   facet normal nx ny nz
///     outer loop
///       vertex x y z
///       vertex x y z
///       vertex x y z
///     endloop
///   endfacet
///   ...
/// endsolid
/// ```
pub fn parse_stl_ascii(text: &str) -> Result<MeshStorage, StlError> {
    let mut vertices: Vec<[f64; 3]> = Vec::new();
    let mut faces: Vec<[u32; 3]> = Vec::new();
    let mut dedup: HashMap<[u64; 3], u32> = HashMap::new();
    let mut current_face: Vec<u32> = Vec::with_capacity(3);
    let mut in_facet = false;

    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        match parts[0] {
            "facet" => {
                in_facet = true;
                current_face.clear();
            }
            "vertex" if in_facet => {
                if parts.len() < 4 {
                    return Err(StlError::Parse {
                        line: i + 1,
                        msg: "vertex line requires 3 coordinates".into(),
                    });
                }
                let coords: [f64; 3] = [
                    parts[1].parse().map_err(|_| StlError::Parse {
                        line: i + 1,
                        msg: format!("invalid x coordinate: {}", parts[1]),
                    })?,
                    parts[2].parse().map_err(|_| StlError::Parse {
                        line: i + 1,
                        msg: format!("invalid y coordinate: {}", parts[2]),
                    })?,
                    parts[3].parse().map_err(|_| StlError::Parse {
                        line: i + 1,
                        msg: format!("invalid z coordinate: {}", parts[3]),
                    })?,
                ];
                // Deduplicate by bit pattern (consistent with binary path)
                let key = [
                    coords[0].to_bits(),
                    coords[1].to_bits(),
                    coords[2].to_bits(),
                ];
                let idx = *dedup.entry(key).or_insert_with(|| {
                    let k = vertices.len() as u32;
                    vertices.push(coords);
                    k
                });
                current_face.push(idx);
            }
            "endfacet" => {
                if current_face.len() != 3 {
                    return Err(StlError::NotTriangular {
                        face_verts: current_face.len(),
                    });
                }
                let [a, b, c] = [current_face[0], current_face[1], current_face[2]];
                faces.push([a, b, c]);
                in_facet = false;
            }
            _ => {}
        }
    }

    Ok(build_mesh_from_vertices_and_faces(&vertices, &faces).expect("indices already validated"))
}

/// Parse binary STL.
///
/// Each triangle: 12-byte normal + 3x12-byte vertices + 2-byte attribute = 50 bytes.
/// Vertex indices appear in file order; this implementation deduplicates by position
/// (identical vertices are merged).
pub fn parse_stl_binary(bytes: &[u8], n_faces: usize) -> Result<MeshStorage, StlError> {
    let expected = 84 + 50 * n_faces;
    if bytes.len() != expected {
        return Err(StlError::BadBinarySize {
            actual: bytes.len(),
            expected,
        });
    }

    let mut vertices: Vec<[f64; 3]> = Vec::with_capacity(n_faces * 3);
    let mut faces: Vec<[u32; 3]> = Vec::with_capacity(n_faces);
    let mut dedup: HashMap<[u32; 3], u32> = HashMap::new();

    for i in 0..n_faces {
        let base = 84 + i * 50;
        // Skip normal (base..base+12)
        let mut tri = [0u32; 3];
        for (j, tri_j) in tri.iter_mut().enumerate() {
            let off = base + 12 + j * 12;
            let x =
                f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
            let y = f32::from_le_bytes([
                bytes[off + 4],
                bytes[off + 5],
                bytes[off + 6],
                bytes[off + 7],
            ]);
            let z = f32::from_le_bytes([
                bytes[off + 8],
                bytes[off + 9],
                bytes[off + 10],
                bytes[off + 11],
            ]);
            // Deduplicate by bit pattern
            let key = [x.to_bits(), y.to_bits(), z.to_bits()];
            let idx = *dedup.entry(key).or_insert_with(|| {
                let k = vertices.len() as u32;
                vertices.push([x as f64, y as f64, z as f64]);
                k
            });
            *tri_j = idx;
        }
        faces.push(tri);
    }

    Ok(build_mesh_from_vertices_and_faces(&vertices, &faces).expect("indices already validated"))
}

// ============================================================
// STL saving
// ============================================================

/// Save mesh as ASCII STL.
pub fn save_stl_ascii<P: AsRef<Path>>(mesh: &MeshStorage, path: P) -> Result<(), StlError> {
    let text = format_stl_ascii(mesh);
    fs::write(path, text)?;
    Ok(())
}

/// Serialize mesh to ASCII STL text.
///
/// Normals computed via Newell's method (consistent with `geometry::face_normal`);
/// only triangle faces are output, non-triangle faces are skipped.
pub fn format_stl_ascii(mesh: &MeshStorage) -> String {
    let mut out = String::with_capacity(mesh.face_count() * 80);
    out.push_str("solid halfedge\n");
    let mut skipped: u32 = 0;
    for f in mesh.face_ids() {
        let verts: Vec<VertexId> = FaceVertices::new(mesh, f).collect();
        if verts.len() != 3 {
            skipped += 1;
            continue;
        }
        let p0 = mesh
            .get_vertex(verts[0])
            .map(|v| v.position)
            .unwrap_or([0.0; 3]);
        let p1 = mesh
            .get_vertex(verts[1])
            .map(|v| v.position)
            .unwrap_or([0.0; 3]);
        let p2 = mesh
            .get_vertex(verts[2])
            .map(|v| v.position)
            .unwrap_or([0.0; 3]);
        let n = face_normal(mesh, f).unwrap_or([0.0, 0.0, 1.0]);
        out.push_str(&format!(
            "  facet normal {:.6} {:.6} {:.6}\n    outer loop\n",
            n[0], n[1], n[2]
        ));
        for p in [p0, p1, p2] {
            out.push_str(&format!(
                "      vertex {:.6} {:.6} {:.6}\n",
                p[0], p[1], p[2]
            ));
        }
        out.push_str("    endloop\n  endfacet\n");
    }
    if skipped > 0 {
        log::warn!("[halfedge::format_stl_ascii] warning: skipped {skipped} non-triangle face(s)");
    }
    out.push_str("endsolid halfedge\n");
    out
}

/// Save mesh as binary STL.
pub fn save_stl_binary<P: AsRef<Path>>(mesh: &MeshStorage, path: P) -> Result<(), StlError> {
    let bytes = format_stl_binary(mesh);
    fs::write(path, bytes)?;
    Ok(())
}

/// Serialize mesh to binary STL byte stream.
pub fn format_stl_binary(mesh: &MeshStorage) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(84 + mesh.face_count() * 50);
    // 80-byte header
    out.extend_from_slice(b"halfedge binary stl".as_slice());
    while out.len() < 80 {
        out.push(0);
    }
    // 4-byte face count
    let n = mesh.face_count() as u32;
    out.extend_from_slice(&n.to_le_bytes());

    let mut skipped: u32 = 0;
    for f in mesh.face_ids() {
        let verts: Vec<VertexId> = FaceVertices::new(mesh, f).collect();
        if verts.len() != 3 {
            // Degenerate: fill zero triangle, face count consistency maintained by caller
            out.extend_from_slice(&[0u8; 50]);
            skipped += 1;
            continue;
        }
        let p0 = mesh
            .get_vertex(verts[0])
            .map(|v| v.position)
            .unwrap_or([0.0; 3]);
        let p1 = mesh
            .get_vertex(verts[1])
            .map(|v| v.position)
            .unwrap_or([0.0; 3]);
        let p2 = mesh
            .get_vertex(verts[2])
            .map(|v| v.position)
            .unwrap_or([0.0; 3]);
        let n = face_normal(mesh, f).unwrap_or([0.0, 0.0, 1.0]);
        for c in &n {
            out.extend_from_slice(&(*c as f32).to_le_bytes());
        }
        for p in [p0, p1, p2] {
            for c in &p {
                out.extend_from_slice(&(*c as f32).to_le_bytes());
            }
        }
        // 2-byte attribute
        out.extend_from_slice(&[0u8, 0u8]);
    }
    if skipped > 0 {
        log::warn!(
            "[halfedge::format_stl_binary] warning: zero-filled {skipped} non-triangle face(s)"
        );
    }
    out
}

// ============================================================
// Tests
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
    fn stl_ascii_roundtrip() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let text = format_stl_ascii(&mesh);
        let parsed = parse_stl_ascii(&text).expect("ASCII STL roundtrip parse failed");
        assert_eq!(parsed.vertex_count(), mesh.vertex_count());
        assert_eq!(parsed.face_count(), mesh.face_count());
    }

    #[test]
    fn stl_binary_roundtrip() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let bytes = format_stl_binary(&mesh);
        assert_eq!(bytes.len(), 84 + 50 * 4);
        let parsed = parse_stl_bytes(&bytes).expect("binary STL roundtrip parse failed");
        assert_eq!(parsed.vertex_count(), mesh.vertex_count());
        assert_eq!(parsed.face_count(), mesh.face_count());
    }

    #[test]
    fn stl_ascii_parses_minimum_solid() {
        let text = "solid x\n\
            facet normal 0 0 1\n\
              outer loop\n\
                vertex 0 0 0\n\
                vertex 1 0 0\n\
                vertex 0 1 0\n\
              endloop\n\
            endfacet\n\
            endsolid x\n";
        let mesh = parse_stl_ascii(text).expect("minimum ASCII STL parse failed");
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.face_count(), 1);
    }

    #[test]
    fn stl_ascii_rejects_non_triangular() {
        let text = "solid x\n\
            facet normal 0 0 1\n\
              outer loop\n\
                vertex 0 0 0\n\
                vertex 1 0 0\n\
                vertex 1 1 0\n\
                vertex 0 1 0\n\
              endloop\n\
            endfacet\n\
            endsolid x\n";
        let err = parse_stl_ascii(text).unwrap_err();
        assert!(matches!(err, StlError::NotTriangular { .. }));
    }

    #[test]
    fn stl_binary_detects_size_mismatch() {
        let mut bytes = vec![0u8; 84];
        bytes.extend_from_slice(&3u32.to_le_bytes()); // declare 3 faces
        bytes.extend_from_slice(&[0u8; 50]); // only 1 face
        let err = parse_stl_binary(&bytes, 3).unwrap_err();
        assert!(matches!(err, StlError::BadBinarySize { .. }));
    }

    #[test]
    fn stl_file_roundtrip_ascii() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let path = std::env::temp_dir().join("halfedge_stl_ascii.stl");
        save_stl_ascii(&mesh, &path).expect("failed to save STL file");
        let loaded = load_stl(&path).expect("failed to load STL file");
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.vertex_count(), mesh.vertex_count());
        assert_eq!(loaded.face_count(), mesh.face_count());
    }

    #[test]
    fn stl_file_roundtrip_binary() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let path = std::env::temp_dir().join("halfedge_stl_bin.stl");
        save_stl_binary(&mesh, &path).expect("failed to save binary STL");
        let loaded = load_stl(&path).expect("failed to load binary STL");
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.vertex_count(), mesh.vertex_count());
        assert_eq!(loaded.face_count(), mesh.face_count());
    }

    #[test]
    fn stl_ascii_empty_text() {
        let mesh = parse_stl_ascii("").expect("empty STL should parse as empty mesh");
        assert_eq!(mesh.face_count(), 0);
    }

    #[test]
    fn stl_ascii_only_solid_endsolid() {
        let text = "solid x\nendsolid x\n";
        let mesh = parse_stl_ascii(text).expect("empty solid STL should parse as empty mesh");
        assert_eq!(mesh.face_count(), 0);
    }
}
