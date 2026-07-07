//! OBJ format parser/writer.

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

use crate::ids::VertexId;
use crate::storage::MeshStorage;
use crate::traversal::FaceHalfEdges;

use super::build_mesh_from_polygons;

// ============================================================
// Error type
// ============================================================

/// OBJ read/write error.
#[derive(Debug)]
pub enum ObjError {
    /// File IO error.
    Io(std::io::Error),
    /// Parse error: line number and description.
    Parse { line: usize, msg: String },
    /// Face index out of range.
    IndexOutOfRange {
        line: usize,
        idx: i64,
        vertex_count: usize,
    },
    /// Non-triangular face (vertex count != 3).
    NotTriangular { line: usize, face_verts: usize },
}

impl fmt::Display for ObjError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::Parse { line, msg } => write!(f, "parse error on line {}: {}", line, msg),
            Self::IndexOutOfRange {
                line,
                idx,
                vertex_count,
            } => write!(
                f,
                "index {} out of range on line {} (vertex count {})",
                idx, line, vertex_count
            ),
            Self::NotTriangular { line, face_verts } => {
                write!(
                    f,
                    "face on line {} has {} vertices != 3, only triangles supported",
                    line, face_verts
                )
            }
        }
    }
}

impl std::error::Error for ObjError {}

impl From<std::io::Error> for ObjError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ============================================================
// OBJ loading
// ============================================================

/// Load an OBJ file, reading only `v` and `f` lines, supporting arbitrary face vertex counts.
///
/// ```
/// use halfedge::{build_mesh_from_vertices_and_faces, save_obj, load_obj};
///
/// let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
/// let faces = vec![[0u32, 1, 2]];
/// let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
/// let path = std::env::temp_dir().join("halfedge_doc_load.obj");
/// save_obj(&mesh, &path).unwrap();
/// let loaded = load_obj(&path).unwrap();
/// let _ = std::fs::remove_file(&path);
/// assert_eq!(loaded.vertex_count(), 3);
/// assert_eq!(loaded.face_count(), 1);
/// ```
pub fn load_obj<P: AsRef<Path>>(path: P) -> Result<MeshStorage, ObjError> {
    let text = fs::read_to_string(path)?;
    parse_obj(&text)
}

/// Parse OBJ text into a half-edge mesh. Supports triangles and n-gons.
pub fn parse_obj(text: &str) -> Result<MeshStorage, ObjError> {
    let mut vertices: Vec<[f64; 3]> = Vec::new();
    let mut faces: Vec<Vec<u32>> = Vec::new();

    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut tokens = line.split_whitespace();
        let kind = match tokens.next() {
            Some(k) => k,
            None => continue,
        };
        match kind {
            "v" => {
                let coords: Vec<f64> = tokens
                    .take(3)
                    .map(|t| {
                        t.parse::<f64>().map_err(|_| ObjError::Parse {
                            line: line_no + 1,
                            msg: format!("cannot parse vertex coordinate: {}", t),
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if coords.len() != 3 {
                    return Err(ObjError::Parse {
                        line: line_no + 1,
                        msg: "vertex line missing coordinate components".into(),
                    });
                }
                vertices.push([coords[0], coords[1], coords[2]]);
            }
            "f" => {
                let verts: Vec<i64> = tokens
                    .map(|t| {
                        // Support v/vt/vn format, only take first component
                        let v_part = t.split('/').next().unwrap_or(t);
                        v_part.parse::<i64>().map_err(|_| ObjError::Parse {
                            line: line_no + 1,
                            msg: format!("cannot parse face index: {}", t),
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if verts.len() < 3 {
                    return Err(ObjError::NotTriangular {
                        line: line_no + 1,
                        face_verts: verts.len(),
                    });
                }
                let to_zero = |i: i64| -> Result<u32, ObjError> {
                    let zero_based = if i > 0 {
                        (i - 1) as usize
                    } else if i < 0 {
                        // Negative: count from end
                        let n = vertices.len() as i64;
                        if n + i < 0 {
                            return Err(ObjError::IndexOutOfRange {
                                line: line_no + 1,
                                idx: i,
                                vertex_count: vertices.len(),
                            });
                        }
                        (n + i) as usize
                    } else {
                        return Err(ObjError::Parse {
                            line: line_no + 1,
                            msg: "face index cannot be 0".into(),
                        });
                    };
                    if zero_based >= vertices.len() {
                        return Err(ObjError::IndexOutOfRange {
                            line: line_no + 1,
                            idx: i,
                            vertex_count: vertices.len(),
                        });
                    }
                    Ok(zero_based as u32)
                };
                let indices: Vec<u32> = verts
                    .iter()
                    .map(|&i| to_zero(i))
                    .collect::<Result<_, _>>()?;
                faces.push(indices);
            }
            _ => {
                // Ignore other lines (vt, vn, g, o, s, mtllib, usemtl, ...)
            }
        }
    }

    Ok(build_mesh_from_polygons(&vertices, &faces).expect("indices already validated"))
}

// ============================================================
// OBJ saving
// ============================================================

/// Save a mesh to an OBJ file.
///
/// ```
/// use halfedge::{build_mesh_from_vertices_and_faces, save_obj, load_obj};
///
/// let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
/// let faces = vec![[0u32, 1, 2]];
/// let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
/// let path = std::env::temp_dir().join("halfedge_doc_save.obj");
/// save_obj(&mesh, &path).unwrap();
/// let loaded = load_obj(&path).unwrap();
/// let _ = std::fs::remove_file(&path);
/// assert_eq!(loaded.vertex_count(), 3);
/// ```
pub fn save_obj<P: AsRef<Path>>(mesh: &MeshStorage, path: P) -> Result<(), ObjError> {
    let text = format_obj(mesh);
    fs::write(path, text)?;
    Ok(())
}

/// Serialize a mesh to OBJ text.
///
/// - Vertices output in `vertex_ids()` order, assigned 1-based indices;
/// - Faces output in `face_ids()` order, each line `f i j k`.
pub fn format_obj(mesh: &MeshStorage) -> String {
    let mut v_index: HashMap<VertexId, u32> = HashMap::new();
    let mut out = String::new();
    for (next_idx, v_id) in (1u32..).zip(mesh.vertex_ids()) {
        v_index.insert(v_id, next_idx);
        let p = mesh
            .get_vertex(v_id)
            .expect("vertex exists in mesh")
            .position;
        out.push_str(&format!("v {:.6} {:.6} {:.6}\n", p[0], p[1], p[2]));
    }
    let mut skipped: u32 = 0;
    for f_id in mesh.face_ids() {
        let verts: Vec<u32> = FaceHalfEdges::new(mesh, f_id)
            .filter_map(|he| mesh.get_halfedge(he))
            .map(|h| h.vertex)
            .filter_map(|v| v_index.get(&v).copied())
            .collect();
        if verts.len() < 3 {
            skipped += 1;
            continue; // skip degenerate face
        }
        out.push('f');
        for v in &verts {
            out.push(' ');
            out.push_str(&v.to_string());
        }
        out.push('\n');
    }
    if skipped > 0 {
        log::warn!(
            "[halfedge::format_obj] warning: skipped {skipped} degenerate face(s) (vertex count < 3)"
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
    use crate::build_mesh_from_vertices_and_faces;
    use crate::validate::check_topology;

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
    fn obj_roundtrip_quad() {
        let (verts, faces) = make_quad_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let text = format_obj(&mesh);
        let mesh2 = parse_obj(&text).unwrap();
        assert_eq!(mesh2.vertex_count(), mesh.vertex_count());
        assert_eq!(mesh2.face_count(), mesh.face_count());
        assert_eq!(mesh2.halfedge_count(), mesh.halfedge_count());
        assert!(check_topology(&mesh2).is_ok());
    }

    #[test]
    fn obj_roundtrip_tetrahedron() {
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let faces = vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]];
        let mesh = build_mesh_from_vertices_and_faces(&vertices, &faces).unwrap();
        let text = format_obj(&mesh);
        let mesh2 = parse_obj(&text).unwrap();
        assert_eq!(mesh2.vertex_count(), 4);
        assert_eq!(mesh2.face_count(), 4);
        assert_eq!(mesh2.halfedge_count(), 12);
        assert!(check_topology(&mesh2).is_ok());
    }

    #[test]
    fn obj_parse_skips_comments_and_other_lines() {
        let text = r#"
# this is a test OBJ
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
vt 0.0 0.0
vn 0.0 0.0 1.0
f 1 2 3
g mesh
usemtl default
"#;
        let mesh = parse_obj(text).unwrap();
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.face_count(), 1);
        assert_eq!(mesh.halfedge_count(), 6);
    }

    #[test]
    fn obj_parse_supports_v_vt_vn_format() {
        let text = r#"
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f 1/1/1 2/2/1 3/3/1
"#;
        let mesh = parse_obj(text).unwrap();
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.face_count(), 1);
    }

    #[test]
    fn obj_parse_negative_indices() {
        let text = r#"
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f -3 -2 -1
"#;
        let mesh = parse_obj(text).unwrap();
        assert_eq!(mesh.face_count(), 1);
        assert!(check_topology(&mesh).is_ok());
    }

    #[test]
    fn obj_parse_quadrilateral_face_succeeds() {
        let text = r#"
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
"#;
        let mesh = parse_obj(text).expect("quad face OBJ parse should succeed");
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.face_count(), 1);
    }

    #[test]
    fn obj_parse_out_of_range_index_fails() {
        let text = r#"
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 5
"#;
        let result = parse_obj(text);
        match result {
            Err(ObjError::IndexOutOfRange { idx, .. }) => assert_eq!(idx, 5),
            other => panic!("expected IndexOutOfRange error, got: {:?}", other),
        }
    }

    #[test]
    fn obj_save_load_file_roundtrip() {
        let (verts, faces) = make_quad_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let path = std::env::temp_dir().join("halfedge_test_quad.obj");
        save_obj(&mesh, &path).unwrap();
        let loaded = load_obj(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.vertex_count(), mesh.vertex_count());
        assert_eq!(loaded.face_count(), mesh.face_count());
        assert_eq!(loaded.halfedge_count(), mesh.halfedge_count());
        assert!(check_topology(&loaded).is_ok());
    }

    #[test]
    fn obj_parse_empty_text() {
        let mesh = parse_obj("").expect("empty OBJ should parse as empty mesh");
        assert_eq!(mesh.vertex_count(), 0);
        assert_eq!(mesh.face_count(), 0);
    }

    #[test]
    fn obj_parse_only_vertices_no_faces() {
        let text = "v 0 0 0\nv 1 0 0\nv 0 1 0\n";
        let mesh = parse_obj(text).expect("vertices-only OBJ should parse");
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.face_count(), 0);
    }

    #[test]
    fn obj_parse_face_zero_index_fails() {
        let text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n";
        assert!(parse_obj(text).is_err());
    }

    #[test]
    fn obj_parse_face_index_equal_to_vertex_count_fails() {
        let text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n";
        assert!(parse_obj(text).is_err());
    }
}
