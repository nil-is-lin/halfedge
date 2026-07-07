//! OFF format parser/writer (ASCII).

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

use crate::ids::{FaceId, VertexId};
use crate::storage::MeshStorage;
use crate::traversal::FaceVertices;

use super::build_mesh_from_polygons;

// ============================================================
// Error type
// ============================================================

/// OFF parse/serialize error.
#[derive(Debug)]
pub enum OffError {
    Io(std::io::Error),
    Parse { line: usize, msg: String },
}

impl fmt::Display for OffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Parse { line, msg } => write!(f, "OFF parse error on line {line}: {msg}"),
        }
    }
}

impl std::error::Error for OffError {}

impl From<std::io::Error> for OffError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ============================================================
// OFF loading
// ============================================================

/// Load an OFF file (ASCII).
///
/// Format:
/// ```text
/// OFF
/// <vertex_count> <face_count> <edge_count>
/// x y z                # vertex 0
/// ...
/// k v0 v1 ... vk-1     # face 0
/// ...
/// ```
/// First line `OFF` keyword is optional (some files have `OFFN` or numbers).
/// Lines starting with `#` are comments. `edge_count` is usually 0, ignored.
pub fn load_off<P: AsRef<Path>>(path: P) -> Result<MeshStorage, OffError> {
    let text = fs::read_to_string(path)?;
    parse_off(&text)
}

/// Parse OFF text.
pub fn parse_off(text: &str) -> Result<MeshStorage, OffError> {
    let mut lines = text.lines().filter(|l| {
        let t = l.trim();
        !t.is_empty() && !t.starts_with('#')
    });

    // First line may contain OFF keyword
    let first = match lines.next() {
        Some(s) => s,
        None => {
            return Err(OffError::Parse {
                line: 1,
                msg: "OFF file is empty".into(),
            });
        }
    };
    let counts_line = if first.trim().starts_with("OFF") {
        // If first line has no digits after OFF, counts are on the next line
        let rest = first.trim().strip_prefix("OFF").unwrap_or(first).trim();
        if rest.is_empty() {
            lines.next().unwrap_or("")
        } else {
            rest
        }
    } else {
        first
    };

    let count_parts: Vec<&str> = counts_line.split_whitespace().collect();
    if count_parts.len() < 2 {
        return Err(OffError::Parse {
            line: 1,
            msg: "OFF header missing vertex/face count".into(),
        });
    }
    let v_count: usize = count_parts[0].parse().map_err(|_| OffError::Parse {
        line: 1,
        msg: format!("invalid vertex count: {}", count_parts[0]),
    })?;
    let f_count: usize = count_parts[1].parse().map_err(|_| OffError::Parse {
        line: 1,
        msg: format!("invalid face count: {}", count_parts[1]),
    })?;

    let mut vertices: Vec<[f64; 3]> = Vec::with_capacity(v_count);
    let mut faces: Vec<Vec<u32>> = Vec::with_capacity(f_count);
    let mut line_no: usize;

    for (i, raw) in lines.enumerate() {
        line_no = i + 3;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if vertices.len() < v_count {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 3 {
                return Err(OffError::Parse {
                    line: line_no,
                    msg: "vertex line has fewer than 3 coordinates".into(),
                });
            }
            let x: f64 = parts[0].parse().map_err(|_| OffError::Parse {
                line: line_no,
                msg: format!("invalid x coordinate: {}", parts[0]),
            })?;
            let y: f64 = parts[1].parse().map_err(|_| OffError::Parse {
                line: line_no,
                msg: format!("invalid y coordinate: {}", parts[1]),
            })?;
            let z: f64 = parts[2].parse().map_err(|_| OffError::Parse {
                line: line_no,
                msg: format!("invalid z coordinate: {}", parts[2]),
            })?;
            vertices.push([x, y, z]);
        } else if faces.len() < f_count {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }
            let k: usize = parts[0].parse().map_err(|_| OffError::Parse {
                line: line_no,
                msg: format!("invalid face vertex count: {}", parts[0]),
            })?;
            // Degenerate faces (k < 3) handled by build_mesh_from_polygons
            if parts.len() < k + 1 {
                return Err(OffError::Parse {
                    line: line_no,
                    msg: "face line has fewer indices than declared".into(),
                });
            }
            let mut idx = Vec::with_capacity(k);
            for j in 0..k {
                let v: u32 = parts[1 + j].parse().map_err(|_| OffError::Parse {
                    line: line_no,
                    msg: format!("invalid face index: {}", parts[1 + j]),
                })?;
                if v as usize >= v_count {
                    return Err(OffError::Parse {
                        line: line_no,
                        msg: format!("face index {v} out of range (vertex count {v_count})"),
                    });
                }
                idx.push(v);
            }
            faces.push(idx);
        }
    }

    Ok(build_mesh_from_polygons(&vertices, &faces).expect("indices already validated"))
}

// ============================================================
// OFF saving
// ============================================================

/// Serialize mesh to OFF text.
pub fn format_off(mesh: &MeshStorage) -> String {
    let v_ids: Vec<VertexId> = mesh.vertex_ids().collect();
    let f_ids: Vec<FaceId> = mesh.face_ids().collect();
    let mut index_map: HashMap<VertexId, usize> = HashMap::new();
    for (i, &v) in v_ids.iter().enumerate() {
        index_map.insert(v, i);
    }

    let mut out = String::with_capacity(v_ids.len() * 32 + f_ids.len() * 16);
    out.push_str("OFF\n");
    out.push_str(&format!("{} {} 0\n", v_ids.len(), f_ids.len()));
    for &v in &v_ids {
        let p = mesh.get_vertex(v).expect("vertex exists in mesh").position;
        out.push_str(&format!("{:.6} {:.6} {:.6}\n", p[0], p[1], p[2]));
    }
    let mut skipped: u32 = 0;
    for f in &f_ids {
        let verts: Vec<usize> = FaceVertices::new(mesh, *f)
            .filter_map(|v| index_map.get(&v).copied())
            .collect();
        if verts.len() < 3 {
            skipped += 1;
            continue;
        }
        out.push_str(&verts.len().to_string());
        for vi in &verts {
            out.push(' ');
            out.push_str(&vi.to_string());
        }
        out.push('\n');
    }
    if skipped > 0 {
        log::warn!(
            "[halfedge::format_off] warning: skipped {skipped} degenerate face(s) (vertex count < 3)"
        );
    }
    out
}

/// Save mesh to an OFF file.
pub fn save_off<P: AsRef<Path>>(mesh: &MeshStorage, path: P) -> Result<(), OffError> {
    let text = format_off(mesh);
    fs::write(path, text)?;
    Ok(())
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_mesh_from_vertices_and_faces;
    use crate::validate::check_topology;

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
    fn off_roundtrip_tetrahedron() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let text = format_off(&mesh);
        let parsed = parse_off(&text).expect("OFF roundtrip parse failed");
        assert_eq!(parsed.vertex_count(), mesh.vertex_count());
        assert_eq!(parsed.face_count(), mesh.face_count());
        assert!(check_topology(&parsed).is_ok());
    }

    #[test]
    fn off_file_roundtrip() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let path = std::env::temp_dir().join("halfedge_off.off");
        save_off(&mesh, &path).expect("failed to save OFF");
        let loaded = load_off(&path).expect("failed to load OFF");
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.vertex_count(), mesh.vertex_count());
        assert_eq!(loaded.face_count(), mesh.face_count());
    }

    #[test]
    fn off_parse_counts_inline_with_keyword() {
        let text = "OFF 4 4 0\n\
            0 0 0\n1 0 0\n0 1 0\n0 0 1\n\
            3 0 2 1\n3 0 1 3\n3 0 3 2\n3 1 2 3\n";
        let mesh = parse_off(text).expect("OFF inline counts parse failed");
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.face_count(), 4);
    }

    #[test]
    fn off_parse_quadrilateral_face_succeeds() {
        let text = "OFF\n4 1 0\n\
            0 0 0\n1 0 0\n1 1 0\n0 1 0\n\
            4 0 1 2 3\n";
        let mesh = parse_off(text).expect("OFF quad face parse failed");
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.face_count(), 1);
    }

    #[test]
    fn off_parse_out_of_range_index_fails() {
        let text = "OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 5\n";
        let err = parse_off(text).unwrap_err();
        assert!(matches!(err, OffError::Parse { .. }));
    }

    #[test]
    fn off_parse_empty_text_fails() {
        assert!(parse_off("").is_err());
    }

    #[test]
    fn off_parse_only_vertices_no_faces() {
        let text = "OFF\n3 0 0\n0 0 0\n1 0 0\n0 1 0\n";
        let mesh = parse_off(text).expect("vertices-only OFF should parse");
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.face_count(), 0);
    }
}
