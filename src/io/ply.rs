//! PLY format parser/writer (ASCII + binary little-endian).

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

/// PLY parse/serialize error.
#[derive(Debug)]
pub enum PlyError {
    Io(std::io::Error),
    Parse { line: usize, msg: String },
    Unsupported(String),
}

impl fmt::Display for PlyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Parse { line, msg } => write!(f, "PLY parse error on line {line}: {msg}"),
            Self::Unsupported(s) => write!(f, "unsupported PLY feature: {s}"),
        }
    }
}

impl std::error::Error for PlyError {}

impl From<std::io::Error> for PlyError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ============================================================
// PLY internal types
// ============================================================

/// PLY data format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyFormat {
    Ascii,
    BinaryLittleEndian,
    #[allow(dead_code)]
    BinaryBigEndian,
}

/// PLY scalar type and byte width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyType {
    Char,
    UChar,
    Short,
    UShort,
    Int,
    UInt,
    Float,
    Double,
}

impl PlyType {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "char" | "int8" => Some(Self::Char),
            "uchar" | "uint8" => Some(Self::UChar),
            "short" | "int16" => Some(Self::Short),
            "ushort" | "uint16" => Some(Self::UShort),
            "int" | "int32" => Some(Self::Int),
            "uint" | "uint32" => Some(Self::UInt),
            "float" | "float32" => Some(Self::Float),
            "double" | "float64" => Some(Self::Double),
            _ => None,
        }
    }

    fn size(&self) -> usize {
        match self {
            Self::Char | Self::UChar => 1,
            Self::Short | Self::UShort => 2,
            Self::Int | Self::UInt | Self::Float => 4,
            Self::Double => 8,
        }
    }

    /// Read this type as little-endian bytes and return f64 (for vertex coordinates).
    fn read_le_as_f64(&self, bytes: &[u8]) -> Result<f64, PlyError> {
        match self {
            Self::Char => bytes
                .first()
                .map(|b| (*b as i8) as f64)
                .ok_or_else(|| PlyError::Unsupported("byte stream ended unexpectedly".into())),
            Self::UChar => bytes
                .first()
                .map(|b| *b as f64)
                .ok_or_else(|| PlyError::Unsupported("byte stream ended unexpectedly".into())),
            Self::Short => bytes
                .get(0..2)
                .and_then(|s| s.try_into().ok())
                .map(|arr: [u8; 2]| i16::from_le_bytes(arr) as f64)
                .ok_or_else(|| PlyError::Unsupported("byte stream ended unexpectedly".into())),
            Self::UShort => bytes
                .get(0..2)
                .and_then(|s| s.try_into().ok())
                .map(|arr: [u8; 2]| u16::from_le_bytes(arr) as f64)
                .ok_or_else(|| PlyError::Unsupported("byte stream ended unexpectedly".into())),
            Self::Int => bytes
                .get(0..4)
                .and_then(|s| s.try_into().ok())
                .map(|arr: [u8; 4]| i32::from_le_bytes(arr) as f64)
                .ok_or_else(|| PlyError::Unsupported("byte stream ended unexpectedly".into())),
            Self::UInt => bytes
                .get(0..4)
                .and_then(|s| s.try_into().ok())
                .map(|arr: [u8; 4]| u32::from_le_bytes(arr) as f64)
                .ok_or_else(|| PlyError::Unsupported("byte stream ended unexpectedly".into())),
            Self::Float => bytes
                .get(0..4)
                .and_then(|s| s.try_into().ok())
                .map(|arr: [u8; 4]| f32::from_le_bytes(arr) as f64)
                .ok_or_else(|| PlyError::Unsupported("byte stream ended unexpectedly".into())),
            Self::Double => bytes
                .get(0..8)
                .and_then(|s| s.try_into().ok())
                .map(|arr: [u8; 8]| f64::from_le_bytes(arr))
                .ok_or_else(|| PlyError::Unsupported("byte stream ended unexpectedly".into())),
        }
    }

    /// Read this type as little-endian bytes and return u32 (for face indices).
    fn read_le_as_u32(&self, bytes: &[u8]) -> Result<u32, PlyError> {
        match self {
            Self::Char => bytes
                .first()
                .map(|b| (*b as i8) as u32)
                .ok_or_else(|| PlyError::Unsupported("byte stream ended unexpectedly".into())),
            Self::UChar => bytes
                .first()
                .map(|b| *b as u32)
                .ok_or_else(|| PlyError::Unsupported("byte stream ended unexpectedly".into())),
            Self::Short => bytes
                .get(0..2)
                .and_then(|s| s.try_into().ok())
                .map(|arr: [u8; 2]| i16::from_le_bytes(arr) as u32)
                .ok_or_else(|| PlyError::Unsupported("byte stream ended unexpectedly".into())),
            Self::UShort => bytes
                .get(0..2)
                .and_then(|s| s.try_into().ok())
                .map(|arr: [u8; 2]| u16::from_le_bytes(arr) as u32)
                .ok_or_else(|| PlyError::Unsupported("byte stream ended unexpectedly".into())),
            Self::Int => bytes
                .get(0..4)
                .and_then(|s| s.try_into().ok())
                .map(|arr: [u8; 4]| i32::from_le_bytes(arr) as u32)
                .ok_or_else(|| PlyError::Unsupported("byte stream ended unexpectedly".into())),
            Self::UInt => bytes
                .get(0..4)
                .and_then(|s| s.try_into().ok())
                .map(|arr: [u8; 4]| u32::from_le_bytes(arr))
                .ok_or_else(|| PlyError::Unsupported("byte stream ended unexpectedly".into())),
            Self::Float | Self::Double => Err(PlyError::Unsupported(
                "float type used as index type".into(),
            )),
        }
    }

    /// Write f64 as this type in little-endian bytes.
    fn write_le_from_f64(&self, v: f64) -> Vec<u8> {
        match self {
            Self::Char => (v as i8).to_le_bytes().to_vec(),
            Self::UChar => (v as u8).to_le_bytes().to_vec(),
            Self::Short => (v as i16).to_le_bytes().to_vec(),
            Self::UShort => (v as u16).to_le_bytes().to_vec(),
            Self::Int => (v as i32).to_le_bytes().to_vec(),
            Self::UInt => (v as u32).to_le_bytes().to_vec(),
            Self::Float => (v as f32).to_le_bytes().to_vec(),
            Self::Double => v.to_le_bytes().to_vec(),
        }
    }

    /// Write u32 as this type in little-endian bytes.
    fn write_le_from_u32(&self, v: u32) -> Vec<u8> {
        match self {
            Self::Char => (v as i8).to_le_bytes().to_vec(),
            Self::UChar => (v as u8).to_le_bytes().to_vec(),
            Self::Short => (v as i16).to_le_bytes().to_vec(),
            Self::UShort => (v as u16).to_le_bytes().to_vec(),
            Self::Int => (v as i32).to_le_bytes().to_vec(),
            Self::UInt => v.to_le_bytes().to_vec(),
            Self::Float | Self::Double => (v as f32).to_le_bytes().to_vec(),
        }
    }
}

/// PLY header parse result.
struct PlyHeader {
    format: PlyFormat,
    vertex_count: usize,
    /// Vertex attributes (name, type), in order of appearance.
    vertex_props: Vec<(String, PlyType)>,
    face_count: usize,
    /// Face index list (count_type, index_type), None means no faces.
    face_list: Option<(PlyType, PlyType)>,
}

/// Parse PLY header text portion. Returns header and binary start offset.
fn parse_ply_header(text_or_bytes: &[u8]) -> Result<(PlyHeader, usize), PlyError> {
    let needle = b"end_header\n";
    let end_pos = text_or_bytes
        .windows(needle.len())
        .position(|w| w == needle)
        .ok_or_else(|| PlyError::Parse {
            line: 0,
            msg: "missing 'end_header' line".into(),
        })?;

    let header_str =
        std::str::from_utf8(&text_or_bytes[..end_pos]).map_err(|_| PlyError::Parse {
            line: 0,
            msg: "PLY header is not valid UTF-8".into(),
        })?;

    let mut format = PlyFormat::Ascii;
    let mut vertex_count: usize = 0;
    let mut face_count: usize = 0;
    let mut vertex_props: Vec<(String, PlyType)> = Vec::new();
    let mut face_list: Option<(PlyType, PlyType)> = None;
    let mut current_element: Option<String> = None;

    for (line_no, line) in header_str.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("comment") {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        match parts[0] {
            "ply" | "end_header" => {}
            "format" => {
                if parts.len() < 2 {
                    return Err(PlyError::Parse {
                        line: line_no + 1,
                        msg: "format line missing type".into(),
                    });
                }
                format = match parts[1] {
                    "ascii" => PlyFormat::Ascii,
                    "binary_little_endian" => PlyFormat::BinaryLittleEndian,
                    "binary_big_endian" => PlyFormat::BinaryBigEndian,
                    other => {
                        return Err(PlyError::Unsupported(format!(
                            "unknown PLY format: {other}"
                        )));
                    }
                };
            }
            "element" => {
                if parts.len() < 3 {
                    return Err(PlyError::Parse {
                        line: line_no + 1,
                        msg: "element line missing name or count".into(),
                    });
                }
                let name = parts[1].to_string();
                let count: usize = parts[2].parse().map_err(|_| PlyError::Parse {
                    line: line_no + 1,
                    msg: format!("invalid element count: {}", parts[2]),
                })?;
                match name.as_str() {
                    "vertex" => {
                        vertex_count = count;
                        current_element = Some("vertex".into());
                    }
                    "face" => {
                        face_count = count;
                        current_element = Some("face".into());
                    }
                    _ => {
                        current_element = Some(name);
                    }
                }
            }
            "property" => {
                let elem = match &current_element {
                    Some(e) => e.as_str(),
                    None => continue,
                };
                if elem == "vertex" {
                    if parts.len() < 3 {
                        return Err(PlyError::Parse {
                            line: line_no + 1,
                            msg: "vertex property line too short".into(),
                        });
                    }
                    let ty = PlyType::from_str(parts[1]).ok_or_else(|| {
                        PlyError::Unsupported(format!("unknown type: {}", parts[1]))
                    })?;
                    let name = parts[2].to_string();
                    vertex_props.push((name, ty));
                } else if elem == "face" {
                    // property list <count_type> <index_type> <name>
                    if parts.len() >= 5 && parts[1] == "list" {
                        let ct = PlyType::from_str(parts[2]).ok_or_else(|| {
                            PlyError::Unsupported(format!("unknown type: {}", parts[2]))
                        })?;
                        let it = PlyType::from_str(parts[3]).ok_or_else(|| {
                            PlyError::Unsupported(format!("unknown type: {}", parts[3]))
                        })?;
                        face_list = Some((ct, it));
                    }
                }
            }
            _ => {}
        }
    }

    let bin_offset = end_pos + needle.len();
    Ok((
        PlyHeader {
            format,
            vertex_count,
            vertex_props,
            face_count,
            face_list,
        },
        bin_offset,
    ))
}

// ============================================================
// PLY loading
// ============================================================

/// Load a PLY file (auto-detect ASCII / binary).
pub fn load_ply<P: AsRef<Path>>(path: P) -> Result<MeshStorage, PlyError> {
    let bytes = fs::read(path)?;
    parse_ply_bytes(&bytes)
}

/// Parse PLY byte stream (auto-detect ASCII / binary).
pub fn parse_ply_bytes(bytes: &[u8]) -> Result<MeshStorage, PlyError> {
    let (header, bin_offset) = parse_ply_header(bytes)?;
    match header.format {
        PlyFormat::Ascii => {
            let text = std::str::from_utf8(bytes).map_err(|_| PlyError::Parse {
                line: 0,
                msg: "PLY ASCII file is not valid UTF-8".into(),
            })?;
            parse_ply_ascii_with_header(text, &header)
        }
        PlyFormat::BinaryLittleEndian => parse_ply_binary_le(&bytes[bin_offset..], &header),
        PlyFormat::BinaryBigEndian => {
            Err(PlyError::Unsupported("big-endian PLY not supported".into()))
        }
    }
}

/// Parse PLY ASCII text (legacy entry point, delegates internally to `parse_ply_bytes`).
pub fn parse_ply(text: &str) -> Result<MeshStorage, PlyError> {
    parse_ply_bytes(text.as_bytes())
}

// ============================================================
// PLY saving
// ============================================================

/// Serialize mesh to PLY ASCII text.
pub fn format_ply(mesh: &MeshStorage) -> String {
    let v_ids: Vec<VertexId> = mesh.vertex_ids().collect();
    let f_ids: Vec<FaceId> = mesh.face_ids().collect();
    let mut index_map: HashMap<VertexId, usize> = HashMap::new();
    for (i, &v) in v_ids.iter().enumerate() {
        index_map.insert(v, i);
    }

    let mut out = String::new();
    out.push_str("ply\n");
    out.push_str("format ascii 1.0\n");
    out.push_str(&format!("element vertex {}\n", v_ids.len()));
    out.push_str("property float x\n");
    out.push_str("property float y\n");
    out.push_str("property float z\n");
    out.push_str(&format!("element face {}\n", f_ids.len()));
    out.push_str("property list uchar int vertex_indices\n");
    out.push_str("end_header\n");

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
            "[halfedge::format_ply] warning: skipped {skipped} degenerate face(s) (vertex count < 3)"
        );
    }
    out
}

/// Save mesh to a PLY file (ASCII format).
pub fn save_ply<P: AsRef<Path>>(mesh: &MeshStorage, path: P) -> Result<(), PlyError> {
    let text = format_ply(mesh);
    fs::write(path, text)?;
    Ok(())
}

/// Serialize mesh to binary PLY byte stream (little-endian, float vertices + uchar/int face indices).
pub fn format_ply_binary(mesh: &MeshStorage) -> Vec<u8> {
    let v_ids: Vec<VertexId> = mesh.vertex_ids().collect();
    let f_ids: Vec<FaceId> = mesh.face_ids().collect();
    let mut index_map: HashMap<VertexId, usize> = HashMap::new();
    for (i, &v) in v_ids.iter().enumerate() {
        index_map.insert(v, i);
    }

    let mut header = String::new();
    header.push_str("ply\n");
    header.push_str("format binary_little_endian 1.0\n");
    header.push_str(&format!("element vertex {}\n", v_ids.len()));
    header.push_str("property float x\n");
    header.push_str("property float y\n");
    header.push_str("property float z\n");
    header.push_str(&format!("element face {}\n", f_ids.len()));
    header.push_str("property list uchar int vertex_indices\n");
    header.push_str("end_header\n");

    let mut out: Vec<u8> = Vec::with_capacity(header.len() + v_ids.len() * 12 + f_ids.len() * 16);
    out.extend_from_slice(header.as_bytes());

    let float_ty = PlyType::Float;
    for &v in &v_ids {
        let p = mesh.get_vertex(v).expect("vertex exists in mesh").position;
        for c in &p {
            out.extend(float_ty.write_le_from_f64(*c));
        }
    }
    let uchar_ty = PlyType::UChar;
    let int_ty = PlyType::Int;
    let mut skipped: u32 = 0;
    for f in &f_ids {
        let verts: Vec<usize> = FaceVertices::new(mesh, *f)
            .filter_map(|v| index_map.get(&v).copied())
            .collect();
        if verts.len() < 3 {
            skipped += 1;
            continue;
        }
        out.extend(uchar_ty.write_le_from_u32(verts.len() as u32));
        for vi in &verts {
            out.extend(int_ty.write_le_from_u32(*vi as u32));
        }
    }
    if skipped > 0 {
        log::warn!(
            "[halfedge::format_ply_binary] warning: skipped {skipped} degenerate face(s) (vertex count < 3)"
        );
    }
    out
}

/// Save mesh to a binary PLY file.
pub fn save_ply_binary<P: AsRef<Path>>(mesh: &MeshStorage, path: P) -> Result<(), PlyError> {
    let bytes = format_ply_binary(mesh);
    fs::write(path, bytes)?;
    Ok(())
}

// ============================================================
// PLY internal parsing helpers
// ============================================================

/// Parse PLY ASCII text using a pre-parsed header. Skips header lines and reads data directly.
fn parse_ply_ascii_with_header(text: &str, header: &PlyHeader) -> Result<MeshStorage, PlyError> {
    let mut end_line: Option<usize> = None;
    for (i, line) in text.lines().enumerate() {
        if line.trim() == "end_header" {
            end_line = Some(i);
            break;
        }
    }
    let start_line = end_line.ok_or_else(|| PlyError::Parse {
        line: 0,
        msg: "missing 'end_header' line".into(),
    })? + 1;

    // Find x/y/z property indices (attributes may be more than 3)
    let x_idx = header
        .vertex_props
        .iter()
        .position(|(n, _)| n == "x")
        .unwrap_or(0);
    let y_idx = header
        .vertex_props
        .iter()
        .position(|(n, _)| n == "y")
        .unwrap_or(1);
    let z_idx = header
        .vertex_props
        .iter()
        .position(|(n, _)| n == "z")
        .unwrap_or(2);

    let mut vertices: Vec<[f64; 3]> = Vec::with_capacity(header.vertex_count);
    let mut faces: Vec<Vec<u32>> = Vec::with_capacity(header.face_count);

    for (i, raw) in text.lines().enumerate().skip(start_line) {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if vertices.len() < header.vertex_count {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 3 {
                return Err(PlyError::Parse {
                    line: i + 1,
                    msg: "vertex line has fewer than 3 fields".into(),
                });
            }
            let parse = |s: &str| -> Result<f64, PlyError> {
                s.parse::<f64>().map_err(|_| PlyError::Parse {
                    line: i + 1,
                    msg: format!("invalid vertex coordinate: {s}"),
                })
            };
            let x = parse(parts[x_idx])?;
            let y = parse(parts[y_idx])?;
            let z = parse(parts[z_idx])?;
            vertices.push([x, y, z]);
        } else if header.face_count > 0 {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }
            // First token is face vertex count, rest are indices
            let count: usize = parts[0].parse().map_err(|_| PlyError::Parse {
                line: i + 1,
                msg: format!("invalid face vertex count: {}", parts[0]),
            })?;
            if parts.len() < count + 1 {
                return Err(PlyError::Parse {
                    line: i + 1,
                    msg: "face line has fewer indices than declared".into(),
                });
            }
            let mut indices: Vec<u32> = Vec::with_capacity(count);
            for k in 0..count {
                let idx: u32 = parts[1 + k].parse().map_err(|_| PlyError::Parse {
                    line: i + 1,
                    msg: format!("invalid face index: {}", parts[1 + k]),
                })?;
                indices.push(idx);
            }
            // Degenerate faces (< 3 indices) handled by build_mesh_from_polygons
            faces.push(indices);
        }
    }

    Ok(build_mesh_from_polygons(&vertices, &faces).expect("indices already validated"))
}

/// Parse PLY binary little-endian data (bytes after end_header).
fn parse_ply_binary_le(data: &[u8], header: &PlyHeader) -> Result<MeshStorage, PlyError> {
    let mut vertices: Vec<[f64; 3]> = Vec::with_capacity(header.vertex_count);
    let mut faces: Vec<Vec<u32>> = Vec::with_capacity(header.face_count);

    // Find x/y/z attributes
    let x_idx = header
        .vertex_props
        .iter()
        .position(|(n, _)| n == "x")
        .unwrap_or(0);
    let y_idx = header
        .vertex_props
        .iter()
        .position(|(n, _)| n == "y")
        .unwrap_or(1);
    let z_idx = header
        .vertex_props
        .iter()
        .position(|(n, _)| n == "z")
        .unwrap_or(2);

    // Compute per-vertex byte size
    let vertex_stride: usize = header.vertex_props.iter().map(|(_, t)| t.size()).sum();
    if vertex_stride == 0 && header.vertex_count > 0 {
        return Err(PlyError::Unsupported("vertex has no properties".into()));
    }

    let mut offset = 0usize;
    for _ in 0..header.vertex_count {
        if offset + vertex_stride > data.len() {
            return Err(PlyError::Parse {
                line: 0,
                msg: "binary vertex data ended unexpectedly".into(),
            });
        }
        let mut field_offsets: Vec<usize> = Vec::with_capacity(header.vertex_props.len());
        let mut cur = offset;
        for (_, ty) in &header.vertex_props {
            field_offsets.push(cur);
            cur += ty.size();
        }
        let x = header.vertex_props[x_idx]
            .1
            .read_le_as_f64(&data[field_offsets[x_idx]..])?;
        let y = header.vertex_props[y_idx]
            .1
            .read_le_as_f64(&data[field_offsets[y_idx]..])?;
        let z = header.vertex_props[z_idx]
            .1
            .read_le_as_f64(&data[field_offsets[z_idx]..])?;
        vertices.push([x, y, z]);
        offset += vertex_stride;
    }

    // Read faces
    if let Some((ct, it)) = header.face_list {
        for _ in 0..header.face_count {
            if offset + ct.size() > data.len() {
                return Err(PlyError::Parse {
                    line: 0,
                    msg: "binary face count ended unexpectedly".into(),
                });
            }
            let count = ct.read_le_as_u32(&data[offset..])? as usize;
            offset += ct.size();
            if offset + it.size() * count > data.len() {
                return Err(PlyError::Parse {
                    line: 0,
                    msg: "binary face indices ended unexpectedly".into(),
                });
            }
            let mut indices: Vec<u32> = Vec::with_capacity(count);
            for _ in 0..count {
                let idx = it.read_le_as_u32(&data[offset..])?;
                offset += it.size();
                indices.push(idx);
            }
            // Degenerate faces (< 3 indices) handled by build_mesh_from_polygons
            faces.push(indices);
        }
    }

    Ok(build_mesh_from_polygons(&vertices, &faces).expect("indices already validated"))
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
    fn ply_binary_roundtrip() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let bytes = format_ply_binary(&mesh);
        let parsed = parse_ply_bytes(&bytes).expect("PLY binary roundtrip parse failed");
        assert_eq!(parsed.vertex_count(), mesh.vertex_count());
        assert_eq!(parsed.face_count(), mesh.face_count());
        assert!(check_topology(&parsed).is_ok());
    }

    #[test]
    fn ply_binary_file_roundtrip() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let path = std::env::temp_dir().join("halfedge_ply_bin.ply");
        save_ply_binary(&mesh, &path).expect("failed to save binary PLY");
        let loaded = load_ply(&path).expect("failed to load binary PLY");
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.vertex_count(), mesh.vertex_count());
        assert_eq!(loaded.face_count(), mesh.face_count());
    }

    #[test]
    fn ply_ascii_still_works_via_parse_ply_bytes() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let text = format_ply(&mesh);
        let bytes = text.into_bytes();
        let parsed = parse_ply_bytes(&bytes).expect("PLY ASCII via bytes parse failed");
        assert_eq!(parsed.vertex_count(), mesh.vertex_count());
        assert_eq!(parsed.face_count(), mesh.face_count());
    }

    #[test]
    fn ply_binary_detects_bad_header() {
        let bytes = b"ply\nformat binary_little_endian 1.0\nelement vertex 0\n";
        let err = parse_ply_bytes(bytes).unwrap_err();
        assert!(matches!(err, PlyError::Parse { .. }));
    }

    #[test]
    fn ply_parse_empty_bytes_fails() {
        assert!(parse_ply_bytes(b"").is_err());
    }

    #[test]
    fn ply_parse_missing_end_header_fails() {
        let bytes = b"ply\nformat ascii 1.0\n";
        assert!(parse_ply_bytes(bytes).is_err());
    }
}
