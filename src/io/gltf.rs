//! glTF/GLB format parser/writer (minimal subset: single mesh primitive, POSITION + indices).

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

use crate::ids::VertexId;
use crate::storage::MeshStorage;
use crate::traversal::FaceVertices;

use super::build_mesh_from_vertices_and_faces;

// ============================================================
// Error type
// ============================================================

/// glTF parse/serialize error.
#[derive(Debug)]
pub enum GltfError {
    Io(std::io::Error),
    Parse(String),
    Unsupported(String),
}

impl fmt::Display for GltfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Parse(s) => write!(f, "glTF parse error: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported glTF feature: {s}"),
        }
    }
}

impl std::error::Error for GltfError {}

impl From<std::io::Error> for GltfError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ============================================================
// GLB constants
// ============================================================

const GLB_MAGIC: u32 = 0x4654_4C47; // "glTF"
const GLB_VERSION: u32 = 2;
const JSON_CHUNK_TYPE: u32 = 0x4E4F_534A; // "JSON"
const BIN_CHUNK_TYPE: u32 = 0x004E_4942; // "BIN\0"

// ============================================================
// Minimal JSON parser (supports only object/array/primitives needed for GLB)
// ============================================================

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    fn as_object(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            JsonValue::Object(v) => Some(v.as_slice()),
            _ => None,
        }
    }
    fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(v) => Some(v.as_slice()),
            _ => None,
        }
    }
    fn as_u32(&self) -> Option<u32> {
        match self {
            JsonValue::Number(n) if *n >= 0.0 => Some(*n as u32),
            _ => None,
        }
    }
    #[allow(dead_code)]
    fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
    fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object()
            .and_then(|o| o.iter().find(|(k, _)| k == key).map(|(_, v)| v))
    }
}

fn parse_minimal_json(text: &str) -> Result<JsonValue, GltfError> {
    let bytes = text.as_bytes();
    let mut pos = 0;
    skip_ws(bytes, &mut pos);
    let v = parse_value(bytes, &mut pos)?;
    Ok(v)
}

fn skip_ws(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() {
        match bytes[*pos] {
            b' ' | b'\t' | b'\n' | b'\r' => *pos += 1,
            _ => break,
        }
    }
}

fn parse_value(bytes: &[u8], pos: &mut usize) -> Result<JsonValue, GltfError> {
    skip_ws(bytes, pos);
    if *pos >= bytes.len() {
        return Err(GltfError::Parse("JSON ended unexpectedly".into()));
    }
    match bytes[*pos] {
        b'{' => parse_object(bytes, pos),
        b'[' => parse_array(bytes, pos),
        b'"' => parse_string(bytes, pos).map(JsonValue::String),
        b't' | b'f' => parse_bool(bytes, pos).map(JsonValue::Bool),
        b'n' => parse_null(bytes, pos).map(|_| JsonValue::Null),
        b'-' | b'0'..=b'9' => parse_number(bytes, pos).map(JsonValue::Number),
        c => Err(GltfError::Parse(format!(
            "unexpected JSON character '{}' at position {}",
            c as char, pos
        ))),
    }
}

fn parse_object(bytes: &[u8], pos: &mut usize) -> Result<JsonValue, GltfError> {
    *pos += 1; // skip '{'
    let mut entries: Vec<(String, JsonValue)> = Vec::new();
    skip_ws(bytes, pos);
    if *pos < bytes.len() && bytes[*pos] == b'}' {
        *pos += 1;
        return Ok(JsonValue::Object(entries));
    }
    loop {
        skip_ws(bytes, pos);
        if *pos >= bytes.len() || bytes[*pos] != b'"' {
            return Err(GltfError::Parse("expected string key in object".into()));
        }
        let key = parse_string(bytes, pos)?;
        skip_ws(bytes, pos);
        if *pos >= bytes.len() || bytes[*pos] != b':' {
            return Err(GltfError::Parse("expected ':' after key".into()));
        }
        *pos += 1;
        let val = parse_value(bytes, pos)?;
        entries.push((key, val));
        skip_ws(bytes, pos);
        if *pos >= bytes.len() {
            return Err(GltfError::Parse("object ended unexpectedly".into()));
        }
        match bytes[*pos] {
            b',' => {
                *pos += 1;
                continue;
            }
            b'}' => {
                *pos += 1;
                return Ok(JsonValue::Object(entries));
            }
            c => {
                return Err(GltfError::Parse(format!(
                    "expected ',' or '}}' in object, got '{}'",
                    c as char
                )));
            }
        }
    }
}

fn parse_array(bytes: &[u8], pos: &mut usize) -> Result<JsonValue, GltfError> {
    *pos += 1; // skip '['
    let mut items: Vec<JsonValue> = Vec::new();
    skip_ws(bytes, pos);
    if *pos < bytes.len() && bytes[*pos] == b']' {
        *pos += 1;
        return Ok(JsonValue::Array(items));
    }
    loop {
        let val = parse_value(bytes, pos)?;
        items.push(val);
        skip_ws(bytes, pos);
        if *pos >= bytes.len() {
            return Err(GltfError::Parse("array ended unexpectedly".into()));
        }
        match bytes[*pos] {
            b',' => {
                *pos += 1;
                continue;
            }
            b']' => {
                *pos += 1;
                return Ok(JsonValue::Array(items));
            }
            c => {
                return Err(GltfError::Parse(format!(
                    "expected ',' or ']' in array, got '{}'",
                    c as char
                )));
            }
        }
    }
}

fn parse_string(bytes: &[u8], pos: &mut usize) -> Result<String, GltfError> {
    if *pos >= bytes.len() || bytes[*pos] != b'"' {
        return Err(GltfError::Parse("expected '\"'".into()));
    }
    *pos += 1;
    let mut s = String::new();
    while *pos < bytes.len() {
        let c = bytes[*pos];
        *pos += 1;
        match c {
            b'"' => return Ok(s),
            b'\\' => {
                if *pos >= bytes.len() {
                    return Err(GltfError::Parse("incomplete escape sequence".into()));
                }
                let esc = bytes[*pos];
                *pos += 1;
                match esc {
                    b'"' => s.push('"'),
                    b'\\' => s.push('\\'),
                    b'/' => s.push('/'),
                    b'n' => s.push('\n'),
                    b't' => s.push('\t'),
                    b'r' => s.push('\r'),
                    b'b' => s.push('\u{08}'),
                    b'f' => s.push('\u{0C}'),
                    _ => s.push(esc as char),
                }
            }
            _ => s.push(c as char),
        }
    }
    Err(GltfError::Parse("unterminated string".into()))
}

fn parse_bool(bytes: &[u8], pos: &mut usize) -> Result<bool, GltfError> {
    if bytes[*pos..].starts_with(b"true") {
        *pos += 4;
        Ok(true)
    } else if bytes[*pos..].starts_with(b"false") {
        *pos += 5;
        Ok(false)
    } else {
        Err(GltfError::Parse("invalid bool value".into()))
    }
}

fn parse_null(bytes: &[u8], pos: &mut usize) -> Result<(), GltfError> {
    if bytes[*pos..].starts_with(b"null") {
        *pos += 4;
        Ok(())
    } else {
        Err(GltfError::Parse("invalid null value".into()))
    }
}

fn parse_number(bytes: &[u8], pos: &mut usize) -> Result<f64, GltfError> {
    let start = *pos;
    if *pos < bytes.len() && bytes[*pos] == b'-' {
        *pos += 1;
    }
    while *pos < bytes.len() {
        match bytes[*pos] {
            b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-' => *pos += 1,
            _ => break,
        }
    }
    let s = std::str::from_utf8(&bytes[start..*pos])
        .map_err(|_| GltfError::Parse("invalid number".into()))?;
    s.parse::<f64>()
        .map_err(|_| GltfError::Parse(format!("invalid number: {s}")))
}

// ============================================================
// GLB document structure
// ============================================================

#[derive(Debug)]
struct GltfAccessor {
    buffer_view: u32,
    component_type: u32, // 5120 BYTE / 5121 UBYTE / 5122 SHORT / 5123 USHORT / 5125 UINT / 5126 FLOAT
    count: u32,
    byte_offset: u32,
}

#[derive(Debug)]
struct GltfBufferView {
    #[allow(dead_code)]
    buffer: u32,
    byte_offset: u32,
    #[allow(dead_code)]
    byte_length: u32,
}

#[derive(Debug)]
struct GltfPrimitive {
    attributes: HashMap<String, u32>,
    indices: Option<u32>,
    mode: u32, // 4 = TRIANGLES
}

#[derive(Debug)]
struct GltfMesh {
    primitives: Vec<GltfPrimitive>,
}

#[derive(Debug)]
struct GltfDoc {
    buffer_views: Vec<GltfBufferView>,
    accessors: Vec<GltfAccessor>,
    meshes: Vec<GltfMesh>,
}

impl GltfDoc {
    fn from_json(json: &JsonValue) -> Result<Self, GltfError> {
        let bvs_json = json
            .get("bufferViews")
            .and_then(|v| v.as_array())
            .ok_or_else(|| GltfError::Parse("missing bufferViews".into()))?;
        let accs_json = json
            .get("accessors")
            .and_then(|v| v.as_array())
            .ok_or_else(|| GltfError::Parse("missing accessors".into()))?;
        let meshes_json = json
            .get("meshes")
            .and_then(|v| v.as_array())
            .ok_or_else(|| GltfError::Parse("missing meshes".into()))?;

        let mut buffer_views: Vec<GltfBufferView> = Vec::with_capacity(bvs_json.len());
        for bv in bvs_json {
            buffer_views.push(GltfBufferView {
                buffer: bv.get("buffer").and_then(|v| v.as_u32()).unwrap_or(0),
                byte_offset: bv.get("byteOffset").and_then(|v| v.as_u32()).unwrap_or(0),
                byte_length: bv
                    .get("byteLength")
                    .and_then(|v| v.as_u32())
                    .ok_or_else(|| GltfError::Parse("bufferView missing byteLength".into()))?,
            });
        }

        let mut accessors: Vec<GltfAccessor> = Vec::with_capacity(accs_json.len());
        for acc in accs_json {
            accessors.push(GltfAccessor {
                buffer_view: acc
                    .get("bufferView")
                    .and_then(|v| v.as_u32())
                    .ok_or_else(|| GltfError::Parse("accessor missing bufferView".into()))?,
                component_type: acc
                    .get("componentType")
                    .and_then(|v| v.as_u32())
                    .ok_or_else(|| GltfError::Parse("accessor missing componentType".into()))?,
                count: acc
                    .get("count")
                    .and_then(|v| v.as_u32())
                    .ok_or_else(|| GltfError::Parse("accessor missing count".into()))?,
                byte_offset: acc.get("byteOffset").and_then(|v| v.as_u32()).unwrap_or(0),
            });
        }

        let mut meshes: Vec<GltfMesh> = Vec::with_capacity(meshes_json.len());
        for m in meshes_json {
            let prims_json = m
                .get("primitives")
                .and_then(|v| v.as_array())
                .ok_or_else(|| GltfError::Parse("mesh missing primitives".into()))?;
            let mut primitives: Vec<GltfPrimitive> = Vec::with_capacity(prims_json.len());
            for p in prims_json {
                let mut attributes: HashMap<String, u32> = HashMap::new();
                if let Some(attrs) = p.get("attributes").and_then(|v| v.as_object()) {
                    for (k, v) in attrs {
                        if let Some(idx) = v.as_u32() {
                            attributes.insert(k.clone(), idx);
                        }
                    }
                }
                let indices = p.get("indices").and_then(|v| v.as_u32());
                let mode = p.get("mode").and_then(|v| v.as_u32()).unwrap_or(4);
                primitives.push(GltfPrimitive {
                    attributes,
                    indices,
                    mode,
                });
            }
            meshes.push(GltfMesh { primitives });
        }

        Ok(GltfDoc {
            buffer_views,
            accessors,
            meshes,
        })
    }
}

fn read_accessor_f32x3(
    doc: &GltfDoc,
    acc_idx: u32,
    bin: &[u8],
) -> Result<Vec<[f64; 3]>, GltfError> {
    let acc = doc
        .accessors
        .get(acc_idx as usize)
        .ok_or_else(|| GltfError::Parse(format!("accessor {acc_idx} out of bounds")))?;
    if acc.component_type != 5126 {
        return Err(GltfError::Unsupported(format!(
            "unsupported POSITION componentType {} (only 5126 FLOAT supported)",
            acc.component_type
        )));
    }
    let bv = doc
        .buffer_views
        .get(acc.buffer_view as usize)
        .ok_or_else(|| GltfError::Parse(format!("bufferView {} out of bounds", acc.buffer_view)))?;
    let start = (bv.byte_offset + acc.byte_offset) as usize;
    let end = start + (acc.count as usize) * 12;
    if end > bin.len() {
        return Err(GltfError::Parse("accessor exceeds bufferView range".into()));
    }
    let mut out: Vec<[f64; 3]> = Vec::with_capacity(acc.count as usize);
    let mut off = start;
    for _ in 0..acc.count {
        let x = f32::from_le_bytes([bin[off], bin[off + 1], bin[off + 2], bin[off + 3]]);
        let y = f32::from_le_bytes([bin[off + 4], bin[off + 5], bin[off + 6], bin[off + 7]]);
        let z = f32::from_le_bytes([bin[off + 8], bin[off + 9], bin[off + 10], bin[off + 11]]);
        out.push([x as f64, y as f64, z as f64]);
        off += 12;
    }
    Ok(out)
}

fn read_accessor_u32(doc: &GltfDoc, acc_idx: u32, bin: &[u8]) -> Result<Vec<u32>, GltfError> {
    let acc = doc
        .accessors
        .get(acc_idx as usize)
        .ok_or_else(|| GltfError::Parse(format!("accessor {acc_idx} out of bounds")))?;
    let elem_size: usize = match acc.component_type {
        5121 => 1, // UBYTE
        5123 => 2, // USHORT
        5125 => 4, // UINT
        other => {
            return Err(GltfError::Unsupported(format!(
                "unsupported indices componentType {other}"
            )));
        }
    };
    let bv = doc
        .buffer_views
        .get(acc.buffer_view as usize)
        .ok_or_else(|| GltfError::Parse(format!("bufferView {} out of bounds", acc.buffer_view)))?;
    let start = (bv.byte_offset + acc.byte_offset) as usize;
    let end = start + (acc.count as usize) * elem_size;
    if end > bin.len() {
        return Err(GltfError::Parse("accessor exceeds bufferView range".into()));
    }
    let mut out: Vec<u32> = Vec::with_capacity(acc.count as usize);
    let mut off = start;
    for _ in 0..acc.count {
        let v: u32 = match acc.component_type {
            5121 => bin[off] as u32,
            5123 => u16::from_le_bytes([bin[off], bin[off + 1]]) as u32,
            5125 => u32::from_le_bytes([bin[off], bin[off + 1], bin[off + 2], bin[off + 3]]),
            other => {
                return Err(GltfError::Unsupported(format!(
                    "accessor component_type {other} (only 5121/5123/5125 supported for indices)"
                )));
            }
        };
        out.push(v);
        off += elem_size;
    }
    Ok(out)
}

// ============================================================
// GLB loading
// ============================================================

/// Load a GLB file (glTF binary container).
///
/// Only supports a minimal subset: single mesh primitive with POSITION accessor and
/// index accessor. Does not support materials, textures, animations, skins, cameras,
/// or node hierarchies.
pub fn load_glb<P: AsRef<Path>>(path: P) -> Result<MeshStorage, GltfError> {
    let bytes = fs::read(path)?;
    parse_glb(&bytes)
}

/// Parse GLB byte stream.
pub fn parse_glb(bytes: &[u8]) -> Result<MeshStorage, GltfError> {
    if bytes.len() < 12 {
        return Err(GltfError::Parse("file too short for GLB header".into()));
    }
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != GLB_MAGIC {
        return Err(GltfError::Parse(format!(
            "invalid GLB magic: 0x{magic:08X}"
        )));
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != GLB_VERSION {
        return Err(GltfError::Unsupported(format!(
            "unsupported GLB version {version} (only 2 supported)"
        )));
    }
    let _total_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;

    // Parse chunks (each chunk: 4-byte length + 4-byte type + data, 4-byte aligned)
    let mut offset = 12usize;
    let mut json_chunk: Option<&[u8]> = None;
    let mut bin_chunk: Option<&[u8]> = None;

    while offset + 8 <= bytes.len() {
        let chunk_len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        let chunk_type = u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
        offset += 8;
        if offset + chunk_len > bytes.len() {
            return Err(GltfError::Parse("chunk length exceeds file size".into()));
        }
        let data = &bytes[offset..offset + chunk_len];
        offset += chunk_len;
        // 4-byte alignment padding
        while offset < bytes.len() && !offset.is_multiple_of(4) {
            offset += 1;
        }
        match chunk_type {
            JSON_CHUNK_TYPE => json_chunk = Some(data),
            BIN_CHUNK_TYPE => bin_chunk = Some(data),
            _ => {}
        }
    }

    let json_bytes = json_chunk.ok_or_else(|| GltfError::Parse("missing JSON chunk".into()))?;
    let json_str = std::str::from_utf8(json_bytes)
        .map_err(|_| GltfError::Parse("JSON chunk is not valid UTF-8".into()))?;
    let bin = bin_chunk.ok_or_else(|| GltfError::Parse("missing BIN chunk".into()))?;

    let json = parse_minimal_json(json_str)?;
    let gltf = GltfDoc::from_json(&json)?;

    // Find first mesh's first primitive
    let prim = gltf
        .meshes
        .first()
        .and_then(|m| m.primitives.first())
        .ok_or_else(|| GltfError::Parse("no mesh primitive found".into()))?;

    // Read POSITION accessor
    let pos_acc_idx = prim
        .attributes
        .get("POSITION")
        .copied()
        .ok_or_else(|| GltfError::Parse("primitive missing POSITION attribute".into()))?;
    let positions = read_accessor_f32x3(&gltf, pos_acc_idx, bin)?;

    // Read indices (optional)
    let indices: Vec<u32> = if let Some(idx_acc) = prim.indices {
        read_accessor_u32(&gltf, idx_acc, bin)?
    } else {
        // No indices: non-indexed draw, sequential 0..N
        (0..positions.len() as u32).collect()
    };

    // Convert to triangle faces: if mode != 4 (TRIANGLES) not supported
    if prim.mode != 4 {
        return Err(GltfError::Unsupported(format!(
            "unsupported primitive mode {} (only 4 = TRIANGLES supported)",
            prim.mode
        )));
    }

    if !indices.len().is_multiple_of(3) {
        return Err(GltfError::Parse(format!(
            "index count {} is not a multiple of 3",
            indices.len()
        )));
    }

    let mut faces: Vec<[u32; 3]> = Vec::with_capacity(indices.len() / 3);
    for tri in indices.chunks_exact(3) {
        faces.push([tri[0], tri[1], tri[2]]);
    }

    Ok(build_mesh_from_vertices_and_faces(&positions, &faces).expect("indices already validated"))
}

// ============================================================
// GLB saving
// ============================================================

/// Serialize mesh to GLB byte stream (minimal subset).
///
/// Output structure:
/// - 12-byte GLB header (magic, version=2, total_length)
/// - JSON chunk: describes buffers / bufferViews / accessors / meshes
/// - BIN chunk: position (float32 x 3N) + indices (uint32 x M)
///
/// Only generates single mesh / single primitive (mode=4 TRIANGLES), no materials.
/// Non-triangle faces are skipped.
pub fn format_glb(mesh: &MeshStorage) -> Vec<u8> {
    let v_ids: Vec<VertexId> = mesh.vertex_ids().collect();
    let mut index_map: HashMap<VertexId, u32> = HashMap::new();
    for (i, &v) in v_ids.iter().enumerate() {
        index_map.insert(v, i as u32);
    }

    // Build BIN data
    let mut bin: Vec<u8> = Vec::with_capacity(v_ids.len() * 12 + mesh.face_count() * 12);
    for &v in &v_ids {
        let p = mesh.get_vertex(v).expect("vertex exists in mesh").position;
        for c in &p {
            bin.extend_from_slice(&(*c as f32).to_le_bytes());
        }
    }
    let pos_byte_len = bin.len();
    let pos_byte_offset = 0u32;

    let mut index_count: u32 = 0;
    let mut skipped: u32 = 0;
    for f in mesh.face_ids() {
        let verts: Vec<u32> = FaceVertices::new(mesh, f)
            .filter_map(|v| index_map.get(&v).copied())
            .collect();
        if verts.len() != 3 {
            skipped += 1;
            continue;
        }
        for vi in &verts {
            bin.extend_from_slice(&vi.to_le_bytes());
        }
        index_count += 3;
    }
    let idx_byte_offset = pos_byte_len as u32;
    let idx_byte_len = bin.len() - pos_byte_len;

    // Build JSON (manually assembled to avoid serde_json dependency)
    let json = format!(
        r#"{{"asset":{{"version":"2.0","generator":"halfedge"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"indices":1,"mode":4}}]}}],"buffers":[{{"byteLength":{}}}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":{},"target":34962}},{{"buffer":0,"byteOffset":{},"byteLength":{},"target":34963}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":{},"type":"VEC3","max":[0,0,0],"min":[0,0,0]}},{{"bufferView":1,"componentType":5125,"count":{},"type":"SCALAR"}}]}}"#,
        bin.len(),
        pos_byte_len,
        idx_byte_offset,
        idx_byte_len,
        v_ids.len(),
        index_count
    );

    // Chunk data must be 4-byte aligned (pad JSON with spaces)
    let mut json_bytes = json.into_bytes();
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(b' ');
    }
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }

    let total_len = 12 + 8 + json_bytes.len() + 8 + bin.len();
    let mut out: Vec<u8> = Vec::with_capacity(total_len);
    out.extend_from_slice(&GLB_MAGIC.to_le_bytes());
    out.extend_from_slice(&GLB_VERSION.to_le_bytes());
    out.extend_from_slice(&(total_len as u32).to_le_bytes());

    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&JSON_CHUNK_TYPE.to_le_bytes());
    out.extend_from_slice(&json_bytes);

    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(&BIN_CHUNK_TYPE.to_le_bytes());
    out.extend_from_slice(&bin);

    let _ = pos_byte_offset;
    if skipped > 0 {
        log::warn!("[halfedge::format_glb] warning: skipped {skipped} non-triangle face(s)");
    }
    out
}

/// Save mesh to a GLB file.
pub fn save_glb<P: AsRef<Path>>(mesh: &MeshStorage, path: P) -> Result<(), GltfError> {
    let bytes = format_glb(mesh);
    fs::write(path, bytes)?;
    Ok(())
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
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
    fn glb_roundtrip_tetrahedron() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let bytes = format_glb(&mesh);
        let parsed = parse_glb(&bytes).expect("GLB roundtrip parse failed");
        assert_eq!(parsed.vertex_count(), mesh.vertex_count());
        assert_eq!(parsed.face_count(), mesh.face_count());
        assert!(check_topology(&parsed).is_ok());
    }

    #[test]
    fn glb_file_roundtrip() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces).unwrap();
        let path = std::env::temp_dir().join("halfedge_glb.glb");
        save_glb(&mesh, &path).expect("failed to save GLB");
        let loaded = load_glb(&path).expect("failed to load GLB");
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.vertex_count(), mesh.vertex_count());
        assert_eq!(loaded.face_count(), mesh.face_count());
    }

    #[test]
    fn glb_detects_bad_magic() {
        let bytes = [0u8; 32];
        let err = parse_glb(&bytes).unwrap_err();
        assert!(matches!(err, GltfError::Parse(_)));
    }

    #[test]
    fn glb_icosphere_roundtrip() {
        let mesh = crate::test_util::build_icosphere(1);
        let bytes = format_glb(&mesh);
        let parsed = parse_glb(&bytes).expect("GLB icosphere roundtrip failed");
        assert_eq!(parsed.vertex_count(), mesh.vertex_count());
        assert_eq!(parsed.face_count(), mesh.face_count());
    }
}
