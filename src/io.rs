//! IO 模块：OBJ / PLY / STL 格式读写 + 从顶点/面索引构建半边网格
//!
//! ## 功能
//! - [`build_mesh_from_vertices_and_faces`]：从顶点位置数组与三角面索引数组
//!   构建**完整**半边网格（含 twin / next / prev / 边界环）。
//! - [`build_mesh_from_polygons`]：从任意边数多边形构建半边网格。
//! - OBJ：`load_obj` / `parse_obj`（支持三角面及 n-gon），
//!   `save_obj` / `format_obj`（序列化为 OBJ 文本，输出任意边数面）。
//! - PLY：`load_ply` / `parse_ply` / `save_ply` / `format_ply`（ASCII 格式）。
//! - STL：`load_stl` / `parse_stl_ascii` / `parse_stl_binary` / `parse_stl_bytes`
//!   （自动判别 ASCII / 二进制），`save_stl_ascii` / `save_stl_binary` /
//!   `format_stl_ascii` / `format_stl_binary`。
//! - 统一入口：`load_mesh` / `save_mesh`，按扩展名 `.obj` / `.ply` / `.stl` 分派。
//!
//! ## OBJ 格式约定
//! ```text
//! v x y z          # 顶点（1-based 索引）
//! f i j k ...      # 面（顶点索引，支持 v/vt/vn 形式，仅取 v；支持三角及以上边数）
//! ```
//! - 顶点索引 1-based；负数表示从末尾倒数（OBJ 标准）。
//! - 非 `v` / `f` 行（如 `vt`、`vn`、`#`、空行）被忽略。
//! - 面少于 3 顶点时报错。
//!
//! ## 边界半边构建
//! 1. 每个三角面创建 3 条内部半边，按 CCW 设置 `next/prev/face`；
//! 2. 用 `HashMap<(u32, u32), HalfEdgeId>` 记录每条有向边；
//! 3. 对每条有向边 `(a, b)`，查 `(b, a)`：
//!    - 命中：互设 twin（内部边）；
//!    - 未命中：新建边界半边（`face = None`），与原半边互设 twin；
//! 4. 边界半边的 `next/prev` 由公式给出：
//!    $$
//!    \text{bh.next} = \text{bh.twin.prev.twin}, \quad
//!    \text{bh.prev} = \text{bh.twin.next.twin}
//!    $$

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

use crate::ids::{FaceId, HalfEdgeId, VertexId};
use crate::storage::{Face, HalfEdge, MeshStorage, Vertex};
use crate::traversal::{FaceHalfEdges, FaceVertices};

// ============================================================
// 错误类型
// ============================================================

/// OBJ 读写错误。
#[derive(Debug)]
pub enum ObjError {
    /// 文件 IO 错误。
    Io(std::io::Error),
    /// 解析错误：行号与描述。
    Parse { line: usize, msg: String },
    /// 面索引越界。
    IndexOutOfRange {
        line: usize,
        idx: i64,
        vertex_count: usize,
    },
    /// 非三角面（顶点数 ≠ 3）。
    NotTriangular { line: usize, face_verts: usize },
}

impl fmt::Display for ObjError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO 错误: {}", e),
            Self::Parse { line, msg } => write!(f, "第 {} 行解析错误: {}", line, msg),
            Self::IndexOutOfRange {
                line,
                idx,
                vertex_count,
            } => write!(
                f,
                "第 {} 行索引 {} 越界（当前顶点数 {}）",
                line, idx, vertex_count
            ),
            Self::NotTriangular { line, face_verts } => {
                write!(f, "第 {} 行面顶点数 {} ≠ 3，仅支持三角面", line, face_verts)
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
// 从顶点 + 面索引构建半边网格
// ============================================================

/// 从顶点位置数组与三角面索引数组构建完整半边网格。
///
/// - `vertices`：顶点位置 `[[x, y, z], ...]`；
/// - `faces`：三角面索引 `[[v0, v1, v2], ...]`，0-based，CCW 朝向。
///
/// 自动构建 twin / next / prev / 边界环 / 顶点入口 / 面入口。
pub fn build_mesh_from_vertices_and_faces(
    vertices: &[[f64; 3]],
    faces: &[[u32; 3]],
) -> MeshStorage {
    let mut mesh = MeshStorage::new();
    // 预分配容量：半边数上限 = 3 * F（内部） + 3 * F（边界 twin） = 6 * F
    mesh.reserve(vertices.len(), faces.len() * 6, faces.len());

    // 1. 创建所有顶点
    let v_ids: Vec<VertexId> = vertices
        .iter()
        .map(|p| mesh.add_vertex(Vertex::new(*p)))
        .collect();

    // 2. 为每个面创建 3 条内部半边
    let mut edge_map: HashMap<(u32, u32), HalfEdgeId> = HashMap::new();
    for face_idx in faces {
        let [i0, i1, i2] = *face_idx;
        let v0 = v_ids[i0 as usize];
        let v1 = v_ids[i1 as usize];
        let v2 = v_ids[i2 as usize];

        let h0 = mesh.add_halfedge(HalfEdge::new(v1)); // v0 → v1
        let h1 = mesh.add_halfedge(HalfEdge::new(v2)); // v1 → v2
        let h2 = mesh.add_halfedge(HalfEdge::new(v0)); // v2 → v0

        let f_id = mesh.add_face(Face::new());
        for (he, next, prev) in [(h0, h1, h2), (h1, h2, h0), (h2, h0, h1)] {
            let h = mesh.get_halfedge_mut(he).unwrap();
            h.next = Some(next);
            h.prev = Some(prev);
            h.face = Some(f_id);
        }
        mesh.get_face_mut(f_id).unwrap().halfedge = Some(h0);

        edge_map.insert((i0, i1), h0);
        edge_map.insert((i1, i2), h1);
        edge_map.insert((i2, i0), h2);

        // 顶点 outgoing 入口（若未设置）
        if mesh.get_vertex(v0).unwrap().halfedge.is_none() {
            mesh.get_vertex_mut(v0).unwrap().halfedge = Some(h0);
        }
        if mesh.get_vertex(v1).unwrap().halfedge.is_none() {
            mesh.get_vertex_mut(v1).unwrap().halfedge = Some(h1);
        }
        if mesh.get_vertex(v2).unwrap().halfedge.is_none() {
            mesh.get_vertex_mut(v2).unwrap().halfedge = Some(h2);
        }
    }

    // 3. 建立 twin 关系；缺失反向边时创建边界半边
    let directed_edges: Vec<(u32, u32, HalfEdgeId)> =
        edge_map.iter().map(|((a, b), h)| (*a, *b, *h)).collect();

    let mut boundary_twins: Vec<HalfEdgeId> = Vec::new();
    for (a, b, h) in &directed_edges {
        // 跳过已处理（h.twin 已设置）
        if mesh.get_halfedge(*h).unwrap().twin.is_some() {
            continue;
        }
        if let Some(reverse_h) = edge_map.get(&(*b, *a)) {
            // 内部边
            mesh.get_halfedge_mut(*h).unwrap().twin = Some(*reverse_h);
            mesh.get_halfedge_mut(*reverse_h).unwrap().twin = Some(*h);
        } else {
            // 边界边：创建反向边界半边
            let origin_v = v_ids[*a as usize]; // 反向半边的 tip = 原半边的 origin
            let twin_id = mesh.add_halfedge(HalfEdge::new(origin_v));
            mesh.get_halfedge_mut(*h).unwrap().twin = Some(twin_id);
            mesh.get_halfedge_mut(twin_id).unwrap().twin = Some(*h);
            // 边界半边 face=None, next/prev 留待第 4 步
            boundary_twins.push(twin_id);
        }
    }

    // 4. 设置边界半边的 next/prev
    // 算法：从 bh.twin 出发沿 CCW 方向绕 bh.tip 旋转，直到找到边界 outgoing；
    //      从 bh 出发沿 CW 方向绕 bh.origin 旋转，直到找到 twin 为边界的 outgoing。
    for bh in &boundary_twins {
        // bh.next：绕 bh.tip CCW 走，找边界 outgoing
        let mut cur = mesh.get_halfedge(*bh).unwrap().twin.unwrap();
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
            mesh.get_halfedge_mut(*bh).unwrap().next = Some(n);
        }

        // bh.prev：绕 bh.origin CW 走，找 twin 为边界的 outgoing
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
            mesh.get_halfedge_mut(*bh).unwrap().prev = Some(p);
        }
    }

    mesh
}

/// 从顶点位置数组与任意多边形面索引数组构建完整半边网格。
///
/// 与 [`build_mesh_from_vertices_and_faces`] 不同，本函数接受任意边数的多边形
/// （三角形、四边形、五边形等），适用于 Catmull-Clark 细分等需要多边形输入的场景。
///
/// # 约定
/// - 每个面的顶点索引按 CCW（从面外侧看逆时针）排列；
/// - 自动建立 twin / next / prev / 边界环；
/// - **注意**：输出的多边形面不通过 [`crate::validate::validate_topology`] 的
///   三角面校验（`FaceNotTriangular`），但其他校验项仍满足。
pub fn build_mesh_from_polygons(vertices: &[[f64; 3]], faces: &[Vec<u32>]) -> MeshStorage {
    let mut mesh = MeshStorage::new();
    // 预分配容量：半边数上限 = Σ k_i（内部） + Σ k_i（边界 twin） = 2 * Σ k_i
    let total_he: usize = faces
        .iter()
        .map(|f| f.len())
        .sum::<usize>()
        .saturating_mul(2);
    mesh.reserve(vertices.len(), total_he, faces.len());

    // 1. 创建所有顶点
    let v_ids: Vec<VertexId> = vertices
        .iter()
        .map(|p| mesh.add_vertex(Vertex::new(*p)))
        .collect();

    // 2. 为每个面创建 k 条内部半边
    let mut edge_map: HashMap<(u32, u32), HalfEdgeId> = HashMap::new();
    for face_idx in faces {
        let k = face_idx.len();
        if k < 3 {
            continue; // 退化面，跳过
        }
        // 创建 k 条半边
        let mut he_ids: Vec<HalfEdgeId> = Vec::with_capacity(k);
        for i in 0..k {
            let v_from = v_ids[face_idx[i] as usize];
            let v_to = v_ids[face_idx[(i + 1) % k] as usize];
            let h = mesh.add_halfedge(HalfEdge::new(v_to)); // v_from → v_to
            he_ids.push(h);
            // 顶点 outgoing 入口（若未设置）
            if mesh.get_vertex(v_from).unwrap().halfedge.is_none() {
                mesh.get_vertex_mut(v_from).unwrap().halfedge = Some(h);
            }
        }
        // 创建面并设置 next/prev/face
        let f_id = mesh.add_face(Face::new());
        for i in 0..k {
            let next = he_ids[(i + 1) % k];
            let prev = he_ids[(i + k - 1) % k];
            let h = mesh.get_halfedge_mut(he_ids[i]).unwrap();
            h.next = Some(next);
            h.prev = Some(prev);
            h.face = Some(f_id);
        }
        mesh.get_face_mut(f_id).unwrap().halfedge = Some(he_ids[0]);
        // 注册有向边
        for i in 0..k {
            let a = face_idx[i];
            let b = face_idx[(i + 1) % k];
            edge_map.insert((a, b), he_ids[i]);
        }
    }

    // 3. 建立 twin 关系；缺失反向边时创建边界半边
    let directed_edges: Vec<(u32, u32, HalfEdgeId)> =
        edge_map.iter().map(|((a, b), h)| (*a, *b, *h)).collect();

    let mut boundary_twins: Vec<HalfEdgeId> = Vec::new();
    for (a, b, h) in &directed_edges {
        if mesh.get_halfedge(*h).unwrap().twin.is_some() {
            continue;
        }
        if let Some(reverse_h) = edge_map.get(&(*b, *a)) {
            mesh.get_halfedge_mut(*h).unwrap().twin = Some(*reverse_h);
            mesh.get_halfedge_mut(*reverse_h).unwrap().twin = Some(*h);
        } else {
            let origin_v = v_ids[*a as usize];
            let twin_id = mesh.add_halfedge(HalfEdge::new(origin_v));
            mesh.get_halfedge_mut(*h).unwrap().twin = Some(twin_id);
            mesh.get_halfedge_mut(twin_id).unwrap().twin = Some(*h);
            boundary_twins.push(twin_id);
        }
    }

    // 4. 设置边界半边的 next/prev（与三角版相同算法）
    for bh in &boundary_twins {
        let mut cur = mesh.get_halfedge(*bh).unwrap().twin.unwrap();
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
            mesh.get_halfedge_mut(*bh).unwrap().next = Some(n);
        }

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
            mesh.get_halfedge_mut(*bh).unwrap().prev = Some(p);
        }
    }

    mesh
}

// ============================================================
// OBJ 加载
// ============================================================

/// 加载 OBJ 文件，仅读取 `v` 与 `f` 行，支持任意边数面。
pub fn load_obj<P: AsRef<Path>>(path: P) -> Result<MeshStorage, ObjError> {
    let text = fs::read_to_string(path)?;
    parse_obj(&text)
}

/// 解析 OBJ 文本为半边网格。支持三角面及 n-gon。
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
                            msg: format!("无法解析顶点坐标: {}", t),
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if coords.len() != 3 {
                    return Err(ObjError::Parse {
                        line: line_no + 1,
                        msg: "顶点行缺少坐标分量".into(),
                    });
                }
                vertices.push([coords[0], coords[1], coords[2]]);
            }
            "f" => {
                let verts: Vec<i64> = tokens
                    .map(|t| {
                        // 支持 v/vt/vn 形式，仅取第一个分量
                        let v_part = t.split('/').next().unwrap_or(t);
                        v_part.parse::<i64>().map_err(|_| ObjError::Parse {
                            line: line_no + 1,
                            msg: format!("无法解析面索引: {}", t),
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
                        // 负数：从末尾倒数
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
                            msg: "面索引不能为 0".into(),
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
                // 忽略其他行（vt, vn, g, o, s, mtllib, usemtl, ...）
            }
        }
    }

    Ok(build_mesh_from_polygons(&vertices, &faces))
}

// ============================================================
// OBJ 保存
// ============================================================

/// 将网格保存为 OBJ 文件。
pub fn save_obj<P: AsRef<Path>>(mesh: &MeshStorage, path: P) -> Result<(), ObjError> {
    let text = format_obj(mesh);
    fs::write(path, text)?;
    Ok(())
}

/// 将网格序列化为 OBJ 文本。
///
/// - 顶点按 `vertex_ids()` 顺序输出，分配 1-based 索引；
/// - 面按 `face_ids()` 顺序输出，每行 `f i j k`。
pub fn format_obj(mesh: &MeshStorage) -> String {
    let mut v_index: HashMap<VertexId, u32> = HashMap::new();
    let mut out = String::new();
    for (next_idx, v_id) in (1u32..).zip(mesh.vertex_ids()) {
        v_index.insert(v_id, next_idx);
        let p = mesh.get_vertex(v_id).unwrap().position;
        out.push_str(&format!("v {:.6} {:.6} {:.6}\n", p[0], p[1], p[2]));
    }
    for f_id in mesh.face_ids() {
        let verts: Vec<u32> = FaceHalfEdges::new(mesh, f_id)
            .filter_map(|he| mesh.get_halfedge(he))
            .map(|h| h.vertex)
            .filter_map(|v| v_index.get(&v).copied())
            .collect();
        if verts.len() < 3 {
            continue; // 跳过退化面
        }
        out.push('f');
        for v in &verts {
            out.push(' ');
            out.push_str(&v.to_string());
        }
        out.push('\n');
    }
    out
}

// ============================================================
// PLY I/O
// ============================================================

/// PLY 解析/序列化错误。
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
            Self::Parse { line, msg } => write!(f, "PLY parse error at line {line}: {msg}"),
            Self::Unsupported(s) => write!(f, "Unsupported PLY feature: {s}"),
        }
    }
}

impl std::error::Error for PlyError {}

impl From<std::io::Error> for PlyError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// 加载 PLY 文件（ASCII 格式）。
pub fn load_ply<P: AsRef<Path>>(path: P) -> Result<MeshStorage, PlyError> {
    let text = fs::read_to_string(path)?;
    parse_ply(&text)
}

/// 解析 PLY ASCII 文本。
pub fn parse_ply(text: &str) -> Result<MeshStorage, PlyError> {
    let mut lines = text.lines().enumerate();
    let mut vertex_count: usize = 0;
    let mut _face_count: usize = 0;
    let mut in_header = true;
    let mut vertices: Vec<[f64; 3]> = Vec::new();
    let mut faces: Vec<Vec<u32>> = Vec::new();

    for (line_no, raw) in &mut lines {
        let line = raw.trim();
        if line.is_empty() || (in_header && line.starts_with("comment")) {
            continue;
        }
        if in_header {
            if line == "end_header" {
                in_header = false;
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 && parts[0] == "element" {
                if parts[1] == "vertex" {
                    vertex_count = parts[2].parse().map_err(|_| PlyError::Parse {
                        line: line_no + 1,
                        msg: "invalid vertex count".into(),
                    })?;
                } else if parts[1] == "face" {
                    _face_count = parts[2].parse().map_err(|_| PlyError::Parse {
                        line: line_no + 1,
                        msg: "invalid face count".into(),
                    })?;
                }
            }
        } else {
            if vertices.len() < vertex_count {
                let coords: Vec<f64> = line
                    .split_whitespace()
                    .take(3)
                    .map(|s| {
                        s.parse::<f64>().map_err(|_| PlyError::Parse {
                            line: line_no + 1,
                            msg: format!("invalid vertex coordinate: {s}"),
                        })
                    })
                    .collect::<Result<_, _>>()?;
                if coords.len() >= 3 {
                    vertices.push([coords[0], coords[1], coords[2]]);
                }
            } else {
                let indices: Vec<u32> = line
                    .split_whitespace()
                    .skip(1) // skip the count prefix
                    .map(|s| {
                        s.parse::<u32>().map_err(|_| PlyError::Parse {
                            line: line_no + 1,
                            msg: format!("invalid face index: {s}"),
                        })
                    })
                    .collect::<Result<_, _>>()?;
                if indices.len() >= 3 {
                    faces.push(indices);
                }
            }
        }
    }

    Ok(build_mesh_from_polygons(&vertices, &faces))
}

/// 将网格序列化为 PLY ASCII 文本。
pub fn format_ply(mesh: &MeshStorage) -> String {
    let v_ids: Vec<VertexId> = mesh.vertex_ids().collect();
    let f_ids: Vec<FaceId> = mesh.face_ids().collect();
    let mut index_map: std::collections::HashMap<VertexId, usize> =
        std::collections::HashMap::new();
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
        let p = mesh.get_vertex(v).unwrap().position;
        out.push_str(&format!("{:.6} {:.6} {:.6}\n", p[0], p[1], p[2]));
    }
    for f in &f_ids {
        let verts: Vec<usize> = FaceVertices::new(mesh, *f)
            .filter_map(|v| index_map.get(&v).copied())
            .collect();
        if verts.len() < 3 {
            continue;
        }
        out.push_str(&verts.len().to_string());
        for vi in &verts {
            out.push(' ');
            out.push_str(&vi.to_string());
        }
        out.push('\n');
    }
    out
}

/// 将网格保存为 PLY 文件（ASCII 格式）。
pub fn save_ply<P: AsRef<Path>>(mesh: &MeshStorage, path: P) -> Result<(), PlyError> {
    let text = format_ply(mesh);
    fs::write(path, text)?;
    Ok(())
}

// ============================================================
// STL I/O（ASCII + 二进制）
// ============================================================

/// STL 解析/序列化错误。
#[derive(Debug)]
pub enum StlError {
    Io(std::io::Error),
    /// ASCII 解析错误：行号 + 描述。
    Parse {
        line: usize,
        msg: String,
    },
    /// 二进制文件大小不匹配（实际长度 / 期望长度）。
    BadBinarySize {
        actual: usize,
        expected: usize,
    },
    /// 面顶点数 ≠ 3。
    NotTriangular {
        face_verts: usize,
    },
}

impl fmt::Display for StlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Parse { line, msg } => write!(f, "STL parse error at line {line}: {msg}"),
            Self::BadBinarySize { actual, expected } => write!(
                f,
                "STL binary size mismatch: actual {actual} bytes, expected {expected} bytes"
            ),
            Self::NotTriangular { face_verts } => {
                write!(f, "STL face vertex count {face_verts} ≠ 3")
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

/// 加载 STL 文件。自动识别 ASCII（首行含 "solid"）与二进制格式。
///
/// 注意：仅依据首行 "solid" 判断并不可靠（部分二进制文件首部恰好为 "solid"）。
/// 本实现采用启发式：若首行前 5 个字节为 `solid` 且文件大小恰好等于
/// `84 + 50 * N`（N 为面数），则视为二进制；否则视为 ASCII。
pub fn load_stl<P: AsRef<Path>>(path: P) -> Result<MeshStorage, StlError> {
    let bytes = fs::read(path)?;
    parse_stl_bytes(&bytes)
}

/// 解析 STL 字节流（自动判别 ASCII / 二进制）。
pub fn parse_stl_bytes(bytes: &[u8]) -> Result<MeshStorage, StlError> {
    // 二进制 STL：80 字节 header + 4 字节面数 + 50*N 字节三角形数据
    if bytes.len() >= 84 {
        let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
        let expected = 84 + 50 * n;
        // 启发式：大小恰好匹配 → 二进制；否则按 ASCII 解析
        if bytes.len() == expected {
            return parse_stl_binary(bytes, n);
        }
    }
    // 否则按 ASCII 解析
    let text = std::str::from_utf8(bytes).map_err(|_| StlError::Parse {
        line: 0,
        msg: "file is not valid UTF-8".into(),
    })?;
    parse_stl_ascii(text)
}

/// 解析 ASCII STL。
///
/// 格式：
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
    use std::collections::HashMap;
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
                        msg: "vertex line needs 3 coordinates".into(),
                    });
                }
                let coords: [f64; 3] = [
                    parts[1].parse().map_err(|_| StlError::Parse {
                        line: i + 1,
                        msg: format!("invalid x: {}", parts[1]),
                    })?,
                    parts[2].parse().map_err(|_| StlError::Parse {
                        line: i + 1,
                        msg: format!("invalid y: {}", parts[2]),
                    })?,
                    parts[3].parse().map_err(|_| StlError::Parse {
                        line: i + 1,
                        msg: format!("invalid z: {}", parts[3]),
                    })?,
                ];
                // 用位模式做精确去重（与 binary 路径一致）
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

    Ok(build_mesh_from_vertices_and_faces(&vertices, &faces))
}

/// 解析二进制 STL。
///
/// 每个 triangle：12 字节法向 + 3×12 字节顶点 + 2 字节属性 = 50 字节。
/// 顶点索引在文件内顺序出现；本实现按位置去重（完全相同的顶点合并）。
pub fn parse_stl_binary(bytes: &[u8], n_faces: usize) -> Result<MeshStorage, StlError> {
    use std::collections::HashMap;
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
        // 跳过法向（base..base+12）
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
            // 用位模式做精确去重
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

    Ok(build_mesh_from_vertices_and_faces(&vertices, &faces))
}

/// 保存网格为 ASCII STL。
pub fn save_stl_ascii<P: AsRef<Path>>(mesh: &MeshStorage, path: P) -> Result<(), StlError> {
    let text = format_stl_ascii(mesh);
    fs::write(path, text)?;
    Ok(())
}

/// 序列化网格为 ASCII STL 文本。
///
/// 法向用 Newell 方法（与 `geometry::face_normal` 一致）计算；
/// 仅输出三角面，非三角面跳过。
pub fn format_stl_ascii(mesh: &MeshStorage) -> String {
    use crate::geometry::face_normal;
    let mut out = String::with_capacity(mesh.face_count() * 80);
    out.push_str("solid halfedge\n");
    for f in mesh.face_ids() {
        let verts: Vec<VertexId> = crate::traversal::FaceVertices::new(mesh, f).collect();
        if verts.len() != 3 {
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
    out.push_str("endsolid halfedge\n");
    out
}

/// 保存网格为二进制 STL。
pub fn save_stl_binary<P: AsRef<Path>>(mesh: &MeshStorage, path: P) -> Result<(), StlError> {
    let bytes = format_stl_binary(mesh);
    fs::write(path, bytes)?;
    Ok(())
}

/// 序列化网格为二进制 STL 字节流。
pub fn format_stl_binary(mesh: &MeshStorage) -> Vec<u8> {
    use crate::geometry::face_normal;
    let mut out: Vec<u8> = Vec::with_capacity(84 + mesh.face_count() * 50);
    // 80 字节 header
    out.extend_from_slice(b"halfedge binary stl".as_slice());
    while out.len() < 80 {
        out.push(0);
    }
    // 4 字节面数
    let n = mesh.face_count() as u32;
    out.extend_from_slice(&n.to_le_bytes());

    for f in mesh.face_ids() {
        let verts: Vec<VertexId> = crate::traversal::FaceVertices::new(mesh, f).collect();
        if verts.len() != 3 {
            // 退化：填零三角，但保留面计数一致性由调用方保证
            out.extend_from_slice(&[0u8; 50]);
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
        // 2 字节属性
        out.extend_from_slice(&[0u8, 0u8]);
    }
    out
}

// ============================================================
// 自动检测格式
// ============================================================

/// 统一 I/O 错误类型。
#[derive(Debug)]
pub enum MeshError {
    Obj(ObjError),
    Ply(PlyError),
    Stl(StlError),
    UnsupportedFormat(String),
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Obj(e) => write!(f, "OBJ error: {e}"),
            Self::Ply(e) => write!(f, "PLY error: {e}"),
            Self::Stl(e) => write!(f, "STL error: {e}"),
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

/// 自动检测文件格式并加载网格。
///
/// 根据文件扩展名选择解析器：
/// - `.obj` → OBJ（支持三角面与 n-gon）
/// - `.ply` → PLY ASCII
/// - `.stl` → STL（自动判别 ASCII / 二进制）
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
        other => Err(MeshError::UnsupportedFormat(other.into())),
    }
}

/// 自动检测文件格式并保存网格。
///
/// 根据文件扩展名选择序列化器：
/// - `.obj` → OBJ
/// - `.ply` → PLY ASCII
/// - `.stl` → STL ASCII（若需二进制，请直接调用 `save_stl_binary`）
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
        other => Err(MeshError::UnsupportedFormat(other.into())),
    }
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::check_topology;

    /// 两个三角形拼成的四边形：
    /// v0-v1-v2 三角形 + v0-v2-v3 三角形（CCW 朝向 +z）
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
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces);
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.face_count(), 2);
        // 两个三角形共享 1 条边，其余 4 条是边界
        // 总半边数 = 2*3 + 4*2 = 14 (内部边 1 对 twin + 边界 4 对 twin)
        // 内部半边 = 6 (3*2)，边界 twin = 4*1 = 4，总 = 6+4 = 10
        // 等等：内部边 1 条 → twin 对 1 → 2 半边；边界 4 条 → 4 twin 对 → 8 半边
        // 总半边 = 2 + 8 = 10
        assert_eq!(mesh.halfedge_count(), 10);
    }

    #[test]
    fn build_mesh_passes_full_validation() {
        let (verts, faces) = make_quad_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces);
        assert!(
            check_topology(&mesh).is_ok(),
            "构建的网格应通过完整校验: {:?}",
            check_topology(&mesh)
        );
    }

    #[test]
    fn build_mesh_closed_tetrahedron() {
        // 四面体：4 顶点 4 三角面，闭合
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let faces = vec![
            [0, 1, 2], // 底面（CCW 朝 -z）
            [0, 2, 3], // 侧面 1
            [0, 3, 1], // 侧面 2
            [1, 3, 2], // 侧面 3
        ];
        let mesh = build_mesh_from_vertices_and_faces(&vertices, &faces);
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.face_count(), 4);
        // 闭合：6 条边 * 2 半边 = 12 半边
        assert_eq!(mesh.halfedge_count(), 12);
        assert!(check_topology(&mesh).is_ok());
    }

    #[test]
    fn obj_roundtrip_quad() {
        let (verts, faces) = make_quad_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces);
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
        let mesh = build_mesh_from_vertices_and_faces(&vertices, &faces);
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
# 这是一个测试 OBJ
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
        assert_eq!(mesh.halfedge_count(), 6); // 1 三角形 = 3 内部 + 3 边界 twin
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
        let mesh = parse_obj(text).expect("四边形面 OBJ 解析应成功");
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
            other => panic!("期望 IndexOutOfRange 错误，实际: {:?}", other),
        }
    }

    #[test]
    fn obj_save_load_file_roundtrip() {
        let (verts, faces) = make_quad_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces);
        let path = std::env::temp_dir().join("halfedge_test_quad.obj");
        save_obj(&mesh, &path).unwrap();
        let loaded = load_obj(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.vertex_count(), mesh.vertex_count());
        assert_eq!(loaded.face_count(), mesh.face_count());
        assert_eq!(loaded.halfedge_count(), mesh.halfedge_count());
        assert!(check_topology(&loaded).is_ok());
    }

    // ---------- 自动检测格式 ----------

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
        let path = std::env::temp_dir().join("halfedge_autodetect.off");
        let err = load_mesh(&path).unwrap_err();
        assert!(matches!(err, MeshError::UnsupportedFormat(_)));
    }

    // ---------- STL 测试 ----------

    fn make_tetra_data() -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        // 标准四面体：4 顶点 4 面，CCW 朝外
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let faces = vec![
            [0, 2, 1], // 底面，朝 -z
            [0, 1, 3], // 前面
            [0, 3, 2], // 左面
            [1, 2, 3], // 斜面
        ];
        (vertices, faces)
    }

    #[test]
    fn stl_ascii_roundtrip() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces);
        let text = format_stl_ascii(&mesh);
        let parsed = parse_stl_ascii(&text).expect("ASCII STL 往返解析失败");
        assert_eq!(parsed.vertex_count(), mesh.vertex_count());
        assert_eq!(parsed.face_count(), mesh.face_count());
    }

    #[test]
    fn stl_binary_roundtrip() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces);
        let bytes = format_stl_binary(&mesh);
        // 84 + 4*50 = 284 字节
        assert_eq!(bytes.len(), 84 + 50 * 4);
        let parsed = parse_stl_bytes(&bytes).expect("二进制 STL 往返解析失败");
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
        let mesh = parse_stl_ascii(text).expect("最小 ASCII STL 解析失败");
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
        // 长度对不上声明的面数
        let mut bytes = vec![0u8; 84];
        bytes.extend_from_slice(&3u32.to_le_bytes()); // 声明 3 面
        bytes.extend_from_slice(&[0u8; 50]); // 实际只有 1 面
        let err = parse_stl_binary(&bytes, 3).unwrap_err();
        assert!(matches!(err, StlError::BadBinarySize { .. }));
    }

    #[test]
    fn stl_file_roundtrip_ascii() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces);
        let path = std::env::temp_dir().join("halfedge_stl_ascii.stl");
        save_stl_ascii(&mesh, &path).expect("保存 STL 文件失败");
        let loaded = load_stl(&path).expect("加载 STL 文件失败");
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.vertex_count(), mesh.vertex_count());
        assert_eq!(loaded.face_count(), mesh.face_count());
    }

    #[test]
    fn stl_file_roundtrip_binary() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces);
        let path = std::env::temp_dir().join("halfedge_stl_bin.stl");
        save_stl_binary(&mesh, &path).expect("保存二进制 STL 失败");
        let loaded = load_stl(&path).expect("加载二进制 STL 失败");
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.vertex_count(), mesh.vertex_count());
        assert_eq!(loaded.face_count(), mesh.face_count());
    }

    #[test]
    fn auto_detect_stl() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces);
        let path = std::env::temp_dir().join("halfedge_autodetect.stl");
        save_mesh(&mesh, &path).expect("save_mesh(.stl) 失败");
        let loaded = load_mesh(&path).expect("load_mesh(.stl) 失败");
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.face_count(), mesh.face_count());
    }
}
