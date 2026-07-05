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
    let n_verts = v_ids.len();
    for face_idx in faces {
        let [i0, i1, i2] = *face_idx;
        // 越界检查：公开构建函数不应静默 panic
        if (i0 as usize) >= n_verts
            || (i1 as usize) >= n_verts
            || (i2 as usize) >= n_verts
        {
            panic!(
                "build_mesh_from_vertices_and_faces: 面索引 [{}, {}, {}] 越界（顶点总数 {}）",
                i0, i1, i2, n_verts
            );
        }
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
    let n_verts = v_ids.len();
    let mut skipped_degenerate: u32 = 0;
    for face_idx in faces {
        let k = face_idx.len();
        if k < 3 {
            skipped_degenerate += 1;
            continue; // 退化面，跳过
        }
        // 越界检查：公开构建函数不应静默 panic
        for idx in face_idx {
            if (*idx as usize) >= n_verts {
                panic!(
                    "build_mesh_from_polygons: 面索引 {} 越界（顶点总数 {}）",
                    idx, n_verts
                );
            }
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

    if skipped_degenerate > 0 {
        eprintln!(
            "[halfedge::build_mesh_from_polygons] 警告：跳过 {skipped_degenerate} 个退化面（顶点数 < 3）"
        );
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
    let mut skipped: u32 = 0;
    for f_id in mesh.face_ids() {
        let verts: Vec<u32> = FaceHalfEdges::new(mesh, f_id)
            .filter_map(|he| mesh.get_halfedge(he))
            .map(|h| h.vertex)
            .filter_map(|v| v_index.get(&v).copied())
            .collect();
        if verts.len() < 3 {
            skipped += 1;
            continue; // 跳过退化面
        }
        out.push('f');
        for v in &verts {
            out.push(' ');
            out.push_str(&v.to_string());
        }
        out.push('\n');
    }
    if skipped > 0 {
        eprintln!("[halfedge::format_obj] 警告：跳过 {skipped} 个退化面（顶点数 < 3）");
    }
    out
}

// ============================================================
// PLY I/O（ASCII + 二进制小端）
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
            Self::Io(e) => write!(f, "IO 错误: {e}"),
            Self::Parse { line, msg } => write!(f, "第 {line} 行 PLY 解析错误: {msg}"),
            Self::Unsupported(s) => write!(f, "不支持的 PLY 特性: {s}"),
        }
    }
}

impl std::error::Error for PlyError {}

impl From<std::io::Error> for PlyError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// PLY 数据格式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyFormat {
    Ascii,
    BinaryLittleEndian,
    #[allow(dead_code)]
    BinaryBigEndian,
}

/// PLY 标量类型与字节宽度。
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

    /// 以小端字节读取该类型并返回 f64（用于顶点坐标）。
    fn read_le_as_f64(&self, bytes: &[u8]) -> Result<f64, PlyError> {
        match self {
            Self::Char => bytes
                .first()
                .map(|b| (*b as i8) as f64)
                .ok_or_else(|| PlyError::Unsupported("字节流意外结束".into())),
            Self::UChar => bytes
                .first()
                .map(|b| *b as f64)
                .ok_or_else(|| PlyError::Unsupported("字节流意外结束".into())),
            Self::Short => bytes
                .get(0..2)
                .and_then(|s| s.try_into().ok())
                .map(|arr: [u8; 2]| i16::from_le_bytes(arr) as f64)
                .ok_or_else(|| PlyError::Unsupported("字节流意外结束".into())),
            Self::UShort => bytes
                .get(0..2)
                .and_then(|s| s.try_into().ok())
                .map(|arr: [u8; 2]| u16::from_le_bytes(arr) as f64)
                .ok_or_else(|| PlyError::Unsupported("字节流意外结束".into())),
            Self::Int => bytes
                .get(0..4)
                .and_then(|s| s.try_into().ok())
                .map(|arr: [u8; 4]| i32::from_le_bytes(arr) as f64)
                .ok_or_else(|| PlyError::Unsupported("字节流意外结束".into())),
            Self::UInt => bytes
                .get(0..4)
                .and_then(|s| s.try_into().ok())
                .map(|arr: [u8; 4]| u32::from_le_bytes(arr) as f64)
                .ok_or_else(|| PlyError::Unsupported("字节流意外结束".into())),
            Self::Float => bytes
                .get(0..4)
                .and_then(|s| s.try_into().ok())
                .map(|arr: [u8; 4]| f32::from_le_bytes(arr) as f64)
                .ok_or_else(|| PlyError::Unsupported("字节流意外结束".into())),
            Self::Double => bytes
                .get(0..8)
                .and_then(|s| s.try_into().ok())
                .map(|arr: [u8; 8]| f64::from_le_bytes(arr))
                .ok_or_else(|| PlyError::Unsupported("字节流意外结束".into())),
        }
    }

    /// 以小端字节读取该类型并返回 u32（用于面索引）。
    fn read_le_as_u32(&self, bytes: &[u8]) -> Result<u32, PlyError> {
        match self {
            Self::Char => bytes
                .first()
                .map(|b| (*b as i8) as u32)
                .ok_or_else(|| PlyError::Unsupported("字节流意外结束".into())),
            Self::UChar => bytes
                .first()
                .map(|b| *b as u32)
                .ok_or_else(|| PlyError::Unsupported("字节流意外结束".into())),
            Self::Short => bytes
                .get(0..2)
                .and_then(|s| s.try_into().ok())
                .map(|arr: [u8; 2]| i16::from_le_bytes(arr) as u32)
                .ok_or_else(|| PlyError::Unsupported("字节流意外结束".into())),
            Self::UShort => bytes
                .get(0..2)
                .and_then(|s| s.try_into().ok())
                .map(|arr: [u8; 2]| u16::from_le_bytes(arr) as u32)
                .ok_or_else(|| PlyError::Unsupported("字节流意外结束".into())),
            Self::Int => bytes
                .get(0..4)
                .and_then(|s| s.try_into().ok())
                .map(|arr: [u8; 4]| i32::from_le_bytes(arr) as u32)
                .ok_or_else(|| PlyError::Unsupported("字节流意外结束".into())),
            Self::UInt => bytes
                .get(0..4)
                .and_then(|s| s.try_into().ok())
                .map(|arr: [u8; 4]| u32::from_le_bytes(arr))
                .ok_or_else(|| PlyError::Unsupported("字节流意外结束".into())),
            Self::Float | Self::Double => {
                Err(PlyError::Unsupported("浮点类型用作索引类型".into()))
            }
        }
    }

    /// 将 f64 写为该类型的小端字节。
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

    /// 将 u32 写为该类型的小端字节。
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

/// PLY header 解析结果。
struct PlyHeader {
    format: PlyFormat,
    vertex_count: usize,
    /// 顶点属性 (name, type)，按出现顺序。
    vertex_props: Vec<(String, PlyType)>,
    face_count: usize,
    /// 面索引 list 的 (count_type, index_type)，None 表示无面。
    face_list: Option<(PlyType, PlyType)>,
}

/// 解析 PLY header 文本部分。返回 header 与剩余二进制起始偏移。
fn parse_ply_header(text_or_bytes: &[u8]) -> Result<(PlyHeader, usize), PlyError> {
    // 找到 "end_header\n" 的字节偏移
    let needle = b"end_header\n";
    let end_pos = text_or_bytes
        .windows(needle.len())
        .position(|w| w == needle)
        .ok_or_else(|| PlyError::Parse {
            line: 0,
            msg: "缺少 'end_header' 行".into(),
        })?;

    let header_str =
        std::str::from_utf8(&text_or_bytes[..end_pos]).map_err(|_| PlyError::Parse {
            line: 0,
            msg: "PLY 头部不是有效 UTF-8".into(),
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
                        msg: "format 行缺少类型".into(),
                    });
                }
                format = match parts[1] {
                    "ascii" => PlyFormat::Ascii,
                    "binary_little_endian" => PlyFormat::BinaryLittleEndian,
                    "binary_big_endian" => PlyFormat::BinaryBigEndian,
                    other => {
                        return Err(PlyError::Unsupported(format!(
                            "未知 PLY 格式: {other}"
                        )));
                    }
                };
            }
            "element" => {
                if parts.len() < 3 {
                    return Err(PlyError::Parse {
                        line: line_no + 1,
                        msg: "element 行缺少名称或计数".into(),
                    });
                }
                let name = parts[1].to_string();
                let count: usize = parts[2].parse().map_err(|_| PlyError::Parse {
                    line: line_no + 1,
                    msg: format!("无效的 element 计数: {}", parts[2]),
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
                            msg: "vertex property 行过短".into(),
                        });
                    }
                    let ty = PlyType::from_str(parts[1]).ok_or_else(|| {
                        PlyError::Unsupported(format!("未知类型: {}", parts[1]))
                    })?;
                    let name = parts[2].to_string();
                    vertex_props.push((name, ty));
                } else if elem == "face" {
                    // property list <count_type> <index_type> <name>
                    if parts.len() >= 5 && parts[1] == "list" {
                        let ct = PlyType::from_str(parts[2]).ok_or_else(|| {
                            PlyError::Unsupported(format!("未知类型: {}", parts[2]))
                        })?;
                        let it = PlyType::from_str(parts[3]).ok_or_else(|| {
                            PlyError::Unsupported(format!("未知类型: {}", parts[3]))
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

/// 加载 PLY 文件（自动判别 ASCII / 二进制）。
pub fn load_ply<P: AsRef<Path>>(path: P) -> Result<MeshStorage, PlyError> {
    let bytes = fs::read(path)?;
    parse_ply_bytes(&bytes)
}

/// 解析 PLY 字节流（自动判别 ASCII / 二进制）。
pub fn parse_ply_bytes(bytes: &[u8]) -> Result<MeshStorage, PlyError> {
    let (header, bin_offset) = parse_ply_header(bytes)?;
    match header.format {
        PlyFormat::Ascii => {
            // 将全文件转字符串走 ASCII 路径
            let text = std::str::from_utf8(bytes).map_err(|_| PlyError::Parse {
                line: 0,
                msg: "PLY ASCII 文件不是有效 UTF-8".into(),
            })?;
            parse_ply_ascii_with_header(text, &header)
        }
        PlyFormat::BinaryLittleEndian => parse_ply_binary_le(&bytes[bin_offset..], &header),
        PlyFormat::BinaryBigEndian => {
            Err(PlyError::Unsupported("不支持大端 PLY".into()))
        }
    }
}

/// 解析 PLY ASCII 文本（保留旧入口，内部委托给 \texttt{parse\_ply\_bytes}）。
pub fn parse_ply(text: &str) -> Result<MeshStorage, PlyError> {
    parse_ply_bytes(text.as_bytes())
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
        eprintln!("[halfedge::format_ply] 警告：跳过 {skipped} 个退化面（顶点数 < 3）");
    }
    out
}

/// 将网格保存为 PLY 文件（ASCII 格式）。
pub fn save_ply<P: AsRef<Path>>(mesh: &MeshStorage, path: P) -> Result<(), PlyError> {
    let text = format_ply(mesh);
    fs::write(path, text)?;
    Ok(())
}

/// 使用已解析 header 解析 PLY ASCII 文本。跳过 header 行直接读取数据。
fn parse_ply_ascii_with_header(text: &str, header: &PlyHeader) -> Result<MeshStorage, PlyError> {
    // 找到 end_header 行
    let mut end_line: Option<usize> = None;
    for (i, line) in text.lines().enumerate() {
        if line.trim() == "end_header" {
            end_line = Some(i);
            break;
        }
    }
    let start_line = end_line.ok_or_else(|| PlyError::Parse {
        line: 0,
        msg: "缺少 'end_header' 行".into(),
    })? + 1;

    // 找到顶点 x/y/z 属性的索引（属性可能不止 3 个）
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
                    msg: "vertex 行字段数少于 3".into(),
                });
            }
            let parse = |s: &str| -> Result<f64, PlyError> {
                s.parse::<f64>().map_err(|_| PlyError::Parse {
                    line: i + 1,
                    msg: format!("无效顶点坐标: {s}"),
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
            // 第一个 token 是面顶点数，剩余是索引
            let count: usize = parts[0].parse().map_err(|_| PlyError::Parse {
                line: i + 1,
                msg: format!("无效面顶点数: {}", parts[0]),
            })?;
            if parts.len() < count + 1 {
                return Err(PlyError::Parse {
                    line: i + 1,
                    msg: "face 行索引数少于声明值".into(),
                });
            }
            let mut indices: Vec<u32> = Vec::with_capacity(count);
            for k in 0..count {
                let idx: u32 = parts[1 + k].parse().map_err(|_| PlyError::Parse {
                    line: i + 1,
                    msg: format!("无效面索引: {}", parts[1 + k]),
                })?;
                indices.push(idx);
            }
            // 退化面（< 3 索引）交给 build_mesh_from_polygons 统一警告并跳过
            faces.push(indices);
        }
    }

    Ok(build_mesh_from_polygons(&vertices, &faces))
}

/// 解析 PLY 二进制小端数据（end_header 之后的字节）。
fn parse_ply_binary_le(data: &[u8], header: &PlyHeader) -> Result<MeshStorage, PlyError> {
    let mut vertices: Vec<[f64; 3]> = Vec::with_capacity(header.vertex_count);
    let mut faces: Vec<Vec<u32>> = Vec::with_capacity(header.face_count);

    // 找到 x/y/z 属性
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

    // 计算每顶点字节数
    let vertex_stride: usize = header.vertex_props.iter().map(|(_, t)| t.size()).sum();
    if vertex_stride == 0 && header.vertex_count > 0 {
        return Err(PlyError::Unsupported("顶点无属性".into()));
    }

    let mut offset = 0usize;
    for _ in 0..header.vertex_count {
        if offset + vertex_stride > data.len() {
            return Err(PlyError::Parse {
                line: 0,
                msg: "二进制顶点数据意外结束".into(),
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

    // 读面
    if let Some((ct, it)) = header.face_list {
        for _ in 0..header.face_count {
            if offset + ct.size() > data.len() {
                return Err(PlyError::Parse {
                    line: 0,
                    msg: "二进制面计数意外结束".into(),
                });
            }
            let count = ct.read_le_as_u32(&data[offset..])? as usize;
            offset += ct.size();
            if offset + it.size() * count > data.len() {
                return Err(PlyError::Parse {
                    line: 0,
                    msg: "二进制面索引意外结束".into(),
                });
            }
            let mut indices: Vec<u32> = Vec::with_capacity(count);
            for _ in 0..count {
                let idx = it.read_le_as_u32(&data[offset..])?;
                offset += it.size();
                indices.push(idx);
            }
            // 退化面（< 3 索引）交给 build_mesh_from_polygons 统一警告并跳过
            faces.push(indices);
        }
    }

    Ok(build_mesh_from_polygons(&vertices, &faces))
}

/// 序列化网格为二进制 PLY 字节流（小端，float 顶点 + uchar/int 面索引）。
pub fn format_ply_binary(mesh: &MeshStorage) -> Vec<u8> {
    let v_ids: Vec<VertexId> = mesh.vertex_ids().collect();
    let f_ids: Vec<FaceId> = mesh.face_ids().collect();
    let mut index_map: std::collections::HashMap<VertexId, usize> =
        std::collections::HashMap::new();
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
        let p = mesh.get_vertex(v).unwrap().position;
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
        eprintln!("[halfedge::format_ply_binary] 警告：跳过 {skipped} 个退化面（顶点数 < 3）");
    }
    out
}

/// 将网格保存为二进制 PLY 文件。
pub fn save_ply_binary<P: AsRef<Path>>(mesh: &MeshStorage, path: P) -> Result<(), PlyError> {
    let bytes = format_ply_binary(mesh);
    fs::write(path, bytes)?;
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
            Self::Io(e) => write!(f, "IO 错误: {e}"),
            Self::Parse { line, msg } => write!(f, "第 {line} 行 STL 解析错误: {msg}"),
            Self::BadBinarySize { actual, expected } => write!(
                f,
                "STL 二进制大小不匹配：实际 {actual} 字节，期望 {expected} 字节"
            ),
            Self::NotTriangular { face_verts } => {
                write!(f, "STL 面顶点数 {face_verts} ≠ 3")
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
        msg: "文件不是有效 UTF-8".into(),
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
                        msg: "vertex 行需要 3 个坐标".into(),
                    });
                }
                let coords: [f64; 3] = [
                    parts[1].parse().map_err(|_| StlError::Parse {
                        line: i + 1,
                        msg: format!("无效 x 坐标: {}", parts[1]),
                    })?,
                    parts[2].parse().map_err(|_| StlError::Parse {
                        line: i + 1,
                        msg: format!("无效 y 坐标: {}", parts[2]),
                    })?,
                    parts[3].parse().map_err(|_| StlError::Parse {
                        line: i + 1,
                        msg: format!("无效 z 坐标: {}", parts[3]),
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
    let mut skipped: u32 = 0;
    for f in mesh.face_ids() {
        let verts: Vec<VertexId> = crate::traversal::FaceVertices::new(mesh, f).collect();
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
        eprintln!("[halfedge::format_stl_ascii] 警告：跳过 {skipped} 个非三角面");
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

    let mut skipped: u32 = 0;
    for f in mesh.face_ids() {
        let verts: Vec<VertexId> = crate::traversal::FaceVertices::new(mesh, f).collect();
        if verts.len() != 3 {
            // 退化：填零三角，但保留面计数一致性由调用方保证
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
        // 2 字节属性
        out.extend_from_slice(&[0u8, 0u8]);
    }
    if skipped > 0 {
        eprintln!("[halfedge::format_stl_binary] 警告：填零 {skipped} 个非三角面");
    }
    out
}

// ============================================================
// OFF I/O（ASCII）
// ============================================================

/// OFF 解析/序列化错误。
#[derive(Debug)]
pub enum OffError {
    Io(std::io::Error),
    Parse { line: usize, msg: String },
}

impl fmt::Display for OffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO 错误: {e}"),
            Self::Parse { line, msg } => write!(f, "第 {line} 行 OFF 解析错误: {msg}"),
        }
    }
}

impl std::error::Error for OffError {}

impl From<std::io::Error> for OffError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// 加载 OFF 文件（ASCII）。
///
/// 格式：
/// ```text
/// OFF
/// <vertex_count> <face_count> <edge_count>
/// x y z                # vertex 0
/// ...
/// k v0 v1 ... vk-1     # face 0
/// ...
/// ```
/// 第一行 `OFF` 关键字可选（部分文件首行是 `OFFN` 或带数字）。
/// `#` 开头行视为注释。`edge_count` 通常为 0，被忽略。
pub fn load_off<P: AsRef<Path>>(path: P) -> Result<MeshStorage, OffError> {
    let text = fs::read_to_string(path)?;
    parse_off(&text)
}

/// 解析 OFF 文本。
pub fn parse_off(text: &str) -> Result<MeshStorage, OffError> {
    let mut lines = text.lines().filter(|l| {
        let t = l.trim();
        !t.is_empty() && !t.starts_with('#')
    });

    // 第一行可能含 OFF 关键字
    let first = match lines.next() {
        Some(s) => s,
        None => {
            return Err(OffError::Parse {
                line: 1,
                msg: "OFF 文件为空".into(),
            });
        }
    };
    let counts_line = if first.trim().starts_with("OFF") {
        // 若首行除 OFF 外无数字，则计数在下一行
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
            msg: "OFF 头部缺少顶点/面计数".into(),
        });
    }
    let v_count: usize = count_parts[0].parse().map_err(|_| OffError::Parse {
        line: 1,
        msg: format!("无效顶点计数: {}", count_parts[0]),
    })?;
    let f_count: usize = count_parts[1].parse().map_err(|_| OffError::Parse {
        line: 1,
        msg: format!("无效面计数: {}", count_parts[1]),
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
                    msg: "vertex 行坐标数少于 3".into(),
                });
            }
            let x: f64 = parts[0].parse().map_err(|_| OffError::Parse {
                line: line_no,
                msg: format!("无效 x 坐标: {}", parts[0]),
            })?;
            let y: f64 = parts[1].parse().map_err(|_| OffError::Parse {
                line: line_no,
                msg: format!("无效 y 坐标: {}", parts[1]),
            })?;
            let z: f64 = parts[2].parse().map_err(|_| OffError::Parse {
                line: line_no,
                msg: format!("无效 z 坐标: {}", parts[2]),
            })?;
            vertices.push([x, y, z]);
        } else if faces.len() < f_count {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }
            let k: usize = parts[0].parse().map_err(|_| OffError::Parse {
                line: line_no,
                msg: format!("无效面顶点数: {}", parts[0]),
            })?;
            // 退化面（k < 3）交给 build_mesh_from_polygons 统一警告并跳过
            if parts.len() < k + 1 {
                return Err(OffError::Parse {
                    line: line_no,
                    msg: "face 行索引数少于声明值".into(),
                });
            }
            let mut idx = Vec::with_capacity(k);
            for j in 0..k {
                let v: u32 = parts[1 + j].parse().map_err(|_| OffError::Parse {
                    line: line_no,
                    msg: format!("无效面索引: {}", parts[1 + j]),
                })?;
                if v as usize >= v_count {
                    return Err(OffError::Parse {
                        line: line_no,
                        msg: format!("面索引 {v} 越界（顶点计数 {v_count}）"),
                    });
                }
                idx.push(v);
            }
            faces.push(idx);
        }
    }

    Ok(build_mesh_from_polygons(&vertices, &faces))
}

/// 将网格序列化为 OFF 文本。
pub fn format_off(mesh: &MeshStorage) -> String {
    let v_ids: Vec<VertexId> = mesh.vertex_ids().collect();
    let f_ids: Vec<FaceId> = mesh.face_ids().collect();
    let mut index_map: std::collections::HashMap<VertexId, usize> =
        std::collections::HashMap::new();
    for (i, &v) in v_ids.iter().enumerate() {
        index_map.insert(v, i);
    }

    let mut out = String::with_capacity(v_ids.len() * 32 + f_ids.len() * 16);
    out.push_str("OFF\n");
    out.push_str(&format!("{} {} 0\n", v_ids.len(), f_ids.len()));
    for &v in &v_ids {
        let p = mesh.get_vertex(v).unwrap().position;
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
        eprintln!("[halfedge::format_off] 警告：跳过 {skipped} 个退化面（顶点数 < 3）");
    }
    out
}

/// 将网格保存为 OFF 文件。
pub fn save_off<P: AsRef<Path>>(mesh: &MeshStorage, path: P) -> Result<(), OffError> {
    let text = format_off(mesh);
    fs::write(path, text)?;
    Ok(())
}

// ============================================================
// glTF GLB I/O（最小子集：单 mesh primitive，POSITION + indices）
// ============================================================

/// glTF 解析/序列化错误。
#[derive(Debug)]
pub enum GltfError {
    Io(std::io::Error),
    Parse(String),
    Unsupported(String),
}

impl fmt::Display for GltfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO 错误: {e}"),
            Self::Parse(s) => write!(f, "glTF 解析错误: {s}"),
            Self::Unsupported(s) => write!(f, "不支持的 glTF 特性: {s}"),
        }
    }
}

impl std::error::Error for GltfError {}

impl From<std::io::Error> for GltfError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

const GLB_MAGIC: u32 = 0x4654_4C47; // "glTF"
const GLB_VERSION: u32 = 2;
const JSON_CHUNK_TYPE: u32 = 0x4E4F_534A; // "JSON"
const BIN_CHUNK_TYPE: u32 = 0x004E_4942; // "BIN\0"

/// 加载 GLB 文件（glTF 二进制容器）。
///
/// 仅支持最小子集：单 mesh primitive，含 POSITION accessor 与索引 accessor。
/// 不支持材质、纹理、动画、skin、camera、node 层级等高级特性。
pub fn load_glb<P: AsRef<Path>>(path: P) -> Result<MeshStorage, GltfError> {
    let bytes = fs::read(path)?;
    parse_glb(&bytes)
}

/// 解析 GLB 字节流。
pub fn parse_glb(bytes: &[u8]) -> Result<MeshStorage, GltfError> {
    if bytes.len() < 12 {
        return Err(GltfError::Parse("文件过短，无法构成 GLB 头部".into()));
    }
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != GLB_MAGIC {
        return Err(GltfError::Parse(format!(
            "无效 GLB magic: 0x{magic:08X}"
        )));
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != GLB_VERSION {
        return Err(GltfError::Unsupported(format!(
            "不支持 GLB 版本 {version}（仅支持 2）"
        )));
    }
    let _total_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;

    // 解析 chunks（每 chunk: 4 字节长度 + 4 字节类型 + 数据，需 4 字节对齐）
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
            return Err(GltfError::Parse("chunk 长度超过文件大小".into()));
        }
        let data = &bytes[offset..offset + chunk_len];
        offset += chunk_len;
        // 4 字节对齐填充
        while offset < bytes.len() && !offset.is_multiple_of(4) {
            offset += 1;
        }
        match chunk_type {
            JSON_CHUNK_TYPE => json_chunk = Some(data),
            BIN_CHUNK_TYPE => bin_chunk = Some(data),
            _ => {}
        }
    }

    let json_bytes = json_chunk.ok_or_else(|| GltfError::Parse("缺少 JSON chunk".into()))?;
    let json_str = std::str::from_utf8(json_bytes)
        .map_err(|_| GltfError::Parse("JSON chunk 不是有效 UTF-8".into()))?;
    let bin = bin_chunk.ok_or_else(|| GltfError::Parse("缺少 BIN chunk".into()))?;

    let json = parse_minimal_json(json_str)?;
    let gltf = GltfDoc::from_json(&json)?;

    // 找第一个 mesh 的第一个 primitive
    let prim = gltf
        .meshes
        .first()
        .and_then(|m| m.primitives.first())
        .ok_or_else(|| GltfError::Parse("未找到 mesh primitive".into()))?;

    // 读 POSITION accessor
    let pos_acc_idx = prim
        .attributes
        .get("POSITION")
        .copied()
        .ok_or_else(|| GltfError::Parse("primitive 缺少 POSITION 属性".into()))?;
    let positions = read_accessor_f32x3(&gltf, pos_acc_idx, bin)?;

    // 读 indices（可选）
    let indices: Vec<u32> = if let Some(idx_acc) = prim.indices {
        read_accessor_u32(&gltf, idx_acc, bin)?
    } else {
        // 无索引：非 indexed draw，顺序 0..N
        (0..positions.len() as u32).collect()
    };

    // 转三角面：若 mode != 4（TRIANGLES）则不支持
    if prim.mode != 4 {
        return Err(GltfError::Unsupported(format!(
            "不支持 primitive mode {}（仅支持 4 = TRIANGLES）",
            prim.mode
        )));
    }

    if !indices.len().is_multiple_of(3) {
        return Err(GltfError::Parse(format!(
            "索引数 {} 不是 3 的倍数",
            indices.len()
        )));
    }

    let mut faces: Vec<[u32; 3]> = Vec::with_capacity(indices.len() / 3);
    for tri in indices.chunks_exact(3) {
        faces.push([tri[0], tri[1], tri[2]]);
    }

    Ok(build_mesh_from_vertices_and_faces(&positions, &faces))
}

/// 序列化网格为 GLB 字节流（最小子集）。
///
/// 输出结构：
/// - 12 字节 GLB header（magic, version=2, total_length）
/// - JSON chunk：描述 buffers / bufferViews / accessors / meshes
/// - BIN chunk：position (float32 × 3N) + indices (uint32 × M)
///
/// 仅生成单 mesh / 单 primitive（mode=4 TRIANGLES），无材质。
/// 非三角面被跳过。
pub fn format_glb(mesh: &MeshStorage) -> Vec<u8> {
    let v_ids: Vec<VertexId> = mesh.vertex_ids().collect();
    let mut index_map: std::collections::HashMap<VertexId, u32> = std::collections::HashMap::new();
    for (i, &v) in v_ids.iter().enumerate() {
        index_map.insert(v, i as u32);
    }

    // 构造 BIN 数据
    let mut bin: Vec<u8> = Vec::with_capacity(v_ids.len() * 12 + mesh.face_count() * 12);
    for &v in &v_ids {
        let p = mesh.get_vertex(v).unwrap().position;
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

    // 构造 JSON（手动拼接避免依赖 serde_json）
    let json = format!(
        r#"{{"asset":{{"version":"2.0","generator":"halfedge"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"indices":1,"mode":4}}]}}],"buffers":[{{"byteLength":{}}}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":{},"target":34962}},{{"buffer":0,"byteOffset":{},"byteLength":{},"target":34963}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":{},"type":"VEC3","max":[0,0,0],"min":[0,0,0]}},{{"bufferView":1,"componentType":5125,"count":{},"type":"SCALAR"}}]}}"#,
        bin.len(),
        pos_byte_len,
        idx_byte_offset,
        idx_byte_len,
        v_ids.len(),
        index_count
    );

    // chunk 数据需 4 字节对齐（用空格填充 JSON）
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

    // 隐藏未使用变量警告
    let _ = pos_byte_offset;
    if skipped > 0 {
        eprintln!("[halfedge::format_glb] 警告：跳过 {skipped} 个非三角面");
    }
    out
}

/// 将网格保存为 GLB 文件。
pub fn save_glb<P: AsRef<Path>>(mesh: &MeshStorage, path: P) -> Result<(), GltfError> {
    let bytes = format_glb(mesh);
    fs::write(path, bytes)?;
    Ok(())
}

// --- 极简 JSON 解析（仅支持 GLB 中所需的对象/数组/原始值） ---

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

/// 极简 JSON 解析器。支持 object / array / string / number / bool / null。
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
        return Err(GltfError::Parse("JSON 意外结束".into()));
    }
    match bytes[*pos] {
        b'{' => parse_object(bytes, pos),
        b'[' => parse_array(bytes, pos),
        b'"' => parse_string(bytes, pos).map(JsonValue::String),
        b't' | b'f' => parse_bool(bytes, pos).map(JsonValue::Bool),
        b'n' => parse_null(bytes, pos).map(|_| JsonValue::Null),
        b'-' | b'0'..=b'9' => parse_number(bytes, pos).map(JsonValue::Number),
        c => Err(GltfError::Parse(format!(
            "位置 {} 处出现意外 JSON 字符 '{}'",
            pos, c as char
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
            return Err(GltfError::Parse("对象中应为字符串键".into()));
        }
        let key = parse_string(bytes, pos)?;
        skip_ws(bytes, pos);
        if *pos >= bytes.len() || bytes[*pos] != b':' {
            return Err(GltfError::Parse("键后应为 ':'".into()));
        }
        *pos += 1;
        let val = parse_value(bytes, pos)?;
        entries.push((key, val));
        skip_ws(bytes, pos);
        if *pos >= bytes.len() {
            return Err(GltfError::Parse("对象意外结束".into()));
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
                    "对象中应为 ',' 或 '}}'，实际为 '{}'",
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
            return Err(GltfError::Parse("数组意外结束".into()));
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
                    "数组中应为 ',' 或 ']'，实际为 '{}'",
                    c as char
                )));
            }
        }
    }
}

fn parse_string(bytes: &[u8], pos: &mut usize) -> Result<String, GltfError> {
    if *pos >= bytes.len() || bytes[*pos] != b'"' {
        return Err(GltfError::Parse("应为 '\"'".into()));
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
                    return Err(GltfError::Parse("不完整的转义序列".into()));
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
    Err(GltfError::Parse("字符串未终止".into()))
}

fn parse_bool(bytes: &[u8], pos: &mut usize) -> Result<bool, GltfError> {
    if bytes[*pos..].starts_with(b"true") {
        *pos += 4;
        Ok(true)
    } else if bytes[*pos..].starts_with(b"false") {
        *pos += 5;
        Ok(false)
    } else {
        Err(GltfError::Parse("无效 bool 值".into()))
    }
}

fn parse_null(bytes: &[u8], pos: &mut usize) -> Result<(), GltfError> {
    if bytes[*pos..].starts_with(b"null") {
        *pos += 4;
        Ok(())
    } else {
        Err(GltfError::Parse("无效 null 值".into()))
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
        .map_err(|_| GltfError::Parse("无效数字".into()))?;
    s.parse::<f64>()
        .map_err(|_| GltfError::Parse(format!("无效数字: {s}")))
}

// --- GLB 文档结构 ---

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
    attributes: std::collections::HashMap<String, u32>,
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
            .ok_or_else(|| GltfError::Parse("缺少 bufferViews".into()))?;
        let accs_json = json
            .get("accessors")
            .and_then(|v| v.as_array())
            .ok_or_else(|| GltfError::Parse("缺少 accessors".into()))?;
        let meshes_json = json
            .get("meshes")
            .and_then(|v| v.as_array())
            .ok_or_else(|| GltfError::Parse("缺少 meshes".into()))?;

        let mut buffer_views: Vec<GltfBufferView> = Vec::with_capacity(bvs_json.len());
        for bv in bvs_json {
            buffer_views.push(GltfBufferView {
                buffer: bv.get("buffer").and_then(|v| v.as_u32()).unwrap_or(0),
                byte_offset: bv.get("byteOffset").and_then(|v| v.as_u32()).unwrap_or(0),
                byte_length: bv
                    .get("byteLength")
                    .and_then(|v| v.as_u32())
                    .ok_or_else(|| GltfError::Parse("bufferView 缺少 byteLength".into()))?,
            });
        }

        let mut accessors: Vec<GltfAccessor> = Vec::with_capacity(accs_json.len());
        for acc in accs_json {
            accessors.push(GltfAccessor {
                buffer_view: acc
                    .get("bufferView")
                    .and_then(|v| v.as_u32())
                    .ok_or_else(|| GltfError::Parse("accessor 缺少 bufferView".into()))?,
                component_type: acc
                    .get("componentType")
                    .and_then(|v| v.as_u32())
                    .ok_or_else(|| GltfError::Parse("accessor 缺少 componentType".into()))?,
                count: acc
                    .get("count")
                    .and_then(|v| v.as_u32())
                    .ok_or_else(|| GltfError::Parse("accessor 缺少 count".into()))?,
                byte_offset: acc.get("byteOffset").and_then(|v| v.as_u32()).unwrap_or(0),
            });
        }

        let mut meshes: Vec<GltfMesh> = Vec::with_capacity(meshes_json.len());
        for m in meshes_json {
            let prims_json = m
                .get("primitives")
                .and_then(|v| v.as_array())
                .ok_or_else(|| GltfError::Parse("mesh 缺少 primitives".into()))?;
            let mut primitives: Vec<GltfPrimitive> = Vec::with_capacity(prims_json.len());
            for p in prims_json {
                let mut attributes: std::collections::HashMap<String, u32> =
                    std::collections::HashMap::new();
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
        .ok_or_else(|| GltfError::Parse(format!("accessor {acc_idx} 越界")))?;
    if acc.component_type != 5126 {
        return Err(GltfError::Unsupported(format!(
            "不支持 POSITION componentType {}（仅支持 5126 FLOAT）",
            acc.component_type
        )));
    }
    let bv = doc
        .buffer_views
        .get(acc.buffer_view as usize)
        .ok_or_else(|| GltfError::Parse(format!("bufferView {} 越界", acc.buffer_view)))?;
    let start = (bv.byte_offset + acc.byte_offset) as usize;
    let end = start + (acc.count as usize) * 12;
    if end > bin.len() {
        return Err(GltfError::Parse("accessor 超出 bufferView 范围".into()));
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
        .ok_or_else(|| GltfError::Parse(format!("accessor {acc_idx} 越界")))?;
    let elem_size: usize = match acc.component_type {
        5121 => 1, // UBYTE
        5123 => 2, // USHORT
        5125 => 4, // UINT
        other => {
            return Err(GltfError::Unsupported(format!(
                "不支持 indices componentType {other}"
            )));
        }
    };
    let bv = doc
        .buffer_views
        .get(acc.buffer_view as usize)
        .ok_or_else(|| GltfError::Parse(format!("bufferView {} 越界", acc.buffer_view)))?;
    let start = (bv.byte_offset + acc.byte_offset) as usize;
    let end = start + (acc.count as usize) * elem_size;
    if end > bin.len() {
        return Err(GltfError::Parse("accessor 超出 bufferView 范围".into()));
    }
    let mut out: Vec<u32> = Vec::with_capacity(acc.count as usize);
    let mut off = start;
    for _ in 0..acc.count {
        let v: u32 = match acc.component_type {
            5121 => bin[off] as u32,
            5123 => u16::from_le_bytes([bin[off], bin[off + 1]]) as u32,
            5125 => u32::from_le_bytes([bin[off], bin[off + 1], bin[off + 2], bin[off + 3]]),
            _ => unreachable!(),
        };
        out.push(v);
        off += elem_size;
    }
    Ok(out)
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
    Off(OffError),
    Gltf(GltfError),
    UnsupportedFormat(String),
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Obj(e) => write!(f, "OBJ 错误: {e}"),
            Self::Ply(e) => write!(f, "PLY 错误: {e}"),
            Self::Stl(e) => write!(f, "STL 错误: {e}"),
            Self::Off(e) => write!(f, "OFF 错误: {e}"),
            Self::Gltf(e) => write!(f, "glTF 错误: {e}"),
            Self::UnsupportedFormat(s) => write!(f, "不支持的格式: {s}"),
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

/// 自动检测文件格式并加载网格。
///
/// 根据文件扩展名选择解析器：
/// - `.obj` → OBJ（支持三角面与 n-gon）
/// - `.ply` → PLY（自动判别 ASCII / 二进制小端）
/// - `.stl` → STL（自动判别 ASCII / 二进制）
/// - `.off` → OFF（ASCII）
/// - `.glb` / `.gltf` → glTF GLB（最小子集）
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

/// 自动检测文件格式并保存网格。
///
/// 根据文件扩展名选择序列化器：
/// - `.obj` → OBJ（ASCII）
/// - `.ply` → PLY（ASCII；若需二进制请直接调用 `save_ply_binary`）
/// - `.stl` → STL（ASCII；若需二进制请直接调用 `save_stl_binary`）
/// - `.off` → OFF（ASCII）
/// - `.glb` / `.gltf` → glTF GLB（二进制）
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
        let path = std::env::temp_dir().join("halfedge_autodetect.unknown");
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

    // ---------- PLY 二进制测试 ----------

    #[test]
    fn ply_binary_roundtrip() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces);
        let bytes = format_ply_binary(&mesh);
        let parsed = parse_ply_bytes(&bytes).expect("PLY 二进制往返解析失败");
        assert_eq!(parsed.vertex_count(), mesh.vertex_count());
        assert_eq!(parsed.face_count(), mesh.face_count());
        assert!(check_topology(&parsed).is_ok());
    }

    #[test]
    fn ply_binary_file_roundtrip() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces);
        let path = std::env::temp_dir().join("halfedge_ply_bin.ply");
        save_ply_binary(&mesh, &path).expect("保存二进制 PLY 失败");
        let loaded = load_ply(&path).expect("加载二进制 PLY 失败");
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.vertex_count(), mesh.vertex_count());
        assert_eq!(loaded.face_count(), mesh.face_count());
    }

    #[test]
    fn ply_ascii_still_works_via_parse_ply_bytes() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces);
        let text = format_ply(&mesh);
        let bytes = text.into_bytes();
        let parsed = parse_ply_bytes(&bytes).expect("PLY ASCII via bytes 解析失败");
        assert_eq!(parsed.vertex_count(), mesh.vertex_count());
        assert_eq!(parsed.face_count(), mesh.face_count());
    }

    #[test]
    fn ply_binary_detects_bad_header() {
        // 缺少 end_header 应失败
        let bytes = b"ply\nformat binary_little_endian 1.0\nelement vertex 0\n";
        let err = parse_ply_bytes(bytes).unwrap_err();
        assert!(matches!(err, PlyError::Parse { .. }));
    }

    // ---------- OFF 测试 ----------

    #[test]
    fn off_roundtrip_tetrahedron() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces);
        let text = format_off(&mesh);
        let parsed = parse_off(&text).expect("OFF 往返解析失败");
        assert_eq!(parsed.vertex_count(), mesh.vertex_count());
        assert_eq!(parsed.face_count(), mesh.face_count());
        assert!(check_topology(&parsed).is_ok());
    }

    #[test]
    fn off_file_roundtrip() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces);
        let path = std::env::temp_dir().join("halfedge_off.off");
        save_off(&mesh, &path).expect("保存 OFF 失败");
        let loaded = load_off(&path).expect("加载 OFF 失败");
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.vertex_count(), mesh.vertex_count());
        assert_eq!(loaded.face_count(), mesh.face_count());
    }

    #[test]
    fn off_parse_counts_inline_with_keyword() {
        // 首行带 OFF 关键字 + 计数
        let text = "OFF 4 4 0\n\
            0 0 0\n1 0 0\n0 1 0\n0 0 1\n\
            3 0 2 1\n3 0 1 3\n3 0 3 2\n3 1 2 3\n";
        let mesh = parse_off(text).expect("OFF 行内计数解析失败");
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.face_count(), 4);
    }

    #[test]
    fn off_parse_quadrilateral_face_succeeds() {
        let text = "OFF\n4 1 0\n\
            0 0 0\n1 0 0\n1 1 0\n0 1 0\n\
            4 0 1 2 3\n";
        let mesh = parse_off(text).expect("OFF 四边形面解析失败");
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
    fn auto_detect_off() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces);
        let path = std::env::temp_dir().join("halfedge_autodetect.off");
        save_mesh(&mesh, &path).expect("save_mesh(.off) 失败");
        let loaded = load_mesh(&path).expect("load_mesh(.off) 失败");
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.face_count(), mesh.face_count());
    }

    // ---------- glTF GLB 测试 ----------

    #[test]
    fn glb_roundtrip_tetrahedron() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces);
        let bytes = format_glb(&mesh);
        let parsed = parse_glb(&bytes).expect("GLB 往返解析失败");
        assert_eq!(parsed.vertex_count(), mesh.vertex_count());
        assert_eq!(parsed.face_count(), mesh.face_count());
        assert!(check_topology(&parsed).is_ok());
    }

    #[test]
    fn glb_file_roundtrip() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces);
        let path = std::env::temp_dir().join("halfedge_glb.glb");
        save_glb(&mesh, &path).expect("保存 GLB 失败");
        let loaded = load_glb(&path).expect("加载 GLB 失败");
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
        let parsed = parse_glb(&bytes).expect("GLB icosphere 往返失败");
        assert_eq!(parsed.vertex_count(), mesh.vertex_count());
        assert_eq!(parsed.face_count(), mesh.face_count());
    }

    #[test]
    fn auto_detect_glb() {
        let (verts, faces) = make_tetra_data();
        let mesh = build_mesh_from_vertices_and_faces(&verts, &faces);
        let path = std::env::temp_dir().join("halfedge_autodetect.glb");
        save_mesh(&mesh, &path).expect("save_mesh(.glb) 失败");
        let loaded = load_mesh(&path).expect("load_mesh(.glb) 失败");
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.face_count(), mesh.face_count());
    }

    // ---------- 空集合 / 越界 / 退化面 边界测试 ----------

    #[test]
    fn build_mesh_empty_inputs_returns_empty_mesh() {
        let mesh = build_mesh_from_vertices_and_faces(&[], &[]);
        assert_eq!(mesh.vertex_count(), 0);
        assert_eq!(mesh.face_count(), 0);
    }

    #[test]
    fn build_mesh_vertices_no_faces() {
        let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mesh = build_mesh_from_vertices_and_faces(&vertices, &[]);
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.face_count(), 0);
    }

    #[test]
    fn build_polygons_empty_inputs_returns_empty_mesh() {
        let mesh = build_mesh_from_polygons(&[], &[]);
        assert_eq!(mesh.vertex_count(), 0);
        assert_eq!(mesh.face_count(), 0);
    }

    #[test]
    #[should_panic(expected = "越界")]
    fn build_mesh_face_index_out_of_range_panics() {
        let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = [[0u32, 1, 5]];
        let _ = build_mesh_from_vertices_and_faces(&vertices, &faces);
    }

    #[test]
    #[should_panic(expected = "越界")]
    fn build_polygons_face_index_out_of_range_panics() {
        let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = [vec![0u32, 1, 5]];
        let _ = build_mesh_from_polygons(&vertices, &faces);
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
        let mesh = build_mesh_from_polygons(&vertices, &faces);
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
        let mesh = build_mesh_from_polygons(&vertices, &faces);
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.face_count(), 1);
    }

    // ---------- OBJ 损坏输入测试 ----------

    #[test]
    fn obj_parse_empty_text() {
        let mesh = parse_obj("").expect("空 OBJ 应解析为空网格");
        assert_eq!(mesh.vertex_count(), 0);
        assert_eq!(mesh.face_count(), 0);
    }

    #[test]
    fn obj_parse_only_vertices_no_faces() {
        let text = "v 0 0 0\nv 1 0 0\nv 0 1 0\n";
        let mesh = parse_obj(text).expect("仅顶点 OBJ 应解析成功");
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.face_count(), 0);
    }

    #[test]
    fn obj_parse_face_zero_index_fails() {
        // OBJ 索引从 1 开始，0 无效
        let text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n";
        assert!(parse_obj(text).is_err());
    }

    #[test]
    fn obj_parse_face_index_equal_to_vertex_count_fails() {
        // 3 顶点，索引 4（1-based）→ 0-based 为 3，等于顶点数 → 越界
        let text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n";
        assert!(parse_obj(text).is_err());
    }

    // ---------- PLY 损坏输入测试 ----------

    #[test]
    fn ply_parse_empty_bytes_fails() {
        assert!(parse_ply_bytes(b"").is_err());
    }

    #[test]
    fn ply_parse_missing_end_header_fails() {
        let bytes = b"ply\nformat ascii 1.0\n";
        assert!(parse_ply_bytes(bytes).is_err());
    }

    // ---------- STL 损坏输入测试 ----------

    #[test]
    fn stl_ascii_empty_text() {
        let mesh = parse_stl_ascii("").expect("空 STL 应解析为空网格");
        assert_eq!(mesh.face_count(), 0);
    }

    #[test]
    fn stl_ascii_only_solid_endsolid() {
        let text = "solid x\nendsolid x\n";
        let mesh = parse_stl_ascii(text).expect("空 solid STL 应解析为空网格");
        assert_eq!(mesh.face_count(), 0);
    }

    // ---------- OFF 损坏输入测试 ----------

    #[test]
    fn off_parse_empty_text_fails() {
        assert!(parse_off("").is_err());
    }

    #[test]
    fn off_parse_only_vertices_no_faces() {
        let text = "OFF\n3 0 0\n0 0 0\n1 0 0\n0 1 0\n";
        let mesh = parse_off(text).expect("仅顶点 OFF 应解析成功");
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.face_count(), 0);
    }
}
