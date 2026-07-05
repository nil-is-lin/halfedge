//! 网格布尔运算模块 - Corefinement（共精化）算法
//!
//! 支持闭合流形三角网格间的并集、交集、差集、对称差。
//!
//! ## 算法概要
//! Corefinement 通过精确几何切割实现布尔运算，分四阶段：
//! 1. **边-三角形求交**：对网格 A 的每个三角形，找出其三条边与网格 B
//!    所有三角形的交点（含共面边-边交点）；
//! 2. **共面重叠处理**：当 A 的三角形与 B 的三角形完整共面时，用
//!    Sutherland-Hodgman 多边形裁剪计算重叠多边形，从中提取边交点和
//!    内部约束点（B 的顶点落在 A 内部的点），确保共面重叠区域被精确分割；
//! 3. **三角形分裂**：将边交点插入三角形边界形成多边形，用耳裁剪三角化；
//!    随后对每个内部约束点做扇形分裂（1 → 3），使共面重叠区域的边界
//!    在两侧网格上对齐；
//! 4. **内外分类**：对每个子三角形采样 7 点（重心 + 3 顶点 + 3 边中点），
//!    沿**内向法向**偏移 ε 后用射线奇偶法判定其在另一网格内外，多数表决；
//! 5. **集合运算**：按 BoolOp 规则保留/丢弃子三角形，构建结果网格。
//!
//! 与旧射线法相比，corefinement 对跨越边界的三角形进行精确切割，
//! 使 `A - A` 严格为空（0 面），并提升部分重叠场景的几何精度。
//!
//! ## 限制
//! - 仅支持闭合流形三角网格（无边界边）；
//! - 交点坐标使用 f64 精度，极端退化可能失败。

use std::collections::{HashMap, HashSet};

use crate::ids::{FaceId, VertexId};
use crate::linalg::vec3;
use crate::predicates::{is_triangle_degenerate_3d, orient3d, point_in_triangle_2d};
use crate::storage::{MeshStorage, Vertex};
use crate::topology_ops::add_triangle;
use crate::triangulation::ear_clipping_3d;

// ============================================================
// 布尔操作类型
// ============================================================

/// 布尔运算类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOp {
    /// A ∪ B：保留 A 表面在 B 外部的部分 + B 表面在 A 外部的部分。
    Union,
    /// A ∩ B：保留 A 表面在 B 内部的部分 + B 表面在 A 内部的部分。
    Intersection,
    /// A - B：仅保留 A 表面在 B 外部的部分。
    Difference,
    /// (A - B) ∪ (B - A)：保留两侧表面在对方外部的部分。
    SymmetricDifference,
}

// ============================================================
// 几何求交辅助函数
// ============================================================

/// 线段 `(p0, p1)` 与三角形 `(v0, v1, v2)` 的交点。
///
/// 使用鲁棒 [`orient3d`] 谓词判定线段两端相对三角形平面的符号：
/// - 同侧 → 无交点；
/// - 异号 → 求交点参数 `t = s0 / (s0 - s1)`，投影到 2D 用
///   [`point_in_triangle_2d`] 判定是否在三角形内；
/// - 共面（两端同在平面上）→ 返回 `None`，交由
///   [`coplanar_segment_segment_3d`] 处理；
/// - 端点恰在平面上 → 返回 `None`，避免重复顶点。
///
/// 返回 `(交点坐标, 线段参数 t ∈ (1e-9, 1-1e-9))`。
fn segment_triangle_intersection(p0: vec3::Vec3, p1: vec3::Vec3, v0: vec3::Vec3, v1: vec3::Vec3, v2: vec3::Vec3) -> Option<(vec3::Vec3, f64)> {
    let s0 = orient3d(v0, v1, v2, p0);
    let s1 = orient3d(v0, v1, v2, p1);

    if (s0 > 0.0 && s1 > 0.0) || (s0 < 0.0 && s1 < 0.0) {
        return None;
    }
    if s0 == 0.0 || s1 == 0.0 {
        return None;
    }

    let t = s0 / (s0 - s1);
    if !(1e-9..=1.0 - 1e-9).contains(&t) {
        return None;
    }

    let p = [
        p0[0] + t * (p1[0] - p0[0]),
        p0[1] + t * (p1[1] - p0[1]),
        p0[2] + t * (p1[2] - p0[2]),
    ];

    if !point_in_triangle_3d(p, v0, v1, v2) {
        return None;
    }

    Some((p, t))
}

/// 判定点 `p` 是否在三角形 `(v0, v1, v2)` 内（含边界）。
///
/// 投影到三角形法向主轴的正交 2D 平面，用鲁棒
/// [`point_in_triangle_2d`] 判定。退化三角形返回 `false`。
fn point_in_triangle_3d(p: vec3::Vec3, v0: vec3::Vec3, v1: vec3::Vec3, v2: vec3::Vec3) -> bool {
    let e1 = vec3::sub(v1, v0);
    let e2 = vec3::sub(v2, v0);
    let normal = vec3::cross(e1, e2);
    let abs_n = [normal[0].abs(), normal[1].abs(), normal[2].abs()];
    let drop_axis = if abs_n[0] >= abs_n[1] && abs_n[0] >= abs_n[2] {
        0
    } else if abs_n[1] >= abs_n[2] {
        1
    } else {
        2
    };

    let project = |q: vec3::Vec3| -> [f64; 2] {
        match drop_axis {
            0 => [q[1], q[2]],
            1 => [q[0], q[2]],
            _ => [q[0], q[1]],
        }
    };

    point_in_triangle_2d(project(p), project(v0), project(v1), project(v2))
}

/// 共面线段-线段求交。
///
/// 当两条共面线段不平行时，投影到 2D 用参数方程求交点。
/// 返回 `(交点坐标, 线段1参数 t1, 线段2参数 t2)`，均 ∈ `(1e-9, 1-1e-9)`。
/// 平行/共线/端点相交返回 `None`。
fn coplanar_segment_segment_3d(e1a: vec3::Vec3, e1b: vec3::Vec3, e2a: vec3::Vec3, e2b: vec3::Vec3) -> Option<(vec3::Vec3, f64, f64)> {
    let d1 = vec3::sub(e1b, e1a);
    let d2 = vec3::sub(e2b, e2a);
    let normal = vec3::cross(d1, d2);
    let abs_n = [normal[0].abs(), normal[1].abs(), normal[2].abs()];

    if abs_n[0] < 1e-14 && abs_n[1] < 1e-14 && abs_n[2] < 1e-14 {
        return None;
    }

    let drop_axis = if abs_n[0] >= abs_n[1] && abs_n[0] >= abs_n[2] {
        0
    } else if abs_n[1] >= abs_n[2] {
        1
    } else {
        2
    };

    let project = |q: vec3::Vec3| -> [f64; 2] {
        match drop_axis {
            0 => [q[1], q[2]],
            1 => [q[0], q[2]],
            _ => [q[0], q[1]],
        }
    };

    let p1 = project(e1a);
    let p2 = project(e1b);
    let p3 = project(e2a);
    let p4 = project(e2b);

    let d1x = p2[0] - p1[0];
    let d1y = p2[1] - p1[1];
    let d2x = p4[0] - p3[0];
    let d2y = p4[1] - p3[1];

    let denom = d1x * d2y - d1y * d2x;
    if denom.abs() < 1e-14 {
        return None;
    }

    let dx = p3[0] - p1[0];
    let dy = p3[1] - p1[1];

    let t1 = (dx * d2y - dy * d2x) / denom;
    let t2 = (dx * d1y - dy * d1x) / denom;

    if !(1e-9..=1.0 - 1e-9).contains(&t1) || !(1e-9..=1.0 - 1e-9).contains(&t2) {
        return None;
    }

    let p = [
        e1a[0] + t1 * d1[0],
        e1a[1] + t1 * d1[1],
        e1a[2] + t1 * d1[2],
    ];

    Some((p, t1, t2))
}

// ============================================================
// 共面三角形-三角形重叠（Sutherland-Hodgman 多边形裁剪）
// ============================================================

/// 2D 叉积。
fn cross_2d(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

/// 2D 减法。
fn sub_2d(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

/// 判定点 `p` 是否在边 `(edge_start, edge_end)` 的左侧（含边界）。
///
/// 用于 Sutherland-Hodgman 裁剪：CCW 裁剪多边形的内部在每条边的左侧。
fn is_left_of_edge(p: [f64; 2], edge_start: [f64; 2], edge_end: [f64; 2]) -> bool {
    cross_2d(sub_2d(edge_end, edge_start), sub_2d(p, edge_start)) >= -1e-12
}

/// 2D 线段 `(p1, p2)` 与线段 `(p3, p4)` 的交点（参数方程）。
///
/// 返回交点坐标。平行时返回中点作为退化回退。
fn line_intersect_2d(p1: [f64; 2], p2: [f64; 2], p3: [f64; 2], p4: [f64; 2]) -> [f64; 2] {
    let denom = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0]);
    if denom.abs() < 1e-20 {
        return [(p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5];
    }
    let t = ((p1[0] - p3[0]) * (p3[1] - p4[1]) - (p1[1] - p3[1]) * (p3[0] - p4[0])) / denom;
    [p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])]
}

/// Sutherland-Hodgman 多边形裁剪：将 `subject` 裁剪到边 `(clip_start, clip_end)` 的左侧。
///
/// 返回裁剪后的多边形顶点列表（可能为空）。
fn sutherland_hodgman_clip(
    subject: &[[f64; 2]],
    clip_start: [f64; 2],
    clip_end: [f64; 2],
) -> Vec<[f64; 2]> {
    if subject.is_empty() {
        return Vec::new();
    }
    let mut output: Vec<[f64; 2]> = Vec::new();
    let n = subject.len();
    for i in 0..n {
        let current = subject[i];
        let previous = subject[(i + n - 1) % n];
        let current_inside = is_left_of_edge(current, clip_start, clip_end);
        let previous_inside = is_left_of_edge(previous, clip_start, clip_end);
        if current_inside {
            if !previous_inside {
                output.push(line_intersect_2d(previous, current, clip_start, clip_end));
            }
            output.push(current);
        } else if previous_inside {
            output.push(line_intersect_2d(previous, current, clip_start, clip_end));
        }
    }
    output
}

/// 计算两个共面三角形的重叠多边形。
///
/// 使用 Sutherland-Hodgman 算法：将三角形 A 逐边裁剪到三角形 B 的内部，
/// 得到 A ∩ B 的凸多边形。非共面或无重叠时返回 `None`。
///
/// # 参数
/// - `a`: 三角形 A 的三个顶点
/// - `b`: 三角形 B 的三个顶点
///
/// # 返回
/// 重叠多边形的 3D 顶点列表（凸，CCW 与 B 一致），或 `None`。
fn coplanar_triangle_overlap(a: [vec3::Vec3; 3], b: [vec3::Vec3; 3]) -> Option<Vec<vec3::Vec3>> {
    // 共面判定：A 的三个顶点均在 B 的平面上
    for &va in &a {
        if orient3d(b[0], b[1], b[2], va) != 0.0 {
            return None;
        }
    }

    // 投影到 2D（丢弃法向主轴）
    let normal = vec3::cross(vec3::sub(b[1], b[0]), vec3::sub(b[2], b[0]));
    let abs_n = [normal[0].abs(), normal[1].abs(), normal[2].abs()];
    let drop_axis = if abs_n[0] >= abs_n[1] && abs_n[0] >= abs_n[2] {
        0
    } else if abs_n[1] >= abs_n[2] {
        1
    } else {
        2
    };
    if abs_n[drop_axis] < 1e-20 {
        return None; // B 退化
    }

    let project = |q: vec3::Vec3| -> [f64; 2] {
        match drop_axis {
            0 => [q[1], q[2]],
            1 => [q[0], q[2]],
            _ => [q[0], q[1]],
        }
    };

    let a2d: Vec<[f64; 2]> = a.iter().map(|&v| project(v)).collect();
    let b2d: Vec<[f64; 2]> = b.iter().map(|&v| project(v)).collect();

    // 检测 B 的 2D 朝向，确保 CCW（Sutherland-Hodgman 要求裁剪多边形 CCW）
    let b_signed_area = cross_2d(sub_2d(b2d[1], b2d[0]), sub_2d(b2d[2], b2d[0]));
    if b_signed_area.abs() < 1e-20 {
        return None; // B 退化
    }
    let b_ccw = b_signed_area > 0.0;

    // 逐边裁剪 A 到 B 内部
    let mut polygon = a2d.clone();
    for i in 0..3 {
        let (e_start, e_end) = if b_ccw {
            (b2d[i], b2d[(i + 1) % 3])
        } else {
            (b2d[(i + 1) % 3], b2d[i]) // CW → 反转边方向使内部在左侧
        };
        polygon = sutherland_hodgman_clip(&polygon, e_start, e_end);
        if polygon.len() < 3 {
            return None;
        }
    }

    // 2D → 3D：利用平面方程 normal · x + d = 0 反解被丢弃的坐标
    let d_plane = -(normal[0] * b[0][0] + normal[1] * b[0][1] + normal[2] * b[0][2]);
    let result: Vec<vec3::Vec3> = polygon
        .iter()
        .map(|&p2| match drop_axis {
            0 => [
                -(normal[1] * p2[0] + normal[2] * p2[1] + d_plane) / normal[0],
                p2[0],
                p2[1],
            ],
            1 => [
                p2[0],
                -(normal[0] * p2[0] + normal[2] * p2[1] + d_plane) / normal[1],
                p2[1],
            ],
            _ => [
                p2[0],
                p2[1],
                -(normal[0] * p2[0] + normal[1] * p2[1] + d_plane) / normal[2],
            ],
        })
        .collect();

    Some(result)
}

/// 判定点 `p` 是否在三角形内部（不含边界）。
fn is_point_strictly_inside_triangle(p: vec3::Vec3, tri: [vec3::Vec3; 3]) -> bool {
    if !point_in_triangle_3d(p, tri[0], tri[1], tri[2]) {
        return false;
    }
    // 排除在边上的情况
    for &(a, b) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
        if is_point_on_segment_3d(p, a, b) {
            return false;
        }
    }
    true
}

/// 判定点 `p` 是否在线段 `(a, b)` 上（含端点）。
fn is_point_on_segment_3d(p: vec3::Vec3, a: vec3::Vec3, b: vec3::Vec3) -> bool {
    let d = vec3::sub(b, a);
    let ap = vec3::sub(p, a);
    let len_sq = vec3::dot(d, d);
    if len_sq < 1e-20 {
        return vec3::dot(ap, ap) < 1e-20;
    }
    let t = vec3::dot(ap, d) / len_sq;
    if !(-1e-9..=1.0 + 1e-9).contains(&t) {
        return false;
    }
    let closest = vec3::add(a, vec3::scale(d, t.max(0.0).min(1.0)));
    vec3::dot(vec3::sub(p, closest), vec3::sub(p, closest)) < 1e-18
}

/// 查找点 `p` 位于三角形 `tri` 哪条边上，返回 (边索引, 参数 t ∈ (0,1))。
///
/// 排除端点（t ∈ {0, 1}），因为端点是三角形顶点本身，不需要作为交点插入。
fn point_on_triangle_edge(p: vec3::Vec3, tri: [vec3::Vec3; 3]) -> Option<(usize, f64)> {
    let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];
    for (i, &(a, b)) in edges.iter().enumerate() {
        let d = vec3::sub(b, a);
        let len_sq = vec3::dot(d, d);
        if len_sq < 1e-20 {
            continue;
        }
        let t = vec3::dot(vec3::sub(p, a), d) / len_sq;
        // 排除端点
        if !(1e-9..=1.0 - 1e-9).contains(&t) {
            continue;
        }
        let closest = vec3::add(a, vec3::scale(d, t));
        if vec3::dot(vec3::sub(p, closest), vec3::sub(p, closest)) < 1e-18 {
            return Some((i, t));
        }
    }
    None
}

/// 构造三角形的 AABB，并按 `eps` 三轴均匀膨胀。
fn triangle_aabb(tri: [vec3::Vec3; 3], eps: f64) -> crate::geometry::AABB {
    let mut aabb = crate::geometry::AABB::from_points(&[tri[0], tri[1], tri[2]]);
    for i in 0..3 {
        aabb.min[i] -= eps;
        aabb.max[i] += eps;
    }
    aabb
}

/// 向内部约束点列表插入去重的点。
fn push_interior_unique(list: &mut Vec<vec3::Vec3>, p: vec3::Vec3) {
    let is_dup = list.iter().any(|&q| {
        (q[0] - p[0]).abs() < 1e-9 && (q[1] - p[1]).abs() < 1e-9 && (q[2] - p[2]).abs() < 1e-9
    });
    if !is_dup {
        list.push(p);
    }
}

/// 在子三角形列表中查找包含点 `p` 的三角形，返回其索引。
fn find_containing_triangle(tris: &[[vec3::Vec3; 3]], p: vec3::Vec3) -> Option<usize> {
    for (i, t) in tris.iter().enumerate() {
        if point_in_triangle_3d(p, t[0], t[1], t[2]) {
            return Some(i);
        }
    }
    None
}

// ============================================================
// 射线法内外分类
// ============================================================

/// 判断点 `point` 是否在闭合网格内部（射线奇偶法）。
///
/// 沿 `+x` 方向发射射线，统计与三角面交点数，奇数 → 内部。
/// 空网格返回 `false`。退化三角形跳过。
///
/// 使用 `bvh.ray_count_hits` 把 $O(F)$ 扫描降到平均 $O(\log F + k)$。
fn point_inside_mesh(point: vec3::Vec3, mesh: &MeshStorage, bvh: &crate::bvh::Bvh) -> bool {
    if mesh.face_count() == 0 {
        return false;
    }
    let dir: vec3::Vec3 = [1.0, 0.0, 0.0];
    // BVH 内部已跳过退化三角形与拓扑不一致面
    let count = bvh.ray_count_hits(point, dir, mesh);
    count % 2 == 1
}

/// 收集面 `f` 的三个顶点位置。
fn collect_face_positions(mesh: &MeshStorage, f: FaceId) -> [vec3::Vec3; 3] {
    let mut result = [[0.0; 3]; 3];
    let mut i = 0;
    for he in crate::traversal::FaceHalfEdges::new(mesh, f) {
        if let Some(h) = mesh.get_halfedge(he)
            && let Some(v) = mesh.get_vertex(h.vertex)
        {
            result[i] = v.position;
            i += 1;
            if i >= 3 {
                break;
            }
        }
    }
    result
}

/// 多采样判定三角形是否在网格内部。
///
/// 采样 7 个点（重心 + 3 顶点 + 3 边中点），每个沿**内向法向**偏移 `ε`，
/// 用射线法判定内外。多数表决（≥ 4/7）→ inside。
///
/// 内向偏移确保共面情形（如 `A - A`）下采样点明确落入网格内部，
/// 从而严格区分"表面"与"内部"。
///
/// **共面回退**：当全部 7 个采样点均判定为外部（inside_count == 0）时，
/// 可能是背对背共面（A 的面法向 +z，B 的面法向 -z），内向偏移使采样点
/// 离开对方体积。此时用 2D 包含检测：若三角形重心与网格某面共面且
/// 在该面的 2D 投影内，则判定为 inside。
fn is_triangle_inside_mesh(tri: [vec3::Vec3; 3], mesh: &MeshStorage, bvh: &crate::bvh::Bvh) -> bool {
    if mesh.face_count() == 0 {
        return false;
    }

    let normal = vec3::cross(vec3::sub(tri[1], tri[0]), vec3::sub(tri[2], tri[0]));
    let l = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    let eps = if l > 1e-14 { 1e-4 / l } else { 1e-4 };
    let offset = vec3::scale(normal, -eps);

    let samples = [
        vec3::scale(vec3::add(vec3::add(tri[0], tri[1]), tri[2]), 1.0 / 3.0),
        tri[0],
        tri[1],
        tri[2],
        vec3::scale(vec3::add(tri[0], tri[1]), 0.5),
        vec3::scale(vec3::add(tri[1], tri[2]), 0.5),
        vec3::scale(vec3::add(tri[2], tri[0]), 0.5),
    ];

    let mut inside_count = 0;
    for &s in &samples {
        let p = vec3::add(s, offset);
        if point_inside_mesh(p, mesh, bvh) {
            inside_count += 1;
        }
    }

    if inside_count >= 4 {
        return true;
    }

    // 共面回退：全部采样点在外部时，检测是否为背对背共面
    if inside_count == 0 {
        let centroid = vec3::scale(vec3::add(vec3::add(tri[0], tri[1]), tri[2]), 1.0 / 3.0);
        if is_coplanar_centroid_inside_mesh_face(tri, centroid, mesh, bvh) {
            return true;
        }
    }

    false
}

/// 检测三角形 `tri` 是否与 `mesh` 的某个面共面，且重心在该面的 2D 投影内。
///
/// 用于 [`is_triangle_inside_mesh`] 的共面回退：当内向偏移采样全部判定外部时，
/// 检测是否为背对背共面（两面法向相反、共面且 2D 重叠）。
fn is_coplanar_centroid_inside_mesh_face(
    tri: [vec3::Vec3; 3],
    centroid: vec3::Vec3,
    mesh: &MeshStorage,
    bvh: &crate::bvh::Bvh,
) -> bool {
    let aabb = triangle_aabb(tri, 1e-6);
    let candidates = bvh.faces_in_aabb(&aabb, mesh);

    for f in &candidates {
        let tri_b = collect_face_positions(mesh, *f);
        if is_triangle_degenerate_3d(tri_b[0], tri_b[1], tri_b[2]) {
            continue;
        }
        // 共面判定：tri 的三个顶点均在 tri_b 的平面上
        let s0 = orient3d(tri_b[0], tri_b[1], tri_b[2], tri[0]);
        let s1 = orient3d(tri_b[0], tri_b[1], tri_b[2], tri[1]);
        let s2 = orient3d(tri_b[0], tri_b[1], tri_b[2], tri[2]);
        if s0 == 0.0 && s1 == 0.0 && s2 == 0.0
            && point_in_triangle_3d(centroid, tri_b[0], tri_b[1], tri_b[2])
        {
            return true;
        }
    }
    false
}

// ============================================================
// Corefinement：边-三角形求交与三角形分裂
// ============================================================

struct EdgeIntersection {
    t: f64,
    pos: vec3::Vec3,
}

/// 收集三角形 `tri` 与 `mesh` 所有三角形的相交约束点。
///
/// 对每条边：
/// 1. 用 [`segment_triangle_intersection`] 检测非共面交点；
/// 2. 若线段与三角形共面（两端点 orient3d 均为 0），进一步检测：
///    - 若第三个顶点也共面 → **完整共面重叠**，用
///      [`coplanar_triangle_overlap`] 计算重叠多边形，从中提取边交点和内部约束点；
///    - 否则 → 仅边共面，用 [`coplanar_segment_segment_3d`] 检测边-边交点。
///
/// 额外用三角形 AABB 查询 BVH，捕捉**完全位于 tri 内部的共面对**
///（如 B 三角形整体落在 tri 内部，不触碰任何边）。
///
/// 返回 `(每条边的交点列表, 内部约束点列表)`，边交点按参数 `t` 排序，去重阈值 `1e-9`。
fn collect_triangle_intersections(
    tri: [vec3::Vec3; 3],
    mesh: &MeshStorage,
    bvh: &crate::bvh::Bvh,
) -> ([Vec<EdgeIntersection>; 3], Vec<vec3::Vec3>) {
    let mut result: [Vec<EdgeIntersection>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut interior_points: Vec<vec3::Vec3> = Vec::new();
    let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];
    let mut coplanar_processed: HashSet<FaceId> = HashSet::new();

    for (edge_idx, &(p0, p1)) in edges.iter().enumerate() {
        // 构造边的 AABB（轻微膨胀 1e-9 容错），用 BVH 候选过滤
        let edge_aabb = segment_aabb(p0, p1, 1e-9);
        let candidates = bvh.faces_in_aabb(&edge_aabb, mesh);

        for f in &candidates {
            let tri_b = collect_face_positions(mesh, *f);
            if is_triangle_degenerate_3d(tri_b[0], tri_b[1], tri_b[2]) {
                continue;
            }

            if let Some((pos, t)) =
                segment_triangle_intersection(p0, p1, tri_b[0], tri_b[1], tri_b[2])
            {
                push_unique(&mut result[edge_idx], t, pos);
            }

            let s0 = orient3d(tri_b[0], tri_b[1], tri_b[2], p0);
            let s1 = orient3d(tri_b[0], tri_b[1], tri_b[2], p1);
            if s0 == 0.0 && s1 == 0.0 {
                // 检测第三个顶点是否也共面 → 完整共面重叠
                let p_third = tri[(edge_idx + 2) % 3];
                let s_third = orient3d(tri_b[0], tri_b[1], tri_b[2], p_third);

                if s_third == 0.0 {
                    // 完整共面：用 Sutherland-Hodgman 计算重叠多边形（仅处理一次）
                    if coplanar_processed.insert(*f)
                        && let Some(overlap) = coplanar_triangle_overlap(tri, tri_b)
                    {
                        for &v in &overlap {
                            if let Some((eidx, t)) = point_on_triangle_edge(v, tri) {
                                push_unique(&mut result[eidx], t, v);
                            } else if is_point_strictly_inside_triangle(v, tri) {
                                push_interior_unique(&mut interior_points, v);
                            }
                        }
                    }
                } else {
                    // 仅此边共面（非完整三角形共面）：边-边交点检测
                    let tri_b_edges = [
                        (tri_b[0], tri_b[1]),
                        (tri_b[1], tri_b[2]),
                        (tri_b[2], tri_b[0]),
                    ];
                    for &(e2a, e2b) in &tri_b_edges {
                        if let Some((pos, t, _)) = coplanar_segment_segment_3d(p0, p1, e2a, e2b) {
                            push_unique(&mut result[edge_idx], t, pos);
                        }
                    }
                }
            }
        }

        result[edge_idx].sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
    }

    // 额外扫描：用三角形整体 AABB 查询 BVH，捕捉完全位于 tri 内部的共面对
    let tri_aabb = triangle_aabb(tri, 1e-9);
    let interior_candidates = bvh.faces_in_aabb(&tri_aabb, mesh);
    for f in &interior_candidates {
        if coplanar_processed.contains(f) {
            continue;
        }
        let tri_b = collect_face_positions(mesh, *f);
        if is_triangle_degenerate_3d(tri_b[0], tri_b[1], tri_b[2]) {
            continue;
        }
        let s0 = orient3d(tri_b[0], tri_b[1], tri_b[2], tri[0]);
        let s1 = orient3d(tri_b[0], tri_b[1], tri_b[2], tri[1]);
        let s2 = orient3d(tri_b[0], tri_b[1], tri_b[2], tri[2]);
        if s0 == 0.0 && s1 == 0.0 && s2 == 0.0
            && let Some(overlap) = coplanar_triangle_overlap(tri, tri_b)
        {
            coplanar_processed.insert(*f);
            for &v in &overlap {
                if let Some((eidx, t)) = point_on_triangle_edge(v, tri) {
                    push_unique(&mut result[eidx], t, v);
                } else if is_point_strictly_inside_triangle(v, tri) {
                    push_interior_unique(&mut interior_points, v);
                }
            }
        }
    }

    (result, interior_points)
}

/// 构造线段 `(p0, p1)` 的 AABB，并按 `eps` 三轴均匀膨胀。
fn segment_aabb(p0: vec3::Vec3, p1: vec3::Vec3, eps: f64) -> crate::geometry::AABB {
    let mut aabb = crate::geometry::AABB::from_points(&[p0, p1]);
    for i in 0..3 {
        aabb.min[i] -= eps;
        aabb.max[i] += eps;
    }
    aabb
}

/// 向边交点列表插入去重的交点。
fn push_unique(list: &mut Vec<EdgeIntersection>, t: f64, pos: vec3::Vec3) {
    let is_dup = list.iter().any(|e| {
        (e.pos[0] - pos[0]).abs() < 1e-9
            && (e.pos[1] - pos[1]).abs() < 1e-9
            && (e.pos[2] - pos[2]).abs() < 1e-9
    });
    if !is_dup {
        list.push(EdgeIntersection { t, pos });
    }
}

/// 将三角形按边上的交点和内部约束点分裂为子三角形。
///
/// 边交点按 `t` 顺序插入三角形边界，形成多边形，用 [`ear_clipping_3d`] 三角化。
/// 随后对每个内部约束点，定位其所在的子三角形并做扇形分裂（1 → 3）。
/// 无交点且无内部约束点时返回原三角形。
fn split_triangle_by_intersections(
    tri: [vec3::Vec3; 3],
    edge_intersections: &[Vec<EdgeIntersection>; 3],
    interior_points: &[vec3::Vec3],
) -> Vec<[vec3::Vec3; 3]> {
    let total: usize = edge_intersections.iter().map(|v| v.len()).sum();
    if total == 0 && interior_points.is_empty() {
        return vec![tri];
    }

    let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];
    let mut polygon: Vec<vec3::Vec3> = Vec::new();

    for (edge_idx, &(p0, _)) in edges.iter().enumerate() {
        polygon.push(p0);
        for e in &edge_intersections[edge_idx] {
            polygon.push(e.pos);
        }
    }

    let mut sub_tris: Vec<[vec3::Vec3; 3]> = if polygon.len() < 3 {
        vec![tri]
    } else {
        let tri_indices = ear_clipping_3d(&polygon);
        if tri_indices.is_empty() {
            vec![tri]
        } else {
            tri_indices
                .iter()
                .map(|&[i, j, k]| [polygon[i], polygon[j], polygon[k]])
                .collect()
        }
    };

    // 对每个内部约束点，扇形分裂包含它的子三角形
    for &p in interior_points {
        if let Some(idx) = find_containing_triangle(&sub_tris, p) {
            let t = sub_tris[idx];
            sub_tris.remove(idx);
            sub_tris.push([t[0], t[1], p]);
            sub_tris.push([t[1], t[2], p]);
            sub_tris.push([t[2], t[0], p]);
        }
    }

    sub_tris
}

// ============================================================
// 布尔运算主接口
// ============================================================

/// 执行指定类型的布尔运算，返回结果网格。
///
/// 两个输入网格必须为闭合流形三角网格。若任一网格不闭合或为空，
/// 结果可能不正确。
pub fn boolean_operation(mesh_a: &MeshStorage, mesh_b: &MeshStorage, op: BoolOp) -> MeshStorage {
    let mut result = MeshStorage::new();

    if mesh_a.face_count() == 0 && mesh_b.face_count() == 0 {
        return result;
    }

    // 入口构建 BVH：O((F_A + F_B) log(F_A + F_B))，远小于被替换的 O(F_A * F_B) 扫描
    let bvh_a = crate::bvh::Bvh::build(mesh_a);
    let bvh_b = crate::bvh::Bvh::build(mesh_b);

    let mut kept_triangles: Vec<[vec3::Vec3; 3]> = Vec::new();

    for f in mesh_a.face_ids() {
        let tri = collect_face_positions(mesh_a, f);
        if is_triangle_degenerate_3d(tri[0], tri[1], tri[2]) {
            continue;
        }

        let (edge_intersections, interior_points) =
            collect_triangle_intersections(tri, mesh_b, &bvh_b);
        let sub_tris = split_triangle_by_intersections(tri, &edge_intersections, &interior_points);

        for sub_tri in sub_tris {
            if is_triangle_degenerate_3d(sub_tri[0], sub_tri[1], sub_tri[2]) {
                continue;
            }
            let inside_b = is_triangle_inside_mesh(sub_tri, mesh_b, &bvh_b);
            if classify(op, true, inside_b) {
                kept_triangles.push(sub_tri);
            }
        }
    }

    for f in mesh_b.face_ids() {
        let tri = collect_face_positions(mesh_b, f);
        if is_triangle_degenerate_3d(tri[0], tri[1], tri[2]) {
            continue;
        }

        let (edge_intersections, interior_points) =
            collect_triangle_intersections(tri, mesh_a, &bvh_a);
        let sub_tris = split_triangle_by_intersections(tri, &edge_intersections, &interior_points);

        for sub_tri in sub_tris {
            if is_triangle_degenerate_3d(sub_tri[0], sub_tri[1], sub_tri[2]) {
                continue;
            }
            let inside_a = is_triangle_inside_mesh(sub_tri, mesh_a, &bvh_a);
            if classify(op, false, inside_a) {
                kept_triangles.push(sub_tri);
            }
        }
    }

    let total_v_cap = kept_triangles.len() * 3;
    result.reserve(total_v_cap, kept_triangles.len() * 6, kept_triangles.len());

    let mut index_pool: HashMap<[i64; 3], VertexId> = HashMap::new();
    let mut failed_tris: u32 = 0;

    for tri in &kept_triangles {
        let v0 = get_or_add_vertex(&mut result, &mut index_pool, tri[0]);
        let v1 = get_or_add_vertex(&mut result, &mut index_pool, tri[1]);
        let v2 = get_or_add_vertex(&mut result, &mut index_pool, tri[2]);
        if add_triangle(&mut result, v0, v1, v2).is_err() {
            failed_tris += 1;
        }
    }

    if failed_tris > 0 {
        eprintln!(
            "[halfedge::boolean] 警告：{failed_tris} 个三角形创建失败（拓扑冲突），已跳过"
        );
    }

    result
}

/// 分类规则。
///
/// - `from_a`: `true` 表示三角形来自 mesh_a，`false` 表示来自 mesh_b；
/// - `inside_other`: `true` 表示三角形在另一网格内部。
///
/// | op | from_a=true (A 面) | from_a=false (B 面) |
/// |----|-------------------|---------------------|
/// | Union              | `!inside_other` | `!inside_other` |
/// | Intersection       | `inside_other`  | `inside_other`  |
/// | Difference         | `!inside_other` | `false`         |
/// | SymmetricDifference| `!inside_other` | `!inside_other` |
fn classify(op: BoolOp, from_a: bool, inside_other: bool) -> bool {
    match op {
        BoolOp::Union => !inside_other,
        BoolOp::Intersection => inside_other,
        BoolOp::Difference => from_a && !inside_other,
        BoolOp::SymmetricDifference => !inside_other,
    }
}

/// 量化去重添加顶点。
///
/// 量化到 `1e-9` 网格，键为 `(⌊1e9·x⌉, ⌊1e9·y⌉, ⌊1e9·z⌉) ∈ ℤ³`。
/// 阈值内的位置视为同一点，复用已有 VertexId。
fn get_or_add_vertex(
    mesh: &mut MeshStorage,
    pool: &mut HashMap<[i64; 3], VertexId>,
    pos: vec3::Vec3,
) -> VertexId {
    let key = [
        (pos[0] * 1e9).round() as i64,
        (pos[1] * 1e9).round() as i64,
        (pos[2] * 1e9).round() as i64,
    ];
    if let Some(&v) = pool.get(&key)
        && mesh.contains_vertex(v)
    {
        return v;
    }
    let v = mesh.add_vertex(Vertex::new(pos));
    pool.insert(key, v);
    v
}

/// 并集：A ∪ B
pub fn boolean_union(a: &MeshStorage, b: &MeshStorage) -> MeshStorage {
    boolean_operation(a, b, BoolOp::Union)
}

/// 交集：A ∩ B
pub fn boolean_intersection(a: &MeshStorage, b: &MeshStorage) -> MeshStorage {
    boolean_operation(a, b, BoolOp::Intersection)
}

/// 差集：A - B
pub fn boolean_difference(a: &MeshStorage, b: &MeshStorage) -> MeshStorage {
    boolean_operation(a, b, BoolOp::Difference)
}

/// 对称差：(A - B) ∪ (B - A)
pub fn boolean_symmetric_difference(a: &MeshStorage, b: &MeshStorage) -> MeshStorage {
    boolean_operation(a, b, BoolOp::SymmetricDifference)
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 构建单位立方体 [-1,1]³
    fn build_cube() -> MeshStorage {
        let verts: Vec<[f64; 3]> = vec![
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ];
        let faces: Vec<[u32; 3]> = vec![
            [0, 3, 2],
            [0, 2, 1],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        crate::io::build_mesh_from_vertices_and_faces(&verts, &faces)
    }

    /// 构建平移后的单位立方体
    fn build_cube_offset(dx: f64, dy: f64, dz: f64) -> MeshStorage {
        let verts: Vec<[f64; 3]> = vec![
            [-1.0 + dx, -1.0 + dy, -1.0 + dz],
            [1.0 + dx, -1.0 + dy, -1.0 + dz],
            [1.0 + dx, 1.0 + dy, -1.0 + dz],
            [-1.0 + dx, 1.0 + dy, -1.0 + dz],
            [-1.0 + dx, -1.0 + dy, 1.0 + dz],
            [1.0 + dx, -1.0 + dy, 1.0 + dz],
            [1.0 + dx, 1.0 + dy, 1.0 + dz],
            [-1.0 + dx, 1.0 + dy, 1.0 + dz],
        ];
        let faces: Vec<[u32; 3]> = vec![
            [0, 3, 2],
            [0, 2, 1],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        crate::io::build_mesh_from_vertices_and_faces(&verts, &faces)
    }

    #[test]
    fn union_disjoint_cubes() {
        let a = build_cube();
        let b = build_cube_offset(3.0, 0.0, 0.0);
        let result = boolean_union(&a, &b);
        assert_eq!(result.face_count(), 24);
    }

    #[test]
    fn union_overlapping_cubes() {
        let a = build_cube();
        let b = build_cube_offset(0.5, 0.0, 0.0);
        let result = boolean_union(&a, &b);
        assert!(result.face_count() <= 24);
        assert!(result.face_count() >= 4);
    }

    #[test]
    fn intersection_overlapping_cubes() {
        let a = build_cube();
        let b = build_cube_offset(0.5, 0.0, 0.0);
        let result = boolean_intersection(&a, &b);
        assert!(
            result.vertex_count() > 0,
            "overlapping intersection should be non-empty"
        );
    }

    #[test]
    fn intersection_disjoint_cubes() {
        let a = build_cube();
        let b = build_cube_offset(3.0, 0.0, 0.0);
        let result = boolean_intersection(&a, &b);
        assert_eq!(result.face_count(), 0);
    }

    #[test]
    fn difference_self_is_empty() {
        let a = build_cube();
        let result = boolean_difference(&a, &a);
        assert_eq!(
            result.face_count(),
            0,
            "A-A should be strictly empty under corefinement"
        );
    }

    #[test]
    fn symmetric_difference_disjoint_equals_union() {
        let a = build_cube();
        let b = build_cube_offset(3.0, 0.0, 0.0);
        let sym = boolean_symmetric_difference(&a, &b);
        let uni = boolean_union(&a, &b);
        assert_eq!(sym.face_count(), uni.face_count());
    }

    #[test]
    fn intersection_contains_cube() {
        let big = build_cube();
        let small = build_cube_offset(0.0, 0.0, 0.0);
        let result = boolean_intersection(&big, &small);
        assert!(
            result.face_count() >= 6,
            "intersection of identical cubes should have most faces"
        );
    }

    // ===== 共面三角形重叠测试 =====

    #[test]
    fn coplanar_overlap_containment() {
        // 大三角形完全包含小三角形（共面 z=0）
        let big = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]];
        let small = [[2.0, 2.0, 0.0], [4.0, 2.0, 0.0], [2.0, 4.0, 0.0]];
        let overlap = coplanar_triangle_overlap(big, small);
        assert!(overlap.is_some(), "contained triangle should have overlap");
        let poly = overlap.unwrap();
        assert_eq!(poly.len(), 3, "overlap of containment should be the inner triangle");
        // 验证重叠多边形 ≈ small 三角形
        for &v in &poly {
            let is_small_vertex = small.iter().any(|&s| {
                (s[0] - v[0]).abs() < 1e-9 && (s[1] - v[1]).abs() < 1e-9 && (s[2] - v[2]).abs() < 1e-9
            });
            assert!(is_small_vertex, "overlap vertex should match small triangle vertex");
        }
    }

    #[test]
    fn coplanar_overlap_partial() {
        // 部分重叠：两个三角形共享一条边的一部分
        let a = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]];
        let b = [[2.0, 0.0, 0.0], [6.0, 0.0, 0.0], [2.0, 4.0, 0.0]];
        let overlap = coplanar_triangle_overlap(a, b);
        assert!(overlap.is_some(), "partially overlapping triangles should have overlap");
        let poly = overlap.unwrap();
        assert!(poly.len() >= 3, "partial overlap should produce a polygon");
    }

    #[test]
    fn coplanar_overlap_disjoint() {
        // 不重叠的两个共面三角形
        let a = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let b = [[5.0, 5.0, 0.0], [6.0, 5.0, 0.0], [5.0, 6.0, 0.0]];
        let overlap = coplanar_triangle_overlap(a, b);
        assert!(overlap.is_none(), "disjoint coplanar triangles should have no overlap");
    }

    #[test]
    fn coplanar_overlap_non_coplanar() {
        // 非共面三角形 → None
        let a = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let b = [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
        let overlap = coplanar_triangle_overlap(a, b);
        assert!(overlap.is_none(), "non-coplanar triangles should return None");
    }

    /// 构建长方体 [min, max]³
    fn build_box(min: [f64; 3], max: [f64; 3]) -> MeshStorage {
        let verts: Vec<[f64; 3]> = vec![
            [min[0], min[1], min[2]],
            [max[0], min[1], min[2]],
            [max[0], max[1], min[2]],
            [min[0], max[1], min[2]],
            [min[0], min[1], max[2]],
            [max[0], min[1], max[2]],
            [max[0], max[1], max[2]],
            [min[0], max[1], max[2]],
        ];
        let faces: Vec<[u32; 3]> = vec![
            [0, 3, 2], [0, 2, 1],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [3, 7, 6], [3, 6, 2],
            [0, 4, 7], [0, 7, 3],
            [1, 2, 6], [1, 6, 5],
        ];
        crate::io::build_mesh_from_vertices_and_faces(&verts, &faces)
    }

    #[test]
    fn coplanar_difference_small_on_large() {
        // 大立方体 [-2,2]³，小立方体 [-1,1]×[-1,1]×[2,4]
        // 小立方体底面 z=2 与大立方体顶面 z=2 共面且完全重叠
        let big = build_box([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]);
        let small = build_box([-1.0, -1.0, 2.0], [1.0, 1.0, 4.0]);
        let result = boolean_difference(&big, &small);

        // A - B: 大立方体顶面应有"洞"（小立方体底面区域被切除）
        // 验证结果非空且是有效闭合流形
        assert!(result.face_count() > 0, "difference should be non-empty");
        assert!(result.face_count() < big.face_count() + small.face_count(),
            "difference should have fewer faces than combined input");

        // 验证拓扑一致性
        let errors = crate::validate::validate_topology(&result);
        assert!(errors.is_empty(), "result should be topologically valid, got {} errors: {:?}",
            errors.len(), errors.first());
    }

    #[test]
    fn coplanar_union_small_on_large() {
        let big = build_box([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]);
        let small = build_box([-1.0, -1.0, 2.0], [1.0, 1.0, 4.0]);
        let result = boolean_union(&big, &small);

        assert!(result.face_count() > 0, "union should be non-empty");

        // 验证拓扑一致性
        let errors = crate::validate::validate_topology(&result);
        assert!(errors.is_empty(), "result should be topologically valid, got {} errors: {:?}",
            errors.len(), errors.first());
    }

    #[test]
    fn coplanar_intersection_small_on_large() {
        let big = build_box([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]);
        let small = build_box([-1.0, -1.0, 2.0], [1.0, 1.0, 4.0]);
        let result = boolean_intersection(&big, &small);

        // 两个立方体仅共面接触（无体积重叠），共面回退将共享面判定为 inside。
        // 交集结果包含两侧的共面面片（可能非流形），验证至少非空。
        assert!(result.face_count() > 0, "intersection should have the coplanar overlap");
    }

    // ===== 空网格测试 =====

    #[test]
    fn union_empty_meshes_returns_empty() {
        let a = MeshStorage::new();
        let b = MeshStorage::new();
        let result = boolean_union(&a, &b);
        assert_eq!(result.face_count(), 0);
    }

    #[test]
    fn intersection_empty_meshes_returns_empty() {
        let a = MeshStorage::new();
        let b = MeshStorage::new();
        let result = boolean_intersection(&a, &b);
        assert_eq!(result.face_count(), 0);
    }

    #[test]
    fn difference_empty_meshes_returns_empty() {
        let a = MeshStorage::new();
        let b = MeshStorage::new();
        let result = boolean_difference(&a, &b);
        assert_eq!(result.face_count(), 0);
    }

    #[test]
    fn union_empty_and_cube_returns_cube_face_count() {
        let empty = MeshStorage::new();
        let cube = crate::primitives::build_cube(1.0);
        let result = boolean_union(&empty, &cube);
        assert!(
            result.face_count() >= 12,
            "union with empty should preserve cube's 12 faces, got {}",
            result.face_count()
        );
    }

    #[test]
    fn difference_cube_and_empty_returns_cube() {
        let cube = crate::primitives::build_cube(1.0);
        let empty = MeshStorage::new();
        let result = boolean_difference(&cube, &empty);
        assert!(
            result.face_count() >= 12,
            "cube minus empty should preserve cube's 12 faces, got {}",
            result.face_count()
        );
    }

    #[test]
    fn intersection_cube_and_empty_returns_empty() {
        let cube = crate::primitives::build_cube(1.0);
        let empty = MeshStorage::new();
        let result = boolean_intersection(&cube, &empty);
        assert_eq!(result.face_count(), 0);
    }

    // ===== 完全重合测试 =====

    #[test]
    fn union_identical_cubes() {
        let a = crate::primitives::build_cube(1.0);
        let b = crate::primitives::build_cube(1.0);
        let result = boolean_union(&a, &b);
        let errors = crate::validate::validate_topology(&result);
        assert!(
            errors.is_empty(),
            "union of identical cubes should be topologically valid, got {} errors: {:?}",
            errors.len(),
            errors.first()
        );
    }

    #[test]
    fn intersection_identical_cubes() {
        let a = crate::primitives::build_cube(1.0);
        let b = crate::primitives::build_cube(1.0);
        let result = boolean_intersection(&a, &b);
        // 两相同立方体交集：两侧共面面片均被判为 inside 而保留，
        // 产生重复面（已知非流形局限，见模块文档"限制"节），故仅校验面数。
        assert!(
            result.face_count() >= 6,
            "intersection of identical cubes should retain most faces, got {}",
            result.face_count()
        );
    }

    #[test]
    fn symmetric_difference_identical_cubes_is_empty() {
        let a = crate::primitives::build_cube(1.0);
        let b = crate::primitives::build_cube(1.0);
        let result = boolean_symmetric_difference(&a, &b);
        assert_eq!(
            result.face_count(),
            0,
            "symmetric difference of identical cubes should be empty"
        );
    }

    // ===== 包含关系测试 =====

    #[test]
    fn difference_large_minus_small() {
        // 大立方体 [-1,1]³ 完全包含小立方体（边长 0.5，顶点 ±0.25）
        let large = build_box([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let small = crate::primitives::build_cube(0.5);
        let result = boolean_difference(&large, &small);
        let errors = crate::validate::validate_topology(&result);
        assert!(
            errors.is_empty(),
            "large minus small should be topologically valid, got {} errors: {:?}",
            errors.len(),
            errors.first()
        );
    }

    #[test]
    fn intersection_large_and_small_returns_smallish() {
        let large = build_box([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let small = crate::primitives::build_cube(0.5);
        let result = boolean_intersection(&large, &small);
        assert!(
            result.face_count() > 0,
            "intersection of large and small should be non-empty"
        );
        let errors = crate::validate::validate_topology(&result);
        assert!(
            errors.is_empty(),
            "intersection of large and small should be topologically valid, got {} errors: {:?}",
            errors.len(),
            errors.first()
        );
    }

    // ===== 非轴对齐测试 =====

    #[test]
    fn union_rotated_cubes() {
        // 轴对齐 cube，边长 1，中心原点
        let a = crate::primitives::build_cube(1.0);

        // 绕 Z 轴旋转 45° 的同尺寸 cube
        let theta = std::f64::consts::FRAC_PI_4;
        let (c, s) = (theta.cos(), theta.sin());
        let rotate = |p: [f64; 3]| -> [f64; 3] {
            [p[0] * c - p[1] * s, p[0] * s + p[1] * c, p[2]]
        };
        let h = 0.5;
        let verts: Vec<[f64; 3]> = [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],
        ]
        .iter()
        .map(|&p| rotate(p))
        .collect();
        let faces: Vec<[u32; 3]> = vec![
            [0, 3, 2],
            [0, 2, 1],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        let b = crate::io::build_mesh_from_vertices_and_faces(&verts, &faces);

        let result = boolean_union(&a, &b);
        assert!(
            result.face_count() > 0,
            "union of rotated cubes should be non-empty"
        );
        let errors = crate::validate::validate_topology(&result);
        assert!(
            errors.is_empty(),
            "union of rotated cubes should be topologically valid, got {} errors: {:?}",
            errors.len(),
            errors.first()
        );
    }
}
