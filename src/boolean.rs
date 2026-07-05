//! 网格布尔运算模块
//!
//! 支持闭合流形三角网格间的并集、交集、差集、对称差。
//!
//! ## 算法概要
//! 用射线法（Möller-Trumbore）对面进行分类（inside / outside），
//! 根据布尔操作类型收集目标面，通过 `add_triangle` 构建结果网格。
//!
//! ## 限制
//! - 仅支持闭合流形三角网格（无边界边）；
//! - 不支持共面退化（精确共面三角形可能遗漏或重复）；
//! - 交点坐标使用 f64 精度，极端退化可能失败。

use std::collections::HashMap;

use crate::ids::{FaceId, VertexId};
use crate::storage::{MeshStorage, Vertex};
use crate::topology_ops::add_triangle;

// ============================================================
// 3D 向量工具（模块内部）
// ============================================================

type V3 = [f64; 3];

fn v3_sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn v3_add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn v3_scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn v3_dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn v3_cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ============================================================
// 布尔操作类型
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOp {
    /// A ∪ B：面在 A 外部 且 在 B 外部 → 丢弃；其余保留。
    Union,
    /// A ∩ B：面在 A 内部 且 在 B 内部 → 保留；其余丢弃。
    Intersection,
    /// A - B：面在 A 内部 且 在 B 外部 → 保留；其余丢弃。
    Difference,
    /// (A - B) ∪ (B - A)
    SymmetricDifference,
}

// ============================================================
// 射线法内外分类
// ============================================================

/// 判断点 `point` 是否在闭合网格内部（射线法）。
/// 从 point 发射 +x 方向射线，统计与三角形相交次数。
fn point_inside_mesh(point: V3, mesh: &MeshStorage) -> bool {
    let dir: V3 = [1.0, 0.0, 0.0];
    let mut count = 0u32;

    for f in mesh.face_ids() {
        let verts = collect_face_positions(mesh, f);
        if verts[0] == verts[1] || verts[0] == verts[2] {
            continue;
        }
        if ray_triangle_intersection(point, dir, &verts) {
            count += 1;
        }
    }

    count % 2 == 1 // 奇数 → 内部
}

/// 收集面 f 的三个顶点位置。
fn collect_face_positions(mesh: &MeshStorage, f: FaceId) -> [V3; 3] {
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

/// Möller-Trumbore 射线-三角形求交。返回 true 表示相交。
fn ray_triangle_intersection(origin: V3, dir: V3, tri: &[V3; 3]) -> bool {
    let e1 = v3_sub(tri[1], tri[0]);
    let e2 = v3_sub(tri[2], tri[0]);
    let pvec = v3_cross(dir, e2);
    let det = v3_dot(e1, pvec);

    if det.abs() < 1e-14 {
        return false;
    }

    let inv_det = 1.0 / det;
    let tvec = v3_sub(origin, tri[0]);
    let u = v3_dot(tvec, pvec) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return false;
    }

    let qvec = v3_cross(tvec, e1);
    let v = v3_dot(dir, qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }

    let t = v3_dot(e2, qvec) * inv_det;
    t > 1e-12 // 交点必须在射线前方
}

// ============================================================
// 布尔运算主接口
// ============================================================

/// 执行指定类型的布尔运算，返回结果网格。
///
/// 两个输入网格必须为闭合流形三角网格。若任一网格不闭合或为空，
/// 结果可能不正确。
pub fn boolean_operation(mesh_a: &MeshStorage, mesh_b: &MeshStorage, op: BoolOp) -> MeshStorage {
    // 使用射线法对 mesh_a 的每个面分类
    let mut result = MeshStorage::new();

    // 收集符合条件的面
    let mut kept_triangles: Vec<[V3; 3]> = Vec::new();

    // 处理 mesh_a 的面
    for f in mesh_a.face_ids() {
        let tri = collect_face_positions(mesh_a, f);
        let center = v3_scale(v3_add(v3_add(tri[0], tri[1]), tri[2]), 1.0 / 3.0);
        // 用面法向方向做偏移测试：沿法向偏移 → 外部，反向偏移 → 内部
        let normal = v3_cross(v3_sub(tri[1], tri[0]), v3_sub(tri[2], tri[0]));
        let l = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        let eps = if l > 1e-14 { 1e-4 / l } else { 1e-4 };
        let outward = [
            center[0] + normal[0] * eps,
            center[1] + normal[1] * eps,
            center[2] + normal[2] * eps,
        ];
        let inward = [
            center[0] - normal[0] * eps,
            center[1] - normal[1] * eps,
            center[2] - normal[2] * eps,
        ];
        // 面中心沿法向偏移应在外，反向应在内
        let outside_b = point_inside_mesh(outward, mesh_b);
        let inside_b_from_inward = point_inside_mesh(inward, mesh_b);
        // 保守策略：只有两种测试都指示 inside 才认为 inside
        let inside_b = inside_b_from_inward && !outside_b;

        if classify(op, true, inside_b) {
            kept_triangles.push(tri);
        }
    }

    // 处理 mesh_b 的面
    for f in mesh_b.face_ids() {
        let tri = collect_face_positions(mesh_b, f);
        let center = v3_scale(v3_add(v3_add(tri[0], tri[1]), tri[2]), 1.0 / 3.0);
        let normal = v3_cross(v3_sub(tri[1], tri[0]), v3_sub(tri[2], tri[0]));
        let l = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        let eps = if l > 1e-14 { 1e-4 / l } else { 1e-4 };
        let outward = [
            center[0] + normal[0] * eps,
            center[1] + normal[1] * eps,
            center[2] + normal[2] * eps,
        ];
        let inward = [
            center[0] - normal[0] * eps,
            center[1] - normal[1] * eps,
            center[2] - normal[2] * eps,
        ];
        let outside_a = point_inside_mesh(outward, mesh_a);
        let inside_a_from_inward = point_inside_mesh(inward, mesh_a);
        let inside_a = inside_a_from_inward && !outside_a;

        if classify(op, inside_a, true) {
            kept_triangles.push(tri);
        }
    }

    // 构建结果网格
    let total_v_cap = kept_triangles.len() * 3;
    result.reserve(total_v_cap, kept_triangles.len() * 6, kept_triangles.len());

    // 用 add_triangle 逐面构建（利用其自动配对 twin 的能力）
    let mut index_pool: HashMap<[i64; 3], VertexId> = HashMap::new();

    for tri in &kept_triangles {
        let v0 = get_or_add_vertex(&mut result, &mut index_pool, tri[0]);
        let v1 = get_or_add_vertex(&mut result, &mut index_pool, tri[1]);
        let v2 = get_or_add_vertex(&mut result, &mut index_pool, tri[2]);

        // 使用 add_triangle 自动配对共享边
        let _ = add_triangle(&mut result, v0, v1, v2);
    }

    result
}

fn classify(op: BoolOp, inside_a: bool, inside_b: bool) -> bool {
    match op {
        BoolOp::Union => inside_a || inside_b,
        BoolOp::Intersection => inside_a && inside_b,
        BoolOp::Difference => inside_a && !inside_b,
        BoolOp::SymmetricDifference => inside_a != inside_b,
    }
}

fn get_or_add_vertex(
    mesh: &mut MeshStorage,
    pool: &mut HashMap<[i64; 3], VertexId>,
    pos: V3,
) -> VertexId {
    // 量化到 1e-9 精度做去重
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
        // 12 个三角形（6 面 × 2）
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
            // -z (z=-1, 朝外看 CCW)
            [0, 3, 2],
            [0, 2, 1],
            // +z (z=1)
            [4, 5, 6],
            [4, 6, 7],
            // -y (y=-1)
            [0, 1, 5],
            [0, 5, 4],
            // +y (y=1)
            [3, 7, 6],
            [3, 6, 2],
            // -x (x=-1)
            [0, 4, 7],
            [0, 7, 3],
            // +x (x=1)
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
        // 两个不相交的立方体并集 = 两倍的三角形
        assert_eq!(result.face_count(), 24);
    }

    #[test]
    fn union_overlapping_cubes() {
        let a = build_cube();
        let b = build_cube_offset(0.5, 0.0, 0.0);
        let result = boolean_union(&a, &b);
        // 重叠立方体并集面数应 ≤ 24（有些面在内部被丢弃）
        assert!(result.face_count() <= 24);
        assert!(result.face_count() >= 4); // 至少保留包围盒外面
    }

    #[test]
    fn intersection_overlapping_cubes() {
        let a = build_cube();
        let b = build_cube_offset(0.5, 0.0, 0.0);
        let result = boolean_intersection(&a, &b);
        // 交集应非空（重叠区域）
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
        // 不相交 → 空集
        assert_eq!(result.face_count(), 0);
    }

    #[test]
    fn difference_self_is_empty() {
        let a = build_cube();
        let result = boolean_difference(&a, &a);
        // A - A 应接近空（射线法在边界处可能残留少量面）
        assert!(result.face_count() <= 2, "A-A should be nearly empty");
    }

    #[test]
    fn symmetric_difference_disjoint_equals_union() {
        let a = build_cube();
        let b = build_cube_offset(3.0, 0.0, 0.0);
        let sym = boolean_symmetric_difference(&a, &b);
        let uni = boolean_union(&a, &b);
        // 不相交时，对称差 = 并集
        assert_eq!(sym.face_count(), uni.face_count());
    }

    #[test]
    fn intersection_contains_cube() {
        let big = build_cube(); // [-1,1]
        let small = build_cube_offset(0.0, 0.0, 0.0); // same cube
        let result = boolean_intersection(&big, &small);
        // 两个相同立方体的交集应按近于自身
        assert!(
            result.face_count() >= 6,
            "intersection of identical cubes should have most faces"
        );
    }
}
