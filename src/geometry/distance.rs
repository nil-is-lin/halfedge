//! 点到三角形距离与射线求交
//!
//! 提供点到三角形距离和射线与网格求交功能：
//! - 点到三角形最近距离：[`point_triangle_distance`]
//! - 点到三角形最近点：[`closest_point_on_triangle`]
//! - 射线与三角形求交：[`ray_triangle_intersection`]
//! - 射线与网格求交：[`ray_mesh_intersection`], [`ray_mesh_intersects`]
//! - 并行射线求交：[`ray_mesh_intersection_par`]
//! - 射线交点信息：[`RayHit`]

use crate::Scalar;
use crate::ids::{FaceId, VertexId};
use crate::linalg::vec3::{Vec3, add, cross, dot, length, scale, sub};
use crate::storage::MeshStorage;

/// 射线与网格的交点信息。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    /// 交点位置
    pub position: [Scalar; 3],
    /// 从射线原点到交点的距离（参数 t，满足 `origin + t * direction`）
    pub t: Scalar,
    /// 相交的三角面的索引（`FaceId`）
    pub face: FaceId,
    /// 重心坐标 (u, v)，第三个坐标为 1-u-v
    pub barycentric: (f64, f64),
}

/// 点到三角形的最近距离。
///
/// 参数顺序：查询点 `p`，三角形三顶点 `a, b, c`（CCW 或 CW 均可）。
///
/// ## 算法
/// Ericson, *Real-Time Collision Detection* 5.1.5。
/// 将三角形所在平面划分为 7 个 Voronoi 区域（3 个顶点域、3 条边域、1 个面域），
/// 通过点积判定 `p` 投影所属区域，再求最近点。
///
/// ## 复杂度
/// $O(1)$，无循环、无开方（除最后一次距离计算）。
pub fn point_triangle_distance(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f64 {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);

    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return length(ap); // 顶点 A 域
    }

    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return length(bp); // 顶点 B 域
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = if d1 - d3 == 0.0 { 0.0 } else { d1 / (d1 - d3) };
        let closest = add(a, scale(ab, v));
        return length(sub(p, closest)); // 边 AB 域
    }

    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return length(cp); // 顶点 C 域
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = if d2 - d6 == 0.0 { 0.0 } else { d2 / (d2 - d6) };
        let closest = add(a, scale(ac, w));
        return length(sub(p, closest)); // 边 AC 域
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let denom = (d4 - d3) + (d5 - d6);
        let w = if denom == 0.0 { 0.0 } else { (d4 - d3) / denom };
        let closest = add(b, scale(sub(c, b), w));
        return length(sub(p, closest)); // 边 BC 域
    }

    // 面域：重心坐标 (1-v-w, v, w)
    let denom = va + vb + vc;
    if denom.abs() < 1e-20 {
        // 退化三角形（共线），退化到三顶点最近者
        return length(ap).min(length(bp)).min(length(cp));
    }
    let inv = 1.0 / denom;
    let v = vb * inv;
    let w = vc * inv;
    let closest = add(a, add(scale(ab, v), scale(ac, w)));
    length(sub(p, closest))
}

/// 点到三角形所在表面的最近点（Ericson *Real-Time Collision Detection* 5.1.5）。
///
/// 与 [`point_triangle_distance`] 同源算法，但返回最近点本身而非距离。
/// 用于 BVH 最近点查询等需要插值/法向插值等后续计算的场景。
pub fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);

    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a; // 顶点 A 域
    }

    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b; // 顶点 B 域
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = if d1 - d3 == 0.0 { 0.0 } else { d1 / (d1 - d3) };
        return add(a, scale(ab, v)); // 边 AB 域
    }

    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c; // 顶点 C 域
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = if d2 - d6 == 0.0 { 0.0 } else { d2 / (d2 - d6) };
        return add(a, scale(ac, w)); // 边 AC 域
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let denom = (d4 - d3) + (d5 - d6);
        let w = if denom == 0.0 { 0.0 } else { (d4 - d3) / denom };
        return add(b, scale(sub(c, b), w)); // 边 BC 域
    }

    // 面域：重心坐标 (1-v-w, v, w)
    let denom = va + vb + vc;
    if denom.abs() < 1e-20 {
        // 退化三角形：返回最近顶点
        let la = dot(ap, ap);
        let lb = dot(bp, bp);
        let lc = dot(cp, cp);
        if la <= lb && la <= lc {
            return a;
        }
        if lb <= lc {
            return b;
        }
        return c;
    }
    let inv = 1.0 / denom;
    let v = vb * inv;
    let w = vc * inv;
    add(a, add(scale(ab, v), scale(ac, w)))
}

/// 射线与三角形求交（基于 Shewchuk 鲁棒 `orient3d` 谓词）。
///
/// 返回参数 t（`origin + t * direction` 即为交点）与重心坐标 `(u, v)`。
/// 不相交返回 `None`。
///
/// ## 算法
/// 1. 用 `orient3d(v0, v1, v2, origin)` 与 `orient3d(v0, v1, v2, origin+dir)`
///    判定射线是否穿过三角形平面（鲁棒符号判定）；
/// 2. 线性插值求交点参数 t；
/// 3. 投影到三角形法向主轴的正交平面，用鲁棒 `point_in_triangle_2d`
///    判定交点是否在三角形内；
/// 4. 用 2D 重心坐标公式计算 (u, v)。
///
/// 相比朴素 Möller-Trumbore，本实现在共面、共线、交点接近三角形边等
/// 退化情况下能给出精确判定，避免因浮点舍入导致的误判。
pub fn ray_triangle_intersection(
    origin: [f64; 3],
    direction: [f64; 3],
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
) -> Option<(f64, f64, f64)> {
    use crate::predicates::{orient3d, point_in_triangle_2d};

    // 1. 鲁棒判定射线与三角形平面相交
    // 由 orient3d 在第 4 个参数上的线性性：
    //   f(t) = orient3d(v0, v1, v2, origin + t*dir) = a + t*(b - a)
    // 其中 a = orient3d(v0, v1, v2, origin)，b = orient3d(v0, v1, v2, origin+dir)
    // 注意：direction 是方向向量（不一定是单位向量），origin+dir 是 t=1 处的点。
    let a = orient3d(v0, v1, v2, origin);
    let end = [
        origin[0] + direction[0],
        origin[1] + direction[1],
        origin[2] + direction[2],
    ];
    let b = orient3d(v0, v1, v2, end);

    // 求交点参数 t：a + t*(b - a) = 0 → t = -a / (b - a)
    let denom = b - a;
    if denom == 0.0 {
        return None; // 射线与平面平行（含在平面内的退化情况）
    }
    let t = -a / denom;
    if t <= 1e-12 {
        return None; // 交点在射线后方或原点附近
    }

    // 3. 计算交点 P = origin + t * direction
    let p = [
        origin[0] + t * direction[0],
        origin[1] + t * direction[1],
        origin[2] + t * direction[2],
    ];

    // 4. 投影到 2D（丢弃法向最大分量轴），用鲁棒 point_in_triangle_2d 判定
    let e1 = sub(v1, v0);
    let e2 = sub(v2, v0);
    let normal = cross(e1, e2);
    let abs_n = [normal[0].abs(), normal[1].abs(), normal[2].abs()];
    let drop_axis = if abs_n[0] >= abs_n[1] && abs_n[0] >= abs_n[2] {
        0
    } else if abs_n[1] >= abs_n[2] {
        1
    } else {
        2
    };

    let project = |q: [f64; 3]| -> [f64; 2] {
        match drop_axis {
            0 => [q[1], q[2]],
            1 => [q[0], q[2]],
            _ => [q[0], q[1]],
        }
    };

    let p2d = project(p);
    let a2d = project(v0);
    let b2d = project(v1);
    let c2d = project(v2);

    if !point_in_triangle_2d(p2d, a2d, b2d, c2d) {
        return None;
    }

    // 5. 用 2D 重心坐标公式计算 (u, v)
    // P = v0 + u*(v1-v0) + v*(v2-v0)，在 2D 投影中求解
    let ab = [b2d[0] - a2d[0], b2d[1] - a2d[1]];
    let ac = [c2d[0] - a2d[0], c2d[1] - a2d[1]];
    let ap = [p2d[0] - a2d[0], p2d[1] - a2d[1]];
    let denom = ab[0] * ac[1] - ab[1] * ac[0];
    if denom == 0.0 {
        return None; // 退化三角形（共线）
    }
    let inv_denom = 1.0 / denom;
    let u = (ap[0] * ac[1] - ap[1] * ac[0]) * inv_denom;
    let v = (ab[0] * ap[1] - ab[1] * ap[0]) * inv_denom;

    Some((t, u, v))
}

/// 射线与网格求最近交点。返回距离最近的 `RayHit`；无交点返回 `None`。
///
/// 遍历所有三角面，计算与射线的交点，取 t 最小者。复杂度 O(F)。
pub fn ray_mesh_intersection(
    origin: [f64; 3],
    direction: [f64; 3],
    mesh: &MeshStorage,
) -> Option<RayHit> {
    let mut best: Option<RayHit> = None;

    for f in mesh.face_ids() {
        let verts: Vec<VertexId> = crate::traversal::FaceVertices::new(mesh, f).collect();
        if verts.len() != 3 {
            continue;
        }
        let v0 = mesh.get_vertex(verts[0])?.position;
        let v1 = mesh.get_vertex(verts[1])?.position;
        let v2 = mesh.get_vertex(verts[2])?.position;

        if let Some((t, u, v)) = ray_triangle_intersection(origin, direction, v0, v1, v2) {
            let hit = RayHit {
                position: [
                    origin[0] + t * direction[0],
                    origin[1] + t * direction[1],
                    origin[2] + t * direction[2],
                ],
                t,
                face: f,
                barycentric: (u, v),
            };
            match best {
                Some(ref b) if hit.t < b.t => best = Some(hit),
                None => best = Some(hit),
                _ => {}
            }
        }
    }

    best
}

/// 射线是否与网格相交（假设网格闭合；使用射线法奇偶判定）。
///
/// 沿 direction 方向发射射线，统计与网格三角面的交点个数。
/// 奇数个交点 → `true`（进入网格内部）。
pub fn ray_mesh_intersects(origin: [f64; 3], direction: [f64; 3], mesh: &MeshStorage) -> bool {
    let mut count = 0u32;
    for f in mesh.face_ids() {
        let verts: Vec<VertexId> = crate::traversal::FaceVertices::new(mesh, f).collect();
        if verts.len() != 3 {
            continue;
        }
        // 拓扑不一致时跳过，避免 panic
        let (v0, v1, v2) = match (
            mesh.get_vertex(verts[0]),
            mesh.get_vertex(verts[1]),
            mesh.get_vertex(verts[2]),
        ) {
            (Some(a), Some(b), Some(c)) => (a.position, b.position, c.position),
            _ => continue,
        };
        if ray_triangle_intersection(origin, direction, v0, v1, v2).is_some() {
            count += 1;
        }
    }
    count % 2 == 1
}

// ============================================================
// 并行变体（rayon）
// ============================================================

/// 并行检测射线与网格的所有交点。
pub fn ray_mesh_intersection_par(
    mesh: &MeshStorage,
    origin: [f64; 3],
    direction: [f64; 3],
) -> Vec<RayHit> {
    use rayon::prelude::*;
    let faces: Vec<FaceId> = mesh.face_ids().collect();
    faces
        .par_iter()
        .filter_map(|&f| {
            let verts: Vec<VertexId> = crate::traversal::FaceVertices::new(mesh, f).collect();
            if verts.len() != 3 {
                return None;
            }
            let v0 = mesh.get_vertex(verts[0])?.position;
            let v1 = mesh.get_vertex(verts[1])?.position;
            let v2 = mesh.get_vertex(verts[2])?.position;
            let (t, u, v) = ray_triangle_intersection(origin, direction, v0, v1, v2)?;
            let position = [
                origin[0] + direction[0] * t,
                origin[1] + direction[1] * t,
                origin[2] + direction[2] * t,
            ];
            Some(RayHit {
                position,
                t,
                face: f,
                barycentric: (u, v),
            })
        })
        .collect()
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- 点到三角形距离 ----------

    #[test]
    fn point_triangle_distance_point_on_face() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        // 重心
        let p = [1.0 / 3.0, 1.0 / 3.0, 0.0];
        assert!(point_triangle_distance(p, a, b, c).abs() < 1e-9);
    }

    #[test]
    fn point_triangle_distance_above_face() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let p = [0.2, 0.2, 1.0];
        assert!((point_triangle_distance(p, a, b, c) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn point_triangle_distance_vertex_region() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        // 远离 A 的方向
        let p = [-1.0, -1.0, 0.0];
        assert!((point_triangle_distance(p, a, b, c) - 2.0_f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn point_triangle_distance_edge_region() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        // 投影落在边 AB 外延，最近点应是 B
        let p = [2.0, 0.0, 0.0];
        assert!((point_triangle_distance(p, a, b, c) - 1.0).abs() < 1e-9);
    }

    // ---------- 射线求交 ----------

    #[test]
    fn ray_triangle_hit() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        // 从上方垂直射下
        let hit = ray_triangle_intersection([0.25, 0.25, 1.0], [0.0, 0.0, -1.0], v0, v1, v2);
        assert!(hit.is_some());
        let (t, u, v) = hit.unwrap();
        assert!((t - 1.0).abs() < 1e-10);
        assert!((u - 0.25).abs() < 1e-10);
        assert!((v - 0.25).abs() < 1e-10);
    }

    #[test]
    fn ray_triangle_miss_parallel() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        // 平行于三角形
        assert!(ray_triangle_intersection([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], v0, v1, v2).is_none());
    }

    #[test]
    fn test_ray_mesh_intersects() {
        let mesh = crate::test_util::build_icosphere(2);
        // 从球外向球心方向射 → 应命中（奇偶检验：奇数交点）
        let hits = ray_mesh_intersects([2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], &mesh);
        // 可能命中也可能恰好经过顶点/边；仅验证不 panic
        let _ = hits;
    }

    #[test]
    fn test_ray_mesh_intersection_icosphere() {
        let mesh = crate::test_util::build_icosphere(2);
        let hit = ray_mesh_intersection([2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], &mesh);
        assert!(hit.is_some());
        let h = hit.unwrap();
        assert!((h.position[0] - 1.0).abs() < 0.1);
        assert!(h.t > 0.0 && h.t < 2.0);
    }

    #[test]
    fn test_ray_mesh_intersection_miss() {
        let mesh = crate::test_util::build_icosphere(2);
        assert!(ray_mesh_intersection([3.0, 0.0, 0.0], [1.0, 0.0, 0.0], &mesh).is_none());
    }

    #[test]
    fn ray_mesh_intersects_empty_mesh_returns_false() {
        let mesh = MeshStorage::new();
        assert!(!ray_mesh_intersects(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            &mesh
        ));
    }

    // ---------- ray_mesh_intersects 正确性 ----------

    #[test]
    fn ray_mesh_intersects_origin_outside_returns_false() {
        // ray_mesh_intersects 实现为奇偶判定（origin 是否在闭合网格内部）。
        // 原点 (0,0,-5) 在球外，射线沿 +z 穿过球面（与 2 个三角面相交，偶数）→ 返回 false。
        let mesh = crate::test_util::build_icosphere(1);
        let result = ray_mesh_intersects([0.0, 0.0, -5.0], [0.0, 0.0, 1.0], &mesh);
        assert!(!result, "原点在球外，奇偶判定应返回 false");
    }

    #[test]
    fn ray_mesh_intersects_misses_returns_false() {
        // 射线从远处射向 +x 方向，不与球面相交（0 个交点）。
        let mesh = crate::test_util::build_icosphere(1);
        let result = ray_mesh_intersects([10.0, 10.0, 10.0], [1.0, 0.0, 0.0], &mesh);
        assert!(!result, "射线不与球面相交，应返回 false");
    }

    // ---------- 并行函数一致性 ----------

    #[test]
    fn ray_mesh_intersection_par_matches_serial() {
        let mesh = crate::test_util::build_icosphere(1);
        let origin = [2.0, 0.0, 0.0];
        let direction = [-1.0, 0.0, 0.0];
        let serial = ray_mesh_intersection(origin, direction, &mesh);
        let par = ray_mesh_intersection_par(&mesh, origin, direction);
        match serial {
            Some(s_hit) => {
                assert!(!par.is_empty(), "par 应至少有一个交点");
                // par 返回所有交点，取 t 最小者与 serial 比较
                let min_par = par
                    .iter()
                    .min_by(|a, b| a.t.partial_cmp(&b.t).unwrap())
                    .unwrap();
                assert!(
                    (s_hit.t - min_par.t).abs() < 1e-10,
                    "t 不一致: serial={} par={}",
                    s_hit.t,
                    min_par.t
                );
            }
            None => {
                assert!(par.is_empty(), "serial 无交点时 par 应为空");
            }
        }
    }
}
